/*
Copyright 2026 The llm-d Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package proxy

import (
	"context"
	"fmt"
	"net/http"
	"sync/atomic"
	"time"

	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/common/observability/semconv"
	"github.com/llm-d/llm-d-router/pkg/common/observability/tracing"
	"github.com/llm-d/llm-d-router/pkg/sidecar/metrics"
)

const (
	// requestHeaderDataParallelRank pins a request to a specific vLLM
	// data-parallel rank, set on both requests of a disagg pair (see pickDPRank).
	requestHeaderDataParallelRank = "x-data-parallel-rank"

	requestFieldTransferID = "transfer_id"
)

// concurrentPrefillGrace limits how long prefill may continue after a
// successful decode returns. A variable so tests can shorten it.
var concurrentPrefillGrace = 30 * time.Second

// runConcurrentPD fires the prefill and decode requests of a concurrent-dispatch
// P/D protocol (Mooncake, SGLang) in parallel: prefill runs in a goroutine
// and only its status and duration are recorded, while decode runs on the
// calling goroutine and streams its response to w. connector is recorded on
// both spans to identify the protocol.
//
// Decode cannot emit output before the prefill KV has arrived, so its first
// body byte is the commit point: until then its output is held, and a prefill
// that fails first cancels decode and is returned to the client verbatim.
// After it, prefill's HTTP response has only informational value.
func (s *Server) runConcurrentPD(
	w http.ResponseWriter,
	r *http.Request,
	prefillBody, decodeBody []byte,
	prefillHost, connector string,
	// prepareRequests lets a connector mutate the cloned requests before dispatch,
	// e.g. Mooncake sets a DP-rank header so prefill lands on the same rank decode reads from.
	prepareRequests func(prefillReq, decodeReq *http.Request),
	toolFields *toolCallingRequest,
) {
	defer toolFields.report(r.Context())
	tracer := tracing.Tracer(tracerScope)
	ctx, cancelDecode := context.WithCancel(r.Context())
	defer cancelDecode()

	// WithoutCancel for prefill so it isn't aborted when the decode response finishes first.
	prefillCtx, cancelPrefill := context.WithCancel(context.WithoutCancel(ctx))
	// Client disconnect aborts prefill too. Stop on return: net/http also cancels
	// r.Context() once the handler finishes, which must not abort prefill.
	defer context.AfterFunc(r.Context(), cancelPrefill)()
	prefillReq := cloneRequestWithBody(prefillCtx, r, prefillBody)
	decodeReq := cloneRequestWithBody(ctx, r, decodeBody)
	if prepareRequests != nil {
		prepareRequests(prefillReq, decodeReq)
	}

	// Prefill runs in a goroutine; decode runs on the calling goroutine and
	// writes the response through dcw.
	dcw := newDeferredCommitWriter(w)
	dcw.commitOnWrite = true
	ctx, prefillSpan := tracer.Start(ctx, "prefill",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	prefillReq = prefillReq.WithContext(trace.ContextWithSpan(prefillReq.Context(), prefillSpan))
	prefillSpan.SetAttributes(
		semconv.LLMDPDProxyPrefillTarget(prefillHost),
		semconv.LLMDPDProxyConnector(connector),
		semconv.LLMDPDProxyPrefillAsync(true),
	)
	prefillStart := time.Now()

	prefillHandler, err := s.prefillerProxyHandler(prefillHost)
	if err != nil {
		prefillSpan.SetStatus(codes.Error, "failed to create prefill handler")
		prefillSpan.End()
		if err := errorBadGateway(err, w); err != nil {
			s.logger.Error(err, "failed to send error response to client")
		}
		return
	}

	// Aggregate before starting the worker so reporting does not wait for prefill.
	toolFields.compareDispatchedForSpan(prefillBody, prefillSpan)
	// Stored before dcw.abort, so a discarded decode can relay it.
	var prefillResp atomic.Pointer[bufferedResponseWriter]
	go func() {
		defer cancelPrefill()
		defer prefillSpan.End()
		defer func() {
			if rec := recover(); rec != nil && rec != http.ErrAbortHandler {
				s.logger.Error(fmt.Errorf("prefill handler panicked (%T)", rec), "panic in prefill request")
			}
		}()
		pw := &bufferedResponseWriter{}
		prefillHandler.ServeHTTP(pw, prefillReq)
		prefillResp.Store(pw)
		prefillDuration := time.Since(prefillStart)
		prefillSpan.SetAttributes(
			semconv.LLMDPDProxyPrefillStatusCode(pw.statusCode),
			semconv.LLMDPDProxyPrefillDurationMs(float64(prefillDuration.Milliseconds())),
		)
		// A prefill the sidecar cancelled is not a prefill error.
		if isHTTPError(pw.statusCode) && prefillCtx.Err() == nil {
			metrics.RecordError(metrics.StagePrefill)
			prefillSpan.SetStatus(codes.Error, "prefill request failed")
			if dcw.abort() {
				cancelDecode()
			} else {
				// Decode output reached the client, so the KV did arrive.
				s.logger.Info("concurrent-dispatch prefill failed after decode committed; response unchanged",
					"connector", connector, "status", pw.statusCode)
			}
		}
		metrics.RecordPrefillDuration(prefillDuration)
		s.logger.V(logging.DEBUG).Info("concurrent-dispatch prefill request completed", "connector", connector, "status", pw.statusCode)
	}()

	// Decode Stage
	ctx, decodeSpan := tracer.Start(ctx, "decode",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	defer decodeSpan.End()

	decodeSpan.SetAttributes(
		semconv.LLMDPDProxyConnector(connector),
		semconv.LLMDPDProxyDecodeConcurrentWithPrefill(true),
		semconv.LLMDPDProxyDecodeTarget(s.config.DecoderURL.Host),
	)
	decodeStart := time.Now()
	toolFields.compareDispatchedForSpan(decodeBody, decodeSpan)

	// The reverse proxy panics with http.ErrAbortHandler when a stream breaks
	// after headers were sent, including when cancelDecode interrupts a body
	// copy. Recovered so the response is still ours to settle; replayed below
	// when it reached the client.
	decodeAborted := false
	func() {
		defer func() {
			if rec := recover(); rec != nil {
				if rec != http.ErrAbortHandler {
					panic(rec)
				}
				decodeAborted = true
			}
		}()
		s.decoderProxy.ServeHTTP(dcw, decodeReq.WithContext(ctx))
	}()
	decodeDuration := time.Since(decodeStart)

	switch {
	case prefillCtx.Err() != nil:
	case decodeAborted || dcw.failed():
		// The request is over once decode has failed; a prefill still running
		// would compute KV nobody consumes.
		cancelPrefill()
	default:
		// Prefill has no deadline of its own; one that never responds would
		// hold the goroutine and connection indefinitely.
		// Not awaited: the client's response must not wait on the prefill response.
		graceTimer := time.AfterFunc(concurrentPrefillGrace, func() {
			s.logger.Error(nil, "concurrent-dispatch prefill did not complete within the grace period after decode; cancelled",
				"connector", connector)
			cancelPrefill()
		})
		context.AfterFunc(prefillCtx, func() { graceTimer.Stop() })
	}

	if dcw.discarded() {
		// Prefill failed before decode produced output: the prefill response
		// owns the client response and decode is not sampled.
		decodeSpan.SetStatus(codes.Error, "decode cancelled after prefill failure")
		if writeErr := prefillResp.Load().relayTo(w); writeErr != nil {
			s.logger.Error(writeErr, "failed to send prefill error to client (concurrent-dispatch)")
		}
		return
	}

	metrics.RecordDecodeDuration(decodeDuration)
	decodeSpan.SetAttributes(semconv.LLMDPDProxyDecodeDurationMs(float64(decodeDuration.Milliseconds())))
	if decodeAborted {
		metrics.RecordError(metrics.StageDecode)
		decodeSpan.SetStatus(codes.Error, "decode handler aborted")
		if dcw.responseStarted() {
			// Already on the wire; net/http drops the connection.
			panic(http.ErrAbortHandler)
		}
		if err := errorBadGateway(errDecodeAborted, w); err != nil {
			s.logger.Error(err, "failed to send decode abort error to client (concurrent-dispatch)")
		}
		return
	}
	// A header-only decode response never reached the commit point.
	dcw.commit()
	if dcw.failed() {
		metrics.RecordError(metrics.StageDecode)
		decodeSpan.SetStatus(codes.Error, "decode request failed")
	}

	// End-to-end P/D timing. True TTFT captures time from gateway request start
	// to decode start; prefill duration is tracked in the async prefill span.
	if currentSpan := trace.SpanFromContext(ctx); currentSpan.SpanContext().IsValid() {
		var totalDuration time.Duration
		var trueTTFT time.Duration
		if requestStartValue := ctx.Value(requestStartTimeKey); requestStartValue != nil {
			if requestStart, ok := requestStartValue.(time.Time); ok {
				totalDuration = time.Since(requestStart)
				trueTTFT = decodeStart.Sub(requestStart)
			}
		}

		currentSpan.SetAttributes(
			semconv.LLMDPDProxyTotalDurationMs(float64(totalDuration.Milliseconds())),
			semconv.LLMDPDProxyTrueTTFTMs(float64(trueTTFT.Milliseconds())),
			semconv.LLMDPDProxyDecodeDurationMsSummary(float64(decodeDuration.Milliseconds())),
			semconv.LLMDPDProxyConcurrentPD(true),
		)
	}
}

// recordDecodeAbort records decode duration and a decode error when decode
// dispatch did not return normally. The reverse proxy panics with
// http.ErrAbortHandler when a stream breaks after headers were sent (client
// disconnect, load balancer timeout), which skips the metrics recorded after
// dispatch. Deferred without recover, so the panic still propagates.
func recordDecodeAbort(returned *bool, start time.Time) {
	if !*returned {
		metrics.RecordDecodeDuration(time.Since(start))
		metrics.RecordError(metrics.StageDecode)
	}
}
