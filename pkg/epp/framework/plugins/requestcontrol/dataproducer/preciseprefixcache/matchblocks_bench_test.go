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

package preciseprefixcache

import (
	"fmt"
	"testing"

	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
)

// benchLookup mirrors the promptLookup struct in producer.go.
type benchLookup struct {
	keys      []kvblock.BlockHash
	keyToPods map[kvblock.BlockHash][]kvblock.PodEntry
}

type benchScenario struct {
	name    string
	addrs   []string
	lookups []benchLookup
}

func benchAddr(i int) string {
	return fmt.Sprintf("10.0.%d.%d:8080", i/256, i%256)
}

// buildBenchData models one Produce call: `endpoints` candidate pods, one
// prompt of `blocks` keys. Endpoint i holds a prefix of hold(i) blocks, each
// held block carrying the entries produced by entryFor(i, j). Every key also
// carries two foreign-pod entries so map lookups do real work even when the
// measured pod holds nothing.
func buildBenchData(endpoints, blocks int, hold func(i int) int,
	entryFor func(i, j int) []kvblock.PodEntry,
) benchScenario {
	addrs := make([]string, endpoints)
	for i := range addrs {
		addrs[i] = benchAddr(i)
	}

	keys := make([]kvblock.BlockHash, blocks)
	keyToPods := make(map[kvblock.BlockHash][]kvblock.PodEntry, blocks)
	foreign := []kvblock.PodEntry{
		{PodIdentifier: "10.9.0.1:8080", DeviceTier: "gpu"},
		{PodIdentifier: "10.9.0.2:8080", DeviceTier: "gpu"},
	}
	for j := 0; j < blocks; j++ {
		keys[j] = kvblock.BlockHash(0x1000 + j)
		keyToPods[keys[j]] = append([]kvblock.PodEntry{}, foreign...)
	}
	for i := 0; i < endpoints; i++ {
		n := hold(i)
		for j := 0; j < n && j < blocks; j++ {
			keyToPods[keys[j]] = append(keyToPods[keys[j]], entryFor(i, j)...)
		}
	}

	return benchScenario{
		name:    fmt.Sprintf("E%d_B%d", endpoints, blocks),
		addrs:   addrs,
		lookups: []benchLookup{{keys: keys, keyToPods: keyToPods}},
	}
}

// benchScenarios crosses hit profiles with cluster sizes. Cost scales with
// the held-prefix length per endpoint, not with prompt length: a zero-hit
// endpoint short-circuits at the first block.
func benchScenarios() []benchScenario {
	profiles := []struct {
		name     string
		holdFrac float64
		entryFor func(blocks int) func(i, j int) []kvblock.PodEntry
	}{
		{
			name:     "zero_hit",
			holdFrac: 0,
			entryFor: func(int) func(i, j int) []kvblock.PodEntry {
				return func(i, j int) []kvblock.PodEntry {
					return []kvblock.PodEntry{{PodIdentifier: benchAddr(i), DeviceTier: "gpu"}}
				}
			},
		},
		{
			name:     "half_hit_gpu",
			holdFrac: 0.5,
			entryFor: func(int) func(i, j int) []kvblock.PodEntry {
				return func(i, j int) []kvblock.PodEntry {
					return []kvblock.PodEntry{{PodIdentifier: benchAddr(i), DeviceTier: "gpu"}}
				}
			},
		},
		{
			name:     "full_hit_gpu",
			holdFrac: 1.0,
			entryFor: func(int) func(i, j int) []kvblock.PodEntry {
				return func(i, j int) []kvblock.PodEntry {
					return []kvblock.PodEntry{{PodIdentifier: benchAddr(i), DeviceTier: "gpu"}}
				}
			},
		},
		{
			name:     "full_hit_gpu_cpu",
			holdFrac: 1.0,
			entryFor: func(int) func(i, j int) []kvblock.PodEntry {
				return func(i, j int) []kvblock.PodEntry {
					return []kvblock.PodEntry{
						{PodIdentifier: benchAddr(i), DeviceTier: "gpu"},
						{PodIdentifier: benchAddr(i), DeviceTier: "cpu"},
					}
				}
			},
		},
		{
			name:     "tier_break_half",
			holdFrac: 1.0,
			entryFor: func(blocks int) func(i, j int) []kvblock.PodEntry {
				return func(i, j int) []kvblock.PodEntry {
					tier := "gpu"
					if j >= blocks/2 {
						tier = "cpu" // union continues past the tier break
					}
					return []kvblock.PodEntry{{PodIdentifier: benchAddr(i), DeviceTier: tier}}
				}
			},
		},
		{
			name:     "speculative_prefix",
			holdFrac: 0.25,
			entryFor: func(int) func(i, j int) []kvblock.PodEntry {
				return func(i, j int) []kvblock.PodEntry {
					return []kvblock.PodEntry{{PodIdentifier: benchAddr(i), Speculative: true}}
				}
			},
		},
	}

	sizes := []struct{ endpoints, blocks int }{
		{10, 128},   // small pool, 2k-token prompt
		{30, 512},   // medium pool, 8k-token prompt
		{100, 2048}, // large pool, 32k-token prompt
	}

	var out []benchScenario
	for _, p := range profiles {
		for _, s := range sizes {
			holdFrac := p.holdFrac
			entryFor := p.entryFor(s.blocks)
			sc := buildBenchData(s.endpoints, s.blocks,
				func(int) int { return int(holdFrac * float64(s.blocks)) },
				entryFor)
			sc.name = fmt.Sprintf("%s/E%d_B%d", p.name, s.endpoints, s.blocks)
			out = append(out, sc)
		}
	}
	return out
}

var (
	benchSinkInt int
	benchSinkMap map[string]int
)

// benchmarkRefPair runs the per-endpoint loop against the pre-change pair of
// functions (refMatchedBlockCount + refMatchedBlockCountByTier in
// difffuzz_test.go), so before/after numbers come from one bench invocation.
func benchmarkRefPair(b *testing.B, sc benchScenario) {
	b.ReportAllocs()
	for b.Loop() {
		for _, addr := range sc.addrs {
			cachedBlocks := 0
			cachedBlocksByTier := map[string]int{}
			for _, lu := range sc.lookups {
				cachedBlocks += refMatchedBlockCount(lu.keys, lu.keyToPods, addr)
				for tier, n := range refMatchedBlockCountByTier(lu.keys, lu.keyToPods, addr) {
					cachedBlocksByTier[tier] += n
				}
			}
			benchSinkInt += cachedBlocks
			benchSinkMap = cachedBlocksByTier
		}
	}
}

// BenchmarkMatchedBlocksBefore measures the two-pass, sets-based counting that
// preceded matchedBlocks; BenchmarkMatchedBlocks measures the current code.
func BenchmarkMatchedBlocksBefore(b *testing.B) {
	for _, sc := range benchScenarios() {
		b.Run(sc.name, func(b *testing.B) {
			benchmarkRefPair(b, sc)
		})
	}
}

// BenchmarkMatchedBlocks measures the per-endpoint match-count loop of
// produceFromBlockKeys across hit profiles and cluster sizes.
func BenchmarkMatchedBlocks(b *testing.B) {
	for _, sc := range benchScenarios() {
		b.Run(sc.name, func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				for _, addr := range sc.addrs {
					cachedBlocks := 0
					cachedBlocksByTier := map[string]int{}
					for _, lu := range sc.lookups {
						blockCount, byTier := matchedBlocks(lu.keys, lu.keyToPods, addr)
						cachedBlocks += blockCount
						for tier, n := range byTier {
							cachedBlocksByTier[tier] += n
						}
					}
					benchSinkInt += cachedBlocks
					benchSinkMap = cachedBlocksByTier
				}
			}
		})
	}
}
