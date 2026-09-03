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
	"math/rand"
	"testing"

	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/util/sets"

	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
)

// Reference implementations: the straightforward two-pass semantics, used only
// to differentially verify matchedBlocks.

func refMatchedBlockCount(keys []kvblock.BlockHash, keyToPods map[kvblock.BlockHash][]kvblock.PodEntry, podID string) int {
	count := 0
	for _, key := range keys {
		found := false
		for _, e := range keyToPods[key] {
			if e.PodIdentifier == podID {
				found = true
				break
			}
		}
		if !found {
			break
		}
		count++
	}
	return count
}

func refMatchedBlockCountByTier(keys []kvblock.BlockHash, keyToPods map[kvblock.BlockHash][]kvblock.PodEntry, podID string) map[string]int {
	counts := map[string]int{}
	var alive sets.Set[string]
	for _, key := range keys {
		tiersAtKey := sets.New[string]()
		for _, e := range keyToPods[key] {
			if e.PodIdentifier == podID {
				if e.Speculative {
					tiersAtKey.Insert(attrprefix.SpeculativeTierKey)
				} else {
					tiersAtKey.Insert(e.DeviceTier)
				}
			}
		}
		if alive == nil {
			alive = tiersAtKey
		} else {
			alive = alive.Intersection(tiersAtKey)
		}
		if alive.Len() == 0 {
			break
		}
		for tier := range alive {
			counts[tier]++
		}
	}
	return counts
}

// TestMatchedBlocksDifferentialFuzz compares matchedBlocks against the
// straightforward reference on randomized index shapes.
func TestMatchedBlocksDifferentialFuzz(t *testing.T) {
	rnd := rand.New(rand.NewSource(42))
	pods := []string{"10.0.0.1:8000", "10.0.0.2:8000", "10.0.0.3:8000"}
	tiers := []string{"gpu", "cpu", ""}

	for iter := 0; iter < 20000; iter++ {
		n := rnd.Intn(10) // 0..9 keys
		keys := make([]kvblock.BlockHash, n)
		for i := range keys {
			keys[i] = kvblock.BlockHash(i + 1)
		}

		keyToPods := map[kvblock.BlockHash][]kvblock.PodEntry{}
		for _, k := range keys {
			if rnd.Intn(3) == 0 {
				continue // key absent from lookup result
			}
			entries := rnd.Intn(4)
			for e := 0; e < entries; e++ {
				keyToPods[k] = append(keyToPods[k], kvblock.PodEntry{
					PodIdentifier: pods[rnd.Intn(len(pods))],
					DeviceTier:    tiers[rnd.Intn(len(tiers))],
					Speculative:   rnd.Intn(4) == 0,
				})
			}
		}

		for _, pod := range pods {
			wantCount := refMatchedBlockCount(keys, keyToPods, pod)
			wantByTier := refMatchedBlockCountByTier(keys, keyToPods, pod)
			gotCount, gotByTier := matchedBlocks(keys, keyToPods, pod)

			assert.Equal(t, wantCount, gotCount, "iter %d pod %s keys %v pods %v", iter, pod, keys, keyToPods)
			assert.Equal(t, wantByTier, gotByTier, "iter %d pod %s keys %v pods %v", iter, pod, keys, keyToPods)
			if t.Failed() {
				t.Fatalf("divergence at iter %d", iter)
			}
		}
	}
}
