//go:build amd64

package floats

import "testing"

// TestAllPass3Split32PortableForced: the path of a CPU without AVX2+FMA, forced on any host,
// satisfies the same bound at every row count.
func TestAllPass3Split32PortableForced(t *testing.T) {
	saved := hasAVX2FMA
	defer func() { hasAVX2FMA = saved }()
	hasAVX2FMA = false
	TestAllPass3Split32Bound(t)
	TestAllPass3Split32Matches64(t)
	TestAllPass3Plan32RunTickMatches(t)
}

// TestAllPass3Split32TailMask: the mask of a tail of r rows has its first r entries set.
func TestAllPass3Split32TailMask(t *testing.T) {
	for r := 1; r < 8; r++ {
		m := allPass32TailMask[8-r : 16-r]
		for i, v := range m {
			if (v != 0) != (i < r) {
				t.Fatalf("tail of %d rows: mask %v", r, m)
			}
		}
	}
}
