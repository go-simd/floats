//go:build amd64

package floats

import "testing"

// TestPSMixPlan32PortableForced: the path of a CPU without AVX2+FMA, forced on any host,
// satisfies the same bound and the same checks.
func TestPSMixPlan32PortableForced(t *testing.T) {
	saved := hasAVX2FMA
	defer func() { hasAVX2FMA = saved }()
	hasAVX2FMA = false
	TestPSMixPlan32Bound(t)
	TestPSMixPlan32OneRun(t)
	TestPSMixPlan32Overwrite(t)
}
