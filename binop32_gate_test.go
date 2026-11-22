//go:build loong64 || ppc64le || riscv64 || s390x

package floats

import "testing"

// TestBinop32Gates reruns the float32 elementwise suites with the CPU-feature gate
// of the kernels off, so the portable fallback branch of binop32_<arch>.go is
// covered (and checked) on a CPU that has the kernels, and again with it on, so
// the kernel branch is exercised wherever the CPU has the feature.
func TestBinop32Gates(t *testing.T) {
	for _, on := range []bool{false, true} {
		restore, had := setBinop32Gate(on)
		name := "fallback"
		if on {
			name = "kernel"
			if !had {
				restore()
				t.Log("CPU lacks the feature; kernel branch not exercised on this host")
				continue
			}
		}
		for _, tc := range []struct {
			name string
			fn   func(*testing.T)
		}{
			{"BitIdentical", TestBinop32BitIdentical},
			{"Sqrt", TestSqrt32BitIdentical},
			{"Aliasing", TestBinop32Aliasing},
			{"Lengths", TestBinop32Lengths},
			{"SqrtSpecials", TestSqrt32Specials},
		} {
			t.Run(name+"/"+tc.name, tc.fn)
		}
		restore()
	}
}
