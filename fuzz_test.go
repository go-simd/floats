package floats

import (
	"encoding/binary"
	"math"
	"testing"
)

// bytesToF64 decodes the fuzz corpus into a float64 slice, skipping non-finite
// values (NaN/Inf) so the relative-error comparison is meaningful — the kernels
// and reference handle them identically (IEEE arithmetic) but a tolerance check
// against ±Inf/NaN is not well defined.
func bytesToF64(data []byte) []float64 {
	n := len(data) / 8
	out := make([]float64, 0, n)
	for i := 0; i < n; i++ {
		f := math.Float64frombits(binary.LittleEndian.Uint64(data[i*8:]))
		if math.IsNaN(f) || math.IsInf(f, 0) {
			f = 0
		}
		// Clamp magnitude so squared sums stay finite.
		if math.Abs(f) > 1e150 {
			f = 0
		}
		out = append(out, f)
	}
	return out
}

func bytesToF32(data []byte) []float32 {
	n := len(data) / 4
	out := make([]float32, 0, n)
	for i := 0; i < n; i++ {
		f := math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
		if isNaN32(f) || math.IsInf(float64(f), 0) {
			f = 0
		}
		if math.Abs(float64(f)) > 1e18 {
			f = 0
		}
		out = append(out, f)
	}
	return out
}

// FuzzDot drives Dot/Float32Dot with two equal-length slices carved from the
// fuzz input and checks them against the lane-blocked scalar reference within
// the documented tolerance. This is the cross-architecture correctness gate:
// the same fuzzer runs under qemu for every non-native target.
func FuzzDot(f *testing.F) {
	f.Add([]byte{1, 2, 3, 4, 5, 6, 7, 8})
	f.Add(make([]byte, 8*20))
	f.Fuzz(func(t *testing.T, data []byte) {
		a := bytesToF64(data)
		half := len(a) / 2
		x, y := a[:half], a[half:half*2]
		// Condition-number-aware bound: a dot's error scales with Σ|aᵢbᵢ|, not the
		// (possibly near-zero) result, so cancellation between large terms must not
		// be flagged as a kernel bug.
		if got, want := Dot(x, y), dotLanes(x, y); !closeDot(got, want, absSumProd64(x, y), 1e-9) {
			t.Errorf("Dot mismatch: got %v want %v (n=%d)", got, want, len(x))
		}
		af := bytesToF32(data)
		fhalf := len(af) / 2
		xf, yf := af[:fhalf], af[fhalf:fhalf*2]
		if got, want := float64(Float32Dot(xf, yf)), float64(dotLanes32(xf, yf)); !closeDot(got, want, absSumProd32(xf, yf), 1e-3) {
			t.Errorf("Float32Dot mismatch: got %v want %v (n=%d)", got, want, len(xf))
		}
	})
}

// FuzzDistance is the analogous gate for the squared-difference reduction behind
// Distance/Float32Distance (and thus the L2 path of vector search).
func FuzzDistance(f *testing.F) {
	f.Add([]byte{1, 2, 3, 4, 5, 6, 7, 8})
	f.Add(make([]byte, 8*20))
	f.Fuzz(func(t *testing.T, data []byte) {
		a := bytesToF64(data)
		half := len(a) / 2
		x, y := a[:half], a[half:half*2]
		want := math.Sqrt(sumSqDiffLanes(x, y))
		if got := Distance(x, y); !closeRel(got, want, 1e-9) {
			t.Errorf("Distance mismatch: got %v want %v (n=%d)", got, want, len(x))
		}
		af := bytesToF32(data)
		fhalf := len(af) / 2
		xf, yf := af[:fhalf], af[fhalf:fhalf*2]
		wantf := math.Sqrt(float64(sumSqDiffLanes32(xf, yf)))
		if got := float64(Float32Distance(xf, yf)); !closeRel(got, wantf, 1e-3) {
			t.Errorf("Float32Distance mismatch: got %v want %v (n=%d)", got, wantf, len(xf))
		}
	})
}
