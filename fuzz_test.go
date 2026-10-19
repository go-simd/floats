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

// rawF64 decodes the fuzz input into float64s WITHOUT filtering: NaN, ±Inf,
// subnormals and huge values all stay, because the elementwise kernels promise
// bit identity with math.FMA for every input, not just finite ones.
func rawF64(data []byte) []float64 {
	out := make([]float64, len(data)/8)
	for i := range out {
		out[i] = math.Float64frombits(binary.LittleEndian.Uint64(data[i*8:]))
	}
	return out
}

// FuzzAxpy checks Axpy, Xpby, Scale and AxpyDot against the math.FMA reference
// bit for bit (the updated slice) and within the reduction tolerance (the
// AxpyDot sum, when finite). The scalar and the start offset come from the
// input too, so lengths, alignments and IEEE specials are all explored.
func FuzzAxpy(f *testing.F) {
	f.Add(uint64(0x3ff8000000000000), uint8(0), []byte{1, 2, 3, 4, 5, 6, 7, 8})
	f.Add(uint64(0x7ff8000000000001), uint8(1), make([]byte, 8*40))
	f.Add(uint64(0xbfe0000000000000), uint8(3), make([]byte, 8*61))
	f.Fuzz(func(t *testing.T, alphaBits uint64, offset uint8, data []byte) {
		alpha := math.Float64frombits(alphaBits)
		v := rawF64(data)
		off := int(offset % 4)
		if off > len(v) {
			off = len(v)
		}
		v = v[off:]
		n := len(v) / 3
		x, y0, w := v[:n], v[n:2*n], v[2*n:3*n]

		y, want := append([]float64(nil), y0...), append([]float64(nil), y0...)
		Axpy(alpha, x, y)
		axpyRef(alpha, x, want)
		checkSame(t, "FuzzAxpy Axpy", y, want, n, off)

		Xpby(x, alpha, y)
		xpbyRef(alpha, x, want)
		checkSame(t, "FuzzAxpy Xpby", y, want, n, off)

		Scale(alpha, y)
		scaleRef(alpha, want)
		checkSame(t, "FuzzAxpy Scale", y, want, n, off)

		got := AxpyDot(alpha, x, y, w)
		wantDot := axpyDotRef(alpha, x, want, w)
		checkSame(t, "FuzzAxpy AxpyDot", y, want, n, off)
		if mag := absSumProd64(want, w); !math.IsInf(mag, 0) && !math.IsNaN(mag) && !closeDot(got, wantDot, mag, 1e-9) {
			t.Errorf("AxpyDot sum: got %v want %v (n=%d)", got, wantDot, n)
		}
	})
}
