package floats

import (
	"math"
	"math/rand"
	"testing"
)

// relTol is the relative-error bound the SIMD kernels must meet against the
// lane-blocked scalar reference. The kernel and reference reduce in slightly
// different lane structures (hardware vector width vs laneCount), so equality is
// not bit-exact; a handful of ULP accumulated over a reduction is expected. 1e-5
// (f32) and 1e-12 (f64) comfortably bound that while still catching real bugs
// (a wrong lane, a dropped tail element, an operand swap all blow past it).
const (
	relTol64 = 1e-12
	relTol32 = 1e-5
)

func closeRel(got, want, tol float64) bool {
	if math.IsNaN(got) && math.IsNaN(want) {
		return true
	}
	d := math.Abs(got - want)
	a := math.Abs(want)
	if a < 1 {
		return d <= tol // absolute near zero
	}
	return d/a <= tol
}

// sizes spans below/at/above the vector strides (2,4,8) plus larger blocks and a
// few odd tails, so every kernel exercises its vector body, horizontal fold and
// scalar tail.
var sizes = []int{0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 33, 64, 100, 127, 256, 1000, 4095}

func randF64(n int, rng *rand.Rand) []float64 {
	s := make([]float64, n)
	for i := range s {
		s[i] = rng.NormFloat64()
	}
	return s
}

func randF32(n int, rng *rand.Rand) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = float32(rng.NormFloat64())
	}
	return s
}

func TestDot(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for _, n := range sizes {
		a, b := randF64(n, rng), randF64(n, rng)
		if got, want := Dot(a, b), dotLanes(a, b); !closeRel(got, want, relTol64) {
			t.Errorf("Dot n=%d: got %v want %v", n, got, want)
		}
		af, bf := randF32(n, rng), randF32(n, rng)
		if got, want := float64(Float32Dot(af, bf)), float64(dotLanes32(af, bf)); !closeRel(got, want, relTol32) {
			t.Errorf("Float32Dot n=%d: got %v want %v", n, got, want)
		}
	}
}

func TestSum(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	for _, n := range sizes {
		a := randF64(n, rng)
		if got, want := Sum(a), sumLanes(a); !closeRel(got, want, relTol64) {
			t.Errorf("Sum n=%d: got %v want %v", n, got, want)
		}
		af := randF32(n, rng)
		if got, want := float64(Float32Sum(af)), float64(sumLanes32(af)); !closeRel(got, want, relTol32) {
			t.Errorf("Float32Sum n=%d: got %v want %v", n, got, want)
		}
	}
}

func TestDistance(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	for _, n := range sizes {
		a, b := randF64(n, rng), randF64(n, rng)
		if got, want := Distance(a, b), math.Sqrt(sumSqDiffLanes(a, b)); !closeRel(got, want, relTol64) {
			t.Errorf("Distance n=%d: got %v want %v", n, got, want)
		}
		af, bf := randF32(n, rng), randF32(n, rng)
		want := math.Sqrt(float64(sumSqDiffLanes32(af, bf)))
		if got := float64(Float32Distance(af, bf)); !closeRel(got, want, relTol32) {
			t.Errorf("Float32Distance n=%d: got %v want %v", n, got, want)
		}
	}
}

func TestCosineSimilarity(t *testing.T) {
	rng := rand.New(rand.NewSource(4))
	for _, n := range sizes {
		if n == 0 {
			continue // 0/0 = NaN, handled separately
		}
		a, b := randF64(n, rng), randF64(n, rng)
		d := dotLanes(a, b)
		want := d / math.Sqrt(dotLanes(a, a)*dotLanes(b, b))
		if got := CosineSimilarity(a, b); !closeRel(got, want, relTol64) {
			t.Errorf("CosineSimilarity n=%d: got %v want %v", n, got, want)
		}
		af, bf := randF32(n, rng), randF32(n, rng)
		df := float64(dotLanes32(af, bf))
		wantf := df / math.Sqrt(float64(dotLanes32(af, af))*float64(dotLanes32(bf, bf)))
		if got := float64(Float32CosineSimilarity(af, bf)); !closeRel(got, wantf, relTol32) {
			t.Errorf("Float32CosineSimilarity n=%d: got %v want %v", n, got, wantf)
		}
	}
	// Identical unit-ish vectors -> cosine ~ 1.
	v := []float64{1, 2, 3, 4, 5}
	if got := CosineSimilarity(v, v); !closeRel(got, 1, 1e-12) {
		t.Errorf("CosineSimilarity(v,v) = %v want 1", got)
	}
	// Orthogonal -> 0.
	if got := CosineSimilarity([]float64{1, 0}, []float64{0, 1}); !closeRel(got, 0, 1e-12) {
		t.Errorf("orthogonal cosine = %v want 0", got)
	}
	// Zero magnitude -> NaN.
	if got := CosineSimilarity([]float64{0, 0}, []float64{1, 1}); !math.IsNaN(got) {
		t.Errorf("zero-magnitude cosine = %v want NaN", got)
	}
	if got := float64(Float32CosineSimilarity([]float32{0, 0}, []float32{1, 1})); !math.IsNaN(got) {
		t.Errorf("zero-magnitude f32 cosine = %v want NaN", got)
	}
}

func TestMinMax(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	for _, n := range sizes {
		if n == 0 {
			continue
		}
		a := randF64(n, rng)
		wmin, wmax := a[0], a[0]
		for _, v := range a {
			if v < wmin {
				wmin = v
			}
			if v > wmax {
				wmax = v
			}
		}
		if got := Min(a); got != wmin {
			t.Errorf("Min n=%d: got %v want %v", n, got, wmin)
		}
		if got := Max(a); got != wmax {
			t.Errorf("Max n=%d: got %v want %v", n, got, wmax)
		}
		af := randF32(n, rng)
		fwmin, fwmax := af[0], af[0]
		for _, v := range af {
			if v < fwmin {
				fwmin = v
			}
			if v > fwmax {
				fwmax = v
			}
		}
		if got := Float32Min(af); got != fwmin {
			t.Errorf("Float32Min n=%d: got %v want %v", n, got, fwmin)
		}
		if got := Float32Max(af); got != fwmax {
			t.Errorf("Float32Max n=%d: got %v want %v", n, got, fwmax)
		}
	}
	// NaN propagation.
	if !math.IsNaN(Min([]float64{1, math.NaN(), 2})) {
		t.Error("Min should propagate NaN")
	}
	if !math.IsNaN(Max([]float64{1, math.NaN(), 2})) {
		t.Error("Max should propagate NaN")
	}
	if !isNaN32(Float32Min([]float32{1, float32(math.NaN()), 2})) {
		t.Error("Float32Min should propagate NaN")
	}
	if !isNaN32(Float32Max([]float32{1, float32(math.NaN()), 2})) {
		t.Error("Float32Max should propagate NaN")
	}
}

func TestPanics(t *testing.T) {
	mustPanic := func(name string, f func()) {
		defer func() {
			if recover() == nil {
				t.Errorf("%s did not panic", name)
			}
		}()
		f()
	}
	mustPanic("Dot", func() { Dot([]float64{1}, []float64{1, 2}) })
	mustPanic("Distance", func() { Distance([]float64{1}, []float64{1, 2}) })
	mustPanic("Cosine", func() { CosineSimilarity([]float64{1}, []float64{1, 2}) })
	mustPanic("Min", func() { Min(nil) })
	mustPanic("Max", func() { Max(nil) })
	mustPanic("F32Dot", func() { Float32Dot([]float32{1}, []float32{1, 2}) })
	mustPanic("F32Distance", func() { Float32Distance([]float32{1}, []float32{1, 2}) })
	mustPanic("F32Cosine", func() { Float32CosineSimilarity([]float32{1}, []float32{1, 2}) })
	mustPanic("F32Min", func() { Float32Min(nil) })
	mustPanic("F32Max", func() { Float32Max(nil) })
}

// known-value sanity checks (exact for small integer inputs).
func TestKnownValues(t *testing.T) {
	a := []float64{1, 2, 3, 4}
	b := []float64{5, 6, 7, 8}
	if got := Dot(a, b); got != 70 {
		t.Errorf("Dot = %v want 70", got)
	}
	if got := Sum(a); got != 10 {
		t.Errorf("Sum = %v want 10", got)
	}
	if got := Distance([]float64{0, 0}, []float64{3, 4}); got != 5 {
		t.Errorf("Distance = %v want 5", got)
	}
	af := []float32{1, 2, 3, 4}
	bf := []float32{5, 6, 7, 8}
	if got := Float32Dot(af, bf); got != 70 {
		t.Errorf("Float32Dot = %v want 70", got)
	}
	if got := Float32Sum(af); got != 10 {
		t.Errorf("Float32Sum = %v want 10", got)
	}
	if got := Float32Distance([]float32{0, 0}, []float32{3, 4}); got != 5 {
		t.Errorf("Float32Distance = %v want 5", got)
	}
}
