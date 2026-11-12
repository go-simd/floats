package floats

import (
	"fmt"
	"math"
	"math/big"
	"math/rand"
	"testing"
)

func bigDotIdx(a, b []float64, idx []int) *big.Float {
	s := new(big.Float).SetPrec(300)
	for _, i := range idx {
		p := new(big.Float).SetPrec(300).Mul(new(big.Float).SetFloat64(a[i]), new(big.Float).SetFloat64(b[i]))
		s.Add(s, p)
	}
	return s
}

// The sums agree with a 300-bit reference within the recursive-sum bound
// n·eps·Σ|a_i b_i| for every length mod 4, with repeated and permuted indices.
func TestDotIdxDotGather(t *testing.T) {
	r := rand.New(rand.NewSource(5))
	for n := 0; n < 40; n++ {
		a, b, d := make([]float64, 40), make([]float64, 40), make([]float64, n)
		for i := range a {
			a[i], b[i] = r.NormFloat64(), r.NormFloat64()
		}
		for i := range d {
			d[i] = r.NormFloat64()
		}
		idx := make([]int, n)
		for k := range idx {
			idx[k] = r.Intn(40)
		}
		abs := 0.0
		for _, i := range idx {
			abs += math.Abs(a[i] * b[i])
		}
		want, _ := bigDotIdx(a, b, idx).Float64()
		if got := DotIdx(a, b, idx); math.Abs(got-want) > float64(n+1)*0x1p-52*abs {
			t.Fatalf("DotIdx n=%d: %g, want %g", n, got, want)
		}
		var w2 float64
		wb := new(big.Float).SetPrec(300)
		abs2 := 0.0
		for k, i := range idx {
			wb.Add(wb, new(big.Float).SetPrec(300).Mul(new(big.Float).SetFloat64(a[i]), new(big.Float).SetFloat64(d[k])))
			abs2 += math.Abs(a[i] * d[k])
		}
		w2, _ = wb.Float64()
		if got := DotGather(a, idx, d); math.Abs(got-w2) > float64(n+1)*0x1p-52*abs2 {
			t.Fatalf("DotGather n=%d: %g, want %g", n, got, w2)
		}
		e := append([]float64(nil), d...)
		AxpyGather(2, a, idx, e)
		for k, i := range idx {
			if e[k] != math.FMA(2, a[i], d[k]) {
				t.Fatalf("AxpyGather n=%d k=%d", n, k)
			}
		}
	}
}

func TestGatherPanics(t *testing.T) {
	for name, f := range map[string]func(){
		"DotIdx":      func() { DotIdx([]float64{1}, []float64{1}, []int{1}) },
		"DotGather":   func() { DotGather([]float64{1}, []int{0, 0}, []float64{1}) },
		"AxpyGather":  func() { AxpyGather(1, []float64{1}, []int{0, 0}, []float64{1}) },
		"DotGatherIx": func() { DotGather([]float64{1}, []int{3}, []float64{1}) },
		// len(d) < len(idx) but cap(d) >= len(idx): must still panic.
		"DotGatherCap":  func() { DotGather([]float64{1}, []int{0, 0}, make([]float64, 1, 8)) },
		"AxpyGatherCap": func() { AxpyGather(1, []float64{1}, []int{0, 0}, make([]float64, 1, 8)) },
		"DotIdxNeg":     func() { DotIdx([]float64{1}, []float64{1}, []int{-1}) },
	} {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("%s did not panic", name)
				}
			}()
			f()
		}()
	}
}

// FuzzGather carves a, b, d and an index list from the input and checks the
// three kernels against the 300-bit reference within n·(eps·Σ|a_i b_i| + 2^-1074)
// (the absolute term covers products that underflow to subnormals), and
// AxpyGather bit for bit against math.FMA.
func FuzzGather(f *testing.F) {
	f.Add([]byte{1, 2, 3, 4, 5, 6, 7, 8})
	f.Add(make([]byte, 8*40))
	f.Fuzz(func(t *testing.T, data []byte) {
		v := bytesToF64(data)
		m := len(v) / 3
		if m == 0 {
			return
		}
		a, b, d := v[:m], v[m:2*m], v[2*m:3*m]
		idx := make([]int, len(d))
		for k := range idx {
			idx[k] = int(math.Float64bits(d[k])>>7) % m
		}
		n := float64(len(idx) + 1)
		abs, abs2 := 0.0, 0.0
		wb := new(big.Float).SetPrec(300)
		for k, i := range idx {
			abs += math.Abs(a[i] * b[i])
			abs2 += math.Abs(a[i] * d[k])
			wb.Add(wb, new(big.Float).SetPrec(300).Mul(new(big.Float).SetFloat64(a[i]), new(big.Float).SetFloat64(d[k])))
		}
		want, _ := bigDotIdx(a, b, idx).Float64()
		if got := DotIdx(a, b, idx); math.Abs(got-want) > n*(0x1p-52*abs+0x1p-1074) {
			t.Fatalf("DotIdx: %g, want %g", got, want)
		}
		want2, _ := wb.Float64()
		if got := DotGather(a, idx, d); math.Abs(got-want2) > n*(0x1p-52*abs2+0x1p-1074) {
			t.Fatalf("DotGather: %g, want %g", got, want2)
		}
		e := append([]float64(nil), d...)
		AxpyGather(0.5, a, idx, e)
		for k, i := range idx {
			if e[k] != math.FMA(0.5, a[i], d[k]) {
				t.Fatalf("AxpyGather k=%d", k)
			}
		}
	})
}

var gatherSink float64

func gatherData(n int) (a, b, d []float64, idx []int) {
	r := rand.New(rand.NewSource(1))
	a, b, d = make([]float64, 4*n), make([]float64, 4*n), make([]float64, n)
	for i := range a {
		a[i], b[i] = r.NormFloat64(), r.NormFloat64()
	}
	for i := range d {
		d[i] = r.NormFloat64()
	}
	idx = r.Perm(4 * n)[:n] // active set: n scattered indices of 4n
	return
}

func BenchmarkGather(b *testing.B) {
	for _, n := range []int{10, 32, 100} {
		x, y, d, idx := gatherData(n)
		b.Run(fmt.Sprintf("DotIdx/n=%d", n), func(b *testing.B) {
			for b.Loop() {
				gatherSink = DotIdx(x, y, idx)
			}
		})
		b.Run(fmt.Sprintf("DotIdxScalar/n=%d", n), func(b *testing.B) {
			for b.Loop() {
				s := 0.0
				for _, i := range idx {
					s += x[i] * y[i]
				}
				gatherSink = s
			}
		})
		b.Run(fmt.Sprintf("DotGather/n=%d", n), func(b *testing.B) {
			for b.Loop() {
				gatherSink = DotGather(x, idx, d)
			}
		})
		b.Run(fmt.Sprintf("DotGatherScalar/n=%d", n), func(b *testing.B) {
			for b.Loop() {
				s := 0.0
				for k, i := range idx {
					s += x[i] * d[k]
				}
				gatherSink = s
			}
		})
		b.Run(fmt.Sprintf("AxpyGather/n=%d", n), func(b *testing.B) {
			for b.Loop() {
				AxpyGather(1e-9, x, idx, d)
			}
		})
		b.Run(fmt.Sprintf("AxpyGatherScalar/n=%d", n), func(b *testing.B) {
			for b.Loop() {
				for k, i := range idx {
					d[k] += 1e-9 * x[i]
				}
			}
		})
	}
}
