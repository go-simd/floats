package floats

import (
	"math"
	"math/big"
	"math/rand"
	"testing"
)

// The reductions (Dot, Sum, the squared distance, and their float32 forms) sum
// in an order that is NOT specified: the amd64 kernels run several independent
// vector accumulators to hide the FMA/ADD latency, so the order depends on the
// kernel. The contract is therefore an error bound, not equality with another
// float64 code: for n terms, recursive summation in any order, with or without
// FMA, satisfies
//
//	|got - exact| <= gamma_n * sum|t_i|,  gamma_n = n*u / (1 - n*u),
//
// u the unit roundoff (2^-53, 2^-24). exact is computed with math/big at 512
// bits, so it is exact for these inputs.

func redExact(terms func(i int) *big.Float, n int) *big.Float {
	s := new(big.Float).SetPrec(512)
	for i := 0; i < n; i++ {
		s.Add(s, terms(i))
	}
	return s
}

func redBig(x float64) *big.Float { return new(big.Float).SetPrec(512).SetFloat64(x) }

// redErrBound reports whether |got-exact| <= gamma_n*mag (with a 1.01 slack for
// the final conversion of mag to float64).
func redErrBound(got float64, exact *big.Float, n int, mag, u float64) (ok bool, err float64, bound float64) {
	e := new(big.Float).SetPrec(512).Sub(redBig(got), exact)
	err, _ = new(big.Float).Abs(e).Float64()
	nu := float64(n) * u
	bound = nu / (1 - nu) * mag * 1.01
	return err <= bound, err, bound
}

type genKind int

const (
	genNormal genKind = iota
	genCancel         // large alternating-sign terms: sum << sum|t|
	genMixedScale
)

func redGenVec(n int, rng *rand.Rand, k genKind) []float64 {
	v := make([]float64, n)
	for i := range v {
		x := rng.NormFloat64()
		switch k {
		case genCancel:
			x = math.Copysign(1e6+rng.Float64(), float64(1-2*(i&1))) * (1 + 1e-3*rng.NormFloat64())
		case genMixedScale:
			x *= math.Ldexp(1, rng.Intn(40)-20)
		}
		v[i] = x
	}
	return v
}

var boundSizes = []int{0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 35, 63, 64, 65, 100, 127, 128, 129, 255, 256, 257, 1000, 4097, 100003}

func TestReductionsMeetErrorBound(t *testing.T) {
	rng := rand.New(rand.NewSource(20261122))
	const u64, u32 = 0x1p-53, 0x1p-24
	for _, kind := range []genKind{genNormal, genCancel, genMixedScale} {
		for _, n := range boundSizes {
			a, b := redGenVec(n, rng, kind), redGenVec(n, rng, kind)

			// float64 Dot
			ex := redExact(func(i int) *big.Float { return new(big.Float).SetPrec(512).Mul(redBig(a[i]), redBig(b[i])) }, n)
			var mag float64
			for i := range a {
				mag += math.Abs(a[i] * b[i])
			}
			if ok, e, bd := redErrBound(Dot(a, b), ex, n, mag, u64); !ok {
				t.Errorf("Dot kind=%d n=%d: error %g > bound %g", kind, n, e, bd)
			}

			// float64 Sum
			ex = redExact(func(i int) *big.Float { return redBig(a[i]) }, n)
			mag = 0
			for _, v := range a {
				mag += math.Abs(v)
			}
			if ok, e, bd := redErrBound(Sum(a), ex, n, mag, u64); !ok {
				t.Errorf("Sum kind=%d n=%d: error %g > bound %g", kind, n, e, bd)
			}

			// float64 sum of squared differences (a-b is rounded once, then
			// squared and accumulated: n+1 roundings per term at most, so n+2
			// in the bound).
			ex = redExact(func(i int) *big.Float {
				d := new(big.Float).SetPrec(512).Sub(redBig(a[i]), redBig(b[i]))
				return d.Mul(d, d)
			}, n)
			mag = 0
			for i := range a {
				d := a[i] - b[i]
				mag += d * d
			}
			if ok, e, bd := redErrBound(sumSqDiff(a, b), ex, n+2, mag, u64); !ok {
				t.Errorf("sumSqDiff kind=%d n=%d: error %g > bound %g", kind, n, e, bd)
			}

			// float32 forms: inputs are exact float32 values, the oracle is the
			// exact sum of their float64-exact products.
			af, bfv := make([]float32, n), make([]float32, n)
			for i := range af {
				af[i], bfv[i] = float32(a[i]), float32(b[i])
			}
			ex = redExact(func(i int) *big.Float {
				return new(big.Float).SetPrec(512).Mul(redBig(float64(af[i])), redBig(float64(bfv[i])))
			}, n)
			mag = 0
			for i := range af {
				mag += math.Abs(float64(af[i]) * float64(bfv[i]))
			}
			if ok, e, bd := redErrBound(float64(Float32Dot(af, bfv)), ex, n+1, mag, u32); !ok {
				t.Errorf("Float32Dot kind=%d n=%d: error %g > bound %g", kind, n, e, bd)
			}
			ex = redExact(func(i int) *big.Float { return redBig(float64(af[i])) }, n)
			mag = 0
			for _, v := range af {
				mag += math.Abs(float64(v))
			}
			if ok, e, bd := redErrBound(float64(Float32Sum(af)), ex, n+1, mag, u32); !ok {
				t.Errorf("Float32Sum kind=%d n=%d: error %g > bound %g", kind, n, e, bd)
			}
			ex = redExact(func(i int) *big.Float {
				d := new(big.Float).SetPrec(512).Sub(redBig(float64(af[i])), redBig(float64(bfv[i])))
				return d.Mul(d, d)
			}, n)
			mag = 0
			for i := range af {
				d := float64(af[i]) - float64(bfv[i])
				mag += d * d
			}
			if ok, e, bd := redErrBound(float64(sumSqDiff32(af, bfv)), ex, n+3, mag, u32); !ok {
				t.Errorf("sumSqDiff32 kind=%d n=%d: error %g > bound %g", kind, n, e, bd)
			}
		}
	}
}
