package floats

import (
	"math"
	"math/big"
	"math/rand"
	"testing"
)

const logRefPrec = 300

func bigF(x float64) *big.Float { return new(big.Float).SetPrec(logRefPrec).SetFloat64(x) }

// refAtanhSeries is 2*atanh(t) = 2*sum t^(2n+1)/(2n+1), for |t| <= 1/3.
func refAtanhSeries(t *big.Float) *big.Float {
	t2 := new(big.Float).SetPrec(logRefPrec).Mul(t, t)
	sum := new(big.Float).SetPrec(logRefPrec)
	pow := new(big.Float).SetPrec(logRefPrec).Set(t)
	for n := 0; n < 400; n++ {
		sum.Add(sum, new(big.Float).SetPrec(logRefPrec).Quo(pow, big.NewFloat(float64(2*n+1))))
		pow.Mul(pow, t2)
	}
	return sum.Mul(sum, big.NewFloat(2))
}

var logRefLn2 = refAtanhSeries(new(big.Float).SetPrec(logRefPrec).Quo(bigF(1), bigF(3)))

// refLog is log(x) for x > 0 to 300 bits, independent of the code under test:
// x = m * 2^e with m in [1,2), log m = 2*atanh((m-1)/(m+1)).
func refLog(x float64) *big.Float {
	m := bigF(x)
	e := m.MantExp(m) - 1 // x = m' * 2^(e+1), m' in [0.5,1) -> m = 2m' in [1,2)
	m.SetMantExp(m, 1)
	t := new(big.Float).SetPrec(logRefPrec).Quo(
		new(big.Float).SetPrec(logRefPrec).Sub(m, bigF(1)),
		new(big.Float).SetPrec(logRefPrec).Add(m, bigF(1)))
	r := refAtanhSeries(t)
	return r.Add(r, new(big.Float).SetPrec(logRefPrec).Mul(logRefLn2, big.NewFloat(float64(e))))
}

// logULP is |got - exact| in units of the spacing of doubles at exact.
func logULP(got float64, exact *big.Float) float64 {
	if exact.Sign() == 0 {
		return math.Abs(got) * math.Ldexp(1, 1074)
	}
	e := exact.MantExp(nil) - 1
	d := bigF(got)
	d.Sub(d, exact)
	d.SetMantExp(d, -(max(e-52, -1074)))
	f, _ := d.Float64()
	return math.Abs(f)
}

// TestLogTable checks every table entry against the properties log_data.c
// says it was built for: c = 1/invc lies in its subinterval of [0x1.6p-1,
// 0x1.6p0), logc is log(c) to within the rounding of invc, and 0x1.8p9 + logc
// is exact (so k*ln2hi + logc is computed without rounding).
func TestLogTable(t *testing.T) {
	for i, e := range logTab {
		lo := math.Float64frombits(logOff + uint64(i)<<45)
		hi := math.Float64frombits(logOff + uint64(i+1)<<45)
		if c := 1 / e.invc; c < lo || c >= hi {
			t.Errorf("entry %d: 1/invc = %v outside [%v, %v)", i, c, lo, hi)
		}
		neg := refLog(e.invc)
		d := bigF(e.logc)
		d.Add(d, neg) // logc - log(c) ~= logc + log(invc)
		if f, _ := d.Float64(); math.Abs(f) > 0x1p-52 {
			t.Errorf("entry %d: logc %v differs from -log(invc) by %v", i, e.logc, f)
		}
		if (0x1.8p9+e.logc)-0x1.8p9 != e.logc {
			t.Errorf("entry %d: 0x1.8p9 + logc is not exact", i)
		}
	}
}

// TestLogULP measures the worst error over random inputs across the whole
// positive range (log-uniform exponent), the region near 1 and subnormals.
func TestLogULP(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	var xs []float64
	for i := 0; i < 2500; i++ {
		xs = append(xs,
			math.Ldexp(1+rng.Float64(), rng.Intn(2046)-1022), // whole normal range
			1+(rng.Float64()*2-1)*0x1.2p-4,                   // near 1 (both paths)
			math.Ldexp(rng.Float64(), -1022),                 // subnormal
			0.5+rng.Float64()*2,                              // the table's own range
		)
	}
	worst, worstX, goWorst := 0.0, 0.0, 0.0
	for _, x := range xs {
		if x <= 0 {
			continue
		}
		ref := refLog(x)
		if e := logULP(log1(x), ref); e > worst {
			worst, worstX = e, x
		}
		goWorst = max(goWorst, logULP(math.Log(x), ref))
	}
	t.Logf("log: worst %.4f ULP at x=%v over %d inputs (math.Log: %.4f)", worst, worstX, len(xs), goWorst)
	if worst > 0.53 {
		t.Errorf("log error %.4f ULP at x=%v exceeds 0.53", worst, worstX)
	}
}

func TestLogSpecial(t *testing.T) {
	for _, c := range []struct{ x, want float64 }{
		{1, 0}, {0, math.Inf(-1)}, {math.Copysign(0, -1), math.Inf(-1)},
		{math.Inf(1), math.Inf(1)}, {math.E, 1}, {2, math.Ln2},
	} {
		if got := log1(c.x); got != c.want {
			t.Errorf("log(%v) = %v, want %v", c.x, got, c.want)
		}
	}
	for _, x := range []float64{-1, math.Inf(-1), math.NaN(), -1e-310} {
		if got := log1(x); !math.IsNaN(got) {
			t.Errorf("log(%v) = %v, want NaN", x, got)
		}
	}
	if got := log1(5e-324); got != -744.4400719213812 {
		t.Errorf("log(5e-324) = %v", got)
	}
}

// TestLogLoopMatchesLog1: Log's loop and log1 give the same bits everywhere.
func TestLogLoopMatchesLog1(t *testing.T) {
	rng := rand.New(rand.NewSource(4))
	xs := []float64{0, negZero, 1, 1 - 0x1p-4, 1 + 0x1.09p-4, 0x1p-1022, 5e-324, -3,
		math.Inf(1), math.Inf(-1), math.NaN(), math.MaxFloat64}
	for i := 0; i < 20000; i++ {
		xs = append(xs, math.Ldexp(rng.Float64()*2-0.5, rng.Intn(2100)-1074), 1+(rng.Float64()-0.5)*0.3)
	}
	got := make([]float64, len(xs))
	Log(got, xs)
	for i, x := range xs {
		w := log1(x)
		if math.Float64bits(got[i]) != math.Float64bits(w) && !(math.IsNaN(w) && math.IsNaN(got[i])) {
			t.Fatalf("Log(%v) = %v, log1 = %v", x, got[i], w)
		}
	}
}

// TestLogPanics: a short dst is refused, a long one is left untouched past len(src).
func TestLogPanics(t *testing.T) {
	long := []float64{7, 7, 7}
	Log(long, []float64{1, 1})
	if long[2] != 7 {
		t.Fatalf("Log wrote past len(src): %v", long)
	}
	defer func() {
		if recover() == nil {
			t.Fatal("Log with a short dst did not panic")
		}
	}()
	Log(make([]float64, 1), []float64{1, 1})
}
