package floats

import (
	"math"
	"math/big"
	"math/rand"
	"testing"
)

const refPrec = 300

// bigLn2 is ln 2 to refPrec bits: 2*atanh(1/3) = 2*sum (1/3)^(2n+1)/(2n+1).
func bigLn2() *big.Float {
	third := new(big.Float).SetPrec(refPrec).Quo(big.NewFloat(1).SetPrec(refPrec), big.NewFloat(3))
	ninth := new(big.Float).SetPrec(refPrec).Mul(third, third)
	sum := new(big.Float).SetPrec(refPrec)
	pow := new(big.Float).SetPrec(refPrec).Set(third)
	for n := 0; n < 300; n++ {
		term := new(big.Float).SetPrec(refPrec).Quo(pow, big.NewFloat(float64(2*n+1)))
		sum.Add(sum, term)
		pow.Mul(pow, ninth)
	}
	return sum.Mul(sum, big.NewFloat(2))
}

var ln2Big = bigLn2()

// refExp is exp(x) to refPrec bits, independent of the code under test:
// x = k*ln2 + r with |r| <= ln2/2, exp(r) by its Taylor series, times 2^k.
func refExp(x float64) *big.Float {
	bx := new(big.Float).SetPrec(refPrec).SetFloat64(x)
	q := new(big.Float).SetPrec(refPrec).Quo(bx, ln2Big)
	qf, _ := q.Float64()
	k := math.Round(qf)
	r := new(big.Float).SetPrec(refPrec).Mul(ln2Big, big.NewFloat(k))
	r.Sub(bx, r)
	sum := new(big.Float).SetPrec(refPrec).SetInt64(1)
	term := new(big.Float).SetPrec(refPrec).SetInt64(1)
	for n := 1; n < 80; n++ {
		term.Mul(term, r)
		term.Quo(term, big.NewFloat(float64(n)))
		sum.Add(sum, term)
	}
	return sum.SetMantExp(sum, int(k))
}

// ulpErr is |got - exact| in units of the float64 spacing at exact (2^-1074
// in the subnormal range).
func ulpErr(got float64, exact *big.Float) float64 {
	e := exact.MantExp(nil) - 1 // exact = m * 2^e, 1 <= m < 2
	ulpExp := max(e-52, -1074)
	d := new(big.Float).SetPrec(refPrec).SetFloat64(got)
	d.Sub(d, exact)
	d.SetMantExp(d, -ulpExp)
	f, _ := d.Float64()
	return math.Abs(f)
}

// TestExpTable re-derives every table entry from 2^(k/128) computed to 300
// bits (2 square-rooted seven times, then raised to k): H[k] must be the double
// nearest 2^(k/N) and tail[k] the double nearest (2^(k/N) - H)/H.
func TestExpTable(t *testing.T) {
	root := new(big.Float).SetPrec(refPrec).SetInt64(2)
	for i := 0; i < expTableBits; i++ {
		root.Sqrt(root)
	}
	e := new(big.Float).SetPrec(refPrec).SetInt64(1)
	for k := 0; k < expN; k++ {
		h := math.Float64frombits(expTab[2*k+1] + uint64(k)<<45)
		if want, _ := e.Float64(); h != want {
			t.Fatalf("H[%d] = %v, nearest double to 2^(k/N) is %v", k, h, want)
		}
		tail := new(big.Float).SetPrec(refPrec).SetFloat64(h)
		tail.Sub(e, tail)
		tail.Quo(tail, new(big.Float).SetFloat64(h))
		got := math.Float64frombits(expTab[2*k])
		if want, _ := tail.Float64(); got != want {
			t.Fatalf("tail[%d] = %v, want %v", k, got, want)
		}
		e.Mul(e, root)
	}
}

// TestExpULP measures the worst error against the 256-bit reference over the
// whole finite domain, the region near 0, the overflow edge and the subnormal
// range, and requires it to stay within Arm's documented 0.511 ULP (+ margin).
func TestExpULP(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	var xs []float64
	n := 5000
	if testing.Short() {
		n = 2000
	}
	for i := 0; i < n; i++ {
		xs = append(xs,
			rng.Float64()*1454.78-745,                        // the whole finite range
			(rng.Float64()*2-1)*math.Ldexp(1, -rng.Intn(60)), // near 0, log-uniform
			709+rng.Float64()*0.7827,                         // just below overflow
			-745.13+rng.Float64()*37,                         // subnormal and near-subnormal results
		)
	}
	worst, worstX, goWorst := 0.0, 0.0, 0.0
	for _, x := range xs {
		ref := refExp(x)
		if e := ulpErr(exp1(x), ref); e > worst {
			worst, worstX = e, x
		}
		goWorst = max(goWorst, ulpErr(math.Exp(x), ref))
	}
	t.Logf("exp: worst %.4f ULP at x=%v over %d inputs (math.Exp: %.4f)", worst, worstX, len(xs), goWorst)
	if worst > 0.52 {
		t.Errorf("exp error %.4f ULP at x=%v exceeds 0.52", worst, worstX)
	}
}

// TestExpLoopMatchesExp1: Exp's inlined main path and exp1 give the same bits
// for every input, special or not.
func TestExpLoopMatchesExp1(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	xs := []float64{0, 1e-300, math.Ldexp(1, -54), math.Ldexp(1, -55), 511.99, 512, -512,
		709.78, -745.13, 1000, math.Inf(1), math.Inf(-1), math.NaN()}
	for i := 0; i < 20000; i++ {
		xs = append(xs, rng.Float64()*1460-750, (rng.Float64()*2-1)*math.Ldexp(1, -rng.Intn(70)))
	}
	got := make([]float64, len(xs))
	Exp(got, xs)
	for i, x := range xs {
		if w := exp1(x); math.Float64bits(got[i]) != math.Float64bits(w) && !(math.IsNaN(w) && math.IsNaN(got[i])) {
			t.Fatalf("Exp(%v) = %v, exp1 = %v", x, got[i], w)
		}
	}
}

func TestExpSpecial(t *testing.T) {
	for _, c := range []struct{ x, want float64 }{
		{0, 1}, {math.Copysign(0, -1), 1}, {1e-300, 1}, {-1e-20, 1},
		{math.Inf(1), math.Inf(1)}, {math.Inf(-1), 0},
		{709.8, math.Inf(1)}, {1000, math.Inf(1)}, {2000, math.Inf(1)},
		{-745.2, 0}, {-1000, 0}, {-2000, 0},
		{1, math.E},
	} {
		if got := exp1(c.x); got != c.want {
			t.Errorf("exp(%v) = %v, want %v", c.x, got, c.want)
		}
	}
	// Against the reference: the subnormal range (just above ln(2^-1075) exp
	// rounds up to the smallest subnormal; arm64's math.Exp gives 0), and the
	// top of the finite range, where amd64's math.Exp (golang/go#81995) returns +Inf from
	// x >= 1023.5*ln2 ~ 709.436 (its k = round(x*log2e) reaches 1024 and the
	// biased exponent check calls that overflow, though fr*2^1024 is finite).
	for _, x := range []float64{-745.13, -745.1332, -740, -720.5, -708.5, -708.39,
		709.436, 709.5, 709.78, 709.782712893383} {
		want, _ := refExp(x).Float64()
		if got := exp1(x); got != want {
			t.Errorf("exp(%v) = %v, want %v", x, got, want)
		}
	}
	if got := exp1(math.NaN()); !math.IsNaN(got) {
		t.Errorf("exp(NaN) = %v", got)
	}
	dst := make([]float64, 3)
	Exp(dst, []float64{0, 1, math.Inf(-1)})
	for i, w := range []float64{1, math.E, 0} {
		if dst[i] != w {
			t.Errorf("Exp[%d] = %v, want %v", i, dst[i], w)
		}
	}
}

// TestExpPanics: a short dst is refused, a long one is left untouched past len(src).
func TestExpPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("Exp with a short dst did not panic")
		}
	}()
	long := []float64{7, 7, 7}
	Exp(long, []float64{0, 0})
	if long[2] != 7 {
		t.Fatalf("Exp wrote past len(src): %v", long)
	}
	Exp(make([]float64, 1), []float64{0, 0})
}
