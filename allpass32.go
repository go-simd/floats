package floats

import "unsafe"

// # Three-link all-pass rows in float32 split planes (Parametric Stereo)
//
// AllPass3Split32 is AllPass3 (allpass.go) for float32 samples held as two planes, one
// of real parts and one of imaginary parts, so that a 256-bit register holds eight rows
// (a 128-bit NEON register four) instead of two (one). The recurrence is the same: for each
// row k,
//
//	x      = Phi[k] * in[k]
//	for l = 0, 1, 2:
//	    p  = Q_l[k] * old_l[k]          (the delayed state, rotated)
//	    c  = x + C_l[k]*p               (the new state of the link)
//	    cur_l[k] = c
//	    x  = p - C_l[k]*c               (the output of the link)
//	out[k] = x
//
// with a complex product written re = dr*ar - di*ai, im = dr*ai + di*ar for the coefficient a
// and the data d. A complex number is a pair of elements at the same index of a Split32.
//
// Contract: an error bound, not bit identity with another code. The AVX2+FMA and NEON kernels
// fuse the second product of each component and the multiply-add of each link, the portable
// loop rounds every product; they differ in the last bits, and every one of them satisfies the
// bound below. With u = 2^-24 and exact arithmetic on the same float32 inputs, the complex
// error (the modulus of the difference) of every p, c and x is bounded by the recurrence, in
// which |.| is the modulus of the exact value:
//
//	e(x_{-1}) <= 3u |Phi| |in|
//	e(p_l)    <= 3u |Q_l| |old_l|
//	e(c_l)    <= e(x_{l-1}) + |C_l| e(p_l) + u (|C_l| |p_l| + |c_l|)
//	e(x_l)    <= e(p_l) + |C_l| e(c_l) + u (|C_l| |c_l| + |x_l|)
//
// (a component of a complex product carries at most three roundings, each at most u times the
// magnitude of the value rounded, and their sum is at most 3u|a||d| in modulus; a link carries
// two, of C*p and of the sum). It is a few units of u times the largest magnitude in the chain
// for |C| <= 1. The tests assert it for out and for every cur_l against a 200-bit computation
// (TestAllPass3Split32Bound, FuzzAllPass3Split32). Inputs must be finite and keep every
// intermediate value within the float32 range; the bound is relative to the magnitudes, so it
// does not cover values that underflow to a denormal.
//
// Memory: in, old_l and cur_l must hold at least len(out.Re) elements in each plane, as must
// each of the eleven tables of t and out.Im; nothing past len(out.Re) is read or written (the
// tail of eight rows is a masked access on AVX2, single-lane on NEON). The written slices (out
// and cur_l, both planes) must not overlap one another or any operand that is read, not even
// exactly: checked, panics with a message saying so. len(out.Re) = 0 does nothing.

// Split32 is a complex slice in two float32 planes: element k is complex(Re[k], Im[k]).
type Split32 struct{ Re, Im []float32 }

// AllPassRows32 holds the constants of the rows of a layout: Phi the rotation of the
// pre-delay, Q_l the rotation of link l (each as two planes), C_l its real coefficient.
type AllPassRows32 struct {
	PhiRe, PhiIm, Q0Re, Q0Im, Q1Re, Q1Im, Q2Re, Q2Im []float32
	C0, C1, C2                                       []float32
}

// AllPass3Split32 computes the decorrelator rows described above for k < len(out.Re). It
// panics if an operand is shorter than that or a written slice overlaps another operand.
func AllPass3Split32(out, in Split32, t *AllPassRows32, old, cur *[3]Split32) {
	n := len(out.Re)
	if n == 0 {
		return
	}
	if allPass32Short(n, out, in, t, old, cur) {
		panic("floats: AllPass3Split32: an operand is shorter than len(out.Re)")
	}
	if spansMeet(allPass32Written(n, out, cur), allPass32Read(n, in, t, old)) {
		panic("floats: AllPass3Split32: a written slice overlaps another operand")
	}
	allPass3Split32(n, out, in, t, old, cur)
}

// allPass32Short reports whether any operand holds fewer than n elements (out.Re has n by
// the definition of n).
func allPass32Short(n int, out, in Split32, t *AllPassRows32, old, cur *[3]Split32) bool {
	short := len(out.Im) < n || len(in.Re) < n || len(in.Im) < n ||
		len(t.PhiRe) < n || len(t.PhiIm) < n || len(t.Q0Re) < n || len(t.Q0Im) < n ||
		len(t.Q1Re) < n || len(t.Q1Im) < n || len(t.Q2Re) < n || len(t.Q2Im) < n ||
		len(t.C0) < n || len(t.C1) < n || len(t.C2) < n
	for l := range old {
		short = short || len(old[l].Re) < n || len(old[l].Im) < n || len(cur[l].Re) < n || len(cur[l].Im) < n
	}
	return short
}

func fspan32(s []float32, n int) span {
	lo := int64(uint64(uintptr(unsafe.Pointer(unsafe.SliceData(s)))))
	return span{lo, lo + int64(n)*4}
}

// allPass32Written is the byte ranges of the n first elements of everything the kernel writes.
func allPass32Written(n int, out Split32, cur *[3]Split32) []span {
	w := make([]span, 0, 8)
	w = append(w, fspan32(out.Re, n), fspan32(out.Im, n))
	for l := range cur {
		w = append(w, fspan32(cur[l].Re, n), fspan32(cur[l].Im, n))
	}
	return w
}

// allPass32Read is the byte ranges of the n first elements of everything the kernel reads.
func allPass32Read(n int, in Split32, t *AllPassRows32, old *[3]Split32) []span {
	r := make([]span, 0, 19)
	r = append(r, fspan32(in.Re, n), fspan32(in.Im, n),
		fspan32(t.PhiRe, n), fspan32(t.PhiIm, n), fspan32(t.Q0Re, n), fspan32(t.Q0Im, n),
		fspan32(t.Q1Re, n), fspan32(t.Q1Im, n), fspan32(t.Q2Re, n), fspan32(t.Q2Im, n),
		fspan32(t.C0, n), fspan32(t.C1, n), fspan32(t.C2, n))
	for l := range old {
		r = append(r, fspan32(old[l].Re, n), fspan32(old[l].Im, n))
	}
	return r
}

// allPass3Split32Ref is the portable loop: every product rounded to float32 on its own (the
// conversions keep a compiler from fusing them), and the reference of the kernels' bound.
func allPass3Split32Ref(n int, out, in Split32, t *AllPassRows32, old, cur *[3]Split32) {
	for k := 0; k < n; k++ {
		xr, xi := cmul32(t.PhiRe[k], t.PhiIm[k], in.Re[k], in.Im[k])
		q := [3][2][]float32{{t.Q0Re, t.Q0Im}, {t.Q1Re, t.Q1Im}, {t.Q2Re, t.Q2Im}}
		c := [3][]float32{t.C0, t.C1, t.C2}
		for l := 0; l < 3; l++ {
			pr, pi := cmul32(q[l][0][k], q[l][1][k], old[l].Re[k], old[l].Im[k])
			cl := c[l][k]
			cr, ci := xr+float32(cl*pr), xi+float32(cl*pi)
			cur[l].Re[k], cur[l].Im[k] = cr, ci
			xr, xi = pr-float32(cl*cr), pi-float32(cl*ci)
		}
		out.Re[k], out.Im[k] = xr, xi
	}
}

// cmul32 is the product of the coefficient (ar, ai) and the data (dr, di), each product
// rounded to float32 on its own.
func cmul32(ar, ai, dr, di float32) (float32, float32) {
	return float32(dr*ar) - float32(di*ai), float32(dr*ai) + float32(di*ar)
}
