package floats

// # Windowed overlap-add of lapped transforms (TDAC)
//
// The inner loop of the synthesis of every MDCT codec (AAC, Vorbis, Opus CELT, AC-3,
// MP3): two half-blocks, the second half of the previous block and the first half
// of the current one, are combined under a window of the overlap's length into the
// samples they cover. With n = len(a), j = n-1-i and a window w of 2n points:
//
//	dst[i]     = a[i]*w[2n-1-i] - b[j]*w[i]
//	dst[2n-1-i] = a[i]*w[i]     + b[j]*w[2n-1-i]      for i < n
//
// (the time-domain aliasing of the two blocks cancels when w satisfies the
// Princen-Bradley condition w[i]^2 + w[i+n]^2 = 1). Reading b and w from both ends
// and writing dst from both ends is what the auto-vectoriser cannot do and a
// SIMD kernel does with a lane reversal.
//
// Contract (go-fft#121, the fleet rule: a stated bound, not bit identity): every
// output is the exact value of its formula on the inputs, rounded with an error of
// at most 2 units of the last place of the larger of the two products' magnitudes,
// that is |error| <= 2 * 2^-p * (|a[i] w| + |b[j] w'|) with p = 24 (float32) or
// 53 (float64). A kernel may fuse a product with the sum (one rounding fewer) and
// the portable code does not, so the two can differ in the last bit; both are
// within the bound, which the tests assert against a 200-bit computation.
//
// dst must not overlap a, b or win.

// OverlapWindow computes the windowed overlap-add of a and b under win into dst
// for n = len(a): dst has at least 2n elements, b at least n and win at least 2n.
// It writes dst[:2n] and panics if a slice is too short.
func OverlapWindow(dst, a, b, win []float64) {
	n := len(a)
	if len(b) < n || len(win) < 2*n || len(dst) < 2*n {
		panic("floats: OverlapWindow: len(b) < len(a), or len(win) or len(dst) < 2*len(a)")
	}
	if n == 0 {
		return
	}
	overlapWindow64(dst[:2*n], a, b[:n], win[:2*n])
}

// Float32OverlapWindow is OverlapWindow on float32 values, computed in float32.
func Float32OverlapWindow(dst, a, b, win []float32) {
	n := len(a)
	if len(b) < n || len(win) < 2*n || len(dst) < 2*n {
		panic("floats: Float32OverlapWindow: len(b) < len(a), or len(win) or len(dst) < 2*len(a)")
	}
	if n == 0 {
		return
	}
	overlapWindow32(dst[:2*n], a, b[:n], win[:2*n])
}

// overlapRef64 is the portable loop and the reference of the kernels: every
// product and every sum rounded (the conversions keep the compiler from fusing).
func overlapRef64(dst, a, b, win []float64) { overlapTail64(dst, a, b, win, 0) }

// overlapTail64 is the loop from step from on: the kernels do the blocks that fit
// and this finishes what is left.
func overlapTail64(dst, a, b, win []float64, from int) {
	n := len(a)
	for i := from; i < n; i++ {
		j := n - 1 - i
		s0, s1 := a[i], b[j]
		wi, wj := win[i], win[2*n-1-i]
		dst[i] = float64(s0*wj) - float64(s1*wi)
		dst[2*n-1-i] = float64(s0*wi) + float64(s1*wj)
	}
}

// overlapRef32 is overlapRef64 in float32.
func overlapRef32(dst, a, b, win []float32) { overlapTail32(dst, a, b, win, 0) }

func overlapTail32(dst, a, b, win []float32, from int) {
	n := len(a)
	for i := from; i < n; i++ {
		j := n - 1 - i
		s0, s1 := a[i], b[j]
		wi, wj := win[i], win[2*n-1-i]
		dst[i] = float32(s0*wj) - float32(s1*wi)
		dst[2*n-1-i] = float32(s0*wi) + float32(s1*wj)
	}
}
