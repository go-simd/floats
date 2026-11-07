package floats

import "math"

// log32 is the float32 log of v0.32.0 and v0.33.0, kept as an independent
// oracle for logCR (TestLogCRMatchesLog32): it widens to this package's
// float64 Log (0.504 ULP), rounds once, and takes the 24 inputs whose float64
// result lies next to a float32 rounding midpoint from a table. It shares no
// code or constant with logCR.
func log32(dst, src []float32) {
	var buf [via64Chunk]float64
	src = src[:len(dst)]
	for i := 0; i < len(dst); i += via64Chunk {
		b := buf[:min(via64Chunk, len(dst)-i)]
		s := src[i : i+len(b)]
		for j := range b {
			b[j] = float64(s[j])
		}
		Log(b, b)
		d := dst[i : i+len(b)]
		for j, v := range b {
			f := float32(v)
			// Within 8 float64 ULP of a float32 rounding midpoint (the 29
			// bits float32 drops are 0x10000000 +- 8): the double rounding
			// may be wrong, so take the table's answer.
			if math.Float64bits(v)&(1<<29-1)-(1<<28-8) < 16 {
				f = log32Exception(s[j], f)
			}
			d[j] = f
		}
	}
}

// log32Exceptions are the float32 inputs whose float64 log lies within 2
// float64 ULP of a float32 rounding midpoint, sorted, each with its correctly
// rounded log decided by a 300-bit reference (NDARRAY_EXHAUSTIVE=1
// NDARRAY_DUMP_HARD=1 on the v0.32.0 TestLog32CorrectlyRounded): 24 inputs, 5
// of which rounded to the wrong float32 without this table. They stay the
// hardest known inputs for any float32 log (TestLog32Exceptions).
var log32Exceptions = [...][2]uint32{
	{0x0dc8bba4, 0xc289bac4}, {0x111c87f8, 0xc28085df}, {0x1a8446cb, 0xc24d0a45},
	{0x1f116ab8, 0xc233b53e}, {0x28e3fa26, 0xc1fa75e0}, {0x29e6126b, 0xc1ef4c02},
	{0x29fd22f8, 0xc1ee8859}, {0x2c4c24b7, 0xc1d48710}, {0x2e492984, 0xc1be76ee},
	{0x38dcbe38, 0xc1128ba4}, {0x3bf86ef0, 0xc09c399e}, {0x3c413d3a, 0xc08e158f},
	{0x41178feb, 0x400fe5e7}, {0x464d5b2b, 0x4117bcf1}, {0x4665a9a6, 0x41198725},
	{0x4c5d65a5, 0x418f034b}, {0x4d604ebe, 0x419a352c}, {0x4e85f412, 0x41a6b811},
	{0x5ee8984e, 0x422e4a21}, {0x65d890d3, 0x4254d1f9}, {0x66a8c860, 0x42595e46},
	{0x66abbd63, 0x4259700e}, {0x6f31a8ec, 0x42845a89}, {0x79e7ec37, 0x42a1ffb7},
}

// log32Exception returns the table's correctly rounded log of x if x is in
// log32Exceptions, and f otherwise.
func log32Exception(x, f float32) float32 {
	bits := math.Float32bits(x)
	lo, hi := 0, len(log32Exceptions)
	for lo < hi {
		m := (lo + hi) / 2
		switch e := log32Exceptions[m]; {
		case e[0] == bits:
			return math.Float32frombits(e[1])
		case e[0] < bits:
			lo = m + 1
		default:
			hi = m
		}
	}
	return f
}
