package floats

import (
	"math"
	"math/big"
	"math/rand"
	"testing"
	"unsafe"
)

// mulRowsShapes are the (n, rows, lda, ldb) of the tests: every column count from 1 to
// 70 (the 16, 4, 2 and 1 column blocks and their tails), the row counts of a window sum
// and a few more, packed strides, padded strides and zero strides.
func mulRowsShapes() (shapes [][4]int) {
	for n := 1; n <= 70; n++ {
		for _, rows := range []int{1, 2, 3, 5, 10} {
			shapes = append(shapes, [4]int{n, rows, n, n}, [4]int{n, rows, n + 3, 2*n + 1}, [4]int{n, rows, 0, n})
		}
	}
	for _, n := range []int{127, 128, 129, 257, 1000} {
		shapes = append(shapes, [4]int{n, 7, n, n + 64}, [4]int{n, 64, 128, 128})
	}
	return shapes
}

// mulRowsInputs fills a matrix of rows rows read down the columns, with signed zeros,
// infinities and ordinary values of several magnitudes.
func mulRowsInputs(rng *rand.Rand, n int) []float64 {
	c := make([]float64, n)
	for i := range c {
		switch rng.Intn(24) {
		case 0:
			c[i] = 0
		case 1:
			c[i] = math.Copysign(0, -1)
		case 2:
			c[i] = math.Inf(1 - 2*rng.Intn(2))
		default:
			c[i] = rng.NormFloat64() * math.Pow(10, float64(rng.Intn(9)-4))
		}
	}
	return c
}

// mulRowsWant is the definition from the contract, one math.FMA per term in increasing k.
func mulRowsWant(dst, a []float64, lda int, b []float64, ldb int, rows int) []float64 {
	want := append([]float64(nil), dst...)
	for k := 0; k < rows; k++ {
		for j := range want {
			want[j] = math.FMA(a[k*lda+j], b[k*ldb+j], want[j])
		}
	}
	return want
}

// mulRowsDiff reports the first index where a and b differ, a NaN equal to any NaN (the
// payload of a NaN is left to the hardware).
func mulRowsDiff(a, b []float64) int {
	for i := range a {
		if math.IsNaN(a[i]) && math.IsNaN(b[i]) {
			continue
		}
		if math.Float64bits(a[i]) != math.Float64bits(b[i]) {
			return i
		}
	}
	return -1
}

// TestMulAddRowsExact: the kernel, the public function and the portable loop equal the
// definition bit for bit at every shape.
func TestMulAddRowsExact(t *testing.T) {
	rng := rand.New(rand.NewSource(201))
	for _, s := range mulRowsShapes() {
		n, rows, lda, ldb := s[0], s[1], s[2], s[3]
		a := mulRowsInputs(rng, (rows-1)*lda+n)
		b := mulRowsInputs(rng, (rows-1)*ldb+n)
		d0 := mulRowsInputs(rng, n)
		want := mulRowsWant(d0, a, lda, b, ldb, rows)
		got := append([]float64(nil), d0...)
		MulAddRows(got, a, lda, b, ldb, rows)
		if i := mulRowsDiff(got, want); i >= 0 {
			t.Fatalf("MulAddRows n=%d rows=%d lda=%d ldb=%d element %d: %x, want %x", n, rows, lda, ldb, i, math.Float64bits(got[i]), math.Float64bits(want[i]))
		}
		ref := append([]float64(nil), d0...)
		mulAddRowsRef(ref, a, lda, b, ldb, rows)
		if i := mulRowsDiff(ref, want); i >= 0 {
			t.Fatalf("mulAddRowsRef n=%d rows=%d element %d differs from the definition", n, rows, i)
		}
	}
}

// TestMulAddRowsAgainstExact: against a 300-bit evaluation of dst + Σ a*b, the result is
// within rows+1 units of the last place of the sum of the magnitudes (the bound of rows
// fused additions), for the window sum of a QMF bank (64 columns, 10 rows) and others.
func TestMulAddRowsAgainstExact(t *testing.T) {
	rng := rand.New(rand.NewSource(202))
	for _, s := range [][4]int{{64, 10, 64, 64}, {64, 5, 128, 256}, {33, 8, 40, 40}, {1000, 3, 1000, 1000}} {
		n, rows, lda, ldb := s[0], s[1], s[2], s[3]
		a := make([]float64, (rows-1)*lda+n)
		b := make([]float64, (rows-1)*ldb+n)
		for i := range a {
			a[i] = rng.NormFloat64()
		}
		for i := range b {
			b[i] = rng.NormFloat64()
		}
		d := make([]float64, n)
		for i := range d {
			d[i] = rng.NormFloat64()
		}
		d0 := append([]float64(nil), d...)
		MulAddRows(d, a, lda, b, ldb, rows)
		var worst float64
		for j := 0; j < n; j++ {
			exact := new(big.Float).SetPrec(300).SetFloat64(d0[j])
			mag := math.Abs(d0[j])
			for k := 0; k < rows; k++ {
				p := new(big.Float).SetPrec(300).SetFloat64(a[k*lda+j])
				p.Mul(p, new(big.Float).SetPrec(300).SetFloat64(b[k*ldb+j]))
				exact.Add(exact, p)
				mag += math.Abs(a[k*lda+j] * b[k*ldb+j])
			}
			diff := new(big.Float).SetPrec(300).Sub(new(big.Float).SetPrec(300).SetFloat64(d[j]), exact)
			e, _ := diff.Float64()
			if r := math.Abs(e) / (mag * 0x1p-53); r > worst {
				worst = r
			}
		}
		t.Logf("n=%d rows=%d: worst error %.3f units of 2^-53 of the sum of magnitudes (bound %d)", n, rows, worst, rows+1)
		if worst > float64(rows+1) {
			t.Errorf("n=%d rows=%d: error %.3f units, bound %d", n, rows, worst, rows+1)
		}
	}
}

func TestMulAddRowsPanicsAndBounds(t *testing.T) {
	mustPanic := func(name, want string, f func()) {
		t.Helper()
		defer func() {
			r, _ := recover().(string)
			if r != want {
				t.Errorf("%s: panic %q, want %q", name, r, want)
			}
		}()
		f()
	}
	d, a, b := make([]float64, 8), make([]float64, 40), make([]float64, 40)
	mustPanic("rows < 0", "floats: MulAddRows: rows < 0", func() { MulAddRows(d, a, 8, b, 8, -1) })
	mustPanic("lda < 0", "floats: MulAddRows: lda or ldb < 0", func() { MulAddRows(d, a, -8, b, 8, 2) })
	mustPanic("ldb < 0", "floats: MulAddRows: lda or ldb < 0", func() { MulAddRows(d, a, 8, b, -8, 2) })
	mustPanic("short a", "floats: MulAddRows: a or b shorter than (rows-1)*stride + len(dst)", func() { MulAddRows(d, a[:15], 8, b, 8, 2) })
	mustPanic("short b", "floats: MulAddRows: a or b shorter than (rows-1)*stride + len(dst)", func() { MulAddRows(d, a, 8, b[:15], 8, 2) })
	mustPanic("overflow", "floats: MulAddRows: rows*stride overflows", func() { MulAddRows(d, a, math.MaxInt/2, b, 8, 4) })
	MulAddRows(nil, a, 8, b, 8, 3)   // no columns: nothing
	MulAddRows(d, nil, 8, nil, 8, 0) // no rows: nothing, nothing read
	buf := make([]float64, 64)
	mustPanic("dst in a", "floats: MulAddRows: dst overlaps a or b", func() { MulAddRows(buf[:8], buf[:16], 8, b, 8, 2) })
	mustPanic("dst in b", "floats: MulAddRows: dst overlaps a or b", func() { MulAddRows(buf[8:16], a, 8, buf[:16], 8, 2) })
	mustPanic("shifted", "floats: MulAddRows: dst overlaps a or b", func() { MulAddRows(buf[4:12], buf[8:24], 8, b, 8, 2) })
	mustPanic("across element types", "floats: MulAddRows: dst overlaps a or b", func() {
		MulAddRows(unsafe.Slice((*float64)(unsafe.Pointer(&buf[8])), 8), buf[:16], 8, b, 8, 2)
	})
	// the part of a that is not read may overlap dst (lda = 16: row 1 starts at 16, dst is 0..7)
	buf2 := make([]float64, 64)
	MulAddRows(buf2[8:16], buf2[:8], 16, b, 8, 1)
	// a and b may be one slice
	rng := rand.New(rand.NewSource(203))
	one := mulRowsInputs(rng, 24)
	d1 := mulRowsInputs(rng, 8)
	want := mulRowsWant(d1, one, 8, one, 8, 3)
	MulAddRows(d1, one, 8, one, 8, 3)
	if i := mulRowsDiff(d1, want); i >= 0 {
		t.Fatalf("a == b: element %d differs", i)
	}
	// nothing written outside dst, a and b unmodified, with real extra elements around
	for _, s := range mulRowsShapes() {
		n, rows, lda, ldb := s[0], s[1], s[2], s[3]
		a := mulRowsInputs(rng, (rows-1)*lda+n+3)
		b := mulRowsInputs(rng, (rows-1)*ldb+n+3)
		a0, b0 := append([]float64(nil), a...), append([]float64(nil), b...)
		pad := make([]float64, n+16)
		for i := range pad {
			pad[i] = 12345
		}
		MulAddRows(pad[8:8+n], a, lda, b, ldb, rows)
		for i := range pad {
			if (i < 8 || i >= 8+n) && pad[i] != 12345 {
				t.Fatalf("n=%d rows=%d: wrote outside dst at %d", n, rows, i)
			}
		}
		if mulRowsDiff(a, a0) >= 0 || mulRowsDiff(b, b0) >= 0 {
			t.Fatalf("n=%d rows=%d: an input changed", n, rows)
		}
	}
}

// FuzzMulAddRows: for any shape and any bytes, the public function equals the
// definition bit for bit.
func FuzzMulAddRows(f *testing.F) {
	f.Add(uint8(0), uint8(0), uint8(0), uint8(0), []byte{1, 2, 3})
	f.Add(uint8(63), uint8(9), uint8(0), uint8(64), []byte("window sum"))
	f.Add(uint8(16), uint8(3), uint8(5), uint8(5), []byte{255, 0, 128})
	f.Add(uint8(3), uint8(1), uint8(0), uint8(0), []byte{})
	f.Fuzz(func(t *testing.T, n8, rows8, pa, pb uint8, data []byte) {
		n, rows := int(n8)%80, int(rows8)%12
		lda, ldb := n+int(pa)%70, n+int(pb)%70
		if pa%7 == 0 {
			lda = 0
		}
		fill := func(m int, salt int) []float64 {
			v := make([]float64, m)
			for i := range v {
				if len(data) > 0 {
					v[i] = math.Float64frombits(uint64(data[(i*3+salt)%len(data)])<<56 | uint64(data[(i+salt)%len(data)])<<48 | uint64(i))
				}
			}
			return v
		}
		a, b, d := fill(max(rows-1, 0)*lda+n, 1), fill(max(rows-1, 0)*ldb+n, 2), fill(n, 3)
		want := mulRowsWant(d, a, lda, b, ldb, rows)
		MulAddRows(d, a, lda, b, ldb, rows)
		if i := mulRowsDiff(d, want); i >= 0 {
			t.Fatalf("n=%d rows=%d element %d differs from the definition", n, rows, i)
		}
	})
}

// BenchmarkMulAddRows is the window sum of a 64-band QMF synthesis: 64 columns, 5 rows
// of stride 128 against 5 rows of stride 256 (twice per output block).
func BenchmarkMulAddRows(b *testing.B) {
	rng := rand.New(rand.NewSource(204))
	win := mulRowsInputs(rng, 5*128)
	v := mulRowsInputs(rng, 5*256)
	d := make([]float64, 64)
	b.Run("kernel/64x5", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			MulAddRows(d, win, 128, v, 256, 5)
		}
	})
	b.Run("portable/64x5", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			mulAddRowsRef(d, win, 128, v, 256, 5)
		}
	})
	b.Run("rowloop/64x5", func(b *testing.B) { // a pass per row, as a Go loop does it
		for i := 0; i < b.N; i++ {
			for k := 0; k < 5; k++ {
				w, x := win[k*128:], v[k*256:]
				for q := range d {
					d[q] = math.FMA(w[q], x[q], d[q])
				}
			}
		}
	})
}
