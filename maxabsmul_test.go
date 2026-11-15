package floats

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"
)

// cmaxRef is C's fmax (a NaN argument loses; equal arguments give the second),
// the way lsoda.c's vmnorm loop is written: independent of maxAbsMulRef.
func cmaxRef(a, b float64) float64 {
	switch {
	case a != a:
		return b
	case b != b:
		return a
	case a > b:
		return a
	}
	return b
}

func vmnormRef(v, w []float64) float64 {
	vm := 0.0
	for i, x := range v {
		vm = cmaxRef(vm, math.Abs(x)*w[i])
	}
	return vm
}

func mamSpecial(rng *rand.Rand) float64 {
	sp := []float64{0, math.Copysign(0, -1), 1, -1, 0.5, math.NaN(),
		math.Float64frombits(0xfff8000000000123), math.Inf(1), math.Inf(-1),
		5e-324, -5e-324, 1e308, -1e308, 0x1p-1022}
	if rng.Intn(3) == 0 {
		return sp[rng.Intn(len(sp))]
	}
	return rng.NormFloat64() * math.Pow(10, float64(rng.Intn(9)-4))
}

func checkMaxAbsMul(t *testing.T, label string, v, w []float64) {
	t.Helper()
	want := vmnormRef(v, w)
	if got := MaxAbsMul(v, w); math.Float64bits(got) != math.Float64bits(want) {
		t.Fatalf("%s n=%d: %v (%#x), want %v (%#x)", label, len(v), got, math.Float64bits(got), want, math.Float64bits(want))
	}
	if got := maxAbsMulRef(v, w); math.Float64bits(got) != math.Float64bits(want) {
		t.Fatalf("%s ref n=%d: %v, want %v", label, len(v), got, want)
	}
}

// Bit identity with the fmax loop on specials (NaN, ±Inf, ±0, subnormals,
// negative weights), every length 0..70, four start offsets.
func TestMaxAbsMulBitIdentical(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	for n := 0; n <= 70; n++ {
		for trial := 0; trial < 60; trial++ {
			for off := 0; off < 4; off++ {
				v, w := make([]float64, n+off), make([]float64, n+off+3)
				for i := range v {
					v[i] = mamSpecial(rng)
				}
				for i := range w {
					w[i] = mamSpecial(rng)
				}
				checkMaxAbsMul(t, "random", v[off:], w[off:])
			}
		}
	}
}

// Ordinary positive weights (ewset's): the maximum planted at every position.
func TestMaxAbsMulPositions(t *testing.T) {
	if got := MaxAbsMul(nil, nil); got != 0 || math.Signbit(got) {
		t.Fatalf("MaxAbsMul(nil) = %v", got)
	}
	for n := 1; n <= 67; n++ {
		for pos := 0; pos < n; pos++ {
			v, w := make([]float64, n), make([]float64, n)
			for i := range v {
				v[i], w[i] = float64(i%5)-2, 1.5
			}
			v[pos] = -1e6
			checkMaxAbsMul(t, "planted", v, w)
			v[pos] = math.NaN()
			checkMaxAbsMul(t, "nan", v, w)
			w[pos] = 0
			v[pos] = math.Inf(1)
			checkMaxAbsMul(t, "inf*0", v, w)
		}
		// All zero products, of both signs: the sign of the zero follows the loop.
		for _, wz := range []float64{0, math.Copysign(0, -1)} {
			v, w := make([]float64, n), make([]float64, n)
			for i := range v {
				v[i], w[i] = []float64{0, math.Copysign(0, -1), 3}[i%3], wz
			}
			checkMaxAbsMul(t, "zeros", v, w)
		}
	}
}

func TestMaxAbsMulPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("no panic for len(w) < len(v)")
		}
	}()
	MaxAbsMul(make([]float64, 3), make([]float64, 2))
}

// FuzzMaxAbsMul: bit identity with the fmax loop on arbitrary bit patterns.
func FuzzMaxAbsMul(f *testing.F) {
	f.Add(uint8(0), []byte{1, 2, 3, 4, 5, 6, 7, 8})
	f.Add(uint8(1), make([]byte, 16*45))
	f.Fuzz(func(t *testing.T, offset uint8, data []byte) {
		x := make([]float64, len(data)/8)
		for i := range x {
			x[i] = math.Float64frombits(binary.LittleEndian.Uint64(data[i*8:]))
		}
		off := int(offset % 4)
		if off > len(x) {
			off = len(x)
		}
		x = x[off:]
		n := len(x) / 2
		checkMaxAbsMul(t, "fuzz", x[:n], x[n:2*n])
	})
}
