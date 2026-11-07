package floats

import (
	"math"
	"math/big"
	"math/rand"
	"testing"
)

// TestExpCRULP bounds expCR's float64 error against the 300-bit reference
// over the inputs float32 exp needs (|x| <= 104, the float32 overflow and
// underflow edges included). Measured worst: 1.006 ULP. The exhaustive
// float32 test certifies an expCR result only when it is more than 4 float64
// ULP from a rounding midpoint, so this bound must stay below 2 (half the
// margin).
func TestExpCRULP(t *testing.T) {
	r := rand.New(rand.NewSource(11))
	var worst float64
	var at float64
	for i := 0; i < 200000; i++ {
		x := float64(float32(r.Float64()*193 - 104)) // a float32 input, widened
		if e := ulpErr(expCR(x), refExp(x)); e > worst {
			worst, at = e, x
		}
	}
	t.Logf("expCR: worst %.3f float64 ULP at x = %v", worst, at)
	if worst >= 2 {
		t.Errorf("expCR error %.3f ULP at %v: the certifier's 4-ULP margin needs it below 2", worst, at)
	}
}

// TestExp32CRCorrectlyRounded: the prototype, through the same exhaustive
// certifier as Exp32 (NDARRAY_EXHAUSTIVE=1 for all 2^32 inputs).
func TestExp32CRCorrectlyRounded(t *testing.T) {
	runExhaustive(t, exhaustiveCase{"expCR", exp32CR, expCRSlice, refExp, exp32Special, 4}, exhaustiveStride())
}

// TestExpCRTable re-derives the 64 mantissas of 2^(i/64) from the 300-bit
// reference: the table was copied, not trusted.
func TestExpCRTable(t *testing.T) {
	root := new(big.Float).SetPrec(refPrec).SetInt64(2)
	for range 6 { // 2^(1/64)
		root.Sqrt(root)
	}
	e := new(big.Float).SetPrec(refPrec).SetInt64(1)
	for i, m := range crMant {
		want, _ := e.Float64()
		if got := math.Float64frombits(0x3ff0000000000000 | m); got != want {
			t.Errorf("crMant[%d] gives %v, the nearest double to 2^(%d/64) is %v", i, got, i, want)
		}
		e.Mul(e, root)
	}
}
