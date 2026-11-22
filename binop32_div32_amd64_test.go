//go:build amd64

package floats

import (
	"math"
	"testing"
)

// TestDiv32AVX2HalvesBitIdentical checks the AVX2 divide with 128-bit halves
// feeding YMM divides (div32AVX2Halves, floats#93) and the plain AVX2 divide
// against the scalar loop at lengths around the 16- and 8-element steps and at
// large lengths, with dst exactly aliasing a and b, and with a sentinel after dst
// to catch an overrun.
func TestDiv32AVX2HalvesBitIdentical(t *testing.T) {
	if !hasAVX2FMA {
		t.Skip("no AVX2+FMA")
	}
	kernels := map[string]func(dst, a, b []float32){
		"halves": div32AVX2Halves,
		"plain":  div32AVX2,
	}
	lens := []int{0, 1, 7, 8, 9, 15, 16, 17, 23, 24, 25, 31, 32, 33, 255, 1000, 65535, 65536, 65537, 100003}
	for name, kern := range kernels {
		for _, n := range lens {
			a, b := make([]float32, n), make([]float32, n)
			for i := range a {
				a[i], b[i] = 0.25+float32(i%97)*0.37, -3+float32(i%13)*0.71
			}
			if n > 3 {
				b[1], a[2], b[2], a[3] = 0, 0, 0, float32(math.Inf(1))
			}
			want := make([]float32, n)
			div32Ref(want, a, b)
			for _, mode := range []string{"disjoint", "dst=a", "dst=b"} {
				x, y := append([]float32(nil), a...), append([]float32(nil), b...)
				buf := make([]float32, n+1)
				buf[n] = 7
				dst := buf[:n]
				switch mode {
				case "dst=a":
					kern(x, x, y)
					dst = x
				case "dst=b":
					kern(y, x, y)
					dst = y
				default:
					kern(dst, x, y)
				}
				for i := 0; i < n; i++ {
					w, g := want[i], dst[i]
					if math.Float32bits(w) != math.Float32bits(g) && !(w != w && g != g) {
						t.Fatalf("%s %s n=%d i=%d: got %v want %v", name, mode, n, i, g, w)
					}
				}
				if mode == "disjoint" && buf[n] != 7 {
					t.Fatalf("%s n=%d: wrote past dst", name, n)
				}
			}
		}
	}
}

// TestIsZen3 decodes CPUID leaf 1 EAX values: family 19h with the Zen 3 models
// (Milan-X 00A00F11 is model 01h, Vermeer 00A20F10 model 21h, Cezanne 00A50F00
// model 50h) selects the halves kernel; Zen 4 (Raphael 00A60F12 model 61h, Genoa
// 00A10F11 model 11h), Zen 2 (Rome 00830F10, family 17h), Zen 5 (family 1Ah) and
// an Intel signature do not.
func TestIsZen3(t *testing.T) {
	for _, c := range []struct {
		name string
		eax  uint32
		want bool
	}{
		{"Milan-X", 0x00a00f11, true},
		{"Vermeer", 0x00a20f10, true},
		{"Cezanne", 0x00a50f00, true},
		{"Raphael (Zen 4)", 0x00a60f12, false},
		{"Genoa (Zen 4)", 0x00a10f11, false},
		{"Rome (Zen 2)", 0x00830f10, false},
		{"Zen 5", 0x00b40f40, false},
		{"Cascade Lake", 0x00050657, false},
	} {
		if got := isZen3(c.eax); got != c.want {
			t.Errorf("%s (%#x): isZen3 = %v, want %v", c.name, c.eax, got, c.want)
		}
	}
	t.Logf("div32HalvesCPU = %v", div32HalvesCPU)
}
