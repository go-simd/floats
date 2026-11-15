//go:build ignore

// Command gen produces maxmin_s390x.s with go-asmgen: the s390x vector-facility
// (z13 baseline) float64 pairwise Maximum and Minimum kernels
// (go-ndarray/ndarray#98, #118).
//
// Rule (maxmin.go): a NaN b[i] wins, else a NaN a[i] wins, else the larger
// (smaller), and for two zeros the AND (OR) of the bits.
//
// Sources: z/Architecture Principles of Operation (SA22-7832). The PDF exceeds
// the size the fetch tool reads (10 MB) and was NOT opened; the semantics below
// were established by execution under QEMU (maxmin_test.go, bit for bit):
//
//   - VFCHDB / VFCEDB (VECTOR FP COMPARE HIGH / EQUAL, long format): each element
//     of the result is all ones when the comparison holds and zero otherwise,
//     including when an operand is a NaN. Compare high is "second operand greater
//     than third"; Go's `VFCHDB x, y, m` is m = (x > y) (read off the bytes cmd/asm
//     emits, and confirmed by the tests).
//
//   - VSEL: Go `VSEL x, y, m, d` is d = m ? x : y (bitwise).
//
//   - VN, VO: bitwise AND, OR. All in the z13 vector facility, and the z14 VFMAX
//     (which has a NaN and zero rule of its own) is not used, so no gate is needed.
//
//     t = (b > a) ; t = t ? b : a                  Maximum   (a > b for Minimum)
//     t = (b == b) ? t : b                         a NaN b wins
//     t = (a == b) ? (a AND b) : t                 two zeros   (OR for Minimum)
//
// Every selected element is a copy of a or b, so NaNs keep their bits. Each kernel
// does n/8 blocks of eight elements (n a positive multiple of 8; the Go caller
// finishes the last seven); VLM loads and VSTM stores four vectors, a block is
// computed before it is stored, so dst may alias a or b exactly.
//
// Run: go run -modfile=<modfile requiring github.com/go-asmgen/asmgen v0.23.0> maxmin_s390x_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/s390x"
)

func main() {
	f := emit.NewFile("s390x")
	f.Add(kernel("maximumVX", true))
	f.Add(kernel("minimumVX", false))
	if err := os.WriteFile("maxmin_s390x.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote maxmin_s390x.s")
}

// kernel: a in V0..V3, b in V4..V7, t in V8..V11, m in V12..V15, e in V16..V19.
func kernel(name string, isMax bool) *emit.Function {
	sig := s390x.Layout([]string{"dst", "a", "b", "n"},
		[]s390x.Type{s390x.Ptr, s390x.Ptr, s390x.Ptr, s390x.Int64}, nil, nil)
	b := s390x.NewFunc(name, sig, 0)
	b.LoadArg("dst", "R1").LoadArg("a", "R2").LoadArg("b", "R3").LoadArg("n", "R4").
		Raw("loop:").Raw("VLM (R2), V0, V3").Raw("VLM (R3), V4, V7")
	for k := 0; k < 4; k++ {
		a, bb, t, m, e := k, 4+k, 8+k, 12+k, 16+k
		zop := "VN"
		if isMax {
			b.Raw("VFCHDB V%d, V%d, V%d", bb, a, t) // t = b > a
		} else {
			zop = "VO"
			b.Raw("VFCHDB V%d, V%d, V%d", a, bb, t) // t = a > b
		}
		b.Raw("VSEL V%d, V%d, V%d, V%d", bb, a, t, t). // t = t ? b : a
								Raw("VFCEDB V%d, V%d, V%d", bb, bb, m).      // m = b is not a NaN
								Raw("VSEL V%d, V%d, V%d, V%d", t, bb, m, t). // t = m ? t : b
								Raw("VFCEDB V%d, V%d, V%d", a, bb, e).       // e = a == b
								Raw("%s V%d, V%d, V%d", zop, a, bb, m).      // m = a op b
								Raw("VSEL V%d, V%d, V%d, V%d", m, t, e, t)   // t = e ? m : t
	}
	b.Raw("VSTM V8, V11, (R1)").
		Raw("ADD $64, R1").Raw("ADD $64, R2").Raw("ADD $64, R3").Raw("SUB $8, R4").
		Raw("CMP R4, $0").Raw("BGT loop").Ret()
	return b.Func()
}
