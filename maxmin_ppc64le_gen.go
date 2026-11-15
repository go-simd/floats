//go:build ignore

// Command gen produces maxmin_ppc64le.s with go-asmgen: the ppc64le VSX (POWER8
// baseline) float64 pairwise Maximum and Minimum kernels (go-ndarray/ndarray#98,
// #118).
//
// Rule (maxmin.go): a NaN b[i] wins, else a NaN a[i] wins, else the larger
// (smaller), and for two zeros the AND (OR) of the bits.
//
// Sources: Power ISA Version 3.0B, Book I, VSX Vector Floating-Point
// Instructions. The ISA PDF was NOT opened (the fetch of it did not succeed; only a
// QEMU decoder listing of xvcmpgtdp's XX3 layout was seen); the encodings below
// are from the ISA as known and were verified by execution under QEMU POWER9
// (maxmin_test.go, bit for bit). Every instruction is in ISA 2.06, so valid on
// POWER8:
//
//   - xvcmpeqdp (XX3-form, XO 99) and xvcmpgtdp (XO 107): each doubleword of the
//     result is all ones when the comparison holds, zero otherwise, a NaN operand
//     included; xvcmpgtdp T,A,B is A > B. Go's assembler lacks them: WORD.
//
//   - xxsel (XX4-form): T = (A AND NOT C) OR (B AND C), C being the mask; Go
//     assembles `XXSEL A, B, C, T`. XXLAND / XXLOR: bitwise.
//
//     t = (b > a) ; t = t ? b : a                  Maximum   (a > b for Minimum)
//     t = (b == b) ? t : b                         a NaN b wins
//     t = (a == b) ? (a AND b) : t                 two zeros   (OR for Minimum)
//
// Every selected doubleword is a copy of a or b, so NaNs keep their bits.
//
// Lane order: LXVD2X/STXVD2X swap the doublewords of a vector on little-endian;
// every vector is loaded and stored that way, so a lane-wise operation pairs the
// same elements. Each kernel does n/8 blocks of eight elements (n a positive
// multiple of 8; the Go caller finishes the last seven); a block is computed before
// it is stored, so dst may alias a or b exactly. R14..R16 hold the byte offsets
// LXVD2X's indexed form needs.
//
// Run: go run -modfile=<modfile requiring github.com/go-asmgen/asmgen v0.23.0> maxmin_ppc64le_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/ppc64"
)

func main() {
	f := emit.NewFile("ppc64le")
	f.Add(kernel("maximumVSX", true))
	f.Add(kernel("minimumVSX", false))
	if err := os.WriteFile("maxmin_ppc64le.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote maxmin_ppc64le.s")
}

// xx3 encodes an XX3-form instruction: VSR[t] = op(VSR[a], VSR[b]).
func xx3(xo uint32, t, a, b int) uint32 {
	return 60<<26 | uint32(t&31)<<21 | uint32(a&31)<<16 | uint32(b&31)<<11 | xo<<3 |
		uint32(a>>5)<<2 | uint32(b>>5)<<1 | uint32(t>>5)
}

func cmp(mnem string, xo uint32, t, a, b int) string {
	return fmt.Sprintf("WORD $0x%08x // %s VS%d, VS%d, VS%d", xx3(xo, t, a, b), mnem, t, a, b)
}

func vidx(k int) string { return [...]string{"R0", "R14", "R15", "R16"}[k] }

// kernel: a in VS32..VS35, b in VS36..VS39, t in VS40..VS43, m in VS44..VS47,
// e in VS48..VS51.
func kernel(name string, isMax bool) *emit.Function {
	sig := ppc64.Layout([]string{"dst", "a", "b", "n"},
		[]ppc64.Type{ppc64.Ptr, ppc64.Ptr, ppc64.Ptr, ppc64.Int64}, nil, nil)
	bl := ppc64.NewFunc(name, sig, 0)
	bl.LoadArg("dst", "R3").LoadArg("a", "R5").LoadArg("b", "R6").LoadArg("n", "R4").
		Raw("MOVD $16, R14").Raw("MOVD $32, R15").Raw("MOVD $48, R16").
		Label("loop")
	for k := 0; k < 4; k++ {
		bl.Raw("LXVD2X (R5)(%s), VS%d", vidx(k), 32+k)
	}
	for k := 0; k < 4; k++ {
		bl.Raw("LXVD2X (R6)(%s), VS%d", vidx(k), 36+k)
	}
	for k := 0; k < 4; k++ {
		a, b, t, m, e := 32+k, 36+k, 40+k, 44+k, 48+k
		zop := "XXLAND"
		if isMax {
			bl.Raw("%s", cmp("xvcmpgtdp", 107, t, b, a)) // t = b > a
		} else {
			zop = "XXLOR"
			bl.Raw("%s", cmp("xvcmpgtdp", 107, t, a, b)) // t = a > b
		}
		bl.Raw("XXSEL VS%d, VS%d, VS%d, VS%d", a, b, t, t). // t = t ? b : a
									Raw("%s", cmp("xvcmpeqdp", 99, m, b, b)).        // m = b is not a NaN
									Raw("XXSEL VS%d, VS%d, VS%d, VS%d", b, t, m, t). // t = m ? t : b
									Raw("%s", cmp("xvcmpeqdp", 99, e, a, b)).        // e = a == b
									Raw("%s VS%d, VS%d, VS%d", zop, a, b, m).        // m = a op b
									Raw("XXSEL VS%d, VS%d, VS%d, VS%d", t, m, e, t)  // t = e ? m : t
	}
	for k := 0; k < 4; k++ {
		bl.Raw("STXVD2X VS%d, (R3)(%s)", 40+k, vidx(k))
	}
	bl.Raw("ADD $64, R3").Raw("ADD $64, R5").Raw("ADD $64, R6").Raw("ADD $-8, R4").
		Raw("CMP R4, $0").Raw("BGT loop").Ret()
	return bl.Func()
}
