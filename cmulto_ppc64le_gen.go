//go:build ignore

// Command gen produces cmulto_ppc64le.s with go-asmgen: the VSX (POWER8 baseline, so no
// gate) kernel of CMulTo (cmulto.go), n complex numbers in one call, four per step then
// the last two and one when n has those bits (go-simd/floats, "CMulTo: no s390x/ppc64le
// kernel").
//
// A VSX register holds exactly one complex number. LXVD2X/STXVD2X move the doubleword at
// the lower address to doubleword 0 of the register in either endianness, so with
// doubleword numbering (which XXPERMDI uses) the real part is doubleword 0. For
// c = (ar, ai) of a and w = (br, bi) of b:
//
//	S1 = (br, br)  XXPERMDI DM=0 (w, w)      S2 = (bi, bi)  XXPERMDI DM=3 (w, w)
//	N  = -S2       XVNEGDP                    W2 = (-bi, bi) XXPERMDI DM=0 (N, S2)
//	SW = (ai, ar)  XXPERMDI DM=2 (c, c)       P  = c*S1      XVMULDP
//	P  = SW*W2 + P = (ar*br - ai*bi, ai*br + ar*bi)           XVMADDADP
//
// The released assembler has no XV arithmetic (as for the other ppc64le kernels), so
// those are WORDs of their Power ISA v3.0B encodings (XX3-form, opcode 60: xvmuldp XO
// 112, xvmaddadp 97; XX2-form xvnegdp XO 505; xxpermdi XO 10 with DM at bit 8), on the
// upper registers VS32..VS63 (the Altivec VRs), so no FPR is touched. Reads of a step are
// done before its stores, so dst may be a or b exactly.
//
// Run: go run -modfile=<modfile requiring github.com/go-asmgen/asmgen v0.23.2> cmulto_ppc64le_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/ppc64"
)

func main() {
	f := emit.NewFile("ppc64le")
	f.Add(kernel().Func())
	if err := os.WriteFile("cmulto_ppc64le.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote cmulto_ppc64le.s")
}

// xx3 encodes an XX3-form VSX instruction on VS registers t, a, b (0..63); low carries
// the XO shifted into place and any DM.
func xx3(low, t, a, b int) string {
	w := 0xF0000000 | (t&31)<<21 | (a&31)<<16 | (b&31)<<11 | low |
		((a>>5)&1)<<2 | ((b>>5)&1)<<1 | (t>>5)&1
	return fmt.Sprintf("WORD $0x%08x", w)
}

func xvmuldp(t, a, b int) string   { return xx3(112<<3, t, a, b) }
func xvmaddadp(t, a, b int) string { return xx3(97<<3, t, a, b) }
func xxpermdi(t, a, b, dm int) string {
	return xx3(10<<3|dm<<8, t, a, b)
}

// xvnegdp is the XX2 form: T at 21, B at 11, XO 505 at bit 2, BX bit 1, TX bit 0.
func xvnegdp(t, b int) string {
	w := 0xF0000000 | (t&31)<<21 | (b&31)<<11 | 505<<2 | ((b>>5)&1)<<1 | (t>>5)&1
	return fmt.Sprintf("WORD $0x%08x", w)
}

// load emits the two loads of the complex number at byte offset off (an index register
// holds it), registers VS(v)..: c = v, w = v+1.
func load(b *ppc64.Builder, off, v int) {
	b.Raw("LXVD2X (R4)(%s), VS%d", idx(off), v)
	b.Raw("LXVD2X (R5)(%s), VS%d", idx(off), v+1)
}

// idx names the register holding the byte offset off (R0 reads as zero).
func idx(off int) string {
	if off == 0 {
		return "R0"
	}
	return fmt.Sprintf("R%d", 7+off/16) // R8, R9, R10 for 16, 32, 48
}

// compute emits the product into VS(v+6) and stores it at byte offset off.
func compute(b *ppc64.Builder, off, v int) {
	c, w, s1, s2, sw, p, w2 := v, v+1, v+2, v+3, v+4, v+5, v+6
	b.Raw("%s", xxpermdi(s1, w, w, 0))
	b.Raw("%s", xxpermdi(s2, w, w, 3))
	b.Raw("%s", xvnegdp(w2, s2))
	b.Raw("%s", xxpermdi(w2, w2, s2, 0))
	b.Raw("%s", xxpermdi(sw, c, c, 2))
	b.Raw("%s", xvmuldp(p, c, s1))
	b.Raw("%s", xvmaddadp(p, sw, w2))
	b.Raw("STXVD2X VS%d, (R3)(%s)", p, idx(off))
}

// block emits k numbers (k <= 4): all loads, then the arithmetic.
func block(b *ppc64.Builder, k int) {
	for i := 0; i < k; i++ {
		load(b, 16*i, 32+8*i)
	}
	for i := 0; i < k; i++ {
		compute(b, 16*i, 32+8*i)
	}
}

func kernel() *ppc64.Builder {
	sig := abi.LayoutArgs([]abi.Arg{abi.Scalar("dst", abi.Ptr), abi.Scalar("a", abi.Ptr), abi.Scalar("b", abi.Ptr), abi.Scalar("n", abi.Int64)}, nil)
	b := ppc64.NewFunc("cmulToVSX", sig, 0)
	// R3 dst, R4 a, R5 b, R6 n complex numbers, R7 scratch; R8..R10 = 16, 32, 48.
	b.LoadArg("dst", "R3").LoadArg("a", "R4").LoadArg("b", "R5").LoadArg("n", "R6")
	b.Raw("MOVD $16, R8").Raw("MOVD $32, R9").Raw("MOVD $48, R10")
	b.Raw("CMP R6, $4").Raw("BLT two").Label("loop")
	block(b, 4)
	b.Raw("ADD $64, R3").Raw("ADD $64, R4").Raw("ADD $64, R5").Raw("ADD $-4, R6").
		Raw("CMP R6, $4").Raw("BGE loop").Label("two")
	b.Raw("ANDCC $2, R6, R7").Raw("BEQ one")
	block(b, 2)
	b.Raw("ADD $32, R3").Raw("ADD $32, R4").Raw("ADD $32, R5").Label("one")
	b.Raw("ANDCC $1, R6, R7").Raw("BEQ done")
	block(b, 1)
	b.Label("done").Ret()
	return b
}
