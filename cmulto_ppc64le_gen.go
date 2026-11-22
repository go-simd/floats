//go:build ignore

// Command gen produces cmulto_ppc64le.s with go-asmgen: the VSX (POWER8 baseline, so no
// gate) kernel of CMulTo (cmulto.go), n complex numbers in one call.
//
// Pairs of numbers are split into real and imaginary vectors, multiplied and
// re-interleaved (fewer permutes than one number per register, which lost to Go's loop on
// POWER9). A VSX register holds exactly one complex number. LXVD2X/STXVD2X move the
// doubleword at the lower address to doubleword 0 of the register in either endianness,
// so with doubleword numbering (which XXPERMDI uses) the real part is doubleword 0. For
// c0, c1 (two numbers of a) and w0, w1 (of b):
//
//	AR = (c0.re, c1.re)  XXPERMDI DM=0 (c0, c1)     AI = (c0.im, c1.im)  XXPERMDI DM=3 (c0, c1)
//	BR, BI the same of w0, w1
//	RE = AR*BR - AI*BI   XVMULDP, XVMSUBADP         IM = AR*BI + AI*BR   XVMULDP, XVMADDADP
//	out0 = (RE.0, IM.0)  XXPERMDI DM=0 (RE, IM)     out1 = (RE.1, IM.1)  XXPERMDI DM=3 (RE, IM)
//
// A last odd number uses a one-number form (splat, swap, negate). The released
// assembler has no XV arithmetic (as for the other ppc64le kernels), so those are WORDs
// of their Power ISA v3.0B encodings (XX3-form, opcode 60: xvmuldp XO 112, xvmaddadp 97,
// xvmsubadp 113; XX2-form xvnegdp XO 505; xxpermdi XO 10 with DM at bit 8), on the upper
// registers VS32..VS63 (the Altivec VRs), so no FPR is touched.
//
// Schedule (measured on POWER9, which the first form lost on): the n&3 numbers that do
// not fill a block of four are done FIRST, so that their dependent chain overlaps the main
// loop instead of ending the call; the main loop is software-pipelined, the loads of the
// next block of four (two register sets of 16) issued before the arithmetic of the
// current one, so that no vector operation waits in the issue queue for a load. Reads of
// a block are done before its stores, so dst may be a or b exactly.
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
func xvmsubadp(t, a, b int) string { return xx3(113<<3, t, a, b) }
func xxpermdi(t, a, b, dm int) string {
	return xx3(10<<3|dm<<8, t, a, b)
}

// xvnegdp is the XX2 form: T at 21, B at 11, XO 505 at bit 2, BX bit 1, TX bit 0.
func xvnegdp(t, b int) string {
	w := 0xF0000000 | (t&31)<<21 | (b&31)<<11 | 505<<2 | ((b>>5)&1)<<1 | (t>>5)&1
	return fmt.Sprintf("WORD $0x%08x", w)
}

// ptrs names the three base registers a step reads and writes through.
type ptrs struct{ dst, a, b string }

// idx names the register holding the byte offset off (0..112; R0 reads as zero).
func idx(off int) string {
	return [...]string{"R0", "R8", "R9", "R10", "R11", "R12", "R15", "R16"}[off/16]
}

// pairLoad loads two numbers of a and of b at byte offsets off and off+16 into
// VS(v)..VS(v+3): c0 c1 w0 w1.
func pairLoad(b *ppc64.Builder, p ptrs, off, v int) {
	b.Raw("LXVD2X (%s)(%s), VS%d", p.a, idx(off), v)
	b.Raw("LXVD2X (%s)(%s), VS%d", p.a, idx(off+16), v+1)
	b.Raw("LXVD2X (%s)(%s), VS%d", p.b, idx(off), v+2)
	b.Raw("LXVD2X (%s)(%s), VS%d", p.b, idx(off+16), v+3)
}

// pairCompute multiplies the pair loaded into VS(v).. and stores it at byte offset off
// of dst. VS(v+4)..VS(v+7) are AR AI BR BI.
func pairCompute(b *ppc64.Builder, p ptrs, off, v int) {
	c0, c1, w0, w1, ar, ai, br, bi := v, v+1, v+2, v+3, v+4, v+5, v+6, v+7
	b.Raw("%s", xxpermdi(ar, c0, c1, 0))
	b.Raw("%s", xxpermdi(ai, c0, c1, 3))
	b.Raw("%s", xxpermdi(br, w0, w1, 0))
	b.Raw("%s", xxpermdi(bi, w0, w1, 3))
	b.Raw("%s", xvmuldp(c0, ai, bi))
	b.Raw("%s", xvmsubadp(c0, ar, br))
	b.Raw("%s", xvmuldp(c1, ar, bi))
	b.Raw("%s", xvmaddadp(c1, ai, br))
	b.Raw("%s", xxpermdi(w0, c0, c1, 0))
	b.Raw("%s", xxpermdi(w1, c0, c1, 3))
	b.Raw("STXVD2X VS%d, (%s)(%s)", w0, p.dst, idx(off))
	b.Raw("STXVD2X VS%d, (%s)(%s)", w1, p.dst, idx(off+16))
}

// single is one number at byte offset off in the one-number form, VS(v)..VS(v+6). With
// c = (ar, ai) of a and w = (br, bi) of b:
//
//	S1 = (br, br)  XXPERMDI DM=0 (w, w)      S2 = (bi, bi)  XXPERMDI DM=3 (w, w)
//	N  = -S2       XVNEGDP                    W2 = (-bi, bi) XXPERMDI DM=0 (N, S2)
//	SW = (ai, ar)  XXPERMDI DM=2 (c, c)       P  = c*S1      XVMULDP
//	P  = SW*W2 + P = (ar*br - ai*bi, ai*br + ar*bi)           XVMADDADP
func single(b *ppc64.Builder, p ptrs, off, v int) {
	c, w, s1, s2, sw, pr, w2 := v, v+1, v+2, v+3, v+4, v+5, v+6
	b.Raw("LXVD2X (%s)(%s), VS%d", p.a, idx(off), c)
	b.Raw("LXVD2X (%s)(%s), VS%d", p.b, idx(off), w)
	b.Raw("%s", xxpermdi(s1, w, w, 0))
	b.Raw("%s", xxpermdi(s2, w, w, 3))
	b.Raw("%s", xvnegdp(w2, s2))
	b.Raw("%s", xxpermdi(w2, w2, s2, 0))
	b.Raw("%s", xxpermdi(sw, c, c, 2))
	b.Raw("%s", xvmuldp(pr, c, s1))
	b.Raw("%s", xvmaddadp(pr, sw, w2))
	b.Raw("STXVD2X VS%d, (%s)(%s)", pr, p.dst, idx(off))
}

// quadLoad and quadCompute are two pairs, a block of four numbers, in the register set
// starting at v (16 registers).
func quadLoad(b *ppc64.Builder, p ptrs, off, v int) {
	pairLoad(b, p, off, v)
	pairLoad(b, p, off+32, v+8)
}

func quadCompute(b *ppc64.Builder, p ptrs, off, v int) {
	pairCompute(b, p, off, v)
	pairCompute(b, p, off+32, v+8)
}

func kernel() *ppc64.Builder {
	sig := abi.LayoutArgs([]abi.Arg{abi.Scalar("dst", abi.Ptr), abi.Scalar("a", abi.Ptr), abi.Scalar("b", abi.Ptr), abi.Scalar("n", abi.Int64)}, nil)
	b := ppc64.NewFunc("cmulToVSX", sig, 0)
	// R3 dst, R4 a, R5 b, R6 n complex numbers, R7 n&3; R8..R12, R15, R16 = 16..112.
	b.LoadArg("dst", "R3").LoadArg("a", "R4").LoadArg("b", "R5").LoadArg("n", "R6")
	b.Raw("MOVD $16, R8").Raw("MOVD $32, R9").Raw("MOVD $48, R10").Raw("MOVD $64, R11").
		Raw("MOVD $80, R12").Raw("MOVD $96, R15").Raw("MOVD $112, R16")

	// The n&3 last numbers first, through R18..R20 (the tail of dst, a, b).
	t := ptrs{"R18", "R19", "R20"}
	b.Raw("ANDCC $3, R6, R7").Raw("BEQ main")
	b.Raw("SUB R7, R6, R17").Raw("SLD $4, R17").
		Raw("ADD R17, R3, R18").Raw("ADD R17, R4, R19").Raw("ADD R17, R5, R20")
	b.Raw("ANDCC $2, R7, R17").Raw("BEQ one")
	pairLoad(b, t, 0, 32)
	pairCompute(b, t, 0, 32)
	b.Raw("ADD $32, R18").Raw("ADD $32, R19").Raw("ADD $32, R20").Label("one")
	b.Raw("ANDCC $1, R7, R17").Raw("BEQ main")
	single(b, t, 0, 32)
	b.Label("main")

	// Blocks of four, two register sets A and B, the loads of block k+1 before the
	// arithmetic of block k.
	m := ptrs{"R3", "R4", "R5"}
	const setA, setB = 32, 48
	adv := func() {
		b.Raw("ADD $64, R3").Raw("ADD $64, R4").Raw("ADD $64, R5")
	}
	// R21 = blocks of four. The loop runs on CTR (bdnz resolves early, with no direction
	// prediction: a compare-and-branch exit mispredicted once per call from 96 numbers on,
	// measured with perf), (R21-1)/2 times, two blocks per pass, each pass loading the two
	// blocks after the one it computes; what is left (one block, or two) follows it.
	b.Raw("SUB R7, R6, R6").Raw("SRD $2, R6, R21")
	b.Raw("CMP R21, $1").Raw("BLT done")
	quadLoad(b, m, 0, setA)
	b.Raw("ADD $-1, R21, R22").Raw("SRD $1, R22, R22").Raw("CMP R22, $0").Raw("BEQ rest")
	b.Raw("MOVD R22, CTR")
	b.Label("loop")
	quadLoad(b, m, 64, setB)
	quadCompute(b, m, 0, setA)
	adv()
	quadLoad(b, m, 64, setA)
	quadCompute(b, m, 0, setB)
	adv()
	b.Raw("BDNZ loop")
	b.Label("rest")
	b.Raw("ANDCC $1, R21, R23").Raw("BNE lastA")
	quadLoad(b, m, 64, setB)
	quadCompute(b, m, 0, setA)
	adv()
	quadCompute(b, m, 0, setB)
	b.Raw("BR done")
	b.Label("lastA")
	quadCompute(b, m, 0, setA)
	b.Label("done").Ret()
	return b
}
