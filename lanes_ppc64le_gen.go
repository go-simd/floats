//go:build ignore

// Command gen produces lanes_ppc64le.s with go-asmgen: the ppc64le VSX (POWER8
// baseline) float64 elementwise kernels Add, Sub, Mul, Div, Sqrt of go-ndarray
// (internal/kernels/asmgen/ppc64le, moved here: floats#30) and the
// eight-accumulator pairwise-sum block of SumPairwise.
//
// Go's ppc64 assembler has the VSX loads, stores and doubleword permutes
// (LXVD2X, STXVD2X, XXPERMDI) but no vector double arithmetic; the arithmetic is
// go-asmgen's VSX encoders (XVADDDP, XVSUBDP, XVMULDP, XVDIVDP, XVSQRTDP), which
// emit WORDs. One IEEE operation per element, no fused instruction: bit-identical
// to the scalar loops.
//
// Lane order: LXVD2X/STXVD2X swap the doublewords of a vector on little-endian.
// Every vector of one kernel is loaded and stored that way, so a lane-wise
// operation pairs the same elements in every operand and the store puts each
// result back at its own index.
//
// pwBlockVSX(a, n): 8 <= n <= 128. The eight accumulators r[0..7] of NumPy's
// DOUBLE_pairwise_sum are four two-lane registers VS0..VS3 = {r0,r1} {r2,r3}
// {r4,r5} {r6,r7} (in either lane order), loaded from a[0:8] and then added to
// each following block of eight, so every lane receives exactly the additions of
// the scalar r[j]. The fold ((r0+r1)+(r2+r3)) + ((r4+r5)+(r6+r7)) adds each
// register to its own doubleword swap (XXPERMDI $2, the sum of two doubles is
// commutative) and then adds the scalar views F0..F3; the leftover n % 8
// elements are added to the result one at a time, as NumPy does.
//
// Registers: R3.. hold the arguments; R14..R16 the byte offsets LXVD2X's indexed
// form needs (R0 reads as 0).
//
// Run: go run -modfile=<modfile requiring github.com/go-asmgen/asmgen v0.23.0> lanes_ppc64le_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/ppc64"
)

func main() {
	f := emit.NewFile("ppc64le")
	f.Add(sqrtKernel())
	for _, k := range []struct {
		name, vec, sca string
	}{{"addVSX", "add", "FADD"}, {"subVSX", "sub", "FSUB"}, {"mulVSX", "mul", "FMUL"}, {"divVSX", "div", "FDIV"}} {
		f.Add(binKernel(k.name, k.vec, k.sca))
	}
	f.Add(pwBlockKernel())
	if err := os.WriteFile("lanes_ppc64le.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote lanes_ppc64le.s")
}

// offsets loads the byte offsets 16, 32, 48 into R14, R15, R16.
func offsets(b *ppc64.Builder) {
	b.Raw("MOVD $16, R14").Raw("MOVD $32, R15").Raw("MOVD $48, R16")
}

// vidx is the index register for vector k (0..3) of a 64-byte block.
func vidx(k int) string { return [...]string{"R0", "R14", "R15", "R16"}[k] }

// sqrtKernel builds sqrtVSX(dst, src *float64, n int).
func sqrtKernel() *emit.Function {
	sig := ppc64.Layout([]string{"dst", "src", "n"}, []ppc64.Type{ppc64.Ptr, ppc64.Ptr, ppc64.Int64}, nil, nil)
	b := ppc64.NewFunc("sqrtVSX", sig, 0)
	b.LoadArg("dst", "R3").LoadArg("src", "R5").LoadArg("n", "R4")
	offsets(b)
	b.Label("qloop").Raw("CMP R4, $8").Raw("BLT qtail")
	for k := 0; k < 4; k++ {
		b.Raw("LXVD2X (R5)(%s), VS%d", vidx(k), 32+k)
	}
	for k := 0; k < 4; k++ {
		b.XVSQRTDP(36+k, 32+k)
	}
	for k := 0; k < 4; k++ {
		b.Raw("STXVD2X VS%d, (R3)(%s)", 36+k, vidx(k))
	}
	b.Raw("ADD $64, R3").Raw("ADD $64, R5").Raw("ADD $-8, R4").Raw("BR qloop")
	b.Label("qtail").Raw("CMP R4, $0").Raw("BEQ qdone").
		Raw("FMOVD 0(R5), F1").Raw("FSQRT F1, F1").Raw("FMOVD F1, 0(R3)").
		Raw("ADD $8, R3").Raw("ADD $8, R5").Raw("ADD $-1, R4").Raw("BR qtail")
	b.Label("qdone").Ret()
	return b.Func()
}

// binKernel builds name(dst, a, b *float64, n int): dst[i] = a[i] OP b[i], 8
// doubles per iteration with the VSX op and a scalar tail with sca. Each block
// is loaded and computed before it is stored, so dst may alias an input exactly.
func binKernel(name, op, sca string) *emit.Function {
	sig := ppc64.Layout([]string{"dst", "a", "b", "n"},
		[]ppc64.Type{ppc64.Ptr, ppc64.Ptr, ppc64.Ptr, ppc64.Int64}, nil, nil)
	b := ppc64.NewFunc(name, sig, 0)
	b.LoadArg("dst", "R3").LoadArg("a", "R5").LoadArg("b", "R6").LoadArg("n", "R4")
	offsets(b)
	l := name[:3]
	b.Label(l+"loop").Raw("CMP R4, $8").Raw("BLT %stail", l)
	for k := 0; k < 4; k++ {
		b.Raw("LXVD2X (R5)(%s), VS%d", vidx(k), 32+k)
		b.Raw("LXVD2X (R6)(%s), VS%d", vidx(k), 36+k)
	}
	for k := 0; k < 4; k++ {
		switch op {
		case "add":
			b.XVADDDP(40+k, 32+k, 36+k)
		case "sub":
			b.XVSUBDP(40+k, 32+k, 36+k)
		case "mul":
			b.XVMULDP(40+k, 32+k, 36+k)
		case "div":
			b.XVDIVDP(40+k, 32+k, 36+k)
		}
	}
	for k := 0; k < 4; k++ {
		b.Raw("STXVD2X VS%d, (R3)(%s)", 40+k, vidx(k))
	}
	b.Raw("ADD $64, R3").Raw("ADD $64, R5").Raw("ADD $64, R6").Raw("ADD $-8, R4").Raw("BR %sloop", l)
	b.Label(l+"tail").Raw("CMP R4, $0").Raw("BEQ %sdone", l).
		Raw("FMOVD 0(R5), F1").Raw("FMOVD 0(R6), F2").Raw("%s F2, F1", sca).Raw("FMOVD F1, 0(R3)").
		Raw("ADD $8, R3").Raw("ADD $8, R5").Raw("ADD $8, R6").Raw("ADD $-1, R4").Raw("BR %stail", l)
	b.Label(l + "done").Ret()
	return b.Func()
}

// pwBlockKernel builds pwBlockVSX(a *float64, n int) float64 for 8 <= n <= 128.
func pwBlockKernel() *emit.Function {
	sig := ppc64.Layout([]string{"a", "n"}, []ppc64.Type{ppc64.Ptr, ppc64.Int64},
		[]string{"ret"}, []ppc64.Type{ppc64.Float64})
	b := ppc64.NewFunc("pwBlockVSX", sig, 0)
	b.LoadArg("a", "R3").LoadArg("n", "R4")
	offsets(b)
	for k := 0; k < 4; k++ {
		b.Raw("LXVD2X (R3)(%s), VS%d", vidx(k), k) // r[0..7] = a[0..7]
	}
	b.Raw("ADD $64, R3").Raw("ADD $-8, R4")
	b.Label("ploop").Raw("CMP R4, $8").Raw("BLT pfold")
	for k := 0; k < 4; k++ {
		b.Raw("LXVD2X (R3)(%s), VS%d", vidx(k), 4+k)
	}
	for k := 0; k < 4; k++ {
		b.XVADDDP(k, k, 4+k)
	}
	b.Raw("ADD $64, R3").Raw("ADD $-8, R4").Raw("BR ploop")
	b.Label("pfold")
	for k := 0; k < 4; k++ {
		b.Raw("XXPERMDI VS%d, VS%d, $2, VS%d", k, k, 4+k) // swap the doublewords
		b.XVADDDP(k, k, 4+k)                              // both lanes = r_pair
	}
	b.Raw("FADD F1, F0").Raw("FADD F3, F2").Raw("FADD F2, F0") // ((p0+p1)+(p2+p3))
	b.Label("ptail").Raw("CMP R4, $0").Raw("BEQ pdone").
		Raw("FMOVD 0(R3), F1").Raw("FADD F1, F0").
		Raw("ADD $8, R3").Raw("ADD $-1, R4").Raw("BR ptail")
	b.Label("pdone").StoreRet("F0", "ret").Ret()
	return b.Func()
}
