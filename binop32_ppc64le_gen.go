//go:build ignore

// Command gen produces binop32_ppc64le.s with go-asmgen: the float32 elementwise
// Add32, Sub32, Mul32, Div32 and Sqrt32 VSX kernels (go-ndarray/ndarray's, moved
// here so the kernel lives once).
//
// XVADDSP / XVSUBSP / XVMULSP / XVDIVSP / XVSQRTSP (go-asmgen's VSX encoders, WORDs)
// are one IEEE binary32 operation per lane, and the scalar tail uses FADDS / FSUBS
// / FMULS / FDIVS / FSQRTS, so every result is the correctly rounded one,
// bit-identical to the scalar Go loop. 16 elements (four registers) per iteration,
// then one. LXVW4X / STXVW4X put each word at the same lane in every operand and
// back at its own index. Only ISA-2.06 VSX operations: valid on POWER8. Each block
// is loaded, computed and stored before the next, so dst may alias an input
// exactly.
//
// Run: go run -modfile=<modfile requiring github.com/go-asmgen/asmgen v0.23.0> binop32_ppc64le_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/ppc64"
)

func main() {
	f := emit.NewFile("ppc64le")
	for _, k := range []struct{ name, vec, sca string }{
		{"add32VSX", "add", "FADDS"}, {"sub32VSX", "sub", "FSUBS"},
		{"mul32VSX", "mul", "FMULS"}, {"div32VSX", "div", "FDIVS"},
	} {
		f.Add(bin32Kernel(k.name, k.vec, k.sca))
	}
	f.Add(sqrt32Kernel())
	if err := os.WriteFile("binop32_ppc64le.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote binop32_ppc64le.s")
}

// offsets loads the byte offsets 16, 32, 48 into R14, R15, R16: LXVW4X takes only
// a (base)(index) address.
func offsets(b *ppc64.Builder) {
	b.Raw("MOVD $16, R14").Raw("MOVD $32, R15").Raw("MOVD $48, R16")
}

// vidx is the index register for vector k (0..3) of a 64-byte block.
func vidx(k int) string { return [...]string{"R0", "R14", "R15", "R16"}[k] }

// sqrt32Kernel builds sqrt32VSX(dst, src *float32, n int).
func sqrt32Kernel() *emit.Function {
	sig := ppc64.Layout([]string{"dst", "src", "n"}, []ppc64.Type{ppc64.Ptr, ppc64.Ptr, ppc64.Int64}, nil, nil)
	b := ppc64.NewFunc("sqrt32VSX", sig, 0)
	b.LoadArg("dst", "R3").LoadArg("src", "R5").LoadArg("n", "R4")
	offsets(b)
	b.Label("qloop").Raw("CMP R4, $16").Raw("BLT qtail")
	for k := 0; k < 4; k++ {
		b.Raw("LXVW4X (R5)(%s), VS%d", vidx(k), 32+k)
	}
	for k := 0; k < 4; k++ {
		b.XVSQRTSP(36+k, 32+k)
	}
	for k := 0; k < 4; k++ {
		b.Raw("STXVW4X VS%d, (R3)(%s)", 36+k, vidx(k))
	}
	b.Raw("ADD $64, R3").Raw("ADD $64, R5").Raw("ADD $-16, R4").Raw("BR qloop")
	b.Label("qtail").Raw("CMP R4, $0").Raw("BEQ qdone").
		Raw("FMOVS 0(R5), F1").Raw("FSQRTS F1, F1").Raw("FMOVS F1, 0(R3)").
		Raw("ADD $4, R3").Raw("ADD $4, R5").Raw("ADD $-1, R4").Raw("BR qtail")
	b.Label("qdone").Ret()
	return b.Func()
}

// bin32Kernel builds name(dst, a, b *float32, n int): dst[i] = a[i] OP b[i].
func bin32Kernel(name, op, sca string) *emit.Function {
	sig := ppc64.Layout([]string{"dst", "a", "b", "n"},
		[]ppc64.Type{ppc64.Ptr, ppc64.Ptr, ppc64.Ptr, ppc64.Int64}, nil, nil)
	b := ppc64.NewFunc(name, sig, 0)
	b.LoadArg("dst", "R3").LoadArg("a", "R5").LoadArg("b", "R6").LoadArg("n", "R4")
	offsets(b)
	b.Label("floop").Raw("CMP R4, $16").Raw("BLT ftail")
	for k := 0; k < 4; k++ {
		b.Raw("LXVW4X (R5)(%s), VS%d", vidx(k), 32+k)
		b.Raw("LXVW4X (R6)(%s), VS%d", vidx(k), 36+k)
	}
	for k := 0; k < 4; k++ {
		switch op {
		case "add":
			b.XVADDSP(40+k, 32+k, 36+k)
		case "sub":
			b.XVSUBSP(40+k, 32+k, 36+k)
		case "mul":
			b.XVMULSP(40+k, 32+k, 36+k)
		case "div":
			b.XVDIVSP(40+k, 32+k, 36+k)
		}
	}
	for k := 0; k < 4; k++ {
		b.Raw("STXVW4X VS%d, (R3)(%s)", 40+k, vidx(k))
	}
	b.Raw("ADD $64, R3").Raw("ADD $64, R5").Raw("ADD $64, R6").Raw("ADD $-16, R4").Raw("BR floop")
	b.Label("ftail").Raw("CMP R4, $0").Raw("BEQ fdone").
		Raw("FMOVS 0(R5), F1").Raw("FMOVS 0(R6), F2").Raw("%s F2, F1", sca).Raw("FMOVS F1, 0(R3)").
		Raw("ADD $4, R3").Raw("ADD $4, R5").Raw("ADD $4, R6").Raw("ADD $-1, R4").Raw("BR ftail")
	b.Label("fdone").Ret()
	return b.Func()
}
