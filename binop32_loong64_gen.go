//go:build ignore

// Command gen produces binop32_loong64.s with go-asmgen: the float32 elementwise
// Add32, Sub32, Mul32, Div32 and Sqrt32 LSX kernels (go-ndarray/ndarray's, moved
// here so the kernel lives once).
//
// VFADDF / VFSUBF / VFMULF / VFDIVF / VFSQRTF (vfadd.s, vfsub.s, vfmul.s,
// vfdiv.s, vfsqrt.s) on a 128-bit LSX register are one IEEE binary32 operation
// per lane, and the scalar tail uses ADDF / SUBF / MULF / DIVF / SQRTF, so every
// result is the correctly rounded one, bit-identical to the scalar Go loop. 16
// elements (four registers, four independent chains) per iteration, then one.
// Each block is loaded, computed and stored before the next, so dst may alias an
// input exactly. LSX comes with every LASX CPU; the Go side gates on hasLASX.
//
// Run: go run -modfile=<modfile requiring github.com/go-asmgen/asmgen v0.23.0> binop32_loong64_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/loong64"
)

func main() {
	f := emit.NewFile("loong64")
	for _, k := range []struct{ name, vec, sca string }{
		{"add32LSX", "VADDF", "ADDF"}, {"sub32LSX", "VSUBF", "SUBF"},
		{"mul32LSX", "VMULF", "MULF"}, {"div32LSX", "VDIVF", "DIVF"},
	} {
		f.Add(bin32Kernel(k.name, k.vec, k.sca))
	}
	f.Add(sqrt32Kernel())
	if err := os.WriteFile("binop32_loong64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote binop32_loong64.s")
}

// sqrt32Kernel builds sqrt32LSX(dst, src *float32, n int).
func sqrt32Kernel() *emit.Function {
	sig := loong64.Layout([]string{"dst", "src", "n"}, []loong64.Type{loong64.Ptr, loong64.Ptr, loong64.Int64}, nil, nil)
	b := loong64.NewFunc("sqrt32LSX", sig, 0)
	b.LoadArg("dst", "R4").LoadArg("src", "R6").LoadArg("n", "R5")
	b.Label("qloop").Raw("MOVV $16, R12").Raw("BLT R5, R12, qtail")
	for k := 0; k < 4; k++ {
		b.Raw("VMOVQ %d(R6), V%d", 16*k, k)
	}
	for k := 0; k < 4; k++ {
		b.Raw("VFSQRTF V%d, V%d", k, 4+k)
	}
	for k := 0; k < 4; k++ {
		b.Raw("VMOVQ V%d, %d(R4)", 4+k, 16*k)
	}
	b.Raw("ADDV $64, R4").Raw("ADDV $64, R6").Raw("ADDV $-16, R5").Raw("JMP qloop")
	b.Label("qtail").Raw("BEQ R5, qdone").
		Raw("MOVF 0(R6), F1").Raw("SQRTF F1, F1").Raw("MOVF F1, 0(R4)").
		Raw("ADDV $4, R4").Raw("ADDV $4, R6").Raw("ADDV $-1, R5").Raw("JMP qtail")
	b.Label("qdone").Ret()
	return b.Func()
}

// bin32Kernel builds name(dst, a, b *float32, n int): dst[i] = a[i] OP b[i].
func bin32Kernel(name, vec, sca string) *emit.Function {
	sig := loong64.Layout([]string{"dst", "a", "b", "n"},
		[]loong64.Type{loong64.Ptr, loong64.Ptr, loong64.Ptr, loong64.Int64}, nil, nil)
	b := loong64.NewFunc(name, sig, 0)
	b.LoadArg("dst", "R4").LoadArg("a", "R6").LoadArg("b", "R7").LoadArg("n", "R5")
	b.Label("floop").Raw("MOVV $16, R12").Raw("BLT R5, R12, ftail")
	for k := 0; k < 4; k++ {
		b.Raw("VMOVQ %d(R6), V%d", 16*k, k)
		b.Raw("VMOVQ %d(R7), V%d", 16*k, 4+k)
	}
	for k := 0; k < 4; k++ {
		b.Raw("%s V%d, V%d, V%d", vec, 4+k, k, 8+k) // V(8+k) = V(k) OP V(4+k)
	}
	for k := 0; k < 4; k++ {
		b.Raw("VMOVQ V%d, %d(R4)", 8+k, 16*k)
	}
	b.Raw("ADDV $64, R4").Raw("ADDV $64, R6").Raw("ADDV $64, R7").Raw("ADDV $-16, R5").Raw("JMP floop")
	b.Label("ftail").Raw("BEQ R5, fdone").
		Raw("MOVF 0(R6), F1").Raw("MOVF 0(R7), F2").Raw("%s F2, F1", sca).Raw("MOVF F1, 0(R4)").
		Raw("ADDV $4, R4").Raw("ADDV $4, R6").Raw("ADDV $4, R7").Raw("ADDV $-1, R5").Raw("JMP ftail")
	b.Label("fdone").Ret()
	return b.Func()
}
