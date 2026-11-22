//go:build ignore

// Command gen produces binop32_s390x.s with go-asmgen: the float32 elementwise
// Add32, Sub32, Mul32, Div32 and Sqrt32 VXE kernels (go-ndarray/ndarray's, moved
// here so the kernel lives once).
//
// VFASB / VFSSB / VFMSB / VFDSB / VFSQSB (single-precision vector operations of
// the vector-enhancements facility 1, z14; go-asmgen emits them as WORD+BYTE+BYTE)
// are one IEEE binary32 operation per lane, and the scalar tail uses FADDS / FSUBS
// / FMULS / FDIVS / FSQRTS, so every result is the correctly rounded one,
// bit-identical to the scalar Go loop. 16 elements (four registers) per iteration,
// then one. Each block is loaded, computed and stored before the next, so dst may
// alias an input exactly. The Go side gates on hasVXE.
//
// Operand order, measured under s390x emulation: "FSUBS F1, F0" is F0 = F0 - F1.
//
// Run: go run -modfile=<modfile requiring github.com/go-asmgen/asmgen v0.23.0> binop32_s390x_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/s390x"
)

func main() {
	f := emit.NewFile("s390x")
	for _, k := range []struct{ name, op, sca string }{
		{"add32VXE", "add", "FADDS"}, {"sub32VXE", "sub", "FSUBS"},
		{"mul32VXE", "mul", "FMULS"}, {"div32VXE", "div", "FDIVS"},
	} {
		f.Add(bin32Kernel(k.name, k.op, k.sca))
	}
	f.Add(sqrt32Kernel())
	if err := os.WriteFile("binop32_s390x.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote binop32_s390x.s")
}

// sqrt32Kernel builds sqrt32VXE(dst, src *float32, n int).
func sqrt32Kernel() *emit.Function {
	sig := s390x.Layout([]string{"dst", "src", "n"},
		[]s390x.Type{s390x.Ptr, s390x.Ptr, s390x.Int64}, nil, nil)
	b := s390x.NewFunc("sqrt32VXE", sig, 0)
	b.LoadArg("dst", "R1").LoadArg("src", "R2").LoadArg("n", "R4")
	b.Raw("loop:").Raw("CMP R4, $16").Raw("BLT tail").
		Raw("VLM (R2), V0, V3")
	for r := 0; r < 4; r++ {
		b.VFSQSB(r, r)
	}
	b.Raw("VSTM V0, V3, (R1)").
		Raw("ADD $64, R1").Raw("ADD $64, R2").Raw("SUB $16, R4").Raw("BR loop").
		Raw("tail:").Raw("CMP R4, $0").Raw("BEQ done").
		Raw("FMOVS (R2), F0").Raw("FSQRTS F0, F0").Raw("FMOVS F0, (R1)").
		Raw("ADD $4, R1").Raw("ADD $4, R2").Raw("SUB $1, R4").Raw("BR tail").
		Raw("done:").Ret()
	return b.Func()
}

// bin32Kernel builds NAME(dst, a, b *float32, n int): dst = a OP b.
func bin32Kernel(name, op, sca string) *emit.Function {
	sig := s390x.Layout([]string{"dst", "a", "b", "n"},
		[]s390x.Type{s390x.Ptr, s390x.Ptr, s390x.Ptr, s390x.Int64}, nil, nil)
	b := s390x.NewFunc(name, sig, 0)
	b.LoadArg("dst", "R1").LoadArg("a", "R2").LoadArg("b", "R3").LoadArg("n", "R4")
	b.Raw("loop:").Raw("CMP R4, $16").Raw("BLT tail").
		Raw("VLM (R2), V0, V3"). // a
		Raw("VLM (R3), V4, V7")  // b
	for r := 0; r < 4; r++ { // V_r = a OP b
		switch op {
		case "add":
			b.VFASB(r, r, 4+r)
		case "sub":
			b.VFSSB(r, r, 4+r)
		case "mul":
			b.VFMSB(r, r, 4+r)
		case "div":
			b.VFDSB(r, r, 4+r)
		}
	}
	b.Raw("VSTM V0, V3, (R1)").
		Raw("ADD $64, R1").Raw("ADD $64, R2").Raw("ADD $64, R3").Raw("SUB $16, R4").Raw("BR loop").
		Raw("tail:").Raw("CMP R4, $0").Raw("BEQ done").
		Raw("FMOVS (R2), F0").Raw("FMOVS (R3), F1").
		Raw(sca + " F1, F0").
		Raw("FMOVS F0, (R1)").
		Raw("ADD $4, R1").Raw("ADD $4, R2").Raw("ADD $4, R3").Raw("SUB $1, R4").Raw("BR tail").
		Raw("done:").Ret()
	return b.Func()
}
