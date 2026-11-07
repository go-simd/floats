//go:build ignore

// Command gen produces binop_amd64.s with go-asmgen: the float64 elementwise
// Add, Sub, Div, Sqrt and ClampScalar kernels (K1 of the go-scikit performance
// campaign: they were private to go-ndarray's internal/kernels).
//
// Every instruction is a plain AVX one, one IEEE rounding per element, so the
// result is bit-identical to the scalar Go loop on every input:
//
//   - VADDPD / VSUBPD / VDIVPD / VSQRTPD (packed double) and VADDSD / VSUBSD /
//     VDIVSD / VSQRTSD (scalar double): Intel SDM vol. 2.
//   - ClampScalar: VMAXPD / VMINPD with a broadcast bound. VMAXPD/VMINPD return
//     the second source (the memory / last Plan 9 operand) when the operands are
//     equal or either is a NaN, and compare strictly: VMAXPD(src1, src2) = src1 >
//     src2 ? src1 : src2. `VMAXPD v, lo, r` is lo > v ? lo : v, `VMINPD r, hi, r`
//     is hi < r ? hi : r: the reference of clampF64 in kernels_amd64_gen.go.
//
// No FMA appears (nothing here multiplies and adds). Each 4-vector block is
// loaded, computed and stored before the next, so dst may alias an input exactly.
//
// Run: go run -modfile=<modfile requiring github.com/go-asmgen/asmgen v0.23.0> binop_amd64_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/amd64"
	"github.com/go-asmgen/asmgen/emit"
)

func main() {
	f := emit.NewFile("amd64")
	f.Add(binop("addAVX2", "VADDPD", "VADDSD").Func())
	f.Add(binop("subAVX2", "VSUBPD", "VSUBSD").Func())
	f.Add(binop("divAVX2", "VDIVPD", "VDIVSD").Func())
	f.Add(sqrt().Func())
	f.Add(clampScalar().Func())
	if err := os.WriteFile("binop_amd64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote binop_amd64.s")
}

// elemLoop emits the x4 / x1 / scalar loop skeleton of kernels_amd64_gen.go.
func elemLoop(b *amd64.Builder, vec func(off int, reg string), sca func()) {
	regs := []string{"Y1", "Y2", "Y3", "Y4"}
	b.Raw("XORQ AX, AX").
		Label("uloop").
		Raw("LEAQ 16(AX), R8").Raw("CMPQ R8, CX").Raw("JGT vloop")
	for k, r := range regs {
		vec(32*k, r)
	}
	b.Raw("ADDQ $16, AX").Raw("JMP uloop").
		Label("vloop").
		Raw("LEAQ 4(AX), R8").Raw("CMPQ R8, CX").Raw("JGT sloop")
	vec(0, "Y1")
	b.Raw("ADDQ $4, AX").Raw("JMP vloop").
		Label("sloop").
		Raw("CMPQ AX, CX").Raw("JGE done")
	sca()
	b.Raw("ADDQ $1, AX").Raw("JMP sloop").
		Label("done")
}

func mem(base string, off int) string {
	if off == 0 {
		return fmt.Sprintf("(%s)(AX*8)", base)
	}
	return fmt.Sprintf("%d(%s)(AX*8)", off, base)
}

// binop: dst[i] = a[i] OP b[i]; func(dst, a, b []float64).
func binop(name, vop, sop string) *amd64.Builder {
	sig := abi.LayoutArgs([]abi.Arg{abi.Slice("dst"), abi.Slice("a"), abi.Slice("b")}, nil)
	b := amd64.NewFunc(name, sig, 0)
	b.LoadArg("dst_base", "DI").LoadArg("a_base", "SI").LoadArg("a_len", "CX").LoadArg("b_base", "DX")
	elemLoop(b, func(off int, r string) {
		b.Raw("VMOVUPD %s, %s", mem("SI", off), r).
			Raw("%s %s, %s, %s", vop, mem("DX", off), r, r). // r = a OP b
			Raw("VMOVUPD %s, %s", r, mem("DI", off))
	}, func() {
		b.Raw("VMOVSD (SI)(AX*8), X1").
			Raw("%s (DX)(AX*8), X1, X1", sop).
			Raw("VMOVSD X1, (DI)(AX*8)")
	})
	b.Raw("VZEROUPPER").Ret()
	return b
}

// sqrt: dst[i] = sqrt(src[i]); func(dst, src []float64).
func sqrt() *amd64.Builder {
	sig := abi.LayoutArgs([]abi.Arg{abi.Slice("dst"), abi.Slice("src")}, nil)
	b := amd64.NewFunc("sqrtAVX2", sig, 0)
	b.LoadArg("dst_base", "DI").LoadArg("src_base", "SI").LoadArg("src_len", "CX")
	elemLoop(b, func(off int, r string) {
		b.Raw("VSQRTPD %s, %s", mem("SI", off), r).
			Raw("VMOVUPD %s, %s", r, mem("DI", off))
	}, func() {
		b.Raw("VMOVSD (SI)(AX*8), X1").
			Raw("VSQRTSD X1, X1, X1").
			Raw("VMOVSD X1, (DI)(AX*8)")
	})
	b.Raw("VZEROUPPER").Ret()
	return b
}

// clampScalar: dst[i] = (lo > v ? lo : v), then (hi < r ? hi : r), with scalar
// bounds; func(dst, v []float64, lo, hi float64). Y0 / Y5 = broadcast lo / hi
// (X0 / X5 keep them for the scalar tail).
func clampScalar() *amd64.Builder {
	sig := abi.LayoutArgs([]abi.Arg{abi.Slice("dst"), abi.Slice("v"),
		abi.Scalar("lo", abi.Float64), abi.Scalar("hi", abi.Float64)}, nil)
	b := amd64.NewFunc("clampScalarAVX2", sig, 0)
	b.LoadArg("dst_base", "DI").LoadArg("v_base", "SI").LoadArg("v_len", "CX").
		LoadArg("lo", "X0").LoadArg("hi", "X5").
		Raw("VBROADCASTSD X0, Y0").Raw("VBROADCASTSD X5, Y5")
	elemLoop(b, func(off int, r string) {
		b.Raw("VMAXPD %s, Y0, %s", mem("SI", off), r). // r = lo > v ? lo : v
								Raw("VMINPD %s, Y5, %s", r, r). // r = hi < r ? hi : r
								Raw("VMOVUPD %s, %s", r, mem("DI", off))
	}, func() {
		b.Raw("VMAXSD (SI)(AX*8), X0, X1").
			Raw("VMINSD X1, X5, X1").
			Raw("VMOVSD X1, (DI)(AX*8)")
	})
	b.Raw("VZEROUPPER").Ret()
	return b
}
