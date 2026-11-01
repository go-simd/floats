//go:build ignore

// Command gen produces nofma_amd64.s with go-asmgen: the UNFUSED float64
// elementwise kernels (AxpyNoFMA, XpbyNoFMA, Axpby3NoFMA, AxpyDiffNoFMA) and the
// eight-accumulator pairwise-sum block that SumPairwise runs on AVX.
//
// Unfused means one IEEE rounding per binary operation, in the order NumPy
// evaluates the temporaries of the equivalent expression (y + alpha*x is a
// multiply followed by an add; there is no VFMADD anywhere in this file). Every
// instruction used is a plain AVX arithmetic one:
//
//   - VMULPD / VADDPD / VSUBPD (packed double) and VMULSD / VADDSD / VSUBSD
//     (scalar double): Intel SDM vol. 2, MULPD, ADDPD, SUBPD, MULSD, ADDSD, SUBSD.
//   - VHADDPD: horizontal add, dst[0] = src1[0]+src1[1], dst[1] = src2[0]+src2[1],
//     dst[2] = src1[2]+src1[3], dst[3] = src2[2]+src2[3] (Intel SDM, HADDPD).
//   - VEXTRACTF128, VBROADCASTSD, VMOVUPD, VMOVSD, VZEROUPPER.
//
// Run: go run -modfile=<modfile requiring github.com/go-asmgen/asmgen v0.23.0> nofma_amd64_gen.go
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
	f.Add(axpyNoFMA().Func())
	f.Add(xpbyNoFMA().Func())
	f.Add(axpby3NoFMA().Func())
	f.Add(axpyDiffNoFMA().Func())
	f.Add(pwBlock().Func())
	if err := os.WriteFile("nofma_amd64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote nofma_amd64.s")
}

// axpySig is func(<scalar> float64, x, y []float64).
func axpySig(scalar string) abi.Signature {
	return abi.LayoutArgs(
		[]abi.Arg{abi.Scalar(scalar, abi.Float64), abi.Slice("x"), abi.Slice("y")}, nil)
}

// elemLoop emits the x4 / x1 / scalar loop skeleton, as in kernels_amd64_gen.go:
// vec(off, reg) is the per-vector body for YMM reg at byte offset off, sca() the
// scalar tail body (X1). Every access is at an index < n; each block is loaded,
// computed and stored before the next, so dst may alias an input exactly.
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

// axpyNoFMA: y[i] = y[i] + alpha*x[i], the product rounded first.
func axpyNoFMA() *amd64.Builder {
	b := amd64.NewFunc("axpyNoFMAAVX2", axpySig("alpha"), 0)
	b.LoadArg("alpha", "X0").LoadArg("x_base", "SI").LoadArg("x_len", "CX").LoadArg("y_base", "DI").
		Raw("VBROADCASTSD X0, Y0")
	elemLoop(b, func(off int, r string) {
		b.Raw("VMULPD %s, Y0, %s", mem("SI", off), r). // r = alpha*x
								Raw("VADDPD %s, %s, %s", mem("DI", off), r, r). // r = r + y
								Raw("VMOVUPD %s, %s", r, mem("DI", off))
	}, func() {
		b.Raw("VMULSD (SI)(AX*8), X0, X1").
			Raw("VADDSD (DI)(AX*8), X1, X1").
			Raw("VMOVSD X1, (DI)(AX*8)")
	})
	b.Raw("VZEROUPPER").Ret()
	return b
}

// xpbyNoFMA: y[i] = x[i] + beta*y[i], the product rounded first.
func xpbyNoFMA() *amd64.Builder {
	b := amd64.NewFunc("xpbyNoFMAAVX2", axpySig("beta"), 0)
	b.LoadArg("beta", "X0").LoadArg("x_base", "SI").LoadArg("x_len", "CX").LoadArg("y_base", "DI").
		Raw("VBROADCASTSD X0, Y0")
	elemLoop(b, func(off int, r string) {
		b.Raw("VMULPD %s, Y0, %s", mem("DI", off), r). // r = beta*y
								Raw("VADDPD %s, %s, %s", mem("SI", off), r, r). // r = r + x
								Raw("VMOVUPD %s, %s", r, mem("DI", off))
	}, func() {
		b.Raw("VMULSD (DI)(AX*8), X0, X1").
			Raw("VADDSD (SI)(AX*8), X1, X1").
			Raw("VMOVSD X1, (DI)(AX*8)")
	})
	b.Raw("VZEROUPPER").Ret()
	return b
}

// axpby3NoFMA: dst[i] = (alpha*a[i] + beta*b[i]) + gamma*c[i], each product and
// each sum rounded on its own, left to right.
func axpby3NoFMA() *amd64.Builder {
	sig := abi.LayoutArgs(
		[]abi.Arg{abi.Slice("dst"), abi.Scalar("alpha", abi.Float64), abi.Slice("a"),
			abi.Scalar("beta", abi.Float64), abi.Slice("b"),
			abi.Scalar("gamma", abi.Float64), abi.Slice("c")}, nil)
	b := amd64.NewFunc("axpby3NoFMAAVX2", sig, 0)
	b.LoadArg("alpha", "X0").LoadArg("beta", "X10").LoadArg("gamma", "X11").
		LoadArg("dst_base", "DI").LoadArg("a_base", "SI").LoadArg("a_len", "CX").
		LoadArg("b_base", "DX").LoadArg("c_base", "R9").
		Raw("VBROADCASTSD X0, Y0").Raw("VBROADCASTSD X10, Y10").Raw("VBROADCASTSD X11, Y11")
	elemLoop(b, func(off int, r string) {
		b.Raw("VMULPD %s, Y0, %s", mem("SI", off), r). // r = alpha*a
								Raw("VMULPD %s, Y10, Y5", mem("DX", off)). // t = beta*b
								Raw("VADDPD Y5, %s, %s", r, r).            // r = r + t
								Raw("VMULPD %s, Y11, Y5", mem("R9", off)). // t = gamma*c
								Raw("VADDPD Y5, %s, %s", r, r).            // r = r + t
								Raw("VMOVUPD %s, %s", r, mem("DI", off))
	}, func() {
		b.Raw("VMULSD (SI)(AX*8), X0, X1").
			Raw("VMULSD (DX)(AX*8), X10, X5").
			Raw("VADDSD X5, X1, X1").
			Raw("VMULSD (R9)(AX*8), X11, X5").
			Raw("VADDSD X5, X1, X1").
			Raw("VMOVSD X1, (DI)(AX*8)")
	})
	b.Raw("VZEROUPPER").Ret()
	return b
}

// axpyDiffNoFMA: dst[i] = a[i] + f*(b[i] - c[i]): the difference, then the
// product, then the sum, each rounded (NumPy's temporaries for a + F*(b - c)).
func axpyDiffNoFMA() *amd64.Builder {
	sig := abi.LayoutArgs(
		[]abi.Arg{abi.Slice("dst"), abi.Slice("a"), abi.Scalar("f", abi.Float64),
			abi.Slice("b"), abi.Slice("c")}, nil)
	b := amd64.NewFunc("axpyDiffNoFMAAVX2", sig, 0)
	b.LoadArg("f", "X0").LoadArg("dst_base", "DI").LoadArg("a_base", "SI").LoadArg("a_len", "CX").
		LoadArg("b_base", "DX").LoadArg("c_base", "R9").
		Raw("VBROADCASTSD X0, Y0")
	elemLoop(b, func(off int, r string) {
		b.Raw("VMOVUPD %s, %s", mem("DX", off), r).
			Raw("VSUBPD %s, %s, %s", mem("R9", off), r, r). // r = b - c
			Raw("VMULPD %s, Y0, %s", r, r).                 // r = f*r
			Raw("VADDPD %s, %s, %s", mem("SI", off), r, r). // r = r + a
			Raw("VMOVUPD %s, %s", r, mem("DI", off))
	}, func() {
		b.Raw("VMOVSD (DX)(AX*8), X1").
			Raw("VSUBSD (R9)(AX*8), X1, X1").
			Raw("VMULSD X1, X0, X1").
			Raw("VADDSD (SI)(AX*8), X1, X1").
			Raw("VMOVSD X1, (DI)(AX*8)")
	})
	b.Raw("VZEROUPPER").Ret()
	return b
}

// pwBlock is NumPy's DOUBLE_pairwise_sum for 8 <= n <= 128 (the caller
// guarantees both bounds): the eight accumulators r[0..7] live in two YMM
// registers, Y0 = r[0..3] and Y1 = r[4..7], one lane each, so every lane sees
// exactly the additions NumPy's scalar code performs on it. Then
//
//	res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
//
// with VHADDPD forming the pairs (r0+r1, r0+r1, r2+r3, r2+r3), and the n % 8
// remaining elements are added one at a time to res.
func pwBlock() *amd64.Builder {
	sig := abi.LayoutArgs(
		[]abi.Arg{abi.Slice("a")},
		[]abi.Arg{abi.Scalar("ret", abi.Float64)})
	b := amd64.NewFunc("pwBlockAVX2", sig, 0)
	b.LoadArg("a_base", "AX").LoadArg("a_len", "CX").
		Raw("VMOVUPD (AX), Y0").Raw("VMOVUPD 32(AX), Y1").
		Raw("MOVQ CX, DX").Raw("ANDQ $-8, DX"). // DX = n - n%8
		Raw("MOVQ $8, DI").
		Label("loop").
		Raw("CMPQ DI, DX").Raw("JGE reduce").
		Raw("VADDPD (AX)(DI*8), Y0, Y0").
		Raw("VADDPD 32(AX)(DI*8), Y1, Y1").
		Raw("ADDQ $8, DI").Raw("JMP loop").
		Label("reduce").
		Raw("VHADDPD Y0, Y0, Y0").      // r0+r1, r0+r1, r2+r3, r2+r3
		Raw("VHADDPD Y1, Y1, Y1").      // r4+r5, r4+r5, r6+r7, r6+r7
		Raw("VEXTRACTF128 $1, Y0, X2"). // r2+r3
		Raw("VADDSD X2, X0, X0").       // (r0+r1)+(r2+r3)
		Raw("VEXTRACTF128 $1, Y1, X3"). // r6+r7
		Raw("VADDSD X3, X1, X1").       // (r4+r5)+(r6+r7)
		Raw("VADDSD X1, X0, X0").       // res
		Raw("VZEROUPPER").
		Label("tail").
		Raw("CMPQ DI, CX").Raw("JGE done").
		Raw("VADDSD (AX)(DI*8), X0, X0").
		Raw("ADDQ $1, DI").Raw("JMP tail").
		Label("done").StoreRet("X0", "ret").Ret()
	return b
}
