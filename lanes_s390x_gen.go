//go:build ignore

// Command gen produces lanes_s390x.s with go-asmgen: the s390x vector-facility
// (z13 baseline) float64 elementwise kernels Add, Sub, Mul, Div, Sqrt and the
// eight-accumulator pairwise-sum block that SumPairwise runs. The elementwise
// kernels are go-ndarray's addVX ... sqrtVX (internal/kernels/asmgen/s390x),
// moved here (floats#30); the block is new, in NumPy's order.
//
// Each element is one IEEE operation (VFADB, VFSDB, VFMDB, VFDDB, VFSQDB; FADD,
// FSUB, FMUL, FDIV, FSQRT in the tail), no fused instruction, so the results are
// the scalar loop's bit for bit.
//
// Operand order, measured by ndarray under s390x emulation (the IBM manuals give
// the other order): VFSDB V1, V2, V3 is V3 = V2 - V1, VFDDB V1, V2, V3 is
// V3 = V2 / V1, and FSUB F1, F0 is F0 = F0 - F1.
//
// pwBlockVX(a, n): 8 <= n <= 128. The eight accumulators r[0..7] of NumPy's
// DOUBLE_pairwise_sum are four two-lane registers V0..V3 = (r0,r1) (r2,r3) (r4,r5)
// (r6,r7), loaded by VLM from a[0:8] and then added to a[8k : 8k+8] block by
// block, so every lane receives exactly the additions of the scalar r[j]. The
// fold ((r0+r1)+(r2+r3)) + ((r4+r5)+(r6+r7)) adds lane 1 to lane 0 of each
// register (VREPG $1 + WFADB; the sum of two doubles is commutative, so which
// lane is which does not matter on a big-endian machine), then the leftover
// n % 8 elements are added to the result one at a time, as NumPy does.
//
// Run: go run -modfile=<modfile requiring github.com/go-asmgen/asmgen v0.23.0> lanes_s390x_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/s390x"
)

func main() {
	f := emit.NewFile("s390x")
	f.Add(sqrtKernel())
	for _, k := range []struct{ name, vec, sca string }{
		{"addVX", "VFADB", "FADD"}, {"subVX", "VFSDB", "FSUB"},
		{"mulVX", "VFMDB", "FMUL"}, {"divVX", "VFDDB", "FDIV"},
	} {
		f.Add(binKernel(k.name, k.vec, k.sca))
	}
	f.Add(pwBlockKernel())
	if err := os.WriteFile("lanes_s390x.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote lanes_s390x.s")
}

// sqrtKernel builds sqrtVX(dst, src *float64, n int).
func sqrtKernel() *emit.Function {
	sig := s390x.Layout([]string{"dst", "src", "n"},
		[]s390x.Type{s390x.Ptr, s390x.Ptr, s390x.Int64}, nil, nil)
	b := s390x.NewFunc("sqrtVX", sig, 0)
	b.LoadArg("dst", "R1").LoadArg("src", "R2").LoadArg("n", "R4")
	b.Raw("loop:").Raw("CMP R4, $8").Raw("BLT tail").
		Raw("VLM (R2), V0, V3")
	for r := 0; r < 4; r++ {
		b.Raw("VFSQDB V%d, V%d", r, r)
	}
	b.Raw("VSTM V0, V3, (R1)").
		Raw("ADD $64, R1").Raw("ADD $64, R2").Raw("SUB $8, R4").Raw("BR loop").
		Raw("tail:").Raw("CMP R4, $0").Raw("BEQ done").
		Raw("FMOVD (R2), F0").Raw("FSQRT F0, F0").Raw("FMOVD F0, (R1)").
		Raw("ADD $8, R1").Raw("ADD $8, R2").Raw("SUB $1, R4").Raw("BR tail").
		Raw("done:").Ret()
	return b.Func()
}

// binKernel builds NAME(dst, a, b *float64, n int): dst = a OP b. vec computes
// Vd = Vn OP Vm from "vec Vm, Vn, Vd"; sca computes F0 = F0 OP F1 from
// "sca F1, F0". Each 8-element block is loaded and computed before it is
// stored, so dst may alias an input exactly.
func binKernel(name, vec, sca string) *emit.Function {
	sig := s390x.Layout([]string{"dst", "a", "b", "n"},
		[]s390x.Type{s390x.Ptr, s390x.Ptr, s390x.Ptr, s390x.Int64}, nil, nil)
	b := s390x.NewFunc(name, sig, 0)
	b.LoadArg("dst", "R1").LoadArg("a", "R2").LoadArg("b", "R3").LoadArg("n", "R4")
	b.Raw("loop:").Raw("CMP R4, $8").Raw("BLT tail").
		Raw("VLM (R2), V0, V3"). // a
		Raw("VLM (R3), V4, V7")  // b
	for r := 0; r < 4; r++ {
		b.Raw("%s V%d, V%d, V%d", vec, 4+r, r, r) // V_r = a OP b
	}
	b.Raw("VSTM V0, V3, (R1)").
		Raw("ADD $64, R1").Raw("ADD $64, R2").Raw("ADD $64, R3").Raw("SUB $8, R4").Raw("BR loop").
		Raw("tail:").Raw("CMP R4, $0").Raw("BEQ done").
		Raw("FMOVD (R2), F0").Raw("FMOVD (R3), F1").
		Raw(sca + " F1, F0").
		Raw("FMOVD F0, (R1)").
		Raw("ADD $8, R1").Raw("ADD $8, R2").Raw("ADD $8, R3").Raw("SUB $1, R4").Raw("BR tail").
		Raw("done:").Ret()
	return b.Func()
}

// pwBlockKernel builds pwBlockVX(a *float64, n int) float64 for 8 <= n <= 128.
func pwBlockKernel() *emit.Function {
	sig := s390x.Layout([]string{"a", "n"}, []s390x.Type{s390x.Ptr, s390x.Int64},
		[]string{"ret"}, []s390x.Type{s390x.Float64})
	b := s390x.NewFunc("pwBlockVX", sig, 0)
	b.LoadArg("a", "R1").LoadArg("n", "R2").
		Raw("VLM (R1), V0, V3"). // r[0..7] = a[0..7]
		Raw("ADD $64, R1").Raw("SUB $8, R2").
		Raw("loop:").Raw("CMP R2, $8").Raw("BLT fold").
		Raw("VLM (R1), V4, V7")
	for r := 0; r < 4; r++ {
		b.Raw("VFADB V%d, V%d, V%d", 4+r, r, r) // r_pair += a block
	}
	b.Raw("ADD $64, R1").Raw("SUB $8, R2").Raw("BR loop").
		Raw("fold:")
	for r := 0; r < 4; r++ {
		b.Raw("VREPG $1, V%d, V%d", r, 4+r).
			Raw("WFADB V%d, V%d, V%d", r, 4+r, r) // F_r = lane 0 + lane 1
	}
	b.Raw("WFADB V1, V0, V0").Raw("WFADB V3, V2, V2").
		Raw("WFADB V2, V0, V0"). // F0 = ((r0+r1)+(r2+r3)) + ((r4+r5)+(r6+r7))
		Raw("tail:").Raw("CMP R2, $0").Raw("BEQ done").
		Raw("FMOVD (R1), F1").Raw("FADD F1, F0").
		Raw("ADD $8, R1").Raw("SUB $1, R2").Raw("BR tail").
		Raw("done:").
		StoreRet("F0", "ret").
		Ret()
	return b.Func()
}
