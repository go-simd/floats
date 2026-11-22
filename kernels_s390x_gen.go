//go:build ignore

// Command gen produces kernels_s390x.s with go-asmgen: z/Architecture
// vector-facility reduction kernels for the dot product, sum and
// sum-of-squared-differences, in float64 (dotVX, sumVX, sumSqDiffVX) and
// float32 (dot32VX, sum32VX, sumSqDiff32VX).
//
// Eight independent accumulators (V0..V7). A vector FMA or add has a latency
// of several cycles, so the earlier single-accumulator kernels (one V0, stride
// 2) ran at one chain per latency: measured on z17 (linux1, steal 0), the f64
// Sum kernel at 0.62 ns/element against 0.36 ns/element for the plain Go Dot
// loop. The bulk loop processes eight registers per iteration (16 f64 or 32
// f32 elements), folds the accumulators with a balanced add tree into V0, then
// a one-register loop and a scalar loop take the rest. The summation order is
// unspecified: the contract is the error bound gamma_n * sum|terms|
// (reduce_bound_test.go).
//
// The float32 kernels use the single-precision vector-FP ops of the
// vector-enhancements facility 1 (z14): VFASB, VFSSB and VFMASB. The released
// assembler has no mnemonics for them, so they are emitted as their fixed
// encodings through WORD/BYTE (e7 | V1 V2 | V3 .. | .. M | .. | opcode), the
// same way binop32_s390x.s does; the encoder is checked against the bytes the
// assembler produced for VFMADB (e7033300008f). The Go side gates the float32
// kernels on hasVXE and falls back to the lane-blocked scalar reference.
//
// Big-endian note: every operation is an element-wise reduction ending in a
// commutative horizontal sum, so lane numbering does not change the result.
//
// Operand orders (verified against the released assembler + qemu):
//
//	VFMADB Va, Vb, Vc, Vd   Vd = Va*Vb + Vc
//	VFSDB  Va, Vb, Vd       Vd = Vb - Va
//	FADD   Fs, Fd           Fd = Fd + Fs
//	FMADD  Fa, Fb, Fd       Fd = Fd + Fa*Fb
//
// Run: go run -modfile=<modfile requiring github.com/go-asmgen/asmgen v0.23.2> kernels_s390x_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/s390x"
)

const redK = 8

type kind int

const (
	kDot kind = iota
	kSum
	kSSD
)

func main() {
	f := emit.NewFile("s390x")
	for _, k := range []kind{kDot, kSum, kSSD} {
		f.Add(kernel(k, false).Func())
	}
	for _, k := range []kind{kDot, kSum, kSSD} {
		f.Add(kernel(k, true).Func())
	}
	f.Add(axpyF64().Func())
	if err := os.WriteFile("kernels_s390x.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote kernels_s390x.s")
}

// vrrc encodes a word-element (M4=2) VRR-c vector-FP op: V1 = V2 op V3.
func vrrc(op, v1, v2, v3 int) string {
	return fmt.Sprintf("WORD $0xe7%02x%02x00; BYTE $0x20; BYTE $0x%02x", v1<<4|v2, v3<<4, op)
}

// vfmasb encodes VFMA with word elements: V1 = V2*V3 + V4.
func vfmasb(v1, v2, v3, v4 int) string {
	return fmt.Sprintf("WORD $0xe7%02x%02x00; BYTE $0x%02x; BYTE $0x8f", v1<<4|v2, v3<<4|2, v4<<4)
}

func kernel(k kind, f32 bool) *s390x.Builder {
	name := map[kind]string{kDot: "dot", kSum: "sum", kSSD: "sumSqDiff"}[k]
	lanes, esz := 2, 8
	ldF, scalarAdd, scalarFMA, scalarSub := "FMOVD", "FADD", "FMADD", "FSUB"
	rt := abi.Float64
	suffix := ""
	if f32 {
		lanes, esz = 4, 4
		ldF, scalarAdd, scalarFMA, scalarSub = "FMOVS", "FADDS", "FMADDS", "FSUBS"
		rt = abi.Float32
		suffix = "32"
	}
	blk := redK * lanes
	two := k != kSum
	args := []abi.Arg{abi.Slice("a")}
	if two {
		args = append(args, abi.Slice("b"))
	}
	sig := abi.LayoutArgs(args, []abi.Arg{abi.Scalar("ret", rt)})
	b := s390x.NewFunc(name+suffix+"VX", sig, 0)
	b.LoadArg("a_base", "R1").LoadArg("a_len", "R2")
	if two {
		b.LoadArg("b_base", "R3")
	}

	vadd := func(d, x, y int) { // Vd = Vx + Vy
		if f32 {
			b.Raw("%s", vrrc(0xe3, d, x, y))
		} else {
			b.Raw("VFADB V%d, V%d, V%d", y, x, d)
		}
	}
	// op accumulates the registers loaded into ta/tb into accumulator ac.
	op := func(ac, ta, tb int) {
		switch k {
		case kDot:
			if f32 {
				b.Raw("%s", vfmasb(ac, ta, tb, ac))
			} else {
				b.Raw("VFMADB V%d, V%d, V%d, V%d", ta, tb, ac, ac)
			}
		case kSum:
			vadd(ac, ac, ta)
		case kSSD:
			if f32 {
				b.Raw("%s", vrrc(0xe2, ta, ta, tb)) // ta = ta - tb
				b.Raw("%s", vfmasb(ac, ta, ta, ac))
			} else {
				b.Raw("VFSDB V%d, V%d, V%d", tb, ta, ta) // ta = ta - tb
				b.Raw("VFMADB V%d, V%d, V%d, V%d", ta, ta, ac, ac)
			}
		}
	}
	load := func(off, ta, tb int) {
		b.Raw("VL %d(R1), V%d", off, ta)
		if two {
			b.Raw("VL %d(R3), V%d", off, tb)
		}
	}

	b.Raw("MOVD R2, R4")
	for r := 0; r < redK; r++ {
		b.Raw("VZERO V%d", r)
	}
	b.Raw("MOVD $%d, R9", blk).Raw("CMPBLT R4, R9, rtail").
		Label("cloop")
	for kk := 0; kk < redK; kk++ {
		ta, tb := 8+2*(kk&1), 9+2*(kk&1)
		load(16*kk, ta, tb)
		op(kk, ta, tb)
	}
	b.Raw("ADD $128, R1")
	if two {
		b.Raw("ADD $128, R3")
	}
	b.Raw("ADD $%d, R4", -blk).Raw("CMPBGE R4, R9, cloop").
		Label("rtail")
	for _, p := range [][2]int{{0, 1}, {2, 3}, {4, 5}, {6, 7}, {0, 2}, {4, 6}, {0, 4}} {
		vadd(p[0], p[0], p[1])
	}
	b.Raw("MOVD $%d, R9", lanes).
		Label("vloop").
		Raw("CMPBLT R4, R9, vtail")
	load(0, 8, 9)
	op(0, 8, 9)
	b.Raw("ADD $16, R1")
	if two {
		b.Raw("ADD $16, R3")
	}
	b.Raw("ADD $%d, R4", -lanes).Raw("BR vloop").
		Label("vtail")
	if f32 {
		// Extract every lane before the first LDGR: F0..F3 alias the high
		// doublewords of V0..V3, so LDGR F0 would clobber lanes 0-1 of V0.
		gpr := []string{"R7", "R8", "R10", "R11"}
		for l := 0; l < lanes; l++ {
			b.Raw("VLGVF $%d, V0, %s", l, gpr[l]).Raw("SLD $32, %s, %s", gpr[l], gpr[l])
		}
		for l := 0; l < lanes; l++ {
			b.Raw("LDGR %s, F%d", gpr[l], l)
		}
		b.Raw("%s F1, F0", scalarAdd).Raw("%s F3, F2", scalarAdd).Raw("%s F2, F0", scalarAdd)
	} else {
		b.Raw("VLGVG $0, V0, R7").Raw("VLGVG $1, V0, R8").
			Raw("LDGR R7, F0").Raw("LDGR R8, F1").Raw("FADD F1, F0")
	}
	b.Label("sloop").
		Raw("CMPBEQ R4, $0, done").
		Raw("%s 0(R1), F1", ldF)
	switch k {
	case kDot:
		b.Raw("%s 0(R3), F2", ldF).Raw("%s F1, F2, F0", scalarFMA)
	case kSum:
		b.Raw("%s F1, F0", scalarAdd)
	case kSSD:
		b.Raw("%s 0(R3), F2", ldF).Raw("%s F2, F1", scalarSub).Raw("%s F1, F1, F0", scalarFMA)
	}
	b.Raw("ADD $%d, R1", esz)
	if two {
		b.Raw("ADD $%d, R3", esz)
	}
	b.Raw("ADD $-1, R4").Raw("BR sloop").
		Label("done").StoreRet("F0", "ret").Ret()
	return b
}

// axpyF64: y[i] = fma(alpha, x[i], y[i]), eight elements (four vectors) per
// iteration, then two, then one. R1 walks x, R3 walks y, R2 counts what is left.
func axpyF64() *s390x.Builder {
	sig := abi.LayoutArgs(
		[]abi.Arg{abi.Scalar("alpha", abi.Float64), abi.Slice("x"), abi.Slice("y")},
		nil,
	)
	b := s390x.NewFunc("axpyVX", sig, 0)
	b.LoadArg("x_base", "R1").LoadArg("x_len", "R2").LoadArg("y_base", "R3").
		Raw("FMOVD alpha+0(FP), F0").
		Raw("VLREPG alpha+0(FP), V3").
		Label("uloop").
		Raw("CMPBLT R2, $8, vloop").
		Raw("VL (R1), V1").Raw("VL 16(R1), V2").Raw("VL 32(R1), V4").Raw("VL 48(R1), V5").
		Raw("VL (R3), V6").Raw("VL 16(R3), V7").Raw("VL 32(R3), V8").Raw("VL 48(R3), V9").
		Raw("VFMADB V1, V3, V6, V6"). // Vd = Va*Vb + Vc
		Raw("VFMADB V2, V3, V7, V7").
		Raw("VFMADB V4, V3, V8, V8").
		Raw("VFMADB V5, V3, V9, V9").
		Raw("VST V6, (R3)").Raw("VST V7, 16(R3)").Raw("VST V8, 32(R3)").Raw("VST V9, 48(R3)").
		Raw("ADD $64, R1").Raw("ADD $64, R3").Raw("SUB $8, R2").Raw("BR uloop").
		Label("vloop").
		Raw("CMPBLT R2, $2, sloop").
		Raw("VL (R1), V1").Raw("VL (R3), V6").
		Raw("VFMADB V1, V3, V6, V6").
		Raw("VST V6, (R3)").
		Raw("ADD $16, R1").Raw("ADD $16, R3").Raw("SUB $2, R2").Raw("BR vloop").
		Label("sloop").
		Raw("CMPBEQ R2, $0, done").
		Raw("FMOVD (R1), F1").
		Raw("FMOVD (R3), F2").
		Raw("FMADD F0, F1, F2"). // F2 += F0*F1
		Raw("FMOVD F2, (R3)").
		Label("done").Ret()
	return b
}
