//go:build ignore

// Command gen produces cmulto_s390x.s with go-asmgen: the vector-facility (z13
// baseline, so no gate) kernel of CMulTo (cmulto.go), n complex numbers in one call,
// four per step then the last two and one when n has those bits (go-simd/floats,
// "CMulTo: no s390x/ppc64le kernel").
//
// A vector register holds exactly one complex number, element 0 the real part (s390x is
// big-endian: the first float64 in memory is element 0). For c = (ar, ai) of a and
// w = (br, bi) of b:
//
//	V1 = (br, br)  VREPG $0     V2 = (bi, bi) VREPG $1, then V2 ^= sign (VX) = (-bi, bi)
//	V3 = (ai, ar)  VPDI $4      V4 = c*(br, br)  VFMDB
//	V5 = V3*V2 + V4 = (ar*br - ai*bi, ai*br + ar*bi)  VFMADB
//
// The sign vector is built once: VGBM $0x0100 sets byte 7 to 0xff, VESLG $63 moves its
// low bit to the sign bit of element 0. Reads of a step are done before its stores, so
// dst may be a or b exactly. Operand orders: VFMADB Va, Vb, Vc, Vd is Vd = Va*Vb + Vc;
// VFMDB Va, Vb, Vd is Vd = Va*Vb.
//
// Run: go run -modfile=<modfile requiring github.com/go-asmgen/asmgen v0.23.2> cmulto_s390x_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/s390x"
)

func main() {
	f := emit.NewFile("s390x")
	f.Add(kernel().Func())
	if err := os.WriteFile("cmulto_s390x.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote cmulto_s390x.s")
}

// load emits the two loads of the complex number at byte offset off, registers from v.
func load(b *s390x.Builder, off, v int) {
	b.Raw("VL %d(R2), V%d", off, v)   // c
	b.Raw("VL %d(R3), V%d", off, v+1) // w
}

// compute emits the product into V(v+6) and stores it at byte offset off.
func compute(b *s390x.Builder, off, v int) {
	b.Raw("VREPG $0, V%d, V%d", v+1, v+2)
	b.Raw("VREPG $1, V%d, V%d", v+1, v+3)
	b.Raw("VX V31, V%d, V%d", v+3, v+3)
	b.Raw("VPDI $4, V%d, V%d, V%d", v, v, v+4)
	b.Raw("VFMDB V%d, V%d, V%d", v, v+2, v+5)
	b.Raw("VFMADB V%d, V%d, V%d, V%d", v+4, v+3, v+5, v+6)
	b.Raw("VST V%d, %d(R1)", v+6, off)
}

// block emits k numbers (k <= 4): all loads, then the arithmetic, then the stores.
func block(b *s390x.Builder, k int) {
	for i := 0; i < k; i++ {
		load(b, 16*i, 8*i)
	}
	for i := 0; i < k; i++ {
		compute(b, 16*i, 8*i)
	}
}

func kernel() *s390x.Builder {
	sig := abi.LayoutArgs([]abi.Arg{abi.Scalar("dst", abi.Ptr), abi.Scalar("a", abi.Ptr), abi.Scalar("b", abi.Ptr), abi.Scalar("n", abi.Int64)}, nil)
	b := s390x.NewFunc("cmulToVX", sig, 0)
	// R1 dst, R2 a, R3 b, R4 n complex numbers, R5 the bits of n for the last steps.
	b.LoadArg("dst", "R1").LoadArg("a", "R2").LoadArg("b", "R3").LoadArg("n", "R4")
	b.Raw("VGBM $0x0100, V31").Raw("VESLG $63, V31, V31")
	b.Raw("MOVD $4, R9").Raw("CMPBLT R4, R9, two").Label("loop")
	block(b, 4)
	b.Raw("ADD $64, R1").Raw("ADD $64, R2").Raw("ADD $64, R3").Raw("ADD $-4, R4").
		Raw("CMPBGE R4, R9, loop").Label("two")
	b.Raw("MOVD R4, R5").Raw("AND $2, R5").Raw("CMPBEQ R5, $0, one")
	block(b, 2)
	b.Raw("ADD $32, R1").Raw("ADD $32, R2").Raw("ADD $32, R3").Label("one")
	b.Raw("AND $1, R4").Raw("CMPBEQ R4, $0, done")
	block(b, 1)
	b.Label("done").Ret()
	return b
}
