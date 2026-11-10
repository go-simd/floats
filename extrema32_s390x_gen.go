//go:build ignore

// Command gen produces extrema32_s390x.s with go-asmgen: the VXE float32 Max
// and Min kernels max32VXE/min32VXE (go-ndarray's, moved here so the kernel
// lives once), for n >= 16. VFMAXSB/VFMINSB and their scalar W forms come with
// the vector-enhancements facility 1 (z14), so the Go side gates them on
// hasVXE. In mode 1 (Java's Math.max/min) a NaN operand gives NaN and -0 orders
// below +0, Go's rule. The operation is associative under it, so four four-lane
// accumulators seeded with the first 16 elements, folded at the end (lanes 2-3
// onto 0-1 by an 8-byte VSLDB rotation, then lane 1 onto 0), give the
// sequential result.
//
// Run: go run -modfile=<modfile requiring github.com/go-asmgen/asmgen v0.23.0> extrema32_s390x_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/s390x"
)

func main() {
	f := emit.NewFile("s390x")
	f.Add(extremum32Kernel("max32VXE", "VFMAXSB", "WFMAXSB"))
	f.Add(extremum32Kernel("min32VXE", "VFMINSB", "WFMINSB"))
	if err := os.WriteFile("extrema32_s390x.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote extrema32_s390x.s")
}

func extremum32Kernel(name, vec, sca string) *emit.Function {
	sig := s390x.Layout([]string{"a", "n"}, []s390x.Type{s390x.Ptr, s390x.Int64},
		[]string{"ret"}, []s390x.Type{s390x.Float32})
	b := s390x.NewFunc(name, sig, 0)
	b.LoadArg("a", "R1").LoadArg("n", "R2").
		Raw("VLM (R1), V16, V19").
		Raw("ADD $64, R1").Raw("SUB $16, R2").
		Raw("loop:").Raw("CMP R2, $16").Raw("BLT fold").
		Raw("VLM (R1), V0, V3")
	for r := 0; r < 4; r++ {
		b.Raw("%s $1, V%d, V%d, V%d", vec, r, 16+r, 16+r)
	}
	b.Raw("ADD $64, R1").Raw("SUB $16, R2").Raw("BR loop").
		Raw("fold:").
		Raw("%s $1, V17, V16, V16", vec).
		Raw("%s $1, V19, V18, V18", vec).
		Raw("%s $1, V18, V16, V0", vec).
		Raw("VSLDB $8, V0, V0, V1"). // V1 = e2 e3 e0 e1: lanes 0,1 meet 2,3
		Raw("%s $1, V1, V0, V0", vec).
		Raw("VREPF $1, V0, V1").
		Raw("%s $1, F1, F0, F0", sca). // F0 = extremum of every lane
		Raw("tail:").Raw("CMP R2, $0").Raw("BEQ done").
		Raw("FMOVS (R1), F1").
		Raw("%s $1, F1, F0, F0", sca).
		Raw("ADD $4, R1").Raw("SUB $1, R2").Raw("BR tail").
		Raw("done:").
		StoreRet("F0", "ret").
		Ret()
	return b.Func()
}
