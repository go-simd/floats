//go:build !amd64 && !arm64

package floats

// No kernel on this architecture yet: the portable loops, which the compiler
// already vectorises where it can. A kernel is added when it is measured faster
// on real silicon (the dispatch rule of this library).

func add(dst, a, b []float64)                      { addRef(dst, a, b) }
func sub(dst, a, b []float64)                      { subRef(dst, a, b) }
func div(dst, a, b []float64)                      { divRef(dst, a, b) }
func sqrt(dst, src []float64)                      { sqrtRef(dst, src) }
func clampScalar(dst, v []float64, lo, hi float64) { clampScalarRef(dst, v, lo, hi) }
