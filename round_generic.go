//go:build !amd64 && !arm64 && !s390x && !ppc64le && !loong64 && !riscv64

package floats

// No kernel on this architecture: the portable loops (math.Floor and the other
// rounding functions are instructions on some of the targets that land here, and
// the compiler keeps a loop of them tight). amd64, arm64, s390x, ppc64le, loong64
// and riscv64 have kernels (round_<arch>.go).

func vabs(dst, src []float64)    { absRef(dst, src) }
func vneg(dst, src []float64)    { negRef(dst, src) }
func vsquare(dst, src []float64) { squareRef(dst, src) }
func vfloor(dst, src []float64)  { floorRef(dst, src) }
func vceil(dst, src []float64)   { ceilRef(dst, src) }
func vtrunc(dst, src []float64)  { truncRef(dst, src) }
func vrint(dst, src []float64)   { rintRef(dst, src) }
