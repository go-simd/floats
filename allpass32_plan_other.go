//go:build !amd64

package floats

// kernel runs the kernel of a tick through slices: only amd64 has a pointer-level path
// (allpass32_amd64.go).
func (p *AllPass3Plan32) kernel(inOff uint, old, cur *[3]uint) { p.kernelSlices(inOff, old, cur) }
