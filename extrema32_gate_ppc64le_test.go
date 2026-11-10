//go:build ppc64le

package floats

func setExtrema32Gate(on bool) func() {
	saved := hasVSX
	hasVSX = on
	return func() { hasVSX = saved }
}
