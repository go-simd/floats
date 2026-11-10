//go:build loong64

package floats

func setExtrema32Gate(on bool) func() {
	saved := hasLASX
	hasLASX = on
	return func() { hasLASX = saved }
}
