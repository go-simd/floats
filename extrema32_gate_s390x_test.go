//go:build s390x

package floats

func setExtrema32Gate(on bool) func() {
	saved := hasVXE
	hasVXE = on
	return func() { hasVXE = saved }
}
