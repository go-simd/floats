//go:build riscv64

package floats

func setExtrema32Gate(on bool) func() {
	saved := hasRVV
	hasRVV = on
	return func() { hasRVV = saved }
}
