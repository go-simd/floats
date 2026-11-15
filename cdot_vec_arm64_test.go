//go:build arm64

package floats

// cdotVector reports whether CDotC runs a vector kernel on this machine.
func cdotVector() bool { return true }
