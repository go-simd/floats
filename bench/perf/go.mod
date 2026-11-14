module github.com/go-simd/floats/bench/perf

go 1.27.3

require (
	github.com/go-scikit/refbench v0.4.0
	github.com/go-simd/floats v0.21.0
)

require (
	github.com/go-fleettools/fleettools v0.81.0 // indirect
	golang.org/x/sys v0.49.0 // indirect
)

replace github.com/go-simd/floats => ../..
