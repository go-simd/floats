package floats

// This file holds the lane-blocked scalar reference for every reduction. It is
// always compiled (no build tag): it is the oracle the differential tests and
// fuzzers compare the SIMD kernels against, and it is also the kernel the
// generic (non-SIMD) build calls directly.
//
// Each reference keeps laneCount independent accumulators and assigns element i
// to lane i%laneCount, exactly mirroring how the SIMD kernels stripe a vector
// register across the input; the tail (len not a multiple of laneCount) is
// folded into the low lanes in order. The lanes are then summed low-to-high.
// Because the kernels reduce in this same structure, kernel and reference agree
// to within a few ULP on every architecture, making the tests deterministic.

// dotLanes returns Σ aᵢbᵢ in float64, reduced in the laneCount-lane order.
func dotLanes(a, b []float64) float64 {
	var acc [laneCount]float64
	n := len(a)
	i := 0
	for ; i+laneCount <= n; i += laneCount {
		for j := 0; j < laneCount; j++ {
			acc[j] += a[i+j] * b[i+j]
		}
	}
	for j := 0; i+j < n; j++ {
		acc[j] += a[i+j] * b[i+j]
	}
	return foldLanes(acc)
}

// sumLanes returns Σ aᵢ in float64, reduced in the laneCount-lane order.
func sumLanes(a []float64) float64 {
	var acc [laneCount]float64
	n := len(a)
	i := 0
	for ; i+laneCount <= n; i += laneCount {
		for j := 0; j < laneCount; j++ {
			acc[j] += a[i+j]
		}
	}
	for j := 0; i+j < n; j++ {
		acc[j] += a[i+j]
	}
	return foldLanes(acc)
}

// sumSqDiffLanes returns Σ(aᵢ−bᵢ)² in float64, reduced in the laneCount-lane
// order.
func sumSqDiffLanes(a, b []float64) float64 {
	var acc [laneCount]float64
	n := len(a)
	i := 0
	for ; i+laneCount <= n; i += laneCount {
		for j := 0; j < laneCount; j++ {
			d := a[i+j] - b[i+j]
			acc[j] += d * d
		}
	}
	for j := 0; i+j < n; j++ {
		d := a[i+j] - b[i+j]
		acc[j] += d * d
	}
	return foldLanes(acc)
}

// foldLanes sums the lane accumulators low-to-high.
func foldLanes(acc [laneCount]float64) float64 {
	s := acc[0]
	for j := 1; j < laneCount; j++ {
		s += acc[j]
	}
	return s
}

// ---- float32 references (accumulated in float32) ----

func dotLanes32(a, b []float32) float32 {
	var acc [laneCount]float32
	n := len(a)
	i := 0
	for ; i+laneCount <= n; i += laneCount {
		for j := 0; j < laneCount; j++ {
			acc[j] += a[i+j] * b[i+j]
		}
	}
	for j := 0; i+j < n; j++ {
		acc[j] += a[i+j] * b[i+j]
	}
	return foldLanes32(acc)
}

func sumLanes32(a []float32) float32 {
	var acc [laneCount]float32
	n := len(a)
	i := 0
	for ; i+laneCount <= n; i += laneCount {
		for j := 0; j < laneCount; j++ {
			acc[j] += a[i+j]
		}
	}
	for j := 0; i+j < n; j++ {
		acc[j] += a[i+j]
	}
	return foldLanes32(acc)
}

func sumSqDiffLanes32(a, b []float32) float32 {
	var acc [laneCount]float32
	n := len(a)
	i := 0
	for ; i+laneCount <= n; i += laneCount {
		for j := 0; j < laneCount; j++ {
			d := a[i+j] - b[i+j]
			acc[j] += d * d
		}
	}
	for j := 0; i+j < n; j++ {
		d := a[i+j] - b[i+j]
		acc[j] += d * d
	}
	return foldLanes32(acc)
}

func foldLanes32(acc [laneCount]float32) float32 {
	s := acc[0]
	for j := 1; j < laneCount; j++ {
		s += acc[j]
	}
	return s
}
