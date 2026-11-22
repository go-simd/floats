//go:build ppc64le || s390x

package floats

// Short float64 reductions for the arches whose vector kernels start at a
// threshold (vsxMin64, vxMin64). Four independent accumulators shorten the
// FMA/add dependency chain below the plain loop's, so the span just under the
// kernel threshold is no slower than the kernel's setup would make it.

func dotSmall(a, b []float64) float64 {
	b = b[:len(a)]
	var s0, s1, s2, s3 float64
	i := 0
	for ; i+4 <= len(a); i += 4 {
		s0 += a[i] * b[i]
		s1 += a[i+1] * b[i+1]
		s2 += a[i+2] * b[i+2]
		s3 += a[i+3] * b[i+3]
	}
	for ; i < len(a); i++ {
		s0 += a[i] * b[i]
	}
	return (s0 + s1) + (s2 + s3)
}

func sumSmall(a []float64) float64 {
	var s0, s1, s2, s3 float64
	i := 0
	for ; i+4 <= len(a); i += 4 {
		s0 += a[i]
		s1 += a[i+1]
		s2 += a[i+2]
		s3 += a[i+3]
	}
	for ; i < len(a); i++ {
		s0 += a[i]
	}
	return (s0 + s1) + (s2 + s3)
}

func sumSqDiffSmall(a, b []float64) float64 {
	b = b[:len(a)]
	var s0, s1, s2, s3 float64
	i := 0
	for ; i+4 <= len(a); i += 4 {
		d0 := a[i] - b[i]
		d1 := a[i+1] - b[i+1]
		d2 := a[i+2] - b[i+2]
		d3 := a[i+3] - b[i+3]
		s0 += d0 * d0
		s1 += d1 * d1
		s2 += d2 * d2
		s3 += d3 * d3
	}
	for ; i < len(a); i++ {
		d := a[i] - b[i]
		s0 += d * d
	}
	return (s0 + s1) + (s2 + s3)
}
