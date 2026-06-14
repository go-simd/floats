package floats

import "math"

// Float32Dot returns the dot product a·b = Σ aᵢbᵢ over float32 slices,
// accumulated in float32. It panics if len(a) != len(b).
func Float32Dot(a, b []float32) float32 {
	if len(a) != len(b) {
		panic("floats: slice length mismatch")
	}
	return dot32(a, b)
}

// Float32Sum returns Σ aᵢ over a float32 slice, accumulated in float32. The sum
// of an empty slice is 0.
func Float32Sum(a []float32) float32 { return sum32(a) }

// Float32Min returns the smallest element of a. It panics if a is empty. If a
// contains a NaN the result is NaN.
func Float32Min(a []float32) float32 {
	if len(a) == 0 {
		panic("floats: Min of empty slice")
	}
	m := a[0]
	for _, v := range a[1:] {
		if v < m || isNaN32(v) {
			m = v
		}
	}
	return m
}

// Float32Max returns the largest element of a. It panics if a is empty. If a
// contains a NaN the result is NaN.
func Float32Max(a []float32) float32 {
	if len(a) == 0 {
		panic("floats: Max of empty slice")
	}
	m := a[0]
	for _, v := range a[1:] {
		if v > m || isNaN32(v) {
			m = v
		}
	}
	return m
}

// Float32Distance returns the Euclidean (L2) distance ‖a−b‖₂ over float32
// slices. The squared-difference sum is accumulated in float32 and the square
// root taken in float64 for accuracy. It panics if len(a) != len(b).
func Float32Distance(a, b []float32) float32 {
	if len(a) != len(b) {
		panic("floats: slice length mismatch")
	}
	return float32(math.Sqrt(float64(sumSqDiff32(a, b))))
}

// Float32CosineSimilarity returns a·b / (‖a‖₂‖b‖₂) over float32 slices — the
// relevance score for float32 embedding vectors. It panics if len(a) != len(b).
// If either vector has zero magnitude the result is NaN.
func Float32CosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) {
		panic("floats: slice length mismatch")
	}
	d := float64(dot32(a, b))
	na := float64(dot32(a, a))
	nb := float64(dot32(b, b))
	return float32(d / math.Sqrt(na*nb))
}

// isNaN32 reports whether f is a float32 NaN, without converting to float64
// (which would also be NaN but allocates intent confusion); a NaN is the only
// value not equal to itself.
func isNaN32(f float32) bool { return f != f }
