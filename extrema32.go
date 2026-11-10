package floats

// maxRef32 is the oracle for the float32 maximum, and the implementation where
// there is no kernel: four accumulators, so four independent chains, folded with
// the same NaN-propagating max (Go's builtin: NaN if either operand is NaN, -0
// below +0, as numpy.max). Max is associative under that rule, so the grouping
// does not change the result. a is non-empty.
func maxRef32(a []float32) float32 {
	if len(a) < 4 {
		m := a[0]
		for _, v := range a[1:] {
			m = max(m, v)
		}
		return m
	}
	m0, m1, m2, m3 := a[0], a[1], a[2], a[3]
	s := a[4:]
	for len(s) >= 4 {
		m0, m1, m2, m3 = max(m0, s[0]), max(m1, s[1]), max(m2, s[2]), max(m3, s[3])
		s = s[4:]
	}
	m := max(max(m0, m1), max(m2, m3))
	for _, v := range s {
		m = max(m, v)
	}
	return m
}

// minRef32 is the dual of maxRef32.
func minRef32(a []float32) float32 {
	if len(a) < 4 {
		m := a[0]
		for _, v := range a[1:] {
			m = min(m, v)
		}
		return m
	}
	m0, m1, m2, m3 := a[0], a[1], a[2], a[3]
	s := a[4:]
	for len(s) >= 4 {
		m0, m1, m2, m3 = min(m0, s[0]), min(m1, s[1]), min(m2, s[2]), min(m3, s[3])
		s = s[4:]
	}
	m := min(min(m0, m1), min(m2, m3))
	for _, v := range s {
		m = min(m, v)
	}
	return m
}
