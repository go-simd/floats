package floats

import (
	"errors"
	"fmt"
)

// # AllPass3Plan32: the memory of a split-plane decorrelator, proved once
//
// AllPass3Split32 proves its operands on every call, which costs more than the kernel when it
// runs once per slot on 23 to 50 rows (see AllPass3Plan, which has the same reason). An
// AllPass3Plan32 moves the proof to the allocation of the memory: NewAllPass3TickPlan32 checks
// the tables, the output, the input ring and the three link rings once, and RunTick(t) names
// the memory by a tick counter only, so no slice that was not proved reaches the kernel and
// a call checks nothing.
//
// Every ring is a pair of planes (Re, Im) of the same length: a flat slice of slots of
// stride elements, of which a call uses the first n. The plan owns the output it was given.

// AllPass3Plan32 is a set of memory proved for AllPass3Split32. It is not safe for concurrent use.
type AllPass3Plan32 struct {
	n        int
	t        AllPassRows32
	out      Split32
	in       Split32
	inStride int
	inSlots  int
	w        [3]Split32
	wStride  [3]int
	wSlots   [3]int
	inDelay  int
}

// NewAllPass3TickPlan32 proves the memory of n = len(t.PhiRe) rows for a decorrelator driven
// by a tick counter, and returns the plan. It returns an error, and no plan, if
//
//   - any table of t does not hold n elements, or n is 0,
//   - a plane of out holds fewer than n elements,
//   - the two planes of a ring differ in length, a stride is below n, or a ring holds fewer
//     than one slot (the input) or two slots (a link ring: one to read and another to write),
//   - inDelay is negative or not below the number of slots of the input ring,
//   - any of out and the three link rings (the written memory, every plane) overlaps any
//     other of them, the input ring or a table. The tables and the input ring may overlap one
//     another: they are only read.
//
// The slices are kept: the caller must not reslice them elsewhere to write the proved memory
// while a plan uses it through another name; the plan guarantees that its own calls never
// write outside out and the link rings.
func NewAllPass3TickPlan32(t *AllPassRows32, out, in Split32, inStride int, w [3]Split32, wStride [3]int, inDelay int) (*AllPass3Plan32, error) {
	n := len(t.PhiRe)
	if n == 0 {
		return nil, errors.New("floats: AllPass3Plan32: no rows")
	}
	tabs := [11][]float32{t.PhiRe, t.PhiIm, t.Q0Re, t.Q0Im, t.Q1Re, t.Q1Im, t.Q2Re, t.Q2Im, t.C0, t.C1, t.C2}
	for i, s := range tabs {
		if len(s) < n {
			return nil, fmt.Errorf("floats: AllPass3Plan32: table %d holds %d elements, want %d", i, len(s), n)
		}
	}
	if len(out.Re) < n || len(out.Im) < n {
		return nil, fmt.Errorf("floats: AllPass3Plan32: out holds %d and %d elements, want %d", len(out.Re), len(out.Im), n)
	}
	if len(in.Re) != len(in.Im) || inStride < n || len(in.Re) < inStride {
		return nil, errors.New("floats: AllPass3Plan32: the input ring holds no slot of inStride >= n elements in two planes of one length")
	}
	p := &AllPass3Plan32{n: n, t: *t, out: Split32{out.Re[:n:n], out.Im[:n:n]}, in: in, inStride: inStride, inSlots: len(in.Re) / inStride, w: w, wStride: wStride}
	for l := range w {
		if len(w[l].Re) != len(w[l].Im) || wStride[l] < n || len(w[l].Re) < 2*wStride[l] {
			return nil, fmt.Errorf("floats: AllPass3Plan32: link ring %d holds fewer than two slots of >= n elements in two planes of one length", l)
		}
		p.wSlots[l] = len(w[l].Re) / wStride[l]
	}
	if inDelay < 0 || inDelay >= p.inSlots {
		return nil, fmt.Errorf("floats: AllPass3Plan32: input delay %d outside [0, %d)", inDelay, p.inSlots)
	}
	p.inDelay = inDelay
	written := make([]span, 0, 8)
	written = append(written, fspan32(out.Re, len(out.Re)), fspan32(out.Im, len(out.Im)))
	for l := range w {
		written = append(written, fspan32(w[l].Re, len(w[l].Re)), fspan32(w[l].Im, len(w[l].Im)))
	}
	read := make([]span, 0, 13)
	read = append(read, fspan32(in.Re, len(in.Re)), fspan32(in.Im, len(in.Im)))
	for _, s := range tabs {
		read = append(read, fspan32(s, n))
	}
	if spansMeet(written, read) {
		return nil, errors.New("floats: AllPass3Plan32: written memory (out, link rings) overlaps another operand")
	}
	return p, nil
}

// Rows is the number of rows n of the plan.
func (p *AllPass3Plan32) Rows() int { return p.n }

// Out is the output of the last RunTick: n elements per plane, owned by the plan's caller.
func (p *AllPass3Plan32) Out() Split32 { return p.out }

// RunTick computes the rows of tick t: link l, whose ring holds m slots, reads slot (t+1) mod m
// and writes slot t mod m; the input is slot (t - inDelay) mod s of the input ring of s slots;
// the result goes to Out. The slots follow from t and from sizes proved at construction (every
// ring of a link holds at least two slots, so no link reads the slot it writes, and every slot
// is in range), so a call checks nothing and cannot panic for any t, negative or not; a
// negative t reduces as uint(t) does, memory-safe but not continuous.
func (p *AllPass3Plan32) RunTick(t int) {
	var old, cur [3]Split32
	u := uint(t)
	n := uint(p.n)
	for l := range old {
		m := uint(p.wSlots[l])
		c := u % m
		o := c + 1
		if o == m {
			o = 0
		}
		s := uint(p.wStride[l])
		w := &p.w[l]
		old[l] = Split32{w.Re[o*s : o*s+n], w.Im[o*s : o*s+n]}
		cur[l] = Split32{w.Re[c*s : c*s+n], w.Im[c*s : c*s+n]}
	}
	k := uint(p.inSlots)
	i := u%k + k - uint(p.inDelay)
	if i >= k {
		i -= k
	}
	s := uint(p.inStride)
	allPass3Split32(p.n, p.out, Split32{p.in.Re[i*s : i*s+n], p.in.Im[i*s : i*s+n]}, &p.t, &old, &cur)
}
