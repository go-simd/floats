package floats

import (
	"errors"
	"fmt"
)

// # AllPass3Plan: the memory of a decorrelator, proved once
//
// AllPass3 proves its sixteen operands on every call (lengths and the absence of overlap
// between everything written and everything read: about fifty comparisons), which costs
// more than the kernel itself when it runs once per slot on 23 to 50 rows. An
// AllPass3Plan moves that proof to the allocation of the memory, as a plane of an image
// codec does: NewAllPass3Plan checks the tables, the output, the input ring and the three
// link rings once, and keeps them. A call then names the memory by slot numbers only, so
// no slice that was not proved can reach the kernel; each call checks that the slot
// numbers are in range and that no link reads the slot it writes (eight integer
// comparisons), which is all that remains to be proved because the rings are proved
// disjoint from one another and from the output.
//
// The input ring is a flat slice of slots of inStride elements, of which a call reads the
// first n; link l has its own flat ring of slots of stride[l] elements. The plan owns the
// output slice it was given (read it with Out after a call).

// AllPass3Plan is a set of memory proved for AllPass3. It is not safe for concurrent use.
type AllPass3Plan struct {
	n        int
	t        AllPassRows
	out      []complex128
	in       []complex128
	inStride int
	inSlots  int
	w        [3][]complex128
	wStride  [3]int
	wSlots   [3]int
}

// NewAllPass3Plan proves the memory of n = len(t.Phi) rows and returns the plan. It
// returns an error, and no plan, if
//
//   - any table of t does not hold n elements, or n is 0,
//   - out holds fewer than n elements,
//   - a stride is below n, or a ring holds fewer than one slot (the input) or two slots
//     (a link ring: one slot to read and another to write),
//   - any of out and the three link rings (the written memory) overlaps any other of
//     them, the input ring or a table. The tables and the input ring may overlap one
//     another: they are only read.
//
// The slices are kept: the caller must not reslice them elsewhere to write the proved
// memory while a plan uses it through another name; the plan guarantees that its own
// calls never write outside out and the link rings.
func NewAllPass3Plan(t *AllPassRows, out, in []complex128, inStride int, w [3][]complex128, wStride [3]int) (*AllPass3Plan, error) {
	n := len(t.Phi)
	if n == 0 {
		return nil, errors.New("floats: AllPass3Plan: no rows")
	}
	tabs := [7]int{len(t.Phi), len(t.Q0), len(t.Q1), len(t.Q2), len(t.C0), len(t.C1), len(t.C2)}
	for i, l := range tabs {
		if l < n {
			return nil, fmt.Errorf("floats: AllPass3Plan: table %d holds %d elements, want %d", i, l, n)
		}
	}
	if len(out) < n {
		return nil, fmt.Errorf("floats: AllPass3Plan: out holds %d elements, want %d", len(out), n)
	}
	if inStride < n || len(in) < inStride {
		return nil, errors.New("floats: AllPass3Plan: the input ring holds no slot of inStride >= n elements")
	}
	p := &AllPass3Plan{n: n, t: *t, out: out[:n:n], in: in, inStride: inStride, inSlots: len(in) / inStride, w: w, wStride: wStride}
	for l := range w {
		if wStride[l] < n || len(w[l]) < 2*wStride[l] {
			return nil, fmt.Errorf("floats: AllPass3Plan: link ring %d holds fewer than two slots of >= n elements", l)
		}
		p.wSlots[l] = len(w[l]) / wStride[l]
	}
	written := [4]span{cspan(out, len(out)), cspan(w[0], len(w[0])), cspan(w[1], len(w[1])), cspan(w[2], len(w[2]))}
	read := [8]span{cspan(in, len(in)),
		cspan(t.Phi, n), cspan(t.Q0, n), cspan(t.Q1, n), cspan(t.Q2, n),
		fspan(t.C0, n), fspan(t.C1, n), fspan(t.C2, n)}
	if spansMeet(written[:], read[:]) {
		return nil, errors.New("floats: AllPass3Plan: written memory (out, link rings) overlaps another operand")
	}
	return p, nil
}

// spansMeet reports whether a written span meets another written one or a read one.
func spansMeet(written, read []span) bool {
	var hit int64
	for i := range written {
		for _, b := range written[i+1:] {
			hit |= written[i].meets(b)
		}
		for _, b := range read {
			hit |= written[i].meets(b)
		}
	}
	return hit < 0
}

// Rows is the number of rows n of the plan.
func (p *AllPass3Plan) Rows() int { return p.n }

// Out is the output of the last Run: n elements, owned by the plan's caller.
func (p *AllPass3Plan) Out() []complex128 { return p.out }

// Run computes the rows of one slot: the input is slot inSlot of the input ring, link l
// reads its state in slot oldSlot[l] of its ring and writes the new one to slot
// curSlot[l], and the result goes to Out. It panics, before reading or writing anything,
// if a slot number is out of range or a link reads the slot it writes.
func (p *AllPass3Plan) Run(inSlot int, oldSlot, curSlot [3]int) {
	if uint(inSlot) >= uint(p.inSlots) {
		panic("floats: AllPass3Plan.Run: input slot out of range")
	}
	var old, cur [3][]complex128
	for l := range old {
		o, c := oldSlot[l], curSlot[l]
		if uint(o) >= uint(p.wSlots[l]) || uint(c) >= uint(p.wSlots[l]) {
			panic("floats: AllPass3Plan.Run: link slot out of range")
		}
		if o == c {
			panic("floats: AllPass3Plan.Run: a link reads the slot it writes")
		}
		s := p.wStride[l]
		old[l] = p.w[l][o*s : o*s+p.n]
		cur[l] = p.w[l][c*s : c*s+p.n]
	}
	s := p.inStride
	allPass3(p.n, p.out, p.in[inSlot*s:inSlot*s+p.n], &p.t, &old, &cur)
}
