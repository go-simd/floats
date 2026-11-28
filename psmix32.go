package floats

import (
	"errors"
	"fmt"
)

// # Parametric Stereo mixing of a slot, in float32 split planes
//
// A PSMixPlan32 mixes the hybrid rows of one slot, a whole slot of short runs of rows per call.
// The rows are grouped in runs of consecutive rows that share their gains (the rows of one
// parameter band: about 3.5 rows per run in the 34-band layout, which is why one call per run
// does not pay). For each run [Start, End) with band b and source q, and every row k of it,
// with s the mono row, d the decorrelated row of source q and the four gains of the band
// (gLs, gLd, gRs, gRd = gains[4b .. 4b+3]) and its transient gain tg = tgs[b]:
//
//	l[k] = gLs*s[k] + (gLd*tg)*d[k]
//	r[k] = gRs*s[k] + (gRd*tg)*d[k]
//
// on the real plane and on the imaginary plane alike (the gains are real). The decorrelated
// signal of a run comes from one of three sources (the all-pass rows, the long delay, the short
// delay), each a pair of planes indexed by the same row numbers as s, l and r.
//
// Contract: an error bound, not bit identity with another code. The AVX2+FMA and NEON kernels
// fuse the second product of each row into the sum; the portable loop rounds every product and
// the sum. With u = 2^-24 and exact arithmetic on the same float32 inputs, every output
// component satisfies
//
//	|out - exact| <= 3u (|gXs*s| + |gXd*tg*d|)
//
// (the product gXd*tg, the product gXs*s and the product with d each carry one rounding, the
// sum a fourth, of which the fused form saves one). The tests assert it against a 200-bit
// math/big computation (TestPSMixPlan32Bound, FuzzPSMixPlan32). Inputs must be finite and keep
// every intermediate value within the float32 range; the bound is relative to the magnitudes,
// so it does not cover values that underflow to a denormal.
//
// Memory, and the one liberty the kernels take. A vector kernel handles a short run with one
// full-width load and store, not a masked one (a masked store is several times slower on Zen 4
// and there is no masked store on NEON), so it may compute, and write to l and r, up to 7 rows
// past the end of a run. The runs are processed in increasing row order and do not overlap, so
// every such row that belongs to a later run is overwritten by it with the right value; the
// others, the rows between two runs and the 7 rows after the last, hold UNSPECIFIED values
// after a call. To keep that inside memory the plan proves, Run requires every plane of s, l
// and r, and of every source that a run uses, to hold at least maxEnd+7 elements (maxEnd the
// end of the last run), and reads at most that many of them: nothing past maxEnd+7 is touched.
// The written planes l and r must not overlap each other or any plane that is read; Run does
// not check that (a check costs about what the kernel does), and a violation gives wrong
// values, never an access outside the lengths above.

// psMixSlack is the number of elements past the last run that a kernel may read and write.
const psMixSlack = 7

// PSMixRun is rows [Start, End) of a slot that share the gains of parameter band Band and the
// decorrelated signal of source Src (0, 1 or 2).
type PSMixRun struct{ Start, End, Band, Src int }

// psMixRun32 is a PSMixRun as the kernels read it: four int32.
type psMixRun32 struct{ start, end, band, src int32 }

// PSMixPlan32 is a list of runs proved once, for Run. It is immutable and may be shared by
// goroutines, as long as the planes they pass are not.
type PSMixPlan32 struct {
	runs   []psMixRun32
	nBands int
	maxEnd int
	srcEnd [3]int
}

// NewPSMixPlan32 checks the runs for nBands bands and returns the plan. It returns an error,
// and no plan, if there is no run, nBands is not positive, a run is empty or negative, has a
// band outside [0, nBands) or a source outside [0, 3), or starts before the end of the run that
// precedes it (the runs must be in increasing row order and must not overlap), or a row number
// is 2^30 or more.
func NewPSMixPlan32(runs []PSMixRun, nBands int) (*PSMixPlan32, error) {
	if len(runs) == 0 || nBands <= 0 {
		return nil, errors.New("floats: PSMixPlan32: no run or no band")
	}
	p := &PSMixPlan32{nBands: nBands, runs: make([]psMixRun32, len(runs))}
	prev := 0
	for i, r := range runs {
		switch {
		case r.Start < prev:
			return nil, fmt.Errorf("floats: PSMixPlan32: run %d starts at %d, before the end %d of the run before it", i, r.Start, prev)
		case r.End <= r.Start || r.End >= 1<<30:
			return nil, fmt.Errorf("floats: PSMixPlan32: run %d is [%d, %d)", i, r.Start, r.End)
		case r.Band < 0 || r.Band >= nBands:
			return nil, fmt.Errorf("floats: PSMixPlan32: run %d has band %d outside [0, %d)", i, r.Band, nBands)
		case r.Src < 0 || r.Src >= 3:
			return nil, fmt.Errorf("floats: PSMixPlan32: run %d has source %d outside [0, 3)", i, r.Src)
		}
		p.runs[i] = psMixRun32{int32(r.Start), int32(r.End), int32(r.Band), int32(r.Src)}
		prev = r.End
		p.srcEnd[r.Src] = r.End
	}
	p.maxEnd = prev
	return p, nil
}

// Runs is the number of runs of the plan.
func (p *PSMixPlan32) Runs() int { return len(p.runs) }

// Run mixes the runs of the plan: gains holds four values per band (gLs, gLd, gRs, gRd of band
// b at 4b..4b+3), tg one transient gain per band, s the mono rows, d the three decorrelated
// sources and l and r the outputs. It panics if gains or tg is shorter than the plan's bands
// need, or a plane shorter than maxEnd+7 elements (a source that no run uses may be empty).
func (p *PSMixPlan32) Run(gains, tg []float32, s Split32, d *[3]Split32, l, r Split32) {
	need := p.maxEnd + psMixSlack
	short := len(gains) < 4*p.nBands || len(tg) < p.nBands ||
		len(s.Re) < need || len(s.Im) < need || len(l.Re) < need || len(l.Im) < need || len(r.Re) < need || len(r.Im) < need
	for q := range d {
		if e := p.srcEnd[q]; e > 0 && (len(d[q].Re) < e+psMixSlack || len(d[q].Im) < e+psMixSlack) {
			short = true
		}
	}
	if short {
		panic("floats: PSMixPlan32.Run: an operand is shorter than the plan needs")
	}
	psMix32(p, gains, tg, s, d, l, r)
}

// psMix32Ref is the portable loop: exactly the rows of the runs, every product rounded to
// float32 on its own (the conversions keep a compiler from fusing them), and the reference of
// the kernels' bound.
func psMix32Ref(p *PSMixPlan32, gains, tg []float32, s Split32, d *[3]Split32, l, r Split32) {
	for _, run := range p.runs {
		g := gains[4*run.band : 4*run.band+4]
		gLs, gRs := g[0], g[2]
		gLd, gRd := float32(g[1]*tg[run.band]), float32(g[3]*tg[run.band])
		src := d[run.src]
		for k := run.start; k < run.end; k++ {
			l.Re[k] = float32(gLs*s.Re[k]) + float32(gLd*src.Re[k])
			r.Re[k] = float32(gRs*s.Re[k]) + float32(gRd*src.Re[k])
			l.Im[k] = float32(gLs*s.Im[k]) + float32(gLd*src.Im[k])
			r.Im[k] = float32(gRs*s.Im[k]) + float32(gRd*src.Im[k])
		}
	}
}
