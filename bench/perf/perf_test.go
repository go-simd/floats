package perf

// Every public function of go-simd/floats against NumPy, one core, interleaved
// rounds, measured by refbench (performance campaign wave W1; it replaces
// bench/numpycmp).
//
//	cd bench/perf
//	go test -run TestPerf .                          CI mode: allocations, table, denominator (no Python, no timing)
//	go test -run TestPerf -cpu 1 -perf.run .         measure (refuses at a 1-minute load of 8 or more)
//	go test -run TestPerf -perf.record .             allocation counts only (a deferred report)
//	go test -run TestDataHash -floats.hash .         check that Go and NumPy hold the same bytes (needs uv)
//
// Both sides build their inputs by one formula, lo + (hi-lo)*frac(p*(i+1)), with
// a different p per array (the golden-ratio family the other go-scikit
// harnesses use). The NumPy side is the idiomatic statement a user would write
// (a temporary where NumPy has no fused form), under refs.txt's pinned NumPy.

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"flag"
	"fmt"
	"math"
	"os/exec"
	"sync"
	"testing"
	"time"

	"github.com/go-scikit/refbench"
	"github.com/go-scikit/refbench/rbtest"
	"github.com/go-simd/floats"
)

var hashFlag = flag.Bool("floats.hash", false, "TestDataHash: compare the Go inputs with NumPy's (needs uv and network)")

var sizes = []int{1_000, 100_000, 1_000_000}

// the generator parameters of the six arrays a, b, c, d (output, zeros), lo, hi.
var params = [6]struct{ p, lo, hi float64 }{
	{0.6180339887498949, -1, 1},
	{0.7548776662466927, -1, 1},
	{0.5698402909980532, -1, 1},
	{0, 0, 0}, // d: zeros
	{0.5698402909980532, -1, -0.25},
	{0.7548776662466927, 0.25, 1},
}

// gen is lo + (hi-lo)*frac(p*(i+1)); the explicit conversions forbid fused
// multiply-adds, so NumPy's temporaries and these values agree bit for bit.
func gen(n int, p, lo, hi float64) []float64 {
	out := make([]float64, n)
	if p == 0 {
		return out
	}
	for i := range out {
		t := float64(float64(i+1) * p)
		f := t - math.Floor(t)
		out[i] = lo + float64((hi-lo)*f)
	}
	return out
}

// pySetup builds the same arrays under NumPy (float64, or float32 by cast).
func pySetup(f32 bool) string {
	g := "lambda p,lo,hi: lo + (hi-lo)*(((np.arange(n)+1.0)*p) % 1.0)"
	s := fmt.Sprintf("(lambda g: (g(%v,-1,1), g(%v,-1,1), g(%v,-1,1), np.zeros(n), g(%v,-1,-0.25), g(%v,0.25,1)))(%s)",
		params[0].p, params[1].p, params[2].p, params[4].p, params[5].p, g)
	if f32 {
		return "tuple(x.astype(np.float32) for x in " + s + ")"
	}
	return s
}

type data struct {
	a, b, c, d, lo, hi []float64
	a32, b32           []float32
}

func newData(n int) *data {
	x := &data{}
	for i, dst := range []*[]float64{&x.a, &x.b, &x.c, &x.d, &x.lo, &x.hi} {
		*dst = gen(n, params[i].p, params[i].lo, params[i].hi)
	}
	x.a32, x.b32 = make([]float32, n), make([]float32, n)
	for i := range x.a {
		x.a32[i], x.b32[i] = float32(x.a[i]), float32(x.b[i])
	}
	return x
}

// bytes64 / bytes32 hash a slice as NumPy's tobytes does (little-endian).
func hashArrays(f32 bool, x *data) string {
	h := sha256.New()
	put64 := func(v []float64) {
		for _, f := range v {
			binary.Write(h, binary.LittleEndian, math.Float64bits(f))
		}
	}
	put32 := func(v []float32) {
		for _, f := range v {
			binary.Write(h, binary.LittleEndian, math.Float32bits(f))
		}
	}
	if f32 {
		c32, d32, lo32, hi32 := make([]float32, len(x.a)), make([]float32, len(x.a)), make([]float32, len(x.a)), make([]float32, len(x.a))
		for i := range x.a {
			c32[i], lo32[i], hi32[i] = float32(x.c[i]), float32(x.lo[i]), float32(x.hi[i])
		}
		for _, v := range [][]float32{x.a32, x.b32, c32, d32, lo32, hi32} {
			put32(v)
		}
	} else {
		for _, v := range [][]float64{x.a, x.b, x.c, x.d, x.lo, x.hi} {
			put64(v)
		}
	}
	return hex.EncodeToString(h.Sum(nil))
}

var sink float64

type op struct {
	fn     string // the public function
	parity string
	f32    bool
	py     string // lambda a,b,c,d,lo,hi: the NumPy statement
	body   func(x *data) func()
}

const args = "lambda a,b,c,d,lo,hi: "

var ops = []op{
	{"Dot", refbench.ParityTolerance, false, args + "np.dot(a, b)", func(x *data) func() { return func() { sink = floats.Dot(x.a, x.b) } }},
	{"Sum", refbench.ParityTolerance, false, args + "np.sum(a)", func(x *data) func() { return func() { sink = floats.Sum(x.a) } }},
	{"SumPairwise", refbench.ParityExact, false, args + "np.sum(a)", func(x *data) func() { return func() { sink = floats.SumPairwise(x.a) } }},
	{"Min", refbench.ParityExact, false, args + "np.min(a)", func(x *data) func() { return func() { sink = floats.Min(x.a) } }},
	{"Max", refbench.ParityExact, false, args + "np.max(a)", func(x *data) func() { return func() { sink = floats.Max(x.a) } }},
	{"Distance", refbench.ParityTolerance, false, args + "np.linalg.norm(a - b)", func(x *data) func() { return func() { sink = floats.Distance(x.a, x.b) } }},
	{"CosineSimilarity", refbench.ParityTolerance, false, args + "np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))", func(x *data) func() { return func() { sink = floats.CosineSimilarity(x.a, x.b) } }},
	{"Float32Dot", refbench.ParityTolerance, true, args + "np.dot(a, b)", func(x *data) func() { return func() { sink = float64(floats.Float32Dot(x.a32, x.b32)) } }},
	{"Float32Sum", refbench.ParityTolerance, true, args + "np.sum(a)", func(x *data) func() { return func() { sink = float64(floats.Float32Sum(x.a32)) } }},
	{"Float32Min", refbench.ParityExact, true, args + "np.min(a)", func(x *data) func() { return func() { sink = float64(floats.Float32Min(x.a32)) } }},
	{"Float32Max", refbench.ParityExact, true, args + "np.max(a)", func(x *data) func() { return func() { sink = float64(floats.Float32Max(x.a32)) } }},
	{"Float32Distance", refbench.ParityTolerance, true, args + "np.linalg.norm(a - b)", func(x *data) func() { return func() { sink = float64(floats.Float32Distance(x.a32, x.b32)) } }},
	{"Float32CosineSimilarity", refbench.ParityTolerance, true, args + "np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))", func(x *data) func() { return func() { sink = float64(floats.Float32CosineSimilarity(x.a32, x.b32)) } }},
	{"Axpy", refbench.ParityTolerance, false, args + "np.add(d, 0.5*a, out=d)", func(x *data) func() { return func() { floats.Axpy(0.5, x.a, x.d) } }},
	{"Xpby", refbench.ParityTolerance, false, args + "np.add(a, 0.5*d, out=d)", func(x *data) func() { return func() { floats.Xpby(x.a, 0.5, x.d) } }},
	{"AxpyDot", refbench.ParityTolerance, false, args + "(np.add(d, 0.5*a, out=d), np.dot(d, b))", func(x *data) func() { return func() { sink = floats.AxpyDot(0.5, x.a, x.d, x.b) } }},
	{"Scale", refbench.ParityExact, false, args + "np.multiply(d, 1.0, out=d)", func(x *data) func() { return func() { floats.Scale(1.0, x.d) } }},
	{"Mul", refbench.ParityExact, false, args + "np.multiply(a, b, out=d)", func(x *data) func() { return func() { floats.Mul(x.d, x.a, x.b) } }},
	{"Add", refbench.ParityExact, false, args + "np.add(a, b, out=d)", func(x *data) func() { return func() { floats.Add(x.d, x.a, x.b) } }},
	{"Sub", refbench.ParityExact, false, args + "np.subtract(a, b, out=d)", func(x *data) func() { return func() { floats.Sub(x.d, x.a, x.b) } }},
	{"Div", refbench.ParityExact, false, args + "np.divide(a, b, out=d)", func(x *data) func() { return func() { floats.Div(x.d, x.a, x.b) } }},
	{"Sqrt", refbench.ParityExact, false, args + "np.sqrt(hi, out=d)", func(x *data) func() { return func() { floats.Sqrt(x.d, x.hi) } }},
	{"ClampScalar", refbench.ParityExact, false, args + "np.clip(a, -0.5, 0.5, out=d)", func(x *data) func() { return func() { floats.ClampScalar(x.d, x.a, -0.5, 0.5) } }},
	{"Axpby3", refbench.ParityTolerance, false, args + "np.add(0.5*a - 1.5*b, 2.5*c, out=d)", func(x *data) func() { return func() { floats.Axpby3(x.d, 0.5, x.a, -1.5, x.b, 2.5, x.c) } }},
	{"Clamp", refbench.ParityExact, false, args + "np.clip(a, lo, hi, out=d)", func(x *data) func() { return func() { floats.Clamp(x.d, x.a, x.lo, x.hi) } }},
	{"MaxAbs", refbench.ParityExact, false, args + "np.abs(a).max()", func(x *data) func() { return func() { sink = floats.MaxAbs(x.a) } }},
	{"AxpyNoFMA", refbench.ParityExact, false, args + "np.add(d, 0.5*a, out=d)", func(x *data) func() { return func() { floats.AxpyNoFMA(0.5, x.a, x.d) } }},
	{"XpbyNoFMA", refbench.ParityExact, false, args + "np.add(a, 0.5*d, out=d)", func(x *data) func() { return func() { floats.XpbyNoFMA(x.a, 0.5, x.d) } }},
	{"Axpby3NoFMA", refbench.ParityExact, false, args + "np.add(0.5*a + -1.5*b, 2.5*c, out=d)", func(x *data) func() { return func() { floats.Axpby3NoFMA(x.d, 0.5, x.a, -1.5, x.b, 2.5, x.c) } }},
	{"AxpyDiffNoFMA", refbench.ParityExact, false, args + "np.add(a, 0.5*(b - c), out=d)", func(x *data) func() { return func() { floats.AxpyDiffNoFMA(x.d, x.a, 0.5, x.b, x.c) } }},
}

// worker is the one long-lived NumPy process, started at the first timed call.
var (
	workerOnce sync.Once
	worker     *refbench.Worker
	workerErr  error
)

func pyWorker() (*refbench.Worker, error) {
	workerOnce.Do(func() {
		uv, err := exec.LookPath("uv")
		if err != nil {
			workerErr = err
			return
		}
		worker, workerErr = refbench.StartPython(context.Background(), uv, "refs.txt")
	})
	return worker, workerErr
}

func cases() []refbench.Case {
	var out []refbench.Case
	for _, n := range sizes {
		x := newData(n)
		for _, o := range ops {
			body := o.body(x)
			setup, call := pySetup(o.f32), o.py
			n := n
			out = append(out, refbench.Case{
				Name:      fmt.Sprintf("%s/%d", o.fn, n),
				Func:      o.fn,
				Reference: "numpy",
				Parity:    o.parity,
				Go: func(iters int) {
					for range iters {
						body()
					}
				},
				Ref: func(iters int) (time.Duration, error) {
					w, err := pyWorker()
					if err != nil {
						return 0, err
					}
					return w.Timer(setup, call, n)(iters)
				},
			})
		}
	}
	return out
}

func TestPerf(t *testing.T) {
	rbtest.Perf(t, refbench.CIConfig{Root: "../..", Cases: cases()}, refbench.Options{
		Repo:        "go-simd/floats",
		RefVersions: map[string]string{"numpy": "2.5.4 (refs.txt)"},
	})
}

// TestDataHash checks rule 2 of the campaign: the arrays Go times are the bytes
// NumPy times.
func TestDataHash(t *testing.T) {
	if !*hashFlag {
		t.Skip("run with -floats.hash (needs uv and network)")
	}
	w, err := pyWorker()
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	for _, n := range sizes {
		x := newData(n)
		for _, f32 := range []bool{false, true} {
			want, err := w.DataHash(pySetup(f32), n)
			if err != nil {
				t.Fatal(err)
			}
			if got := hashArrays(f32, x); got != want {
				t.Errorf("n=%d float32=%v: Go %s, NumPy %s", n, f32, got, want)
			}
		}
	}
}
