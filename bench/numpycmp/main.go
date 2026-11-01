// Command numpycmp measures go-simd/floats Mul, Axpby3, Clamp and MaxAbs against
// the plain Go loop and against NumPy (np.multiply, an a*x+b*y+c*z expression,
// np.clip, np.abs(x).max()) on one
// core, in interleaved rounds: every round runs the Go benchmark binary and then
// NumPy's own `python3 -m timeit` for each (op, n), and the median over rounds is
// reported. It refuses to run when the 1-minute load is at or above 8, because
// ratios measured on a loaded machine mean nothing.
//
//	go test -c -o /path/floats.test ../..      (from this directory)
//	go run . --binary /path/floats.test
package main

import (
	"fmt"
	"io"
	"os"
	"os/exec"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/spf13/cobra"
)

var osExit = os.Exit

func main() {
	if err := RootCmd().Execute(); err != nil {
		osExit(1)
	}
}

// op couples a Go benchmark (it has /simd and /naive sub-benchmarks) with the
// NumPy statement that does the same job.
type op struct {
	name, bench, stmt string
}

var ops = []op{
	{"Mul", "BenchmarkMul", "np.multiply(a, b, out=d)"},
	{"Axpby3", "BenchmarkAxpby3", "d[:] = 0.5*a - 1.5*b + 2.5*c"},
	{"Clamp", "BenchmarkClamp", "np.clip(a, lo, hi, out=d)"},
	{"MaxAbs", "BenchmarkMaxAbs", "np.abs(a).max()"},
	{"SumPairwise", "BenchmarkSumPairwise", "np.sum(a)"},
}

var sizes = []int{1000, 100000, 1000000}

func numpySetup(n int) string {
	return fmt.Sprintf("import numpy as np; n=%d; r=np.random.default_rng(7); "+
		"a=r.standard_normal(n); b=r.standard_normal(n); c=r.standard_normal(n); d=np.empty(n); "+
		"lo=-np.abs(r.standard_normal(n)); hi=np.abs(r.standard_normal(n))", n)
}

var timeitRe = regexp.MustCompile(`best of \d+: ([0-9.]+) (nsec|usec|msec|sec) per loop`)

// parseTimeit returns the best time per loop, in ns, from `python -m timeit`.
func parseTimeit(out string) (float64, error) {
	m := timeitRe.FindStringSubmatch(out)
	if m == nil {
		return 0, fmt.Errorf("no timeit result in %q", out)
	}
	v, err := strconv.ParseFloat(m[1], 64)
	if err != nil {
		return 0, err
	}
	return v * map[string]float64{"nsec": 1, "usec": 1e3, "msec": 1e6, "sec": 1e9}[m[2]], nil
}

var goBenchRe = regexp.MustCompile(`(?m)^(Benchmark\S+?)(?:-\d+)?\s+\d+\s+([0-9.]+) ns/op`)

// parseGoBench returns ns/op keyed by benchmark name (without the -N suffix).
func parseGoBench(out string) map[string]float64 {
	res := map[string]float64{}
	for _, m := range goBenchRe.FindAllStringSubmatch(out, -1) {
		if v, err := strconv.ParseFloat(m[2], 64); err == nil {
			res[m[1]] = v
		}
	}
	return res
}

var loadRe = regexp.MustCompile(`([0-9]+[.,][0-9]+)`)

// parseLoad returns the first number of `sysctl -n vm.loadavg` ("{ 1.2 3.4 5.6 }").
func parseLoad(out string) (float64, error) {
	m := loadRe.FindString(out)
	if m == "" {
		return 0, fmt.Errorf("no load average in %q", out)
	}
	return strconv.ParseFloat(strings.Replace(m, ",", ".", 1), 64)
}

func median(v []float64) float64 {
	s := append([]float64(nil), v...)
	sort.Float64s(s)
	if len(s) == 0 {
		return 0
	}
	if len(s)%2 == 1 {
		return s[len(s)/2]
	}
	return (s[len(s)/2-1] + s[len(s)/2]) / 2
}

type config struct {
	binary, python string
	rounds         int
	maxLoad        float64
	out            io.Writer
	run            func(env []string, name string, args ...string) (string, error)
}

func execRun(env []string, name string, args ...string) (string, error) {
	c := exec.Command(name, args...)
	c.Env = append(os.Environ(), env...)
	c.Stdin = nil
	b, err := c.CombinedOutput()
	return string(b), err
}

func measure(cfg config) error {
	out, err := cfg.run(nil, "sysctl", "-n", "vm.loadavg")
	if err != nil {
		out, err = cfg.run(nil, "cat", "/proc/loadavg")
	}
	if err != nil {
		return fmt.Errorf("cannot read the load average: %w", err)
	}
	load, err := parseLoad(out)
	if err != nil {
		return err
	}
	if load >= cfg.maxLoad {
		return fmt.Errorf("1-minute load %.1f >= %.1f: ratios would be meaningless, defer", load, cfg.maxLoad)
	}
	oneThread := []string{"OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", "MKL_NUM_THREADS=1", "VECLIB_MAXIMUM_THREADS=1"}
	fmt.Fprintf(cfg.out, "load %.1f, %d interleaved rounds, median ns/op, one core\n\n", load, cfg.rounds)
	fmt.Fprintln(cfg.out, "| op | n | simd | Go loop | NumPy | simd vs Go loop | simd vs NumPy |")
	fmt.Fprintln(cfg.out, "|---|---|---|---|---|---|---|")
	for _, o := range ops {
		for _, n := range sizes {
			var simd, naive, numpy []float64
			for r := 0; r < cfg.rounds; r++ {
				res, err := cfg.run(nil, cfg.binary, "-test.run=^$", "-test.cpu=1", "-test.benchtime=300ms",
					fmt.Sprintf("-test.bench=^%s$/^%d$", o.bench, n))
				if err != nil {
					return fmt.Errorf("%s n=%d: %v\n%s", o.name, n, err, res)
				}
				m := parseGoBench(res)
				s, ok1 := m[fmt.Sprintf("%s/%d/simd", o.bench, n)]
				g, ok2 := m[fmt.Sprintf("%s/%d/naive", o.bench, n)]
				if !ok1 || !ok2 {
					return fmt.Errorf("%s n=%d: benchmark lines missing in %q", o.name, n, res)
				}
				simd, naive = append(simd, s), append(naive, g)
				res, err = cfg.run(oneThread, cfg.python, "-m", "timeit", "-r", "5", "-s", numpySetup(n), o.stmt)
				if err != nil {
					return fmt.Errorf("numpy %s n=%d: %v\n%s", o.name, n, err, res)
				}
				t, err := parseTimeit(res)
				if err != nil {
					return err
				}
				numpy = append(numpy, t)
			}
			s, g, p := median(simd), median(naive), median(numpy)
			fmt.Fprintf(cfg.out, "| %s | %d | %.0f | %.0f | %.0f | %.2fx | %.2fx |\n", o.name, n, s, g, p, g/s, p/s)
		}
	}
	return nil
}

// RootCmd is the command, driven by tests through SetArgs/SetOut/SetErr.
func RootCmd() *cobra.Command {
	cfg := config{run: execRun}
	cmd := &cobra.Command{
		Use:          "numpycmp",
		Short:        "benchmark go-simd/floats against the Go loop and NumPy, one core, interleaved",
		SilenceUsage: true,
		RunE: func(cmd *cobra.Command, _ []string) error {
			cfg.out = cmd.OutOrStdout()
			if cfg.binary == "" {
				return fmt.Errorf("--binary is required (go test -c -o floats.test of the root package)")
			}
			return measure(cfg)
		},
	}
	cmd.Flags().StringVar(&cfg.binary, "binary", "", "compiled floats test binary (go test -c)")
	cmd.Flags().StringVar(&cfg.python, "python", "python3", "python with numpy")
	cmd.Flags().IntVar(&cfg.rounds, "rounds", 5, "interleaved rounds")
	cmd.Flags().Float64Var(&cfg.maxLoad, "max-load", 8, "refuse to run at or above this 1-minute load")
	return cmd
}
