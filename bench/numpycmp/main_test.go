package main

import (
	"bytes"
	"errors"
	"strings"
	"testing"
)

func TestParseTimeit(t *testing.T) {
	for in, want := range map[string]float64{
		"5000 loops, best of 5: 41.2 usec per loop\n": 41200,
		"20 loops, best of 5: 1.5 msec per loop":      1.5e6,
		"1000000 loops, best of 5: 300 nsec per loop": 300,
		"1 loop, best of 5: 2 sec per loop":           2e9,
	} {
		got, err := parseTimeit(in)
		if err != nil || got != want {
			t.Errorf("parseTimeit(%q) = %v, %v; want %v", in, got, err, want)
		}
	}
	if _, err := parseTimeit("garbage"); err == nil {
		t.Error("garbage parsed")
	}
}

func TestParseGoBench(t *testing.T) {
	out := "goos: darwin\nBenchmarkMul/1000/simd  \t 1000\t 123.5 ns/op\t 1.0 MB/s\n" +
		"BenchmarkMul/1000/naive-8 \t 100\t 456 ns/op\nPASS\n"
	m := parseGoBench(out)
	if m["BenchmarkMul/1000/simd"] != 123.5 || m["BenchmarkMul/1000/naive"] != 456 || len(m) != 2 {
		t.Errorf("parseGoBench = %v", m)
	}
}

func TestParseLoadAndMedian(t *testing.T) {
	if v, err := parseLoad("{ 7.25 3.1 2.0 }"); err != nil || v != 7.25 {
		t.Errorf("parseLoad = %v %v", v, err)
	}
	if v, err := parseLoad("0.52 0.58 0.59 1/467 12345"); err != nil || v != 0.52 {
		t.Errorf("parseLoad proc = %v %v", v, err)
	}
	if _, err := parseLoad("x"); err == nil {
		t.Error("no number parsed")
	}
	if median([]float64{3, 1, 2}) != 2 || median([]float64{4, 1, 2, 3}) != 2.5 || median(nil) != 0 {
		t.Error("median")
	}
}

// fake drives measure without a machine: load, then Go bench output, then timeit.
func fake(load string) func(env []string, name string, args ...string) (string, error) {
	return func(env []string, name string, args ...string) (string, error) {
		switch {
		case name == "sysctl":
			return load, nil
		case name == "py":
			return "100 loops, best of 5: 2 usec per loop", nil
		}
		b := ""
		for _, a := range args {
			if strings.HasPrefix(a, "-test.bench=") {
				b = a
			}
		}
		// -test.bench=^BenchmarkMul$/^1000$
		name2 := strings.TrimPrefix(strings.SplitN(b, "$", 2)[0], "-test.bench=^")
		n := strings.TrimSuffix(strings.SplitN(b, "/^", 2)[1], "$")
		return name2 + "/" + n + "/simd 10 100 ns/op\n" + name2 + "/" + n + "/naive 10 300 ns/op\n", nil
	}
}

func TestMeasure(t *testing.T) {
	var out bytes.Buffer
	cfg := config{binary: "bin", python: "py", rounds: 3, maxLoad: 8, out: &out, run: fake("{ 1.0 1.0 1.0 }")}
	if err := measure(cfg); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.String(), "| Mul | 1000 | 100 | 300 | 2000 | 3.00x | 20.00x |") {
		t.Errorf("table:\n%s", out.String())
	}

	cfg.run = fake("{ 9.5 1.0 1.0 }")
	if err := measure(cfg); err == nil || !strings.Contains(err.Error(), "defer") {
		t.Errorf("loaded machine not refused: %v", err)
	}
	cfg.run = func([]string, string, ...string) (string, error) { return "", errors.New("boom") }
	if err := measure(cfg); err == nil {
		t.Error("no load source not reported")
	}
	cfg.run = fake("{ garbage }")
	if err := measure(cfg); err == nil {
		t.Error("bad load not reported")
	}
}

func TestMeasureErrors(t *testing.T) {
	base := fake("{ 1.0 1.0 1.0 }")
	for name, f := range map[string]func(env []string, n string, a ...string) (string, error){
		"go fails": func(e []string, n string, a ...string) (string, error) {
			if n == "bin" {
				return "x", errors.New("boom")
			}
			return base(e, n, a...)
		},
		"go lines missing": func(e []string, n string, a ...string) (string, error) {
			if n == "bin" {
				return "PASS", nil
			}
			return base(e, n, a...)
		},
		"numpy fails": func(e []string, n string, a ...string) (string, error) {
			if n == "py" {
				return "", errors.New("boom")
			}
			return base(e, n, a...)
		},
		"numpy unparsable": func(e []string, n string, a ...string) (string, error) {
			if n == "py" {
				return "nothing", nil
			}
			return base(e, n, a...)
		},
	} {
		cfg := config{binary: "bin", python: "py", rounds: 1, maxLoad: 8, out: &bytes.Buffer{}, run: f}
		if err := measure(cfg); err == nil {
			t.Errorf("%s: no error", name)
		}
	}
}

func TestRootCmd(t *testing.T) {
	cmd := RootCmd()
	var out bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetErr(&out)
	cmd.SetArgs(nil)
	if err := cmd.Execute(); err == nil {
		t.Error("--binary not required")
	}
	cmd2 := RootCmd()
	cmd2.SetArgs([]string{"--binary", "/nonexistent", "--max-load", "0"})
	cmd2.SetOut(&out)
	cmd2.SetErr(&out)
	if err := cmd2.Execute(); err == nil {
		t.Error("max-load 0 should refuse")
	}
}
