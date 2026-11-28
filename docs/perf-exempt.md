# Public functions without a NumPy timing row

Every public function has a case in `bench/perf/perf_test.go` or a line here with
its reason (`cd bench/perf && go test -run TestPerf .` checks it).

- `Abs`: NumPy timing deferred (`np.abs`). Timed against the scalar Go loop on one Zen 3 core, README. A valid refbench run needs a 1-minute load below 8; it was 15 to 60 on the machine that wrote perf.json when this was added.
- `Neg`: NumPy timing deferred (`np.negative`), as `Abs`.
- `Square`: NumPy timing deferred (`np.square`), as `Abs`.
- `Floor`: NumPy timing deferred (`np.floor`), as `Abs`.
- `Ceil`: NumPy timing deferred (`np.ceil`), as `Abs`.
- `Trunc`: NumPy timing deferred (`np.trunc`), as `Abs`.
- `RoundEven`: NumPy timing deferred (`np.rint`), as `Abs`.
- `Maximum`: NumPy timing deferred (`np.maximum`). Timed against the scalar Go loop on one Zen 3 core, README. A valid refbench run needs a 1-minute load below 8.
- `Minimum`: NumPy timing deferred (`np.minimum`), as `Maximum`.
- `Add32`: NumPy timing deferred (`np.add` on float32). Timed against the scalar Go loop by `go test -bench Binop32`. A valid refbench run needs a 1-minute load below 8; it was 36 to 45 when this was added.
- `Sub32`: NumPy timing deferred (`np.subtract`), as `Add32`.
- `Mul32`: NumPy timing deferred (`np.multiply`), as `Add32`.
- `Div32`: NumPy timing deferred (`np.divide`), as `Add32`.
- `Sqrt32`: NumPy timing deferred (`np.sqrt`), as `Add32`.
- `NewAllPass3Plan`: the proved-memory form of `AllPass3` (allocation, no kernel work); timed in go-avkit/decode against the same loop in Go, as `AllPass3`.
- `AllPass3Plan.Run`: the kernel of `AllPass3` behind slot numbers; as `AllPass3`, timed in go-avkit/decode (`BenchmarkAllPass3Plan` here against the checked call).
- `NewAllPass3TickPlan`: `NewAllPass3Plan` plus an input delay (allocation, no kernel work); timed in go-avkit/decode, as `AllPass3`.
- `AllPass3Plan.RunTick`: the kernel of `AllPass3` behind a tick counter, no per-call check; as `AllPass3`, timed in go-avkit/decode (`BenchmarkAllPass3PlanTick` here against `Run`).
- `AllPass3Plan.Out`: an accessor of the plan's output slice, no work to time.
- `AllPass3Plan.Rows`: an accessor of the plan's row count, no work to time.
- `MirrorCombine`: no NumPy equivalent (a fused pair pass of a real transform); timed in the DCT-II/III of go-fft/fft#162 against FFTW, not against NumPy.
- `CMulZipRev`: no NumPy equivalent (a reversed gather fused with a complex multiply); timed in the DCT-III of go-fft/fft#162 against FFTW, not against NumPy.
- `MirrorCombineSplit`: no NumPy equivalent, as `MirrorCombine`.
- `CMulTo`: NumPy timing deferred (`np.multiply` on complex128; a valid refbench run needs a 1-minute load below 8, it was 25 to 45 on the arm64 machine that writes perf.json). Timed ad hoc (v0.32.0 kernel) on one Zen 4 core against `np.multiply(a, b, out=d)`: 14 ns against 299 at 64 numbers, 287 against 772 at 1024, 21.9 us against 24.4 us at 65536; README.
- `MulAddRows32`: no NumPy equivalent (a row-wise float32 window sum of a polyphase bank); timed against the portable float64 `MulAddRows` on the same shape (13.7 ns against 23.8 ns, 64 columns, 10 rows, one Zen 4 core) by `go test -bench MulAddRows32`, and in go-avkit/decode#77.
- `AllPass3`: no NumPy equivalent (three Schroeder links of the Parametric Stereo decorrelator over complex rows); timed in go-avkit/decode against the same loop in Go, not against NumPy.
- `AllPass3Split32`: no NumPy equivalent (the float32 split-plane form of `AllPass3`); timed here against the float64 `AllPass3` by `BenchmarkAllPass3Split32`, and in go-avkit/decode#77.
- `NewAllPass3TickPlan32`: the proved-memory form of `AllPass3Split32` (allocation, no kernel work); as `AllPass3Split32`.
- `NewPSMixPlan32`, `PSMixPlan32.Runs`: construction and accessor of the plan of `PSMixPlan32.Run` (allocation, no kernel work).
- `PSMixPlan32.Run`: no NumPy equivalent (the PS mixing of HE-AAC v2 over a slot of short runs); timed against the plain-Go loop by `BenchmarkPSMix32`, and in go-avkit/decode#77.
- `AllPass3Plan32.RunTick`: the kernel of `AllPass3Split32` behind a tick counter, no per-call check; `BenchmarkAllPass3Split32/plan-tick`.
- `AllPass3Plan32.Out`: an accessor of the plan's output planes, no work to time.
- `AllPass3Plan32.Rows`: an accessor of the plan's row count, no work to time.
