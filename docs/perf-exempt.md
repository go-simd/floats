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
- `MirrorCombine`: no NumPy equivalent (a fused pair pass of a real transform); timed in the DCT-II/III of go-fft/fft#162 against FFTW, not against NumPy.
- `CMulZipRev`: no NumPy equivalent (a reversed gather fused with a complex multiply); timed in the DCT-III of go-fft/fft#162 against FFTW, not against NumPy.
- `MirrorCombineSplit`: no NumPy equivalent, as `MirrorCombine`.
- `CMulTo`: NumPy timing deferred (`np.multiply` on complex128; a valid refbench run needs a 1-minute load below 8, it was 25 to 45 on the arm64 machine that writes perf.json). Timed ad hoc on one Zen 4 core against `np.multiply(a, b, out=d)`: 14 ns against 299 at 64 numbers, 287 against 772 at 1024, 21.9 us against 24.4 us at 65536; README.
- `AllPass3`: no NumPy equivalent (three Schroeder links of the Parametric Stereo decorrelator over complex rows); timed in go-avkit/decode against the same loop in Go, not against NumPy.
