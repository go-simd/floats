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
- `MirrorCombine`: no NumPy equivalent (a fused pair pass of a real transform); timed in the DCT-II/III of go-fft/fft#162 against FFTW, not against NumPy.
- `MirrorCombineSplit`: no NumPy equivalent, as `MirrorCombine`.
