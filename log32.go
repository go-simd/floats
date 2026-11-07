package floats

import (
	"math"
	"math/bits"
)

// logCR is CORE-MATH's correctly rounded binary32 logarithm (Alexei Sibidanov
// and Paul Zimmermann, src/binary32/log/logf.c, MIT), one C operation per Go
// operation. x = 2^e·m, m in [1,2); tr[j] ≈ 1/m on 64 intervals so that
// z = m·tr[j] - 1 is small; log x = e·ln2 + tl[j] + log1p(z), with
// tl[j] = -log(tr[j]). A degree-3 polynomial gives a lower bound r of the
// exact value with r + 0x1.f06p-33 an upper bound; when both round to the
// same float32 that float32 is the answer. Otherwise (762 750 of the 2^31
// positive finite inputs, about 1 in 2800, measured by TestLogCRMatchesLog32)
// a degree-8 polynomial with a split ln2 decides it.
//
// Every product is converted explicitly to float64, which Go defines as
// rounding it, so no architecture fuses it into an FMA: the operations are
// the C source's, unfused, everywhere. TestLogCRMatchesLog32 holds the result
// to log32, already proven correctly rounded, over every float32.
func logCR(x float32) float32 {
	ux := math.Float32bits(x)
	if ux < 1<<23 || ux >= 0x7f800000 {
		if ux == 0 || ux >= 0x7f800000 {
			return logCRSpecial(x)
		}
		n := bits.LeadingZeros32(ux) - 8 // subnormal: normalise the mantissa
		ux <<= n
		ux -= uint32(n) << 23
	}
	if ux == 127<<23 {
		return 0
	}
	m := ux & (1<<23 - 1)
	j := (m + 1<<(23-7)) >> (23 - 6)
	e := float64(int32(ux)>>23 - 127)
	z := float64(math.Float64frombits((uint64(m)|1023<<23)<<(52-23))*logCRTr[j]) - 1
	z2 := float64(z * z)
	r := ((float64(e*0x1.62e42fefa39efp-1) + logCRTl[j]) + float64(z*logCRB[0])) +
		float64(z2*(logCRB[1]+float64(z*logCRB[2])))
	ub, lb := float32(r), float32(r+0x1.f06p-33)
	if ub == lb {
		return ub
	}
	return logCRSlow(x, z, z2, e, j)
}

// logCRSlow is logf.c's second step, for the inputs whose bounds straddle a
// float32 rounding boundary.
func logCRSlow(x float32, z, z2, e float64, j uint32) float32 {
	c := &logCRC
	f := float64(z2 * ((c[0] + float64(z*c[1])) + float64(z2*((c[2]+float64(z*c[3]))+
		float64(z2*(c[4]+float64(z*c[5])+float64(z2*c[6])))))))
	if math.Abs(float64(x-1)) < 0x1p-10 {
		return float32(z + f)
	}
	f -= float64(0x1.0ca86c3898dp-49 * e)
	f += z
	f += logCRTl[j] - logCRTl[0]
	el := float64(e * 0x1.62e42fefa3ap-1)
	r := el + f
	ub := float32(r)
	if math.Float64bits(r)&(1<<28-1) == 0 {
		dr := (el - r) + f
		r += float64(dr * 64)
		ub = float32(r)
	}
	return ub
}

// Float32Log writes the correctly rounded float32 log(src[i]) into dst[i] for
// i < len(src) (proven over all 2^32 inputs by TestLog32CorrectlyRounded, which
// is env-gated): logCR, CORE-MATH's correctly rounded logf (MIT), on an AVX2
// kernel on amd64 and a NEON kernel on arm64, in Go elsewhere. log(±0) = -Inf,
// log(+Inf) = +Inf, negative and NaN inputs give NaN. Nothing is allocated.
// Moved here from go-ndarray's internal/kernels with the kernels of
// go-ndarray#94 and #95. It panics if len(dst) < len(src).
func Float32Log(dst, src []float32) {
	if len(dst) < len(src) {
		panic("floats: Float32Log: len(dst) < len(src)")
	}
	log32Fast(dst[:len(src)], src)
}

// logCRSpecial is logf.c's as_special: ±0 → -Inf, +Inf → +Inf, NaN → NaN,
// negatives → NaN.
func logCRSpecial(x float32) float32 {
	switch ux := math.Float32bits(x); {
	case ux<<1 == 0:
		return float32(math.Inf(-1))
	case ux == 0x7f800000:
		return x
	case ux<<1 > 0xff000000:
		return x + x
	}
	return float32(math.NaN())
}

// logCRSlice writes logCR(src[i]) into dst[i].
func logCRSlice(dst, src []float32) {
	src = src[:len(dst)]
	for i, x := range src {
		dst[i] = logCR(x)
	}
}

var logCRTr = [65]float64{
	0x1p+0, 0x1.f81f82p-1, 0x1.f07c1fp-1, 0x1.e9131acp-1,
	0x1.e1e1e1ep-1, 0x1.dae6077p-1, 0x1.d41d41dp-1, 0x1.cd85689p-1,
	0x1.c71c71cp-1, 0x1.c0e0704p-1, 0x1.bacf915p-1, 0x1.b4e81b5p-1,
	0x1.af286bdp-1, 0x1.a98ef6p-1, 0x1.a41a41ap-1, 0x1.9ec8e95p-1,
	0x1.999999ap-1, 0x1.948b0fdp-1, 0x1.8f9c19p-1, 0x1.8acb90fp-1,
	0x1.8618618p-1, 0x1.8181818p-1, 0x1.7d05f41p-1, 0x1.78a4c81p-1,
	0x1.745d174p-1, 0x1.702e05cp-1, 0x1.6c16c17p-1, 0x1.6816817p-1,
	0x1.642c859p-1, 0x1.605816p-1, 0x1.5c9882cp-1, 0x1.58ed231p-1,
	0x1.5555555p-1, 0x1.51d07ebp-1, 0x1.4e5e0a7p-1, 0x1.4afd6ap-1,
	0x1.47ae148p-1, 0x1.446f865p-1, 0x1.4141414p-1, 0x1.3e22cbdp-1,
	0x1.3b13b14p-1, 0x1.3813814p-1, 0x1.3521cfbp-1, 0x1.323e34ap-1,
	0x1.2f684bep-1, 0x1.2c9fb4ep-1, 0x1.29e412ap-1, 0x1.27350b9p-1,
	0x1.2492492p-1, 0x1.21fb781p-1, 0x1.1f7047ep-1, 0x1.1cf06aep-1,
	0x1.1a7b961p-1, 0x1.1811812p-1, 0x1.15b1e5fp-1, 0x1.135c811p-1,
	0x1.1111111p-1, 0x1.0ecf56cp-1, 0x1.0c9715p-1, 0x1.0a6810ap-1,
	0x1.0842108p-1, 0x1.0624dd3p-1, 0x1.041041p-1, 0x1.0204081p-1, 0.5,
}

var logCRTl = [65]float64{
	-0x1.3b40815cd0628p-45, 0x1.fc0a890fbb514p-7, 0x1.f829b1e780b98p-6, 0x1.77458f532c948p-5,
	0x1.f0a30c2114ef2p-5, 0x1.341d793bbc7f7p-4, 0x1.6f0d28d256172p-4, 0x1.a926d3a6acb89p-4,
	0x1.e2707722ae90cp-4, 0x1.0d77e7a90896cp-3, 0x1.29552f6fff036p-3, 0x1.44d2b6c5b7831p-3,
	0x1.5ff306ee78ee7p-3, 0x1.7ab890410d41cp-3, 0x1.9525a9e3451c7p-3, 0x1.af3c94ed0bb06p-3,
	0x1.c8ff7c59a9535p-3, 0x1.e27076d5aedf9p-3, 0x1.fb9186b5e393ep-3, 0x1.0a324e38b8e6dp-2,
	0x1.1675cacaba398p-2, 0x1.22941fc0f76efp-2, 0x1.2e8e2bc311abap-2, 0x1.3a64c56b14373p-2,
	0x1.4618bc31c5c4cp-2, 0x1.51aad874df5b7p-2, 0x1.5d1bdbea80754p-2, 0x1.686c81d331238p-2,
	0x1.739d7f6dbcd9p-2, 0x1.7eaf83c82ad4dp-2, 0x1.89a3385813fe4p-2, 0x1.947941aa91484p-2,
	0x1.9f323edbf95d5p-2, 0x1.a9cec9a4205d3p-2, 0x1.b44f77c5c8cecp-2, 0x1.beb4d9ea71905p-2,
	0x1.c8ff7c69a97abp-2, 0x1.d32fe7f38e95fp-2, 0x1.dd46a0501c22ap-2, 0x1.e7442617e8511p-2,
	0x1.f128f5eaf0476p-2, 0x1.faf588dd8f0a8p-2, 0x1.02552a5edcfc4p-1, 0x1.0723e5c64de05p-1,
	0x1.0be72e3852947p-1, 0x1.109f39d554b5cp-1, 0x1.154c3d2c4d4aep-1, 0x1.19ee6b38bc834p-1,
	0x1.1e85f5ef03f95p-1, 0x1.23130d7fabe07p-1, 0x1.2795e1219afep-1, 0x1.2c0e9ec9c8d5p-1,
	0x1.307d7337f0f83p-1, 0x1.34e289cb4e098p-1, 0x1.393e0d42e28dep-1, 0x1.3d9026ad555bfp-1,
	0x1.41d8fe8667173p-1, 0x1.4618bc1ec5d87p-1, 0x1.4a4f85d303d8p-1, 0x1.4e7d8127f5a75p-1,
	0x1.52a2d26dbc47p-1, 0x1.56bf9d597f25ep-1, 0x1.5ad404cb59df2p-1, 0x1.5ee02a928153ap-1,
	0x1.62e42fefa38b4p-1,
}

var logCRB = [3]float64{0x1.00000006342eap+0, -0x1.0001f7fdc3977p-1, 0x1.554a4e5cae9cfp-2}

var logCRC = [7]float64{
	-0x1p-1, 0x1.55555555571cap-2, -0x1.0000000002d85p-2, 0x1.9999987d0c963p-3,
	-0x1.555554059a8bbp-3, 0x1.24aebcf71a38fp-3, -0x1.001c73915d758p-3,
}
