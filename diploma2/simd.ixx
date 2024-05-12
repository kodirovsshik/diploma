
module;

#include <immintrin.h>


export module diploma.simd;

extern "C"
{
	//MSVC bugs workaround
	//the following definitions were moved out of their headers because they are being imported twice:

	extern int _mm_cvt_ss2si(__m128 _A);
	extern int _mm_cvtt_ss2si(__m128 _A);
	extern __m128 _mm_cvt_si2ss(__m128, int);
	extern float _mm_cvtss_f32(__m128 _A);

	extern __m256  __cdecl _mm256_castpd_ps(__m256d);
	extern __m256d __cdecl _mm256_castps_pd(__m256);
	extern __m256i __cdecl _mm256_castps_si256(__m256);
	extern __m256i __cdecl _mm256_castpd_si256(__m256d);
	extern __m256  __cdecl _mm256_castsi256_ps(__m256i);
	extern __m256d __cdecl _mm256_castsi256_pd(__m256i);
	extern __m128  __cdecl _mm256_castps256_ps128(__m256);
	extern __m128d __cdecl _mm256_castpd256_pd128(__m256d);
	extern __m128i __cdecl _mm256_castsi256_si128(__m256i);
	extern __m256  __cdecl _mm256_castps128_ps256(__m128);
	extern __m256d __cdecl _mm256_castpd128_pd256(__m128d);
	extern __m256i __cdecl _mm256_castsi128_si256(__m128i);
}



#define xinline __forceinline

#define __f(i, N) ((i < N) ? -1 : 0)
#define avx_8load_mask(N) _mm256_set_epi32(__f(7, N), __f(6, N), __f(5, N), __f(4, N), __f(3, N), __f(2, N), __f(1, N), __f(0, N))

template<size_t N>
xinline constexpr __m256i get_load_mask8()
{
	static_assert(N <= 8);
	static_assert(N >= 0);
	return avx_8load_mask(N);
}
xinline __m256i get_load_mask8_rt(size_t n)
{
	return avx_8load_mask(n);
}

#undef __f



export
{
	struct avx256
	{
		using fp32 = __m256;
		using i32 = __m256i;

		static constexpr size_t bits = 256;
		static constexpr size_t bytes = 32;

		avx256() = delete;

		xinline static fp32 load_fp32(const float* p)
		{
			return _mm256_loadu_ps(p);
		}

		template<size_t n>
		xinline static fp32 load_n_ct(const float* p)
		{
			return _mm256_maskload_ps(p, get_load_mask8<n>());
		}

		xinline static fp32 load_fp32_n(const float* p, size_t n)
		{
			return _mm256_maskload_ps(p, get_load_mask8_rt(n));
		}

		//xinline static i32 set_epi32(int e7, int e6, int e5, int e4, int e3, int e2, int e1, int e0)
		//{
		//	return _mm256_set_epi32(e7, e6, e5, e4, e3, e2, e1, e0);
		//}
	};
	struct avx512
	{
		using fp32 = __m512;
		using i32 = __m512i;

		static constexpr size_t bits = 512;
		static constexpr size_t bytes = 64;

		avx512() = delete;

		xinline static fp32 load_fp32(const float* p)
		{
			return _mm512_loadu_ps(p);
		}

		xinline static fp32 load_fp32_n(const float* p, size_t n)
		{
			return _mm512_maskz_loadu_ps(-1 << (8 - n), p);
		}
	};





	xinline avx256::fp32 avx_fmadd(avx256::fp32 m1, avx256::fp32 m2, avx256::fp32 a)
	{
		return _mm256_fmadd_ps(m1, m2, a);
	}
	xinline avx512::fp32 avx_fmadd(avx512::fp32 m1, avx512::fp32 m2, avx512::fp32 a)
	{
		return _mm512_fmadd_ps(m1, m2, a);
	}

	template<size_t N>
	xinline float avx_reduce_n_ct(avx256::fp32 ymm)
	{
		static_assert(N >= 1 && N <= 8);
		if constexpr (N > 1) ymm = _mm256_hadd_ps(ymm, _mm256_setzero_ps());
		if constexpr (N > 2) ymm = _mm256_hadd_ps(ymm, _mm256_setzero_ps());
		if constexpr (N > 4) ymm = _mm256_add_ps(ymm, _mm256_castps128_ps256(_mm256_extractf32x4_ps(ymm, 1)));
		return _mm256_cvtss_f32(ymm);
	}

	xinline float avx_reduce(avx256::fp32 ymm)
	{
		ymm = _mm256_hadd_ps(ymm, _mm256_setzero_ps());
		ymm = _mm256_hadd_ps(ymm, _mm256_setzero_ps());
		ymm = _mm256_add_ps(ymm, _mm256_castps128_ps256(_mm256_extractf32x4_ps(ymm, 1)));
		return _mm256_cvtss_f32(ymm);
	}
	xinline float avx_reduce(avx512::fp32 reg)
	{
		return _mm512_reduce_add_ps(reg);
	}
}
