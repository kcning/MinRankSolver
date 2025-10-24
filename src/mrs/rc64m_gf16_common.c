#include "rc64m_gf16.h"
#include "grp64_gf16.h"

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

#if defined(__AVX512F__)

static inline __m256i
rc64m_gf16_mul_per_row_avx512(const Grp64GF16* restrict row,
                              const RC64MGF16* restrict n) {
    const Grp64GF16* src = rc64m_gf16_raddr((RC64MGF16*)n, 0);
    __m512i prod, p0, p1;
    p0 = grp64_gf16_mul_scalar_from_bs_adj_avx512_no_split(src + 0, row, 0);
    p1 = grp64_gf16_mul_scalar_from_bs_adj_avx512_no_split(src + 2, row, 2);
    prod = _mm512_xor_si512(p0, p1);
    src += 4;

    for(uint32_t j = 4; j < 64; j += 4, src += 4) {
        p0 = grp64_gf16_mul_scalar_from_bs_adj_avx512_no_split(src,row,j);
        p1 = grp64_gf16_mul_scalar_from_bs_adj_avx512_no_split(src+2,row,j+2);
        prod = _mm512_xor_si512(prod, p0);
        prod = _mm512_xor_si512(prod, p1);
    }

    __m256i vhi = _mm512_extracti64x4_epi64(prod, 1);
    __m256i vlo = _mm512_extracti64x4_epi64(prod, 0);
    return _mm256_xor_si256(vhi, vlo);
}

#endif
