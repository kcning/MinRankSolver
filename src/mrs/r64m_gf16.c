#include "r64m_gf16.h"
#include "grp64_gf16.h"
#include "util.h"
#include "rc64m_gf16.h"
#include <stdint.h>
#include <string.h>
#include <stdalign.h>
#include "rc64m_gf16_common.c"

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

/* ========================================================================
 * struct R64MGF16 definition
 * ======================================================================== */

struct R64MGF16 {
    uint64_t rnum; // uint32_t will cause padding. Might as well use 64-bit
    // NOTE: same alignment requirement as Grp64GF16
    alignas(32) Grp64GF16 rows[];
};

/* ========================================================================
 * function implementations
 * ======================================================================== */

/* usage: Compute the size of memory needed for R64MGF16
 * params:
 *      1) rnum: number of rows
 * return: size of memory needed in bytes */
uint64_t
r64m_gf16_memsize(uint32_t rnum) {
    return sizeof(R64MGF16) + sizeof(Grp64GF16) * rnum;
}

/* usage: Create a R64MGF16 matrix. The matrix is not initialized
 * params:
 *      1) rnum: number of rows
 * return: ptr to struct R64MGF16. NULL on faliure */
R64MGF16*
r64m_gf16_create(uint32_t rnum) {
    static_assert(sizeof(Grp64GF16) == 32, "size of Grp64GF16 is not 32 bytes");
    // NOTE: same alignment requirement as Grp64GF16
    R64MGF16* m = aligned_alloc(32, r64m_gf16_memsize(rnum));
    if(!m)
        return NULL;

    m->rnum = rnum;
    return m;
}

/* usage: Release a struct R64MGF16
 * params:
 *      1) m: ptr to struct R64MGF16
 * return: void */
void
r64m_gf16_free(R64MGF16* m) {
    free(m);
}

/* usage: Given a R64MGF16 matrix, return the number of rows
 * params:
 *      1) m: ptr to a struct R64MGF16
 * return: the number of rows */
uint32_t
r64m_gf16_rnum(const R64MGF16* m) {
    return m->rnum;
}

/* usage: Given a R64MGF16 matrix and row index, return the address
 *      to the row
 * params:
 *      1) m: ptr to a struct R64MGF16
 *      2) i: index of the row, starting from 0
 * return: address to the i-th row */
Grp64GF16*
r64m_gf16_raddr(R64MGF16* m, uint32_t i) {
    return m->rows + i;
}

/* usage: Given a R64MGF16 matrix and both row and column indices, return
 *      the coefficient of the given entry.
 * params:
 *      1) m: ptr to a struct R64MGF16
 *      2) ri: index of the row, starting from 0
 *      3) ci: index of the column, from 0 ~ 63
 * return: coefficient of the entry */
gf16_t
r64m_gf16_at(const R64MGF16* m, uint32_t ri, uint32_t ci) {
    return grp64_gf16_at(r64m_gf16_raddr((R64MGF16*) m, ri), ci);
}

/* usage: Given a R64MGF16 matrix and both row and column indices, set
 *      the coefficient of the target entry to the given value.
 * params:
 *      1) m: ptr to a struct R64MGF16
 *      2) ri: index of the row, starting from 0
 *      3) ci: index of the column, from 0 ~ 63
 *      4) v: the new coefficient
 * return: void */
void
r64m_gf16_set_at(R64MGF16* m, uint64_t ri, uint64_t ci, gf16_t v) {
    grp64_gf16_set_at(r64m_gf16_raddr(m, ri), ci, v);
}

/* usage: Reset a struct R64MGF16 to zero matrix
 * params:
 *      1) m: ptr to a struct R64MGF16
 * return: void */
void
r64m_gf16_zero(R64MGF16* const m) {
    memset(m->rows, 0x0, sizeof(Grp64GF16) * r64m_gf16_rnum(m));
}

/* usage: Given a struct R64MGF16, populate it with random coefficients.
 * params:
 *      1) m: ptr to a struct R64MGF16
 * return: void */
void
r64m_gf16_rand(R64MGF16* m) {
    for(uint32_t i = 0; i < r64m_gf16_rnum(m); ++i)
        grp64_gf16_rand(r64m_gf16_raddr(m, i));
}

/* usage: Given 2 struct R64MGF16, copy the 2nd R64MGF16 into the 1st one
 * params:
 *      1) dst: ptr to the struct R64MGF16 to copy to
 *      2) src: ptr to the struct R64MGF16 to copy from
 * return: void */
void
r64m_gf16_copy(R64MGF16* restrict dst, const R64MGF16* restrict src) {
    assert(r64m_gf16_rnum(dst) == r64m_gf16_rnum(src));
    memcpy(dst->rows, src->rows, sizeof(Grp64GF16) * r64m_gf16_rnum(dst));
}

#if defined(__AVX512F__)

static force_inline void
r64m_gf16_gramian_avx512(const R64MGF16* restrict m, RC64MGF16* restrict p) {
    const Grp64GF16* m_row = r64m_gf16_raddr((R64MGF16*) m, 0);
    for(uint32_t i = 0; i < 64; i += 4) {
        __m512i p0 = grp64_gf16_mul_scalar_from_bs_1x2_avx512(m_row,m_row,i);
        __m512i p1 = grp64_gf16_mul_scalar_from_bs_1x2_avx512(m_row,m_row,i+2);
        Grp64GF16* dst0 = rc64m_gf16_raddr(p, i);
        Grp64GF16* dst1 = rc64m_gf16_raddr(p, i + 2);
        _mm512_store_si512(dst0, p0);
        _mm512_store_si512(dst1, p1);
    }
    ++m_row;

    for(uint64_t ri = 1; ri < r64m_gf16_rnum(m); ++ri, ++m_row) {
        for(uint32_t i = 0; i < 64; i += 4) {
            __m512i p0 =
                grp64_gf16_mul_scalar_from_bs_1x2_avx512(m_row, m_row, i);
            __m512i p1 =
                grp64_gf16_mul_scalar_from_bs_1x2_avx512(m_row, m_row, i + 2);
            Grp64GF16* dst0 = rc64m_gf16_raddr(p, i);
            Grp64GF16* dst1 = rc64m_gf16_raddr(p, i + 2);
            __m512i v0 = _mm512_load_si512(dst0);
            __m512i v1 = _mm512_load_si512(dst1);
            _mm512_store_si512(dst0, _mm512_xor_si512(v0, p0));
            _mm512_store_si512(dst1, _mm512_xor_si512(v1, p1));
        }
    }
}

#elif defined(__AVX2__)

static force_inline void
r64m_gf16_gramian_avx2(const R64MGF16* restrict m, RC64MGF16* restrict p) {
    const Grp64GF16* m_row = r64m_gf16_raddr((R64MGF16*) m, 0);
    const __m256i v_1st = _mm256_load_si256((__m256i*) m_row->b);
    for(uint32_t i = 0; i < 64; i += 2) {
        Grp64GF16* dst0 = rc64m_gf16_raddr(p, i);
        Grp64GF16* dst1 = rc64m_gf16_raddr(p, i + 1);
        __m256i p0 = grp64_gf16_mul_scalar_from_bs_avx2(v_1st, m_row, i);
        __m256i p1 = grp64_gf16_mul_scalar_from_bs_avx2(v_1st, m_row, i + 1);
        _mm256_store_si256((__m256i*) dst0->b, p0);
        _mm256_store_si256((__m256i*) dst1->b, p1);
    }

    for(uint64_t ri = 1; ri < r64m_gf16_rnum(m); ++ri) {
        const __m256i v = _mm256_load_si256((__m256i*) (++m_row)->b);
        for(uint32_t i = 0; i < 64; i += 2) {
            Grp64GF16* dst0 = rc64m_gf16_raddr(p, i);
            Grp64GF16* dst1 = rc64m_gf16_raddr(p, i + 1);
            __m256i p0 = grp64_gf16_mul_scalar_from_bs_avx2(v, m_row, i);
            __m256i p1 = grp64_gf16_mul_scalar_from_bs_avx2(v, m_row, i + 1);
            __m256i va = _mm256_load_si256((__m256i*) dst0->b);
            __m256i vb = _mm256_load_si256((__m256i*) dst1->b);
            _mm256_store_si256((__m256i*) dst0->b, _mm256_xor_si256(va, p0));
            _mm256_store_si256((__m256i*) dst1->b, _mm256_xor_si256(vb, p1));
        }
    }
}

#else

static force_inline void
r64m_gf16_gramian_naive(const R64MGF16* restrict m, RC64MGF16* restrict p) {
    rc64m_gf16_zero(p);
    for(uint64_t ri = 0; ri < r64m_gf16_rnum(m); ++ri) {
        const Grp64GF16* m_row = r64m_gf16_raddr((R64MGF16*) m, ri);
        for(uint32_t i = 0; i < 64; i += 2) {
            Grp64GF16* dst0 = rc64m_gf16_raddr(p, i);
            Grp64GF16* dst1 = rc64m_gf16_raddr(p, i + 1);
            grp64_gf16_fmaddi_scalar_bs(dst0, m_row, m_row, i);
            grp64_gf16_fmaddi_scalar_bs(dst1, m_row, m_row, i + 1);
        }
    }
}

#endif

/* usage: Given a R64MGF16 matrix m and a container, compute the Gramian
 *      matrix of m, i.e. m.transpose() * m, and store it into the container.
 *      Note that the product has dimension 64 x 64.
 * params:
 *      1) m: ptr to a struct R64MGF16
 *      2) p: ptr to a struct RC64MGF16, container for the result
 * return: void */
void
r64m_gf16_gramian(const R64MGF16* restrict m, RC64MGF16* restrict p) {
#if defined(__AVX512F__)
    r64m_gf16_gramian_avx512(m, p);
#elif defined(__AVX2__)
    r64m_gf16_gramian_avx2(m, p);
#else
    r64m_gf16_gramian_naive(m, p);
#endif
}

/* usage: Given a R64MGF16, find the columns that are fully zero
 * params:
 *      1) m: ptr to struct R64MGF16
 * return: a 64-bit integer that encodes zero columns. If the first column
 *      is fully zero, the LSB is set to 1, and so on. */
uint64_t
r64m_gf16_zc_pos(const R64MGF16* m) {
    uint64_t zp = UINT64_MAX;
    for(uint32_t i = 0; i < r64m_gf16_rnum(m) && zp; ++i) {
        const Grp64GF16* row = r64m_gf16_raddr((R64MGF16*)m, i);
        zp &= grp64_gf16_zpos(row);
    }

    return zp;
}

/* usage: Given a R64MGF16, find the columns whose selected rows are fully zero
 * params:
 *      1) m: ptr to struct R64MGF16
 *      2) ridxs: a uint32_t array that stores the indices of the selected
 *          rows
 *      3) sz: size of ridxs
 * return: a 64-bit integer that encodes zero columns. If the first column
 *      is fully zero, the LSB is set to 1, and so on. */
uint64_t
r64m_gf16_subset_zc_pos(const R64MGF16* restrict m,
                        const uint32_t* restrict ridxs, uint32_t sz) {
    uint64_t zp = UINT64_MAX;
    for(uint32_t i = 0; i < sz; ++i) {
        uint32_t ri = ridxs[i];
        const Grp64GF16* row = r64m_gf16_raddr((R64MGF16*)m, ri);
        zp &= grp64_gf16_zpos(row);
    }
    return zp;
}

/* usage: Given a R64MGF16, find the columns that are not fully zero
 * params:
 *      1) m: ptr to struct R64MGF16
 * return: a 64-bit integer that encodes non-zero columns. If the first column
 *      is not fully zero, the LSB is set to 1, and so on. */
uint64_t
r64m_gf16_nzc_pos(const R64MGF16* m) {
    return ~r64m_gf16_zc_pos(m);
}

#if defined(__AVX512F__)

static force_inline void
r64m_gf16_fma_avx512(R64MGF16* restrict a, const R64MGF16* restrict b,
                     const RC64MGF16* restrict c) {
    for(uint32_t i = 0; i < r64m_gf16_rnum(a); ++i) {
        Grp64GF16* dst = r64m_gf16_raddr(a, i);
        __m256i prod = _mm256_load_si256((__m256i*) dst);
        const Grp64GF16* b_row = r64m_gf16_raddr((R64MGF16*) b, i);
        __m256i res = rc64m_gf16_mul_per_row_avx512(b_row, c);
        _mm256_store_si256((__m256i*) dst, _mm256_xor_si256(prod, res));
    }
}

#elif defined(__AVX2__)

static force_inline void
r64m_gf16_fma_avx2(R64MGF16* restrict a, const R64MGF16* restrict b,
                   const RC64MGF16* restrict c) {
    for(uint32_t i = 0; i < r64m_gf16_rnum(a); ++i) {
        Grp64GF16* dst = r64m_gf16_raddr(a, i);
        __m256i prod = _mm256_load_si256((__m256i*) dst);
        const Grp64GF16* b_row = r64m_gf16_raddr((R64MGF16*) b, i);
        const __m256i* src = (__m256i*) rc64m_gf16_raddr((RC64MGF16*)c, 0);
        for(uint32_t j = 0; j < 64; j += 2, src += 2) {
            __m256i v0 = _mm256_load_si256(src);
            __m256i v1 = _mm256_load_si256(src + 1);
            __m256i p0 = grp64_gf16_mul_scalar_from_bs_avx2(v0, b_row, j);
            __m256i p1 = grp64_gf16_mul_scalar_from_bs_avx2(v1, b_row, j + 1);
            prod = _mm256_xor_si256(prod, p0);
            prod = _mm256_xor_si256(prod, p1);
        }
        _mm256_store_si256((__m256i*) dst, prod);
    }
}

#else

static force_inline void
r64m_gf16_fma_naive(R64MGF16* restrict a, const R64MGF16* restrict b,
                    const RC64MGF16* restrict c) {
    Grp64GF16* dst = r64m_gf16_raddr(a, 0);
    for(uint32_t i = 0; i < r64m_gf16_rnum(a); ++i, ++dst) {
        const Grp64GF16* b_row = r64m_gf16_raddr((R64MGF16*) b, i);
        const Grp64GF16* src = rc64m_gf16_raddr((RC64MGF16*)c, 0);
        for(uint32_t j = 0; j < 64; j += 2, src += 2) {
            grp64_gf16_fmaddi_scalar_bs(dst, src, b_row, j);
            grp64_gf16_fmaddi_scalar_bs(dst, src + 1, b_row, j + 1);
        }
    }
}

#endif

/* usage: Given 2 R64MGF16 A and B, and a RC64MGF16 C, compute
 *      A + B * C and store the result back into A
 * params:
 *      1) a: ptr to struct R64MGF16, storing the matrix A
 *      2) b: ptr to struct R64MGF16, storing the matrix B
 *      3) c: ptr to struct RC64MGF16, storing the matrix C
 * return: void */
void
r64m_gf16_fma(R64MGF16* restrict a, const R64MGF16* restrict b,
              const RC64MGF16* restrict c) {
    assert(r64m_gf16_rnum(a) == r64m_gf16_rnum(b));
#if defined(__AVX512F__)
    r64m_gf16_fma_avx512(a, b, c);
#elif defined(__AVX2__)
    r64m_gf16_fma_avx2(a, b, c);
#else
    r64m_gf16_fma_naive(a, b, c);
#endif
}

#if defined(__AVX512F__)

static force_inline void
r64m_gf16_fma_diag_avx512(R64MGF16* restrict a, const R64MGF16* restrict b,
                          const RC64MGF16* restrict c, const uint64_t d) {
    const __m256i vd = _mm256_set1_epi64x(d);
    const Grp64GF16* b_row = r64m_gf16_raddr((R64MGF16*) b, 0);
    for(uint32_t i = 0; i < r64m_gf16_rnum(a); ++i, ++b_row) {
        __m256i prod = rc64m_gf16_mul_per_row_avx512(b_row, c);
        __m256i* dst = (__m256i*) r64m_gf16_raddr(a, i);
        __m256i v = _mm256_load_si256(dst);
        prod = _mm256_and_si256(prod, vd);
        _mm256_store_si256(dst, _mm256_xor_si256(prod, v));
    }
}

#elif defined(__AVX2__)

static force_inline void
r64m_gf16_fma_diag_avx2(R64MGF16* restrict a, const R64MGF16* restrict b,
                        const RC64MGF16* restrict c, const uint64_t d) {
    const __m256i vd = _mm256_set1_epi64x(d);
    const Grp64GF16* b_row = r64m_gf16_raddr((R64MGF16*) b, 0);
    for(uint32_t i = 0; i < r64m_gf16_rnum(a); ++i, ++b_row) {
        const Grp64GF16* src = rc64m_gf16_raddr((RC64MGF16*)c, 0);
        __m256i prod = _mm256_setzero_si256();
        for(uint32_t j = 0; j < 64; j += 2, src += 2) {
            __m256i v0 = _mm256_load_si256( (__m256i*) src->b );
            __m256i v1 = _mm256_load_si256( (__m256i*) ((src + 1)->b) );
            __m256i p0 = grp64_gf16_mul_scalar_from_bs_avx2(v0, b_row, j);
            __m256i p1 = grp64_gf16_mul_scalar_from_bs_avx2(v1, b_row, j+1);
            prod = _mm256_xor_si256(prod, p0);
            prod = _mm256_xor_si256(prod, p1);
        }
        __m256i* dst = (__m256i*) r64m_gf16_raddr(a, i);
        __m256i v = _mm256_load_si256(dst);
        prod = _mm256_and_si256(prod, vd);
        _mm256_store_si256(dst, _mm256_xor_si256(prod, v));
    }
}

#else

static force_inline void
r64m_gf16_fma_diag_naive(R64MGF16* restrict a, const R64MGF16* restrict b,
                         const RC64MGF16* restrict c, const uint64_t d) {
    const Grp64GF16* b_row = r64m_gf16_raddr((R64MGF16*) b, 0);
    Grp64GF16* dst = r64m_gf16_raddr(a, 0);
    for(uint32_t i = 0; i < r64m_gf16_rnum(a); ++i, ++b_row, ++dst) {
        const Grp64GF16* src = rc64m_gf16_raddr((RC64MGF16*)c, 0);

        for(uint32_t j = 0; j < 64; j += 2, src += 2) {
            grp64_gf16_fmaddi_scalar_mask_bs(dst, src, b_row, j, d);
            grp64_gf16_fmaddi_scalar_mask_bs(dst, src + 1, b_row, j + 1, d);
        }
    }
}

#endif

/* usage: Given 2 R64MGF16 A and B, a RC64MGF16 C, and a 64x64 diagonal
 *      matrix D with coefficients either 1 and 0, compute A + B * C * D
 *      and store the result back into A
 * params:
 *      1) a: ptr to struct R64MGF16, storing the matrix A
 *      2) b: ptr to struct R64MGF16, storing the matrix B
 *      3) c: ptr to struct RC64MGF16, storing the matrix C
 *      4) d: a 64-bit integer that encodes the diagonal matrix D. If the LSB
 *          is 1, then entry (0, 0) of D is 1. Otherwise 0.
 * return: void */
void
r64m_gf16_fma_diag(R64MGF16* restrict a, const R64MGF16* restrict b,
                   const RC64MGF16* restrict c, const uint64_t d) {
    assert(r64m_gf16_rnum(a) == r64m_gf16_rnum(b));
#if defined(__AVX512F__)
    r64m_gf16_fma_diag_avx512(a, b, c, d);
#elif defined(__AVX2__)
    r64m_gf16_fma_diag_avx2(a, b, c, d);
#else
    r64m_gf16_fma_diag_naive(a, b, c, d);
#endif
}

#if defined(__AVX512F__)

static force_inline void
r64m_gf16_diag_fma_avx512(R64MGF16* restrict a, const R64MGF16* restrict b,
                          const RC64MGF16* restrict c, uint64_t d) {
    const __m256i vm = _mm256_set1_epi64x(d);
    for(uint32_t i = 0; i < r64m_gf16_rnum(a); ++i) {
        const Grp64GF16* b_row = r64m_gf16_raddr((R64MGF16*) b, i);
        __m256i* dst = (__m256i*) r64m_gf16_raddr(a, i);
        __m256i prod = _mm256_and_si256(vm, _mm256_load_si256(dst));
        __m256i res = rc64m_gf16_mul_per_row_avx512(b_row, c);
        _mm256_store_si256(dst, _mm256_xor_si256(prod, res));
    }
}

#elif defined(__AVX2__)

static force_inline void
r64m_gf16_diag_fma_avx2(R64MGF16* restrict a, const R64MGF16* restrict b,
                        const RC64MGF16* restrict c, uint64_t d) {
    const __m256i vm = _mm256_set1_epi64x(d);
    for(uint32_t i = 0; i < r64m_gf16_rnum(a); ++i) {
        __m256i* dst = (__m256i*) r64m_gf16_raddr(a, i);
        __m256i prod = _mm256_and_si256(vm, _mm256_load_si256(dst));
        const Grp64GF16* b_row = r64m_gf16_raddr((R64MGF16*) b, i);
        const Grp64GF16* src = rc64m_gf16_raddr((RC64MGF16*)c, 0);
        for(uint32_t j = 0; j < 64; j += 2, src += 2) {
            __m256i v0 = _mm256_load_si256( (__m256i*) src->b );
            __m256i v1 = _mm256_load_si256( (__m256i*) ((src + 1)->b) );
            __m256i p0 = grp64_gf16_mul_scalar_from_bs_avx2(v0, b_row, j);
            __m256i p1 = grp64_gf16_mul_scalar_from_bs_avx2(v1, b_row, j+1);
            prod = _mm256_xor_si256(prod, p0);
            prod = _mm256_xor_si256(prod, p1);
        }
        _mm256_store_si256(dst, prod);
    }
}

#else

static force_inline void
r64m_gf16_diag_fma_naive(R64MGF16* restrict a, const R64MGF16* restrict b,
                         const RC64MGF16* restrict c, uint64_t d) {
    Grp64GF16* dst = r64m_gf16_raddr(a, 0);
    for(uint32_t i = 0; i < r64m_gf16_rnum(a); ++i, ++dst) {
        const Grp64GF16* b_row = r64m_gf16_raddr((R64MGF16*) b, i);
        grp64_gf16_zero_subset(dst, d);
        const Grp64GF16* src = rc64m_gf16_raddr((RC64MGF16*)c, 0);
        for(uint32_t j = 0; j < 64; j += 2, src += 2) {
            grp64_gf16_fmaddi_scalar_bs(dst, src, b_row, j);
            grp64_gf16_fmaddi_scalar_bs(dst, src + 1, b_row, j + 1);
        }
    }
}

#endif

/* usage: Given 2 R64MGF16 A and B, a RC64MGF16 C, and a 64x64 diagonal
 *      matrix D with coefficients either 1 and 0, compute A * D + B * C
 *      and store the result back into A
 * params:
 *      1) a: ptr to struct R64MGF16, storing the matrix A
 *      2) b: ptr to struct R64MGF16, storing the matrix B
 *      3) c: ptr to struct RC64MGF16, storing the matrix C
 *      4) d: a 64-bit integer that encodes the diagonal matrix D. If the LSB
 *          is 1, then entry (0, 0) of D is 1. Otherwise 0.
 * return: void */
void
r64m_gf16_diag_fma(R64MGF16* restrict a, const R64MGF16* restrict b,
                   const RC64MGF16* restrict c, uint64_t d) {
    assert(r64m_gf16_rnum(a) == r64m_gf16_rnum(b));
#if defined(__AVX512F__)
    r64m_gf16_diag_fma_avx512(a, b, c, d);
#elif defined(__AVX2__)
    r64m_gf16_diag_fma_avx2(a, b, c, d);
#else
    r64m_gf16_diag_fma_naive(a, b, c, d);
#endif
}

/* usage: Given 2 R64MGF16 A and B, and a RC64MGF16 C, compute
 *      A - B * C and store the result back into A
 * params:
 *      1) a: ptr to struct R64MGF16, storing the matrix A
 *      2) b: ptr to struct R64MGF16, storing the matrix B
 *      3) c: ptr to struct RC64MGF16, storing the matrix C
 * return: void */
void
r64m_gf16_fms(R64MGF16* restrict a, const R64MGF16* restrict b,
              const RC64MGF16* restrict c) {
    // In GF(16), addition is subtraction
    r64m_gf16_fma(a, b, c);
}

/* usage: Given 2 R64MGF16 A and B, a RC64MGF16 C, and a 64x64 diagonal
 *      matrix D with coefficients either 1 and 0, compute A - B * C * D
 *      and store the result back into A
 * params:
 *      1) a: ptr to struct R64MGF16, storing the matrix A
 *      2) b: ptr to struct R64MGF16, storing the matrix B
 *      3) c: ptr to struct RC64MGF16, storing the matrix C
 *      4) d: a 64-bit integer that encodes the diagonal matrix D. If the LSB
 *          is 1, then entry (0, 0) of D is 1. Otherwise 0.
 * return: void */
void
r64m_gf16_fms_diag(R64MGF16* restrict a, const R64MGF16* restrict b,
                   const RC64MGF16* restrict c, uint64_t d) {
    // In GF(16), addition is subtraction
    r64m_gf16_fma_diag(a, b, c, d);
}

/* usage: Given 2 R64MGF16 A and B, replace a subset of columns of A with
 *      corresponding columns of B
 * params:
 *      1) a: ptr to struct R64MGF16, storing the matrix A
 *      2) b: ptr to struct R64MGF16, storing the matrix B
 *      3) di: a 64-bit integer that encodes which columns of A to keep. If
 *          the LSB is 1, then the first column of A is keep. If 0, then the
 *          first column of A is replaced by the first column  of B
 * return: void */
void
r64m_gf16_mixi(R64MGF16* restrict a, const R64MGF16* restrict b, uint64_t di) {
    uint64_t head = r64m_gf16_rnum(a) & ~0x1ULL;
    Grp64GF16* dst = r64m_gf16_raddr(a, 0);
    const Grp64GF16* src = r64m_gf16_raddr((R64MGF16*)b, 0);
    uint64_t ri = 0;
    for(; ri < head; ri += 2, src += 2, dst += 2)
        grp64_gf16_mixi_x2(dst, src, di);

    if(ri < r64m_gf16_rnum(a))
        grp64_gf16_mixi(dst, src, di);
}

/* usage: Given 2 R64MGF16 A and B, compute of A + B and store the result
 *      back into A
 * params:
 *      1) a: ptr to struct R64MGF16, storing the matrix A
 *      2) b: ptr to struct R64MGF16, storing the matrix B
 * return: void */
void
r64m_gf16_addi(R64MGF16* restrict a, const R64MGF16* restrict b) {
    assert(r64m_gf16_rnum(a) == r64m_gf16_rnum(b));
    Grp64GF16* dst = r64m_gf16_raddr(a, 0);
    const Grp64GF16* src = r64m_gf16_raddr((R64MGF16*)b, 0);
    uint64_t head = r64m_gf16_rnum(a) & ~0x1ULL;
    uint64_t ri = 0;
    for(; ri < head; ri += 2, src += 2, dst += 2) {
        grp64_gf16_addi_x2(dst, src);
    }
    if(ri < r64m_gf16_rnum(a))
        grp64_gf16_addi(dst, src);
}
