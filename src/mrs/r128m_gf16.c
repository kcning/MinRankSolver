#include "r128m_gf16.h"
#include <stdint.h>
#include <string.h>
#include <stdalign.h>

/* ========================================================================
 * struct R128MGF16 definition
 * ======================================================================== */

struct R128MGF16 {
    uint64_t rnum; // uint32_t will cause padding. Might as well use 64-bit
    // aligned to 64-byte boundary for AVX512
    alignas(64) Grp128GF16 rows[];
};

/* ========================================================================
 * function implementations
 * ======================================================================== */

/* usage: Compute the size of memory needed for R128MGF16
 * params:
 *      1) rnum: number of rows
 * return: size of memory needed in bytes */
uint64_t
r128m_gf16_memsize(uint32_t rnum) {
    return sizeof(R128MGF16) + sizeof(Grp128GF16) * rnum;
}

/* usage: Create a R128MGF16 matrix. The matrix is not initialized
 * params:
 *      1) rnum: number of rows
 * return: ptr to struct R128MGF16. NULL on faliure */
R128MGF16*
r128m_gf16_create(uint32_t rnum) {
    static_assert(sizeof(Grp128GF16) == 64, "size of Grp128GF16 is not 64");
    // aligned to 64-byte boundary for AVX512
    R128MGF16* m = aligned_alloc(64, r128m_gf16_memsize(rnum));
    if(!m)
        return NULL;

    m->rnum = rnum;
    return m;
}

/* usage: Release a struct R128MGF16
 * params:
 *      1) m: ptr to struct R128MGF16
 * return: void */
void
r128m_gf16_free(R128MGF16* m) {
    free(m);
}

/* usage: Given a R128MGF16 matrix, return the number of rows
 * params:
 *      1) m: ptr to a struct R128MGF16
 * return: the number of rows */
uint32_t
r128m_gf16_rnum(const R128MGF16* m) {
    return m->rnum;
}

/* usage: Given a R128MGF16 matrix and row index, return the address
 *      to the row
 * params:
 *      1) m: ptr to a struct R128MGF16
 *      2) i: index of the row, starting from 0
 * return: address to the i-th row */
Grp128GF16*
r128m_gf16_raddr(R128MGF16* m, uint32_t i) {
    return m->rows + i;
}

/* usage: Given a R128MGF16 matrix and both row and column indices, return
 *      the coefficient of the given entry.
 * params:
 *      1) m: ptr to a struct R128MGF16
 *      2) ri: index of the row, starting from 0
 *      3) ci: index of the column, from 0 ~ 63
 * return: coefficient of the entry */
gf16_t
r128m_gf16_at(const R128MGF16* m, uint32_t ri, uint32_t ci) {
    return grp128_gf16_at(r128m_gf16_raddr((R128MGF16*) m, ri), ci);
}

/* usage: Given a R128MGF16 matrix and both row and column indices, set
 *      the coefficient of the target entry to the given value.
 * params:
 *      1) m: ptr to a struct R128MGF16
 *      2) ri: index of the row, starting from 0
 *      3) ci: index of the column, from 0 ~ 63
 *      4) v: the new coefficient
 * return: void */
void
r128m_gf16_set_at(R128MGF16* m, uint64_t ri, uint64_t ci, gf16_t v) {
    grp128_gf16_set_at(r128m_gf16_raddr(m, ri), ci, v);
}

/* usage: Reset a struct R128MGF16 to zero matrix
 * params:
 *      1) m: ptr to a struct R128MGF16
 * return: void */
void
r128m_gf16_zero(R128MGF16* const m) {
    memset(m->rows, 0x0, sizeof(Grp128GF16) * r128m_gf16_rnum(m));
}

/* usage: Given a struct R128MGF16, populate it with random coefficients.
 * params:
 *      1) m: ptr to a struct R128MGF16
 * return: void */
void
r128m_gf16_rand(R128MGF16* m) {
    for(uint32_t i = 0; i < r128m_gf16_rnum(m); ++i)
        grp128_gf16_rand(r128m_gf16_raddr(m, i));
}

/* usage: Given 2 struct R128MGF16, copy the 2nd R128MGF16 into the 1st one
 * params:
 *      1) dst: ptr to the struct R128MGF16 to copy to
 *      2) src: ptr to the struct R128MGF16 to copy from
 * return: void */
void
r128m_gf16_copy(R128MGF16* restrict dst, const R128MGF16* restrict src) {
    assert(r128m_gf16_rnum(dst) == r128m_gf16_rnum(src));
    memcpy(dst->rows, src->rows, sizeof(Grp128GF16) * r128m_gf16_rnum(dst));
}

#if defined(__AVX512F__)

void
r128m_gf16_gramian_avx512(const R128MGF16* restrict m, RC128MGF16* restrict p) {
    const Grp128GF16* m_row = r128m_gf16_raddr((R128MGF16*) m, 0);
    const __m512i v_1st = _mm512_load_si512(m_row);
    Grp128GF16* dst = rc128m_gf16_raddr(p, 0);
    for(uint32_t i = 0; i < 128; i += 2, dst += 2) {
        __m512i p0 = grp128_gf16_mul_scalar_bs_avx512(v_1st, m_row, i);
        __m512i p1 = grp128_gf16_mul_scalar_bs_avx512(v_1st, m_row, i + 1);
        _mm512_store_si512(dst, p0);
        _mm512_store_si512(dst + 1, p1);
    }
    ++m_row;

    for(uint64_t ri = 1; ri < r128m_gf16_rnum(m); ++ri, ++m_row) {
        Grp128GF16* dst = rc128m_gf16_raddr(p, 0);
        __m512i v = _mm512_load_si512(m_row);
        for(uint32_t i = 0; i < 128; i += 2, dst += 2) {
            __m512i p0 = grp128_gf16_mul_scalar_bs_avx512(v, m_row, i);
            __m512i p1 = grp128_gf16_mul_scalar_bs_avx512(v, m_row, i + 1);
            __m512i d0 = _mm512_load_si512(dst);
            __m512i d1 = _mm512_load_si512(dst + 1);
            _mm512_store_si512(dst, _mm512_xor_si512(d0, p0));
            _mm512_store_si512(dst + 1, _mm512_xor_si512(d1, p1));
        }
    }
}

#elif defined(__AVX2__)

void
r128m_gf16_gramian_avx2(const R128MGF16* restrict m, RC128MGF16* restrict p) {
    const Grp128GF16* m_row = r128m_gf16_raddr((R128MGF16*) m, 0);
    __m256i* maddr = (__m256i*) m_row->b;
    const __m256i v0_1st = _mm256_load_si256(maddr);
    const __m256i v1_1st = _mm256_load_si256(maddr + 1);
    __m256i* dst = (__m256i*) rc128m_gf16_raddr(p, 0);
    for(uint32_t i = 0; i < 128; i += 2, dst += 4) {
        __m256i p0, p1, p2, p3;
        p0 = grp128_gf16_mul_scalar_bs_avx2(&p1, v0_1st, v1_1st, m_row, i);
        p2 = grp128_gf16_mul_scalar_bs_avx2(&p3, v0_1st, v1_1st, m_row, i + 1);
        _mm256_store_si256(dst, p0);
        _mm256_store_si256(dst + 1, p1);
        _mm256_store_si256(dst + 2, p2);
        _mm256_store_si256(dst + 3, p3);
    }
    ++m_row;

    for(uint64_t ri = 1; ri < r128m_gf16_rnum(m); ++ri, ++m_row) {
        maddr = (__m256i*) m_row->b;
        const __m256i v0 = _mm256_load_si256(maddr);
        const __m256i v1 = _mm256_load_si256(maddr + 1);
        dst = (__m256i*) rc128m_gf16_raddr(p, 0);
        for(uint32_t i = 0; i < 128; i += 2, dst += 4) {
            __m256i p0, p1, p2, p3;
            p0 = grp128_gf16_mul_scalar_bs_avx2(&p1, v0, v1, m_row, i);
            p2 = grp128_gf16_mul_scalar_bs_avx2(&p3, v0, v1, m_row, i + 1);
            __m256i d0 = _mm256_load_si256(dst);
            __m256i d1 = _mm256_load_si256(dst + 1);
            __m256i d2 = _mm256_load_si256(dst + 2);
            __m256i d3 = _mm256_load_si256(dst + 3);
            _mm256_store_si256(dst, _mm256_xor_si256(d0, p0));
            _mm256_store_si256(dst + 1, _mm256_xor_si256(d1, p1));
            _mm256_store_si256(dst + 2, _mm256_xor_si256(d2, p2));
            _mm256_store_si256(dst + 3, _mm256_xor_si256(d3, p3));
        }
    }
}

#else

void
r128m_gf16_gramian_naive(const R128MGF16* restrict m, RC128MGF16* restrict p) {
    rc128m_gf16_zero(p);
    for(uint64_t ri = 0; ri < r128m_gf16_rnum(m); ++ri) {
        const Grp128GF16* m_row = r128m_gf16_raddr((R128MGF16*) m, ri);

        Grp128GF16* dst = rc128m_gf16_raddr(p, 0);
        for(uint32_t i = 0; i < 128; i += 2, dst += 2) {
            grp128_gf16_fmaddi_scalar_bs(dst, m_row, m_row, i);
            grp128_gf16_fmaddi_scalar_bs(dst + 1, m_row, m_row, i + 1);
        }
    }
}

#endif

/* usage: Given a R128MGF16 matrix m and a container, compute the Gramian
 *      matrix of m, i.e. m.transpose() * m, and store it into the container.
 *      Note that the product has dimension 128x128.
 * params:
 *      1) m: ptr to a struct R128MGF16
 *      2) p: ptr to a struct RC128MGF16, container for the result
 * return: void */
void
r128m_gf16_gramian(const R128MGF16* restrict m, RC128MGF16* restrict p) {
#if defined(__AVX512F__)
    r128m_gf16_gramian_avx512(m, p);
#elif defined(__AVX2__)
    r128m_gf16_gramian_avx2(m, p);
#else
    r128m_gf16_gramian_naive(m, p);
#endif
}

/* usage: Given a R128MGF16, find the columns that are fully zero
 * params:
 *      1) m: ptr to struct R128MGF16
 *      2) out: ptr to a uint128_t which on return encodes zero columns. If the
 *          first column is fully zero, the LSB is set to 1, and so on.
 * return: void */
void
r128m_gf16_zc_pos(const R128MGF16* restrict m, uint128_t* restrict out) {
    uint128_t_max(out);
    for(uint32_t i = 0; i < r128m_gf16_rnum(m); ++i) {
        const Grp128GF16* row = r128m_gf16_raddr((R128MGF16*)m, i);
        uint128_t tmp; grp128_gf16_zpos(&tmp, row);
        uint128_t_andi(out, &tmp);
        if(uint128_t_is_zero(out))
            break;
    }
}

/* usage: Given a R128MGF16, find the columns whose selected rows are fully zero
 * params:
 *      1) m: ptr to struct R128MGF16
 *      2) ridxs: a uint32_t array that stores the indices of the selected
 *          rows
 *      3) sz: size of ridxs
 *      4) out: ptr to a uint128_t which on return encodes zero columns. If the
 *          first column is fully zero, the LSB is set to 1, and so on.
 * return: void */
void
r128m_gf16_subset_zc_pos(const R128MGF16* restrict m,
                         const uint32_t* restrict ridxs, uint32_t sz,
                         uint128_t* restrict out) {
    uint128_t_max(out);
    for(uint32_t i = 0; i < sz; ++i) {
        uint32_t ri = ridxs[i];
        const Grp128GF16* row = r128m_gf16_raddr((R128MGF16*)m, ri);
        uint128_t tmp; grp128_gf16_zpos(&tmp, row);
        uint128_t_andi(out, &tmp);
        if(uint128_t_is_zero(out))
            break;
    }
}

/* usage: Given a R128MGF16, find the columns that are not fully zero
 * params:
 *      1) m: ptr to struct R128MGF16
 *      2) out: ptr to a uint128_t which on return encodes non-zero columns. If
 *          the first column is not fully zero, the LSB is set to 1, and so on.
 * return: void */
void
r128m_gf16_nzc_pos(const R128MGF16* restrict m, uint128_t* restrict out) {
    r128m_gf16_zc_pos(m, out);
    uint128_t_negi(out);
}

#if defined(__AVX512F__)

void
r128m_gf16_fma_avx512(R128MGF16* restrict a, const R128MGF16* restrict b,
                      const RC128MGF16* restrict c) {
    Grp128GF16* dst = r128m_gf16_raddr(a, 0);
    for(uint32_t i = 0; i < r128m_gf16_rnum(a); ++i, ++dst) {
        const Grp128GF16* b_row = r128m_gf16_raddr((R128MGF16*) b, i);
        const Grp128GF16* src = rc128m_gf16_raddr((RC128MGF16*) c, 0);
        __m512i prod = _mm512_load_si512(dst);
        for(uint32_t j = 0; j < 128; j += 2, src += 2) {
            __m512i v0 = _mm512_load_si512(src);
            __m512i v1 = _mm512_load_si512(src + 1);
            __m512i p0 = grp128_gf16_mul_scalar_bs_avx512(v0, b_row, j);
            __m512i p1 = grp128_gf16_mul_scalar_bs_avx512(v1, b_row, j + 1);
            prod = _mm512_xor_si512(prod, p0);
            prod = _mm512_xor_si512(prod, p1);
        }
        _mm512_store_si512(dst, prod);
    }
}

#elif defined(__AVX2__)

void
r128m_gf16_fma_avx2(R128MGF16* restrict a, const R128MGF16* restrict b,
                    const RC128MGF16* restrict c) {
    __m256i* dst = (__m256i*) r128m_gf16_raddr(a, 0);
    for(uint32_t i = 0; i < r128m_gf16_rnum(a); ++i, dst += 2) {
        const Grp128GF16* b_row = r128m_gf16_raddr((R128MGF16*) b, i);
        const __m256i* src = (__m256i*) rc128m_gf16_raddr((RC128MGF16*)c, 0);
        __m256i prod0 = _mm256_load_si256(dst);
        __m256i prod1 = _mm256_load_si256(dst + 1);
        for(uint32_t j = 0; j < 128; j += 2, src += 4) {
            __m256i v0 = _mm256_load_si256(src);
            __m256i v1 = _mm256_load_si256(src + 1);
            __m256i v2 = _mm256_load_si256(src + 2);
            __m256i v3 = _mm256_load_si256(src + 3);
            __m256i p0, p1, p2, p3;
            p0 = grp128_gf16_mul_scalar_bs_avx2(&p1, v0, v1, b_row, j);
            p2 = grp128_gf16_mul_scalar_bs_avx2(&p3, v2, v3, b_row, j + 1);
            prod0 = _mm256_xor_si256(prod0, p0);
            prod1 = _mm256_xor_si256(prod1, p1);
            prod0 = _mm256_xor_si256(prod0, p2);
            prod1 = _mm256_xor_si256(prod1, p3);
        }
        _mm256_store_si256(dst, prod0);
        _mm256_store_si256(dst + 1, prod1);
    }
}

#else

void
r128m_gf16_fma_naive(R128MGF16* restrict a, const R128MGF16* restrict b,
                     const RC128MGF16* restrict c) {
    for(uint32_t i = 0; i < r128m_gf16_rnum(a); ++i) {
        const Grp128GF16* b_row = r128m_gf16_raddr((R128MGF16*) b, i);
        Grp128GF16* dst = r128m_gf16_raddr(a, i);
        const Grp128GF16* src = rc128m_gf16_raddr((RC128MGF16*)c, 0);
        for(uint32_t j = 0; j < 128; j += 2, src += 2) {
            grp128_gf16_fmaddi_scalar_bs(dst, src, b_row, j);
            grp128_gf16_fmaddi_scalar_bs(dst, src + 1, b_row, j + 1);
        }
    }
}

#endif

/* usage: Given 2 R128MGF16 A and B, and a RC128MGF16 C, compute
 *      A + B * C and store the result back into A
 * params:
 *      1) a: ptr to struct R128MGF16, storing the matrix A
 *      2) b: ptr to struct R128MGF16, storing the matrix B
 *      3) c: ptr to struct RC128MGF16, storing the matrix C
 * return: void */
void
r128m_gf16_fma(R128MGF16* restrict a, const R128MGF16* restrict b,
               const RC128MGF16* restrict c) {
    assert(r128m_gf16_rnum(a) == r128m_gf16_rnum(b));
#if defined(__AVX512F__)
    r128m_gf16_fma_avx512(a, b, c);
#elif defined(__AVX2__)
    r128m_gf16_fma_avx2(a, b, c);
#else
    r128m_gf16_fma_naive(a, b, c);
#endif
}

#if defined(__AVX512F__)
void
r128m_gf16_fma_diag_internal_avx512(R128MGF16* restrict a,
                                    const R128MGF16* restrict b,
                                    const RC128MGF16* restrict c,
                                    const __m512i vd) {
    Grp128GF16* dst = r128m_gf16_raddr(a, 0);
    for(uint32_t i = 0; i < r128m_gf16_rnum(a); ++i, ++dst) {
        const Grp128GF16* b_row = r128m_gf16_raddr((R128MGF16*) b, i);
        const Grp128GF16* src = rc128m_gf16_raddr((RC128MGF16*) c, 0);
        __m512i prod = _mm512_setzero_si512();
        for(uint32_t j = 0; j < 128; j += 2, src += 2) {
            __m512i v0 = _mm512_load_si512(src);
            __m512i v1 = _mm512_load_si512(src + 1);
            __m512i p0 = grp128_gf16_mul_scalar_bs_avx512(v0, b_row, j);
            __m512i p1 = grp128_gf16_mul_scalar_bs_avx512(v1, b_row, j + 1);
            prod = _mm512_xor_si512(prod, p0);
            prod = _mm512_xor_si512(prod, p1);
        }
        prod = _mm512_and_si512(prod, vd);
        __m512i d = _mm512_load_si512(dst);
        _mm512_store_si512(dst, _mm512_xor_si512(prod, d));
    }
}

#elif defined(__AVX2__)

void
r128m_gf16_fma_diag_internal_avx2(R128MGF16* restrict a,
                                  const R128MGF16* restrict b,
                                  const RC128MGF16* restrict c,
                                  const __m256i vd) {
    for(uint32_t i = 0; i < r128m_gf16_rnum(a); ++i) {
        const Grp128GF16* b_row = r128m_gf16_raddr((R128MGF16*) b, i);
        __m256i* dst = (__m256i*) r128m_gf16_raddr(a, i);
        const __m256i* src = (__m256i*) rc128m_gf16_raddr((RC128MGF16*)c, 0);
        __m256i prod0 = _mm256_setzero_si256();
        __m256i prod1 = _mm256_setzero_si256();
        for(uint32_t j = 0; j < 128; j += 2, src += 4) {
            __m256i v0 = _mm256_load_si256(src);
            __m256i v1 = _mm256_load_si256(src + 1);
            __m256i v2 = _mm256_load_si256(src + 2);
            __m256i v3 = _mm256_load_si256(src + 3);
            __m256i p0, p1, p2, p3;
            p0 = grp128_gf16_mul_scalar_bs_avx2(&p1, v0, v1, b_row, j);
            p2 = grp128_gf16_mul_scalar_bs_avx2(&p3, v2, v3, b_row, j + 1);
            prod0 = _mm256_xor_si256(prod0, p0);
            prod1 = _mm256_xor_si256(prod1, p1);
            prod0 = _mm256_xor_si256(prod0, p2);
            prod1 = _mm256_xor_si256(prod1, p3);
        }
        prod0 = _mm256_and_si256(prod0, vd);
        prod1 = _mm256_and_si256(prod1, vd);
        __m256i d0 = _mm256_load_si256(dst);
        __m256i d1 = _mm256_load_si256(dst + 1);
        _mm256_store_si256(dst, _mm256_xor_si256(d0, prod0));
        _mm256_store_si256(dst + 1, _mm256_xor_si256(d1, prod1));
    }
}

#else

void
r128m_gf16_fma_diag_naive(R128MGF16* restrict a, const R128MGF16* restrict b,
                          const RC128MGF16* restrict c,
                          const uint128_t* restrict d) {
    for(uint32_t i = 0; i < r128m_gf16_rnum(a); ++i) {
        const Grp128GF16* b_row = r128m_gf16_raddr((R128MGF16*) b, i);
        Grp128GF16* dst = r128m_gf16_raddr(a, i);
        const Grp128GF16* src = rc128m_gf16_raddr((RC128MGF16*)c, 0);
        for(uint32_t j = 0; j < 128; j += 2, src += 2) {
            grp128_gf16_fmaddi_scalar_mask_bs(dst, src, b_row, j, d);
            grp128_gf16_fmaddi_scalar_mask_bs(dst, src + 1, b_row, j + 1, d);
        }
    }
}

#endif

/* usage: Given 2 R128MGF16 A and B, a RC128MGF16 C, and a 128x128 diagonal
 *      matrix D with coefficients either 1 and 0, compute A + B * C * D
 *      and store the result back into A
 * params:
 *      1) a: ptr to struct R128MGF16, storing the matrix A
 *      2) b: ptr to struct R128MGF16, storing the matrix B
 *      3) c: ptr to struct RC128MGF16, storing the matrix C
 *      4) d: ptr to a uint128_t which encodes the diagonal matrix D. If the
 *          LSB is 1, then entry (0, 0) of D is 1. Otherwise 0.
 * return: void */
void
r128m_gf16_fma_diag(R128MGF16* restrict a, const R128MGF16* restrict b,
                    const RC128MGF16* restrict c,
                    const uint128_t* restrict d) {
    assert(r128m_gf16_rnum(a) == r128m_gf16_rnum(b));
#if defined(__AVX512F__)
    __m512i vd = _mm512_castsi128_si512(_mm_load_si128((__m128i*)d));
    vd = _mm512_shuffle_i64x2(vd, vd, 0x0); // [mask, mask, mask, mask]
    r128m_gf16_fma_diag_internal_avx512(a, b, c, vd);
#elif defined(__AVX2__)
    // [mask, U]
    __m256i vd = _mm256_castsi128_si256(_mm_load_si128((__m128i*)d));
    vd = _mm256_permute2x128_si256(vd, vd, 0x0); // [mask, mask]
    r128m_gf16_fma_diag_internal_avx2(a, b, c, vd);
#else
    r128m_gf16_fma_diag_naive(a, b, c, d);
#endif
}

#if defined(__AVX2__)

/* usage: Given 2 R128MGF16 A and B, a RC128MGF16 C, and a 128x128 diagonal
 *      matrix D with coefficients either 1 and 0, compute A + B * C * D
 *      and store the result back into A
 * params:
 *      1) a: ptr to struct R128MGF16, storing the matrix A
 *      2) b: ptr to struct R128MGF16, storing the matrix B
 *      3) c: ptr to struct RC128MGF16, storing the matrix C
 *      4) d: a 256-bit register which encodes the diagonal matrix d. if the
 *          lsb is 1, then entry (0, 0) of d is 1. otherwise 0. the upper
 *          128 bits need to be duplicate of the lower 128 bits.
 * return: void */
void
r128m_gf16_fma_diag_avx2(R128MGF16* restrict a, const R128MGF16* restrict b,
                         const RC128MGF16* restrict c, const __m256i d) {
    assert(r128m_gf16_rnum(a) == r128m_gf16_rnum(b));
#if defined(__AVX512F__)
    __m512i vd = _mm512_castsi256_si512(d);
    vd = _mm512_shuffle_i64x2(vd, vd, 0x0); // [mask, mask, mask, mask]
    r128m_gf16_fma_diag_internal_avx512(a, b, c, vd);
#else
    r128m_gf16_fma_diag_internal_avx2(a, b, c, d);
#endif
}

#endif

#if defined(__AVX512F__)

void
r128m_gf16_diag_fma_internal_avx512(R128MGF16* restrict a,
                                    const R128MGF16* restrict b,
                                    const RC128MGF16* restrict c,
                                    const __m512i vd) {
    Grp128GF16* dst = r128m_gf16_raddr(a, 0);
    for(uint32_t i = 0; i < r128m_gf16_rnum(a); ++i, ++dst) {
        const Grp128GF16* b_row = r128m_gf16_raddr((R128MGF16*) b, i);
        const Grp128GF16* src = rc128m_gf16_raddr((RC128MGF16*) c, 0);
        __m512i prod = _mm512_and_si512(vd, _mm512_load_si512(dst));
        for(uint32_t j = 0; j < 128; j += 2, src += 2) {
            __m512i v0 = _mm512_load_si512(src);
            __m512i v1 = _mm512_load_si512(src + 1);
            __m512i p0 = grp128_gf16_mul_scalar_bs_avx512(v0, b_row, j);
            __m512i p1 = grp128_gf16_mul_scalar_bs_avx512(v1, b_row, j + 1);
            prod = _mm512_xor_si512(prod, p0);
            prod = _mm512_xor_si512(prod, p1);
        }
        _mm512_store_si512(dst, prod);
    }
}

#elif defined(__AVX2__)

void
r128m_gf16_diag_fma_internal_avx2(R128MGF16* restrict a,
                                  const R128MGF16* restrict b,
                                  const RC128MGF16* restrict c, const __m256i vd) {
    const Grp128GF16* b_row = r128m_gf16_raddr((R128MGF16*) b, 0);
    __m256i* dst = (__m256i*) r128m_gf16_raddr(a, 0);
    for(uint32_t i = 0; i < r128m_gf16_rnum(a); ++i, ++b_row, dst += 2) {
        __m256i prod0 = _mm256_load_si256(dst);
        __m256i prod1 = _mm256_load_si256(dst + 1);
        prod0 = _mm256_and_si256(prod0, vd);
        prod1 = _mm256_and_si256(prod1, vd);

        const __m256i* src = (__m256i*) rc128m_gf16_raddr((RC128MGF16*)c, 0);
        for(uint32_t j = 0; j < 128; j += 2, src += 4) {
            __m256i v0 = _mm256_load_si256(src);
            __m256i v1 = _mm256_load_si256(src + 1);
            __m256i v2 = _mm256_load_si256(src + 2);
            __m256i v3 = _mm256_load_si256(src + 3);
            __m256i p0, p1, p2, p3;
            p0 = grp128_gf16_mul_scalar_bs_avx2(&p1, v0, v1, b_row, j);
            p2 = grp128_gf16_mul_scalar_bs_avx2(&p3, v2, v3, b_row, j + 1);
            prod0 = _mm256_xor_si256(prod0, p0);
            prod1 = _mm256_xor_si256(prod1, p1);
            prod0 = _mm256_xor_si256(prod0, p2);
            prod1 = _mm256_xor_si256(prod1, p3);
        }
        _mm256_store_si256(dst, prod0);
        _mm256_store_si256(dst + 1, prod1);
    }
}

#else

void
r128m_gf16_diag_fma_naive(R128MGF16* restrict a, const R128MGF16* restrict b,
                          const RC128MGF16* restrict c,
                          const uint128_t* restrict d) {
    const Grp128GF16* b_row = r128m_gf16_raddr((R128MGF16*) b, 0);
    Grp128GF16* dst = r128m_gf16_raddr(a, 0);
    for(uint32_t i = 0; i < r128m_gf16_rnum(a); ++i, ++b_row, ++dst) {
        grp128_gf16_zero_subset(dst, d);
        const Grp128GF16* src = rc128m_gf16_raddr((RC128MGF16*)c, 0);
        for(uint32_t j = 0; j < 128; j += 2, src += 2) {
            grp128_gf16_fmaddi_scalar_bs(dst, src, b_row, j);
            grp128_gf16_fmaddi_scalar_bs(dst, src + 1, b_row, j + 1);
        }
    }
}

#endif

/* usage: Given 2 R128MGF16 A and B, a RC128MGF16 C, and a 128x128 diagonal
 *      matrix D with coefficients either 1 and 0, compute A * D + B * C
 *      and store the result back into A
 * params:
 *      1) a: ptr to struct R128MGF16, storing the matrix A
 *      2) b: ptr to struct R128MGF16, storing the matrix B
 *      3) c: ptr to struct RC128MGF16, storing the matrix C
 *      4) d: ptr to a uint128_t which encodes the diagonal matrix D. If the
 *          LSB is 1, then entry (0, 0) of D is 1. Otherwise 0.
 * return: void */
void
r128m_gf16_diag_fma(R128MGF16* restrict a, const R128MGF16* restrict b,
                    const RC128MGF16* restrict c,
                    const uint128_t* restrict d) {
    assert(r128m_gf16_rnum(a) == r128m_gf16_rnum(b));

#if defined(__AVX512F__)
    __m512i vd = _mm512_castsi128_si512(_mm_load_si128((__m128i*)d));
    vd = _mm512_shuffle_i64x2(vd, vd, 0x0); // [mask, mask, mask, mask]
    r128m_gf16_diag_fma_internal_avx512(a, b, c, vd);
#elif defined(__AVX2__)
    // [mask, U]
    __m256i vd = _mm256_castsi128_si256(_mm_load_si128((__m128i*)d));
    vd = _mm256_permute2x128_si256(vd, vd, 0x0); // [mask, mask]
    r128m_gf16_diag_fma_internal_avx2(a, b, c, vd);
#else
    r128m_gf16_diag_fma_naive(a, b, c, d);
#endif
}

#if defined(__AVX2__)

/* usage: Given 2 R128MGF16 A and B, a RC128MGF16 C, and a 128x128 diagonal
 *      matrix D with coefficients either 1 and 0, compute A * D + B * C
 *      and store the result back into A
 * params:
 *      1) a: ptr to struct R128MGF16, storing the matrix A
 *      2) b: ptr to struct R128MGF16, storing the matrix B
 *      3) c: ptr to struct RC128MGF16, storing the matrix C
 *      4) d: a 256-bit register which encodes the diagonal matrix d. if the
 *          lsb is 1, then entry (0, 0) of d is 1. otherwise 0. the upper
 *          128 bits need to be duplicate of the lower 128 bits.
 * return: void */
void
r128m_gf16_diag_fma_avx2(R128MGF16* restrict a, const R128MGF16* restrict b,
                         const RC128MGF16* restrict c,
                         const __m256i d) {
    assert(r128m_gf16_rnum(a) == r128m_gf16_rnum(b));
#if defined(__AVX512F__)
    __m512i vd = _mm512_castsi256_si512(d);
    vd = _mm512_shuffle_i64x2(vd, vd, 0x0); // [mask, mask, mask, mask]
    r128m_gf16_diag_fma_internal_avx512(a, b, c, vd);
#else
    r128m_gf16_diag_fma_internal_avx2(a, b, c, d);
#endif
}

#endif

/* usage: Given 2 R128MGF16 A and B, and a RC128MGF16 C, compute
 *      A - B * C and store the result back into A
 * params:
 *      1) a: ptr to struct R128MGF16, storing the matrix A
 *      2) b: ptr to struct R128MGF16, storing the matrix B
 *      3) c: ptr to struct RC128MGF16, storing the matrix C
 * return: void */
void
r128m_gf16_fms(R128MGF16* restrict a, const R128MGF16* restrict b,
               const RC128MGF16* restrict c) {
    // In GF(16), addition is subtraction
    r128m_gf16_fma(a, b, c);
}

/* usage: Given 2 R128MGF16 A and B, a RC128MGF16 C, and a 128x128 diagonal
 *      matrix D with coefficients either 1 and 0, compute A - B * C * D
 *      and store the result back into A
 * params:
 *      1) a: ptr to struct R128MGF16, storing the matrix A
 *      2) b: ptr to struct R128MGF16, storing the matrix B
 *      3) c: ptr to struct RC128MGF16, storing the matrix C
 *      4) d: ptr to a uint128_t which encodes the diagonal matrix D. If the
 *          LSB is 1, then entry (0, 0) of D is 1. Otherwise 0.
 * return: void */
void
r128m_gf16_fms_diag(R128MGF16* restrict a, const R128MGF16* restrict b,
                    const RC128MGF16* restrict c,
                    const uint128_t* restrict d) {
    // In GF(16), addition is subtraction
    r128m_gf16_fma_diag(a, b, c, d);
}

#if defined(__AVX2__)

/* usage: Given 2 R128MGF16 A and B, a RC128MGF16 C, and a 128x128 diagonal
 *      matrix D with coefficients either 1 and 0, compute A - B * C * D
 *      and store the result back into A
 * params:
 *      1) a: ptr to struct R128MGF16, storing the matrix A
 *      2) b: ptr to struct R128MGF16, storing the matrix B
 *      3) c: ptr to struct RC128MGF16, storing the matrix C
 *      4) d: a 256-bit register which encodes the diagonal matrix d. if the
 *          lsb is 1, then entry (0, 0) of d is 1. otherwise 0. the upper
 *          128 bits need to be duplicate of the lower 128 bits.
 * return: void */
void
r128m_gf16_fms_diag_avx2(R128MGF16* restrict a, const R128MGF16* restrict b,
                         const RC128MGF16* restrict c, const __m256i d) {
    // In GF(16), addition is subtraction
    r128m_gf16_fma_diag_avx2(a, b, c, d);
}

#endif

/* usage: Given 2 R128MGF16 A and B, replace a subset of columns of A with
 *      corresponding columns of B
 * params:
 *      1) a: ptr to struct R128MGF16, storing the matrix A
 *      2) b: ptr to struct R128MGF16, storing the matrix B
 *      3) di: ptr to a uint128_t which encodes the diagonal matrix D. If the
 *          LSB is 1, then entry (0, 0) of D is 1. Otherwise 0.
 * return: void */
void
r128m_gf16_mixi(R128MGF16* restrict a, const R128MGF16* restrict b,
                const uint128_t* restrict di) {
    uint32_t head = r128m_gf16_rnum(a) & ~0x1U;
    Grp128GF16* dst = r128m_gf16_raddr(a, 0);
    const Grp128GF16* src = r128m_gf16_raddr((R128MGF16*)b, 0);
    uint32_t ri = 0;
    for(; ri < head; ri += 2, src += 2, dst += 2) {
        grp128_gf16_mixi(dst, src, di);
        grp128_gf16_mixi(dst + 1, src + 1, di);
    }

    if(ri < r128m_gf16_rnum(a))
        grp128_gf16_mixi(dst, src, di);
}

#if defined(__AVX2__)

/* usage: Given 2 R128MGF16 A and B, replace a subset of columns of A with
 *      corresponding columns of B
 * params:
 *      1) a: ptr to struct R128MGF16, storing the matrix A
 *      2) b: ptr to struct R128MGF16, storing the matrix B
 *      3) di: a 256 bit register which encodes the diagonal matrix D. If the
 *          LSB is 1, then entry (0, 0) of D is 1. Otherwise 0. The upper
 *          128 bits need to the duplicate of the lower 128 bits.
 * return: void */
void
r128m_gf16_mixi_avx2(R128MGF16* restrict a, const R128MGF16* restrict b,
                     const __m256i di) {
    uint32_t head = r128m_gf16_rnum(a) & ~0x1U;
    Grp128GF16* dst = r128m_gf16_raddr(a, 0);
    const Grp128GF16* src = r128m_gf16_raddr((R128MGF16*)b, 0);
    uint32_t ri = 0;
    for(; ri < head; ri += 2, src += 2, dst += 2) {
        grp128_gf16_mixi_avx2(dst, src, di);
        grp128_gf16_mixi_avx2(dst + 1, src + 1, di);
    }

    if(ri < r128m_gf16_rnum(a))
        grp128_gf16_mixi_avx2(dst, src, di);
}

#endif

/* usage: Given 2 R128MGF16 A and B, compute of A + B and store the result
 *      back into A
 * params:
 *      1) a: ptr to struct R128MGF16, storing the matrix A
 *      2) b: ptr to struct R128MGF16, storing the matrix B
 * return: void */
void
r128m_gf16_addi(R128MGF16* restrict a, const R128MGF16* restrict b) {
    assert(r128m_gf16_rnum(a) == r128m_gf16_rnum(b));
    Grp128GF16* dst = r128m_gf16_raddr(a, 0);
    const Grp128GF16* src = r128m_gf16_raddr((R128MGF16*)b, 0);
    uint64_t head = r128m_gf16_rnum(a) & ~0x1ULL;
    uint64_t ri = 0;
    for(; ri < head; ri += 2, src += 2, dst += 2) {
        grp128_gf16_addi(dst, src);
        grp128_gf16_addi(dst + 1, src + 1);
    }
    if(ri < r128m_gf16_rnum(a))
        grp128_gf16_addi(dst, src);
}
