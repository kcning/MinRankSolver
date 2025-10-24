#include "rc128m_gf16.h"
#include "grp128_gf16.h"
#include <string.h>
#include <stdlib.h>
#include <stdalign.h>

/* ========================================================================
 * struct RC128MGF16 definition
 * ======================================================================== */

struct RC128MGF16 {
    // alignment is the same as Grp128GF16
    Grp128GF16 rows[128];
};

/* ========================================================================
 * function implementations
 * ======================================================================== */

/* usage: compute the size of memory needed for struct RC128MGF16
 * return: size in bytes */
uint64_t
rc128m_gf16_memsize(void) {
    static_assert(sizeof(RC128MGF16) == (128 * 4 / 8 * 128),
                  "Incorrect size for RC128MGF16. Check padding");
    return sizeof(RC128MGF16);
}

/* usage: return the addr of the selected row in a struct RC128MGF16
 * params:
 *      1) m: ptr to struct RC128MGF16
 *      2) i: index of the row
 * return: the row addr as a ptr to struct Grp128GF16 */
Grp128GF16*
rc128m_gf16_raddr(RC128MGF16* m, uint32_t i) {
    return m->rows + i;
}

/* usage: swap 2 rows in a struct RC128MGF16
 * params:
 *      1) m: ptr to struct RC128MGF16
 *      2) i: index of the 1st row
 *      3) j: index of the 2nd row
 * return: void */
void
rc128m_gf16_swap_rows(RC128MGF16* m, uint32_t i, uint32_t j) {
    assert(i < 128 && j < 128);
    Grp128GF16* r0 = rc128m_gf16_raddr(m, i);
    Grp128GF16* r1 = rc128m_gf16_raddr(m, j);
#if defined(__AVX512F__)
    __m512i v0 = _mm512_load_si512(r0);
    __m512i v1 = _mm512_load_si512(r1);
    _mm512_store_si512(r0, v1);
    _mm512_store_si512(r1, v0);
#elif defined(__AVX__)
    __m256i* s0 = (__m256i*) r0->b;
    __m256i* s1 = (__m256i*) r1->b;
    __m256i v0a = _mm256_load_si256(s0);
    __m256i v0b = _mm256_load_si256(s0 + 1);
    __m256i v1a = _mm256_load_si256(s1);
    __m256i v1b = _mm256_load_si256(s1 + 1);
    _mm256_store_si256(s0, v1a);
    _mm256_store_si256(s0 + 1, v1b);
    _mm256_store_si256(s1, v0a);
    _mm256_store_si256(s1 + 1, v0b);
#else
    uint128_t_swap(r0->b, r1->b);
    uint128_t_swap(r0->b + 1, r1->b + 1);
    uint128_t_swap(r0->b + 2, r1->b + 2);
    uint128_t_swap(r0->b + 3, r1->b + 3);
#endif
}

/* usage: return the selected element in a struct RC128MGF16
 * params:
 *      1) m: ptr to struct RC128MGF16
 *      2) i: index of the row
 *      3) j: index of the column
 * return: the (i, j)-th element of the matrix */
gf16_t
rc128m_gf16_at(const RC128MGF16* m, uint32_t i, uint32_t j) {
    assert(i < 128 && j < 128);
    return grp128_gf16_at(rc128m_gf16_raddr((RC128MGF16*)m, i), j);
}

/* usage: set the selected element in a struct RC128MGF16 to the given value
 * params:
 *      1) m: ptr to struct RC128MGF16
 *      2) i: index of the row
 *      3) j: index of the column
 *      4) v: the new value
 * return: void */
void
rc128m_gf16_set_at(RC128MGF16* m, uint32_t i, uint32_t j, gf16_t v) {
    assert(i < 128 && j < 128);
    assert(v <= GF16_MAX);
    grp128_gf16_set_at(rc128m_gf16_raddr(m, i), j, v);
}

/* usage: Create a RC128MGF16 container. Note the matrix is not initialized.
 * params: none
 * return: a ptr to struct RC128MGF16. On failure, return NULL */
RC128MGF16*
rc128m_gf16_create(void) {
    // align to 64-byte boundary for AVX512
    RC128MGF16* m = aligned_alloc(64, sizeof(RC128MGF16));
    return m;
}

/* usage: Create an array of RC128MGF16. None of the  matrices is initialized.
 * params:
 *      1) sz: size of the array
 * return: a ptr to struct RC128MGF16. On failure, return NULL */
RC128MGF16*
rc128m_gf16_arr_create(uint32_t sz) {
    // align to 64-byte boundary for AVX512
    RC128MGF16* m = aligned_alloc(64, sizeof(RC128MGF16) * sz);
    return m;
}

/* usage: given an array of RC128MGF16, return a ptr to its i-th entry.
 * params:
 *      1) m: ptr to an array of RC128MGF16
 *      2) i: index of the entry
 * return: a struct RC128MGF16 ptr to the i-th entry */
RC128MGF16*
rc128m_gf16_arr_at(RC128MGF16* m, uint32_t i) {
    return m + i;
}

/* usage: Release a struct RC128MGF16
 * params:
 *      1) m: ptr to a struct RC128MGF16
 * return: void */
void
rc128m_gf16_free(RC128MGF16* m) {
    free(m);
}

/* usage: Release an array of struct RC128MGF16
 * params:
 *      1) m: ptr to a struct RC128MGF16
 * return: void */
void
rc128m_gf16_arr_free(RC128MGF16* m) {
    free(m);
}

/* usage: Given a struct RC128MGF16, populate it with random coefficients.
 * params:
 *      1) m: ptr to a struct RC128MGF16
 * return: void */
void
rc128m_gf16_rand(RC128MGF16* m) {
    for(uint32_t i = 0; i < 128; ++i) {
        grp128_gf16_rand(rc128m_gf16_raddr(m, i));
    }
}

/* usage: Reset a struct RC128MGF16 to zero matrix
 * params:
 *      1) m: ptr to a struct RC128MGF16
 * return: void */
void
rc128m_gf16_zero(RC128MGF16* m) {
    memset(m, 0x0, sizeof(RC128MGF16));
}

/* usage: given a struct RC128MGF16, set a subset of rows and columns to zero
 * params:
 *      1) m: ptr to a struct RC128MGF16
 *      2) d: ptr to uint128_t that encodes which rows and columns to keep.
 *          If the i-th bit is set, then the i-th row and column are kept.
 *          Otherwise, the i-th row and i-th column are cleared to zero.
 * return: void */
void
rc128m_gf16_zero_subset_rc(RC128MGF16* m, const uint128_t* restrict d) {
    uint128_t tmp; uint128_t_neg(&tmp, d);

    for(uint32_t i = 0; i < 128; ++i) {
        Grp128GF16* row = rc128m_gf16_raddr(m, i);
        grp128_gf16_zero_subset(row, d);
    }

    uint8_t sbidxs[128];
    uint32_t sbnum = uint128_t_sbpos(&tmp, sbidxs);
    for(uint32_t i = 0; i < sbnum; ++i) {
        Grp128GF16* row = rc128m_gf16_raddr(m, sbidxs[i]);
        grp128_gf16_zero(row);
    }
}

#if defined(__AVX2__)

/* usage: given a struct RC128MGF16, set a subset of rows and columns to zero
 * params:
 *      1) m: ptr to a struct RC128MGF16
 *      2) d: a 256-bit register that encodes which rows and columns to keep.
 *          If the i-th bit is set, then the i-th row and column are kept.
 *          Otherwise, the i-th row and i-th column are cleared to zero. Only
 *          the lower 128 bits need to be valid. The upper 128 bits need to
 *          be duplicate of the lower 128 bits.
 * return: void */
void
rc128m_gf16_zero_subset_rc_avx2(RC128MGF16* m, const __m256i mask) {
    uint128_t tmp;
    // negate m because usually m is almost full 1's in its lower 128 bits
    _mm_storeu_si128((__m128i*) &tmp, _mm256_castsi256_si128(~mask));

    for(uint32_t i = 0; i < 128; ++i) {
        Grp128GF16* row = rc128m_gf16_raddr(m, i);
        grp128_gf16_zero_subset_avx2(row, mask);
    }

    uint8_t sbidxs[128];
    uint32_t sbnum = uint128_t_sbpos(&tmp, sbidxs);
    for(uint32_t i = 0; i < sbnum; ++i) {
        Grp128GF16* row = rc128m_gf16_raddr(m, sbidxs[i]);
        grp128_gf16_zero(row);
    }
}

#endif

/* usage: Copy a struct RC128MGF16
 * params:
 *      1) dst: ptr to a struct RC128MGF16 for the copy
 *      2) src: ptr to a struct RC128MGF16. The source
 * return: void */
void
rc128m_gf16_copy(RC128MGF16* restrict dst, const RC128MGF16* restrict src) {
    memcpy(dst, src, sizeof(RC128MGF16));
}

/* usage: Reset a struct RC128MGF16 to identity matrix
 * params:
 *      1) m: ptr to a struct RC128MGF16
 * return: void */
void
rc128m_gf16_identity(RC128MGF16* m) {
#if defined(__AVX512F__)
    __m512i v0 = _mm512_set_epi64(0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x1);
    __m512i v64 = _mm512_bslli_epi128(v0, 8);// 0x0 0x0 0x0 0x0 0x0 0x0 0x1 0x0
    Grp128GF16* row = rc128m_gf16_raddr(m, 0);
    for(uint32_t i = 0; i < 64; ++i, ++row) {
        _mm512_store_si512(row, v0);
        v0 = _mm512_slli_epi64(v0, 1);
    }
    for(uint32_t i = 0; i < 64; ++i, ++row) {
        _mm512_store_si512(row, v64);
        v64 = _mm512_slli_epi64(v64, 1);
    }
#elif defined(__AVX2__)
    __m256i vlo = _mm256_set_epi64x(0x0, 0x0, 0x0, 0x1);
    __m256i vlo64 = _mm256_slli_si256(vlo, 8); // 0x0, 0x0, 0x1, 0x0
    __m256i vhi = _mm256_setzero_si256();
    Grp128GF16* row = rc128m_gf16_raddr(m, 0);
    for(uint32_t i = 0; i < 64; ++i, ++row) {
        __m256i* dst = (__m256i*) row;
        _mm256_store_si256(dst, vlo);
        _mm256_store_si256(dst + 1, vhi);
        vlo = _mm256_slli_epi64(vlo, 1);
    }
    for(uint32_t i = 0; i < 64; ++i, ++row) {
        __m256i* dst = (__m256i*) row;
        _mm256_store_si256(dst, vlo64);
        _mm256_store_si256(dst + 1, vhi);
        vlo64 = _mm256_slli_epi64(vlo64, 1);
    }
#else
    rc128m_gf16_zero(m);
    for(uint32_t i = 0; i < 128; ++i) {
        Grp128GF16* row = rc128m_gf16_raddr(m, i);
        uint128_t_toggle_at(row->b, i);
    }
#endif
}

static inline void
rc128m_gf16_row_reduc(Grp128GF16* restrict dst_row,
                      const Grp128GF16* restrict pvt_row,
                      Grp128GF16* restrict dst_inv_row,
                      const Grp128GF16* restrict inv_row, uint32_t pvt_idx) {
#if defined(__AVX512F__)
    __m512i v0 = _mm512_load_si512(pvt_row);
    __m512i v1 = _mm512_load_si512(inv_row);
    __m512i p0 = grp128_gf16_mul_scalar_bs_avx512(v0, dst_row, pvt_idx);
    __m512i p1 = grp128_gf16_mul_scalar_bs_avx512(v1, dst_row, pvt_idx);
    __m512i d0 = _mm512_load_si512(dst_row);
    __m512i d1 = _mm512_load_si512(dst_inv_row);
    _mm512_store_si512(dst_row, _mm512_xor_si512(d0, p0));
    _mm512_store_si512(dst_inv_row, _mm512_xor_si512(d1, p1));
#elif defined(__AVX2__)
    __m256i* src0 = (__m256i*) pvt_row;
    __m256i* src1 = (__m256i*) inv_row;
    __m256i v0 = _mm256_load_si256(src0);
    __m256i v1 = _mm256_load_si256(src0 + 1);
    __m256i v2 = _mm256_load_si256(src1);
    __m256i v3 = _mm256_load_si256(src1 + 1);
    __m256i p0, p1, p2, p3;
    p0 = grp128_gf16_mul_scalar_bs_avx2(&p1, v0, v1, dst_row, pvt_idx);
    p2 = grp128_gf16_mul_scalar_bs_avx2(&p3, v2, v3, dst_row, pvt_idx);
    __m256i* dst0 = (__m256i*) dst_row;
    __m256i* dst1 = (__m256i*) dst_inv_row;
    __m256i d0 = _mm256_load_si256(dst0);
    __m256i d1 = _mm256_load_si256(dst0 + 1);
    __m256i d2 = _mm256_load_si256(dst1);
    __m256i d3 = _mm256_load_si256(dst1 + 1);
    _mm256_store_si256(dst0, _mm256_xor_si256(d0, p0));
    _mm256_store_si256(dst0 + 1, _mm256_xor_si256(d1, p1));
    _mm256_store_si256(dst1, _mm256_xor_si256(d2, p2));
    _mm256_store_si256(dst1 + 1, _mm256_xor_si256(d3, p3));
#else

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wrestrict"
    // NOTE: dst_inv_row must be processed first. Otherwise dst_row changes
    grp128_gf16_fmsubi_scalar_bs(dst_inv_row, inv_row, dst_row, pvt_idx);
    grp128_gf16_fmsubi_scalar_bs(dst_row, pvt_row, dst_row, pvt_idx);
#pragma GCC diagnostic pop

#endif
}

#if defined(__AVX2__)

__m256i
rc128m_gf16_gj_avx2(RC128MGF16* restrict m, RC128MGF16* restrict inv) {
    // the lower 128 bits of indcols mark the independent columns,
    // while the upper 128 bits are simply a copy of the lower 128 bits
    __m256i indcols = ~_mm256_setzero_si256();// should compile to cmpeq_epi64
    gf16_t inv_coeff;
    for(uint32_t i = 0; i < 128; ++i) { // for each pivot
        uint32_t pvt_ri = i;
        for(; pvt_ri < 128; ++pvt_ri) { // find the pivot row
            const Grp128GF16* row = rc128m_gf16_raddr(m, pvt_ri);
            gf16_t coeff = grp128_gf16_at(row, i);
            if(coeff != 0) {
                inv_coeff = gf16_t_inv(coeff);
                break;
            }
        }

        if(pvt_ri == 128) { // singular column
            __m256i bm = _mm256_set_epi64x(0x0, 0x1, 0x0, 0x1);
            if(i >> 6) // i / 64
                bm = _mm256_bslli_epi128(bm, 8);
            bm = _mm256_slli_epi64(bm, i & 0x3F);
            indcols = _mm256_xor_si256(indcols, bm);
            continue;
        }

        Grp128GF16* pvt_row = rc128m_gf16_raddr(m, pvt_ri);
        Grp128GF16* inv_row = rc128m_gf16_raddr(inv, pvt_ri);
        grp128_gf16_muli_scalar(pvt_row, inv_coeff); // reduce the pivot row
        grp128_gf16_muli_scalar(inv_row, inv_coeff); // same operation to inv

        // row reduction
        for(uint32_t j = 0; j < i; ++j) { // above the current row
            rc128m_gf16_row_reduc(rc128m_gf16_raddr(m, j), pvt_row,
                                  rc128m_gf16_raddr(inv, j), inv_row, i);
        }
        // row reduction is not necessary for rows between the current row and
        // the pivot row
        for(uint32_t j = pvt_ri+1; j < 128; ++j) { // below the pivot row
            rc128m_gf16_row_reduc(rc128m_gf16_raddr(m, j), pvt_row,
                                  rc128m_gf16_raddr(inv, j), inv_row, i);
        }

        rc128m_gf16_swap_rows(m, pvt_ri, i);
        rc128m_gf16_swap_rows(inv, pvt_ri, i);
    }

    return indcols;
}

#endif

/* usage: Given a RC128MGF16 m, perform Gauss-Jordan elimination on it to
 *      identify independent columns, which form an invertible submatrix. The
 *      inverse can also be computed if the caller passes an identity matrix as
 *      inv. Alternatively, if the caller passes the constant column as inv,
 *      the solution of solvable systems can be computed.
 * params:
 *      1) m: ptr to a struct RC128MGF16
 *      2) inv: ptr to a struct RC128MGF16
 *      3) di: ptr to an uint128_t, when the function returns di encodes the
 *              independent columns. If the 1st column is independent, then the
 *              1st bit (0x1ULL) is set, and so on.
 * return: void */
void
rc128m_gf16_gj(RC128MGF16* restrict m, RC128MGF16* restrict inv,
               uint128_t* restrict di) {
    uint128_t_max(di);
    gf16_t inv_coeff;
    for(uint32_t i = 0; i < 128; ++i) { // for each pivot
        uint32_t pvt_ri = i;
        for(; pvt_ri < 128; ++pvt_ri) { // find the pivot row
            const Grp128GF16* row = rc128m_gf16_raddr(m, pvt_ri);
            gf16_t coeff = grp128_gf16_at(row, i);
            if(coeff != 0) {
                inv_coeff = gf16_t_inv(coeff);
                break;
            }
        }

        if(pvt_ri == 128) { // singular column
            uint128_t_toggle_at(di, i);
            continue;
        }

        Grp128GF16* pvt_row = rc128m_gf16_raddr(m, pvt_ri);
        Grp128GF16* inv_row = rc128m_gf16_raddr(inv, pvt_ri);
        grp128_gf16_muli_scalar(pvt_row, inv_coeff); // reduce the pivot row
        grp128_gf16_muli_scalar(inv_row, inv_coeff); // same operation to inv

        // row reduction
        for(uint32_t j = 0; j < i; ++j) { // above the current row
            rc128m_gf16_row_reduc(rc128m_gf16_raddr(m, j), pvt_row,
                                  rc128m_gf16_raddr(inv, j), inv_row, i);
        }
        // row reduction is not necessary for rows between the current row and
        // the pivot row
        for(uint32_t j = pvt_ri+1; j < 128; ++j) { // below the pivot row
            rc128m_gf16_row_reduc(rc128m_gf16_raddr(m, j), pvt_row,
                                  rc128m_gf16_raddr(inv, j), inv_row, i);
        }

        rc128m_gf16_swap_rows(m, pvt_ri, i);
        rc128m_gf16_swap_rows(inv, pvt_ri, i);
    }
}

#if defined(__AVX512F__)

void
rc128m_gf16_mul_naive_avx512(RC128MGF16* restrict p,
                             const RC128MGF16* restrict m,
                             const RC128MGF16* restrict n) {
    const Grp128GF16* m_row = rc128m_gf16_raddr((RC128MGF16*) m, 0);
    Grp128GF16* dst = rc128m_gf16_raddr(p, 0);
    for(uint64_t ri = 0; ri < 128; ++ri, ++m_row, ++dst) {
        const Grp128GF16* src = rc128m_gf16_raddr((RC128MGF16*) n, 0);
        __m512i prod = _mm512_setzero_si512();
        for(uint32_t ci = 0; ci < 128; ci += 2, src += 2) {
            __m512i v0 = _mm512_load_si512(src);
            __m512i v1 = _mm512_load_si512(src + 1);
            __m512i p0 = grp128_gf16_mul_scalar_bs_avx512(v0, m_row, ci);
            __m512i p1 = grp128_gf16_mul_scalar_bs_avx512(v1, m_row, ci + 1);
            prod = _mm512_xor_si512(prod, p0);
            prod = _mm512_xor_si512(prod, p1);
        }
        _mm512_store_si512(dst, prod);
    }
}

#elif defined(__AVX2__)

void
rc128m_gf16_mul_naive_avx2(RC128MGF16* restrict p, const RC128MGF16* restrict m,
                           const RC128MGF16* restrict n) {
    const Grp128GF16* m_row = rc128m_gf16_raddr((RC128MGF16*) m, 0);
    __m256i* dst = (__m256i*) rc128m_gf16_raddr(p, 0);
    for(uint64_t ri = 0; ri < 128; ++ri, ++m_row, dst += 2) {
        const __m256i* src = (__m256i*) rc128m_gf16_raddr((RC128MGF16*)n, 0);
        __m256i prod0 = _mm256_setzero_si256();
        __m256i prod1 = _mm256_setzero_si256();
        for(uint32_t ci = 0; ci < 128; ci += 2, src += 4) {
            __m256i v0 = _mm256_load_si256(src);
            __m256i v1 = _mm256_load_si256(src + 1);
            __m256i v2 = _mm256_load_si256(src + 2);
            __m256i v3 = _mm256_load_si256(src + 3);
            __m256i p0, p1, p2, p3;
            p0 = grp128_gf16_mul_scalar_bs_avx2(&p1, v0, v1, m_row, ci);
            p2 = grp128_gf16_mul_scalar_bs_avx2(&p3, v2, v3, m_row, ci + 1);
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
rc128m_gf16_mul_naive_vanilla(RC128MGF16* restrict p,
                              const RC128MGF16* restrict m,
                              const RC128MGF16* restrict n) {
    rc128m_gf16_zero(p);
    for(uint64_t ri = 0; ri < 128; ++ri) {
        const Grp128GF16* m_row = rc128m_gf16_raddr((RC128MGF16*) m, ri);
        Grp128GF16* dst_row = rc128m_gf16_raddr(p, ri);
        for(uint32_t ci = 0; ci < 128; ci += 2) {
            const Grp128GF16* src0 = rc128m_gf16_raddr((RC128MGF16*)n, ci);
            const Grp128GF16* src1 = rc128m_gf16_raddr((RC128MGF16*)n, ci + 1);
            grp128_gf16_fmaddi_scalar_bs(dst_row, src0, m_row, ci);
            grp128_gf16_fmaddi_scalar_bs(dst_row, src1, m_row, ci + 1);
        }
    }
}

#endif

/* usage: Given 2 struct RC128MGF16 m and n, compute m*n and store the result
 *      into p
 * params:
 *      1) p: ptr to a struct RC128MGF16
 *      2) m: ptr to a struct RC128MGF16
 *      3) n: ptr to a struct RC128MGF16
 * return: void */
void
rc128m_gf16_mul_naive(RC128MGF16* restrict p, const RC128MGF16* restrict m,
                      const RC128MGF16* restrict n) {
#if defined(__AVX512F__)
    rc128m_gf16_mul_naive_avx512(p, m, n);
#elif defined(__AVX2__)
    rc128m_gf16_mul_naive_avx2(p, m, n);
#else
    rc128m_gf16_mul_naive_vanilla(p, m, n);
#endif
}

/* usage: Given 2 RC128MGF16 A and B, replace a subset of columns of A with
 *      corresponding columns of B
 * params:
 *      1) a: ptr to struct RC128MGF16, storing the matrix A
 *      2) b: ptr to struct RC128MGF16, storing the matrix B
 *      3) di: ptr to uint128_t that encodes which columns of A to keep. If
 *          the LSB is 1, then the first column of A is keep. If 0, then the
 *          first column of A is replaced by the first column of B
 * return: void */
void
rc128m_gf16_mixi(RC128MGF16* restrict a, const RC128MGF16* restrict b,
                 const uint128_t* restrict di) {
    Grp128GF16* dst = rc128m_gf16_raddr(a, 0);
    const Grp128GF16* src = rc128m_gf16_raddr((RC128MGF16*)b, 0);
    for(uint32_t ri = 0; ri < 128; ri += 2, src += 2, dst += 2) {
        grp128_gf16_mixi(dst, src, di);
        grp128_gf16_mixi(dst + 1, src + 1, di);
    }
}

#if defined(__AVX2__)

/* usage: Given 2 RC128MGF16 A and B, replace a subset of columns of A with
 *      corresponding columns of B
 * params:
 *      1) a: ptr to struct RC128MGF16, storing the matrix A
 *      2) b: ptr to struct RC128MGF16, storing the matrix B
 *      3) di: a 256 bit register that encodes which columns of A to keep. If
 *          the LSB is 1, then the first column of A is keep. If 0, then the
 *          first column of A is replaced by the first column of B. The upper
 *          128 bits need to be duplicate of the lower 128 bits.
 * return: void */
void
rc128m_gf16_mixi_avx2(RC128MGF16* restrict a, const RC128MGF16* restrict b,
                      const __m256i di) {
    Grp128GF16* dst = rc128m_gf16_raddr(a, 0);
    const Grp128GF16* src = rc128m_gf16_raddr((RC128MGF16*)b, 0);
    for(uint32_t ri = 0; ri < 128; ri += 2, src += 2, dst += 2) {
        grp128_gf16_mixi_avx2(dst, src, di);
        grp128_gf16_mixi_avx2(dst + 1, src + 1, di);
    }
}

#endif

/* usage: Given 2 RC128MGF16 A and B, compute of A + B and store the result
 *      back into A
 * params:
 *      1) a: ptr to struct RC128MGF16, storing the matrix A
 *      2) b: ptr to struct RC128MGF16, storing the matrix B
 * return: void */
void
rc128m_gf16_addi(RC128MGF16* restrict a, const RC128MGF16* restrict b) {
    Grp128GF16* dst = rc128m_gf16_raddr(a, 0);
    const Grp128GF16* src = rc128m_gf16_raddr((RC128MGF16*)b, 0);
    for(uint32_t ri = 0; ri < 128; ri += 2, src += 2, dst += 2) {
        grp128_gf16_addi(dst, src);
        grp128_gf16_addi(dst + 1, src + 1);
    }
}

/* usage: Print a RC128MGF16 matrix
 * params:
 *      1) m: ptr to struct RC128MGF16
 * return: void */
void
rc128m_gf16_print(const RC128MGF16* m) {
    for(uint32_t i = 0; i < 128; ++i) {
        for(uint32_t j = 0; j < 128; ++j) {
            printf("%02d ", rc128m_gf16_at(m, i, j));
        }
        printf("\n");
    }
}

/* usage: Given a RC128MGF16, check if it is symmetric
 * params:
 *      1) m: ptr to struct RC128MGF16
 * return: True if symmetric. False otherwise */
bool
rc128m_gf16_is_symmetric(const RC128MGF16* m) {
    for(uint32_t i = 0; i < 128; ++i)
        for(uint32_t j = 0; j < i; ++j) {
            if(rc128m_gf16_at(m, i, j) != rc128m_gf16_at(m, j, i))
                return false;
        }

    return true;
}
