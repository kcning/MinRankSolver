#include "rc64m_gf16.h"
#include "grp64_gf16.h"
#include <string.h>
#include <stdio.h>
#include <stdalign.h>
#include "uint64a.h"
#include "rc64m_gf16_common.c"

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

/* ========================================================================
 * struct RC64MGF16 definition
 * ======================================================================== */

struct RC64MGF16 {
    // NOTE: same alignment requirement as Grp64GF16
    Grp64GF16 rows[64];
};

/* ========================================================================
 * function implementations
 * ======================================================================== */

/* usage: compute the size of memory needed for struct RC64MGF16
 * return: size in bytes */
uint64_t
rc64m_gf16_memsize(void) {
    return sizeof(RC64MGF16);
}

/* usage: return the addr of the selected row in a struct RC64MGF16
 * params:
 *      1) m: ptr to struct RC64MGF16
 *      2) i: index of the row
 * return: the row addr as a ptr to struct Grp64GF16 */
Grp64GF16*
rc64m_gf16_raddr(RC64MGF16* m, uint32_t i) {
    return m->rows + i;
}

/* usage: swap 2 rows in a struct RC64MGF16
 * params:
 *      1) m: ptr to struct RC64MGF16
 *      2) i: index of the 1st row
 *      3) j: index of the 2nd row
 * return: void */
void
rc64m_gf16_swap_rows(RC64MGF16* m, uint32_t i, uint32_t j) {
    assert(i < 64 && j < 64);
    Grp64GF16* r1 = rc64m_gf16_raddr(m, i);
    Grp64GF16* r2 = rc64m_gf16_raddr(m, j);
#if defined(__AVX__)
    __m256i va = _mm256_load_si256((__m256i*) r1->b);
    __m256i vb = _mm256_load_si256((__m256i*) r2->b);
    _mm256_store_si256((__m256i*) r1->b, vb);
    _mm256_store_si256((__m256i*) r2->b, va);
#else
    uint64_t t0 = r2->b[0];
    uint64_t t1 = r2->b[1];
    uint64_t t2 = r2->b[2];
    uint64_t t3 = r2->b[3];
    r2->b[0] = r1->b[0];
    r2->b[1] = r1->b[1];
    r2->b[2] = r1->b[2];
    r2->b[3] = r1->b[3];
    r1->b[0] = t0;
    r1->b[1] = t1;
    r1->b[2] = t2;
    r1->b[3] = t3;
#endif
}

/* usage: return the selected element in a struct RC64MGF16
 * params:
 *      1) m: ptr to struct RC64MGF16
 *      2) i: index of the row
 *      3) j: index of the column
 * return: the (i, j)-th element of the matrix */
gf16_t
rc64m_gf16_at(const RC64MGF16* m, uint32_t i, uint32_t j) {
    assert(i < 64 && j < 64);
    return grp64_gf16_at(rc64m_gf16_raddr((RC64MGF16*)m, i), j);
}

/* usage: set the selected element in a struct RC64MGF16 to the given value
 * params:
 *      1) m: ptr to struct RC64MGF16
 *      2) i: index of the row
 *      3) j: index of the column
 *      4) v: the new value
 * return: void */
void
rc64m_gf16_set_at(RC64MGF16* m, uint32_t i, uint32_t j, gf16_t v) {
    assert(i < 64 && j < 64);
    assert(v <= GF16_MAX);
    grp64_gf16_set_at(rc64m_gf16_raddr(m, i), j, v);
}

/* usage: Create a RC64MGF16 container. Note the matrix is not initialized.
 * params: none
 * return: a ptr to struct RC64MGF16. On failure, return NULL */
RC64MGF16*
rc64m_gf16_create(void) {
    // NOTE: align to 64-byte boundary for AVX512
    RC64MGF16* m = aligned_alloc(64, sizeof(RC64MGF16));
    return m;
}

/* usage: Create an array of RC64MGF16. None of the matrices is initialized.
 * params:
 *      1) sz: size of the array
 * return: a ptr to struct RC64MGF16. On failure, return NULL */
RC64MGF16*
rc64m_gf16_arr_create(uint32_t sz) {
    // align to 64-byte boundary for AVX512
    RC64MGF16* m = aligned_alloc(64, sizeof(RC64MGF16) * sz);
    return m;
}

/* usage: given an array of RC64MGF16, return a ptr to its i-th entry.
 * params:
 *      1) m: ptr to an array of RC64MGF16
 *      2) i: index of the entry
 * return: a struct RC64MGF16 ptr to the i-th entry */
RC64MGF16*
rc64m_gf16_arr_at(RC64MGF16* m, uint32_t i) {
    return m + i;
}

/* usage: Release a struct RC64MGF16
 * params:
 *      1) m: ptr to a struct RC64MGF16
 * return: void */
void
rc64m_gf16_free(RC64MGF16* m) {
    free(m);
}

/* usage: Release an array of struct RC64MGF16
 * params:
 *      1) m: ptr to a struct RC64MGF16
 * return: void */
void
rc64m_gf16_arr_free(RC64MGF16* m) {
    free(m);
}

/* usage: Given a struct RC64MGF16, populate it with random coefficients.
 * params:
 *      1) m: ptr to a struct RC64MGF16
 * return: void */
void
rc64m_gf16_rand(RC64MGF16* m) {
    for(uint32_t i = 0; i < 64; ++i)
        grp64_gf16_rand(rc64m_gf16_raddr(m, i));
}

/* usage: Reset a struct RC64MGF16 to zero matrix
 * params:
 *      1) m: ptr to a struct RC64MGF16
 * return: void */
void
rc64m_gf16_zero(RC64MGF16* m) {
    memset(m, 0x0, sizeof(RC64MGF16));
}

/* usage: given a struct RC64MGF16, set a subset of rows and columns to zero
 * params:
 *      1) m: ptr to a struct RC64MGF16
 *      2) d: a 64-bit integer that encodes which rows and columns to keep.
 *          If the i-th bit is set, then the i-th row and column are kept.
 *          Otherwise, the i-th row and i-th column are cleared to zero.
 * return: void */
void
rc64m_gf16_zero_subset_rc(RC64MGF16* m, uint64_t d) {
    Grp64GF16* row = rc64m_gf16_raddr(m, 0);
#if defined(__AVX512F__)
    __m512i vm = _mm512_set1_epi64(d);
    for(uint32_t i = 0; i < 32; ++i, row += 2) {
        __m512i v = _mm512_load_si512(row);
        _mm512_store_si512(row, _mm512_and_si512(v, vm));
    }
#else
    for(uint32_t i = 0; i < 64; ++i, ++row)
        grp64_gf16_zero_subset(row, d);
#endif

    // NOTE: mostly of the bits are set, so we compute ~d
    uint8_t sbidxs[64];
    uint32_t sbnum = uint64_t_sbpos(~d, sbidxs);
    for(uint32_t i = 0; i < sbnum; ++i)
        grp64_gf16_zero(rc64m_gf16_raddr(m, sbidxs[i]));
}

/* usage: Copy a struct RC64MGF16
 * params:
 *      1) dst: ptr to a struct RC64MGF16 for the copy
 *      2) src: ptr to a struct RC64MGF16. The source
 * return: void */
void
rc64m_gf16_copy(RC64MGF16* restrict dst, const RC64MGF16* restrict src) {
    memcpy(dst, src, sizeof(RC64MGF16));
}

/* usage: Reset a struct RC64MGF16 to identity matrix
 * params:
 *      1) m: ptr to a struct RC64MGF16
 * return: void */
void
rc64m_gf16_identity(RC64MGF16* m) {
#if defined(__AVX512F__)
    __m512i v = _mm512_set_epi64(0x0, 0x0, 0x0, 0x2, 0x0, 0x0, 0x0, 0x1);
    Grp64GF16* row = rc64m_gf16_raddr(m, 0);
    for(uint32_t i = 0; i < 32; ++i, row += 2) {
        _mm512_store_si512(row->b, v);
        v = _mm512_slli_epi64(v, 2);
    }
#elif defined(__AVX2__)
    __m256i v = _mm256_set_epi64x(0x0, 0x0, 0x0, 0x1);
    Grp64GF16* row = rc64m_gf16_raddr(m, 0);
    for(uint32_t i = 0; i < 64; ++i, ++row) {
        _mm256_store_si256((__m256i*) (row->b), v);
        v = _mm256_slli_epi64(v, 1);
    }
#else
    rc64m_gf16_zero(m);
    uint64_t v = 0x1ULL;
    for(uint32_t i = 0; i < 64; ++i) {
        Grp64GF16* row = rc64m_gf16_raddr(m, i);
        row->b[0] = v;
        v <<= 1;
    }
#endif
}

static inline void
rc64m_gf16_row_reduc(Grp64GF16* restrict dst_row,
                     const Grp64GF16* restrict pvt_row,
                     Grp64GF16* restrict dst_inv_row,
                     const Grp64GF16* restrict inv_row,
                     uint32_t pvt_idx) {
#if defined(__AVX512__)
    __m256i p0, p1;
    grp64_gf16_muli_scalar_from_bs_2x1_avx512(&p0, &p1, pvt_row, inv_row,
                                              dst_row, pvt_idx);
    __m256i v0 = _mm256_load_si256((__m256i*) dst_row);
    __m256i v1 = _mm256_load_si256((__m256i*) dst_inv_row);
    _mm256_store_si256((__m256i*) dst_row, _mm256_xor_si256(v0, p0));
    _mm256_store_si256((__m256i*) dst_inv_row, _mm256_xor_si256(v1, p1));
#elif defined(__AVX2__)
    __m256i b0 = _mm256_load_si256((__m256i*) pvt_row);
    __m256i b1 = _mm256_load_si256((__m256i*) inv_row);
    __m256i p0 = grp64_gf16_mul_scalar_from_bs_avx2(b0, dst_row, pvt_idx);
    __m256i p1 = grp64_gf16_mul_scalar_from_bs_avx2(b1, dst_row, pvt_idx);
    __m256i v0 = _mm256_load_si256((__m256i*) dst_row);
    __m256i v1 = _mm256_load_si256((__m256i*) dst_inv_row);
    _mm256_store_si256((__m256i*) dst_row, _mm256_xor_si256(v0, p0));
    _mm256_store_si256((__m256i*) dst_inv_row, _mm256_xor_si256(v1, p1));
#else

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wrestrict"
    // NOTE: the inverse row must be processed first because dst_row
    // will be modified after the 2nd func call
    grp64_gf16_fmsubi_scalar_bs(dst_inv_row, inv_row, dst_row, pvt_idx);
    grp64_gf16_fmsubi_scalar_bs(dst_row, pvt_row, dst_row, pvt_idx);
#pragma GCC diagnostic pop

#endif
}

/* usage: Given a RC64MGF16 m, perform Gauss-Jordan elimination on it to
 *      identify independent columns, which form an invertible submatrix.  The
 *      inverse can also be computed if the caller passes an identity matrix as
 *      inv. Alternatively, if the caller passes the constant column as inv, the
 *      solution of solvable systems can be computed.
 * params:
 *      1) m: ptr to a struct RC64MGF16
 *      2) inv: ptr to a struct RC64MGF16
 *      3) di: ptr to an uint64_t, when the function returns di encodes the
 *              independent columns. If the 1st column is independent, then the
 *              1st bit (0x1ULL) is set, and so on.
 * return: void */
void
rc64m_gf16_gj(RC64MGF16* restrict m, RC64MGF16* restrict inv,
              uint64_t* restrict di) {
    uint64_t indcols = UINT64_MAX;
    gf16_t inv_coeff;
    for(uint32_t i = 0; i < 64; ++i) { // for each pivot
        uint32_t pvt_ri = i;
        for(; pvt_ri < 64; ++pvt_ri) { // find the pivot row
            const Grp64GF16* row = rc64m_gf16_raddr(m, pvt_ri);
            gf16_t coeff = grp64_gf16_at(row, i);
            if(coeff != 0) {
                inv_coeff = gf16_t_inv(coeff);
                break;
            }
        }

        if(pvt_ri == 64) { // singular column
            indcols ^= 0x1ULL << i;
            continue;
        }

        Grp64GF16* pvt_row = rc64m_gf16_raddr(m, pvt_ri);
        Grp64GF16* inv_row = rc64m_gf16_raddr(inv, pvt_ri);

        // reduce the pivot row, and apply the same to inv
        grp64_gf16_muli_scalar_2x1(pvt_row, inv_row, inv_coeff);

        // row reduction
        for(uint32_t j = 0; j < i; ++j) { // above the current row
            rc64m_gf16_row_reduc(rc64m_gf16_raddr(m, j), pvt_row,
                                 rc64m_gf16_raddr(inv, j), inv_row, i);
        }
        // row reduction is not necessary for rows between the current row and
        // the pivot row
        for(uint32_t j = pvt_ri+1; j < 64; ++j) { // below the pivot row
            rc64m_gf16_row_reduc(rc64m_gf16_raddr(m, j), pvt_row,
                                 rc64m_gf16_raddr(inv, j), inv_row, i);
        }

        rc64m_gf16_swap_rows(m, pvt_ri, i);
        rc64m_gf16_swap_rows(inv, pvt_ri, i);
    }
    *di = indcols;
}

#if defined(__AVX512F__)

static force_inline void
rc64m_gf16_mul_avx512(RC64MGF16* restrict p, const RC64MGF16* restrict m,
                      const RC64MGF16* restrict n) {
    for(uint64_t ri = 0; ri < 64; ++ri) {
        const Grp64GF16* m_row = rc64m_gf16_raddr((RC64MGF16*) m, ri);
        Grp64GF16* dst = rc64m_gf16_raddr(p, ri);
        __m256i res = rc64m_gf16_mul_per_row_avx512(m_row, n);
        _mm256_store_si256((__m256i*) dst, res);
    }
}

#elif defined(__AVX2__)

static force_inline void
rc64m_gf16_mul_avx2(RC64MGF16* restrict p, const RC64MGF16* restrict m,
                    const RC64MGF16* restrict n) {
    for(uint64_t ri = 0; ri < 64; ++ri) {
        const Grp64GF16* m_row = rc64m_gf16_raddr((RC64MGF16*) m, ri);
        __m256i* dst = (__m256i*) rc64m_gf16_raddr(p, ri);
        __m256i prod = _mm256_setzero_si256();
        const __m256i* src = (__m256i*) rc64m_gf16_raddr((RC64MGF16*)n, 0);
        for(uint32_t j = 0; j < 64; j += 2, src += 2) {
            __m256i v0 = _mm256_load_si256(src);
            __m256i v1 = _mm256_load_si256(src + 1);
            __m256i p0 = grp64_gf16_mul_scalar_from_bs_avx2(v0, m_row, j);
            __m256i p1 = grp64_gf16_mul_scalar_from_bs_avx2(v1, m_row, j + 1);
            prod = _mm256_xor_si256(prod, p0);
            prod = _mm256_xor_si256(prod, p1);
        }
        _mm256_store_si256(dst, prod);
    }
}

#else

static force_inline void
rc64m_gf16_mul_naive_vanilla(RC64MGF16* restrict p, const RC64MGF16* restrict m,
                             const RC64MGF16* restrict n) {
    rc64m_gf16_zero(p);
    for(uint64_t ri = 0; ri < 64; ++ri) {
        const Grp64GF16* m_row = rc64m_gf16_raddr((RC64MGF16*) m, ri);
        Grp64GF16* dst = rc64m_gf16_raddr(p, ri);
        const Grp64GF16* src = rc64m_gf16_raddr((RC64MGF16*)n, 0);
        for(uint32_t j = 0; j < 64; j += 2, src += 2) {
            grp64_gf16_fmaddi_scalar_bs(dst, src, m_row, j);
            grp64_gf16_fmaddi_scalar_bs(dst, src + 1, m_row, j + 1);
        }
    }
}

#endif

/* usage: Given 2 struct RC64MGF16 m and n, compute m*n and store the result
 *      into p
 * params:
 *      1) p: ptr to a struct RC64MGF16
 *      2) m: ptr to a struct RC64MGF16
 *      3) n: ptr to a struct RC64MGF16
 * return: void */
void
rc64m_gf16_mul_naive(RC64MGF16* restrict p, const RC64MGF16* restrict m,
                     const RC64MGF16* restrict n) {
#if defined(__AVX512F__)
    rc64m_gf16_mul_avx512(p, m, n);
#elif defined(__AVX2__)
    rc64m_gf16_mul_avx2(p, m, n);
#else
    rc64m_gf16_mul_naive_vanilla(p, m, n);
#endif
}

/* usage: Given 2 RC64MGF16 A and B, replace a subset of columns of A with
 *      corresponding columns of B
 * params:
 *      1) a: ptr to struct RC64MGF16, storing the matrix A
 *      2) b: ptr to struct RC64MGF16, storing the matrix B
 *      3) di: a 64-bit integer that encodes which columns of A to keep. If
 *          the LSB is 1, then the first column of A is keep. If 0, then the
 *          first column of A is replaced by the first column  of B
 * return: void */
void
rc64m_gf16_mixi(RC64MGF16* restrict a, const RC64MGF16* restrict b,
                uint64_t di) {
    Grp64GF16* dst = rc64m_gf16_raddr(a, 0);
    const Grp64GF16* src = rc64m_gf16_raddr((RC64MGF16*)b, 0);
    for(uint32_t ri = 0; ri < 32; ++ri, src += 2, dst += 2)
        grp64_gf16_mixi_x2(dst, src, di);
}

/* usage: Given 2 RC64MGF16 A and B, compute of A + B and store the result
 *      back into A
 * params:
 *      1) a: ptr to struct RC64MGF16, storing the matrix A
 *      2) b: ptr to struct RC64MGF16, storing the matrix B
 * return: void */
void
rc64m_gf16_addi(RC64MGF16* restrict a, const RC64MGF16* restrict b) {
    Grp64GF16* dst = rc64m_gf16_raddr(a, 0);
    const Grp64GF16* src = rc64m_gf16_raddr((RC64MGF16*)b, 0);
    for(uint32_t ri = 0; ri < 32; ++ri, src += 2, dst += 2) {
        grp64_gf16_addi_x2(dst, src);
    }
}

/* usage: Print a RC64MGF16 matrix
 * params:
 *      1) m: ptr to struct RC64MGF16
 * return: void */
void
rc64m_gf16_print(const RC64MGF16* m) {
    for(uint32_t i = 0; i < 64; ++i) {
        for(uint32_t j = 0; j < 64; ++j) {
            printf("%02d ", rc64m_gf16_at(m, i, j));
        }
        printf("\n");
    }
}

/* usage: Given a RC64MGF16, check if it is symmetric
 * params:
 *      1) m: ptr to struct RC64MGF16
 * return: True if symmetric. False otherwise */
bool
rc64m_gf16_is_symmetric(const RC64MGF16* m) {
    for(uint32_t i = 0; i < 64; ++i)
        for(uint32_t j = 0; j < i; ++j) {
            if(rc64m_gf16_at(m, i, j) != rc64m_gf16_at(m, j, i))
                return false;
        }

    return true;
}
