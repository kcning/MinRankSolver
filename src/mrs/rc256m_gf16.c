#include "rc256m_gf16.h"
#include <string.h>
#include <stdlib.h>
#include <stdalign.h>

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX256F__)
#include <immintrin.h>
#endif

/* ========================================================================
 * struct RC256MGF16 definition
 * ======================================================================== */

struct RC256MGF16 {
    // NOTE: alignment requirement for Grp256GF16 is 32-byte boundary
    // but we choose 64-byte boundary for AVX512 instructions
    alignas(64) Grp256GF16 rows[256];
};

/* ========================================================================
 * function implementations
 * ======================================================================== */

/* usage: compute the size of memory needed for struct RC256MGF16
 * return: size in bytes */
uint64_t
rc256m_gf16_memsize(void) {
    static_assert(sizeof(RC256MGF16) == (256 * 4 / 8 * 256),
                  "Incorrect size for RC256MGF16. Check padding");
    return sizeof(RC256MGF16);
}

/* usage: return the addr of the selected row in a struct RC256MGF16
 * params:
 *      1) m: ptr to struct RC256MGF16
 *      2) i: index of the row
 * return: the row addr as a ptr to struct Grp256GF16 */
Grp256GF16*
rc256m_gf16_raddr(RC256MGF16* m, uint32_t i) {
    return m->rows + i;
}

/* usage: swap 2 rows in a struct RC256MGF16
 * params:
 *      1) m: ptr to struct RC256MGF16
 *      2) i: index of the 1st row
 *      3) j: index of the 2nd row
 * return: void */
void
rc256m_gf16_swap_rows(RC256MGF16* m, uint32_t i, uint32_t j) {
    assert(i < 256 && j < 256);
    Grp256GF16* r1 = rc256m_gf16_raddr(m, i);
    Grp256GF16* r2 = rc256m_gf16_raddr(m, j);
    uint256_t_swap(r1->b, r2->b);
    uint256_t_swap(r1->b + 1, r2->b + 1);
    uint256_t_swap(r1->b + 2, r2->b + 2);
    uint256_t_swap(r1->b + 3, r2->b + 3);
}

/* usage: return the selected element in a struct RC256MGF16
 * params:
 *      1) m: ptr to struct RC256MGF16
 *      2) i: index of the row
 *      3) j: index of the column
 * return: the (i, j)-th element of the matrix */
gf16_t
rc256m_gf16_at(const RC256MGF16* m, uint32_t i, uint32_t j) {
    assert(i < 256 && j < 256);
    return grp256_gf16_at(rc256m_gf16_raddr((RC256MGF16*)m, i), j);
}

/* usage: set the selected element in a struct RC256MGF16 to the given value
 * params:
 *      1) m: ptr to struct RC256MGF16
 *      2) i: index of the row
 *      3) j: index of the column
 *      4) v: the new value
 * return: void */
void
rc256m_gf16_set_at(RC256MGF16* m, uint32_t i, uint32_t j, gf16_t v) {
    assert(i < 256 && j < 256);
    assert(v <= GF16_MAX);
    grp256_gf16_set_at(rc256m_gf16_raddr(m, i), j, v);
}

/* usage: Create a RC256MGF16 container. Note the matrix is not initialized.
 * params: none
 * return: a ptr to struct RC256MGF16. On failure, return NULL */
RC256MGF16*
rc256m_gf16_create(void) {
    // NOTE: alignment requirement for Grp256GF16 is 32-byte boundary
    // but we choose 64-byte boundary for AVX512 instructions
    RC256MGF16* m = aligned_alloc(64, sizeof(RC256MGF16));
    return m;
}

/* usage: Release a struct RC256MGF16
 * params:
 *      1) m: ptr to a struct RC256MGF16
 * return: void */
void
rc256m_gf16_free(RC256MGF16* m) {
    free(m);
}

/* usage: Given a struct RC256MGF16, populate it with random coefficients.
 * params:
 *      1) m: ptr to a struct RC256MGF16
 * return: void */
void
rc256m_gf16_rand(RC256MGF16* m) {
    for(uint32_t i = 0; i < 256; ++i) {
        grp256_gf16_rand(rc256m_gf16_raddr(m, i));
    }
}

/* usage: Reset a struct RC256MGF16 to zero matrix
 * params:
 *      1) m: ptr to a struct RC256MGF16
 * return: void */
void
rc256m_gf16_zero(RC256MGF16* m) {
    memset(m, 0x0, sizeof(RC256MGF16));
}

/* usage: given a struct RC256MGF16, set a subset of rows and columns to zero
 * params:
 *      1) m: ptr to a struct RC256MGF16
 *      2) d: ptr to uint256_t that encodes which rows and columns to keep.
 *          If the i-th bit is set, then the i-th row and column are kept.
 *          Otherwise, the i-th row and i-th column are cleared to zero.
 * return: void */
void
rc256m_gf16_zero_subset_rc(RC256MGF16* m, const uint256_t* restrict d) {
    uint8_t sbidxs[256];
    uint32_t sbnum = uint256_t_sbpos(d, sbidxs);
    uint32_t next_nzridx = 0;
    for(uint32_t i = 0; i < 256; ++i) {
        Grp256GF16* row = rc256m_gf16_raddr(m, i);
        if(sbnum && i == sbidxs[next_nzridx]) {
            grp256_gf16_zero_subset(row, d);
            ++next_nzridx;
            --sbnum;
        } else
            grp256_gf16_zero(row);
    }

    assert(sbnum == 0);
}

/* usage: Copy a struct RC256MGF16
 * params:
 *      1) dst: ptr to a struct RC256MGF16 for the copy
 *      2) src: ptr to a struct RC256MGF16. The source
 * return: void */
void
rc256m_gf16_copy(RC256MGF16* restrict dst, const RC256MGF16* restrict src) {
    memcpy(dst, src, sizeof(RC256MGF16));
}

/* usage: Reset a struct RC256MGF16 to identity matrix
 * params:
 *      1) m: ptr to a struct RC256MGF16
 * return: void */
void
rc256m_gf16_identity(RC256MGF16* m) {
    // TODO: optimize with AVX512 and AVX2
    rc256m_gf16_zero(m);
    for(uint32_t i = 0; i < 256; ++i) {
        Grp256GF16* row = rc256m_gf16_raddr(m, i);
        uint256_t_toggle_at(row->b, i);
    }
}

static inline void
rc256m_gf16_row_reduc(Grp256GF16* restrict dst_row,
                      const Grp256GF16* restrict pvt_row,
                      Grp256GF16* restrict dst_inv_row,
                      const Grp256GF16* restrict inv_row, uint32_t pvt_idx) {
    gf16_t mul_scalar = grp256_gf16_at(dst_row, pvt_idx);
    if(mul_scalar == 0)
        return;

    grp256_gf16_fmsubi_scalar(dst_row, pvt_row, mul_scalar);
    grp256_gf16_fmsubi_scalar(dst_inv_row, inv_row, mul_scalar);
}

/* usage: Given a RC256MGF16 m, perform Gauss-Jordan elimination on it to
 *      identify independent columns, which form an invertible submatrix. The
 *      inverse can also be computed if the caller passes an identity matrix as
 *      inv. Alternatively, if the caller passes the constant column as inv,
 *      the solution of solvable systems can be computed.
 * params:
 *      1) m: ptr to a struct RC256MGF16
 *      2) inv: ptr to a struct RC256MGF16
 *      3) di: ptr to an uint256_t, when the function returns di encodes the
 *              independent columns. If the 1st column is independent, then the
 *              1st bit (0x1ULL) is set, and so on.
 * return: void */
void
rc256m_gf16_gj(RC256MGF16* restrict m, RC256MGF16* restrict inv,
               uint256_t* restrict di) {
    uint256_t_max(di);
    gf16_t inv_coeff;
    for(uint32_t i = 0; i < 256; ++i) { // for each pivot
        uint32_t pvt_ri = i;
        for(; pvt_ri < 256; ++pvt_ri) { // find the pivot row
            const Grp256GF16* row = rc256m_gf16_raddr(m, pvt_ri);
            gf16_t coeff = grp256_gf16_at(row, i);
            if(coeff != 0) {
                inv_coeff = gf16_t_inv(coeff);
                break;
            }
        }

        if(pvt_ri == 256) { // singular column
            // TODO: optimize this
            uint256_t_toggle_at(di, i);
            continue;
        }

        Grp256GF16* pvt_row = rc256m_gf16_raddr(m, pvt_ri);
        Grp256GF16* inv_row = rc256m_gf16_raddr(inv, pvt_ri);
        grp256_gf16_muli_scalar(pvt_row, inv_coeff); // reduce the pivot row
        grp256_gf16_muli_scalar(inv_row, inv_coeff); // same operation to inv

        // row reduction
        for(uint32_t j = 0; j < i; ++j) { // above the current row
            rc256m_gf16_row_reduc(rc256m_gf16_raddr(m, j), pvt_row,
                                  rc256m_gf16_raddr(inv, j), inv_row, i);
        }
        // row reduction is not necessary for rows between the current row and
        // the pivot row
        for(uint32_t j = pvt_ri+1; j < 256; ++j) { // below the pivot row
            rc256m_gf16_row_reduc(rc256m_gf16_raddr(m, j), pvt_row,
                                  rc256m_gf16_raddr(inv, j), inv_row, i);
        }

        rc256m_gf16_swap_rows(m, pvt_ri, i);
        rc256m_gf16_swap_rows(inv, pvt_ri, i);
    }
}

/* usage: Given 2 struct RC256MGF16 m and n, compute m*n and store the result
 *      into p
 * params:
 *      1) p: ptr to a struct RC256MGF16
 *      2) m: ptr to a struct RC256MGF16
 *      3) n: ptr to a struct RC256MGF16
 * return: void */
void
rc256m_gf16_mul_naive(RC256MGF16* restrict p, const RC256MGF16* restrict m,
                      RC256MGF16* const restrict n) {
    rc256m_gf16_zero(p);
    for(uint64_t ri = 0; ri < 256; ++ri) {
        const Grp256GF16* m_row = rc256m_gf16_raddr((RC256MGF16*) m, ri);
        Grp256GF16* dst_row = rc256m_gf16_raddr(p, ri);
        for(uint32_t ci = 0; ci < 256; ++ci) {
            gf16_t v = grp256_gf16_at(m_row, ci);
            if(!v)
                continue;
            grp256_gf16_fmaddi_scalar(dst_row, rc256m_gf16_raddr(n, ci), v);
        }
    }
}

/* usage: Given 2 RC256MGF16 A and B, replace a subset of columns of A with
 *      corresponding columns of B
 * params:
 *      1) a: ptr to struct RC256MGF16, storing the matrix A
 *      2) b: ptr to struct RC256MGF16, storing the matrix B
 *      3) di: ptr to uint256_t that encodes which columns of A to keep. If
 *          the LSB is 1, then the first column of A is keep. If 0, then the
 *          first column of A is replaced by the first column of B
 * return: void */
void
rc256m_gf16_mixi(RC256MGF16* restrict a, const RC256MGF16* restrict b,
                 const uint256_t* restrict di) {
    for(uint32_t ri = 0; ri < 256; ++ri) {
        Grp256GF16* dst = rc256m_gf16_raddr(a, ri);
        const Grp256GF16* src = rc256m_gf16_raddr((RC256MGF16*)b, ri);
        grp256_gf16_mixi(dst, src, di);
    }
}

/* usage: Print a RC256MGF16 matrix
 * params:
 *      1) m: ptr to struct RC256MGF16
 * return: void */
void
rc256m_gf16_print(const RC256MGF16* m) {
    for(uint32_t i = 0; i < 256; ++i) {
        for(uint32_t j = 0; j < 256; ++j) {
            printf("%02d ", rc256m_gf16_at(m, i, j));
        }
        printf("\n");
    }
}

/* usage: Given a RC256MGF16, check if it is symmetric
 * params:
 *      1) m: ptr to struct RC256MGF16
 * return: True if symmetric. False otherwise */
bool
rc256m_gf16_is_symmetric(const RC256MGF16* m) {
    for(uint32_t i = 0; i < 256; ++i)
        for(uint32_t j = 0; j < i; ++j) {
            if(rc256m_gf16_at(m, i, j) != rc256m_gf16_at(m, j, i))
                return false;
        }

    return true;
}
