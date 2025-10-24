#include "rc512m_gf16.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdalign.h>

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

/* ========================================================================
 * struct RC512MGF16 definition
 * ======================================================================== */

struct RC512MGF16 {
    alignas(64) Grp512GF16 rows[512];
};

/* ========================================================================
 * function implementations
 * ======================================================================== */

/* usage: compute the size of memory needed for struct RC512MGF16
 * return: size in bytes */
uint64_t
rc512m_gf16_memsize(void) {
    static_assert(sizeof(RC512MGF16) == (512 * 4 / 8 * 512),
                  "Incorrect size for RC512MGF16. Check padding");
    return sizeof(RC512MGF16);
}

/* usage: return the addr of the selected row in a struct RC512MGF16
 * params:
 *      1) m: ptr to struct RC512MGF16
 *      2) i: index of the row
 * return: the row addr as a ptr to struct Grp512GF16 */
Grp512GF16*
rc512m_gf16_raddr(RC512MGF16* m, uint32_t i) {
    return m->rows + i;
}

/* usage: swap 2 rows in a struct RC512MGF16
 * params:
 *      1) m: ptr to struct RC512MGF16
 *      2) i: index of the 1st row
 *      3) j: index of the 2nd row
 * return: void */
void
rc512m_gf16_swap_rows(RC512MGF16* m, uint32_t i, uint32_t j) {
    assert(i < 512 && j < 512);
    Grp512GF16* r1 = rc512m_gf16_raddr(m, i);
    Grp512GF16* r2 = rc512m_gf16_raddr(m, j);
    uint512_t_swap(r1->b, r2->b);
    uint512_t_swap(r1->b + 1, r2->b + 1);
    uint512_t_swap(r1->b + 2, r2->b + 2);
    uint512_t_swap(r1->b + 3, r2->b + 3);
}

/* usage: return the selected element in a struct RC512MGF16
 * params:
 *      1) m: ptr to struct RC512MGF16
 *      2) i: index of the row
 *      3) j: index of the column
 * return: the (i, j)-th element of the matrix */
gf16_t
rc512m_gf16_at(const RC512MGF16* m, uint32_t i, uint32_t j) {
    assert(i < 512 && j < 512);
    return grp512_gf16_at(rc512m_gf16_raddr((RC512MGF16*)m, i), j);
}

/* usage: set the selected element in a struct RC512MGF16 to the given value
 * params:
 *      1) m: ptr to struct RC512MGF16
 *      2) i: index of the row
 *      3) j: index of the column
 *      4) v: the new value
 * return: void */
void
rc512m_gf16_set_at(RC512MGF16* m, uint32_t i, uint32_t j, gf16_t v) {
    assert(i < 512 && j < 512);
    assert(v <= GF16_MAX);
    grp512_gf16_set_at(rc512m_gf16_raddr(m, i), j, v);
}

/* usage: Create a RC512MGF16 container. Note the matrix is not initialized.
 * params: none
 * return: a ptr to struct RC512MGF16. On failure, return NULL */
RC512MGF16*
rc512m_gf16_create(void) {
    // NOTE: alignment requirement for Grp512GF16 is 64-byte boundary
    RC512MGF16* m = aligned_alloc(64, sizeof(RC512MGF16));
    return m;
}

/* usage: Release a struct RC512MGF16
 * params:
 *      1) m: ptr to a struct RC512MGF16
 * return: void */
void
rc512m_gf16_free(RC512MGF16* m) {
    free(m);
}

/* usage: Given a struct RC512MGF16, populate it with random coefficients.
 * params:
 *      1) m: ptr to a struct RC512MGF16
 * return: void */
void
rc512m_gf16_rand(RC512MGF16* m) {
    for(uint32_t i = 0; i < 512; ++i) {
        grp512_gf16_rand(rc512m_gf16_raddr(m, i));
    }
}

/* usage: Reset a struct RC512MGF16 to zero matrix
 * params:
 *      1) m: ptr to a struct RC512MGF16
 * return: void */
void
rc512m_gf16_zero(RC512MGF16* m) {
    memset(m, 0x0, sizeof(RC512MGF16));
}

/* usage: given a struct RC512MGF16, set a subset of rows and columns to zero
 * params:
 *      1) m: ptr to a struct RC512MGF16
 *      2) d: ptr to uint512_t that encodes which rows and columns to keep.
 *          If the i-th bit is set, then the i-th row and column are kept.
 *          Otherwise, the i-th row and i-th column are cleared to zero.
 * return: void */
void
rc512m_gf16_zero_subset_rc(RC512MGF16* m, const uint512_t* restrict d) {
    uint16_t sbidxs[512];
    uint32_t sbnum = uint512_t_sbpos(d, sbidxs);
    uint32_t next_nzridx = 0;
    for(uint32_t i = 0; i < 512; ++i) {
        Grp512GF16* row = rc512m_gf16_raddr(m, i);
        if(sbnum && i == sbidxs[next_nzridx]) {
            grp512_gf16_zero_subset(row, d);
            ++next_nzridx;
            --sbnum;
        } else
            grp512_gf16_zero(row);
    }

    assert(sbnum == 0);
}

/* usage: Copy a struct RC512MGF16
 * params:
 *      1) dst: ptr to a struct RC512MGF16 for the copy
 *      2) src: ptr to a struct RC512MGF16. The source
 * return: void */
void
rc512m_gf16_copy(RC512MGF16* restrict dst, const RC512MGF16* restrict src) {
    memcpy(dst, src, sizeof(RC512MGF16));
}

/* usage: Reset a struct RC512MGF16 to identity matrix
 * params:
 *      1) m: ptr to a struct RC512MGF16
 * return: void */
void
rc512m_gf16_identity(RC512MGF16* m) {
#if defined(__AVX512F__)

#define store_1_slot_and_zero_3_slots(slot, vec, zero_vec) do { \
    _mm512_store_si512((slot), vec); \
    _mm512_store_si512((slot) + 1, zero_vec); \
    _mm512_store_si512((slot) + 2, zero_vec); \
    _mm512_store_si512((slot) + 3, zero_vec); \
} while(0)

    __m512i zv = _mm512_setzero_si512();
    __m512i v0 = _mm512_set_epi64(0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x1);
    __m512i v1 = _mm512_set_epi64(0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x1, 0x0);
    __m512i v2 = _mm512_set_epi64(0x0, 0x0, 0x0, 0x0, 0x0, 0x1, 0x0, 0x0);
    __m512i v3 = _mm512_set_epi64(0x0, 0x0, 0x0, 0x0, 0x1, 0x0, 0x0, 0x0);
    __m512i v4 = _mm512_set_epi64(0x0, 0x0, 0x0, 0x1, 0x0, 0x0, 0x0, 0x0);
    __m512i v5 = _mm512_set_epi64(0x0, 0x0, 0x1, 0x0, 0x0, 0x0, 0x0, 0x0);
    __m512i v6 = _mm512_set_epi64(0x0, 0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0);
    __m512i v7 = _mm512_set_epi64(0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0);

    uint32_t i = 0;
    for(; i < 64; ++i) {
        Grp512GF16* r0 = rc512m_gf16_raddr(m, i);
        store_1_slot_and_zero_3_slots(r0->b, v0, zv);
        v0 = _mm512_slli_epi64(v0, 1);
    }
    for(; i < 64 * 2; ++i) {
        Grp512GF16* r1 = rc512m_gf16_raddr(m, i);
        store_1_slot_and_zero_3_slots(r1->b, v1, zv);
        v1 = _mm512_slli_epi64(v1, 1);
    }
    for(; i < 64 * 3; ++i) {
        Grp512GF16* r2 = rc512m_gf16_raddr(m, i);
        store_1_slot_and_zero_3_slots(r2->b, v2, zv);
        v2 = _mm512_slli_epi64(v2, 1);
    }
    for(; i < 64 * 4; ++i) {
        Grp512GF16* r3 = rc512m_gf16_raddr(m, i);
        store_1_slot_and_zero_3_slots(r3->b, v3, zv);
        v3 = _mm512_slli_epi64(v3, 1);
    }
    for(; i < 64 * 5; ++i) {
        Grp512GF16* r4 = rc512m_gf16_raddr(m, i);
        store_1_slot_and_zero_3_slots(r4->b, v4, zv);
        v4 = _mm512_slli_epi64(v4, 1);
    }
    for(; i < 64 * 6; ++i) {
        Grp512GF16* r5 = rc512m_gf16_raddr(m, i);
        store_1_slot_and_zero_3_slots(r5->b, v5, zv);
        v5 = _mm512_slli_epi64(v5, 1);
    }
    for(; i < 64 * 7; ++i) {
        Grp512GF16* r6 = rc512m_gf16_raddr(m, i);
        store_1_slot_and_zero_3_slots(r6->b, v6, zv);
        v6 = _mm512_slli_epi64(v6, 1);
    }
    for(; i < 64 * 8; ++i) {
        Grp512GF16* r7 = rc512m_gf16_raddr(m, i);
        store_1_slot_and_zero_3_slots(r7->b, v7, zv);
        v7 = _mm512_slli_epi64(v7, 1);
    }
    assert(i == 512);

#undef store_1_slot_and_zero_3_slots

#else
    rc512m_gf16_zero(m);
    for(uint32_t i = 0; i < 512; ++i) {
        Grp512GF16* row = rc512m_gf16_raddr(m, i);
        uint512_t_toggle_at(row->b, i);
    }
#endif
}

static inline void
rc512m_gf16_row_reduc(Grp512GF16* restrict dst_row,
                      const Grp512GF16* restrict pvt_row,
                      Grp512GF16* restrict dst_inv_row,
                      const Grp512GF16* restrict inv_row, uint32_t pvt_idx) {
    gf16_t mul_scalar = grp512_gf16_at(dst_row, pvt_idx);
    if(mul_scalar == 0)
        return;

    grp512_gf16_fmsubi_scalar(dst_row, pvt_row, mul_scalar);
    grp512_gf16_fmsubi_scalar(dst_inv_row, inv_row, mul_scalar);
}

/* usage: Given a RC512MGF16 m, perform Gauss-Jordan elimination on it to
 *      identify independent columns, which form an invertible submatrix. The
 *      inverse can also be computed if the caller passes an identity matrix as
 *      inv. Alternatively, if the caller passes the constant column as inv,
 *      the solution of solvable systems can be computed.
 * params:
 *      1) m: ptr to a struct RC512MGF16
 *      2) inv: ptr to a struct RC512MGF16
 *      3) di: ptr to an uint512_t, when the function returns di encodes the
 *              independent columns. If the 1st column is independent, then the
 *              1st bit (0x1ULL) is set, and so on.
 * return: void */
void
rc512m_gf16_gj(RC512MGF16* restrict m, RC512MGF16* restrict inv,
               uint512_t* restrict di) {
    uint512_t_max(di);
    gf16_t inv_coeff;
    for(uint32_t i = 0; i < 512; ++i) { // for each pivot
        uint32_t pvt_ri = i;
        for(; pvt_ri < 512; ++pvt_ri) { // find the pivot row
            const Grp512GF16* row = rc512m_gf16_raddr(m, pvt_ri);
            gf16_t coeff = grp512_gf16_at(row, i);
            if(coeff != 0) {
                inv_coeff = gf16_t_inv(coeff);
                break;
            }
        }

        if(pvt_ri == 512) { // singular column
            uint512_t_toggle_at(di, i);
            continue;
        }

        Grp512GF16* pvt_row = rc512m_gf16_raddr(m, pvt_ri);
        Grp512GF16* inv_row = rc512m_gf16_raddr(inv, pvt_ri);
        grp512_gf16_muli_scalar(pvt_row, inv_coeff); // reduce the pivot row
        grp512_gf16_muli_scalar(inv_row, inv_coeff); // same operation to inv

        // row reduction
        for(uint32_t j = 0; j < i; ++j) { // above the current row
            rc512m_gf16_row_reduc(rc512m_gf16_raddr(m, j), pvt_row,
                                  rc512m_gf16_raddr(inv, j), inv_row, i);
        }
        // row reduction is not necessary for rows between the current row and
        // the pivot row
        for(uint32_t j = pvt_ri+1; j < 512; ++j) { // below the pivot row
            rc512m_gf16_row_reduc(rc512m_gf16_raddr(m, j), pvt_row,
                                  rc512m_gf16_raddr(inv, j), inv_row, i);
        }

        rc512m_gf16_swap_rows(m, pvt_ri, i);
        rc512m_gf16_swap_rows(inv, pvt_ri, i);
    }
}

/* usage: Given 2 struct RC512MGF16 m and n, compute m*n and store the result
 *      into p
 * params:
 *      1) p: ptr to a struct RC512MGF16
 *      2) m: ptr to a struct RC512MGF16
 *      3) n: ptr to a struct RC512MGF16
 * return: void */
void
rc512m_gf16_mul_naive(RC512MGF16* restrict p, const RC512MGF16* restrict m,
                      RC512MGF16* const restrict n) {
    rc512m_gf16_zero(p);
    for(uint64_t ri = 0; ri < 512; ++ri) {
        const Grp512GF16* m_row = rc512m_gf16_raddr((RC512MGF16*) m, ri);
        Grp512GF16* dst_row = rc512m_gf16_raddr(p, ri);
        for(uint32_t ci = 0; ci < 512; ++ci) {
            gf16_t v = grp512_gf16_at(m_row, ci);
            if(!v)
                continue;
            grp512_gf16_fmaddi_scalar(dst_row, rc512m_gf16_raddr(n, ci), v);
        }
    }
}

/* usage: Given 2 RC512MGF16 A and B, replace a subset of columns of A with
 *      corresponding columns of B
 * params:
 *      1) a: ptr to struct RC512MGF16, storing the matrix A
 *      2) b: ptr to struct RC512MGF16, storing the matrix B
 *      3) di: ptr to uint512_t that encodes which columns of A to keep. If
 *          the LSB is 1, then the first column of A is keep. If 0, then the
 *          first column of A is replaced by the first column of B
 * return: void */
void
rc512m_gf16_mixi(RC512MGF16* restrict a, const RC512MGF16* restrict b,
                 const uint512_t* restrict di) {
    for(uint32_t ri = 0; ri < 512; ++ri) {
        Grp512GF16* dst = rc512m_gf16_raddr(a, ri);
        const Grp512GF16* src = rc512m_gf16_raddr((RC512MGF16*)b, ri);
        grp512_gf16_mixi(dst, src, di);
    }
}

/* usage: Print a RC512MGF16 matrix
 * params:
 *      1) m: ptr to struct RC512MGF16
 * return: void */
void
rc512m_gf16_print(const RC512MGF16* m) {
    for(uint32_t i = 0; i < 512; ++i) {
        for(uint32_t j = 0; j < 512; ++j) {
            printf("%02d ", rc512m_gf16_at(m, i, j));
        }
        printf("\n");
    }
}

/* usage: Given a RC512MGF16, check if it is symmetric
 * params:
 *      1) m: ptr to struct RC512MGF16
 * return: True if symmetric. False otherwise */
bool
rc512m_gf16_is_symmetric(const RC512MGF16* m) {
    for(uint32_t i = 0; i < 512; ++i)
        for(uint32_t j = 0; j < i; ++j) {
            if(rc512m_gf16_at(m, i, j) != rc512m_gf16_at(m, j, i))
                return false;
        }

    return true;
}
