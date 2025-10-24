#include "rc64m_generic.h"
#include "gf.h"
#include "uint512_t.h"
#include "uint64a.h"
#include <stdio.h>

struct RC64MGeneric {
    uint512_t rows[64]; // 64 x 64 matrix whose elements are in GF (8 bits)
    uint8_t ridxs[64]; // ridxs[i] stores the offset of the i-th row
};

/* ========================================================================
 * struct RC64MGeneric definition
 * ======================================================================== */

/* usage: Compute the size of memory needed for RC64MGeneric
 * params: none
 * return: the size of memory needed in bytes */
uint64_t
rc64m_generic_memsize(void) {
    return sizeof(RC64MGeneric);
}

/* usage: return the address of the selected row in a struct RC64MGeneric
 * params:
 *      1) m: ptr to struct RC64MGeneric
 *      2) i: index of the row
 * return: the row address as a ptr to gf_t */
const gf_t*
rc64m_generic_raddr(const RC64MGeneric* m, uint32_t i) {
    return (gf_t*) (m->rows + m->ridxs[i]);
}

/* usage: return the selected element in a struct RC64MGeneric
 * params:
 *      1) m: ptr to struct RC64MGeneric
 *      2) i: index of the row
 *      3) j: index of the column
 * return: the (i, j)-th element of the matrix */
gf_t
rc64m_generic_at(const RC64MGeneric* m, uint32_t i, uint32_t j) {
    assert(i < 64 && j < 64);
    const gf_t* row = rc64m_generic_raddr(m, i);
    return row[j];
}

/* usage: set the selected element in a struct RC64MGeneric to the given value
 * params:
 *      1) m: ptr to struct RC64MGeneric
 *      2) i: index of the row
 *      3) j: index of the column
 *      4) v: the new value
 * return: void */
void
rc64m_generic_set_at(RC64MGeneric* m, uint32_t i, uint32_t j, gf_t v) {
    assert(i < 64 && j < 64);
    assert(v <= GF_MAX);
    gf_t* row = (gf_t*) rc64m_generic_raddr(m, i);
    row[j] = v;
}

/* usage: Create a RC64MGeneric container. Note the matrix is not initialized.
 * params: none
 * return: a ptr to struct RC64MGeneric. On failure, return NULL */
RC64MGeneric*
rc64m_generic_create(void) {
    // uint512_t_t needs to be aligned to an addr of 64 bytes.
    RC64MGeneric* m = aligned_alloc(64, rc64m_generic_memsize());
    if(!m)
        return NULL;

    for(uint32_t i = 0; i < 64; ++i) {
        m->ridxs[i] = i;
    }

    return m;
}

/* usage: Release a struct RC64MGeneric
 * params:
 *      1) m: ptr to a struct RC64MGeneric
 * return: void */
void
rc64m_generic_free(RC64MGeneric* m) {
    free(m);
}

/* usage: Given a struct RC64MGeneric, populate it with random coefficients.
 * params:
 *      1) m: ptr to a struct RC64MGeneric
 * return: void */
void
rc64m_generic_rand(RC64MGeneric* m) {
    for(uint32_t i = 0; i < 64; ++i) {
        gf_t * raddr = (gf_t*) rc64m_generic_raddr(m, i);
        uint512_t_rand((uint512_t*) raddr);
        gf_t_arr_reduc_64(raddr);
    }
}

/* usage: Reset a struct RC64MGeneric to zero matrix
 * params:
 *      1) m: ptr to a struct RC64MGeneric
 * return: void */
void
rc64m_generic_zero(RC64MGeneric* const m) {
    memset(m->rows, 0x0, sizeof(uint512_t) * 64);
    // leave the ridxs as it is
}

/* usage: Copy a struct RC64MGeneric
 * params:
 *      1) dst: ptr to a struct RC64MGeneric for the copy
 *      2) src: ptr to a struct RC64MGeneric. The source
 * return: void */
void
rc64m_generic_copy(RC64MGeneric* restrict dst, const RC64MGeneric* restrict src) {
    memcpy(dst, src, sizeof(RC64MGeneric));
}

/* usage: Reset a struct RC64MGeneric to identity matrix
 * params:
 *      1) m: ptr to a struct RC64MGeneric
 * return: void */
void
rc64m_generic_identity(RC64MGeneric* const m) {
    memset(m->rows, 0x0, sizeof(uint512_t) * 64);
    for(uint32_t i = 0; i < 64; ++i) {
        gf_t * row = (gf_t*) (m->rows + i); // treat the row as an array of gf_t
        row[i] = 1;

        // reset ridxs
        m->ridxs[i] = i;
    }
}

static inline void
rc64m_generic_swap_rows(RC64MGeneric* m, uint32_t i, uint32_t j) {
    uint8_t tmp = m->ridxs[i];
    m->ridxs[i] = m->ridxs[j];
    m->ridxs[j] = tmp;
}

static inline void
rc64m_generic_row_reduc(gf_t* restrict dst_row, const gf_t* restrict pvt_row,
                        gf_t* restrict dst_inv_row, const gf_t* restrict inv_row,
                        uint32_t pivot_idx) {
    gf_t mul_scalar = dst_row[pivot_idx];
    if(mul_scalar == 0)
        return;

    gf_t_arr_fmsubi_scalar64(dst_row, pvt_row, mul_scalar);
    assert(dst_row[pivot_idx] == 0);
    gf_t_arr_fmsubi_scalar64(dst_inv_row, inv_row, mul_scalar);
}

/* usage: Given a RC64MGeneric m, perform Gauss-Jordan elimination on it to
 *      identify independent columns, which form an invertible submatrix.
 *      The inverse can also be computed if the caller passes an identity matrix
 *      as inv. Alternatively, if the caller passes the constant column as inv, the
 *      solution of solvable systems can be computed.
 * params:
 *      1) m: ptr to a struct RC64MGeneric
 *      2) inv: ptr to a struct RC64MGeneric
 *      3) di: ptr to an uint64_t, when the function returns di encodes the
 *              independent columns. If the 1st column is independent, then the
 *              1st bit (0x1ULL) is set, and so on.
 * return: void */
void
rc64m_generic_gj(RC64MGeneric* restrict m, RC64MGeneric* restrict inv,
                 uint64_t* restrict di) {
    // NOTE: if matrix m is symmetric, one can also work with m^T instead of m.
    // The symmetric is quickly lost however, after row redunction for the first
    // pivot is done.
    uint64_t ind_cols = UINT64_MAX;
    gf_t inv_scalar;
    for(uint32_t i = 0; i < 64; ++i) { // for each pivot
        uint32_t pvt_ri = i;
        for(; pvt_ri < 64; ++pvt_ri) { // find the pivot row
            const gf_t* row = (gf_t*) rc64m_generic_raddr(m, pvt_ri);
            if (row[i] != GF_MIN) { // if not zero
                inv_scalar = gf_t_inv(row[i]);
                break;
            }
        }

        if(pvt_ri == 64) { // singular column
            ind_cols ^= 0x1ULL << i; // clear the bit for the column
            continue;
        }

        gf_t* pivot_row = (gf_t*) rc64m_generic_raddr(m, pvt_ri);
        gf_t* inv_row = (gf_t*) rc64m_generic_raddr(inv, pvt_ri);
        gf_t_arr_muli_scalar64(pivot_row, inv_scalar); // reduce the pivot row
        gf_t_arr_muli_scalar64(inv_row, inv_scalar); // same operation to inv

        // row reduction, apply the same operations to inv to compute the inverse
        for(uint32_t j = 0; j < i; ++j) { // above the current row
            rc64m_generic_row_reduc((gf_t*) rc64m_generic_raddr(m, j), pivot_row,
                                    (gf_t*) rc64m_generic_raddr(inv, j), inv_row, i);
        }
        // row reduction is not necessary for rows between the current row and
        // the pivot row
        for(uint32_t j = pvt_ri+1; j < 64; ++j) { // below the pivot row
            rc64m_generic_row_reduc((gf_t*) rc64m_generic_raddr(m, j), pivot_row,
                                    (gf_t*) rc64m_generic_raddr(inv, j), inv_row, i);
        }

        rc64m_generic_swap_rows(m, pvt_ri, i); // swap rows
        rc64m_generic_swap_rows(inv, pvt_ri, i); // same operation to inv
    }

    *di = ind_cols;
}

/* usage: Given 2 struct RC64MGeneric m and n, compute m*n and store the result into p
 * params:
 *      1) p: ptr to a struct RC64MGeneric
 *      2) m: ptr to a struct RC64MGeneric
 *      3) n: ptr to a struct RC64MGeneric
 * return: void */
void
rc64m_generic_mul_naive(RC64MGeneric* restrict p, const RC64MGeneric* restrict m,
                        RC64MGeneric* const restrict n) {
    rc64m_generic_zero(p);
    for(uint64_t ri = 0; ri < 64; ++ri) {
        const gf_t* m_row = rc64m_generic_raddr(m, ri);
        gf_t* dst_row = (gf_t*) rc64m_generic_raddr(p, ri);
        for(uint32_t ci = 0; ci < 64; ++ci) {
            gf_t v = m_row[ci];
            if(!v)
                continue;
            gf_t_arr_fmaddi_scalar64(dst_row, rc64m_generic_raddr(n, ci), v);
        }
    }
}

/* subroutine of rc64m_generic_mixi: replace a column in matrix A by the
 *      corresponding column in matrix B */
static inline void
rc64m_generic_replace_col(RC64MGeneric* restrict a, const RC64MGeneric* restrict b,
                          uint32_t ci) {
    for(uint32_t i = 0; i < 64; ++i) {
        rc64m_generic_set_at(a, i, ci, rc64m_generic_at(b, i, ci));
    }
}

/* usage: Given 2 RC64MGeneric A and B, replace a subset of columns of A by
 *      corresponding columns of B
 * params:
 *      1) a: ptr to struct RC64MGeneric, storing the matrix A
 *      2) b: ptr to struct RC64MGeneric, storing the matrix B
 *      3) di: a 64-bit integer that encodes which columns of A to keep. If
 *          the LSB is 1, then the first column of A is kept. If 0, then the
 *          first column of A is replaced by the first column  of B
 * return: void */
void
rc64m_generic_mixi(RC64MGeneric* restrict a, const RC64MGeneric* restrict b,
                   uint64_t di) {
    uint8_t sb_idxs[64];
    uint32_t sbnum = uint64_t_sbpos(~di, sb_idxs);

    for(uint32_t ri = 0; ri < 64; ++ri) {
        gf_t* dst = (gf_t*) rc64m_generic_raddr(a, ri);
        const gf_t* src = rc64m_generic_raddr(b, ri);
        for(uint32_t i = 0; i < sbnum; ++i) {
            dst[sb_idxs[i]] = src[sb_idxs[i]];
        }
    }
    /*
    for(uint32_t i = 0; i < 64; ++i) {
        if(!(di & 0x1)) // replace the column
            rc64m_generic_replace_col(a, b, i);
        di >>= 1;
    }
    */
}

/* usage: Given a RC64MGeneric, set the selected column to zero
 * params:
 *      1) m: ptr to struct RC64MGeneric
 *      2) ci: index of the selected column
 * return: void */
void
rc64m_generic_zero_col(RC64MGeneric* m, uint32_t ci) {
    // TODO: optimize this with AVX2
    for(uint32_t i = 0; i < 64; ++i) {
        rc64m_generic_set_at(m, i, ci, 0);
    }
}

/* usage: Given a RC64MGeneric, zero out its selected row
 * params:
 *      1) m: ptr to struct RC64MGeneric
 *      2) i: index of the row
 * return: void */
void
rc64m_generic_zero_row(RC64MGeneric* m, uint32_t i) {
    uint512_t_zero((uint512_t*) rc64m_generic_raddr(m, i));
}

/* usage: Given a RC64MGeneric, keep a subset of its columns while setting the
 *      remaining columns to zero
 * params:
 *      1) m: ptr to struct RC64MGeneric
 *      2) di: a 64-bit integer that encodes which columns to keep. If the LSB
 *          is 1, then the first column is kept, and so on
 * return: void */
void
rc64m_generic_zero_cols(RC64MGeneric* m, uint64_t di) {
    /*
    for(uint32_t i = 0; i < 64; ++i) {
        if(!(di & 0x1)) // zero the column
            rc64m_generic_zero_col(m, i);
        di >>= 1;
    }
    */
    for(uint32_t i = 0; i < 64; ++i) {
        gf_t_arr_zero_64b((gf_t*) rc64m_generic_raddr(m, i), di);
    }
}

/* usage: Print a RC64MGeneric matrix
 * params:
 *      1) m: ptr to struct RC64MGeneric
 * return: void */
void
rc64m_generic_print(const RC64MGeneric* m) {
    for(uint32_t i = 0; i < 64; ++i) {
        for(uint32_t j = 0; j < 64; ++j) {
            printf("%02d ", rc64m_generic_at(m, i, j));
        }
        printf("\n");
    }
}

/* usage: Given a RC64MGeneric, check if it is symmetric
 * params:
 *      1) m: ptr to struct RC64MGeneric
 * return: True if symmetric. False otherwise */
bool
rc64m_generic_is_symmetric(const RC64MGeneric* m) {
    for(uint32_t i = 0; i < 64; ++i)
        for(uint32_t j = 0; j < i; ++j) {
            if(rc64m_generic_at(m, i, j) != rc64m_generic_at(m, j, i))
                return false;
        }

    return true;
}
