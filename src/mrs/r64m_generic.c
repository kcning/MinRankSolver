#include "r64m_generic.h"
#include "gf.h"
#include "uint512_t.h"
#include "uint64a.h"

#include <stdalign.h>
#include <stdlib.h> // random

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif


/* ========================================================================
 * struct R64MGeneric definition
 * ======================================================================== */

struct R64MGeneric {
    uint32_t rnum;
    // TODO: alignment
    uint512_t rows[]; // each GF element takes 8-bit, 64 x 8 = 512 bits
    // L1d is usually 128KB, which is enough for 2048 rows.
};

/* ========================================================================
 * function implementations
 * ======================================================================== */

/* usage: Compute the size of memory needed for a row in R64MGeneric
 * params:
 * return: size of memory needed in bytes */
uint64_t
r64m_generic_row_memsize(void) {
    return sizeof(uint512_t);
}

/* usage: Compute the size of memory needed for R64MGeneric
 * params:
 *      1) rnum: number of rows
 * return: size of memory needed in bytes */
uint64_t
r64m_generic_memsize(uint32_t rnum) {
    return sizeof(R64MGeneric) + sizeof(uint512_t) * rnum;
}

/* usage: Create a R64MGeneric matrix. The matrix is not initialized
 * params:
 *      1) rnum: number of rows
 * return: ptr to struct R64MGeneric. NULL on faliure */
R64MGeneric*
r64m_generic_create(uint32_t rnum) {
    R64MGeneric* m = malloc(r64m_generic_memsize(rnum));
    if(!m)
        return NULL;

    m->rnum = rnum;
    return m;
}

/* usage: Release a struct R64MGeneric
 * params:
 *      1) m: ptr to struct R64MGeneric
 * return: void */
void
r64m_generic_free(R64MGeneric* m) {
    free(m);
}

/* usage: Given a R64MGeneric matrix, return the number of rows
 * params:
 *      1) m: ptr to a struct R64MGeneric
 * return: the number of rows */
uint32_t
r64m_generic_rnum(const R64MGeneric* m) {
    return m->rnum;
}

/* usage: Given a R64MGeneric matrix and row index, return the row
 * params:
 *      1) m: ptr to a struct R64MGeneric
 *      2) i: index of the row, starting from 0
 *      3) r: a dense gf_t array of size 64 for storing the row
 * return: void */
void
r64m_generic_row(R64MGeneric* restrict m, uint32_t i,
                 gf_t* restrict r) {
    uint512_t_copy((uint512_t*)r, r64m_generic_raddr(m, i));
}

/* usage: Given a R64MGeneric matrix and row index, return the address
 *      to the row
 * params:
 *      1) m: ptr to a struct R64MGeneric
 *      2) i: index of the row, starting from 0
 * return: address to the i-th row */
gf_t*
r64m_generic_raddr(R64MGeneric* m, uint32_t i) {
    return (gf_t*) (m->rows + i);
}

/* usage: Given a R64MGeneric matrix and both row and column indices, return
 *      the coefficient of the given entry.
 * params:
 *      1) m: ptr to a struct R64MGeneric
 *      2) ri: index of the row, starting from 0
 *      3) ci: index of the column, from 0 ~ 63
 * return: coefficient of the entry */
gf_t
r64m_generic_at(const R64MGeneric* m, uint32_t ri, uint32_t ci) {
    return r64m_generic_raddr((R64MGeneric*) m, ri)[ci];
}

/* usage: Given a R64MGeneric matrix and both row and column indices, set
 *      the coefficient of the target entry to the given value.
 * params:
 *      1) m: ptr to a struct R64MGeneric
 *      2) ri: index of the row, starting from 0
 *      3) ci: index of the column, from 0 ~ 63
 *      4) v: the new coefficient
 * return: void */
void
r64m_generic_set_at(R64MGeneric* m, uint64_t ri, uint64_t ci, gf_t v) {
    r64m_generic_raddr(m, ri)[ci] = v;
}

/* usage: Reset a struct R64MGeneric to zero matrix
 * params:
 *      1) m: ptr to a struct R64MGeneric
 * return: void */
void
r64m_generic_zero(R64MGeneric* const m) {
    memset(m->rows, 0x0, sizeof(uint512_t) * r64m_generic_rnum(m));
}

/* usage: Given a struct R64MGeneric, populate it with random coefficients.
 * params:
 *      1) m: ptr to a struct R64MGeneric
 * return: void */
void
r64m_generic_rand(R64MGeneric* m) {
    for(uint32_t i = 0; i < r64m_generic_rnum(m); ++i) {
        gf_t * raddr = (gf_t*) r64m_generic_raddr(m, i);
        uint512_t_rand((uint512_t*) raddr);
        gf_t_arr_reduc_64(raddr);
    }
}

/* usage: Given 2 struct R64MGeneric, copy the 2nd R64MGeneric into the 1st one
 * params:
 *      1) dst: ptr to the struct R64MGeneric to copy to
 *      2) src: ptr to the struct R64MGeneric to copy from
 * return: void */
void
r64m_generic_copy(R64MGeneric* restrict dst,
                  const R64MGeneric* restrict src) {
    assert(r64m_generic_rnum(dst) == r64m_generic_rnum(src));
    memcpy(dst->rows, src->rows, sizeof(uint512_t) * r64m_generic_rnum(dst));
}

/* usage: Given 2 struct R64MGeneric, check if they are the same
 * params:
 *      1) a: ptr to the struct R64MGeneric
 *      2) b: ptr to the struct R64MGeneric
 * return: void */
bool
r64m_generic_is_equal(const R64MGeneric* restrict a,
                      const R64MGeneric* restrict b) {
    if(a == b)
        return true;
    assert(r64m_generic_rnum(a) == r64m_generic_rnum(b));
    return 0 == memcmp(a->rows, b->rows, sizeof(uint512_t) * r64m_generic_rnum(a));
}

/* usage: Given a R64MGeneric matrix m and a container, compute the Gramian
 *      matrix of m, i.e. m.transpose() * m, and store it into the container.
 *      Note that the product has dimension 64 x 64.
 * params:
 *      1) m: ptr to a struct R64MGeneric
 *      2) p: ptr to a struct RC64MGeneric, container for the result
 * return: void */
void
r64m_generic_gramian(const R64MGeneric* restrict m, RC64MGeneric* restrict p) {
    rc64m_generic_zero(p);
    for(uint64_t ri = 0; ri < r64m_generic_rnum(m); ++ri) {
        const gf_t* m_row = r64m_generic_raddr((R64MGeneric*) m, ri);

        for(uint32_t i = 0; i < 64; ++i) {
            if(m_row[i] == 0)
                continue;

            gf_t* dst = (gf_t*) rc64m_generic_raddr(p, i);
            gf_t_arr_fmaddi_scalar64(dst, m_row, m_row[i]);
        }
    }
}

/* usage: Given a R64MGeneric, set the selected column to zero
 * params:
 *      1) m: ptr to struct R64MGeneric
 *      2) ci: index of the selected column
 * return: void */
void
r64m_generic_zero_col(R64MGeneric* m, uint32_t ci) {
    for(uint32_t i = 0; i < r64m_generic_rnum(m); ++i) {
        r64m_generic_set_at(m, i, ci, 0);
    }
}

/* usage: Given a R64MGeneric, keep a subset of its columns while setting the
 *      remaining columns to zero
 * params:
 *      1) m: ptr to struct R64MGeneric
 *      2) di: a 64-bit integer that encodes which columns to keep. If the LSB
 *          is 1, then the first column is kept, and so on
 * return: void */
void
r64m_generic_zero_cols(R64MGeneric* m, uint64_t di) {
    uint8_t sb_idxs[64];
    uint32_t sbnum = uint64_t_sbpos(~di, sb_idxs);

    for(uint32_t ri = 0; ri < r64m_generic_rnum(m); ++ri) {
        gf_t* dst = (gf_t*) r64m_generic_raddr(m, ri);
        for(uint32_t i = 0; i < sbnum; ++i) {
            dst[sb_idxs[i]] = 0;
        }
    }
    /*
    for(uint32_t i = 0; i < 64; ++i) {
        if(!(di & 0x1) )// zero the column
            r64m_generic_zero_col(m, i);
        di >>= 1;
    }
    */
}

/* usage: Given a R64MGeneric, find the columns that are fully zero
 * params:
 *      1) m: ptr to struct R64MGeneric
 * return: a 64-bit integer that encodes zero columns. If the first column
 *      is fully zero, the LSB is set to 1, and so on. */
uint64_t
r64m_generic_zc_pos(const R64MGeneric* m) {
    uint64_t zp = UINT64_MAX;
    uint32_t ri = 0;
    do {
        const gf_t* row = r64m_generic_raddr((R64MGeneric*)m, ri++);
        for(uint64_t ci = 0; ci < 64; ++ci) {
            if(row[ci])
                zp &= ~(0x1 << ci);
        }
    } while(zp && ri < r64m_generic_rnum(m));

    return zp;
}

/* usage: Given a R64MGeneric, find the columns that are not fully zero
 * params:
 *      1) m: ptr to struct R64MGeneric
 * return: a 64-bit integer that encodes non-zero columns. If the first column
 *      is not fully zero, the LSB is set to 1, and so on. */
uint64_t
r64m_generic_nzc_pos(const R64MGeneric* m) {
    return ~r64m_generic_zc_pos(m);
}

/* usage : Givena  R64MGeneric, count the number of rows that are fully zero
 * params:
 *      1) m: ptr to struct R64MGeneric
 * return: number of fully zero rows */
uint32_t
r64m_generic_zr_count(const R64MGeneric* m) {
    uint32_t c = 0;
    for(uint32_t i = 0; i < r64m_generic_rnum(m); ++i) {
        const gf_t* row = r64m_generic_raddr((R64MGeneric*) m, i);
        // TODO: optimize
        if(64 == gf_t_arr_zc(row, 64))
            ++c;
    }
    return c;
}

/* usage: Given 2 R64MGeneric A and B, a RC64MGeneric C, and a 64x64 diagonal
 *      matrix D with coefficients either 1 and 0, compute A + B * C * D
 *      and store the result back into A
 * params:
 *      1) a: ptr to struct R64MGeneric, storing the matrix A
 *      2) b: ptr to struct R64MGeneric, storing the matrix B
 *      3) c: ptr to struct RC64MGeneric, storing the matrix C
 *      4) d: a 64-bit integer that encodes the diagonal matrix D. If the LSB
 *          is 1, then entry (0, 0) of D is 1. Otherwise 0.
 * return: void */
void
r64m_generic_fma_diag(R64MGeneric* restrict a, const R64MGeneric* restrict b,
                      const RC64MGeneric* restrict c, uint64_t d) {
    assert(r64m_generic_rnum(a) == r64m_generic_rnum(b));
    for(uint32_t i = 0; i < r64m_generic_rnum(a); ++i) {
        const gf_t* b_row = r64m_generic_raddr((R64MGeneric*) b, i);
        gf_t* dst = (gf_t*) r64m_generic_raddr(a, i);
        for(uint32_t j = 0; j < 64; ++j) {
            if(b_row[j] == 0)
                continue;
            gf_t_arr_fmaddi_scalar_mask64(dst, rc64m_generic_raddr(c, j), b_row[j], d);
        }
    }
}

/* usage: Given 2 R64MGeneric A and B, and a RC64MGeneric C, compute
 *      A + B * C and store the result back into A
 * params:
 *      1) a: ptr to struct R64MGeneric, storing the matrix A
 *      2) b: ptr to struct R64MGeneric, storing the matrix B
 *      3) c: ptr to struct RC64MGeneric, storing the matrix C
 * return: void */
void
r64m_generic_fma(R64MGeneric* restrict a, const R64MGeneric* restrict b,
                 const RC64MGeneric* restrict c) {
    assert(r64m_generic_rnum(a) == r64m_generic_rnum(b));
    for(uint32_t i = 0; i < r64m_generic_rnum(a); ++i) {
        const gf_t* b_row = r64m_generic_raddr((R64MGeneric*) b, i);
        gf_t* dst = (gf_t*) r64m_generic_raddr(a, i);
        for(uint32_t j = 0; j < 64; ++j) {
            if(b_row[j] == 0)
                continue;
            gf_t_arr_fmaddi_scalar64(dst, rc64m_generic_raddr(c, j), b_row[j]);
        }
    }
}

/* usage: Given 2 R64MGeneric A and B, a RC64MGeneric C, and a 64x64 diagonal
 *      matrix D with coefficients either 1 and 0, compute A * D + B * C
 *      and store the result back into A
 * params:
 *      1) a: ptr to struct R64MGeneric, storing the matrix A
 *      2) b: ptr to struct R64MGeneric, storing the matrix B
 *      3) c: ptr to struct RC64MGeneric, storing the matrix C
 *      4) d: a 64-bit integer that encodes the diagonal matrix D. If the LSB
 *          is 1, then entry (0, 0) of D is 1. Otherwise 0.
 * return: void */
void
r64m_generic_diag_fma(R64MGeneric* restrict a, const R64MGeneric* restrict b,
                      const RC64MGeneric* restrict c, uint64_t d) {
    assert(r64m_generic_rnum(a) == r64m_generic_rnum(b));

    gf_t mask[64];
    gf_t_arr_mask_from_64b(mask, d);

    for(uint32_t i = 0; i < r64m_generic_rnum(a); ++i) {
        const gf_t* b_row = r64m_generic_raddr((R64MGeneric*) b, i);
        gf_t* dst = (gf_t*) r64m_generic_raddr(a, i);

        // TODO: save the mask as registers for the next row instead of discarding it
        gf_t_arr_andi_64(dst, mask);
        for(uint32_t j = 0; j < 64; ++j) {
            if(b_row[j] == 0)
                continue;
            gf_t_arr_fmaddi_scalar64(dst, rc64m_generic_raddr(c, j), b_row[j]);
        }
    }
}

/* usage: Given 2 R64MGeneric A and B, a RC64MGeneric C, and a 64x64 diagonal
 *      matrix D with coefficients either 1 and 0, compute A - B * C * D
 *      and store the result back into A
 * params:
 *      1) a: ptr to struct R64MGeneric, storing the matrix A
 *      2) b: ptr to struct R64MGeneric, storing the matrix B
 *      3) c: ptr to struct RC64MGeneric, storing the matrix C
 *      4) d: a 64-bit integer that encodes the diagonal matrix D. If the LSB
 *          is 1, then entry (0, 0) of D is 1. Otherwise 0.
 * return: void */
void
r64m_generic_fms_diag(R64MGeneric* restrict a, const R64MGeneric* restrict b,
                      const RC64MGeneric* restrict c, uint64_t d) {
    assert(r64m_generic_rnum(a) == r64m_generic_rnum(b));
    for(uint32_t i = 0; i < r64m_generic_rnum(a); ++i) {
        const gf_t* b_row = r64m_generic_raddr((R64MGeneric*) b, i);
        gf_t* dst = (gf_t*) r64m_generic_raddr(a, i);
        for(uint32_t j = 0; j < 64; ++j) {
            if(b_row[j] == 0)
                continue;
            gf_t_arr_fmsubi_scalar_mask64(dst, rc64m_generic_raddr(c, j), b_row[j], d);
        }
    }
}

/* usage: Given 2 R64MGeneric A and B, and a RC64MGeneric C, compute
 *      A - B * C and store the result back into A
 * params:
 *      1) a: ptr to struct R64MGeneric, storing the matrix A
 *      2) b: ptr to struct R64MGeneric, storing the matrix B
 *      3) c: ptr to struct RC64MGeneric, storing the matrix C
 * return: void */
void
r64m_generic_fms(R64MGeneric* restrict a, const R64MGeneric* restrict b,
                 const RC64MGeneric* restrict c) {
    assert(r64m_generic_rnum(a) == r64m_generic_rnum(b));
    for(uint32_t i = 0; i < r64m_generic_rnum(a); ++i) {
        const gf_t* b_row = r64m_generic_raddr((R64MGeneric*) b, i);
        gf_t* dst = (gf_t*) r64m_generic_raddr(a, i);
        for(uint32_t j = 0; j < 64; ++j) {
            if(b_row[j] == 0)
                continue;
            gf_t_arr_fmsubi_scalar64(dst, rc64m_generic_raddr(c, j), b_row[j]);
        }
    }
}

/* subroutine of r64m_generic_mixi: replace a column in matrix A by the
 *      corresponding column in matrix B */
static inline void
r64m_generic_replace_col(R64MGeneric* restrict a, const R64MGeneric* restrict b,
                         uint32_t ci) {
    for(uint32_t i = 0; i < r64m_generic_rnum(a); ++i) {
        r64m_generic_set_at(a, i, ci, r64m_generic_at(b, i, ci));
    }
}

/* usage: Given 2 R64MGeneric A and B, replace a subset of columns of A with
 *      corresponding columns of B
 * params:
 *      1) a: ptr to struct R64MGeneric, storing the matrix A
 *      2) b: ptr to struct R64MGeneric, storing the matrix B
 *      3) di: a 64-bit integer that encodes which columns of A to keep. If
 *          the LSB is 1, then the first column of A is keep. If 0, then the
 *          first column of A is replaced by the first column  of B
 * return: void */
void
r64m_generic_mixi(R64MGeneric* restrict a, const R64MGeneric* restrict b,
                  uint64_t di) {
    uint8_t sb_idxs[64];
    uint32_t sbnum = uint64_t_sbpos(~di, sb_idxs);

    for(uint32_t ri = 0; ri < r64m_generic_rnum(a); ++ri) {
        gf_t* dst = (gf_t*) r64m_generic_raddr(a, ri);
        const gf_t* src = r64m_generic_raddr((R64MGeneric*)b, ri);
        for(uint32_t i = 0; i < sbnum; ++i) {
            dst[sb_idxs[i]] = src[sb_idxs[i]];
        }
    }
}

/* usage: Print a R64MGeneric matrix
 * params:
 *      1) m: ptr to struct R64MGeneric
 * return: void */
void
r64m_generic_print(const R64MGeneric* m) {
    for(uint32_t i = 0; i < r64m_generic_rnum(m); ++i) {
        for(uint32_t j = 0; j < 64; ++j) {
            printf("%02d ", r64m_generic_at(m, i, j));
        }
        printf("\n");
    }
}
