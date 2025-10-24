#include "r256m_gf16.h"
#include <stdint.h>
#include <string.h>

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX256F__)
#include <immintrin.h>
#endif

/* ========================================================================
 * struct R256MGF16 definition
 * ======================================================================== */

struct R256MGF16 {
    uint64_t rnum; // uint32_t will cause padding. Might as well use 64-bit
    // 64 - 8 bytes of padding in between
    alignas(64) Grp256GF16 rows[];
};

/* ========================================================================
 * function implementations
 * ======================================================================== */

/* usage: Compute the size of memory needed for R256MGF16
 * params:
 *      1) rnum: number of rows
 * return: size of memory needed in bytes */
uint64_t
r256m_gf16_memsize(uint32_t rnum) {
    return sizeof(R256MGF16) + sizeof(Grp256GF16) * rnum;
}

/* usage: Create a R256MGF16 matrix. The matrix is not initialized
 * params:
 *      1) rnum: number of rows
 * return: ptr to struct R256MGF16. NULL on faliure */
R256MGF16*
r256m_gf16_create(uint32_t rnum) {
    static_assert(sizeof(Grp256GF16) == 128, "size of Grp256GF16 is not 128");
    // NOTE: field 'rows' needs to be aligned to a 64-byte boundary
    R256MGF16* m = aligned_alloc(64, r256m_gf16_memsize(rnum));
    if(!m)
        return NULL;

    m->rnum = rnum;
    return m;
}

/* usage: Release a struct R256MGF16
 * params:
 *      1) m: ptr to struct R256MGF16
 * return: void */
void
r256m_gf16_free(R256MGF16* m) {
    free(m);
}

/* usage: Given a R256MGF16 matrix, return the number of rows
 * params:
 *      1) m: ptr to a struct R256MGF16
 * return: the number of rows */
uint32_t
r256m_gf16_rnum(const R256MGF16* m) {
    return m->rnum;
}

/* usage: Given a R256MGF16 matrix and row index, return the address
 *      to the row
 * params:
 *      1) m: ptr to a struct R256MGF16
 *      2) i: index of the row, starting from 0
 * return: address to the i-th row */
Grp256GF16*
r256m_gf16_raddr(R256MGF16* m, uint32_t i) {
    return m->rows + i;
}

/* usage: Given a R256MGF16 matrix and both row and column indices, return
 *      the coefficient of the given entry.
 * params:
 *      1) m: ptr to a struct R256MGF16
 *      2) ri: index of the row, starting from 0
 *      3) ci: index of the column, from 0 ~ 63
 * return: coefficient of the entry */
gf16_t
r256m_gf16_at(const R256MGF16* m, uint32_t ri, uint32_t ci) {
    return grp256_gf16_at(r256m_gf16_raddr((R256MGF16*) m, ri), ci);
}

/* usage: Given a R256MGF16 matrix and both row and column indices, set
 *      the coefficient of the target entry to the given value.
 * params:
 *      1) m: ptr to a struct R256MGF16
 *      2) ri: index of the row, starting from 0
 *      3) ci: index of the column, from 0 ~ 63
 *      4) v: the new coefficient
 * return: void */
void
r256m_gf16_set_at(R256MGF16* m, uint64_t ri, uint64_t ci, gf16_t v) {
    grp256_gf16_set_at(r256m_gf16_raddr(m, ri), ci, v);
}

/* usage: Reset a struct R256MGF16 to zero matrix
 * params:
 *      1) m: ptr to a struct R256MGF16
 * return: void */
void
r256m_gf16_zero(R256MGF16* const m) {
    memset(m->rows, 0x0, sizeof(Grp256GF16) * r256m_gf16_rnum(m));
}

/* usage: Given a struct R256MGF16, populate it with random coefficients.
 * params:
 *      1) m: ptr to a struct R256MGF16
 * return: void */
void
r256m_gf16_rand(R256MGF16* m) {
    for(uint32_t i = 0; i < r256m_gf16_rnum(m); ++i)
        grp256_gf16_rand(r256m_gf16_raddr(m, i));
}

/* usage: Given 2 struct R256MGF16, copy the 2nd R256MGF16 into the 1st one
 * params:
 *      1) dst: ptr to the struct R256MGF16 to copy to
 *      2) src: ptr to the struct R256MGF16 to copy from
 * return: void */
void
r256m_gf16_copy(R256MGF16* restrict dst, const R256MGF16* restrict src) {
    assert(r256m_gf16_rnum(dst) == r256m_gf16_rnum(src));
    memcpy(dst->rows, src->rows, sizeof(Grp256GF16) * r256m_gf16_rnum(dst));
}

/* usage: Given a R256MGF16 matrix m and a container, compute the Gramian
 *      matrix of m, i.e. m.transpose() * m, and store it into the container.
 *      Note that the product has dimension 64 x 64.
 * params:
 *      1) m: ptr to a struct R256MGF16
 *      2) p: ptr to a struct RC256MGF16, container for the result
 * return: void */
void
r256m_gf16_gramian(const R256MGF16* restrict m, RC256MGF16* restrict p) {
    rc256m_gf16_zero(p);
    for(uint64_t ri = 0; ri < r256m_gf16_rnum(m); ++ri) {
        const Grp256GF16* m_row = r256m_gf16_raddr((R256MGF16*) m, ri);

        for(uint32_t i = 0; i < 256; ++i) {
            // TODO: extract multiple coeffs at once
            gf16_t c = grp256_gf16_at(m_row, i);
            if(c == 0)
                continue;

            Grp256GF16* dst = rc256m_gf16_raddr(p, i);
            grp256_gf16_fmsubi_scalar(dst, m_row, c);
        }
    }
}

/* usage: Given a R256MGF16, find the columns that are fully zero
 * params:
 *      1) m: ptr to struct R256MGF16
 *      2) out: ptr to a uint256_t which on return encodes zero columns. If the
 *          first column is fully zero, the LSB is set to 1, and so on.
 * return: void */
void
r256m_gf16_zc_pos(const R256MGF16* restrict m, uint256_t* restrict out) {
    uint256_t_max(out);
    for(uint32_t i = 0; i < r256m_gf16_rnum(m); ++i) {
        const Grp256GF16* row = r256m_gf16_raddr((R256MGF16*)m, i);
        uint256_t tmp; grp256_gf16_zpos(&tmp, row);
        uint256_t_andi(out, &tmp);
        if(uint256_t_is_zero(out))
            break;
    }
}

/* usage: Given a R256MGF16, find the columns whose selected rows are fully zero
 * params:
 *      1) m: ptr to struct R256MGF16
 *      2) ridxs: a uint32_t array that stores the indices of the selected
 *          rows
 *      3) sz: size of ridxs
 *      4) out: ptr to a uint256_t which on return encodes zero columns. If the
 *          first column is fully zero, the LSB is set to 1, and so on.
 * return: void */
void
r256m_gf16_subset_zc_pos(const R256MGF16* restrict m,
                         const uint32_t* restrict ridxs, uint32_t sz,
                         uint256_t* restrict out) {
    uint256_t_max(out);
    for(uint32_t i = 0; i < sz; ++i) {
        uint32_t ri = ridxs[i];
        const Grp256GF16* row = r256m_gf16_raddr((R256MGF16*)m, ri);
        uint256_t tmp; grp256_gf16_zpos(&tmp, row);
        uint256_t_andi(out, &tmp);
        if(uint256_t_is_zero(out))
            break;
    }
}

/* usage: Given a R256MGF16, find the columns that are not fully zero
 * params:
 *      1) m: ptr to struct R256MGF16
 *      2) out: ptr to a uint256_t which on return encodes non-zero columns. If
 *          the first column is not fully zero, the LSB is set to 1, and so on.
 * return: void */
void
r256m_gf16_nzc_pos(const R256MGF16* restrict m, uint256_t* restrict out) {
    r256m_gf16_zc_pos(m, out);
    uint256_t_negi(out);
}

/* usage: Given 2 R256MGF16 A and B, and a RC256MGF16 C, compute
 *      A + B * C and store the result back into A
 * params:
 *      1) a: ptr to struct R256MGF16, storing the matrix A
 *      2) b: ptr to struct R256MGF16, storing the matrix B
 *      3) c: ptr to struct RC256MGF16, storing the matrix C
 * return: void */
void
r256m_gf16_fma(R256MGF16* restrict a, const R256MGF16* restrict b,
               const RC256MGF16* restrict c) {
    assert(r256m_gf16_rnum(a) == r256m_gf16_rnum(b));
    for(uint32_t i = 0; i < r256m_gf16_rnum(a); ++i) {
        const Grp256GF16* b_row = r256m_gf16_raddr((R256MGF16*) b, i);
        Grp256GF16* dst = r256m_gf16_raddr(a, i);
        for(uint32_t j = 0; j < 256; ++j) {
            gf16_t coeff = grp256_gf16_at(b_row, j);
            if(coeff == 0)
                continue;
            const Grp256GF16* src = rc256m_gf16_raddr((RC256MGF16*)c, j);
            grp256_gf16_fmaddi_scalar(dst, src, coeff);
        }
    }
}

/* usage: Given 2 R256MGF16 A and B, a RC256MGF16 C, and a 64x64 diagonal
 *      matrix D with coefficients either 1 and 0, compute A + B * C * D
 *      and store the result back into A
 * params:
 *      1) a: ptr to struct R256MGF16, storing the matrix A
 *      2) b: ptr to struct R256MGF16, storing the matrix B
 *      3) c: ptr to struct RC256MGF16, storing the matrix C
 *      4) d: ptr to a uint256_t which encodes the diagonal matrix D. If the
 *          LSB is 1, then entry (0, 0) of D is 1. Otherwise 0.
 * return: void */
void
r256m_gf16_fma_diag(R256MGF16* restrict a, const R256MGF16* restrict b,
                    const RC256MGF16* restrict c,
                    const uint256_t* restrict d) {
    assert(r256m_gf16_rnum(a) == r256m_gf16_rnum(b));
    for(uint32_t i = 0; i < r256m_gf16_rnum(a); ++i) {
        const Grp256GF16* b_row = r256m_gf16_raddr((R256MGF16*) b, i);
        Grp256GF16* dst = r256m_gf16_raddr(a, i);
        for(uint32_t j = 0; j < 256; ++j) {
            gf16_t coeff = grp256_gf16_at(b_row, j);
            if(coeff == 0)
                continue;
            const Grp256GF16* src = rc256m_gf16_raddr((RC256MGF16*)c, j);
            grp256_gf16_fmaddi_scalar_mask(dst, src, coeff, d);
        }
    }
}

/* usage: Given 2 R256MGF16 A and B, a RC256MGF16 C, and a 64x64 diagonal
 *      matrix D with coefficients either 1 and 0, compute A * D + B * C
 *      and store the result back into A
 * params:
 *      1) a: ptr to struct R256MGF16, storing the matrix A
 *      2) b: ptr to struct R256MGF16, storing the matrix B
 *      3) c: ptr to struct RC256MGF16, storing the matrix C
 *      4) d: ptr to a uint256_t which encodes the diagonal matrix D. If the
 *          LSB is 1, then entry (0, 0) of D is 1. Otherwise 0.
 * return: void */
void
r256m_gf16_diag_fma(R256MGF16* restrict a, const R256MGF16* restrict b,
                    const RC256MGF16* restrict c,
                    const uint256_t* restrict d) {
    assert(r256m_gf16_rnum(a) == r256m_gf16_rnum(b));
    for(uint32_t i = 0; i < r256m_gf16_rnum(a); ++i) {
        const Grp256GF16* b_row = r256m_gf16_raddr((R256MGF16*) b, i);
        Grp256GF16* dst = r256m_gf16_raddr(a, i);

        grp256_gf16_zero_subset(dst, d);
        for(uint32_t j = 0; j < 256; ++j) {
            gf16_t coeff = grp256_gf16_at(b_row, j);
            if(coeff == 0)
                continue;
            const Grp256GF16* src = rc256m_gf16_raddr((RC256MGF16*)c, j);
            grp256_gf16_fmaddi_scalar(dst, src, coeff);
        }
    }
}

/* usage: Given 2 R256MGF16 A and B, and a RC256MGF16 C, compute
 *      A - B * C and store the result back into A
 * params:
 *      1) a: ptr to struct R256MGF16, storing the matrix A
 *      2) b: ptr to struct R256MGF16, storing the matrix B
 *      3) c: ptr to struct RC256MGF16, storing the matrix C
 * return: void */
void
r256m_gf16_fms(R256MGF16* restrict a, const R256MGF16* restrict b,
               const RC256MGF16* restrict c) {
    // In GF(16), addition is subtraction
    r256m_gf16_fma(a, b, c);
}

/* usage: Given 2 R256MGF16 A and B, a RC256MGF16 C, and a 64x64 diagonal
 *      matrix D with coefficients either 1 and 0, compute A - B * C * D
 *      and store the result back into A
 * params:
 *      1) a: ptr to struct R256MGF16, storing the matrix A
 *      2) b: ptr to struct R256MGF16, storing the matrix B
 *      3) c: ptr to struct RC256MGF16, storing the matrix C
 *      4) d: ptr to a uint256_t which encodes the diagonal matrix D. If the
 *          LSB is 1, then entry (0, 0) of D is 1. Otherwise 0.
 * return: void */
void
r256m_gf16_fms_diag(R256MGF16* restrict a, const R256MGF16* restrict b,
                    const RC256MGF16* restrict c,
                    const uint256_t* restrict d) {
    // In GF(16), addition is subtraction
    r256m_gf16_fma_diag(a, b, c, d);
}

/* usage: Given 2 R256MGF16 A and B, replace a subset of columns of A with
 *      corresponding columns of B
 * params:
 *      1) a: ptr to struct R256MGF16, storing the matrix A
 *      2) b: ptr to struct R256MGF16, storing the matrix B
 *      3) di: ptr to a uint256_t which encodes the diagonal matrix D. If the
 *          LSB is 1, then entry (0, 0) of D is 1. Otherwise 0.
 * return: void */
void
r256m_gf16_mixi(R256MGF16* restrict a, const R256MGF16* restrict b,
                const uint256_t* restrict di) {
    for(uint32_t ri = 0; ri < r256m_gf16_rnum(a); ++ri) {
        Grp256GF16* dst = r256m_gf16_raddr(a, ri);
        const Grp256GF16* src = r256m_gf16_raddr((R256MGF16*)b, ri);
        grp256_gf16_mixi(dst, src, di);
    }
}

/* usage: Given 2 R256MGF16 A and B, compute of A + B and store the result
 *      back into A
 * params:
 *      1) a: ptr to struct R256MGF16, storing the matrix A
 *      2) b: ptr to struct R256MGF16, storing the matrix B
 * return: void */
void
r256m_gf16_addi(R256MGF16* restrict a, const R256MGF16* restrict b) {
    assert(r256m_gf16_rnum(a) == r256m_gf16_rnum(b));
    for(uint64_t ri = 0; ri < r256m_gf16_rnum(a); ++ri) {
        Grp256GF16* dst = r256m_gf16_raddr(a, ri);
        const Grp256GF16* src = r256m_gf16_raddr((R256MGF16*)b, ri);
        grp256_gf16_addi(dst, src);
    }
}
