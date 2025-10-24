#include "c512m_gf16.h"
#include "r512m_gf16.h"

#include <stdio.h>

/* ========================================================================
 * struct C512MGF16 definition
 * ======================================================================== */

/* ========================================================================
 * function implementations
 * ======================================================================== */

/* usage: Compute the size of memory needed for C512MGF16
 * params:
 *      1) cnum: number of columns
 * return: size of memory needed in bytes */
uint64_t
c512m_gf16_memsize(uint32_t cnum) {
    return r512m_gf16_memsize(cnum);
}

/* usage: Create a C512MGF16 matrix. The matrix is not initialized
 * params:
 *      1) cnum: number of columns
 * return: ptr to struct C512MGF16. NULL on failure */
C512MGF16*
c512m_gf16_create(uint32_t cnum) {
    return r512m_gf16_create(cnum);
}

/* usage: Release a struct C512MGF16
 * params:
 *      1) m: ptr to struct C512MGF16
 * return: void */
void
c512m_gf16_free(C512MGF16* m) {
    r512m_gf16_free(m);
}

/* usage: Given a C512MGF16 matrix, return the number of columns
 * params:
 *      1) m: ptr to a struct C512MGF16
 * return: the number of columns */
uint32_t
c512m_gf16_cnum(const C512MGF16* m) {
    return r512m_gf16_rnum(m);
}

/* usage: Given a C512MGF16 matrix and column index, return the address
 *      to the column
 * params:
 *      1) m: ptr to a struct C512MGF16
 *      2) i: index of the column, starting from 0
 * return: address to the i-th column */
Grp512GF16*
c512m_gf16_caddr(C512MGF16* m, uint32_t i) {
    return r512m_gf16_raddr(m, i);
}

/* usage: Given a C512MGF16 matrix and both row and column indices, return
 *      the coefficient of the given entry.
 * params:
 *      1) m: ptr to a struct C512MGF16
 *      2) ri: index of the row, from 0 ~ 63
 *      3) ci: index of the column, start from 0
 * return: coefficient of the entry */
gf16_t
c512m_gf16_at(const C512MGF16* m, uint32_t ri, uint32_t ci) {
    return r512m_gf16_at(m, ci, ri);
}

/* usage: Given a C512MGF16 matrix and both row and column indices, set
 *      the coefficient of the target entry to the given value.
 * params:
 *      1) m: ptr to a struct C512MGF16
 *      2) ri: index of the row, from 0 ~ 63
 *      3) ci: index of the column, start from 0
 *      4) v: the new coefficient
 * return: void */
void
c512m_gf16_set_at(C512MGF16* m, uint64_t ri, uint64_t ci, gf16_t v) {
    r512m_gf16_set_at(m, ci, ri, v);
}

/* usage: Given a C512MGF16 matrix and both row and column indices, add
 *      the given value to the target entry.
 * params:
 *      1) m: ptr to a struct C512MGF16
 *      2) ri: index of the row, from 0 ~ 63
 *      3) ci: index of the column, start from 0
 *      4) v: the value
 * return: void */
void
c512m_gf16_add_at(C512MGF16* m, uint64_t ri, uint64_t ci, gf16_t v) {
    Grp512GF16* col = c512m_gf16_caddr(m, ci);
    grp512_gf16_add_at(col, ri, v);
}

/* usage: Reset a struct C512MGF16 to zero matrix
 * params:
 *      1) m: ptr to a struct C512MGF16
 * return: void */
void
c512m_gf16_zero(C512MGF16* const m) {
    r512m_gf16_zero(m);
}

/* usage: Print a C512MGF16 matrix
 * params:
 *      1) m: ptr to struct C512MGF16
 * return: void */
void
c512m_gf16_print(const C512MGF16* m) {
    for(uint32_t i = 0; i < 512; ++i) {
        for(uint32_t j = 0; j < c512m_gf16_cnum(m); ++j) {
            printf("%02d ", c512m_gf16_at(m, i, j));
        }
        printf("\n");
    }
}

/* usage: Given a C512MGF16 matrix, find the rows whose selected columns
 *      are fully zero
 * params:
 *      1) m: ptr to struct C512MGF16
 *      2) cidxs: a uint32_t array that stores the indices of the selected
 *          columns
 *      3) sz: size of cidxs
 *      4) out: ptr to a uint512_t which upon return encodes the rows whose
 *          selected columns are full zero. If the i-th row meets the criteria,
 *          then the i-th LSB is set to 1.
 * return: void */
void
c512m_gf16_subset_zr_pos(const C512MGF16* restrict m,
                         const uint32_t* restrict cidxs, uint32_t sz,
                         uint512_t* restrict out) {
    r512m_gf16_subset_zc_pos(m, cidxs, sz, out);
}

/* usage: Given a C512MGF16 matrix, find the rows whose selected columns
 *      are not fully zero
 * params:
 *      1) m: ptr to struct C512MGF16
 *      2) cidxs: a uint32_t array that stores the indices of the selected
 *          columns
 *      3) sz: size of cidxs
 *      4) out: ptr to a uint512_t which upon return encodes the rows whose
 *          selected columns are not full zero. If the i-th row meets the
 *          criteria, then the i-th LSB is set to 1.
 * return: void */
void
c512m_gf16_subset_nzr_pos(const C512MGF16* restrict m,
                          const uint32_t* restrict cidxs, uint32_t sz,
                          uint512_t* restrict out) {
    c512m_gf16_subset_zr_pos(m, cidxs, sz, out);
    uint512_t_negi(out);
}
