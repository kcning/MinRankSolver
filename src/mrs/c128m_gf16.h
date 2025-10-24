#ifndef __C128M_GF16_H__
#define __C128M_GF16_H__

#include "r128m_gf16.h"
#include "grp128_gf16.h"

/* Just define it as the transpose of R128MGF16 */
typedef R128MGF16 C128MGF16;

/* ========================================================================
 * function prototypes
 * ======================================================================== */

/* usage: Compute the size of memory needed for C128MGF16
 * params:
 *      1) cnum: number of columns
 * return: size of memory needed in bytes */
uint64_t
c128m_gf16_memsize(uint32_t cnum);

/* usage: Create a C128MGF16 matrix. The matrix is not initialized
 * params:
 *      1) cnum: number of columns
 * return: ptr to struct C128MGF16. NULL on failure */
C128MGF16*
c128m_gf16_create(uint32_t cnum);

/* usage: Release a struct C128MGF16
 * params:
 *      1) m: ptr to struct C128MGF16
 * return: void */
void
c128m_gf16_free(C128MGF16* m);

/* usage: Given a C128MGF16 matrix, return the number of columns
 * params:
 *      1) m: ptr to a struct C128MGF16
 * return: the number of columns */
uint32_t
c128m_gf16_cnum(const C128MGF16* m);

/* usage: Given a C128MGF16 matrix and column index, return the address
 *      to the column
 * params:
 *      1) m: ptr to a struct C128MGF16
 *      2) i: index of the column, starting from 0
 * return: address to the i-th column */
Grp128GF16*
c128m_gf16_caddr(C128MGF16* m, uint32_t i);

/* usage: Given a C128MGF16 matrix and both row and column indices, return
 *      the coefficient of the given entry.
 * params:
 *      1) m: ptr to a struct C128MGF16
 *      2) ri: index of the row, from 0 ~ 63
 *      3) ci: index of the column, start from 0
 * return: coefficient of the entry */
gf16_t
c128m_gf16_at(const C128MGF16* m, uint32_t ri, uint32_t ci);

/* usage: Given a C128MGF16 matrix and both row and column indices, set
 *      the coefficient of the target entry to the given value.
 * params:
 *      1) m: ptr to a struct C128MGF16
 *      2) ri: index of the row, from 0 ~ 63
 *      3) ci: index of the column, start from 0
 *      4) v: the new coefficient
 * return: void */
void
c128m_gf16_set_at(C128MGF16* m, uint64_t ri, uint64_t ci, gf16_t v);

/* usage: Given a C128MGF16 matrix and both row and column indices, add
 *      the given value to the target entry.
 * params:
 *      1) m: ptr to a struct C128MGF16
 *      2) ri: index of the row, from 0 ~ 63
 *      3) ci: index of the column, start from 0
 *      4) v: the value
 * return: void */
void
c128m_gf16_add_at(C128MGF16* m, uint64_t ri, uint64_t ci, gf16_t v);

/* usage: Reset a struct C128MGF16 to zero matrix
 * params:
 *      1) m: ptr to a struct C128MGF16
 * return: void */
void
c128m_gf16_zero(C128MGF16* const m);

/* usage: Print a C128MGF16 matrix
 * params:
 *      1) m: ptr to struct C128MGF16
 * return: void */
void
c128m_gf16_print(const C128MGF16* m);

/* usage: Given a C128MGF16 matrix, find the rows whose selected columns
 *      are fully zero
 * params:
 *      1) m: ptr to struct C128MGF16
 *      2) cidxs: a uint32_t array that stores the indices of the selected
 *          columns
 *      3) sz: size of cidxs
 *      4) out: ptr to a uint128_t which upon return encodes the rows whose
 *          selected columns are full zero. If the i-th row meets the criteria,
 *          then the i-th LSB is set to 1.
 * return: void */
void
c128m_gf16_subset_zr_pos(const C128MGF16* restrict m,
                         const uint32_t* restrict cidxs, uint32_t sz,
                         uint128_t* restrict out);

/* usage: Given a C128MGF16 matrix, find the rows whose selected columns
 *      are not fully zero
 * params:
 *      1) m: ptr to struct C128MGF16
 *      2) cidxs: a uint32_t array that stores the indices of the selected
 *          columns
 *      3) sz: size of cidxs
 *      4) out: ptr to a uint128_t which upon return encodes the rows whose
 *          selected columns are not full zero. If the i-th row meets the
 *          criteria, then the i-th LSB is set to 1.
 * return: void */
void
c128m_gf16_subset_nzr_pos(const C128MGF16* restrict m,
                          const uint32_t* restrict cidxs, uint32_t sz,
                          uint128_t* restrict out);

#endif // __C128M_GF16_H__

