#ifndef __C64M_GENERIC_H__
#define __C64M_GENERIC_H__

#include "r64m_generic.h"

/* Just define it as the transpose of R64MGeneric */
typedef R64MGeneric C64MGeneric;

/* ========================================================================
 * function prototypes
 * ======================================================================== */

/* usage: Compute the size of memory needed for C64MGeneric
 * params:
 *      1) cnum: number of columns
 * return: size of memory needed in bytes */
uint64_t
c64m_generic_memsize(uint32_t cnum);

/* usage: Create a C64MGeneric matrix. The matrix is not initialized
 * params:
 *      1) cnum: number of columns
 * return: ptr to struct C64MGeneric. NULL on failure */
C64MGeneric*
c64m_generic_create(uint32_t cnum);

/* usage: Release a struct C64MGeneric
 * params:
 *      1) m: ptr to struct C64MGeneric
 * return: void */
void
c64m_generic_free(C64MGeneric* m);

/* usage: Given a C64MGeneric matrix, return the number of columns
 * params:
 *      1) m: ptr to a struct C64MGeneric
 * return: the number of columns */
uint32_t
c64m_generic_cnum(const C64MGeneric* m);

/* usage: Given a C64MGeneric matrix and column index, return the column
 * params:
 *      1) m: ptr to a struct C64MGeneric
 *      2) i: index of the column, starting from 0
 *      3) c: a dense gf_t array of size 64 for storing the column
 * return: void */
void
c64m_generic_col(C64MGeneric* restrict m, uint32_t i,
                 gf_t* restrict c);

/* usage: Given a C64MGeneric matrix and column index, return the address
 *      to the column
 * params:
 *      1) m: ptr to a struct C64MGeneric
 *      2) i: index of the column, starting from 0
 * return: address to the i-th column */
gf_t*
c64m_generic_caddr(C64MGeneric* m, uint32_t i);

/* usage: Given a C64MGeneric matrix and both row and column indices, return
 *      the coefficient of the given entry.
 * params:
 *      1) m: ptr to a struct C64MGeneric
 *      2) ri: index of the row, from 0 ~ 63
 *      3) ci: index of the column, start from 0
 * return: coefficient of the entry */
gf_t
c64m_generic_at(const C64MGeneric* m, uint32_t ri, uint32_t ci);

/* usage: Given a C64MGeneric matrix and both row and column indices, set
 *      the coefficient of the target entry to the given value.
 * params:
 *      1) m: ptr to a struct C64MGeneric
 *      2) ri: index of the row, from 0 ~ 63
 *      3) ci: index of the column, start from 0
 *      4) v: the new coefficient
 * return: void */
void
c64m_generic_set_at(C64MGeneric* m, uint64_t ri, uint64_t ci, gf_t v);

/* usage: Given a C64MGeneric matrix and both row and column indices, add
 *      the given value to the target entry.
 * params:
 *      1) m: ptr to a struct C64MGeneric
 *      2) ri: index of the row, from 0 ~ 63
 *      3) ci: index of the column, start from 0
 *      4) v: the value
 * return: void */
void
c64m_generic_add_at(C64MGeneric* m, uint64_t ri, uint64_t ci, gf_t v);

/* usage: Reset a struct C64MGeneric to zero matrix
 * params:
 *      1) m: ptr to a struct C64MGeneric
 * return: void */
void
c64m_generic_zero(C64MGeneric* const m);

/* usage: Print a C64MGeneric matrix
 * params:
 *      1) m: ptr to struct C64MGeneric
 * return: void */
void
c64m_generic_print(const C64MGeneric* m);

/* usage: Given a C64MGeneric matrix, find the rows whose selected columns
 *      are fully zero
 * params:
 *      1) m: ptr to struct C64MGeneric
 *      2) cidxs: a uint32_t array that stores the indices of the selected
 *          columns
 *      3) sz: size of cidxs
 * return: a uint64_t that encodes the rows whose selected columns are full
 *      zero. If the i-th row meets the criteria, then the i-th LSB is set
 *      to 1. */
uint64_t
c64m_generic_subset_zr_pos(const C64MGeneric* restrict m,
                           const uint32_t* restrict cidxs, uint32_t sz);

/* usage: Given a C64MGeneric matrix, find the rows whose selected columns
 *      are not fully zero
 * params:
 *      1) m: ptr to struct C64MGeneric
 *      2) cidxs: a uint32_t array that stores the indices of the selected
 *          columns
 *      3) sz: size of cidxs
 * return: a uint64_t that encodes the rows whose selected columns are not
 *      fully zero. If the i-th row meets the criteria, then the i-th LSB is
 *      set to 1. */
uint64_t
c64m_generic_subset_nzr_pos(const C64MGeneric* restrict m,
                            const uint32_t* restrict cidxs, uint32_t sz);

#endif // __C64M_GENERIC_H__
