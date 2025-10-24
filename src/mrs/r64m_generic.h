#ifndef __R64M_GENERIC_H__
#define __R64M_GENERIC_H__

#include <stdint.h>
#include <stdbool.h>

#include "gf.h"
#include "rc64m_generic.h"

typedef struct R64MGeneric R64MGeneric;

/* ========================================================================
 * function prototypes
 * ======================================================================== */

/* usage: Compute the size of memory needed for a row in R64MGeneric
 * params:
 * return: size of memory needed in bytes */
uint64_t
r64m_generic_row_memsize(void);

/* usage: Compute the size of memory needed for R64MGeneric
 * params:
 *      1) rnum: number of rows
 * return: size of memory needed in bytes */
uint64_t
r64m_generic_memsize(uint32_t rnum);

/* usage: Create a R64MGeneric matrix. The matrix is not initialized
 * params:
 *      1) rnum: number of rows
 * return: ptr to struct R64MGeneric. NULL on faliure */
R64MGeneric*
r64m_generic_create(uint32_t rnum);

/* usage: Release a struct R64MGeneric
 * params:
 *      1) m: ptr to struct R64MGeneric
 * return: void */
void
r64m_generic_free(R64MGeneric* m);

/* usage: Given a R64MGeneric matrix, return the number of rows
 * params:
 *      1) m: ptr to a struct R64MGeneric
 * return: the number of rows */
uint32_t
r64m_generic_rnum(const R64MGeneric* m);

/* usage: Given a R64MGeneric matrix and row index, return the row
 * params:
 *      1) m: ptr to a struct R64MGeneric
 *      2) i: index of the row, starting from 0
 *      3) r: a dense gf_t array of size 64 for storing the row
 * return: void */
void
r64m_generic_row(R64MGeneric* restrict m, uint32_t i,
                 gf_t* restrict r);

/* usage: Given a R64MGeneric matrix and row index, return the address
 *      to the row
 * params:
 *      1) m: ptr to a struct R64MGeneric
 *      2) i: index of the row, starting from 0
 * return: address to the i-th row */
gf_t*
r64m_generic_raddr(R64MGeneric* m, uint32_t i);

/* usage: Given a R64MGeneric matrix and both row and column indices, return
 *      the coefficient of the given entry.
 * params:
 *      1) m: ptr to a struct R64MGeneric
 *      2) ri: index of the row, starting from 0
 *      3) ci: index of the column, from 0 ~ 63
 * return: coefficient of the entry */
gf_t
r64m_generic_at(const R64MGeneric* m, uint32_t ri, uint32_t ci);

/* usage: Given a R64MGeneric matrix and both row and column indices, set
 *      the coefficient of the target entry to the given value.
 * params:
 *      1) m: ptr to a struct R64MGeneric
 *      2) ri: index of the row, starting from 0
 *      3) ci: index of the column, from 0 ~ 63
 *      4) v: the new coefficient
 * return: void */
void
r64m_generic_set_at(R64MGeneric* m, uint64_t ri, uint64_t ci, gf_t v);

/* usage: Reset a struct R64MGeneric to zero matrix
 * params:
 *      1) m: ptr to a struct R64MGeneric
 * return: void */
void
r64m_generic_zero(R64MGeneric* const m);

/* usage: Given a struct R64MGeneric, populate it with random coefficients.
 * params:
 *      1) m: ptr to a struct R64MGeneric
 * return: void */
void
r64m_generic_rand(R64MGeneric* m);

/* usage: Given 2 struct R64MGeneric, copy the 2nd R64MGeneric into the 1st one
 * params:
 *      1) dst: ptr to the struct R64MGeneric to copy to
 *      2) src: ptr to the struct R64MGeneric to copy from
 * return: void */
void
r64m_generic_copy(R64MGeneric* restrict dst,
                  const R64MGeneric* restrict src);

/* usage: Given 2 struct R64MGeneric, check if they are the same
 * params:
 *      1) a: ptr to the struct R64MGeneric
 *      2) b: ptr to the struct R64MGeneric
 * return: void */
bool
r64m_generic_is_equal(const R64MGeneric* restrict a,
                      const R64MGeneric* restrict b);

/* usage: Given a R64MGeneric matrix m and a container, compute the Gramian
 *      matrix of m, i.e. m.transpose() * m, and store it into the container.
 *      Note that the product has dimension 64 x 64.
 * params:
 *      1) m: ptr to a struct R64MGeneric
 *      2) p: ptr to a struct RC64MGeneric, container for the result
 * return: void */
void
r64m_generic_gramian(const R64MGeneric* restrict m, RC64MGeneric* restrict p);

/* usage: Given a R64MGeneric, find the columns that are fully zero
 * params:
 *      1) m: ptr to struct R64MGeneric
 * return: a 64-bit integer that encodes zero columns. If the first column
 *      is fully zero, the LSB is set to 1, and so on. */
uint64_t
r64m_generic_zc_pos(const R64MGeneric* m);

/* usage: Given a R64MGeneric, find the columns that are not fully zero
 * params:
 *      1) m: ptr to struct R64MGeneric
 * return: a 64-bit integer that encodes non-zero columns. If the first column
 *      is not fully zero, the LSB is set to 1, and so on. */
uint64_t
r64m_generic_nzc_pos(const R64MGeneric* m);

/* usage : Givena  R64MGeneric, count the number of rows that are fully zero
 * params:
 *      1) m: ptr to struct R64MGeneric
 * return: number of fully zero rows */
uint32_t
r64m_generic_zr_count(const R64MGeneric* m);

/* usage: Given a R64MGeneric, set the selected column to zero
 * params:
 *      1) m: ptr to struct R64MGeneric
 *      2) ci: index of the selected column
 * return: void */
void
r64m_generic_zero_col(R64MGeneric* m, uint32_t ci);

/* usage: Given a R64MGeneric, keep a subset of its columns while setting the
 *      remaining columns to zero
 * params:
 *      1) m: ptr to struct R64MGeneric
 *      2) di: a 64-bit integer that encodes which columns to keep. If the LSB
 *          is 1, then the first column is kept, and so on
 * return: void */
void
r64m_generic_zero_cols(R64MGeneric* m, uint64_t di);

/* usage: Given 2 R64MGeneric A and B, and a RC64MGeneric C, compute
 *      A + B * C and store the result back into A
 * params:
 *      1) a: ptr to struct R64MGeneric, storing the matrix A
 *      2) b: ptr to struct R64MGeneric, storing the matrix B
 *      3) c: ptr to struct RC64MGeneric, storing the matrix C
 * return: void */
void
r64m_generic_fma(R64MGeneric* restrict a, const R64MGeneric* restrict b,
                 const RC64MGeneric* restrict c);

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
                      const RC64MGeneric* restrict c, uint64_t d);

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
                      const RC64MGeneric* restrict c, uint64_t d);

/* usage: Given 2 R64MGeneric A and B, and a RC64MGeneric C, compute
 *      A - B * C and store the result back into A
 * params:
 *      1) a: ptr to struct R64MGeneric, storing the matrix A
 *      2) b: ptr to struct R64MGeneric, storing the matrix B
 *      3) c: ptr to struct RC64MGeneric, storing the matrix C
 * return: void */
void
r64m_generic_fms(R64MGeneric* restrict a, const R64MGeneric* restrict b,
                 const RC64MGeneric* restrict c);

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
                      const RC64MGeneric* restrict c, uint64_t d);

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
                  uint64_t di);

/* usage: Print a R64MGeneric matrix
 * params:
 *      1) m: ptr to struct R64MGeneric
 * return: void */
void
r64m_generic_print(const R64MGeneric* m);

#endif // __R64M_GENERIC_H__
