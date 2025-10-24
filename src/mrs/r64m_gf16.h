#ifndef __R64M_GF16_H__
#define __R64M_GF16_H__

#include <stdint.h>
#include "gf16.h"
#include "grp64_gf16.h"
#include "rc64m_gf16.h"

typedef struct R64MGF16 R64MGF16;

/* ========================================================================
 * function prototypes
 * ======================================================================== */

/* usage: Compute the size of memory needed for R64MGF16
 * params:
 *      1) rnum: number of rows
 * return: size of memory needed in bytes */
uint64_t
r64m_gf16_memsize(uint32_t rnum);

/* usage: Create a R64MGF16 matrix. The matrix is not initialized
 * params:
 *      1) rnum: number of rows
 * return: ptr to struct R64MGF16. NULL on faliure */
R64MGF16*
r64m_gf16_create(uint32_t rnum);

/* usage: Release a struct R64MGF16
 * params:
 *      1) m: ptr to struct R64MGF16
 * return: void */
void
r64m_gf16_free(R64MGF16* m);

/* usage: Given a R64MGF16 matrix, return the number of rows
 * params:
 *      1) m: ptr to a struct R64MGF16
 * return: the number of rows */
uint32_t
r64m_gf16_rnum(const R64MGF16* m);

/* usage: Given a R64MGF16 matrix and row index, return the address
 *      to the row
 * params:
 *      1) m: ptr to a struct R64MGF16
 *      2) i: index of the row, starting from 0
 * return: address to the i-th row */
Grp64GF16*
r64m_gf16_raddr(R64MGF16* m, uint32_t i);

/* usage: Given a R64MGF16 matrix and both row and column indices, return
 *      the coefficient of the given entry.
 * params:
 *      1) m: ptr to a struct R64MGF16
 *      2) ri: index of the row, starting from 0
 *      3) ci: index of the column, from 0 ~ 63
 * return: coefficient of the entry */
gf16_t
r64m_gf16_at(const R64MGF16* m, uint32_t ri, uint32_t ci);

/* usage: Given a R64MGF16 matrix and both row and column indices, set
 *      the coefficient of the target entry to the given value.
 * params:
 *      1) m: ptr to a struct R64MGF16
 *      2) ri: index of the row, starting from 0
 *      3) ci: index of the column, from 0 ~ 63
 *      4) v: the new coefficient
 * return: void */
void
r64m_gf16_set_at(R64MGF16* m, uint64_t ri, uint64_t ci, gf16_t v);

/* usage: Reset a struct R64MGF16 to zero matrix
 * params:
 *      1) m: ptr to a struct R64MGF16
 * return: void */
void
r64m_gf16_zero(R64MGF16* const m);

/* usage: Given a struct R64MGF16, populate it with random coefficients.
 * params:
 *      1) m: ptr to a struct R64MGF16
 * return: void */
void
r64m_gf16_rand(R64MGF16* m);

/* usage: Given 2 struct R64MGF16, copy the 2nd R64MGF16 into the 1st one
 * params:
 *      1) dst: ptr to the struct R64MGF16 to copy to
 *      2) src: ptr to the struct R64MGF16 to copy from
 * return: void */
void
r64m_gf16_copy(R64MGF16* restrict dst, const R64MGF16* restrict src);

/* usage: Given a R64MGF16 matrix m and a container, compute the Gramian
 *      matrix of m, i.e. m.transpose() * m, and store it into the container.
 *      Note that the product has dimension 64 x 64.
 * params:
 *      1) m: ptr to a struct R64MGF16
 *      2) p: ptr to a struct RC64MGF16, container for the result
 * return: void */
void
r64m_gf16_gramian(const R64MGF16* restrict m, RC64MGF16* restrict p);

/* usage: Given a R64MGF16, find the columns that are fully zero
 * params:
 *      1) m: ptr to struct R64MGF16
 * return: a 64-bit integer that encodes zero columns. If the first column
 *      is fully zero, the LSB is set to 1, and so on. */
uint64_t
r64m_gf16_zc_pos(const R64MGF16* m);

/* usage: Given a R64MGF16, find the columns whose selected rows are fully zero
 * params:
 *      1) m: ptr to struct R64MGF16
 *      2) ridxs: a uint32_t array that stores the indices of the selected
 *          rows
 *      3) sz: size of ridxs
 * return: a 64-bit integer that encodes zero columns. If the first column
 *      is fully zero, the LSB is set to 1, and so on. */
uint64_t
r64m_gf16_subset_zc_pos(const R64MGF16* restrict m,
                        const uint32_t* restrict ridxs, uint32_t sz);

/* usage: Given a R64MGF16, find the columns that are not fully zero
 * params:
 *      1) m: ptr to struct R64MGF16
 * return: a 64-bit integer that encodes non-zero columns. If the first column
 *      is not fully zero, the LSB is set to 1, and so on. */
uint64_t
r64m_gf16_nzc_pos(const R64MGF16* m);

/* usage: Given 2 R64MGF16 A and B, and a RC64MGF16 C, compute
 *      A + B * C and store the result back into A
 * params:
 *      1) a: ptr to struct R64MGF16, storing the matrix A
 *      2) b: ptr to struct R64MGF16, storing the matrix B
 *      3) c: ptr to struct RC64MGF16, storing the matrix C
 * return: void */
void
r64m_gf16_fma(R64MGF16* restrict a, const R64MGF16* restrict b,
              const RC64MGF16* restrict c);

/* usage: Given 2 R64MGF16 A and B, a RC64MGF16 C, and a 64x64 diagonal
 *      matrix D with coefficients either 1 and 0, compute A + B * C * D
 *      and store the result back into A
 * params:
 *      1) a: ptr to struct R64MGF16, storing the matrix A
 *      2) b: ptr to struct R64MGF16, storing the matrix B
 *      3) c: ptr to struct RC64MGF16, storing the matrix C
 *      4) d: a 64-bit integer that encodes the diagonal matrix D. If the LSB
 *          is 1, then entry (0, 0) of D is 1. Otherwise 0.
 * return: void */
void
r64m_gf16_fma_diag(R64MGF16* restrict a, const R64MGF16* restrict b,
                   const RC64MGF16* restrict c, const uint64_t d);

/* usage: Given 2 R64MGF16 A and B, a RC64MGF16 C, and a 64x64 diagonal
 *      matrix D with coefficients either 1 and 0, compute A * D + B * C
 *      and store the result back into A
 * params:
 *      1) a: ptr to struct R64MGF16, storing the matrix A
 *      2) b: ptr to struct R64MGF16, storing the matrix B
 *      3) c: ptr to struct RC64MGF16, storing the matrix C
 *      4) d: a 64-bit integer that encodes the diagonal matrix D. If the LSB
 *          is 1, then entry (0, 0) of D is 1. Otherwise 0.
 * return: void */
void
r64m_gf16_diag_fma(R64MGF16* restrict a, const R64MGF16* restrict b,
                   const RC64MGF16* restrict c, uint64_t d);

/* usage: Given 2 R64MGF16 A and B, and a RC64MGF16 C, compute
 *      A - B * C and store the result back into A
 * params:
 *      1) a: ptr to struct R64MGF16, storing the matrix A
 *      2) b: ptr to struct R64MGF16, storing the matrix B
 *      3) c: ptr to struct RC64MGF16, storing the matrix C
 * return: void */
void
r64m_gf16_fms(R64MGF16* restrict a, const R64MGF16* restrict b,
              const RC64MGF16* restrict c);

/* usage: Given 2 R64MGF16 A and B, a RC64MGF16 C, and a 64x64 diagonal
 *      matrix D with coefficients either 1 and 0, compute A - B * C * D
 *      and store the result back into A
 * params:
 *      1) a: ptr to struct R64MGF16, storing the matrix A
 *      2) b: ptr to struct R64MGF16, storing the matrix B
 *      3) c: ptr to struct RC64MGF16, storing the matrix C
 *      4) d: a 64-bit integer that encodes the diagonal matrix D. If the LSB
 *          is 1, then entry (0, 0) of D is 1. Otherwise 0.
 * return: void */
void
r64m_gf16_fms_diag(R64MGF16* restrict a, const R64MGF16* restrict b,
                   const RC64MGF16* restrict c, uint64_t d);

/* usage: Given 2 R64MGF16 A and B, replace a subset of columns of A with
 *      corresponding columns of B
 * params:
 *      1) a: ptr to struct R64MGF16, storing the matrix A
 *      2) b: ptr to struct R64MGF16, storing the matrix B
 *      3) di: a 64-bit integer that encodes which columns of A to keep. If
 *          the LSB is 1, then the first column of A is keep. If 0, then the
 *          first column of A is replaced by the first column  of B
 * return: void */
void
r64m_gf16_mixi(R64MGF16* restrict a, const R64MGF16* restrict b, uint64_t di);

/* usage: Given 2 R64MGF16 A and B, compute of A + B and store the result
 *      back into A
 * params:
 *      1) a: ptr to struct R64MGF16, storing the matrix A
 *      2) b: ptr to struct R64MGF16, storing the matrix B
 * return: void */
void
r64m_gf16_addi(R64MGF16* restrict a, const R64MGF16* restrict b);

#endif // __R64M_GF16_H__
