#ifndef __RC64M_GENERIC_H__
#define __RC64M_GENERIC_H__

#include <stdint.h>
#include <stdbool.h>
#include <gf.h>

typedef struct RC64MGeneric RC64MGeneric;

/* ========================================================================
 * function prototypes
 * ======================================================================== */

/* usage: Compute the size of memory needed for RC64MGeneric
 * params: none
 * return: the size of memory needed in bytes */
uint64_t
rc64m_generic_memsize(void);

/* usage: return the address of the selected row in a struct RC64MGeneric
 * params:
 *      1) m: ptr to struct RC64MGeneric
 *      2) i: index of the row
 * return: the row address as a ptr to gf_t */
const gf_t*
rc64m_generic_raddr(const RC64MGeneric* m, uint32_t i);

/* usage: return the selected element in a struct RC64MGeneric
 * params:
 *      1) m: ptr to struct RC64MGeneric
 *      2) i: index of the row
 *      3) j: index of the column
 * return: the (i, j)-th element of the matrix */
gf_t
rc64m_generic_at(const RC64MGeneric* m, uint32_t i, uint32_t j);

/* usage: set the selected element in a struct RC64MGeneric to the given value
 * params:
 *      1) m: ptr to struct RC64MGeneric
 *      2) i: index of the row
 *      3) j: index of the column
 *      4) v: the new value
 * return: void */
void
rc64m_generic_set_at(RC64MGeneric* m, uint32_t i, uint32_t j, gf_t v);

/* usage: Create a RC64MGeneric container. Note the matrix is not initialized.
 * params: none
 * return: a ptr to struct RC64MGeneric. On failure, return NULL */
RC64MGeneric*
rc64m_generic_create(void);

/* usage: Release a struct RC64MGeneric
 * params:
 *      1) m: ptr to a struct RC64MGeneric
 * return: void */
void
rc64m_generic_free(RC64MGeneric* m);

/* usage: Given a struct RC64MGeneric, populate it with random coefficients.
 * params:
 *      1) m: ptr to a struct RC64MGeneric
 * return: void */
void
rc64m_generic_rand(RC64MGeneric* m);

/* usage: Reset a struct RC64MGeneric to zero matrix
 * params:
 *      1) m: ptr to a struct RC64MGeneric
 * return: void */
void
rc64m_generic_zero(RC64MGeneric* const m);

/* usage: Copy a struct RC64MGeneric
 * params:
 *      1) dst: ptr to a struct RC64MGeneric for the copy
 *      2) src: ptr to a struct RC64MGeneric. The source
 * return: void */
void
rc64m_generic_copy(RC64MGeneric* restrict dst, const RC64MGeneric* restrict src);

/* usage: Reset a struct RC64MGeneric to identity matrix
 * params:
 *      1) m: ptr to a struct RC64MGeneric
 * return: void */
void
rc64m_generic_identity(RC64MGeneric* const m);

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
                 uint64_t* restrict di);

/* usage: Given 2 struct RC64MGeneric m and n, compute m*n and store the result into p
 * params:
 *      1) p: ptr to a struct RC64MGeneric
 *      2) m: ptr to a struct RC64MGeneric
 *      3) n: ptr to a struct RC64MGeneric
 * return: void */
void
rc64m_generic_mul_naive(RC64MGeneric* restrict p, const RC64MGeneric* restrict m,
                        RC64MGeneric* const restrict n);

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
                   uint64_t di);

/* usage: Given a RC64MGeneric, zero out its selected row
 * params:
 *      1) m: ptr to struct RC64MGeneric
 *      2) i: index of the row
 * return: void */
void
rc64m_generic_zero_row(RC64MGeneric* m, uint32_t i);

/* usage: Given a RC64MGeneric, set the selected column to zero
 * params:
 *      1) m: ptr to struct RC64MGeneric
 *      2) ci: index of the selected column
 * return: void */
void
rc64m_generic_zero_col(RC64MGeneric* m, uint32_t ci);

/* usage: Given a RC64MGeneric, keep a subset of its columns while setting the
 *      remaining columns to zero
 * params:
 *      1) m: ptr to struct RC64MGeneric
 *      2) di: a 64-bit integer that encodes which columns to keep. If the LSB
 *          is 1, then the first column is kept, and so on
 * return: void */
void
rc64m_generic_zero_cols(RC64MGeneric* m, uint64_t di);

/* usage: Print a RC64MGeneric matrix
 * params:
 *      1) m: ptr to struct RC64MGeneric
 * return: void */
void
rc64m_generic_print(const RC64MGeneric* m);

/* usage: Given a RC64MGeneric, check if it is symmetric
 * params:
 *      1) m: ptr to struct RC64MGeneric
 * return: True if symmetric. False otherwise */
bool
rc64m_generic_is_symmetric(const RC64MGeneric* m);

#endif // __RC64M_GENERIC_H__
