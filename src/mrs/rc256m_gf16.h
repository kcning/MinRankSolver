#ifndef __RC256M_GF16_H__
#define __RC256M_GF16_H__

#include <stdint.h>
#include <stdbool.h>
#include "gf16.h"
#include "grp256_gf16.h"

typedef struct RC256MGF16 RC256MGF16;

/* ========================================================================
 * function prototypes
 * ======================================================================== */

/* usage: compute the size of memory needed for struct RC256MGF16
 * return: size in bytes */
uint64_t
rc256m_gf16_memsize(void);

/* usage: return the addr of the selected row in a struct RC256MGF16
 * params:
 *      1) m: ptr to struct RC256MGF16
 *      2) i: index of the row
 * return: the row addr as a ptr to struct Grp256GF16 */
Grp256GF16*
rc256m_gf16_raddr(RC256MGF16* m, uint32_t i);

/* usage: swap 2 rows in a struct RC256MGF16
 * params:
 *      1) m: ptr to struct RC256MGF16
 *      2) i: index of the 1st row
 *      3) j: index of the 2nd row
 * return: void */
void
rc256m_gf16_swap_rows(RC256MGF16* m, uint32_t i, uint32_t j);

/* usage: return the selected element in a struct RC256MGF16
 * params:
 *      1) m: ptr to struct RC256MGF16
 *      2) i: index of the row
 *      3) j: index of the column
 * return: the (i, j)-th element of the matrix */
gf16_t
rc256m_gf16_at(const RC256MGF16* m, uint32_t i, uint32_t j);

/* usage: set the selected element in a struct RC256MGF16 to the given value
 * params:
 *      1) m: ptr to struct RC256MGF16
 *      2) i: index of the row
 *      3) j: index of the column
 *      4) v: the new value
 * return: void */
void
rc256m_gf16_set_at(RC256MGF16* m, uint32_t i, uint32_t j, gf16_t v);

/* usage: Create a RC256MGF16 container. Note the matrix is not initialized.
 * params: none
 * return: a ptr to struct RC256MGF16. On failure, return NULL */
RC256MGF16*
rc256m_gf16_create(void);

/* usage: Release a struct RC256MGF16
 * params:
 *      1) m: ptr to a struct RC256MGF16
 * return: void */
void
rc256m_gf16_free(RC256MGF16* m);

/* usage: Given a struct RC256MGF16, populate it with random coefficients.
 * params:
 *      1) m: ptr to a struct RC256MGF16
 * return: void */
void
rc256m_gf16_rand(RC256MGF16* m);

/* usage: Reset a struct RC256MGF16 to zero matrix
 * params:
 *      1) m: ptr to a struct RC256MGF16
 * return: void */
void
rc256m_gf16_zero(RC256MGF16* m);

/* usage: given a struct RC256MGF16, set a subset of rows and columns to zero
 * params:
 *      1) m: ptr to a struct RC256MGF16
 *      2) d: ptr to uint256_t that encodes which rows and columns to keep.
 *          If the i-th bit is set, then the i-th row and column are kept.
 *          Otherwise, the i-th row and i-th column are cleared to zero.
 * return: void */
void
rc256m_gf16_zero_subset_rc(RC256MGF16* m, const uint256_t* restrict d);

/* usage: Copy a struct RC256MGF16
 * params:
 *      1) dst: ptr to a struct RC256MGF16 for the copy
 *      2) src: ptr to a struct RC256MGF16. The source
 * return: void */
void
rc256m_gf16_copy(RC256MGF16* restrict dst, const RC256MGF16* restrict src);

/* usage: Reset a struct RC256MGF16 to identity matrix
 * params:
 *      1) m: ptr to a struct RC256MGF16
 * return: void */
void
rc256m_gf16_identity(RC256MGF16* m);

/* usage: Given a RC256MGF16 m, perform Gauss-Jordan elimination on it to
 *      identify independent columns, which form an invertible submatrix. The
 *      inverse can also be computed if the caller passes an identity matrix as
 *      inv. Alternatively, if the caller passes the constant column as inv,
 *      the solution of solvable systems can be computed.
 * params:
 *      1) m: ptr to a struct RC256MGF16
 *      2) inv: ptr to a struct RC256MGF16
 *      3) di: ptr to an uint256_t, when the function returns di encodes the
 *              independent columns. If the 1st column is independent, then the
 *              1st bit (0x1ULL) is set, and so on.
 * return: void */
void
rc256m_gf16_gj(RC256MGF16* restrict m, RC256MGF16* restrict inv,
               uint256_t* restrict di);

/* usage: Given 2 struct RC256MGF16 m and n, compute m*n and store the result
 *      into p
 * params:
 *      1) p: ptr to a struct RC256MGF16
 *      2) m: ptr to a struct RC256MGF16
 *      3) n: ptr to a struct RC256MGF16
 * return: void */
void
rc256m_gf16_mul_naive(RC256MGF16* restrict p, const RC256MGF16* restrict m,
                      RC256MGF16* const restrict n);

/* usage: Given 2 RC256MGF16 A and B, replace a subset of columns of A with
 *      corresponding columns of B
 * params:
 *      1) a: ptr to struct RC256MGF16, storing the matrix A
 *      2) b: ptr to struct RC256MGF16, storing the matrix B
 *      3) di: ptr to uint256_t that encodes which columns of A to keep. If
 *          the LSB is 1, then the first column of A is keep. If 0, then the
 *          first column of A is replaced by the first column of B
 * return: void */
void
rc256m_gf16_mixi(RC256MGF16* restrict a, const RC256MGF16* restrict b,
                 const uint256_t* restrict di);

/* usage: Print a RC256MGF16 matrix
 * params:
 *      1) m: ptr to struct RC256MGF16
 * return: void */
void
rc256m_gf16_print(const RC256MGF16* m);

/* usage: Given a RC256MGF16, check if it is symmetric
 * params:
 *      1) m: ptr to struct RC256MGF16
 * return: True if symmetric. False otherwise */
bool
rc256m_gf16_is_symmetric(const RC256MGF16* m);

#endif // __RC256M_GF16_H__
