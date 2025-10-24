/* minrank.h: header file for struct MinRank, representing a MinRank problem */

#ifndef __MINRANK_H__
#define __MINRANK_H__

#include "gfm.h"

typedef struct MinRank MinRank;

/* ========================================================================
 * function prototypes
 * ======================================================================== */

/* usage: Given a struct MinRank, return the number of rows of its matrices
 * params:
 *      1) mr: ptr to MinRank
 * return: number of rows */
uint32_t
minrank_nrow(const MinRank* mr);

/* usage: Given a struct MinRank, return the number of columns of its matrices
 * params:
 *      1) mr: ptr to MinRank
 * return: number of columns */
uint32_t
minrank_ncol(const MinRank* mr);

/* usage: Given a struct MinRank, return its number of matrices
 * params:
 *      1) mr: ptr to MinRank
 * return: number of matrices */
uint32_t
minrank_nmat(const MinRank* mr);

/* usage: Given a struct MinRank, return its target rank
 * params:
 *      1) mr: ptr to MinRank
 * return: the target rank */
uint32_t
minrank_rank(const MinRank* mr);

/* usage: Given a struct MinRank, return its inhomogeneous matrix M0
 * params:
 *      1) mr: ptr to MinRank
 * return: ptr to GFM if the MinRank instance is inhomogeneous, otherwise NULL */
const GFM*
minrank_m0(const MinRank* mr);

/* usage: Given a struct MinRank, return its i-th matrix Mi in the homogeneous part
 * params:
 *      1) mr: ptr to MinRank
 *      2) i: index of the matrix. 1 <= i <= minrank_nmat(mr)
 * return: ptr to GFM, on error NULL */
const GFM*
minrank_matrix(const MinRank* mr, uint32_t i);

/* usage: Given a struct MinRank, return a coefficient in the target matrix
 * params:
 *      1) mr: ptr to MinRank
 *      2) mi: index of the matrix. 0 <= mi <= minrank_nmat(mr)
 *      3) ri: row index
 *      4) ci: column index
 * return: the coefficient in question */
gf_t
minrank_coeff(const MinRank* mr, uint32_t mi, uint32_t ri, uint32_t ci);

/* usage: Given the description of a MinRank problem, create a struct for it
 * params:
 *      1) nrow: number of rows in each matrix
 *      2) ncol: number of columns in each matrix
 *      3) k: number of matrices in the homogeneous part
 *      4) r: target rank
 *      5) m0: optional ptr to struct GFM representing M0. If NULL, the
 *          problem is homogeneous. Note that if not NULL, the ownership of m0 is
 *          transferred to this MinRank instance. One must not call the destructor
 *          on m0. Doing so will result in a double free.
 *      6) ms: optional ptr to an array of struct GFM, representing M1 ~ Mk.
 *          If NULL, the M1 ~ Mk will be randomly sampled. Note that if not
 *          NULL, the ownership of ms is transferred to this MinRank instance.
 *          One must not call the destructor on ms. Doing so will result in a
 *          double free.
 * return: ptr to MinRank, on error return NULL */
MinRank*
minrank_create(uint32_t nrow, uint32_t ncol, uint32_t k, uint32_t r,
               const GFM* restrict m0, const GFM* restrict ms);

/* usage: Release a struct MinRank
 * params:
 *      1) mr: ptr to struct MinRank
 * return: void */
void
minrank_free(MinRank* mr);

/* usage: Given a struct MinRank, compute the number of rows in
 *      M_{\lambda}
 * params:
 *      1) mr: ptr to struct MinRank
 * return: number of rows in M_{\lambda} matrix */
uint32_t
minrank_sum_nrow(const MinRank* mr);

/* usage: Given a struct MinRank, compute the number of columns in
 *      M_{\lambda}
 * params:
 *      1) mr: ptr to struct MinRank
 * return: number of columns in M_{\lambda} matrix */
uint32_t
minrank_sum_ncol(const MinRank* mr);

/* usage: Given a struct MinRank with matrices M0, M1, ..., Mk
 *      create scalar variables x1, x2, ..., xk, compute M_{\lambda} =
 *      M0 + x1M1 + x2M2 + ... + xkMk, and represent the sum as a matrix whose
 *      columns represent x1, ..., xk and the constant term. I.e.:
 *
 *          1  x1  x2 ...  xk
 *      | c00 c01 c02 ... c0k |
 *      | c10 c11 c12 ... c1k |
 *      |       ...           |
 *
 *      Clearly, the resultant matrix has dimension
 *      (mr_nrow(mr) * mr_ncol(mr)) x (k + 1).
 * params:
 *      1) mr: ptr to struct MinRank
 * return: ptr to struct GFM, on error NULL */
GFM*
minrank_sum(const MinRank* mr);

/* usage: Given a struct MinRank, compute the number of rows in its
 *      Kipnis-Shamir matrix
 * params:
 *      1) mr: ptr to struct MinRank
 *      2) c: number of rows in the left multiplier
 *          1 <= c <= minrank_nrow(mr) - minrank_rank(mr)
 * return: number of rows in its Kipnis-Shamir matrix */
static inline uint32_t
minrank_ks_nrow(const MinRank* mr, uint32_t c) {
    return c * minrank_ncol(mr);
}

/* usage: Given a struct MinRank, compute the number of columns in its
 *      Kipnis-Shamir matrix
 * params:
 *      1) mr: ptr to struct MinRank
 *      2) c: number of rows in the left multiplier
 *          1 <= c <= minrank_nrow(mr) - minrank_rank(mr)
 * return: number of columns in its Kipnis-Shamir matrix */
static inline uint32_t
minrank_ks_ncol(const MinRank* mr, uint32_t c) {
    uint32_t n = 1 // constant term
        + minrank_nmat(mr) // linear vars
        + minrank_rank(mr) * c // kernel vars
        + minrank_rank(mr) * c * minrank_nmat(mr); // deg-2 monomials
    return n;
}

/* usage: Given a struct MinRank, compute its Kipnis-Shamir matrix
 * params:
 *      1) mr: ptr to struct MinRank
 *      2) c: number of rows in the left multiplier.
 *          1 <= c <= minrank_nrow(mr) - minrank_rank(mr)
 * return: ptr to struct GFM, on error NULL */
GFM*
minrank_ks(const MinRank* mr, uint32_t c);

#endif // __MINRANK_H__
