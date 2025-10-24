/* gfm.h: define matrices storing elements of GF, up to GF(256) */
#ifndef __GFM_H__
#define __GFM_H__

#include "gf.h"

typedef struct GFM GFM;

/* ========================================================================
 * function prototypes
 * ======================================================================== */

/* usage: Given the size of the matrix and optional its elements, create a GFM
 * params:
 *      1) nrow: number of rows
 *      2) ncol: number of columns
 *      3) vals: optional. Elements of the matrix stored in row-majored format
 * return: ptr to GFM, on error return NULL */
GFM*
gfm_create(uint64_t nrow, uint64_t ncol, const gf_t* vals);

/* usage: Release a struct GFM
 * params:
 *      1) m: ptr to struct GFM
 * return: void */
void
gfm_free(GFM* m);

/* usage: Release an array of struct GFM
 * params:
 *      1) ms: ptr to an array of struct GFM
 *      2) n: size of the array
 * return: void */
void
gfm_arr_free(GFM* ms, uint64_t n);

/* usage: Create an array of struct GFM
 * params:
 *      1) nrow: number of rows
 *      2) ncol: number of columns
 *      3) n: size of the array
 *      4) vals: optional. Elements of the matrices stored in row-majored format
 * return: ptr to GFM, on error return NULL */
GFM*
gfm_arr_create(uint32_t nrow, uint32_t ncol, uint32_t n, const gf_t* vals);

/* usage: Given an array of struct GFM, return its i-th entry
 * params:
 *      1) ms: ptr to an array of struct GFM
 *      2) i: index of target entry
 * return: ptr to target entry */
GFM*
gfm_arr_at(GFM* ms, uint64_t i);

/* usage: Given a struct GFM, set all its coefficients to zero
 * params:
 *      1) m: ptr to struct GFM
 * return: void */
void
gfm_zero(GFM* m);

/* usage: Given a struct GFM, randomize its coefficients
 * params:
 *      1) m: ptr to struct GFM
 * return: void */
void
gfm_rand(GFM* m);

/* usage: Generate an array of randomize GFMs
 * params:
 *      1) nrow: number of rows in each matrix
 *      2) ncol: number of columns in each matrix
 *      3) num: number of matrices
 * return: ptr to an array of GFMs */
GFM*
gfm_rand_matrices(uint64_t nrow, uint64_t ncol, uint64_t num);

/* usage: Generate a randomize GFM
 * params:
 *      1) nrow: number of rows in each matrix
 *      2) ncol: number of columns in each matrix
 * return: ptr to struct GFM */
static inline GFM*
gfm_rand_mat(uint64_t nrow, uint64_t ncol) {
    return gfm_rand_matrices(nrow, ncol, 1);
}

/* usage: Given a struct GFM, return its number of rows
 * params:
 *      1) m: ptr to struct GFM
 * return: number of rows */
uint64_t
gfm_nrow(const GFM* m);

/* usage: Given a struct GFM, return its number of columns
 * params:
 *      1) m: ptr to struct GFM
 * return: number of columns */
uint64_t
gfm_ncol(const GFM* m);

/* usage: Given a struct GFM, return addr to its i-th row as an array of gf_t
 * params:
 *      1) m: ptr to struct GFM
 *      2) ri: index of the row
 * return: ptr to the target row */
const gf_t*
gfm_row_addr(const GFM* m, uint64_t ri);

/* usage: Given a struct GFM and an array of gf_t, copy the array into the i-th
 *      row of the GFM
 * params:
 *      1) m: ptr to struct GFM
 *      2) ri: index of the row
 *      3) row: an array of gf_t whose length >= gfm_ncol(m)
 * return: void */
void
gfm_row_copy_from(GFM* restrict m, uint64_t ri, const gf_t* restrict row);

/* usage: Given a struct GFM and n arrays of gf_t, copy the arrays into the
 *      i-th ~ (i+n)-th rows of the GFM
 * params:
 *      1) m: ptr to struct GFM
 *      2) ri: index of the row
 *      3) n: number of rows to copy
 *      4) rows: an array of size n * gfm_ncol(m) which stores elements of the n arrays
 *              consecutively
 * return: void */
void
gfm_rows_copy_from(GFM* restrict m, uint64_t ri, uint64_t n, const gf_t* restrict rows);

/* usage: Given a struct GFM and both row and column indices, return the coefficient
 *      of the given entry
 * params:
 *      1) m: ptr to struct GFM
 *      2) ri: index of the row
 *      3) ci: index of the column
 * return: coefficient of the target entry */
gf_t
gfm_at(const GFM* m, uint64_t ri, uint64_t ci);

/* usage: Given a struct GFM and both row and column indices, set the coefficient
 *      of the given entry
 * params:
 *      1) m: ptr to struct GFM
 *      2) ri: index of the row
 *      3) ci: index of the column
 *      4) c: coefficient
 * return: void */
void
gfm_set_at(GFM* m, uint64_t ri, uint64_t ci, gf_t c);

/* usage: Given a struct GFM and an array of gf_t, retreat the array as coefficients
 *      stored in row-major format, and copy them into the struct GFM
 * params:
 *      1) m: ptr to struct GFM
 *      2) cs: ptr to gf_t, must be have >= gfm_nrow(m) * gfm_ncol(m) elements
 * return: void */
void
gfm_set_from_arr(GFM* restrict m, const gf_t* restrict cs);

/* usage: Print a struct GFM
 * params:
 *      1) m: ptr to struct GFM
 * return: void */
void
gfm_print(const GFM* m);

/* usage: Given a struct GFM, count the number of zero coefficients in it
 * params:
 *      1) m: ptr to struct GFM
 * return: number of zero coefficients */
uint32_t
gfm_cz(const GFM* m);

/* usage: Given a struct GFM, count the number of nonzero coefficients in it
 * params:
 *      1) m: ptr to struct GFM
 * return: number of zero coefficients */
uint32_t
gfm_cnz(const GFM* m);

/* usage: find the max number of non-zero monomials in any row
 * params:
 *      1) m: ptr to struct GFM
 * return: the max number of non-zero entries in any row */
uint64_t
gfm_find_max_tnum_per_eq(const GFM* m);

#endif // __GFM_H__
