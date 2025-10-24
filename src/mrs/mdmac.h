#ifndef __MDMAC_H__
#define __MDMAC_H__
// multi-degree Macaulay matrix extended from a KS matrix

#include <stdint.h>
#include <stdbool.h>

#include "mdeg.h"
#include "mono.h"
#include "gfa.h"
#include "minrank.h"

typedef struct MDMac MDMac;
typedef struct MDMacColIterator MDMacColIterator;
typedef bool (mdmac_col_iter_cb_t)(const MDeg*);
typedef void (mdmac_iter_rows_cb_t)(uint64_t, uint64_t, void*);

/* ========================================================================
 * function prototypes
 * ======================================================================== */

/* usage: compute the number of columns in a MDMac
 * params:
 *      1) k: number of linear variables
 *      2) r: number of kernel variables in each group
 *      3) mdeg: ptr to struct MDeg, target multi-degree
 * return: void */
uint64_t
mdmac_calc_ncol(uint32_t k, uint32_t r, const MDeg* d);

/* usage: compute the amount of memory needed for a struct MDMac
 * params:
 *      1) k: number of linear variables
 *      2) r: number of kernel variables in each group
 *      3) d: ptr to struct MDeg, target multi-degree
 *      4) m: number of columns of matrices of the input MinRank instance
 *      5) max_tnum: max number of non-zero entries in any rows of the
 *          KS matrix derived from the input MinRank instance
 * return: amount of memory needed in bytes */
size_t
mdmac_calc_memsize(uint32_t k, uint32_t r, const MDeg* d, uint32_t m,
                   uint32_t max_tnum);

/* usage: Given a struct MDMac, return its number of linear variables
 * params:
 *      1) m: ptr to struct MDMac
 * return: number of linear variables */
uint32_t
mdmac_k(const MDMac* m);

/* usage: Given a struct MDMac, return its target rank
 * params:
 *      1) m: ptr to struct MDMac
 * return: target rank */
uint32_t
mdmac_r(const MDMac* m);

/* usage: Given a struct MDMac, return the number of rows its the original KS
 *      matrix
 * params:
 *      1) m: ptr to struct MDMac
 * return: number of rows */
uint32_t
mdmac_c(const MDMac* m);

/* usage: Given a struct MDMac, return the number of columns of matrices in
 *      the original MinRank instance
 * params:
 *      1) m: ptr to struct MDMac
 * return: number of columns */
uint32_t
mdmac_m(const MDMac* m);

/* usage: Given a struct MDMac, return its multi-degree
 * params:
 *      1) m: ptr to struct MDMac
 * return: target multi-degree stored as a uint32_t array of length
 *      mdmac_c(m) + 1, where the 1st integer is the degree of the
 *      linear vars, and the 2nd integer the degree for the first
 *      group of kernel vars, and so on */
const MDeg*
mdmac_mdeg(const MDMac* m);

/* usage: Given a struct MDMac, return its degree
 * params:
 *      1) m: ptr to struct MDMac
 * return: its degree */
uint32_t
mdmac_deg(const MDMac* m);

/* usage: Given a struct MDMac, return its number of rows (equations)
 * params:
 *      1) m: ptr to struct MDMac
 * return: number of rows */
uint64_t
mdmac_nrow(const MDMac* m);

/* usage: Given a struct MDMac, return its number of columns (monomials)
 * params:
 *      1) m: ptr to struct MDMac
 * return: number of columns */
uint64_t
mdmac_ncol(const MDMac* m);

/* usage: Given the number of linear variables, target rank, the number
 *      of rows in the left multiplier, and the multi-degree of the Macaulay
 *      matrix, compute the total number of possible multipliers (monomials)
 * params:
 *      1) k: number of linear variables
 *      2) r: target rank
 *      3) mdeg: ptr to struct MDeg
 * return: total number of multipliers */
uint64_t
mdmac_multiplier_num(uint32_t k, uint32_t r, MDeg* mdeg);

/* usage: Given the max multi-degree, and the current multi-degree, raise the
 *      multi-degree to the next one incrementally
 * params:
 *      1) mdeg: the current multi-degree, which will be updated to the next
 *          one after the function returns
 *      2) max_mdeg: the max multi-degree to consider
 * return: true if the current multi-degree is not the max multi-degree. false
 *      otherwise */
bool
mdmac_mdeg_next(MDeg* restrict mdeg, const MDeg* restrict max_mdeg);

/* usage: Given a multi-degree, pick the first monomial of that multi-degree
 * params:
 *      1) mono: ptr to struct Mono. Container for the first monomial
 *      2) mdeg: ptr to struct MDeg, target multi-degree. Should have the same
 *          degreee as mono
 *      3) k: number of linear variables
 *      4) r: number of kernel variables per row
 * return: void */
void
mdmac_mdeg_first(Mono* restrict mono, const MDeg* restrict mdeg,
                 uint32_t k, uint32_t r);

/* usage: Given a multi-degree, and the current monomial, update the monomial
 *      to the next one
 * params:
 *      1) mono: ptr to struct Mono. Container for the next monomial
 *      2) d: ptr to struct MDeg, target multi-degree. Should have the same
 *          degreee as mono
 *      3) k: number of linear variables
 *      4) r: number of kernel variables per row
 * return: true if the current monomial is not the last one, otherwise false */
bool
mdmac_mdeg_iterate(Mono* restrict mono, const MDeg* restrict d, uint32_t k,
                   uint32_t r);

/* usage: Given the MinRank instance, and the target multi-degree of a Macaulay
 *      matrix, compute the total number of equations in the Macaulay matrix
 * params:
 *      1) mr: ptr to struct MinRank
 *      2) mdeg: ptr to struct MDeg. Target multi-degree
 * return: total number of eqs in the multi-degree Macaulay matrix */
uint64_t
mdmac_eq_num(const MinRank* restrict mr, const MDeg* restrict mdeg);

/* usage: Given a struct MDMac and the row index i, return the i-th row
 *      which represents an eqaution
 * params:
 *      1) m: ptr to struct MDMac
 *      2) i: index of the row
 * return: ptr of struct GFA that stores the i-th row */
const GFA*
mdmac_row(const MDMac* m, uint64_t i);

/* usage: Given a struct MDMac, row index i, and column index j, return
 *      the specified entry
 * params:
 *      1) m: ptr to struct MDMac
 *      2) i: the row index
 *      3) j: the column index
 * return: entry as a gf_t */
gf_t
mdmac_at(const MDMac* m, uint64_t i, uint64_t j);

/* usage: Given a base KS system constructed for a MinRank instance, and a target multi-degree,
 *      compute its multi-degree Macaulay matrix
 * params:
 *      1) ks: ptr to struct GDM that stores the base KS system
 *      2) mr: ptr to struct MinRank that defines the MinRank problem
 *      3) d: ptr to struct MDeg that specifies the target multi-degree
 * return: ptr to struct MDMac */
MDMac*
mdmac_create_from_ks(const GFM* restrict ks, const MinRank* restrict mr, const MDeg* restrict mdeg);

/* usage: Given a struct MDMac, release it
 * params:
 *      1) m: ptr to struct MDMac
 * return: void */
void
mdmac_free(MDMac* m);

/* usage: Given a struct MDMac, dump the number of non-zero entries of each
 *      columns in randomly selected rows
 * params:
 *      1) out: storage for the result. A uint32_t array with size at least
 *          as large as the number of columns
 *      2) m: ptr to struct MDMac
 *      3) nrow: number of random rows to select. Must be <= the number of
 *          rows in MDMac
 *      4) seed: random seed for selecting the rows
 * return: the total number of non-zero entries if success. negative value on
 *      error */
int64_t
mdmac_nznum(uint32_t* restrict out, const MDMac* restrict m, uint64_t nrow,
            int32_t seed);

/* usage: Given a struct MDMac, return the number of columns that correspond to
 *      linear monomials and the constant term.
 * params:
 *      1) m: ptr to struct MDMac
 * return: the number of linear monomials and the constant term. Note that this
 *      is the max number and an MDMac instance might have less in each row
 *      because some of those monomials might be zero */
uint64_t
mdmac_num_linear_col(const MDMac* m);

/* usage: Given a struct MDMac, return the number of columns that correspond to
 *      non-linear monomials
 * params:
 *      1) m: ptr to struct MDMac
 * return: the number of non-linear monomials. Note that this is the max number
 *      and an MDMac instance might have less non-linear monomials in each
 *      row because some of those monomials might be zero */
uint64_t
mdmac_num_nlcol(const MDMac* m);

/* usage: Given a struct MDMac, return the set of column indices that correspond
 *      to the non-linear monomials
 * params:
 *      1) m: ptr to struct MDMac
 *      2) idxs: container for indices
 *      3) sz: size of cidxs. Must >= the number of non-linear monomials
 * return: 0 on success, 1 if cidxs is too small to hold all the indices */
int32_t
mdmac_nlcol_idxs(const MDMac* restrict m, uint64_t* restrict idxs, uint32_t sz);

/* usage: Given a struct MDMac, return the set of column indices that correspond
 *      to the linear monomials
 * params:
 *      1) m: ptr to struct MDMac
 *      2) cidxs: container for indices
 *      3) sz: size of cidxs. Must >= the number of linear monomials
 * return: 0 on success, 1 if cidxs is too small to hold all the indices */
int32_t
mdmac_lcol_idxs(const MDMac* restrict m, uint64_t* restrict idxs, uint32_t sz);

/* usage: map a variable to its column index in the MDMac
 * params:
 *      1) m: ptr to struct MDMac
 *      2) vidx: an index representing the variable. 0 ~ k-1 for
 *          the linear variable, k ~ k + r * c for the kernel
 *          variables
 * return: the column index */
uint64_t
mdmac_vidx_to_midx(const MDMac* restrict m, uint64_t vidx);

/* usage: Given a struct MDMac, print it
 * params:
 *      1) m: ptr to struct MDMac
 * return: void */
void
mdmac_print(const MDMac* m);

/* usage: Given a base KS system constructed for a MinRank instance, and an
 *      array of target multi-degrees, compute a Macaulay matrix whose
 *      monomials satisfy any of the multi-degrees.
 * params:
 *      1) ks: ptr to struct GDM that stores the base KS system
 *      2) mr: ptr to struct MinRank that defines the MinRank problem
 *      3) degs: an array of ptrs to struct MDeg
 *      4) sz: size of degs
 * return: ptr to struct MDMac. NULL on error */
MDMac*
mdmac_combi_create_from_ks(const GFM* restrict ks, const MinRank* restrict mr,
                           const MDeg** restrict degs, uint32_t sz);

/* usage: randomly select rows from a struct MDMac and call a callback function
 *      on each of the selected rows
 * params:
 *      1) full_nrow: number of rows in the full MDMac
 *      2) nrow: number of rows to randomly select
 *      3) seed: seed for the random number generator
 *      4) cb: the callback function, which takes 2 parameters
 *          1st param: how many random rows have been sampled
 *          2nd param: the index of the randomly selected row
 *          3rd param: a generic ptr which can be used to pass arguments to and
 *              retrieve results from the callback function
 *      5) arg: a generic ptr to pass to the callback function
 * return: 0 if success. negative value on error */
int64_t
mdmac_iter_random_rows(uint64_t full_nrow, uint64_t nrow, int32_t seed,
                       mdmac_iter_rows_cb_t* cb, void* arg);

MDMacColIterator*
mdmac_col_iter_create(uint32_t k, uint32_t r, uint32_t c,
                      const MDeg* mdeg, mdmac_col_iter_cb_t* cb);

MDMacColIterator*
mdmac_col_iter_create_from_mdmac(const MDMac* m, mdmac_col_iter_cb_t* cb);

MDMacColIterator*
mdmac_combi_col_iter_create(uint32_t k, uint32_t r, uint32_t c,
                            const MDeg** m_degs, uint32_t degs_sz,
                            const MDeg* mdeg_max, mdmac_col_iter_cb_t* cb);

void
mdmac_col_iter_free(MDMacColIterator* it);

void
mdmac_col_iter_begin(MDMacColIterator* it);

void
mdmac_col_iter_next(MDMacColIterator* it);

bool
mdmac_col_iter_end(const MDMacColIterator* it);

uint64_t
mdmac_col_iter_idx(const MDMacColIterator* it);

void
mdmac_col_iter_set_filter(MDMacColIterator* it, mdmac_col_iter_cb_t* cb);

#endif // __MDMAC_H__
