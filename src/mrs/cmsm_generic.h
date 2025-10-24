#ifndef __CMSMATRIX_GENERIC_H__
#define __CMSMATRIX_GENERIC_H__

#include <stdint.h>

#include "mdmac.h"
#include "matrix_gf16.h"
#include "r64m_generic.h"
#include "thpool.h"

typedef struct CMSMGeneric CMSMGeneric;

/* ========================================================================
 * function prototypes
 * ======================================================================== */

/* usage: given the number of rows and columns, and the total number of
 *      non-zero entries of a matrix, compute the size of struct CMSMGeneric
 *      needed to hold such matrix.
 * params:
 *      1) rnum: number of rows
 *      2) cnum: number of columns
 *      3) nznum: total number of non-zero entries of the matrix
 * return: size in bytes */
size_t
cmsm_generic_calc_mem_size(uint64_t rnum, uint64_t cnum, uint64_t nznum);

/* usage: given a struct CMSMGeneric, return its size
 * params:
 *      1) m: ptr to struct CMSMGeneric
 * return: size in bytes */
size_t
cmsm_generic_mem_size(const CMSMGeneric* m);

/* usage: given a struct CMSMGeneric, return its number of rows
 * params:
 *      1) m: ptr to struct CMSMGeneric
 * return: number of rows */
uint64_t
cmsm_generic_rnum(const CMSMGeneric* m);

/* usage: given a struct CMSMGeneric, return its number of columns
 * params:
 *      1) m: ptr to struct CMSMGeneric
 * return: number of columns */
uint64_t
cmsm_generic_cnum(const CMSMGeneric* m);

/* usage: given a struct CMSMGeneric, return the max number of non-zero entries
 *      in a column
 *      1) m: ptr to struct CMSMGeneric
 * return: max number of non-zero entries in a column */
uint64_t
cmsm_generic_max_tnum(const CMSMGeneric* m);

/* usage: given a struct CMSMGeneric, return the average number of non-zero
 *      entries in a column
 *      1) m: ptr to struct CMSMGeneric
 * return: average number of non-zero entries in a column */
uint64_t
cmsm_generic_avg_tnum(const CMSMGeneric* m);

/* usage: given a struct CMSMGeneric, return the selected entry
 * params:
 *      1) m: ptr to struct CMSMGeneric
 *      2) ri: the row index
 *      3) ci: the column index
 * return: coefficient of the selected entry */
gf_t
cmsm_generic_at(const CMSMGeneric* m, uint64_t ri, uint64_t ci);

/* usage: create and initialize a CMSMGeneric from the selected columns of a
 *      multi-degree Macaulay matrix
 * params:
 *      1) mac: ptr to struct MDMac
 *      2) nrow: number of rows to randomly select
 *      3) row_seed: seed for the random number generator for selecting rows
 *      4) it: ptr to struct MDMacColIterator. An iterator that
 *          returns indices of columns that should be included
 *      5) nznum_per_col: a uint32_t array that stores the non-zero entries of
 *          each column of mac
 *      6) nznum: number of non-zero entries in the selected columns
 * return: ptr to struct CMSMGeneric on success, NULL otherwise */
CMSMGeneric*
cmsm_generic_from_mdmac(const MDMac* restrict mac,
                        uint64_t nrow, int32_t row_seed,
                        MDMacColIterator* restrict it,
                        const uint32_t* restrict nznum_per_col,
                        uint64_t nznum);

/* usage: create and initialize a CMSMGeneric from a full matrix
 * params:
 *      1) a: a gf_t array that stores the matrix
 *      2) rnum: number of rows
 *      3) cnum: number of columns
 * return: ptr to struct CMSMGeneric on success, NULL otherwise */
CMSMGeneric*
cmsm_generic_from_gf_arr(const gf_t* a, uint64_t rnum, uint64_t cnum);

/* usage: given a struct CMSMGeneric, release it
 * params:
 *      1) m: ptr to struct CMSMGeneric
 * return: void */
void
cmsm_generic_free(CMSMGeneric* m);

/* usage: given a struct CMSMGeneric m and a struct R64MGeneric v, compute m^t * v
 * params:
 *      1) res: ptr to struct R64MGeneric for storing the result
 *      2) m: ptr to struct CMSMGeneric
 *      3) v: ptr to struct R64MGeneric
 * return: void */
void
cmsm_generic_tr_mul_r64m(R64MGeneric* restrict res, const CMSMGeneric* restrict m,
                         const R64MGeneric* restrict v);

/* usage: given a struct CMSMGeneric m and a struct R64MGeneric v, compute m * v
 * params:
 *      1) res: ptr to struct R64MGeneric for storing the result
 *      2) m: ptr to struct CMSMGeneric
 *      3) v: ptr to struct R64MGeneric
 * return: void */
void
cmsm_generic_mul_r64m(R64MGeneric* restrict res, const CMSMGeneric* restrict m,
                      const R64MGeneric* restrict v);

/* usage: given a struct CMSMGeneric m and a struct RMGF16 v, compute m * v
 * params:
 *      1) res: ptr to struct RMGF16 for storing the result
 *      2) m: ptr to struct CMSMGeneric
 *      3) v: ptr to struct RMGF16
 * return: void */
void
cmsm_gf16_mul_rm(RMGF16* restrict res, const CMSMGeneric* restrict m,
                 const RMGF16* restrict v);

/* usage: given a struct CMSMGeneric m and a struct RMGF16 v, compute m^t * v
 * params:
 *      1) res: ptr to struct RMGF16 for storing the result
 *      2) m: ptr to struct CMSMGeneric
 *      3) v: ptr to struct RMGF16
 * return: void */
void
cmsm_gf16_tr_mul_rm(RMGF16* restrict res, const CMSMGeneric* restrict m,
                    const RMGF16* restrict v);

/* usage: given a struct CMSMGeneric m and a struct RMGF16 v, compute m * v
 *      in parallel.
 * params:
 *      1) res: ptr to struct RMGF16 for storing the result
 *      2) m: ptr to struct CMSMGeneric
 *      3) v: ptr to struct RMGF16
 *      4) tnum : number of threads to use
 *      5) partials: an array of length tnum of ptr to struct RMGF16.
 *          Each RMGF16 must have the same dimension as res. This array is used
 *          to hold partial results during computation and will be modified.
 *      6) args: ptr to an array of struct RMGF16PArg. Must
 *          have size at least as large as the number of threads to use
 *      7) tp: ptr to a struct Threadpool
 *      8) lock: ptr to pthread_mutex_t. Used for sync and must be initialized.
 * return: void */
void
cmsm_gf16_mul_rm_parallel(RMGF16* restrict res, const CMSMGeneric* restrict m,
                          const RMGF16* restrict v, uint32_t tnum,
                          RMGF16** restrict partials, RMGF16PArg* restrict args,
                          Threadpool* restrict tp,
                          pthread_mutex_t* restrict lock);

/* usage: given a struct CMSMGeneric m and a struct RMGF16 v, compute m^t * v
 *      in parallel
 * params:
 *      1) res: ptr to struct RMGF16 for storing the result
 *      2) m: ptr to struct CMSMGeneric
 *      3) v: ptr to struct RMGF16
 *      4) tnum : number of threads to use
 *      5) args: ptr to an array of struct RMGF16PArg. Must
 *          have size at least as large as the number of threads to use
 *      6) tp: ptr to a struct Threadpool
 * return: void */
void
cmsm_gf16_tr_mul_rm_parallel(RMGF16* restrict res,
                             const CMSMGeneric* restrict m,
                             const RMGF16* restrict v, uint32_t tnum,
                             RMGF16PArg* restrict args,
                             Threadpool* restrict tp);

/* usage: given a CMSMGeneric m, print its enties
 * params:
 *      1) m: ptr to struct CMSMGeneric
 * return: void */
void
cmsm_generic_print(const CMSMGeneric* m);

/* usage: given a CMSMGeneric m, print the row indices of non-zero entries in
 *      each column
 * params:
 *      1) m: ptr to struct CMSMGeneric
 * return: void */
void
cmsm_generic_print_ridxs(const CMSMGeneric* m);

#endif // __CMSMATRIX_GENERIC_H__
