#ifndef __RMSM_GENERIC_H__
#define __RMSM_GENERIC_H__

#include <stdint.h>

#include "mdmac.h"
#include "matrix_gf16.h"
#include "thpool.h"

typedef struct RMSMGeneric RMSMGeneric;

/* ========================================================================
 * function prototypes
 * ======================================================================== */

/* usage: given the number of rows and columns, and the total number of
 *      non-zero entries of a matrix, compute the size of struct RMSMGeneric
 *      needed to hold such matrix.
 * params:
 *      1) rnum: number of rows
 *      2) nznum: total number of non-zero entries of the matrix
 * return: size in bytes */
size_t
rmsm_generic_calc_mem_size(uint64_t rn, uint64_t nznum);

/* usage: given a struct RMSMGeneric, return its number of rows
 * params:
 *      1) m: ptr to struct RMSMGeneric
 * return: number of rows */
uint64_t
rmsm_generic_rnum(const RMSMGeneric* m);

/* usage: given a struct RMSMGeneric, return its number of columns
 * params:
 *      1) m: ptr to struct RMSMGeneric
 * return: number of columns */
uint64_t
rmsm_generic_cnum(const RMSMGeneric* m);

/* usage: create and initialize a RMSMGeneric from the selected columns of a
 *      multi-degree Macaulay matrix
 * params:
 *      1) mac: ptr to struct MDMac
 *      2) col_idxs: indices of the columns that should be included
 *      3) sz: size of col_idxs
 *      4) nznum_per_col: a uint32_t array that stores the non-zero entries of
 *          each column of mac
 *      5) nznum: number of non-zero entries in the selected columns
 * return: ptr to struct RMSMGeneric on success, NULL otherwise */
RMSMGeneric*
rmsm_generic_from_mdmac(const MDMac* restrict mac,
                        const uint64_t* restrict col_idxs, uint64_t sz,
                        const uint32_t* restrict nznum_per_col, uint64_t nznum);

/* usage: given a struct RMSMGeneric, release it
 * params:
 *      1) m: ptr to struct RMSMGeneric
 * return: void */
void
rmsm_generic_free(RMSMGeneric* m);

/* usage: given a struct RMSMGeneric, return the selected row
 * params:
 *      1) m: ptr to struct RMSMGeneric
 *      2) i: index of the row
 * return: ptr to struct GFA that points to the selected column */
const GFA*
rmsm_generic_row(const RMSMGeneric* m, uint64_t i);

/* usage: given a struct RMSMGeneric, return the selected entry
 * params:
 *      1) m: ptr to struct RMSMGeneric
 *      2) ri: the row index
 *      3) ci: the column index
 * return: coefficient of the selected entry */
gf_t
rmsm_generic_at(const RMSMGeneric* m, uint64_t ri, uint64_t ci);

/* usage: given a struct RMSMGeneric m and a struct RMGF16 v, compute m * v
 * params:
 *      1) res: ptr to struct RMGF16 for storing the result
 *      2) m: ptr to struct RMSMGeneric
 *      3) v: ptr to struct RMGF16
 * return: void */
void
rmsm_gf16_mul_rm(RMGF16* restrict res, const RMSMGeneric* restrict m,
                 const RMGF16* restrict v);

/* usage: given a struct RMSMGeneric m and a struct RMGF16 v, compute m * v
 *      in parallel
 * params:
 *      1) res: ptr to struct RMGF16 for storing the result
 *      2) m: ptr to struct RMSMGeneric
 *      3) v: ptr to struct RMGF16
 *      4) tnum : number of threads to use
 *      5) args: ptr to an array of struct RMGF16PArg. Must
 *          have size at least as large as the number of threads to use
 *      6) tp: ptr to a struct Threadpool
 * return: void */
void
rmsm_gf16_mul_rm_parallel(RMGF16* restrict res,
                         const RMSMGeneric* restrict m,
                         const RMGF16* restrict v, uint32_t tnum,
                         RMGF16PArg* restrict args,
                         Threadpool* restrict tp);

#endif // __RMSM_GENERIC_H__
