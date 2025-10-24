#ifndef __BLOCK_LANCZOS_GF16_H__
#define __BLOCK_LANCZOS_GF16_H__

#include <stdint.h>

#include "cmsm_generic.h"
#include "r64m_gf16_parallel.h"
#include "rmsm_generic.h"
#include "thpool.h"
#include "matrix_gf16.h"
#include "util.h"

typedef struct BLKGF16Arg BLKGF16Arg;

/* ========================================================================
 * function prototypes
 * ======================================================================== */

/* usage: Given the rank of the square matrix (or submatrix) to eliminate,
 *      compute the expected number of iterations for Block Lanczos algorithm.
 * params:
 *      1) block_sz: block size
 *      2) q: field size
 *      3) r: rank of the matrix to eliminate
 * return: estimated number of iterations */
uint64_t pure_func
blkgf16_iter_num(uint64_t block_sz, uint64_t r);

/* usage: given a struct BLKGF16Arg, retrieve the container that stores the
 *      Lanczos vector
 * params:
 *      1) arg: ptr to struct BLKGF16Arg
 * return: ptr to struct RMGeneric which stores the Lanczos vector */
RMGF16*
blkgf16_arg_v(BLKGF16Arg* arg);

/* usage: given a struct BLKGF16Arg, retrieve the data structure used for
 *      parallelization.
 * params:
 *      1) arg: ptr to struct BLKGF16Arg
 * return: ptr to struct RMGF16PArg */
RMGF16PArg*
blkgf16_arg_pargs(BLKGF16Arg* arg);

/* usage: create a struct BLKGF16Arg, which is a collection of data
 *      structures used by the Block Lanczos algorithm
 * params:
 *      1) rnum: number of rows of the matrix to eliminate
 *      2) cnum: number of columns of the matrix to eliminate
 *      3) tnum: number of threads to use
 * return: ptr to struct BLKGF16Arg on success, NULL on error */
BLKGF16Arg*
blkgf16_arg_create(uint64_t rnum, uint64_t cnum, uint32_t tnum);

/* usage: Given a struct BLKGF16Arg, free it
 * params:
 *      1) arg: ptr to struct BLKGF16Arg
 * return: void */
void
blkgf16_arg_free(BLKGF16Arg* arg);

/* usage: Given a sparse matrix m stored in column-majored format (CMSMGeneric)
 *      of size N x L and a BLKGF16Arg, find an RMatrix v such that v^T * m = 0
 *      with Block Lanczos algorithm.  The vector v is stored into the given
 *      BLKGF16Arg and can be retrieved by calling `blk_gf_generic_arg_v`.
 * params:
 *      1) arg: ptr to struct BLKGF16Arg, which contains data structures used
 *              as buffers for intermediate computation results. Note that the
 *              dimensions of m must equal the parameters used to create arg
 *      2) cm: ptr to struct CMSMGeneric. rm and cm must contained the same
 *          matrix m
 *      3) tpool: ptr to struct Threadpool
 * return: the number of iterations used to extract v */
uint32_t
blk_lczs_gf16(BLKGF16Arg* restrict arg, const CMSMGeneric* restrict cm,
              Threadpool* restrict tpool);

#endif // __BLOCK_LANCZOS_GF16_H__
