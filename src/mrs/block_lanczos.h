#ifndef __BLOCK_LANCZOS_H__
#define __BLOCK_LANCZOS_H__

#include <stdint.h>

#include "cmsm_generic.h"
#include "thpool.h"
#include "r64m_generic.h" // TODO: define a layer in between to hide the block size
#include "util.h"

typedef struct BLKGenericArg BLKGenericArg;

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
blkgeneric_iter_num(uint64_t block_sz, uint32_t q, uint32_t r);

/* usage: given a struct BLKGenericArg, retrieve the container that stores the
 *      Lanczos vector
 * params:
 *      1) arg: ptr to struct BLKGenericArg
 * return: ptr to struct R64MGeneric which stores the Lanczos vector */
R64MGeneric*
blkgeneric_arg_v(BLKGenericArg* arg);

/* usage: create a struct BLKGenericArg, which is a collection of data
 *      structures used by the Block Lanczos algorithm
 * params:
 *      1) rnum: number of rows of the matrix to eliminate
 *      2) cnum: number of columns of the matrix to eliminate
 *      3) tnum: number of threads to use
 * return: ptr to struct BLKGenericArg on success, NULL on error */
BLKGenericArg*
blkgeneric_arg_create(uint32_t rnum, uint32_t cnum, uint32_t tnum);

/* usage: Given a struct BLKGenericArg, free it
 * params:
 *      1) arg: ptr to struct BLKGenericArg
 * return: void */
void
blkgeneric_arg_free(BLKGenericArg* arg);

// NOTE: containers for the final results and the intermediate results are
// allocated and provided by the caller

/* usage: Given a CMSMGeneric matrix m of size N x L and a BLKGenericArg,
 *      find an RMatrix v such that v^T * m = 0 with Block Lanczos algorithm.
 *      The vector v is stored into the given BLKGenericArg and can be
 *      retrieved by calling `blk_gf_generic_arg_v`.
 * params:
 *      1) arg: ptr to struct BLKGenericArg, which contains data structures used
 *              as buffers for intermediate computation results. Note that the
 *              dimensions of m must equal the parameters used to create arg
 *      2) m: ptr to struct CMSMGeneric
 *      3) tpool: ptr to struct Threadpool
 * return: the number of iterations used to extract v */
uint32_t
block_lczs(BLKGenericArg* restrict arg, const CMSMGeneric* restrict m,
           Threadpool* restrict tpool);

#endif // __BLOCK_LANCZOS_H__
