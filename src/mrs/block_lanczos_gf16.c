#include "block_lanczos_gf16.h"
#include "block_lanczos.h"
#include "matrix_gf16.h"
#include "util.h"
#include "thpool.h"
#include <pthread.h>

/* ========================================================================
 * struct BLKGF16Arg definition
 * ======================================================================== */

struct BLKGF16Arg {
    RMGF16* restrict v;
    RMGF16* restrict p;
    RMGF16* restrict av;
    RMGF16* restrict mtv;
    RCMGF16* restrict vtAv;
    RCMGF16* restrict vtA2v;
    RCMGF16* restrict c;
    RCMGF16* restrict w;
    // containers for parallelization
    RMGF16PArg* restrict pargs;
    RMGF16** restrict av_partials;
    RCMGF16* restrict gramian_partials;
    pthread_mutex_t lock;
    uint32_t tnum; // number of threads to use
};

/* ========================================================================
 * function implementations
 * ======================================================================== */

/* usage: Given the rank of the square matrix (or submatrix) to eliminate,
 *      compute the expected number of iterations for Block Lanczos algorithm.
 * params:
 *      1) block_sz: block size
 *      2) r: rank of the matrix to eliminate
 * return: estimated number of iterations */
uint64_t pure_func
blkgf16_iter_num(uint64_t block_sz, uint64_t r) {
    return blkgeneric_iter_num(block_sz, 16, r);
}

/* usage: given a struct BLKGF16Arg, retrieve the container that stores the
 *      Lanczos vector
 * params:
 *      1) arg: ptr to struct BLKGF16Arg
 * return: ptr to struct RMGF16 which stores the Lanczos vector */
RMGF16*
blkgf16_arg_v(BLKGF16Arg* arg) {
    return arg->v;
}

/* usage: given a struct BLKGF16Arg, retrieve the data structure used for
 *      parallelization.
 * params:
 *      1) arg: ptr to struct BLKGF16Arg
 * return: ptr to struct RMGF16PArg */
RMGF16PArg*
blkgf16_arg_pargs(BLKGF16Arg* arg) {
    return arg->pargs;
}

/* usage: create a struct BLKGF16Arg, which is a collection of data
 *      structures used by the Block Lanczos algorithm
 * params:
 *      1) rnum: number of rows of the matrix to eliminate
 *      2) cnum: number of columns of the matrix to eliminate
 *      3) tnum: number of threads to use
 * return: ptr to struct BLKGF16Arg on success, NULL on error */
BLKGF16Arg*
blkgf16_arg_create(uint64_t rnum, uint64_t cnum, uint32_t tnum) {
    BLKGF16Arg* arg = malloc(sizeof(BLKGF16Arg));
    if(!arg)
        return NULL;

    pthread_mutexattr_t lock_attr;
    if(pthread_mutexattr_init(&lock_attr)) {
        free(arg);
        return NULL;
    }

    memset(arg, 0x0, sizeof(BLKGF16Arg)); // set all ptrs to NULL
    if(pthread_mutexattr_setrobust(&lock_attr, PTHREAD_MUTEX_ROBUST) ||
       pthread_mutex_init(&arg->lock, &lock_attr)) {
        pthread_mutexattr_destroy(&lock_attr);
        free(arg);
        return NULL;
    }

    if(pthread_mutexattr_destroy(&lock_attr))
        goto blkgf16_arg_create_fail;

    if(NULL == (arg->v = rm_gf16_create(rnum)))
        goto blkgf16_arg_create_fail;
    if(NULL == (arg->p = rm_gf16_create(rnum)))
        goto blkgf16_arg_create_fail;
    if(NULL == (arg->av = rm_gf16_create(rnum)))
        goto blkgf16_arg_create_fail;
    if(NULL == (arg->mtv = rm_gf16_create(cnum)))
        goto blkgf16_arg_create_fail;
    if(NULL == (arg->vtAv = rcm_gf16_create()))
        goto blkgf16_arg_create_fail;
    if(NULL == (arg->vtA2v = rcm_gf16_create()))
        goto blkgf16_arg_create_fail;
    if(NULL == (arg->c = rcm_gf16_create()))
        goto blkgf16_arg_create_fail;
    if(NULL == (arg->w = rcm_gf16_create()))
        goto blkgf16_arg_create_fail;
    if(NULL == (arg->pargs = malloc(sizeof(RMGF16PArg) * tnum)))
        goto blkgf16_arg_create_fail;
    if(NULL == (arg->gramian_partials = rcm_gf16_arr_create(tnum)))
        goto blkgf16_arg_create_fail;
    if(NULL == (arg->av_partials = calloc(tnum, sizeof(RMGF16*))))
        goto blkgf16_arg_create_fail;

    for(uint32_t i = 0; i < tnum; ++i) {
        RMGF16* m = rm_gf16_create(rnum);
        if(!m)
            goto blkgf16_arg_create_fail;
        arg->av_partials[i] = m;
    }

    arg->tnum = tnum;
    return arg;

blkgf16_arg_create_fail:
    blkgf16_arg_free(arg);
    return NULL;
}

/* usage: Given a struct BLKGF16Arg, free it
 * params:
 *      1) arg: ptr to struct BLKGF16Arg
 * return: void */
void
blkgf16_arg_free(BLKGF16Arg* arg) {
    if(!arg)
        return;
    // TODO: check return value?
    pthread_mutex_destroy(&arg->lock);
    rm_gf16_free(arg->v);
    rm_gf16_free(arg->p);
    rm_gf16_free(arg->av);
    rm_gf16_free(arg->mtv);
    rcm_gf16_free(arg->vtAv);
    rcm_gf16_free(arg->vtA2v);
    rcm_gf16_free(arg->c);
    rcm_gf16_free(arg->w);
    rcm_gf16_arr_free(arg->gramian_partials);
    if(arg->av_partials) {
        for(uint32_t i = 0; i < arg->tnum; ++i)
            rm_gf16_free(arg->av_partials[i]);
    }
    free(arg->av_partials);
    free(arg->pargs);
    free(arg);
}

static force_inline uint32_t
blk_lczs_gf16_generic(BLKGF16Arg* restrict arg, const CMSMGeneric* restrict cm,
                      Threadpool* restrict tp) {
    // NOTE: containers for the final results and the intermediate results are
    // allocated and provided by the caller

    // init: randomize v, and set p = 0
    rm_gf16_rand(arg->v);
    rm_gf16_zero(arg->p);

    uint64_t iter = 0;
    DiagMGF16 di;
    do {
        cmsm_gf16_tr_mul_rm_parallel(arg->mtv, cm, arg->v, arg->tnum,
                                     arg->pargs, tp);
        cmsm_gf16_mul_rm_parallel(arg->av, cm, arg->mtv, arg->tnum,
                                  arg->av_partials, arg->pargs, tp, &arg->lock);

        // compute vtA2v and vtAv
        rm_gf16_gramian_parallel(arg->mtv, arg->vtAv, arg->tnum,
                                 arg->gramian_partials, arg->pargs, tp);
        rm_gf16_gramian_parallel(arg->av, arg->vtA2v, arg->tnum,
                                 arg->gramian_partials, arg->pargs, tp);

        // perform Gauss-Jordan on vtAv amd compute w_{inv}
        rcm_gf16_copy(arg->c, arg->vtAv); // copy vtAv into tmp (reuse c)
        rcm_gf16_identity(arg->w); // to compute the inverse
        rcm_gf16_gj(arg->c, arg->w, &di);

        // compute w_{inv} from w and indcols
        // NOTE: in most iterations, w has a small rank defect
        if(likely(diagm_gf16_is_not_full_rank(&di)))
            rcm_gf16_zero_subset_rc(arg->w, &di);
        assert(true == rcm_gf16_is_symmetric(arg->w));

        // compute C_{i+1, i}; note that vtA2v will be modified
        rcm_gf16_mixi(arg->vtA2v, arg->vtAv, &di);
        rcm_gf16_mul_naive(arg->c, arg->w, arg->vtA2v);
        // compute vn (stored in Av); note that vtAv will be modified
        rm_gf16_mixi_parallel(arg->av, arg->v, &di, arg->tnum, arg->pargs, tp);
        rm_gf16_fms_diag_parallel(arg->av, arg->p, arg->vtAv, &di, arg->tnum,
                                  arg->pargs, tp);
        rm_gf16_fms_parallel(arg->av, arg->v, arg->c, arg->tnum, arg->pargs, tp);
        // compute pn (stored in p)
        DiagMGF16 ndi; diagm_gf16_negate(&ndi, &di);
        rm_gf16_diag_fma_parallel(arg->p, arg->v, arg->w, &ndi, arg->tnum,
                                  arg->pargs, tp);

        // swap v and Av
        RMGF16* tmp = arg->av;
        arg->av = arg->v;
        arg->v = tmp;

        ++iter;
    } while(likely(diagm_gf16_nonzero(&di)));

    return iter;
}

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
              Threadpool* restrict tpool) {
    return blk_lczs_gf16_generic(arg, cm, tpool);
}
