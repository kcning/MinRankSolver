#include "block_lanczos.h"
#include "r64m_generic.h"

// TODO: define a layer inbetween to hide the block size
// internal containers
#include "rc64m_generic.h"
#include "uint64a.h"

#include <math.h>

/* ========================================================================
 * struct BLKGenericArg definition
 * ======================================================================== */

struct BLKGenericArg {
    R64MGeneric* restrict v;
    R64MGeneric* restrict p;
    R64MGeneric* restrict av;
    R64MGeneric* restrict mtv;
    RC64MGeneric* restrict vtAv;
    RC64MGeneric* restrict vtA2v;
    RC64MGeneric* restrict c;
    RC64MGeneric* restrict w;
    // TODO: containers for parallelization
    uint32_t tnum; // number of threads to use
};

/* ========================================================================
 * function implementations
 * ======================================================================== */

/* usage: Given the rank of the square matrix (or submatrix) to eliminate,
 *      compute the expected number of iterations for Block Lanczos algorithm.
 * params:
 *      1) block_sz: block size
 *      2) q: field size
 *      3) r: rank of the matrix to eliminate
 * return: estimated number of iterations */
uint64_t pure_func
blkgeneric_iter_num(uint64_t block_sz, uint32_t q, uint32_t r) {
    double prob = 1.0 / q;
    double prob_pow_N = pow(prob, block_sz);
    double e1 = 0.0, e2 = prob;

    for(uint32_t i = 2; i <= block_sz; ++i) {
        double e_next = (1 + prob - 2 * prob_pow_N);
        e_next += (1 - prob + prob_pow_N) * e2 + (prob - prob_pow_N) * e1;
        e1 = e2;
        e2 = e_next;
    }
    return r / e2;
}

/* usage: given a struct BLKGenericArg, retrieve the container that stores the
 *      Lanczos vector
 * params:
 *      1) arg: ptr to struct BLKGenericArg
 * return: ptr to struct R64MGeneric which stores the Lanczos vector */
R64MGeneric*
blkgeneric_arg_v(BLKGenericArg* arg) {
    return arg->v;
}

/* usage: create a struct BLKGenericArg, which is a collection of data
 *      structures used by the Block Lanczos algorithm
 * params:
 *      1) rnum: number of rows of the matrix to eliminate
 *      2) cnum: number of columns of the matrix to eliminate
 *      3) tnum: number of threads to use
 * return: ptr to struct BLKGenericArg on success, NULL on error */
BLKGenericArg*
blkgeneric_arg_create(uint32_t rnum, uint32_t cnum, uint32_t tnum) {
    BLKGenericArg* arg = malloc(sizeof(BLKGenericArg));
    if(!arg)
        return NULL;

    memset(arg, 0x0, sizeof(BLKGenericArg)); // set all ptrs to NULL
    if(NULL == (arg->v = r64m_generic_create(rnum))) {
        free(arg);
        return NULL;
    }
    if(NULL == (arg->p = r64m_generic_create(rnum)))
        goto BLKGENERIC_ARG_CREATE_FAIL;
    if(NULL == (arg->av = r64m_generic_create(rnum)))
        goto BLKGENERIC_ARG_CREATE_FAIL;
    if(NULL == (arg->mtv = r64m_generic_create(cnum)))
        goto BLKGENERIC_ARG_CREATE_FAIL;
    if(NULL == (arg->vtAv = rc64m_generic_create()))
        goto BLKGENERIC_ARG_CREATE_FAIL;
    if(NULL == (arg->vtA2v = rc64m_generic_create()))
        goto BLKGENERIC_ARG_CREATE_FAIL;
    if(NULL == (arg->c = rc64m_generic_create()))
        goto BLKGENERIC_ARG_CREATE_FAIL;
    if(NULL == (arg->w = rc64m_generic_create()))
        goto BLKGENERIC_ARG_CREATE_FAIL;

    return arg;
BLKGENERIC_ARG_CREATE_FAIL:
    blkgeneric_arg_free(arg);
    return NULL;
}

/* usage: Given a struct BLKGenericArg, free it
 * params:
 *      1) arg: ptr to struct BLKGenericArg
 * return: void */
void
blkgeneric_arg_free(BLKGenericArg* arg) {
    if(!arg)
        return;
    r64m_generic_free(arg->v);
    r64m_generic_free(arg->p);
    r64m_generic_free(arg->av);
    r64m_generic_free(arg->mtv);
    rc64m_generic_free(arg->vtAv);
    rc64m_generic_free(arg->vtA2v);
    rc64m_generic_free(arg->c);
    rc64m_generic_free(arg->w);
    free(arg);
}

/* subroutine of block_lczs: compute c */
static inline void
block_lczs_cmpc(RC64MGeneric* restrict c, const RC64MGeneric* restrict w,
                const RC64MGeneric* restrict vtAv,
                RC64MGeneric* restrict vtA2v, uint64_t indcols) {
    rc64m_generic_mixi(vtA2v, vtAv, indcols);
    rc64m_generic_mul_naive(c, w, vtA2v);
}

/* subroutine of block_lczs: compute v */
static inline void
block_lczs_cmpv(R64MGeneric* restrict av, const R64MGeneric* restrict v,
                const R64MGeneric* restrict p, const RC64MGeneric* restrict c,
                RC64MGeneric* restrict vtAv, uint64_t di) {
    r64m_generic_mixi(av, v, di); // compute av * di + v * (I - di) and store into av
    r64m_generic_fms_diag(av, p, vtAv, di); // subtract p * vtAv * di
    r64m_generic_fms(av, v, c); // subtract v * c
}

/* usage: Given 2 R64MGeneric p and v, and 1 RCMGeneric w, and a 64-bit integer
 *      that encodes a diagonal matrix di (If the 1st diagonal entry (0, 0) is 1,
 *      then the 1st bit (0x1ULL) is set, and so on), compute v * w + p * (I - di)
 * params:
 *      1) p: ptr to struct R64MGeneric
 *      2) v: ptr to struct R64MGeneric
 *      3) w: ptr to struct RC64MGeneric
 *      4) di: 64-bit integer which encodes a diagonal matrix
 * return: void */
static inline void
block_lczs_cmpp(R64MGeneric* restrict p, const R64MGeneric* restrict v,
                const RC64MGeneric* restrict w, uint64_t di) {
    assert(r64m_generic_rnum(p) == r64m_generic_rnum(v));
    // TODO: it might be possible to compute v * w more efficiently with di,
    // which encodes the non-zero rows/columns of w, since v * w = v * (di * w)
    // = (v * di) * w
    r64m_generic_diag_fma(p, v, w, ~di);
}

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
           Threadpool* restrict tpool) {
    // NOTE: containers for the final results and the intermediate results are
    // allocated and provided by the caller
    // TODO: make use of tpool
    (void) tpool;

    // init: randomize v, and set p = 0
    r64m_generic_rand(arg->v);
    r64m_generic_zero(arg->p);

    uint64_t iter = 0;
    uint64_t indcols = 0;
    do {
        // TODO: compute m^T * v, m * (m^T * v) = Av simultaneously
        cmsm_generic_tr_mul_r64m(arg->mtv, m, arg->v);
        cmsm_generic_mul_r64m(arg->av, m, arg->mtv);

        // compute vtA2v and vtAv
        r64m_generic_gramian(arg->mtv, arg->vtAv);
        r64m_generic_gramian(arg->av, arg->vtA2v);

        // perform Gauss-Jordan on vtAv amd compute w_{inv}
        // TODO: get rid of this copy operation. low priority
        rc64m_generic_copy(arg->c, arg->vtAv); // copy vtAv into tmp (reuse c)
        rc64m_generic_identity(arg->w); // set w to the identity matrix to compute the inverse
        rc64m_generic_gj(arg->c, arg->w, &indcols);

        // compute w_{inv} from w and indcols
        // NOTE: in most iterations, indcols has 64 set bits
        if(unlikely(~indcols)) {
            uint8_t sbidxs[64];
            uint32_t sbnum = uint64_t_sbpos(~indcols, sbidxs);
            for(uint32_t i = 0; i < sbnum; ++i) {
                uint8_t idx = sbidxs[i];
                // TODO: optimize this
                rc64m_generic_zero_row(arg->w, idx);
                rc64m_generic_zero_col(arg->w, idx);
            }
        }
        assert(true == rc64m_generic_is_symmetric(arg->w));

        // compute C_{i+1, i}; note that vtA2v will be modified
        block_lczs_cmpc(arg->c, arg->w, arg->vtAv, arg->vtA2v, indcols); // compute C_{i+1, i}
        // compute vn (stored in Av); note that vtAv will be modified
        block_lczs_cmpv(arg->av, arg->v, arg->p, arg->c, arg->vtAv, indcols); // compute vn (stored in Av)
        // compute pn (stored in p)
        block_lczs_cmpp(arg->p, arg->v, arg->w, indcols); // compute pn (stored in p)

        // swap v and Av
        R64MGeneric* tmp = arg->av;
        arg->av = arg->v;
        arg->v = tmp;

        ++iter;
        //printf("iter: %zu\n", iter);
    } while(likely(indcols));

    return iter;
}
