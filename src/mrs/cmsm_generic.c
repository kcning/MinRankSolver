#include "cmsm_generic.h"
#include "gf.h"
#include "gfa.h"
#include "matrix_gf16.h"
#include "mdmac.h"
#include "thpool.h"
#include <pthread.h>

/* ========================================================================
 * struct CMSMGeneric definition
 * ======================================================================== */

struct CMSMGeneric { // column-major sparse matrix for generic GFs
    uint64_t rnum; // number of rows
    uint64_t cnum; // number of columns
    uint64_t nznum; // number of non-zero entries
    uint64_t max_tnum; // max number of non-zero entries in a column
    uint64_t avg_tnum; // avg number of non-zero entries in a column
    GFA* cols;
    gfa_idx_t memblk[]; // memory block used for sparse columns
};

/* ========================================================================
 * function implementations
 * ======================================================================== */

/* usage: given the number of rows and columns, and the total number of
 *      non-zero entries of a matrix, compute the size of internal buffer
 *      needed to hold such matrix.
 * params:
 *      1) nznum: total number of non-zero entries of the matrix
 * return: size in bytes */
static inline size_t
cmsm_generic_calc_buf_size(uint64_t nznum) {
    return sizeof(gfa_idx_t) * nznum;
}

/* usage: given the number of rows and columns, and the total number of
 *      non-zero entries of a matrix, compute the size of struct CMSMGeneric
 *      needed to hold such matrix.
 * params:
 *      1) rnum: number of rows
 *      2) cnum: number of columns
 *      3) nznum: total number of non-zero entries of the matrix
 * return: size in bytes */
size_t
cmsm_generic_calc_mem_size(uint64_t rnum, uint64_t cnum, uint64_t nznum) {
    (void) rnum;
    size_t buf_size = cmsm_generic_calc_buf_size(nznum);
    return sizeof(CMSMGeneric) + buf_size + gfa_memsize() * cnum;
}

/* usage: given a struct CMSMGeneric, return its size
 * params:
 *      1) m: ptr to struct CMSMGeneric
 * return: size in bytes */
size_t
cmsm_generic_mem_size(const CMSMGeneric* m) {
    return cmsm_generic_calc_mem_size(m->rnum, m->cnum, m->nznum);
}

/* usage: given a struct CMSMGeneric, return its number of rows
 * params:
 *      1) m: ptr to struct CMSMGeneric
 * return: number of rows */
uint64_t
cmsm_generic_rnum(const CMSMGeneric* m) {
    return m->rnum;
}

/* usage: given a struct CMSMGeneric, return its number of columns
 * params:
 *      1) m: ptr to struct CMSMGeneric
 * return: number of columns */
uint64_t
cmsm_generic_cnum(const CMSMGeneric* m) {
    return m->cnum;
}

/* usage: given a struct CMSMGeneric, return the max number of non-zero entries
 *      in a column
 *      1) m: ptr to struct CMSMGeneric
 * return: max number of non-zero entries in a column */
uint64_t
cmsm_generic_max_tnum(const CMSMGeneric* m) {
    return m->max_tnum;
}

/* usage: given a struct CMSMGeneric, return the average number of non-zero
 *      entries in a column
 *      1) m: ptr to struct CMSMGeneric
 * return: average number of non-zero entries in a column */
uint64_t
cmsm_generic_avg_tnum(const CMSMGeneric* m) {
    return m->avg_tnum;
}

/* usage: given a struct CMSMGeneric, return the selected column
 * params:
 *      1) m: ptr to struct CMSMGeneric
 *      2) i: index of the column
 * return: ptr to struct GFA that points to the selected column */
static inline const GFA*
cmsm_generic_col(const CMSMGeneric* m, uint64_t i) {
    assert(i < m->cnum);
    return gfa_arr_at(m->cols, i);
}

/* usage: given a struct CMSMGeneric, return the selected entry
 * params:
 *      1) m: ptr to struct CMSMGeneric
 *      2) ri: the row index
 *      3) ci: the column index
 * return: coefficient of the selected entry */
gf_t
cmsm_generic_at(const CMSMGeneric* m, uint64_t ri, uint64_t ci) {
    const GFA* col = cmsm_generic_col(m, ci);
    for(uint64_t i = 0; i < gfa_size(col); ++i) {
        gfa_idx_t idx; gf_t v = gfa_at(col, i, &idx);
        // TODO: check if col is sorted
        if(idx == ri)
            return v;
        else if(idx > ri)
            return 0;
    }
    return 0;
}

/* wrapper for passing arguments to function cmsm_generic_cmp_col_sz_mdmac */
struct __GFASizeArgMDMac {
    MDMacColIterator* restrict it;
    const uint32_t* restrict sizes;
    uint64_t max;
    uint64_t sum;
};

/* subroutine of cmsm_generic_from_mdmac: return the number of non-zero entries
 * in the given column. If a subset of the columns are selected with field 'mask',
 * then the column index specifies a column in the selected subset. Otherewise,
 * the column index specifies a column in all of the columns. */
static gfa_idx_t
cmsm_generic_cmp_col_sz_mdmac(uint64_t col_idx, GFA* e, void* __arg) {
    (void) col_idx; (void) e;
    struct __GFASizeArgMDMac* arg = (struct __GFASizeArgMDMac*) __arg;
    uint32_t sz = arg->sizes[mdmac_col_iter_idx(arg->it)];
    mdmac_col_iter_next(arg->it);
    // NOTE: this function does not init the column

    // compute some stats
    if(sz > arg->max)
        arg->max = sz;
    arg->sum += sz;
    return sz;
}

struct CMSMGenericCtorArg {
    CMSMGeneric* restrict m;
    const uint64_t* restrict rmap;
    const MDMac* restrict mac;
};

static inline void
cmsm_generic_ctor_cb(uint64_t i, uint64_t ridx, void* __arg) {
    struct CMSMGenericCtorArg* arg = __arg;
    const GFA* row = mdmac_row(arg->mac, ridx);
    for(uint64_t j = 0; j < gfa_size(row); ++j) {
        gfa_idx_t idx; gf_t v = gfa_at(row, j, &idx);
        if(arg->rmap[idx] == UINT64_MAX) // the column is not included. skip it
            continue;

        assert(arg->rmap[idx] < arg->m->cnum);
        GFA * target_col = (GFA*) cmsm_generic_col(arg->m, arg->rmap[idx]);
        // NOTE: the ridx is the row index in the full MDMac, while i is
        // the new row index in the set of selected rows
        gfa_set_at(target_col, gfa_size(target_col), i, v);
        gfa_inc_size(target_col);
    }
}

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
                        uint64_t nznum) {
    size_t buf_size = cmsm_generic_calc_buf_size(nznum);
    CMSMGeneric* m = malloc(sizeof(CMSMGeneric) + buf_size);
    if(!m)
        return NULL;

    // TODO: get rid of reverse map
    uint64_t* rmap = malloc(sizeof(uint64_t) * mdmac_ncol(mac));
    if(!rmap) {
        free(m);
        return NULL;
    }

    memset(rmap, 0xFF, sizeof(uint64_t) * mdmac_ncol(mac));
    uint64_t cnum = 0;
    for(mdmac_col_iter_begin(it); !mdmac_col_iter_end(it); mdmac_col_iter_next(it))
        rmap[mdmac_col_iter_idx(it)] = cnum++;

    struct __GFASizeArgMDMac arg = {
        .it = it, .sizes = nznum_per_col, .max = 0, .sum = 0,
    };
    mdmac_col_iter_begin(it);
    m->cols = gfa_arr_create_f(cnum, m->memblk, &arg, cmsm_generic_cmp_col_sz_mdmac);
    if(!m->cols) {
        free(rmap);
        free(m);
        return NULL;
    }

    m->nznum = nznum;
    m->max_tnum = arg.max;
    m->avg_tnum = arg.sum / cnum;
    m->rnum = nrow;
    m->cnum = cnum;

    for(uint64_t i = 0; i < cnum; ++i) {
        GFA * col = (GFA*) cmsm_generic_col(m, i);
        gfa_set_size(col, 0);
    }

    struct CMSMGenericCtorArg ctor_arg = {
        .m = m, .mac = mac, .rmap = rmap
    };
    int64_t rv = mdmac_iter_random_rows(mdmac_nrow(mac), nrow, row_seed,
                                        cmsm_generic_ctor_cb, &ctor_arg);
    free(rmap);
    if(rv) {
        cmsm_generic_free(m);
        return NULL;
    }

#if !defined(NDEBUG)
    mdmac_col_iter_begin(it);
    for(uint64_t i = 0; i < cnum; ++i) {
        uint64_t cidx = mdmac_col_iter_idx(it);
        assert( gfa_size(gfa_arr_at(m->cols, i)) == nznum_per_col[cidx] );
        mdmac_col_iter_next(it);
    }
#endif
    return m;
}

/* wrapper for passing arguments to function cmsm_generic_cmp_col_sz_gf_arr */
struct __GFASizeArgGFArr {
    const gf_t* restrict mat;
    uint64_t rnum;
    uint64_t cnum;
    uint64_t max_sz;
    uint64_t sum_sz;
};

/* subroutine of cmsm_generic_from_gf_arr: return the number of non-zero entries
 * in the given column. */
static gfa_idx_t
cmsm_generic_cmp_col_sz_gf_arr(uint64_t col_idx, GFA* e, void* __arg) {
    struct __GFASizeArgGFArr* arg = (struct __GFASizeArgGFArr*) __arg;
    // init the column and return its size
    uint64_t sz = 0;
    for(uint64_t ri = 0; ri < arg->rnum; ++ri) {
        gf_t v = arg->mat[ri * arg->cnum + col_idx];
        if(v)
            gfa_set_at(e, sz++, ri, v);
    }
    arg->sum_sz += sz;
    if(arg->max_sz < sz)
        arg->max_sz = sz;

    return sz;
}

/* usage: create and initialize a CMSMGeneric from a full matrix
 * params:
 *      1) a: a gf_t array that stores the matrix
 *      2) rnum: number of rows
 *      3) cnum: number of columns
 * return: ptr to struct CMSMGeneric on success, NULL otherwise */
CMSMGeneric*
cmsm_generic_from_gf_arr(const gf_t* a, uint64_t rnum, uint64_t cnum) {
    uint64_t nznum = gf_t_arr_nzc(a, rnum * cnum);
    CMSMGeneric* m = malloc(sizeof(CMSMGeneric) + cmsm_generic_calc_buf_size(nznum));
    if(!m)
        return NULL;

    m->nznum = nznum;
    struct __GFASizeArgGFArr arg = {
        .mat = a,
        .rnum = rnum,
        .cnum = cnum,
        .max_sz = 0,
        .sum_sz = 0,
    };
    m->cols = gfa_arr_create_f(cnum, m->memblk, &arg,
                               cmsm_generic_cmp_col_sz_gf_arr);
    if(!m->cols) {
        free(m);
        return NULL;
    }

    m->max_tnum = arg.max_sz;
    m->avg_tnum = arg.sum_sz / cnum;
    m->rnum = rnum;
    m->cnum = cnum;
    return m;
}

/* usage: given a struct CMSMGeneric, release it
 * params:
 *      1) m: ptr to struct CMSMGeneric
 * return: void */
void
cmsm_generic_free(CMSMGeneric* m) {
    if(!m)
        return;
    gfa_arr_free(m->cols);
    free(m);
}

/* subroutine of cmsm_mul_r64m_geneirc: compute the linear combinations of rows in v
 * based on coefficients in a row */
static inline void
cmp_linear_combi(gf_t* restrict dst, const GFA* restrict row,
                 const R64MGeneric* restrict v) {
    uint64_t head = gfa_size(row) & ~0x1ULL;
    for(uint64_t i = 0; i < head; i += 2) {
        gfa_idx_t ri0; gf_t c0 = gfa_at(row, i + 0, &ri0);
        gfa_idx_t ri1; gf_t c1 = gfa_at(row, i + 1, &ri1);
        gf_t_arr_fmaddi_scalar64(dst, r64m_generic_raddr((R64MGeneric*)v, ri0), c0);
        gf_t_arr_fmaddi_scalar64(dst, r64m_generic_raddr((R64MGeneric*)v, ri1), c1);
    }
    for(uint64_t i = head; i < gfa_size(row); ++i) {
        gfa_idx_t ridx; gf_t c = gfa_at(row, i, &ridx);
        gf_t_arr_fmaddi_scalar64(dst, r64m_generic_raddr((R64MGeneric*)v, ridx), c);
    }
}

/* usage: given a struct CMSMGeneric m and a struct R64MGeneric v, compute m^t * v
 * params:
 *      1) res: ptr to struct R64MGeneric for storing the result
 *      2) m: ptr to struct CMSMGeneric
 *      3) v: ptr to struct R64MGeneric
 * return: void */
void
cmsm_generic_tr_mul_r64m(R64MGeneric* restrict res, const CMSMGeneric* restrict m,
                         const R64MGeneric* restrict v) {
    assert(r64m_generic_rnum(res) == cmsm_generic_cnum(m));
    assert(r64m_generic_rnum(v) == cmsm_generic_rnum(m));

    r64m_generic_zero(res);
    // each row of m^t (column of m) induces a linear combination of rows of v

    // TODO: parallelize this outer loop
    /*
    uint32_t head = round_down_multiple_4(r64m_generic_rnum(res));
    for(uint32_t i = 0; i < head; i+=4) {
        cmp_linear_combi(r64m_generic_raddr(res, i + 0), cmsm_generic_col(m, i + 0), v);
        cmp_linear_combi(r64m_generic_raddr(res, i + 1), cmsm_generic_col(m, i + 1), v);
        cmp_linear_combi(r64m_generic_raddr(res, i + 2), cmsm_generic_col(m, i + 2), v);
        cmp_linear_combi(r64m_generic_raddr(res, i + 3), cmsm_generic_col(m, i + 3), v);
    }
    for(uint32_t i = head; i < r64m_generic_rnum(res); ++i)
        cmp_linear_combi(r64m_generic_raddr(res, i), cmsm_generic_col(m, i), v);
    /*/
    for(uint64_t i = 0; i < r64m_generic_rnum(res); ++i)
        cmp_linear_combi(r64m_generic_raddr(res, i), cmsm_generic_col(m, i), v);
    //*/
}

/* usage: given a struct CMSMGeneric m and a struct R64MGeneric v, compute m * v
 * params:
 *      1) res: ptr to struct R64MGeneric for storing the result
 *      2) m: ptr to struct CMSMGeneric
 *      3) v: ptr to struct R64MGeneric
 * return: void */
void
cmsm_generic_mul_r64m(R64MGeneric* restrict res, const CMSMGeneric* restrict m,
                      const R64MGeneric* restrict v) {
    assert(cmsm_generic_rnum(m) == r64m_generic_rnum(res));
    assert(cmsm_generic_cnum(m) == r64m_generic_rnum(v));

    r64m_generic_zero(res);
    for(uint64_t ci = 0; ci < cmsm_generic_cnum(m); ++ci) { // left multiplication
        const GFA* col = cmsm_generic_col(m, ci);
        const gf_t* v_row = r64m_generic_raddr((R64MGeneric*) v, ci);
        for(uint64_t j = 0; j < gfa_size(col); ++j) {
            gfa_idx_t ridx; gf_t c = gfa_at(col, j, &ridx);
            gf_t_arr_fmaddi_scalar64(r64m_generic_raddr(res,  ridx), v_row, c);
        }
    }
}

/* usage: given a struct CMSMGeneric m and a struct RMGF16 v, compute m * v
 * params:
 *      1) res: ptr to struct RMGF16 for storing the result
 *      2) m: ptr to struct CMSMGeneric
 *      3) v: ptr to struct RMGF16
 * return: void */
void
cmsm_gf16_mul_rm(RMGF16* restrict res, const CMSMGeneric* restrict m,
                 const RMGF16* restrict v) {
    assert(cmsm_generic_rnum(m) == rm_gf16_rnum(res));
    assert(cmsm_generic_cnum(m) == rm_gf16_rnum(v));

    rm_gf16_zero(res);
    for(uint64_t ci = 0; ci < cmsm_generic_cnum(m); ++ci) { // left multiplication
        const GFA* col = cmsm_generic_col(m, ci);
        const RowGF16* v_row = rm_gf16_raddr((RMGF16*) v, ci);

        uint64_t head = gfa_size(col) & ~0x1ULL;
        uint64_t j = 0;
        for(; j < head; j += 2) {
            gfa_idx_t r0; gf_t c0 = gfa_at(col, j, &r0);
            gfa_idx_t r1; gf_t c1 = gfa_at(col, j + 1, &r1);
#if BLK_LANCZOS_BLOCK_SIZE == 64
            Grp64GF16* dst0 = rm_gf16_raddr(res, r0);
            Grp64GF16* dst1 = rm_gf16_raddr(res, r1);
            grp64_gf16_fmaddi_scalar_2x1(dst0, dst1, v_row, c0, c1);
#else
            row_gf16_fmaddi_scalar(rm_gf16_raddr(res,  r0), v_row, c0);
            row_gf16_fmaddi_scalar(rm_gf16_raddr(res,  r1), v_row, c1);
#endif
        }
        if(j < gfa_size(col)) {
            gfa_idx_t ridx; gf_t c = gfa_at(col, j, &ridx);
            row_gf16_fmaddi_scalar(rm_gf16_raddr(res,  ridx), v_row, c);
        }
    }
}

static void
cmsm_gf16_mul_rm_worker(void* __arg) {
    RMGF16PArg* arg = (RMGF16PArg*) __arg;
    const CMSMGeneric* m = (CMSMGeneric*) arg->c;
    RMGF16* res = arg->a;
    RMGF16* v = (RMGF16*) arg->b;
    pthread_mutex_t* lock = arg->ptr;
    RMGF16* partial = (RMGF16*) arg->d;
    assert(arg->eidx >= arg->sidx);

    rm_gf16_zero(partial);
    for(uint64_t ci = arg->sidx; ci < arg->eidx; ++ci) { // left multiplication
        const GFA* col = cmsm_generic_col(m, ci);
        const RowGF16* v_row = rm_gf16_raddr((RMGF16*) v, ci);
        uint64_t head = gfa_size(col) & ~0x1ULL;
        uint64_t j = 0;
        for(; j < head; j += 2) {
            gfa_idx_t r0; gf_t c0 = gfa_at(col, j, &r0);
            gfa_idx_t r1; gf_t c1 = gfa_at(col, j + 1, &r1);
#if BLK_LANCZOS_BLOCK_SIZE == 64
            Grp64GF16* dst0 = rm_gf16_raddr(partial, r0);
            Grp64GF16* dst1 = rm_gf16_raddr(partial, r1);
            grp64_gf16_fmaddi_scalar_2x1(dst0, dst1, v_row, c0, c1);
#else
            row_gf16_fmaddi_scalar(rm_gf16_raddr(partial,  r0), v_row, c0);
            row_gf16_fmaddi_scalar(rm_gf16_raddr(partial,  r1), v_row, c1);
#endif
        }
        if(j < gfa_size(col)) {
            gfa_idx_t ridx; gf_t c = gfa_at(col, j, &ridx);
            row_gf16_fmaddi_scalar(rm_gf16_raddr(partial,  ridx), v_row, c);
        }
    }

    pthread_mutex_lock(lock);
    // TODO: improve this
    rm_gf16_addi(res, partial);
    pthread_mutex_unlock(lock);
}

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
                          pthread_mutex_t* restrict lock) {
    assert(cmsm_generic_rnum(m) == rm_gf16_rnum(res));
    assert(cmsm_generic_cnum(m) == rm_gf16_rnum(v));
    rm_gf16_zero(res);
    uint64_t strip_sz = cmsm_generic_cnum(m) / tnum;
    uint64_t sidx = 0;
    for(uint64_t i = 0; i < (tnum - 1); ++i) {
        args[i].a = res;
        args[i].b = v;
        args[i].c = (RCMGF16*) m; // cast to the correct type later
        args[i].d = (void*) partials[i];
        args[i].ptr = lock;
        args[i].sidx = sidx;
        sidx += strip_sz;
        args[i].eidx = sidx;
    }
    args[tnum-1].a = res;
    args[tnum-1].b = v;
    args[tnum-1].c = (RCMGF16*) m;
    args[tnum-1].d = (void*) partials[tnum-1];
    args[tnum-1].ptr = lock;
    args[tnum-1].sidx = sidx;
    args[tnum-1].eidx = cmsm_generic_cnum(m);

    for(uint32_t i = 0; i < tnum; ++i) {
        thpool_add_job(tp, cmsm_gf16_mul_rm_worker, args + i);
    }
    thpool_wait_jobs(tp);
}

/* usage: given a struct CMSMGeneric m and a struct RMGF16 v, compute m^t * v
 * params:
 *      1) res: ptr to struct RMGF16 for storing the result
 *      2) m: ptr to struct CMSMGeneric
 *      3) v: ptr to struct RMGF16
 * return: void */
void
cmsm_gf16_tr_mul_rm(RMGF16* restrict res, const CMSMGeneric* restrict m,
                    const RMGF16* restrict v) {
    assert(rm_gf16_rnum(res) == cmsm_generic_cnum(m));
    assert(rm_gf16_rnum(v) == cmsm_generic_rnum(m));

    rm_gf16_zero(res);
    // each row of m^t (column of m) induces a linear combination of rows of v
    for(uint64_t i = 0; i < rm_gf16_rnum(res); ++i) {
        const GFA* col = cmsm_generic_col(m, i);
        RowGF16* dst = rm_gf16_raddr(res, i);
        uint64_t head = gfa_size(col) & ~0x1ULL;
        uint64_t j = 0;
        for(; j < head; j += 2) {
            gfa_idx_t r0; gf_t c0 = gfa_at(col, j, &r0);
            gfa_idx_t r1; gf_t c1 = gfa_at(col, j + 1, &r1);
#if BLK_LANCZOS_BLOCK_SIZE == 64
            Grp64GF16* src0 = rm_gf16_raddr((RMGF16*)v, r0);
            Grp64GF16* src1 = rm_gf16_raddr((RMGF16*)v, r1);
            grp64_gf16_fmaddi_scalar_1x2(dst, src0, src1, c0, c1);
#else
            row_gf16_fmaddi_scalar(dst, rm_gf16_raddr((RMGF16*)v, r0), c0);
            row_gf16_fmaddi_scalar(dst, rm_gf16_raddr((RMGF16*)v, r1), c1);
#endif
        }

        if(j < gfa_size(col)) {
            gfa_idx_t ridx; gf_t c = gfa_at(col, j, &ridx);
            row_gf16_fmaddi_scalar(dst, rm_gf16_raddr((RMGF16*)v, ridx), c);
        }
    }
}

static void
cmsm_gf16_tr_mul_rm_worker(void* __arg) {
    RMGF16PArg* arg = (RMGF16PArg*) __arg;
    const CMSMGeneric* m = (CMSMGeneric*) arg->c;
    RMGF16* v = (RMGF16*) arg->b;
    uint64_t i = arg->sidx;
    RowGF16* dst = rm_gf16_raddr(arg->a, i);
    for(; i < arg->eidx; ++i, ++dst) {
        const GFA* col = cmsm_generic_col(m, i);
        uint64_t head = gfa_size(col) & ~0x1ULL;
        uint64_t j = 0;
        for(; j < head; j += 2) {
            gfa_idx_t r0; gf_t c0 = gfa_at(col, j, &r0);
            gfa_idx_t r1; gf_t c1 = gfa_at(col, j + 1, &r1);
#if BLK_LANCZOS_BLOCK_SIZE == 64
            Grp64GF16* src0 = rm_gf16_raddr(v, r0);
            Grp64GF16* src1 = rm_gf16_raddr(v, r1);
            grp64_gf16_fmaddi_scalar_1x2(dst, src0, src1, c0, c1);
#else
            row_gf16_fmaddi_scalar(dst, rm_gf16_raddr(v, r0), c0);
            row_gf16_fmaddi_scalar(dst, rm_gf16_raddr(v, r1), c1);
#endif
        }

        if(j < gfa_size(col)) {
            gfa_idx_t ridx; gf_t c = gfa_at(col, j, &ridx);
            row_gf16_fmaddi_scalar(dst, rm_gf16_raddr(v, ridx), c);
        }
    }
}

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
                             Threadpool* restrict tp) {
    assert(rm_gf16_rnum(res) == cmsm_generic_cnum(m));
    assert(rm_gf16_rnum(v) == cmsm_generic_rnum(m));
    rm_gf16_zero(res);
    uint64_t strip_sz = rm_gf16_rnum(res) / tnum;
    uint64_t sidx = 0;
    for(uint64_t i = 0; i < (tnum - 1); ++i) {
        args[i].a = res;
        args[i].b = v;
        args[i].c = (RCMGF16*) m; // cast to the correct type later
        args[i].sidx = sidx;
        sidx += strip_sz;
        args[i].eidx = sidx;
    }
    args[tnum-1].a = res;
    args[tnum-1].b = v;
    args[tnum-1].c = (RCMGF16*) m;
    args[tnum-1].sidx = sidx;
    args[tnum-1].eidx = rm_gf16_rnum(res);

    for(uint32_t i = 0; i < tnum; ++i) {
        thpool_add_job(tp, cmsm_gf16_tr_mul_rm_worker, args + i);
    }
    thpool_wait_jobs(tp);
}

/* usage: given a CMSMGeneric m, print its enties
 * params:
 *      1) m: ptr to struct CMSMGeneric
 * return: void */
void
cmsm_generic_print(const CMSMGeneric* m) {
    for(uint64_t i = 0; i < cmsm_generic_rnum(m); ++i) {
        for(uint64_t j = 0; j < cmsm_generic_cnum(m); ++j) {
            printf("%02u ", cmsm_generic_at(m, i, j));
        }
        printf("\n");
    }
}

/* usage: given a CMSMGeneric m, print the row indices of non-zero entries in
 * params:
 *      1) m: ptr to struct CMSMGeneric
 * return: void */
void
cmsm_generic_print_ridxs(const CMSMGeneric* m) {
    for(uint64_t ci = 0; ci < cmsm_generic_cnum(m); ++ci) { // left multiplication
        const GFA* col = cmsm_generic_col(m, ci);
        for(uint64_t j = 0; j < gfa_size(col); ++j) {
            gfa_idx_t ridx; gfa_at(col, j, &ridx);
            printf("%02lu ", (uint64_t) ridx);
        }
        printf("\n");
    }
}
