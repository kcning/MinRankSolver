#include "rmsm_generic.h"
#include "gfa.h"
#include "matrix_gf16.h"
#include "mdmac.h"

/* ========================================================================
 * struct RMSMGeneric definition
 * ======================================================================== */

struct RMSMGeneric { // row-major sparse matrix for generic GFs
    uint64_t rnum; // number of rows
    uint64_t cnum; // number of columns
    uint64_t nznum; // number of non-zero entries
    uint64_t max_tnum; // max number of non-zero entries in a row
    GFA* rows;
    gfa_idx_t memblk[]; // memory block used for sparse rows
};

/* ========================================================================
 * function implementations
 * ======================================================================== */

/* usage: given the number of rows and columns, and the total number of
 *      non-zero entries of a matrix, compute the size of struct RMSMGeneric
 *      needed to hold such matrix.
 * params:
 *      1) rnum: number of rows
 *      2) nznum: total number of non-zero entries of the matrix
 * return: size in bytes */
size_t
rmsm_generic_calc_mem_size(uint64_t rn, uint64_t nznum) {
    return sizeof(RMSMGeneric) + sizeof(gfa_idx_t) * nznum + gfa_memsize() * rn;
}

/* usage: given a struct RMSMGeneric, return its number of rows
 * params:
 *      1) m: ptr to struct RMSMGeneric
 * return: number of rows */
uint64_t
rmsm_generic_rnum(const RMSMGeneric* m) {
    return m->rnum;
}

/* usage: given a struct RMSMGeneric, return its number of columns
 * params:
 *      1) m: ptr to struct RMSMGeneric
 * return: number of columns */
uint64_t
rmsm_generic_cnum(const RMSMGeneric* m) {
    return m->cnum;
}

/* wrapper for passing arguments to function rmsm_generic_init_row */
typedef struct {
    const MDMac* restrict mac;
    const uint64_t* restrict col_idxs;
    uint64_t size;
    uint64_t max;
    uint64_t sum;
} __RMSMGenericInitRowArg;

/* subroutine of rmsm_generic_from_mdmac: return the number of non-zero entries
 * in the selected columns of a row. The subset of the columns are selected with
 * field 'mask' */
static gfa_idx_t
rmsm_generic_init_row(uint64_t row_idx, GFA* restrict e, void* restrict __arg) {
    __RMSMGenericInitRowArg* arg = (__RMSMGenericInitRowArg*) __arg;
    const GFA* row = mdmac_row(arg->mac, row_idx);

    uint64_t sz = 0;
    uint64_t mapped_idx = 0;
    uint64_t next_cidx = arg->col_idxs[0]; // col_idxs should be sorted
    uint64_t i = 0;
    while(i < gfa_size(row)) {
        gfa_idx_t idx; gf_t v = gfa_at(row, i, &idx);
        if(idx < next_cidx) {
            ++i;
            continue;
        }

        if(idx == next_cidx) {
            // map the column index from the original into a new index in the
            // set of selected columns
            gfa_set_at(e, sz++, mapped_idx, v);
            ++i;
        }

        // at this point, idx >= next_cidx. In either case, we need to search
        // for the next column in the current row
        if( (mapped_idx + 1) >= arg->size) // no more columns to check
            break;
        assert( (mapped_idx + 1) < arg->size);
        next_cidx = arg->col_idxs[++mapped_idx];
    }
    assert(sz <= gfa_size(row));
    //gfa_set_size(e, sz);

    arg->sum += sz; // verify correctness
    if(arg->max < sz)
        arg->max = sz;
    return sz;
}

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
                        const uint32_t* restrict nznum_per_col, uint64_t nznum) {
    (void) nznum_per_col;
    if(!col_idxs || 0 == sz)
        return NULL;

    RMSMGeneric* m = malloc(sizeof(RMSMGeneric) + sizeof(gfa_idx_t) * nznum);
    if(!m)
        return NULL;

    __RMSMGenericInitRowArg arg = {
        .mac = mac,
        .col_idxs = col_idxs,
        .size = sz,
        .max = 0,
        .sum = 0,
    };
    m->rows = gfa_arr_create_f(mdmac_nrow(mac), m->memblk, &arg,
                               rmsm_generic_init_row);
    if(!m->rows) {
        free(m);
        return NULL;
    }

    assert(arg.sum == nznum);
    m->nznum = nznum;
    m->max_tnum = arg.max;
    m->rnum = mdmac_nrow(mac);
    m->cnum = sz;
    return m;
}
/* usage: given a struct RMSMGeneric, return the selected row
 * params:
 *      1) m: ptr to struct RMSMGeneric
 *      2) i: index of the row
 * return: ptr to struct GFA that points to the selected column */
const GFA*
rmsm_generic_row(const RMSMGeneric* m, uint64_t i) {
    return gfa_arr_at(m->rows, i);
}

/* usage: given a struct RMSMGeneric, return the selected entry
 * params:
 *      1) m: ptr to struct RMSMGeneric
 *      2) ri: the row index
 *      3) ci: the column index
 * return: coefficient of the selected entry */
gf_t
rmsm_generic_at(const RMSMGeneric* m, uint64_t ri, uint64_t ci) {
    const GFA* row = rmsm_generic_row(m, ri);
    for(uint64_t i = 0; i < gfa_size(row); ++i) {
        gfa_idx_t idx; gf_t v = gfa_at(row, i, &idx);
        // TODO: check if row is sorted
        if(idx == ci)
            return v;
        else if(idx > ci)
            return 0;
    }
    return 0;
}

/* usage: given a struct RMSMGeneric, release it
 * params:
 *      1) m: ptr to struct RMSMGeneric
 * return: void */
void
rmsm_generic_free(RMSMGeneric* m) {
    if(!m)
        return;
    gfa_arr_free(m->rows);
    free(m);
}

/* usage: given a struct RMSMGeneric m and a struct RMGF16 v, compute m * v
 * params:
 *      1) res: ptr to struct RMGF16 for storing the result
 *      2) m: ptr to struct RMSMGeneric
 *      3) v: ptr to struct RMGF16
 * return: void */
void
rmsm_gf16_mul_rm(RMGF16* restrict res, const RMSMGeneric* restrict m,
                 const RMGF16* restrict v) {
    assert(rmsm_generic_rnum(m) == rm_gf16_rnum(res));
    assert(rmsm_generic_cnum(m) == rm_gf16_rnum(v));
    rm_gf16_zero(res);
    for(uint64_t ri = 0; ri < rmsm_generic_rnum(m); ++ri) {// left multiplication
        const GFA* row =  rmsm_generic_row(m, ri);
        RowGF16* dst = rm_gf16_raddr(res, ri);
        uint64_t head = gfa_size(row) & ~0x1ULL;
        uint64_t j = 0;
        for(; j < head; j += 2) {
            gfa_idx_t r0; gf16_t c0 = gfa_at(row, j, &r0);
            gfa_idx_t r1; gf16_t c1 = gfa_at(row, j + 1, &r1);
            row_gf16_fmaddi_scalar(dst, rm_gf16_raddr((RMGF16*)v, r0), c0);
            row_gf16_fmaddi_scalar(dst, rm_gf16_raddr((RMGF16*)v, r1), c1);
        }

        if(j < gfa_size(row)) {
            gfa_idx_t idx; gf16_t coeff = gfa_at(row, j, &idx);
            row_gf16_fmaddi_scalar(dst, rm_gf16_raddr((RMGF16*)v, idx), coeff);
        }
    }
}

static void
rmsm_gf16_mul_rm_worker(void* __arg) {
    RMGF16PArg* arg = (RMGF16PArg*) __arg;
    const RMSMGeneric* m = (RMSMGeneric*) arg->c;
    RMGF16* v = (RMGF16*) arg->b;
    uint64_t i = arg->sidx;
    RowGF16* dst = rm_gf16_raddr(arg->a, i);
    for(; i < arg->eidx; ++i, ++dst) {
        const GFA*row = rmsm_generic_row(m, i);
        uint64_t head = gfa_size(row) & ~0x1ULL;
        uint64_t j = 0;
        for(; j < head; j += 2) {
            gfa_idx_t r0; gf_t c0 = gfa_at(row, j, &r0);
            gfa_idx_t r1; gf_t c1 = gfa_at(row, j + 1, &r1);
#if BLK_LANCZOS_BLOCK_SIZE == 64
            Grp64GF16* src0 = rm_gf16_raddr(v, r0);
            Grp64GF16* src1 = rm_gf16_raddr(v, r1);
            grp64_gf16_fmaddi_scalar_1x2(dst, src0, src1, c0, c1);
#else
            row_gf16_fmaddi_scalar(dst, rm_gf16_raddr(v, r0), c0);
            row_gf16_fmaddi_scalar(dst, rm_gf16_raddr(v, r1), c1);
#endif
        }

        if(j < gfa_size(row)) {
            gfa_idx_t ridx; gf_t c = gfa_at(row, j, &ridx);
            row_gf16_fmaddi_scalar(dst, rm_gf16_raddr(v, ridx), c);
        }
    }
}

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
                          Threadpool* restrict tp) {
    assert(rmsm_generic_rnum(m) == rm_gf16_rnum(res));
    assert(rmsm_generic_cnum(m) == rm_gf16_rnum(v));
    rm_gf16_zero(res);
    uint64_t strip_sz = rm_gf16_rnum(res) / tnum;
    uint64_t sidx = 0;
    for(uint32_t i = 0; i < (tnum - 1); ++i) {
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
        thpool_add_job(tp, rmsm_gf16_mul_rm_worker, args + i);
    }
    thpool_wait_jobs(tp);
}
