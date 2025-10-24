#include <options.h>
#include <gf.h>
#include <gfa.h>
#include <matrix_gf16.h>
#include <thpool.h>
#include <util.h>
#include <gfm.h>
#include <minrank.h>
#include <ks.h>
#include <mdeg.h>
#include <mdmac.h>
#include <cmsm_generic.h>
#include <block_lanczos_gf16.h>
#include <blake2s.h>
#include <hmap.h>
#include <loader.h>
#include <stdint.h>
#include <time.h>
#include <stdlib.h>

// for solving the final linear system
#include <rc64m_gf16.h>
#include <rc128m_gf16.h>
#include <rc256m_gf16.h>
#include <rc512m_gf16.h>

// sc for solution container
uint32_t g_sc_size;
void* (*g_sc_create)(void);
void (*g_sc_zero)(void*);
void* (*g_sc_raddr)(void*, uint32_t);
gf16_t (*g_sc_at)(void*, uint32_t, uint32_t);
void (*g_sc_gj_internal)(void*, void*, void*);
void (*g_sc_free)(void*);
uint64_t (*g_sc_uint_at)(void*, uint32_t);
void (*g_sc_row_set_at)(void*, uint32_t, gf16_t);

typedef union {
    uint512_t b512;
    uint256_t b256;
    uint128_t b128;
    uint64_t b64;
} sc_di_t;

static inline void
g_sc_gj(void* restrict sc, void* restrict inv, sc_di_t* restrict di) {
    void* ptr = NULL;
    if(g_sc_size == 512) {
        ptr = &(di->b512);
    } else if (g_sc_size == 256) {
        ptr = &(di->b256);
    } else if (g_sc_size == 128) {
        ptr = &(di->b128);
    } else {
        ptr = &(di->b64);
    }
    g_sc_gj_internal(sc, inv, ptr);
}

static inline uint64_t
g_sc_popcnt(sc_di_t* di) {
    if(g_sc_size == 512)
        return uint512_t_popcount(&di->b512);
    else if (g_sc_size == 256)
        return uint256_t_popcount(&di->b256);
    else if (g_sc_size == 128)
        return uint128_t_popcount(&di->b128);
    else
        return uint64_popcount(di->b64);
}

static inline void
init_sc_funcs(uint32_t remaining_ncols) {
    assert(remaining_ncols <= 512);
    if(remaining_ncols > 256) {
        g_sc_size = 512;
        g_sc_create = (void*(*)(void)) rc512m_gf16_create;
        g_sc_zero = (void(*)(void*)) rc512m_gf16_zero;
        g_sc_at = (gf16_t(*)(void*, uint32_t, uint32_t)) rc512m_gf16_at;
        g_sc_raddr = (void*(*)(void*, uint32_t)) rc512m_gf16_raddr;
        g_sc_gj_internal = (void(*)(void*, void*, void*)) rc512m_gf16_gj;
        g_sc_free = (void(*)(void*)) rc512m_gf16_free;
        g_sc_uint_at = (uint64_t(*)(void*, uint32_t)) uint512_t_at;
        g_sc_row_set_at = (void(*)(void*, uint32_t, gf16_t)) grp512_gf16_set_at;
    } else if(remaining_ncols > 128) {
        g_sc_size = 256;
        g_sc_create = (void*(*)(void)) rc256m_gf16_create;
        g_sc_zero = (void(*)(void*)) rc256m_gf16_zero;
        g_sc_at = (gf16_t(*)(void*, uint32_t, uint32_t)) rc256m_gf16_at;
        g_sc_raddr = (void*(*)(void*, uint32_t)) rc256m_gf16_raddr;
        g_sc_gj_internal = (void(*)(void*, void*, void*)) rc256m_gf16_gj;
        g_sc_free = (void(*)(void*)) rc256m_gf16_free;
        g_sc_uint_at = (uint64_t(*)(void*, uint32_t)) uint256_t_at;
        g_sc_row_set_at = (void(*)(void*, uint32_t, gf16_t)) grp256_gf16_set_at;
    } else if(remaining_ncols > 64) {
        g_sc_size = 128;
        g_sc_create = (void*(*)(void)) rc128m_gf16_create;
        g_sc_zero = (void(*)(void*)) rc128m_gf16_zero;
        g_sc_at = (gf16_t(*)(void*, uint32_t, uint32_t)) rc128m_gf16_at;
        g_sc_raddr = (void*(*)(void*, uint32_t)) rc128m_gf16_raddr;
        g_sc_gj_internal = (void(*)(void*, void*, void*)) rc128m_gf16_gj;
        g_sc_free = (void(*)(void*)) rc128m_gf16_free;
        g_sc_uint_at = (uint64_t(*)(void*, uint32_t)) uint128_t_at;
        g_sc_row_set_at = (void(*)(void*, uint32_t, gf16_t)) grp128_gf16_set_at;
    } else {
        g_sc_size = 64;
        g_sc_create = (void*(*)(void)) rc64m_gf16_create;
        g_sc_zero = (void(*)(void*)) rc64m_gf16_zero;
        g_sc_at = (gf16_t(*)(void*, uint32_t, uint32_t)) rc64m_gf16_at;
        g_sc_raddr = (void*(*)(void*, uint32_t)) rc64m_gf16_raddr;
        g_sc_gj_internal = (void(*)(void*, void*, void*)) rc64m_gf16_gj;
        g_sc_free = (void(*)(void*)) rc64m_gf16_free;
        g_sc_uint_at = (uint64_t(*)(void*, uint32_t)) uint64_t_at;
        g_sc_row_set_at = (void(*)(void*, uint32_t, gf16_t)) grp64_gf16_set_at;
    }
}

// TODO: fix this estimation
#define LANCZOS_MAX_ITER    (0x1ULL << 3)

/* ========================================================================
 * main
 * ======================================================================== */

/* subroutine of main: find the non-zero vectors in v^T that lead to a linear
 *      combination of rows of cmsm that is zero. The indices of those vectors
 *      are encoded in a DiagMGF16* */
static inline void
verify_nullvec(DiagMGF16* restrict out, RMGF16* restrict p,
               const CMSMGeneric* restrict cmsm, const RMGF16* restrict v) {
    DiagMGF16 zv, zp;
    rm_gf16_zc_pos(v, &zv); // find zero vectors
    cmsm_gf16_tr_mul_rm(p, cmsm, v); // compute v^t * cm (i.e. cm^t * v)
    rm_gf16_zc_pos(p, &zp);
    diagm_gf16_andn(out, &zp, &zv);
}

static inline void
store_vec(void* p, void* sol, uint32_t dst_idx, uint32_t remaining_ncol,
          gf16_t* vec_buf) {
    void* dst = g_sc_raddr(p, dst_idx);
    void* sol_dst = g_sc_raddr(sol, dst_idx);
    g_sc_row_set_at(sol_dst, 0, vec_buf[0]); // constant term
    for(uint32_t k = 1; k < remaining_ncol; ++k) { // variables
        g_sc_row_set_at(dst, k-1, vec_buf[k]);
    }
}

/* subroutine of main: given the positions of non-trivial nullvectors, compute linear
 *      combinations based on them and store the non-duplicate results */
static inline uint32_t
proc_nullvec(Hmap* restrict hmap, void* restrict p, void* restrict sol,
             RMGF16* restrict prod, const RMGF16* restrict v,
             const CMSMGeneric* restrict cmsm_kept, uint32_t tnum,
             RMGF16PArg* restrict args, Threadpool* restrict tp,
             const uint64_t* restrict vmap, MDMacColIterator* restrict it,
#ifdef BLK_LANCZOS_COLLECT_STATS
             uint32_t remaining_ncol, uint64_t* restrict full_count,
             uint64_t* restrict dup_count) {
#else
            uint32_t remaining_ncol) {
#endif
    cmsm_gf16_tr_mul_rm_parallel(prod, cmsm_kept, v, tnum, args, tp);
    // positions of nullvectors that are in the left kernel
   DiagMGF16 valid_nv_pos;
   // NOTE: we simply assume all nullvectors are in the left kernel of
   // the submatrix to eliminate, since heuristically they are.
    rm_gf16_nzc_pos(prod, &valid_nv_pos);

    if(unlikely(diagm_gf16_is_zero(&valid_nv_pos)))
        return 0;

    uint8_t digest[BLAKE2S_HASH_SIZE];
    gf_t vec_buf[remaining_ncol];
    assert(remaining_ncol <= g_sc_size);
    const uint32_t ori_nvcount = hmap_cur_size(hmap);

    for(uint32_t i = 0; i < g_sc_size; ++i) {
        if( !diagm_gf16_at(&valid_nv_pos, i) )
            continue;

        uint32_t dst_idx = hmap_cur_size(hmap);
        if(dst_idx >= g_sc_size) // enough nullvecs
            break;

        // extract the result of linear combi
        for(uint32_t j = 0; j < remaining_ncol; ++j) {
            // To this this, we need to map variable_index into column index in
            // cmsm_kept. We do this with vmap, which maps from variable index
            // to column index in MDMac. remain_cidxs maps from column index
            // in cmsm_kept into column index in MDMac, so we can find column
            // index in cmsm_kept when there's a match. remaining_ncol are
            // typically small.
            // TODO: this is ugly and inefficient
            mdmac_col_iter_begin(it);
            uint64_t col_idx = 0;
            for(; col_idx < remaining_ncol; ++col_idx) {
                if(mdmac_col_iter_idx(it) == vmap[j])
                    break;
                mdmac_col_iter_next(it);
            }
            assert(col_idx != remaining_ncol);
            vec_buf[j] = rm_gf16_at(prod, col_idx, i);
        }

        blake2s(digest, vec_buf, NULL, BLAKE2S_HASH_SIZE, sizeof(gf_t) * remaining_ncol, 0);
        // check if this linear combi has been extracted
        switch(hmap_insert(hmap, digest, NULL)) {
            case HMAP_INSERT_FULL:
                // the bin is full, but other bins might be fine
#ifdef BLK_LANCZOS_COLLECT_STATS
                ++(*full_count);
#endif
                break;
            case HMAP_INSERT_DUP:
#ifdef BLK_LANCZOS_COLLECT_STATS
                ++(*dup_count);
#endif
                break;
            case HMAP_INSERT_SUC:
                // store the linera combi
                store_vec(p, sol, dst_idx, remaining_ncol, vec_buf);
                break;
            default:
                // do nothing
                break;
        }
    }

    return hmap_cur_size(hmap) - ori_nvcount;
}

static inline void
print_sol(void* restrict sol, void* restrict di,
          uint32_t k, uint32_t r, uint32_t c) {
    uint32_t total_vnum = ks_total_var_num(k, r, c);
    for(uint32_t i = total_vnum; i < g_sc_size; ++i) {
        if(g_sc_at(sol, i, 0)) {
            printf_ts("[+] The system has no solution\n");
            break;
        }
    }

    printf_ts("[+] Solution:\n");
    printf("\t\tlinear variables:\n");
    for(uint32_t i = 0; i < k; ++i) { // for each linear var
        if(g_sc_uint_at(di, i)) {
            printf("\t\tlambda_%u = %u\n", i, g_sc_at(sol, i, 0));
        } else {
            printf("\t\tlambda_%u = free variable\n", i);
        }
    }
    printf("\t\tkernel variables:\n");
    for(uint32_t i = k; i < total_vnum; ++i) { // for each kernel var
        uint32_t tmp[2];
        ks_kernel_var_idx_to_2d(tmp, i, k, r);
        if(g_sc_uint_at(di, i)) {
            printf("\t\tx(%u, %u) = %u\n", tmp[0], tmp[1], g_sc_at(sol, i, 0));
        } else {
            printf("\t\tx(%u, %u) = free variable\n", tmp[0], tmp[1]);
        }
    }
}

static inline uint64_t
count_nznum_in_cols(const uint32_t* restrict nznum, MDMacColIterator* restrict it) {
    uint64_t sum = 0;
    for(mdmac_col_iter_begin(it); !mdmac_col_iter_end(it); mdmac_col_iter_next(it)) {
        uint64_t idx = mdmac_col_iter_idx(it);
        sum += nznum[idx];
    }
    return sum;
}

int32_t
main(int32_t argc, char* argv[]) {
    Options* opt = opt_create();
    if(!opt) {
        printf_err_ts("[!] Failed to allocate memory to parse options\n");
        return 1;
    }
    int32_t parse_rv;
    if(0 != (parse_rv = opt_parse(opt, argc, argv)) ) {
        printf_err_ts("[!] Failed to parse options: %s\n",
                      opt_err_code_to_str(parse_rv));
        opt_free(opt);
        return 1;
    }
    if(opt_help(opt)) {
        opt_print_usage(argv[0]);
        opt_free(opt);
        return 0;
    }
    const uint32_t tnum = opt_tpsize(opt); // number of threads to use
    printf_ts("number of threads to use: %u\n", tnum);

    if(opt_new_randseed(opt)) {
        printf_ts("random seed: %u\n", opt_seed(opt));
        srand(opt_seed(opt));
    } else {
        printf_ts("random seed: NULL\n");
        srand(time(NULL));
    }
    printf_ts("max output from system random generator: %d\n", RAND_MAX);

    LoaderGFMfromFileRet rt;
    if(SUCCESS != loader_gfm_from_file(&rt, opt_mr_file(opt))) {
        printf_err_ts("[!] Failed to load input file %s\n", opt_mr_file(opt));
        opt_free(opt);
        return 1;
    }

    const uint32_t k = rt.k;
    const uint32_t r = rt.r;
    const uint32_t c = opt_c(opt);

    int32_t rval = 0; // return value
    // data storage
    Threadpool* tpool = NULL; GFM* ks = NULL; MinRank* mr = NULL;
    const MDeg* mdeg  = NULL; MDMac* mdmac = NULL; MDMacColIterator* it = NULL;
    CMSMGeneric* cmsm = NULL, *cmsm_kept = NULL;
    uint64_t* vmap = NULL; uint32_t* nznum = NULL;
    BLKGF16Arg* blkarg = NULL; RMGF16* nullvec_candidates = NULL;
    RMGF16* p = NULL, *gf_buf = NULL; Hmap* dedup_hmap = NULL;
    void* reduced_mdmac = NULL, *sol = NULL;

    if( !(mr = minrank_create(rt.nrow, rt.ncol, k, r, rt.m0, rt.ms)) ) {
        printf_err_ts("[!] Fail to create MinRank instance\n");
        gfm_free(rt.m0);
        gfm_arr_free(rt.ms, k);
        rval = 1;
        goto main_cleanup;
    }
    printf_ts("[+] Input MinRank instance: %s\n"
              "\t\tdimension of matrices: %u x %u\n"
              "\t\tnumber of matrices: %u\n"
              "\t\ttarget rank: %u\n", opt_mr_file(opt), minrank_nrow(mr),
              minrank_ncol(mr), minrank_nmat(mr), minrank_rank(mr));

    if(opt_ks_rand(opt)) {
        printf_ts("[+] Generating random KS matrix:\n");
        ks = ks_rand(minrank_nmat(mr), minrank_rank(mr), c, minrank_ncol(mr));
    } else {
        printf_ts("[+] Computing KS matrix:\n");
        ks = minrank_ks(mr, c);
    }

    if(!ks) {
        printf_err_ts("[!] Fail to create KS matrix\n");
        rval = 1;
        goto main_cleanup;
    }
    printf("\t\tnumber of rows in left multiplier (parameter c): %u\n"
           "\t\tdimension (logical): %u x %u\n"
           "\t\tdimension (actual): %lu x %lu\n",
           opt_c(opt), opt_c(opt), minrank_ncol(mr), gfm_nrow(ks), gfm_ncol(ks));

    printf_ts("[+] Selected multi-degree(s):\n");
    uint32_t degs_num = opt_mdeg_num(opt);
    for(uint32_t j = 0; j < degs_num; ++j) {
        printf("\t\t( ");
        mdeg = opt_mdeg(opt, j);
        for(uint32_t i = 0; i < c; ++i) {
            printf("%u, ", mdeg_deg(mdeg, i));
        }
        printf("%u ), total: %u\n", mdeg_deg(mdeg, c), mdeg_total_deg(mdeg));
    }

    uint64_t max_tnum = gfm_find_max_tnum_per_eq(ks);
    size_t mdmac_memsize = mdmac_calc_memsize(k, r, mdeg, minrank_ncol(mr),
                                              max_tnum);

    printf_ts("[+] Computing multi-degree Macaulay matrix\n"
              "\t\tmax number of supported rows: 2^64-1\n"
              "\t\tmax number of supported columns: 2^%u-1\n"
              "\t\tmax number of non-zero entries in a row of the base system: %lu\n"
              "\t\tstorage requirement: %.2fMB\n",
              gfa_size_of_idx(), max_tnum, mdmac_memsize / MBFLOAT);

    if(opt_dry(opt))
        goto main_cleanup;

    if( !(tpool = thpool_create(tnum)) ) {
        printf_err_ts("[!] Fail to create thread pool\n");
        rval = 1;
        goto main_cleanup;
    }

    if(degs_num == 1)
        mdmac = mdmac_create_from_ks(ks, mr, mdeg);
    else
        mdmac = mdmac_combi_create_from_ks(ks, mr, opt_degs(opt), degs_num);

    if(!mdmac) {
        printf_err_ts("[!] Fail to create multi-degree Macaulay\n");
        rval = 1;
        goto main_cleanup;
    }
    it = mdmac_col_iter_create_from_mdmac(mdmac, mdeg_is_nonlinear);

    printf("\t\tdimension: %lu x %lu\n", mdmac_nrow(mdmac), mdmac_ncol(mdmac));
    uint32_t target_nv_num = ks_total_var_num(k, r, c) + 1;

    uint64_t cidxs_sz = mdmac_num_nlcol(mdmac);
    uint64_t remaining_ncol = mdmac_ncol(mdmac) - cidxs_sz;
    if(remaining_ncol > 512) {
        printf_err_ts("[!] Resultant matrix with more than %lu columns is not "
                      "supported\n", remaining_ncol);
        goto main_cleanup;
    }
    init_sc_funcs(remaining_ncol);

    uint32_t vnum = ks_total_var_num(k, r, c);
    assert((vnum + 1) == remaining_ncol);
    vmap = malloc(sizeof(uint64_t) * remaining_ncol);
    if(!vmap) {
        printf_err_ts("[!] Fail to create containers for variable map\n");
        rval = 1;
        goto main_cleanup;
    }
    vmap[0] = 0; // constant column
    for(uint32_t i = 0; i < vnum; ++i) // variables (both linear and kernel)
        vmap[1 + i] = mdmac_vidx_to_midx(mdmac, i);

    if( !( nznum = malloc(sizeof(uint32_t) * mdmac_ncol(mdmac)))  ) {
        printf_err_ts("[!] Fail to create containers for column indices\n");
        rval = 1;
        goto main_cleanup;
    }

    const int32_t mac_seed = rand();
    uint64_t cmsm_rnum = opt_mac_nrow(opt);
    if(cmsm_rnum == 0 || cmsm_rnum > mdmac_nrow(mdmac))
        cmsm_rnum = mdmac_nrow(mdmac); // use all rows
    const uint64_t mac_nznum = mdmac_nznum(nznum, mdmac, cmsm_rnum, mac_seed);
    const uint64_t nznum_to_remove = count_nznum_in_cols(nznum, it);
    mdmac_col_iter_set_filter(it, mdeg_is_linear);
    const uint64_t nznum_to_keep = count_nznum_in_cols(nznum, it);
    assert(mac_nznum == (nznum_to_remove + nznum_to_keep));
    double cmsm_total_mem = cmsm_generic_calc_mem_size(cmsm_rnum, cidxs_sz,
                                                       nznum_to_remove);
    cmsm_total_mem += cmsm_generic_calc_mem_size(cmsm_rnum, remaining_ncol,
                                                 nznum_to_keep);
    cmsm_total_mem /= MBFLOAT;
    printf("\t\trows to keep: %lu\n"
           "\t\tcolumns to keep: %lu\n"
           "\t\tcolumns to eliminate: %lu\n"
           "\t\tnumber of non-zero entries: %lu (%.2f%%)\n"
           "\t\tsize of column-majored condensed multi-degree Macaulay: %.2fMB\n",
           cmsm_rnum, remaining_ncol, cidxs_sz, mac_nznum,
           100.0 * mac_nznum / cmsm_rnum / cidxs_sz, cmsm_total_mem);

    if( !(p = rm_gf16_create(cidxs_sz)) ) {
        printf_err_ts("[!] Fail to create RMGF16 matrix for Block Lanczos\n");
        rval = 1;
        goto main_cleanup;
    }

    printf_ts("[+] Condensing multi-degree Macaulay along columns\n");
    mdmac_col_iter_set_filter(it, mdeg_is_nonlinear);
    if( !(cmsm = cmsm_generic_from_mdmac(mdmac, cmsm_rnum, mac_seed, it, nznum,
                                         nznum_to_remove)) ) {
        printf_err_ts("[!] Fail to create column-majored multi-degree Macaulay\n");
        rval = 1;
        goto main_cleanup;
    }
    // the filter for the iterator is still mdeg_is_linear
    mdmac_col_iter_set_filter(it, mdeg_is_linear);
    if( !(cmsm_kept = cmsm_generic_from_mdmac(mdmac, cmsm_rnum, mac_seed, it,
                                              nznum, nznum_to_keep)) ) {
        printf_err_ts("[!] Fail to create column-majored multi-degree Macaulay\n");
        rval = 1;
        goto main_cleanup;
    }
    printf_ts("[+] Done\n");
    printf("\t\tmax number of entries to eliminate in a column: %lu\n"
           "\t\tavg number of entries to eliminate in a column: %lu\n",
           cmsm_generic_max_tnum(cmsm), cmsm_generic_avg_tnum(cmsm));

    if( !(blkarg = blkgf16_arg_create(cmsm_rnum, cidxs_sz, tnum)) ) {
        printf_err_ts("[!] Fail to create containers for Block Lanczos\n");
        rval = 1;
        goto main_cleanup;
    }

    // launch block Lanczos until enough nullvectors are found
    // NOTE: raise the capacity of dedup_hamp to avoid hash collision
    if( !(dedup_hmap = hmap_create(target_nv_num * 10)) ) {
        printf_err_ts("[!] Fail to create Hmap for Block Lanczos\n");
        rval = 1;
        goto main_cleanup;
    }
    if( !(reduced_mdmac = g_sc_create()) || !(sol = g_sc_create()) ) {
        printf_err_ts("[!] Fail to create containers for the resultant matrix\n");
        rval = 1;
        goto main_cleanup;
    }
    g_sc_zero(reduced_mdmac);
    g_sc_zero(sol);

    if( !(gf_buf = rm_gf16_create(remaining_ncol)) ) {
        printf_err_ts("[!] Fail to create buffer to GF vector\n");
        rval = 1;
        goto main_cleanup;
    }

    printf_ts("[+] Try to extract %u nullvectors\n", target_nv_num);
    // TODO: what is the expected rank?
    uint64_t expected_rank = (cidxs_sz > cmsm_rnum) ? cmsm_rnum : cidxs_sz;
    printf("\t\texpected rank of submatrix to eliminate: %lu\n"
           "\t\tblock size: %d\n"
           "\t\texpected number of iterations: %zu\n"
           "\t\tsize of %lu x %d matrix: %.2fMB\n"
           "\t\tsize of %lu x %d matrix: %.2fMB\n"
           "\t\tsize of %d x %d matrix: %.2fKB\n",
           expected_rank,
           BLK_LANCZOS_BLOCK_SIZE,
           blkgf16_iter_num(BLK_LANCZOS_BLOCK_SIZE, expected_rank),
           cmsm_rnum, BLK_LANCZOS_BLOCK_SIZE,
           rm_gf16_memsize(cmsm_rnum) / MBFLOAT,
           mdmac_ncol(mdmac), BLK_LANCZOS_BLOCK_SIZE,
           rm_gf16_memsize(mdmac_ncol(mdmac)) / MBFLOAT,
           BLK_LANCZOS_BLOCK_SIZE, BLK_LANCZOS_BLOCK_SIZE,
           rcm_gf16_memsize() / KBFLOAT);

    mdmac_free(mdmac); // release resources as soon as possible
    mdmac = NULL;
    free(nznum);
    nznum = NULL;

    // the filter for the iterator is still mdeg_is_linear
#ifdef BLK_LANCZOS_COLLECT_STATS
    uint64_t hmap_full_count = 0, hmap_dup_count = 0,
             zero_nv_count = 0, invalid_nv_count = 0;
#endif
    uint64_t iter = 0;
    while(iter++ < LANCZOS_MAX_ITER && hmap_cur_size(dedup_hmap) < target_nv_num) {
        // TODO: record iter_count
        uint32_t iter_count = blk_lczs_gf16(blkarg, cmsm, tpool);
        nullvec_candidates = blkgf16_arg_v(blkarg);
#ifdef BLK_LANCZOS_COLLECT_STATS
        DiagMGF16 nv_pos, zv;
        verify_nullvec(&nv_pos, p, cmsm, nullvec_candidates);
        rm_gf16_zc_pos(nullvec_candidates, &zv); // find zero vectors
        zero_nv_count += diagm_gf16_nzc(&zv);
        invalid_nv_count += diagm_gf16_zc(&nv_pos);
        uint32_t nvc = proc_nullvec(dedup_hmap, reduced_mdmac, sol, gf_buf,
                                    nullvec_candidates, cmsm_kept,
                                    tnum, blkgf16_arg_pargs(blkarg), tpool,
                                    vmap, it, remaining_ncol,
                                    &hmap_full_count, &hmap_dup_count);
#else
        uint32_t nvc = proc_nullvec(dedup_hmap, reduced_mdmac, sol, gf_buf,
                                    nullvec_candidates, cmsm_kept,
                                    tnum, blkgf16_arg_pargs(blkarg), tpool,
                                    vmap, it, remaining_ncol);
#endif
        printf_ts("[+] %zu-th batch: %u iterations, %u nullvectors\n", iter, iter_count, nvc);
    }

    printf_ts("[+] Block Lanczos finished in %zu batches\n"
              "\t\tnullvectors extracted: %zu\n", iter-1, hmap_cur_size(dedup_hmap));
#ifdef BLK_LANCZOS_COLLECT_STATS
    printf("\t\tnullvectors dropped due to capacity: %zu\n"
           "\t\tnullvectors dropped due to duplication: %zu\n"
           "\t\tnullvectors that are full zero: %zu\n"
           "\t\tnullvectors not in the left kernel: %zu\n",
           hmap_full_count, hmap_dup_count, zero_nv_count,
           invalid_nv_count);
#endif

    if(hmap_cur_size(dedup_hmap) >= target_nv_num) {
        printf_ts("[+] Solving the extracted linear system\n");
        if(opt_ks_rand(opt)) {
            printf_ts("[!] This solution is for the randomly sampled KS matrix!\n");
            printf("\t\tNot the original MinRank instance!\n");
        }
        // reduced mdmac from nullvectors is dense and small. Just run Gaussian
        // elimination to extract the linear variables
        sc_di_t di; g_sc_gj(reduced_mdmac, sol, &di);
        if(g_sc_popcnt(&di) < (target_nv_num-1))
            printf_ts("[!] Failed, only %lu nullvectors are independent\n",
                      g_sc_popcnt(&di));
        else
            print_sol(sol, &di, k, r, c);
    }

main_cleanup:
    printf_ts("[+] Releasing resources\n");
    minrank_free(mr); // owns rt.m0 and rt.ms
    gfm_free(ks);
    mdmac_col_iter_free(it);
    mdmac_free(mdmac);
    free(nznum);
    free(vmap);
    cmsm_generic_free(cmsm);
    cmsm_generic_free(cmsm_kept);
    blkgf16_arg_free(blkarg);
    rm_gf16_free(p);
    hmap_free(dedup_hmap);
    if(g_sc_free) {
        g_sc_free(reduced_mdmac);
        g_sc_free(sol);
    }
    rm_gf16_free(gf_buf);
    thpool_destroy(tpool, true);
    opt_free(opt);
    return rval;
}
