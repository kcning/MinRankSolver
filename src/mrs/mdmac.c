#include "mdmac.h"
#include "gfa.h"
#include "gfm.h"
#include "ks.h"
#include "mdeg.h"
#include "minrank.h"
#include "mono.h"
#include "util.h"
#include "bitmap.h"

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

/* ========================================================================
 * struct MDMac definition
 * ======================================================================== */

struct MDMac {
    uint32_t k; // number of linear variables in the original KS matrix
    uint32_t r; // number of kernel variables per row in the original KS matrix
                // also the target rank
    // NOTE: c is also stored in mdeg, but without field c, struct MDeg needs to
    // be padded by 4 bytes anyway. Might as well store it directly to avoid
    // latency caused by indirect access.
    uint32_t c; // number of rows in the original KS matrix
    uint32_t m; // number of columns of matrices in the original MinRank instance
    uint64_t nrow; // number of rows in the multi-degree Macaulay matrix
    uint64_t ncol; // number of columns (monomials) in the Macaulay matrix

    uint32_t degs_sz; // number of multi-degress in degs
    MDeg** restrict degs; // combined multi-degree based on which the Macaulay
                          // is defined. Take precedence over mdeg
    MDeg* restrict mdeg;// multi-degree based on which the Macaulay is defined
                        // If defined over combined multi-degrees, mdeg stores
                        // the minimal multi-degree that defines the set of
                        // monomials >= the union of monomials defined by
                        // individual multi-degrees
    uint64_t* restrict mono_num_per_deg;// i-th: number of deg-i monomials
    GFA* restrict rows; // NOTE: each eq in the multi-degree Macaulay matrix is
                        // represented by m rows. Thus the number of rows is
                        // not the same as the number of equations
    gfa_idx_t memblk[]; // memory block for the sparse rows
};

/* ========================================================================
 * function implementations
 * ======================================================================== */

/* usage: compute the number of columns in a MDMac
 * params:
 *      1) k: number of linear variables
 *      2) r: number of kernel variables in each group
 *      3) mdeg: ptr to struct MDeg, target multi-degree
 * return: void */
uint64_t
mdmac_calc_ncol(uint32_t k, uint32_t r, const MDeg* d) {
    return ks_mdmac_total_mono_num(k, r, d);
}

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
                   uint32_t max_tnum) {
    uint64_t ncol = mdmac_calc_ncol(k, r, d);
    mdeg_create_static_buf(tmp_buf, mdeg_c(d));
    MDeg* tmp_d = mdeg_create_from_arr(mdeg_c(d), tmp_buf);
    mdeg_copy(tmp_d, d);
    uint64_t nrow = m * mdmac_multiplier_num(k, r, tmp_d);
    size_t sz = sizeof(MDMac) + gfa_size_of_element() * nrow * max_tnum;
    sz += sizeof(uint32_t) * ncol + sizeof(uint64_t) * (mdeg_total_deg(d) + 1);
    sz += gfa_memsize() * nrow;
    return sz;
}

/* usage: Given a struct MDMac, return its number of linear variables
 * params:
 *      1) m: ptr to struct MDMac
 * return: number of linear variables */
uint32_t
mdmac_k(const MDMac* m) {
    return m->k;
}

/* usage: Given a struct MDMac, return its target rank
 * params:
 *      1) m: ptr to struct MDMac
 * return: target rank */
uint32_t
mdmac_r(const MDMac* m) {
    return m->r;
}

/* usage: Given a struct MDMac, return the number of rows its the original KS
 *      matrix
 * params:
 *      1) m: ptr to struct MDMac
 * return: number of rows */
uint32_t
mdmac_c(const MDMac* m) {
    return m->c;
}

/* usage: Given a struct MDMac, return the number of columns of matrices in
 *      the original MinRank instance
 * params:
 *      1) m: ptr to struct MDMac
 * return: number of columns */
uint32_t
mdmac_m(const MDMac* m) {
    return m->m;
}

/* usage: Given a struct MDMac, return its multi-degree
 * params:
 *      1) m: ptr to struct MDMac
 * return: target multi-degree stored as a uint32_t array of length
 *      mdmac_c(m) + 1, where the 1st integer is the degree of the
 *      linear vars, and the 2nd integer the degree for the first
 *      group of kernel vars, and so on */
const MDeg*
mdmac_mdeg(const MDMac* m) {
    return m->mdeg;
}

/* usage: Given a struct MDMac, return its degree
 * params:
 *      1) m: ptr to struct MDMac
 * return: its degree */
uint32_t
mdmac_deg(const MDMac* m) {
    return mdeg_total_deg(m->mdeg);
}

/* usage: Given a struct MDMac, return its number of rows (equations)
 * params:
 *      1) m: ptr to struct MDMac
 * return: number of rows */
uint64_t
mdmac_nrow(const MDMac* m) {
    return m->nrow;
}

/* usage: Given a struct MDMac, return its number of columns (monomials)
 * params:
 *      1) m: ptr to struct MDMac
 * return: number of columns */
uint64_t
mdmac_ncol(const MDMac* m) {
    return m->ncol;
}

/* usage: Given the number of linear variables, target rank, the number
 *      of rows in the left multiplier, and the multi-degree of the Macaulay
 *      matrix, compute the total number of possible multipliers (monomials)
 * params:
 *      1) k: number of linear variables
 *      2) r: target rank
 *      3) d: ptr to struct MDeg
 * return: total number of multipliers */
uint64_t
mdmac_multiplier_num(uint32_t k, uint32_t r, MDeg* d) {
    assert(mdeg_lv_deg(d) >= 1);
    mdeg_lv_deg_dec(d);
    uint64_t n = 0;
    for(uint32_t i = 0; i < mdeg_c(d); ++i) {
        //if(mdeg_kv_deg(d, i) == 0)
        //    continue;

        mdeg_kv_deg_dec(d, i);
        n += ks_mdmac_total_mono_num(k, r, d);
        mdeg_kv_deg_inc(d, i);
    }
    mdeg_lv_deg_inc(d); // restore the multi-degree
    return n;
}

/* usage: Given the max multi-degree, and the current multi-degree, raise the
 *      multi-degree to the next one incrementally
 * params:
 *      1) mdeg: the current multi-degree, which will be updated to the next
 *          one after the function returns
 *      2) max_mdeg: the max multi-degree to consider
 * return: true if the current multi-degree is not the max multi-degree. false
 *      otherwise */
bool
mdmac_mdeg_next(MDeg* restrict mdeg, const MDeg* restrict max_mdeg) {
    return mdeg_next(mdeg, max_mdeg);
}

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
                 uint32_t k, uint32_t r) {
    mono_mdeg_first(mono, mdeg, k, r);
}

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
                   uint32_t r) {
    return mono_mdeg_iterate(mono, d, k, r);
}

/* usage: Given the MinRank instance, and the target multi-degree of a Macaulay
 *      matrix, compute the total number of equations in the Macaulay matrix
 * params:
 *      1) mr: ptr to struct MinRank
 *      2) mdeg: ptr to struct MDeg. Target multi-degree
 * return: total number of eqs in the multi-degree Macaulay matrix */
uint64_t
mdmac_eq_num(const MinRank* restrict mr, const MDeg* restrict mdeg) {
    uint32_t c = mdeg_c(mdeg);
    mdeg_create_static_buf(tmp_buf, c);
    MDeg* d = mdeg_create_from_arr(c, tmp_buf);
    mdeg_copy(d, mdeg);
    uint64_t multiplier_num = mdmac_multiplier_num(minrank_nmat(mr), minrank_rank(mr), d);
    return multiplier_num * minrank_ncol(mr);
}

/* usage: Given a struct MDMac and the row index i, return the i-th row
 *      which represents an eqaution
 * params:
 *      1) m: ptr to struct MDMac
 *      2) i: index of the row
 * return: ptr of struct GFA that stores the i-th row */
const GFA*
mdmac_row(const MDMac* m, uint64_t i) {
    return gfa_arr_at(m->rows, i);
}

/* usage: Given a struct MDMac, row index i, and column index j, return
 *      the specified entry
 * params:
 *      1) m: ptr to struct MDMac
 *      2) i: the row index
 *      3) j: the column index
 * return: entry as a gf_t */
gf_t
mdmac_at(const MDMac* m, uint64_t i, uint64_t j) {
    assert(i < mdmac_nrow(m));
    assert(j < mdmac_ncol(m));
    const GFA* row = mdmac_row(m, i);

    for(uint64_t ci = 0; ci < gfa_size(row); ++ci) { // for every non-zero entry in the row
        gfa_idx_t idx; gf_t v = gfa_at(row, ci, &idx);
        assert(idx != KS_MDMAC_MIDX_INVALID);
        if(idx == j)
            return v;
        else if (idx > j)
            break;
    }

    return 0;
}

/* subroutine of mdmac_cmp_mmap_(base|mono): check that the indices is
 * strictly ascending */
static bool __attribute__((unused))
mdmac_mmap_check_ascend(const gfa_idx_t* mmap, uint64_t sz) {
    for(uint64_t i = 0; i < (sz-1); ++i) {
        if(mmap[i] == KS_MDMAC_MIDX_INVALID)
            continue;

        uint64_t j = i+1;
        while(j < sz && mmap[j] == KS_MDMAC_MIDX_INVALID)
            ++j;

        if(j != sz && mmap[i] >= mmap[j])
            return false;
    }
    return true;
}

/* subroutine of mdmac_create_from_ks: map indices of monomials in the base ks system into
 * indices of monomials in the multi-degree macaulay derived from the ks system */
static inline void
mdmac_cmp_mmap_base(gfa_idx_t* restrict mmap, uint32_t k, uint32_t r,
                    const MDeg* restrict d) {
    const uint32_t c = mdeg_c(d);
    const uint32_t vnum = ks_total_var_num(k, r, c);
    mono_create_static_buf(cidxs, 2); // at most 2 vars in a monomial
    uint64_t dst_idx = 0;

    Mono* mul = mono_create_from_arr(0, cidxs);
    mmap[dst_idx++] = ks_mdmac_midx(k, r, d, mul);

    // kernel vars and linear vars
    for(uint32_t i = vnum; i > 0; --i) {
        mul = mono_create_from_arr(1, cidxs);
        mono_set_var(mul, 0, i-1, false); // no need to sort
        mmap[dst_idx++] = ks_mdmac_midx(k, r, d, mul);
    }

    // degree-2 monomials. Note that the base KS system only have vi * xj,
    // where vi is a kernel var and xj is a linear var
    for(uint32_t i = vnum-1; i >= k; --i) {
        for(uint32_t j = k; j > 0; --j) {
            mul = mono_create_from_arr(2, cidxs);
            mono_set_var(mul, 0, j-1, false); // no need to sort
            mono_set_var(mul, 1, i, false);
            mmap[dst_idx++] = ks_mdmac_midx(k, r, d, mul);
        }
    }
    assert(dst_idx == ks_base_total_mono_num(k, r, c));
    assert(mdmac_mmap_check_ascend(mmap, dst_idx) == true);
}

/* subroutine of mdmac_create_from_ks: map indices of monomials in the base ks system
 * multiplied by a monomial into indices of monomials in the multi-degree Macaulay
 * derived from the ks system. `mul` is the temporary storage for a monomial and
 * should have size mdeg_total_deg(mdeg) + 2. */
static inline void
mdmac_cmp_mmap_mono(gfa_idx_t* restrict mmap, Mono* restrict mono,
                    const Mono* restrict mul, uint32_t k, uint32_t r,
                    const MDeg* restrict d) {
    const uint32_t c = mdeg_c(d);
    // The multi-degree (2, 1, 0) is stored at an array [2, 1, 0] of size
    // c + 1 (c = 2 here), and the multiplier mul is of the format [xi, xj, xk],
    // where xi and xj are linear vars while xk is in the 1st group of kernel var.
    // Note that mul must be sorted.

    // NOTE: each monomial in the base KS system is multiplied by the given monomial,
    // which is not the constant 1, thus no monomial will be mapped to the constant
    // term (at index 0).

    const uint32_t vnum = ks_total_var_num(k, r, c);
    uint64_t dst_idx = 0;

    // constant term
    mmap[dst_idx++] = ks_mdmac_midx(k, r, d, mul);

    // kernel vars and linear vars
    for(uint32_t i = vnum; i > 0; --i) {
        mono_copy_partial_from(mono, mul);
        mono_set_deg(mono, mono_deg(mul) + 1);
        // TODO: optimize this to avoid qsort
        mono_set_var(mono, mono_deg(mul), i-1, true);
        mmap[dst_idx++] = ks_mdmac_midx(k, r, d, mono);
    }

    // deg-2 monomials in the base KS system
    for(uint32_t i = vnum-1; i >= k; --i) {
        for(uint32_t j = k; j > 0; --j) {
            mono_copy_partial_from(mono, mul);
            mono_set_deg(mono, mono_deg(mul) + 2);
            mono_set_var(mono, mono_deg(mul), i, false);
            // TODO: optimize this to avoid qsort
            mono_set_var(mono, mono_deg(mul)+1, j-1, true);
            mmap[dst_idx++] = ks_mdmac_midx(k, r, d, mono);
        }
    }
    assert(dst_idx == ks_base_total_mono_num(k, r, c));
    assert(mdmac_mmap_check_ascend(mmap, dst_idx) == true);
}

/* subroutine of mdmac_create_from_ks: given a monomial index map from the base KS system
 * into the multi-degree Macaulay derived from the KS system, fill monomials in the chosen
 * eq of the base KS system into the multi-degree Macaulay */
static inline void
mdmac_fill_in_eqs(MDMac* restrict m, uint64_t row_offset, const GFM* restrict ks,
                  uint64_t ri, const gfa_idx_t* restrict mmap) {
    for(uint64_t i = 0; i < mdmac_m(m); ++i) { // for each row of the chosen eq in the base KS system
        const gf_t* src_eq = gfm_row_addr(ks, ri + i);
        GFA* dst_eq = (GFA*) mdmac_row(m, row_offset + i);

        uint64_t sz = 0;
        for(uint64_t j = 0; j < gfm_ncol(ks); ++j) {
            if(src_eq[j] == 0)
                continue;

            gfa_set_at(dst_eq, sz++, mmap[j], src_eq[j]);
        }
        gfa_set_size(dst_eq, sz);
    }
}

/* subroutine of mdmac_create_from_ks: release temporary buffers */
static inline void
mdmac_create_cleanup(MDeg* restrict d, gfa_idx_t* restrict mmap,
                     Mono* restrict mono, Mono* restrict mul) {
    mdeg_free(d);
    free(mmap);
    mono_free(mono);
    mono_free(mul);
}

/* subroutine of mdmac_create_from_ks: check if the mdeg is valid */
static inline bool
mdmac_check_mdeg(const MDeg* d) {
    // each group of vars, including the linear vars should have at least deg 1
    for(uint32_t i = 0; i <= mdeg_c(d); ++i) {
        if(mdeg_deg(d, i) < 1)
            return false;
    }
    return true;
}

/* usage: Given a base KS system constructed for a MinRank instance, and a target multi-degree,
 *      compute its multi-degree Macaulay matrix
 * params:
 *      1) ks: ptr to struct GDM that stores the base KS system
 *      2) mr: ptr to struct MinRank that defines the MinRank problem
 *      3) d: ptr to struct MDeg that specifies the target multi-degree
 * return: ptr to struct MDMac */
MDMac*
mdmac_create_from_ks(const GFM* restrict ks, const MinRank* restrict mr,
                     const MDeg* restrict d) {
    const uint32_t c = mdeg_c(d);
    assert(ks_base_total_mono_num(minrank_nmat(mr), minrank_rank(mr), c) == gfm_ncol(ks));

    if(!mdmac_check_mdeg(d))
        return NULL;

    const uint64_t mac_col_num = mdmac_calc_ncol(minrank_nmat(mr),
                                                 minrank_rank(mr), d);
    if(mac_col_num > GFA_IDX_MAX)
        return NULL;

    MDMac* m = NULL;
    Mono* mono = NULL;
    Mono* mul = NULL;
    MDeg* cur_mdeg = NULL;
    gfa_idx_t* mmap = NULL;

    const uint64_t max_tnum = gfm_find_max_tnum_per_eq(ks);
    const uint64_t nrow = mdmac_eq_num(mr, d);

    if(unlikely(0 == nrow))
        return NULL;

    const size_t memblk_sz = gfa_size_of_element() * nrow * max_tnum;
    m = malloc(sizeof(MDMac) + memblk_sz + sizeof(uint32_t) * mac_col_num);
    if(!m)
        return NULL;
    memset(m->memblk, 0x0, memblk_sz);
    // right after memblk used by GFA
    m->mdeg = mdeg_dup(d);
    if(!m->mdeg) {
        free(m);
        return NULL;
    }

    m->mono_num_per_deg = malloc(sizeof(uint64_t) * (mdeg_total_deg(d) + 1));
    if(!m->mono_num_per_deg) {
        mdeg_free(m->mdeg);
        free(m);
        return NULL;
    }
    ks_mdmac_calc_mono_nums(m->mono_num_per_deg, minrank_nmat(mr),
                            minrank_rank(mr), m->mdeg);

    m->rows = gfa_arr_create(max_tnum, nrow, m->memblk);
    if(!m->rows) {
        mdeg_free(m->mdeg);
        free(m->mono_num_per_deg);
        free(m);
        return NULL;
    }

    cur_mdeg = mdeg_create_zero(c);
    mmap = malloc(sizeof(gfa_idx_t) *
            ks_base_total_mono_num(minrank_nmat(mr), minrank_rank(mr), c));
    const uint32_t mono_size = mdeg_total_deg(m->mdeg);
    mul = mono_create_container(mono_size); // the multiplier monomial
    mono = mono_create_container(mono_size + 2); // each eq is bilinear, so a
                                                 // monomial from multiplication
                                                 // has at most 2 more vars
    if(!cur_mdeg || !mmap || !mul || !mono) {
        mdmac_create_cleanup(cur_mdeg, mmap, mono, mul);
        mdmac_free(m);
        return NULL;
    }

    m->k = minrank_nmat(mr); m->r = minrank_rank(mr); m->c = c;
    m->m = minrank_ncol(mr); m->nrow = nrow;
    m->ncol = mac_col_num; m->degs = NULL; m->degs_sz = 0;

    // the first m rows in KS are from 1 row in the left multiplier,
    // and thus share the same multi-degree multiplier. We call this
    // a 'group' and compute the multi-degree Macaulay by multiplying
    // this group with all monomials <= a multi-degree to generate
    // rows in the multi-degree Macaulay before moving onto the next
    // group. There are c groups in total.
    //
    // The multi-degree of the multiplier is computed from the target
    // multi-degree and the degree of the different groups of kernel
    // variables in this group of rows. For example, if the target
    // multi-degree is (2, 2, 1), then for the 1st group of m rows
    // in the base KS matrix, they have 1 linear variable, and 1 kernel
    // variable from the 1st row of the left matrix, thus the multiplier should
    // have degree <= (2-1, 2-1, 1) = (1, 1, 1). For the 2nd group of m
    // rows in the base KS matrix, they have 1 linear variable, and 1
    // kernel variable from the 2nd row of the left matrix, thus the
    // multiplier should have degree <= (2-1, 2, 1-1) = (1, 2, 0)

    uint64_t dst_row_offset = 0;
    uint64_t src_row_offset = 0;
    mdeg_lv_deg_dec(m->mdeg); // deg of the linear var in the multiplier
    for(uint32_t i = 0; i < c; ++i) { // for each group of m rows in the KS base system
        mdeg_kv_deg_dec(m->mdeg, i);

        mdeg_zero(cur_mdeg); // start with the constant multiplier 1
        // just fill in the base system
        mdmac_cmp_mmap_base(mmap, mdmac_k(m), mdmac_r(m), d);
        mdmac_fill_in_eqs(m, dst_row_offset, ks, src_row_offset, mmap); // only the (i-1)-th eq in the base ks system
        dst_row_offset += mdmac_m(m);

        // NOTE: some room for optimization here because some multipliers are the same for
        // different multi-degrees
        while(mdmac_mdeg_next(cur_mdeg, m->mdeg)) { // move on to the next multi-degree
            mdmac_mdeg_first(mul, cur_mdeg, mdmac_k(m), mdmac_r(m));
            mdmac_cmp_mmap_mono(mmap, mono, mul, mdmac_k(m), mdmac_r(m), d);
            mdmac_fill_in_eqs(m, dst_row_offset, ks, src_row_offset, mmap); // only the (i-1)-th eq in the base ks system
            dst_row_offset += mdmac_m(m);

            while(mdmac_mdeg_iterate(mul, cur_mdeg, mdmac_k(m), mdmac_r(m))) { // for every remaining monomial of the current multi-degree
                mdmac_cmp_mmap_mono(mmap, mono, mul, mdmac_k(m), mdmac_r(m), d);
                mdmac_fill_in_eqs(m, dst_row_offset, ks, src_row_offset, mmap); // only the (i-1)-th eq in the base ks system
                dst_row_offset += mdmac_m(m);
            }
        }

        src_row_offset += mdmac_m(m);
        mdeg_kv_deg_inc(m->mdeg, i);
    }

    mdeg_lv_deg_inc(m->mdeg);
    mdmac_create_cleanup(cur_mdeg, mmap, mono, mul);
    return m;
}

static inline void
mdmac_free_degs(MDeg** degs, uint32_t sz) {
    if(!degs || sz == 0)
        return;

    for(uint32_t i = 0; i < sz; ++i)
        mdeg_free(degs[i]);
}

/* usage: Given a struct MDMac, release it
 * params:
 *      1) m: ptr to struct MDMac
 * return: void */
void
mdmac_free(MDMac* m) {
    if(!m)
        return;
    gfa_arr_free(m->rows);
    mdeg_free(m->mdeg);
    mdmac_free_degs(m->degs, m->degs_sz);
    free(m->degs);
    free(m->mono_num_per_deg);
    free(m);
}

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
                       mdmac_iter_rows_cb_t* cb, void* arg) {
    if(nrow > full_nrow)
        return -2;

    Bitmap* b = bitmap_create(full_nrow);
    if(!b)
        return -1;

    bitmap_zero(b);
    int32_t old_seed = rand();
    srand(seed);
    uint64_t sample_num = 0; // Floyd's random sampling
    for(uint64_t in = full_nrow - nrow; in < full_nrow && sample_num < nrow; ++in) {
        uint64_t ridx = uint64_rand() % (in + 1);
        if(bitmap_at(b, ridx))
            ridx = in;

        assert(!bitmap_at(b, ridx));
        bitmap_set_true_at(b, ridx);

        cb(sample_num++, ridx, arg);
    }

    bitmap_free(b);
    assert(sample_num == nrow);
    srand(old_seed);
    return 0;
}

struct MDMacNZnumArg {
    uint32_t* restrict out;
    const MDMac* restrict m;
    uint64_t sum;
};

static inline void
mdmac_nznum_inc_col_counter(uint64_t i, uint64_t ridx, void* __arg) {
    (void) i;
    struct MDMacNZnumArg* arg = __arg;

    const GFA* row = mdmac_row(arg->m, ridx);
    arg->sum += gfa_size(row);
    for(uint64_t j = 0; j < gfa_size(row); ++j) {
        gfa_idx_t cidx; gfa_at(row, j, &cidx);
        assert(arg->out[cidx] < UINT32_MAX);
        ++(arg->out[cidx]);
    }
}

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
            int32_t seed) {
    const uint64_t full_nrow = mdmac_nrow(m);
    memset(out, 0x0, sizeof(uint32_t) * mdmac_ncol(m));
    struct MDMacNZnumArg arg = { .out = out, .m = m, .sum = 0 };
    int64_t rv = mdmac_iter_random_rows(full_nrow, nrow, seed,
                                        mdmac_nznum_inc_col_counter, &arg);
    if(rv)
        return rv;

    return arg.sum;
}

/* usage: Given a struct MDMac, return the number of columns that correspond to
 *      linear monomials and the constant term.
 * params:
 *      1) m: ptr to struct MDMac
 * return: the number of linear monomials and the constant term. Note that this
 *      is the max number and an MDMac instance might have less in each row
 *      because some of those monomials might be zero */
uint64_t
mdmac_num_linear_col(const MDMac* m) {
    return 1 + ks_total_var_num(mdmac_k(m), mdmac_r(m), mdmac_c(m));
}

/* usage: Given a struct MDMac, return the number of columns that correspond to
 *      non-linear monomials
 * params:
 *      1) m: ptr to struct MDMac
 * return: the number of non-linear monomials. Note that this is the max number
 *      and an MDMac instance might have less non-linear monomials in each
 *      row because some of those monomials might be zero */
uint64_t
mdmac_num_nlcol(const MDMac* m) {
    return mdmac_ncol(m) - mdmac_num_linear_col(m);
}

struct MDMacColIterator {
    uint64_t idx;
    uint32_t k;
    uint32_t r;
    uint32_t c;
    uint32_t degs_sz;
    Mono* mono;
    MDeg* cur_d;
    const MDeg* max_d;
    const MDeg** degs;
    mdmac_col_iter_cb_t* cb;
    bool mdeg_iter_done;
    bool mono_iter_done;
};

void
mdmac_col_iter_free(MDMacColIterator* it) {
    if(!it)
        return;

    mdeg_free(it->cur_d);
    mdeg_free((MDeg*) it->max_d);
    mono_free(it->mono);
    if(it->degs) {
        for(uint32_t i = 0; i < it->degs_sz; ++i) {
            mdeg_free((MDeg*) it->degs[i]);
        }
        free(it->degs);
    }
    free(it);
}

MDMacColIterator*
mdmac_col_iter_create(uint32_t k, uint32_t r, uint32_t c,
                      const MDeg* mdeg, mdmac_col_iter_cb_t* cb) {
    return mdmac_combi_col_iter_create(k, r, c, (const MDeg**) &mdeg, 1, mdeg, cb);
}

MDMacColIterator*
mdmac_col_iter_create_from_mdmac(const MDMac* m, mdmac_col_iter_cb_t* cb) {
    if(m->degs_sz)
        return mdmac_combi_col_iter_create(mdmac_k(m), mdmac_r(m), mdmac_c(m),
                                           (const MDeg**)m->degs, m->degs_sz,
                                           m->mdeg, cb);
    else
        return mdmac_col_iter_create(mdmac_k(m), mdmac_r(m), mdmac_c(m),
                                     m->mdeg, cb);
}

MDMacColIterator*
mdmac_combi_col_iter_create(uint32_t k, uint32_t r, uint32_t c,
                            const MDeg** m_degs, uint32_t degs_sz,
                            const MDeg* mdeg_max, mdmac_col_iter_cb_t* cb) {
    MDMacColIterator* it = calloc(1, sizeof(MDMacColIterator));
    if(!it)
        return NULL;

    it->idx = UINT64_MAX; it->k = k; it->r = r; it->c = c;
    MDeg** degs = NULL;
    it->degs_sz = degs_sz;
    degs = calloc(degs_sz, sizeof(MDeg*));
    if(!degs) {
        free(it);
        return NULL;
    }
    it->degs = (const MDeg**) degs;
    for(uint32_t i = 0; i < it->degs_sz; ++i) {
        degs[i] = mdeg_dup(m_degs[i]);
        if(!degs[i]) {
            mdmac_col_iter_free(it);
            return NULL;
        }
    }

    it->max_d = mdeg_dup(mdeg_max);
    if(!it->max_d) {
        mdmac_col_iter_free(it);
        return NULL;
    }
    it->cur_d = mdeg_dup(mdeg_max);
    if(!it->cur_d) {
        mdmac_col_iter_free(it);
        return NULL;
    }

    it->mono = mono_create_container(mdeg_total_deg(mdeg_max));
    if(!it->mono) {
        mdmac_col_iter_free(it);
        return NULL;
    }
    it->mdeg_iter_done = true;
    it->mono_iter_done = true;
    it->cb = cb;

    return it;
}

bool
mdmac_col_iter_end(const MDMacColIterator* it) {
    return it->mdeg_iter_done && it->mono_iter_done;
}

uint64_t
mdmac_col_iter_idx(const MDMacColIterator* it) {
    return it->idx;
}

void
mdmac_col_iter_next(MDMacColIterator* it) {
    it->mono_iter_done = !mono_mdeg_iterate(it->mono, it->cur_d, it->k, it->r);
    if(it->mono_iter_done) {
        bool mdeg_valid = false;
        do { // find the next valid multi-degree
            it->mdeg_iter_done = !mdeg_next(it->cur_d, it->max_d);
            mdeg_valid = mdeg_is_le_any(it->cur_d, it->degs, it->degs_sz);
            if(mdeg_valid && it->cb(it->cur_d))
                break;
        } while(!it->mdeg_iter_done);

        if(it->mdeg_iter_done)
            return;

        assert(mdeg_valid && !it->mdeg_iter_done && it->cb(it->cur_d));
        mono_mdeg_first(it->mono, it->cur_d, it->k, it->r);
        it->mono_iter_done = false;
    }

    if(it->degs_sz == 1)
        it->idx = ks_mdmac_midx(it->k, it->r, *(it->degs), it->mono);
    else
        it->idx = ks_mdmac_combi_midx(it->k, it->r, it->degs, it->degs_sz, it->mono);
}

void
mdmac_col_iter_begin(MDMacColIterator* it) {
    it->mdeg_iter_done = false;
    mdeg_zero(it->cur_d); // (0, 0, ... 0) can only lead to the constant 1
    mono_set_deg(it->mono, 0);
    it->mono_iter_done = true;
    if(it->degs_sz == 1)
        it->idx = ks_mdmac_midx(it->k, it->r, *(it->degs), it->mono);
    else
        it->idx = ks_mdmac_combi_midx(it->k, it->r, it->degs, it->degs_sz, it->mono);
    if(!it->cb(it->cur_d)) // doesn't pass the filter
        mdmac_col_iter_next(it);
}

void
mdmac_col_iter_set_filter(MDMacColIterator* it, mdmac_col_iter_cb_t* cb) {
    it->cb = cb;
}

struct MDMacColSplitArg {
    const MDMac* mac;
    uint64_t* idxs;
    uint64_t offset;
};

static inline void
mdmac_iter_mono(uint32_t mono_size, const MDeg* d, struct MDMacColSplitArg* arg) {
    mono_create_static_buf(mono_buf, mono_size);
    Mono* m = mono_create_from_arr(mono_size, mono_buf);
    mono_mdeg_first(m, d, mdmac_k(arg->mac), mdmac_r(arg->mac));
    arg->idxs[arg->offset++] = ks_mdmac_combi_midx(mdmac_k(arg->mac),
                        mdmac_r(arg->mac), (const MDeg**) arg->mac->degs,
                        arg->mac->degs_sz, m);

    while(mono_mdeg_iterate(m, d, mdmac_k(arg->mac), mdmac_r(arg->mac))) {
        arg->idxs[arg->offset++] = ks_mdmac_combi_midx(mdmac_k(arg->mac),
                            mdmac_r(arg->mac), (const MDeg**) arg->mac->degs,
                            arg->mac->degs_sz, m);
    }
}

static inline bool
mdmac_iter_nonlinear_mono(MDeg* d, uint64_t idx, void* __arg) {
    (void) idx;
    struct MDMacColSplitArg* arg = __arg;
    if(mdeg_total_deg(d) >= 2) { // non-linear
        uint32_t mono_size = mdeg_total_deg(arg->mac->mdeg);
        mdmac_iter_mono(mono_size, d, arg);
    }

    return false;
}

static inline bool
mdmac_iter_linear_mono(MDeg* d, uint64_t idx, void* __arg) {
    (void) idx;
    if(mdeg_total_deg(d) < 2) // linear
        mdmac_iter_mono(1, d, __arg);

    return false;
}

/* usage: Given a struct MDMac, return the set of column indices that correspond
 *      to the non-linear monomials
 * params:
 *      1) m: ptr to struct MDMac
 *      2) idxs: container for indices
 *      3) sz: size of cidxs. Must >= the number of non-linear monomials
 * return: 0 on success, 1 if cidxs is too small to hold all the indices */
int32_t
mdmac_nlcol_idxs(const MDMac* restrict m, uint64_t* restrict idxs, uint32_t sz) {
    if(sz < mdmac_num_nlcol(m))
        return 1;

    if(m->degs_sz == 0) {
        uint64_t lcol_num = mdmac_num_linear_col(m);
        for(uint64_t i = 0; i < mdmac_num_nlcol(m); ++i)
            idxs[i] = lcol_num++;
    } else { // combined multi-degrees
        struct MDMacColSplitArg arg = { .mac = m, .idxs = idxs, .offset = 0 };
        mdeg_iter_subdegs_union((const MDeg**) m->degs, m->degs_sz,
                                 mdmac_iter_nonlinear_mono, &arg);
        assert(arg.offset == mdmac_num_nlcol(m));
    }

    return 0;
}

/* usage: Given a struct MDMac, return the set of column indices that correspond
 *      to the linear monomials
 * params:
 *      1) m: ptr to struct MDMac
 *      2) cidxs: container for indices
 *      3) sz: size of cidxs. Must >= the number of linear monomials
 * return: 0 on success, 1 if cidxs is too small to hold all the indices */
int32_t
mdmac_lcol_idxs(const MDMac* restrict m, uint64_t* restrict idxs, uint32_t sz) {
    if(sz < mdmac_num_linear_col(m))
        return 1;

    if(m->degs_sz == 0) {
        uint64_t lcol_num = mdmac_num_linear_col(m);
        for(uint64_t i = 0; i < lcol_num; ++i)
            idxs[i] = i;
    } else { // combined multi-degrees
        struct MDMacColSplitArg arg = { .mac = m, .idxs = idxs, .offset = 0 };
        mdeg_iter_subdegs_union((const MDeg**) m->degs, m->degs_sz,
                                 mdmac_iter_linear_mono, &arg);
        assert(arg.offset == mdmac_num_linear_col(m));
    }

    return 0;
}

/* usage: map a variable to its column index in the MDMac
 * params:
 *      1) m: ptr to struct MDMac
 *      2) vidx: an index representing the variable. 0 ~ k-1 for
 *          the linear variable, k ~ k + r * c for the kernel
 *          variables
 * return: the column index */
uint64_t
mdmac_vidx_to_midx(const MDMac* restrict m, uint64_t vidx) {
    uint32_t k = mdmac_k(m);
    uint32_t r = mdmac_r(m);
    assert(vidx <= ks_total_var_num(k, r, mdmac_c(m)));
    mono_create_static_buf(mono_buf, 1);
    Mono* mono = mono_create_from_arr(1, mono_buf);
    mono_set_var(mono, 0, vidx, false);

    if(m->degs_sz == 0) // single multi-degree
        return ks_mdmac_midx(k, r, m->mdeg, mono);

    // combined multi-degrees
    return ks_mdmac_combi_midx(k, r, (const MDeg**) m->degs, m->degs_sz, mono);
}

/* usage: Given a struct MDMac, print it
 * params:
 *      1) m: ptr to struct MDMac
 * return: void */
void
mdmac_print(const MDMac* m) {
    for(uint64_t i = 0; i < mdmac_nrow(m); ++i) {
        for(uint64_t j = 0; j < mdmac_ncol(m); ++j)
            printf("%02u ", mdmac_at(m, i, j));
        printf("\n");
    }
}

/* usage: Given the number of linear variables, target rank, the number
 *      of rows in the left multiplier, and an array of target multi-degrees
 *      which define a combined multi-degree, compute the total number of
 *      multipliers (monomials) for the MDMac defined over the combined
 *      multi-degree
 * params:
 *      1) k: number of linear variables
 *      2) r: target rank
 *      3) degs: an array of ptrs to struct MDeg
 *      4) sz: size of degs
 * return: total number of multipliers */
uint64_t
mdmac_combi_multiplier_num(uint32_t k, uint32_t r, MDeg** degs, uint32_t sz) {
    const uint32_t c = mdeg_c(degs[0]);
    for(uint32_t j = 0; j < sz; ++j) {
        assert(c == mdeg_c(degs[j]));
        assert(mdeg_lv_deg(degs[j]) >= 1);
        mdeg_lv_deg_dec(degs[j]);
    }

    uint64_t n = 0;
    for(uint32_t i = 0; i < c; ++i) {
        for(uint32_t j = 0; j < sz; ++j) {
            assert(mdeg_kv_deg(degs[j], i) >= 1);
            mdeg_kv_deg_dec(degs[j], i);
        }

        n += ks_mdmac_combi_total_mono_num(k, r, (const MDeg**) degs, sz);

        for(uint32_t j = 0; j < sz; ++j) // restore
            mdeg_kv_deg_inc(degs[j], i);
    }

    for(uint32_t j = 0; j < sz; ++j) // restore
        mdeg_lv_deg_inc(degs[j]);

    return n;
}

/* usage: Given the MinRank instance, and an array of multi-degrees, compute
 *      the number of rows in the MDMac defined over the combined multi-degrees.
 * params:
 *      1) mr: ptr to struct MinRank
 *      2) degs: an array of ptrs to struct MDeg
 *      3) sz: size of degs
 * return: total number of eqs in the multi-degree Macaulay matrix */
uint64_t
mdmac_combi_eq_num(const MinRank* restrict mr, MDeg** degs, uint32_t sz) {
    uint64_t num = mdmac_combi_multiplier_num(minrank_nmat(mr),
                                              minrank_rank(mr), degs, sz);
    return num * minrank_ncol(mr);
}

static inline void
mdmac_combi_cmp_mmap_base(gfa_idx_t* restrict mmap, uint32_t k, uint32_t r,
                          const MDeg** degs, uint32_t degs_sz) {
    const uint32_t c = mdeg_c(degs[0]);
    const uint32_t vnum = ks_total_var_num(k, r, c);
    mono_create_static_buf(cidxs, 2); // at most 2 vars in a monomial
    Mono* mul = mono_create_from_arr(2, cidxs);
    uint64_t dst_idx = 0;
    mono_set_deg(mul, 0);
    mmap[dst_idx++] = ks_mdmac_combi_midx(k, r, degs, degs_sz, mul);

    // kernel vars and linear vars
    mono_set_deg(mul, 1);
    for(uint32_t i = vnum; i > 0; --i) {
        mono_set_var(mul, 0, i-1, false); // no need to sort
        mmap[dst_idx++] = ks_mdmac_combi_midx(k, r, degs, degs_sz, mul);
    }

    // degree-2 monomials. Note that the base KS system only have vi * xj,
    // where vi is a kernel var and xj is a linear var
    mono_set_deg(mul, 2);
    for(uint32_t i = vnum-1; i >= k; --i) {
        mono_set_var(mul, 1, i, false);
        for(uint32_t j = k; j > 0; --j) {
            mono_set_var(mul, 0, j-1, false); // no need to sort
            mmap[dst_idx++] = ks_mdmac_combi_midx(k, r, degs, degs_sz, mul);
        }
    }
    assert(dst_idx == ks_base_total_mono_num(k, r, c));
}

static inline void
mdmac_combi_cmp_mmap_mono(gfa_idx_t* restrict mmap, Mono* restrict mono,
                          const Mono* restrict mul, uint32_t k, uint32_t r,
                          const MDeg** degs, uint32_t degs_sz) {
    const uint32_t c = mdeg_c(degs[0]);
    const uint32_t vnum = ks_total_var_num(k, r, c);
    uint64_t dst_idx = 0;

    // constant term
    mmap[dst_idx++] = ks_mdmac_combi_midx(k, r, degs, degs_sz, mul);

    // kernel vars and linear vars
    for(uint32_t i = vnum; i > 0; --i) {
        mono_copy_partial_from(mono, mul);
        mono_set_deg(mono, mono_deg(mul) + 1);
        mono_set_var(mono, mono_deg(mul), i-1, true);
        mmap[dst_idx++] = ks_mdmac_combi_midx(k, r, degs, degs_sz, mono);
    }

    // deg-2 monomials in the base KS system
    for(uint32_t i = vnum-1; i >= k; --i) {
        for(uint32_t j = k; j > 0; --j) {
            mono_copy_partial_from(mono, mul);
            mono_set_deg(mono, mono_deg(mul) + 2);
            mono_set_var(mono, mono_deg(mul), i, false);
            mono_set_var(mono, mono_deg(mul)+1, j-1, true);
            mmap[dst_idx++] = ks_mdmac_combi_midx(k, r, degs, degs_sz, mono);
        }
    }
    assert(dst_idx == ks_base_total_mono_num(k, r, c));
}

struct MDMacMulAndFillArg {
    uint32_t mono_size;
    uint64_t dst_row_offset;
    uint64_t src_row_offset;
    gfa_idx_t* mmap;
    const MDeg** degs;
    MDMac* m;
    const GFM* ks;
};

static inline bool
mdmac_mul_and_fill(MDeg* d, uint64_t idx, void* __arg) {
    struct MDMacMulAndFillArg* arg = __arg;
    if(unlikely(idx == 0)) { // (0, 0, ... 0)
        mdmac_combi_cmp_mmap_base(arg->mmap, mdmac_k(arg->m), mdmac_r(arg->m),
                                  arg->degs, arg->m->degs_sz);
        mdmac_fill_in_eqs(arg->m, arg->dst_row_offset, arg->ks,
                          arg->src_row_offset, arg->mmap);
        arg->dst_row_offset += mdmac_m(arg->m);
        return false;
    }

    mono_create_static_buf(mul_buf, arg->mono_size);
    mono_create_static_buf(mono_buf, arg->mono_size + 2);
    Mono* mul = mono_create_from_arr(arg->mono_size, mul_buf);
    Mono* mono = mono_create_from_arr(arg->mono_size + 2, mono_buf);

    mdmac_mdeg_first(mul, d, mdmac_k(arg->m), mdmac_r(arg->m));
    mdmac_combi_cmp_mmap_mono(arg->mmap, mono, mul, mdmac_k(arg->m),
                              mdmac_r(arg->m), arg->degs, arg->m->degs_sz);
    mdmac_fill_in_eqs(arg->m, arg->dst_row_offset, arg->ks,
                      arg->src_row_offset, arg->mmap);
    arg->dst_row_offset += mdmac_m(arg->m);

    while(mdmac_mdeg_iterate(mul, d, mdmac_k(arg->m), mdmac_r(arg->m))) {
        mdmac_combi_cmp_mmap_mono(arg->mmap, mono, mul, mdmac_k(arg->m),
                                  mdmac_r(arg->m), arg->degs, arg->m->degs_sz);
        mdmac_fill_in_eqs(arg->m, arg->dst_row_offset, arg->ks,
                          arg->src_row_offset, arg->mmap);
        arg->dst_row_offset += mdmac_m(arg->m);
    }

    return false;
}

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
                           const MDeg** restrict degs, uint32_t sz) {
    assert(ks && mr && degs && sz);
    for(uint32_t i = 0; i < sz; ++i)
        if(!mdmac_check_mdeg(degs[i]))
            return NULL;

    MDMac* m = NULL;
    MDeg** degs_copy = calloc(sz, sizeof(MDeg*));
    if(!degs_copy)
        return NULL;
    for(uint32_t i = 0; i < sz; ++i) {
        degs_copy[i] = mdeg_dup(degs[i]);
        if(!degs_copy[i]) {
            mdmac_free_degs(degs_copy, sz);
            free(degs_copy);
            return NULL;
        }
    }

    const uint32_t c = mdeg_c(degs[0]);
    const uint32_t k = minrank_nmat(mr), r = minrank_rank(mr);
    assert(ks_base_total_mono_num(k, r, c) == gfm_ncol(ks));
    uint64_t ncol = ks_mdmac_combi_total_mono_num(k, r, degs, sz);
    uint64_t nrow = mdmac_combi_eq_num(mr, degs_copy, sz);
    if(unlikely(nrow == 0))
        return NULL;

    const uint64_t max_tnum = gfm_find_max_tnum_per_eq(ks);
    const size_t memblk_sz = gfa_size_of_element() * nrow * max_tnum;
    m = malloc(sizeof(MDMac) + memblk_sz);
    if(!m)
        return NULL;

    m->degs = degs_copy;
    m->degs_sz = sz;
    m->mdeg = NULL;
    memset(m->memblk, 0x0, memblk_sz);
    // right after memblk used by GFA

    // TODO
    m->mono_num_per_deg = NULL;

    m->rows = gfa_arr_create(max_tnum, nrow, m->memblk);
    if(!m->rows) {
        mdmac_free(m);
        return NULL;
    }

    gfa_idx_t* mmap = malloc(sizeof(gfa_idx_t) * ks_base_total_mono_num(k, r, c));
    if(!mmap) {
        mdmac_free(m);
        return NULL;
    }

    m->k = k; m->r = r; m->c = c; m->m = minrank_ncol(mr);
    m->nrow = nrow; m->ncol = ncol;

    m->mdeg = mdeg_create_zero(c);
    if(!m->mdeg) {
        free(mmap);
        mdmac_free(m);
        return NULL;
    }
    mdeg_find_max_mdeg(m->mdeg, degs, sz);
    const uint32_t mono_size = mdeg_total_deg(m->mdeg);

    for(uint32_t i = 0; i < sz; ++i) {
        assert(c == mdeg_c(degs[i]));
        assert(mdeg_lv_deg(degs[i]) >= 1);
        mdeg_lv_deg_dec(m->degs[i]);
    }

    uint64_t dst_row_offset = 0;
    uint64_t src_row_offset = 0;
    for(uint32_t i = 0; i < c; ++i) {
        for(uint32_t j = 0; j < sz; ++j) {
            assert(mdeg_kv_deg(m->degs[j], i) >= 1);
            mdeg_kv_deg_dec(m->degs[j], i);
        }

        struct MDMacMulAndFillArg arg = {
            .mono_size = mono_size, .dst_row_offset = dst_row_offset,
            .src_row_offset = src_row_offset, .mmap = mmap, .degs = degs,
            .m = m, .ks = ks,
        };
        mdeg_iter_subdegs_union((const MDeg**) m->degs, sz,
                                mdmac_mul_and_fill, &arg);
        dst_row_offset = arg.dst_row_offset;
        src_row_offset += mdmac_m(m);

        for(uint32_t j = 0; j < sz; ++j) // restore
            mdeg_kv_deg_inc(m->degs[j], i);
    }

    for(uint32_t i = 0; i < sz; ++i) // restore
        mdeg_lv_deg_inc(m->degs[i]);

    assert(nrow == dst_row_offset);
    free(mmap);
    return m;
}
