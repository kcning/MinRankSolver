#include "ks.h"
#include "math_util.h"
#include "mdeg.h"
#include "mono.h"

#include <stddef.h>
#include <assert.h>

/* usage: Given the index of a linear variable in a KS system, compute its
 *      overall index in the total degree-1 monomial order including the
 *      kernel variables.
 * params:
 *      1) idx: index of the linear variable. 0 for x1, 1 for x2, and so on
 *      2) r: target rank of the MinRank instance that the KS system is for
 *      3) c: number of rows in the KS matrix
 * return: index of the linear variable in the overall degree-1 monomial order */
uint32_t
ks_linear_var_idx(uint32_t idx) {
    // the linear variables are placed first, before the kernel variables
    return idx;
}

/* usage: Compute the index used to specify a kernel index
 * params:
 *      1) ri: row index of the kernel variable. 0 for the first row
 *      2) ci: column index of the kernel variable. 0 for the first column
 *      3) r: target rank of the MinRank instance that the KS system is for
 * return: index used to specify the kernel variable */
uint32_t
ks_kernel_var_idx_from_2d(uint32_t ri, uint32_t ci, uint32_t r) {
    assert(ci < r);
    return ri * r + ci;
}

/* usage: Compute the 2D indices (row index, column index) used to specify
 *      a kernel variable from its 1D index
 * params:
 *      1) idxs: container for the 2D indices. idxs[0] stores the row index,
 *          and idxs[1] the column index
 *      2) idx: 1D index of the kernel variable
 *      3) k: number of linear variables in the KS system
 *      4) r: target rank of the MinRank instance
 * return: void */
void
ks_kernel_var_idx_to_2d(uint32_t idxs[2], uint32_t idx, uint32_t k, uint32_t r) {
    assert(idx >= k); // must be a kernel var
    uint32_t tmp = idx - k;
    idxs[0] = tmp / r; // ri
    idxs[1] = tmp % r; // ci
}

/* usage: Compute the group index of the given kernel variable
 * params:
 *      1) idx: 1D index of the kernel variable
 *      2) k: number of linear variables in the KS system
 *      3) r: target rank of the MinRank instance
 * return: group index of the kernel var */
uint32_t
ks_kernel_var_idx_to_grp_idx(uint32_t idx, uint32_t k, uint32_t r) {
    uint32_t tmp[2];
    ks_kernel_var_idx_to_2d(tmp, idx, k, r);
    return tmp[0];
}

/* usage: Given the indices of a kernel variable in a KS system, compute its
 *      overall index in the total degree-1 monomial order including the
 *      linear variables.
 * params:
 *      1) ri: row index of the kernel variable. 0 for the first row
 *      2) ci: column index of the kernel variable. 0 for the first column
 *      3) k: number of linear variables in the KS system
 *      4) r: target rank of the MinRank instance that the KS system is for
 *      5) c: number of rows in the KS matrix
 * return: index of the kernel variable in the overall degree-1 monomial order */
uint32_t
ks_kernel_var_idx(uint32_t ri, uint32_t ci, uint32_t k, uint32_t r, uint32_t c) {
    (void) c;
    assert(ri < c);
    uint32_t kernel_var_idx = ks_kernel_var_idx_from_2d(ri, ci, r);
    // the linear variables are placed first, before the kernel variables
    return k + kernel_var_idx;
}

/* usage: Given parameters of a KS system, compute the total number of variables
 * params:
 *      1) k: number of linear variables
 *      2) r: target rank of the MinRank instance
 *      3) c: number of rows in the KS matrix
 * return: total number of variables */
uint32_t
ks_total_var_num(uint32_t k, uint32_t r, uint32_t c) {
    return k + r * c;
}

/* usage: Given parameters of a KS system, compute the total number of degree-2
 *      monomials. Note that sequare terms xi^2 are not considered in the base KS system.
 * params:
 *      1) k: number of linear variables
 *      2) r: target rank of the MinRank instance
 *      3) c: number of rows in the KS matrix
 * return: total number of variables */
uint32_t
ks_base_total_d2_num(uint32_t k, uint32_t r, uint32_t c) {
    return k * r * c;
}

/* usage: Given parameters of a KS system, compute the total number of monomials with
 *      degree <= 2
 * params:
 *      1) k: number of linear variables
 *      2) r: target rank of the MinRank instance
 *      3) c: number of rows in the KS matrix
 * return: total number of monomials with degree <= 2 */
uint32_t
ks_base_total_mono_num(uint32_t k, uint32_t r, uint32_t c) {
    return ks_base_total_d2_num(k, r, c) + ks_total_var_num(k, r, c) + 1;
}

/* usage: Given a kernel variable vij, compute the new indices of monomials with deg <= 2
 *      in the base KS system based on multiplier vij
 * params:
 *      1) nidxs: container for the new indices
 *      2) k: number of linear variables
 *      3) r: target rank of the MinkRank instance
 *      4) c: number of rows in the KS matrix
 *      3) vidx: index of the kernel var. Mapped from its row and column indices with
 *          `ks_kernel_var_idx`
 * return: void */
void
ks_base_cmp_idx_map_d1(uint32_t* nidxs, uint32_t k, uint32_t r, uint32_t c, uint32_t vidx) {
    mono_create_static_buf(mono_buf, 2);
    Mono* m = mono_create_from_arr(2, mono_buf); // at most 2 vars in a monomial
    uint32_t dst_idx = 0; // where in nidxs to store the next calculated result

    // constant term * vij
    mono_set_deg(m, 1);
    mono_set_var(m, 0, vidx, false);
    nidxs[dst_idx++] = ks_midx(k, r, c, m);

    // linear var * vij
    mono_set_deg(m, 2);
    mono_set_var(m, 1, vidx, false);
    for(uint32_t i = 0; i < k; ++i) {
        mono_set_var(m, 0, i, false);
        nidxs[dst_idx++] = ks_midx(k, r, c, m);
    }

    assert(dst_idx == (k + 1));

    // NOTE: in a KS system, kernel variables in the left matrix is
    // multiplied with the degree-1 linear variables in M_{\lambda}
    // one by one. Therefore, we do not need to consider higher
    // degree monomials
}

/* usage: Compute the number of monomials of each degree in a
 *      multi-degree Macaulay matrix derived form a base KS system.
 * params:
 *      1) mono_nums: container for the result. Must hold at least
 *          as uint64_t as the total degree of the multi-degree + 1
 *      2) k: number of linear variables
 *      3) r: target rank of the MinkRank instance
 *      4) mdeg: ptr to struct MDeg
 * return: void. On return, mono_nums[i] holds the number of degree-i
 *      monomials */
void
ks_mdmac_calc_mono_nums(uint64_t* restrict mono_nums, uint32_t k, uint32_t r,
                        const MDeg* restrict mdeg) {
    // example: mdeg = (2, 2, 1), k = 6, r = 4, c = 2
    // linear variable: 1 + binom(k+1-1,1)z + binom(k+2-1,2)z^2
    // 1st group of kernel vars: 1 + binom(r+1-1,1)z + binom(r+2-1,2)z^2
    // 2nd group of kernel vars: 1 + binom(r+1-1,1)z
    // their product: 1 + 14z + 95z^2 + 364z^3 + 786z^4 + 840z^5
    // number of constant term: 1
    // number of degree-1 term: 14
    // number of degree-2 term: 95
    // number of degree-3 term: 364
    // number of degree-4 term: 786
    // number of degree-5 term: 840

    // +1 to include the constant term
    const uint32_t out_size = mdeg_total_deg(mdeg) + 1;
    memset(mono_nums, 0x0, sizeof(uint64_t) * out_size);
    mono_nums[0] = 1; // constant
    for(uint32_t i = 1; i <= mdeg_lv_deg(mdeg); ++i) {
        mono_nums[i] = binom(k + i - 1, i);
    }
    // number of terms in the current product
    uint32_t cur_term_num = 1 + mdeg_lv_deg(mdeg);

    uint64_t tmp_prod[out_size];
    for(uint32_t ci = 0; ci < mdeg_c(mdeg); ++ci) {
        // multiply the current product with the constant term
        memcpy(tmp_prod, mono_nums, sizeof(uint64_t) * cur_term_num);
        for(uint32_t i = 1; i <= mdeg_kv_deg(mdeg, ci); ++i) {
            uint64_t coeff = binom(r + i - 1, i);
            for(uint32_t j = 0; j < cur_term_num; ++j) {
                mono_nums[i + j] += coeff * tmp_prod[j];
            }
        }
        cur_term_num += mdeg_kv_deg(mdeg, ci);
    }
}

/* subroutine of ks_midx and ks_mdmac_midx: compute the offset into
 * a group of degree-1 monomials */
static inline uint32_t
ks_midx_d1(uint32_t k, uint32_t r, uint32_t c, uint32_t vidx) {
    return ks_total_var_num(k, r, c) - vidx;
}

/* usage: Compute the column index of the monomial xixj...vpvr..., where xi's are
 *      linear variables and vi's kernel variables. The indices of the kernel
 *      variables are mapped from their original 2D indices to the 1D ones with
 *      function `ks_kernel_var_idx`.
 * params:
 *      1) k: number of linear variables
 *      2) r: target rank of the MinRank instance
 *      3) c: number of rows in the KS matrix
 *      4) m: ptr to struct Mono. the monomial in question
 * return: index of the monomial in the grlex monomial order */
uint32_t
ks_midx(uint32_t k, uint32_t r, uint32_t c, const Mono* m) {
    assert(mono_deg(m) <= 2); // sanity check
     uint32_t idx = 0;
     switch(mono_deg(m)) {
         case 2:
            assert(mono_var(m, 1) >= k);
            idx += ks_base_total_d2_num(k, r, c) - k * (mono_var(m, 1) - k);
            // fall through
        case 1:
            idx += ks_midx_d1(k, r, c, mono_var(m, 0));
            // fall through
        case 0:
            ; // do nothing
    }
    assert(idx < UINT32_MAX);
    return idx;
}

/* usage: Given a multi-degree, compute the number of monomials with that multi-degree
 * params:
 *      1) k: number of linear variables
 *      2) r: target rank of the MinRank instance
 *      3) d: ptr to struct MDeg
 * return: number of monomials with the given multi-degree */
uint64_t
ks_mdmac_mdeg_mono_num(uint32_t k, uint32_t r, const MDeg* d) {
    uint64_t n = binom(k + mdeg_lv_deg(d) - 1, mdeg_lv_deg(d));
    for(uint32_t i = 1; i <= mdeg_c(d); ++i)
        n *= binom(r + mdeg_deg(d, i) - 1, mdeg_deg(d, i));
    return n;
}

/* usage: Given a multi-degree, compute the number of monomials <= that multi-degree
 * params:
 *      1) k: number of linear variables
 *      2) r: target rank of the MinRank instance
 *      3) mdeg: ptr to struct MDeg
 * return: number of monomials with degree <= the given multi-degree */
uint64_t
ks_mdmac_total_mono_num(uint32_t k, uint32_t r, const MDeg* mdeg) {
    uint64_t n = binom(k + mdeg_lv_deg(mdeg), mdeg_lv_deg(mdeg));
    for(uint32_t i = 0; i < mdeg_c(mdeg); ++i)
        n *= binom(r + mdeg_kv_deg(mdeg, i), mdeg_kv_deg(mdeg, i));
    return n;
}

struct CountMonoArg {
    uint32_t k;
    uint32_t r;
    uint64_t count;
};

static inline bool
count_mono_per_mdeg(MDeg* restrict d, uint64_t idx, void* restrict __arg) {
    (void) idx;
    struct CountMonoArg* arg = __arg;
    arg->count += ks_mdmac_mdeg_mono_num(arg->k, arg->r, d);
    return false;
}

/* usage: Given multi-degrees d1, d2, .. dn, compute the number of monomials <=
 *      any of the multi-degrees.
 * params:
 *      1) k: number of linear variables
 *      2) r: target rank of the MinRank instance
 *      3) degs: an array of ptrs to struct MDeg
 *      4) sz: size of degs
 * return: number of monomials */
uint64_t
ks_mdmac_combi_total_mono_num(uint32_t k, uint32_t r,
                              const MDeg** restrict degs, uint32_t sz) {
    assert(sz);
    struct CountMonoArg arg = { .k = k, .r = r, .count = 0 };
    mdeg_iter_subdegs_union(degs, sz, count_mono_per_mdeg, &arg);
    return arg.count;
}

/* usage: Given the total degree of the monomial, check if it's consistent with
 *      its multi-degree. For total degree <= 2, this function returns true
 *      immediately because a multi-degree Macaulay matrix we build from a
 *      KS system does no have such monomials.
 * params:
 *      1) mvnum: total degree of the monomial
 *      2) len: length of the multi-degree array
 *      3) mdeg: multi-degree array. See ks_mdmac_midx for its format
 * return: true if consistent, false otherwise */
bool
ks_mdmac_midx_check(uint32_t mvnum, uint32_t len, const uint32_t mdeg[]) {
    if(mvnum <= 2)
        return true;

    uint32_t sum = sum_arr(mdeg, len);
    return mvnum == sum;
}

/* usage: Compute the column index of the monomial in the full Macaulay
 *      matrix built from a KS system. Note that monomials that will not
 *      appear from multi-degree construction are also included. The indices
 *      of the kernel variables are mapped from their original 2D indices to
 *      the 1D ones with function `ks_kernel_var_idx`.
 * params:
 *      1) k: number of linear variables
 *      2) r: target rank of the MinRank instance
 *      3) c: number of rows in the KS matrix
 *      4) m: ptr to struct Mono. the monomial in question
 * return: index of the monomial in the grlex monomial order */
uint64_t
ks_mac_midx(uint32_t k, uint32_t r, uint32_t c, const Mono* m) {
    uint64_t idx = 0;
    const uint32_t total_vnum = ks_total_var_num(k, r, c);
    for(uint32_t i = mono_deg(m); i > 0; --i) {
        idx += binom(total_vnum + i - 1, i) - binom(mono_var(m, i-1) + i - 1, i);
    }
    return idx;
}

/* subroutine of ks_mdmac_midx */
static inline uint32_t
ks_mdmac_midx_cmp_grp_idx(uint32_t* out, uint32_t var, uint32_t k, uint32_t r) {
    uint32_t grp_idx, vidx_in_grp;
    if(var < k) { // var is a linear var
        grp_idx = 0;
        vidx_in_grp = var;
    } else { // var is a kernel var
        uint32_t tmp[2];
        ks_kernel_var_idx_to_2d(tmp, var, k, r);
        grp_idx = tmp[0] + 1; // the group of linear vars has index 0
        vidx_in_grp = tmp[1];
    }
    *out = vidx_in_grp + 1; // including itself
    return grp_idx;
}

/* subroutine of ks_mdmac_midx*: compute the monomial index of a monomial per
 *      the given multi-degree. The monomial must be valid for the multi-degree
 */
static inline uint64_t
ks_mdmac_midx_internal(uint32_t k, uint32_t r, const MDeg* restrict d,
                       const Mono* restrict m, uint64_t offset,
                       uint64_t last_mdeg_mono_num) {
    const uint32_t c = mdeg_c(d);
    mdeg_create_static_buf(tmp_mdeg_buf, c);
    MDeg* tmp_mdeg = mdeg_create_from_arr(c, tmp_mdeg_buf);
    mdeg_copy(tmp_mdeg, d);

    const uint32_t out_sz = mdeg_total_deg(d) + 1;
    uint64_t mono_nums[out_sz];
    uint64_t tmp_prod[out_sz];
    uint64_t idx = offset;
    uint64_t full_step = last_mdeg_mono_num;
    // fix var one by one, starting from the largest var
    for(uint32_t i = mono_deg(m); i > 0; --i) {
        idx += full_step;
        uint32_t last_var = mono_var(m, i-1);
        uint32_t grp_idx; // which group the last var belongs to
        uint32_t vnum_in_grp; // number of vars <= the last var in its group
        grp_idx = ks_mdmac_midx_cmp_grp_idx(&vnum_in_grp, last_var, k, r);
        assert(grp_idx <= c);

        // polynomial representation for the last group
        uint32_t cur_term_num = 1 + mdeg_deg(tmp_mdeg, grp_idx);
        mono_nums[0] = 1;
        for(uint32_t j = 1; j < cur_term_num; ++j) {
            mono_nums[j] = binom(vnum_in_grp + j - 1, j);
        }
        memset(mono_nums + cur_term_num, 0x0,
               sizeof(uint64_t) * (out_sz - cur_term_num));

        // multiply the current product w/ poly. rep. of the remaining groups
        for(uint32_t j = 0; j < grp_idx; ++j) {
            // multiply the current product w/ the constant term
            memcpy(tmp_prod, mono_nums, sizeof(uint64_t) * cur_term_num);
            uint32_t vnum = (j == 0) ? k : r;
            for(uint32_t jj = 1; jj <= mdeg_deg(tmp_mdeg, j); ++jj) {
                uint64_t coeff = binom(vnum + jj - 1, jj);
                for(uint32_t kk = 0; kk < cur_term_num; ++kk) {
                    mono_nums[jj + kk] += coeff * tmp_prod[kk];
                }
            }
            cur_term_num += mdeg_deg(tmp_mdeg, j);
        }

        mdeg_deg_dec(tmp_mdeg, grp_idx);
        idx -= mono_nums[i];
        assert(mono_nums[i] <= full_step);

        // polynomial representation for the last group
        cur_term_num = 1 + mdeg_deg(tmp_mdeg, grp_idx);
        mono_nums[0] = 1;
        for(uint32_t j = 1; j <= mdeg_deg(tmp_mdeg, grp_idx); ++j) {
            mono_nums[j] = binom(vnum_in_grp + j - 1, j);
        }
        memset(mono_nums + cur_term_num, 0x0,
               sizeof(uint64_t) * (out_sz - cur_term_num));
        // multiply the current product w/ poly. rep. of the remaining groups
        for(uint32_t j = 0; j < grp_idx; ++j) {
            // multiply the current product with the constant term
            memcpy(tmp_prod, mono_nums, sizeof(uint64_t) * cur_term_num);
            uint32_t vnum = (j == 0) ? k : r;
            for(uint32_t jj = 1; jj <= mdeg_deg(tmp_mdeg, j); ++jj) {
                uint64_t coeff = binom(vnum + jj - 1, jj);
                for(uint32_t kk = 0; kk < cur_term_num; ++kk) {
                    mono_nums[jj + kk] += coeff * tmp_prod[kk];
                }
            }
            cur_term_num += mdeg_deg(tmp_mdeg, j);
        }
        full_step = mono_nums[i-1];
    }

    return idx;
}

/* usage: Compute the column index of the monomial in the multi-degree
 *      Macaulay matrix built from a KS system. The indices of the kernel
 *      variables are mapped from their original 2D indices to the 1D ones with
 *      function `ks_kernel_var_idx`.
 * params:
 *      1) k: number of linear variables
 *      2) r: target rank of the MinRank instance
 *      3) d: ptr to struct MDeg. target multi-degree
 *      4) m: ptr to struct Mono. the monomial in question
 * return: index of the monomial in the grlex monomial order.
 *      KS_MDMAC_MIDX_INVALID if the monomial is invalid for the given target
 *      multi-degree */
uint64_t
ks_mdmac_midx(uint32_t k, uint32_t r, const MDeg* restrict d,
              const Mono* restrict m) {
    if(!mono_check_mdeg(m, d, k, r))
        return KS_MDMAC_MIDX_INVALID;

    uint64_t mono_nums[mdeg_total_deg(d) + 1];
    ks_mdmac_calc_mono_nums(mono_nums, k, r, d);
    uint64_t idx = sum_arr(mono_nums, mono_deg(m));
    return ks_mdmac_midx_internal(k, r, d, m, idx, mono_nums[mono_deg(m)]);
}

/* usage: given a monomial that belongs to a multi-degree d, return its index
 *      in the monomials of that multi-degree according to grlex
 * params:
 *      1) k: number of linear variables
 *      2) r: number of kernel variables per row
 *      3) m: ptr to struct Mono
 *      4) d: ptr to struct MDeg
 * return: its index */
uint64_t
ks_mdeg_midx(uint32_t k, uint32_t r, const Mono* restrict m,
             const MDeg* restrict d) {
    assert(mono_deg(m) == mdeg_total_deg(d));
    const uint32_t c = mdeg_c(d);
    mdeg_create_static_buf(tmp_mdeg_buf, c);
    MDeg* tmp_mdeg = mdeg_create_from_arr(c, tmp_mdeg_buf);
    mdeg_copy(tmp_mdeg, d);
    uint32_t vnums[c + 1];
    vnums[0] = k;
    for(uint32_t i = 1; i <= c; ++i)
        vnums[i] = r;

    uint64_t idx = 0;
    uint64_t step_forward = ks_mdmac_mdeg_mono_num(k, r, d);
    for(uint32_t i = mono_deg(m); i > 0; --i) {
        idx += step_forward;
        uint32_t last_var = mono_var(m, i-1);
        uint32_t grp_idx; // which group the last var belongs to
        uint32_t vnum_in_grp; // number of vars <= the last var in its group
        grp_idx = ks_mdmac_midx_cmp_grp_idx(&vnum_in_grp, last_var, k, r);
        assert(grp_idx <= c);
        vnums[grp_idx] = vnum_in_grp;

        uint64_t step_backward = mdeg_mono_num(tmp_mdeg, vnums);
        assert(step_backward <= step_forward);
        idx -= step_backward;
        mdeg_deg_dec(tmp_mdeg, grp_idx); // the last var has been fixed
        step_forward = mdeg_mono_num(tmp_mdeg, vnums);
    }

    return idx;
}

struct CountMonoandCheckArg {
    uint32_t k;
    uint32_t r;
    uint64_t count;
    const MDeg* target_d;
};

static inline bool
count_mono_and_check(MDeg* restrict d, uint64_t idx, void* restrict __arg) {
    (void) idx;
    struct CountMonoandCheckArg* arg = __arg;
    if(mdeg_is_equal(d, arg->target_d))
        return true;

    arg->count += ks_mdmac_mdeg_mono_num(arg->k, arg->r, d);
    return false;
}

/* usage: Given multi-degrees d1, d2, .. dn, compute the column index of a
 *      monomial in a Macaulay matrix defined over those multi-degrees.
 * params:
 *      1) k: number of linear variables
 *      2) r: target rank of the MinRank instance
 *      3) degs: an array of ptrs to struct MDeg
 *      4) sz: size of degs
 *      5) m: ptr to struct Mono. the monomial in question
 * return: index of the monomial in the grlex monomial order.
 *      KS_MDMAC_MIDX_INVALID if the monomial is invalid for all of the
 *      multi-degrees */
uint64_t
ks_mdmac_combi_midx(uint32_t k, uint32_t r, const MDeg** restrict degs,
                    uint32_t sz, const Mono* restrict m) {
    assert(sz);
    mdeg_create_static_buf(target_d_buf, mdeg_c(degs[0]));
    MDeg* target_d = mdeg_create_from_arr(mdeg_c(degs[0]), target_d_buf);
    mono_to_mdeg(target_d, m, k, r);

    if(!mdeg_is_le_any(target_d, degs, sz))
        return KS_MDMAC_MIDX_INVALID;

    struct CountMonoandCheckArg arg = {
        .k = k, .r = r, .count = 0, .target_d = target_d
    };
    mdeg_iter_subdegs_union(degs, sz, count_mono_and_check, &arg);
    return arg.count + ks_mdeg_midx(k, r, m, target_d);
}

static force_inline void
gen_rand_ks_row(GFM* ks, uint32_t dst_ridx, uint32_t k, uint32_t r, uint32_t c,
                uint32_t ri) {
    mono_create_static_buf(mono_buf, 2);
    Mono* m = mono_create_from_arr(2, mono_buf);
    gf_t* dst_row = (gf_t*) gfm_row_addr(ks, dst_ridx);

    mono_set_deg(m, 0); // constant term
    dst_row[ks_midx(k, r, c, m)] = gf_t_rand();

    // linear var
    mono_set_deg(m, 1);
    for(uint32_t i = 0; i < k; ++i) {
        mono_set_var(m, 0, i, false);
        dst_row[ks_midx(k, r, c, m)] = gf_t_rand();
    }

    // kernel var of the selected group
    for(uint32_t i = 0; i < r; ++i) {
        uint32_t vidx = ks_kernel_var_idx(ri, i, k, r, c);
        mono_set_var(m, 0, vidx, false);
        dst_row[ks_midx(k, r, c, m)] = gf_t_rand();
    }

    // one kernel var of the selected group and 1 linear var
    mono_set_deg(m, 2);
    for(uint32_t i = 0; i < k; ++i) {
        mono_set_var(m, 0, i, false);
        for(uint32_t j = 0; j < r; ++j) {
            uint32_t vidx = ks_kernel_var_idx(ri, j, k, r, c);
            mono_set_var(m, 1, vidx, false);
            dst_row[ks_midx(k, r, c, m)] = gf_t_rand();
        }
    }
}

/* usage: Generate a random Kipnis-Shamir matrix
 * params:
 *      1) k: number of linear variables
 *      2) r: target rank of the MinRank instance
 *      3) c: number of groups of kernel variables
 *      4) m: number of columns of matrices in the original MinRank instance
 * return: ptr to struct GFM, on error NULL */
GFM*
ks_rand(uint32_t k, uint32_t r, uint32_t c, uint32_t m) {
    uint32_t nrow = c * m;
    uint32_t ncol = 1 + k + r * c + r * c * k;
    GFM* ks = gfm_create(nrow, ncol, NULL);
    if(!ks)
        return NULL;

    gfm_zero(ks);
    uint32_t dst_row_offset = 0;
    for(uint32_t i = 0; i < c; ++i) {
        for(uint32_t j = 0; j < m; ++j) { // for each m rows
            gen_rand_ks_row(ks, dst_row_offset++, k, r, c, i);
        }
    }

    return ks;
}
