#ifndef __KS_H__
#define __KS_H__

#include <stdint.h>
#include <stdbool.h>

#include "mono.h"
#include "mdeg.h"
#include "gfm.h"

/* usage: Given the index of a linear variable in a KS system, compute its
 *      overall index in the total degree-1 monomial order including the
 *      kernel variables.
 * params:
 *      1) idx: index of the linear variable. 0 for x1, 1 for x2, and so on
 *      2) r: target rank of the MinRank instance that the KS system is for
 *      3) c: number of rows in the KS matrix
 * return: index of the linear variable in the overall degree-1 monomial order */
uint32_t
ks_linear_var_idx(uint32_t idx);

/* usage: Compute the index used to specify a kernel index
 * params:
 *      1) ri: row index of the kernel variable. 0 for the first row
 *      2) ci: column index of the kernel variable. 0 for the first column
 *      3) r: target rank of the MinRank instance that the KS system is for
 * return: index used to specify the kernel variable */
uint32_t
ks_kernel_var_idx_from_2d(uint32_t ri, uint32_t ci, uint32_t r);

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
ks_kernel_var_idx_to_2d(uint32_t idxs[2], uint32_t idx, uint32_t k, uint32_t r);

/* usage: Compute the group index of the given kernel variable
 * params:
 *      1) idx: 1D index of the kernel variable
 *      2) k: number of linear variables in the KS system
 *      3) r: target rank of the MinRank instance
 * return: group index of the kernel var */
uint32_t
ks_kernel_var_idx_to_grp_idx(uint32_t idx, uint32_t k, uint32_t r);

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
ks_kernel_var_idx(uint32_t ri, uint32_t ci, uint32_t k, uint32_t r, uint32_t c);

/* usage: Given parameters of a KS system, compute the total number of variables
 * params:
 *      1) k: number of linear variables
 *      2) r: target rank of the MinRank instance
 *      3) c: number of rows in the KS matrix
 * return: total number of variables */
uint32_t
ks_total_var_num(uint32_t k, uint32_t r, uint32_t c);

/* usage: Given parameters of a KS system, compute the total number of degree-2
 *      monomials. Note that sequare terms xi^2 are not considered in the base KS system.
 * params:
 *      1) k: number of linear variables
 *      2) r: target rank of the MinRank instance
 *      3) c: number of rows in the KS matrix
 * return: total number of variables */
uint32_t
ks_base_total_d2_num(uint32_t k, uint32_t r, uint32_t c);

/* usage: Given parameters of a KS system, compute the total number of monomials with
 *      degree <= 2
 * params:
 *      1) k: number of linear variables
 *      2) r: target rank of the MinRank instance
 *      3) c: number of rows in the KS matrix
 * return: total number of monomials with degree <= 2 */
uint32_t
ks_base_total_mono_num(uint32_t k, uint32_t r, uint32_t c);

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
ks_midx(uint32_t k, uint32_t r, uint32_t c, const Mono* m);

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
                        const MDeg* restrict mdeg);

/* usage: Given a multi-degree, compute the number of monomials with that multi-degree
 * params:
 *      1) k: number of linear variables
 *      2) r: target rank of the MinRank instance
 *      3) d: ptr to struct MDeg
 * return: number of monomials with the given multi-degree */
uint64_t
ks_mdmac_mdeg_mono_num(uint32_t k, uint32_t r, const MDeg* d);

/* usage: Given a multi-degree, compute the number of monomials <= that multi-degree
 * params:
 *      1) k: number of linear variables
 *      2) r: target rank of the MinRank instance
 *      3) mdeg: ptr to struct MDeg
 * return: number of monomials with degree <= the given multi-degree */
uint64_t
ks_mdmac_total_mono_num(uint32_t k, uint32_t r, const MDeg* mdeg);

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
                              const MDeg** restrict degs, uint32_t sz);

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
ks_mdmac_midx_check(uint32_t mvnum, uint32_t len, const uint32_t mdeg[]);

#if defined(GFA_IDX_SIZE_64)

#define KS_MDMAC_MIDX_INVALID   (UINT64_MAX)

#else

#define KS_MDMAC_MIDX_INVALID   (UINT32_MAX)

#endif

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
              const Mono* restrict m);

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
ks_mac_midx(uint32_t k, uint32_t r, uint32_t c, const Mono* m);

/* usage: Given a kernel variable vij, compute the new indices of monomials with deg <= 2
 *      in the base KS system based on multiplier vij
 * params:
 *      1) nidxs: container for the new indices
 *      2) k: number of linear variables
 *      3) r: target rank of the MinRank instance
 *      4) c: number of rows in the KS matrix
 *      3) vidx: index of the kernel var. Mapped from its row and column indices with
 *          `ks_kernel_var_idx`
 * return: void */
void
ks_base_cmp_idx_map_d1(uint32_t* nidxs, uint32_t k, uint32_t r, uint32_t c, uint32_t vidx);

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
                    uint32_t sz, const Mono* restrict m);

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
             const MDeg* restrict d);

/* usage: Generate a random Kipnis-Shamir matrix
 * params:
 *      1) k: number of linear variables
 *      2) r: target rank of the MinRank instance
 *      3) c: number of groups of kernel variables
 *      4) m: number of columns of matrices in the original MinRank instance
 * return: ptr to struct GFM, on error NULL */
GFM*
ks_rand(uint32_t k, uint32_t r, uint32_t c, uint32_t m);

#endif // __KS_H__
