/* minrank.c: implementation of minrank.h */

#include "minrank.h"
#include "gfm.h"
#include "ks.h"
#include "mono.h"

#include <stdint.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>


/* ========================================================================
 * struct MinRank definition
 * ======================================================================== */

struct MinRank {
    // matrices M0, M1, ..., Mk in F(q)^{nxm}
    uint32_t nrow;      // number of rows in a matrix Mi
    uint32_t ncol;      // number of columns in a matrix Mi
    uint32_t nmat;      // number of matrices in the homogeneous part (k)
    uint32_t rank;      // the target rank
    GFM* m0;            // matrix M0 (heterogeneous case)
    GFM* ms;            // matrices M1, M2, ..., Mk
};

/* ========================================================================
 * function implementations
 * ======================================================================== */

/* usage: Given a struct MinRank, return the number of rows of its matrices
 * params:
 *      1) mr: ptr to MinRank
 * return: number of rows */
uint32_t
minrank_nrow(const MinRank* mr) {
    return mr->nrow;
}

/* usage: Given a struct MinRank, return the number of columns of its matrices
 * params:
 *      1) mr: ptr to MinRank
 * return: number of columns */
uint32_t
minrank_ncol(const MinRank* mr) {
    return mr->ncol;
}

/* usage: Given a struct MinRank, return its number of matrices
 * params:
 *      1) mr: ptr to MinRank
 * return: number of matrices */
uint32_t
minrank_nmat(const MinRank* mr) {
    return mr->nmat;
}

/* usage: Given a struct MinRank, return its target rank
 * params:
 *      1) mr: ptr to MinRank
 * return: the target rank */
uint32_t
minrank_rank(const MinRank* mr) {
    return mr->rank;
}

/* usage: Given a struct MinRank, return its inhomogeneous matrix M0
 * params:
 *      1) mr: ptr to MinRank
 * return: ptr to GFM if the MinRank instance is inhomogeneous, otherwise NULL */
const GFM*
minrank_m0(const MinRank* mr) {
    return mr->m0;
}

/* usage: Given a struct MinRank, return its i-th matrix Mi in the homogeneous part
 * params:
 *      1) mr: ptr to MinRank
 *      2) i: index of the matrix. 1 <= i <= minrank_nmat(mr)
 * return: ptr to GFM, on error NULL */
const GFM*
minrank_matrix(const MinRank* mr, uint32_t i) {
    if(!i || i > minrank_nmat(mr))
        return NULL;

    return gfm_arr_at(mr->ms, i-1);
}

/* usage: Given a struct MinRank, return a coefficient in the target matrix
 * params:
 *      1) mr: ptr to MinRank
 *      2) mi: index of the matrix. 0 <= mi <= minrank_nmat(mr)
 *      3) ri: row index
 *      4) ci: column index
 * return: the coefficient in question */
gf_t
minrank_coeff(const MinRank* mr, uint32_t mi, uint32_t ri, uint32_t ci) {
    const GFM* m = NULL;
    if(mi == 0) {
        m = minrank_m0(mr);
    } else {
        m = minrank_matrix(mr, mi);
    }
    assert(m != NULL);
    assert(ri < minrank_nrow(mr));
    assert(ci < minrank_ncol(mr));
    return gfm_at(m, ri, ci);
}

/* usage: Given the description of a MinRank problem, create a struct for it
 * params:
 *      1) nrow: number of rows in each matrix
 *      2) ncol: number of columns in each matrix
 *      3) k: number of matrices in the homogeneous part
 *      4) r: target rank
 *      5) m0: optional ptr to struct GFM representing M0. If NULL, the
 *          problem is homogeneous. Note that if not NULL, the ownership of m0 is
 *          transferred to this MinRank instance. One must not call the destructor
 *          on m0. Doing so will result in a double free.
 *      6) ms: optional ptr to an array of struct GFM, representing M1 ~ Mk.
 *          If NULL, the M1 ~ Mk will be randomly sampled. Note that if not
 *          NULL, the ownership of ms is transferred to this MinRank instance.
 *          One must not call the destructor on ms. Doing so will result in a
 *          double free.
 * return: ptr to MinRank, on error return NULL */
MinRank*
minrank_create(uint32_t nrow, uint32_t ncol, uint32_t k, uint32_t r,
               const GFM* restrict m0, const GFM* restrict ms) {
    MinRank* mr = malloc(sizeof(MinRank));
    if(!mr)
        return NULL;

    mr->m0 = (GFM*) m0; // same operation for both m0 = NULL or not

    if(!ms) {
        mr->ms = gfm_rand_matrices(nrow, ncol, k);
    } else {
        mr->ms = (GFM*) ms;
    }

    mr->nrow = nrow;
    mr->ncol = ncol;
    mr->nmat = k;
    mr->rank = r;

    return mr;
}

/* usage: Release a struct MinRank
 * params:
 *      1) mr: ptr to struct MinRank
 * return: void */
void
minrank_free(MinRank* mr) {
    gfm_arr_free(mr->ms, mr->nmat);
    if(mr->m0)
        gfm_free(mr->m0);

    free(mr);
}

/* usage: Copy sequentially all coefficients of a matrix Mi of a MinRank
 *      instance into a single column of M_{\lambda}
 * params:
 *      1) ml: ptr to struct GFM, storage for M_{\lambda}
 *      2) ci: index of the target column. 0 <= ci <= minrank_nmat(mr)
 *      3) mr: ptr to the MinRank instance
 * return: void */
static inline void
minrank_copy_coeffs_into_col(GFM* restrict ml, uint32_t ci, const MinRank* restrict mr) {
    for(uint32_t old_ri = 0; old_ri < minrank_nrow(mr); ++old_ri) {
        for(uint32_t old_ci = 0; old_ci < minrank_ncol(mr); ++old_ci) {
            uint32_t new_ri = old_ri * minrank_ncol(mr) + old_ci;
            gfm_set_at(ml, new_ri, ci, minrank_coeff(mr, ci, old_ri, old_ci));
        }
    }
}

/* usage: Given a struct MinRank, compute the number of rows in
 *      M_{\lambda}
 * params:
 *      1) mr: ptr to struct MinRank
 * return: number of rows in M_{\lambda} matrix */
uint32_t
minrank_sum_nrow(const MinRank* mr) {
    // each of the coeffs in a row of M_{\lambda} is a linear
    // combination of the linear variables xi's, and requires
    // 1 + minrank_nmat(mr) columns to store. There are
    // minrank_ncol(mr) coeffs in a row, which we store as
    // individually as its own row. Thus the total number of
    // rows of the matrix to store M_{\lambda} becomes
    // minrank_nrow(mr) * minrank_ncol(mr)
    return minrank_nrow(mr) * minrank_ncol(mr);
}

/* usage: Given a struct MinRank, compute the number of columns in
 *      M_{\lambda}
 * params:
 *      1) mr: ptr to struct MinRank
 * return: number of columns in M_{\lambda} matrix */
uint32_t
minrank_sum_ncol(const MinRank* mr) {
    return minrank_nmat(mr) + 1;
}

/* usage: Given a struct MinRank with matrices M0, M1, ..., Mk
 *      create scalar variables x1, x2, ..., xk, compute M_{\lambda} =
 *      M0 + x1M1 + x2M2 + ... + xkMk, and represent the sum as a matrix whose
 *      columns represent x1, ..., xk and the constant term. I.e.:
 *
 *          1  x1  x2 ...  xk
 *      | c00 c01 c02 ... c0k |
 *      | c10 c11 c12 ... c1k |
 *      |       ...           |
 *
 *      Clearly, the resultant matrix has dimension
 *      (mr_nrow(mr) * mr_ncol(mr)) x (k + 1).
 * params:
 *      1) mr: ptr to struct MinRank
 * return: ptr to struct GFM, on error NULL */
GFM*
minrank_sum(const MinRank* mr) {
    uint32_t new_nrow = minrank_sum_nrow(mr);
    uint32_t new_ncol = minrank_sum_ncol(mr);

    GFM* ml = gfm_create(new_nrow, new_ncol, NULL);
    if(!ml)
        return NULL;

    // TODO: optimize this by storing ml as column-major, if necessary

    // constant term
    if(!minrank_m0(mr))
        for(uint32_t ri = 0; ri < new_nrow; ++ri)
            gfm_set_at(ml, ri, 0, 0);
    else
        minrank_copy_coeffs_into_col(ml, 0, mr);

    // linear variables
    for(uint32_t ci = 1; ci < new_ncol; ++ci)
        minrank_copy_coeffs_into_col(ml, ci, mr);

    return ml;
}

static inline void
minrank_ks_copy_rows(GFM* restrict ks, uint32_t dst_offset, uint32_t c,
                     const gf_t* restrict src_rows, const MinRank* restrict mr) {
    mono_create_static_buf(mono_buf, 1);
    Mono* m = mono_create_from_arr(1, mono_buf);

    const uint32_t k = minrank_nmat(mr);
    for(uint32_t i = 0; i < minrank_ncol(mr); ++i) {
        gf_t* dst = (gf_t*) gfm_row_addr(ks, dst_offset + i);
        const gf_t* src = src_rows + (k + 1) * i;
        // constant term
        mono_set_deg(m, 0);
        dst[ks_midx(k, minrank_rank(mr), c, m)] = src[0];

        // ml's monomials are ordered as 1 x1 x2 ... xk while
        // ks's monimials are ordered as 1 x? ... x_{k+1} xk ... x2 x1 ...
        // so we can't copy the memory block directly
        mono_set_deg(m, 1);
        for(uint32_t j = 0; j < k; ++j) { // for x_{j+1}
            mono_set_var(m, 0, j, false);
            uint32_t dst_idx = ks_midx(k, minrank_rank(mr), c, m);
            dst[dst_idx] = src[1+j];
        }
    }
}

static inline void
minrank_ks_add_mul_rows(GFM* restrict ks, uint32_t dst_offset, uint32_t c,
                        const gf_t* restrict src_rows,
                        const MinRank* restrict mr, const uint32_t* mmap) {
    (void) c;
    for(uint32_t i = 0; i < minrank_ncol(mr); ++i) { // for each of the m coeffs
        gf_t* dst = (gf_t*) gfm_row_addr(ks, dst_offset + i);
        const gf_t* src = src_rows + (minrank_nmat(mr) + 1) * i;
        // multiply src row with vi, and save the result into dst
        for(uint32_t j = 0; j <= minrank_nmat(mr); ++j) { // for each term in src
            uint32_t new_idx = mmap[j];
            assert(new_idx < minrank_ks_ncol(mr, c));
            dst[new_idx] = gf_t_add(dst[new_idx], src[j]);
        }
    }
}

static inline void
minrank_ks_lc_rows(GFM* restrict ks, uint32_t dst_offset, const GFM* restrict ml,
                   const MinRank* restrict mr, uint32_t ri, uint32_t c,
                   uint32_t* restrict mmap) {
    // for each kernel var v_{ri, j}, multiply it with the m rows of ml starting from the (n-r+j)*m-th row
    // and add the resultant m rows to the (ri*m)-th ~ (ri + 1)*m-th rows in ks
    uint32_t lc_row_base_offset = (minrank_nrow(mr) - minrank_rank(mr)) * minrank_ncol(mr);
    for(uint32_t ci = 0; ci < minrank_rank(mr); ++ci) { // for each kernel var v_{ri, ci}
        // index of the kernel var. Mapped from its row and column indices with `ks_kernel_var_idx_from_2d`
        uint32_t vidx = ks_kernel_var_idx(ri, ci, minrank_nmat(mr), minrank_rank(mr), c);
        ks_base_cmp_idx_map_d1(mmap, minrank_nmat(mr), minrank_rank(mr), c, vidx);
        const gf_t* src_rows = gfm_row_addr(ml, lc_row_base_offset + ci * minrank_ncol(mr));
        minrank_ks_add_mul_rows(ks, dst_offset, c, src_rows, mr, mmap);
    }
}

/* usage: Given a struct MinRank, compute its Kipnis-Shamir matrix
 * params:
 *      1) mr: ptr to struct MinRank
 *      2) c: number of rows in the left multiplier.
 *          1 <= c <= minrank_nrow(mr) - minrank_rank(mr)
 * return: ptr to struct GFM, on error NULL */
GFM*
minrank_ks(const MinRank* mr, uint32_t c) {
    uint32_t new_nrow = minrank_ks_nrow(mr, c);
    uint32_t new_ncol = minrank_ks_ncol(mr, c);

    GFM* ks = gfm_create(new_nrow, new_ncol, NULL);
    if(!ks)
        return NULL;

    uint32_t* mmap = malloc(sizeof(uint32_t) * new_ncol);
    if(!mmap) {
        gfm_free(ks);
        return NULL;
    }

    const GFM* ml = minrank_sum(mr);
    if(!ml) {
        free(mmap);
        gfm_free(ks);
        return NULL;
    }

    // for each row in the left cxn matrix:
    //
    //    upper part               kernel
    //  <-of I_{n-r}-> <-zero-> <-  vars ->
    //
    //  | 1 0 0 ... 0 0 ... 0 0 v11 ... v1r | ^
    //  | 0 1 0 ... 0 0 ... 0 0 v21 ... v2r | |
    //  | 0 0 1 ... 0 0 ... 0 0 v31 ... v3r | c rows
    //  |       ... 0 0 ... 0 0     ...     | |
    //  | 0 0 0 ... 1 0 ... 0 0 vc1 ... vcr | v
    //   <---- c ---> <-n-r-c-> <--- r ---->
    gfm_zero(ks);
    uint32_t dst_row_offset = 0;
    for(uint32_t i = 0; i < c; ++i) { // for each row in I_c
        // select consecutive n rows in ml, which represent a single row
        // consisting of n coeffs in M_{\lambda}
        const gf_t* src_rows = gfm_row_addr(ml, dst_row_offset);
        // NOTE: we can't copy the rows as a block directly into the rows of ks
        // because each row in ks is longer.
        minrank_ks_copy_rows(ks, dst_row_offset, c, src_rows, mr); // deal with 1 in I_{n_r}
        minrank_ks_lc_rows(ks, dst_row_offset, ml, mr, i, c, mmap); // deal with vi1 ... vir
        dst_row_offset += minrank_ncol(mr);
    }

    free(mmap);
    gfm_free((GFM*) ml);
    return ks;
}
