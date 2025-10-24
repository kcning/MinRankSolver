#ifndef __MATRIX_GF16_H__
#define __MATRIX_GF16_H__

#include "util.h"
#include <stdbool.h>

/* header file for various struct for storing matrices used by Block Lanczos
 * algorithm.  Aliases to the actual implementations aare defined based on the
 * block size.  This header file provides a zero-overhead and unified interface
 * for all implementations; The rest of the program can invoke functions
 * independently from the actual choice of implementation. */

#if defined(__AVX512F__) || defined(__AVX2__)
#define BLK_LANCZOS_BLOCK_SIZE  (128)
#else
#define BLK_LANCZOS_BLOCK_SIZE  (64)
#endif

#if BLK_LANCZOS_BLOCK_SIZE == 512

#include "r512m_gf16.h"
#include "c512m_gf16.h"
#include "rc512m_gf16.h"
#include "grp512_gf16.h"

typedef R512MGF16 RMGF16;
typedef RC512MGF16 RCMGF16;
typedef C512MGF16 CMGF16;
typedef uint512_t DiagMGF16;
typedef Grp512GF16 RowGF16;

/* ========================================================================
 * function prototypes
 * ======================================================================== */

static force_inline gf16_t
row_gf16_at(const RowGF16* row, uint32_t i) {
    return grp512_gf16_at(row, i);
}

static force_inline void
row_gf16_set_at(RowGF16* row, uint32_t i, gf16_t v) {
    grp512_gf16_set_at(row, i, v);
}

static force_inline void
row_gf16_fmaddi_scalar(RowGF16* restrict a, const RowGF16* restrict b,
                       gf16_t c) {
    grp512_gf16_fmaddi_scalar(a, b, c);
}

static force_inline bool
diagm_gf16_is_not_full_rank(const DiagMGF16* d) {
    return !uint512_t_is_max(d);
}

static force_inline bool
diagm_gf16_nonzero(const DiagMGF16* d) {
    return uint512_t_is_not_zero(d);
}

static force_inline bool
diagm_gf16_is_zero(const DiagMGF16* d) {
    return uint512_t_is_zero(d);
}

static force_inline void
diagm_gf16_negate(DiagMGF16* restrict out, const DiagMGF16* restrict d) {
    uint512_t_neg(out, d);
}

static force_inline void
diagm_gf16_andn(DiagMGF16* restrict dst, const DiagMGF16* restrict a,
                const DiagMGF16* restrict b) {
    uint512_t_andn(dst, a, b);
}

static force_inline bool
diagm_gf16_at(const DiagMGF16* m, uint32_t i) {
    return uint512_t_at(m, i);
}

static force_inline uint32_t
diagm_gf16_nzc(const DiagMGF16* m) {
    return uint512_t_popcount(m);
}

static force_inline uint32_t
diagm_gf16_zc(const DiagMGF16* m) {
    return 512 - uint512_t_popcount(m);
}

#define cm_gf16_create(cnum) \
    c512m_gf16_create(cnum)

#define cm_gf16_free(m) \
    c512m_gf16_free(m)

#define cm_gf16_zero(m) \
    c512m_gf16_zero(m)

#define cm_gf16_at(m, i, j) \
    c512m_gf16_at(m, i, j)

static force_inline void
cm_gf16_add_at(CMGF16* m, uint32_t ri, uint32_t ci, gf16_t v) {
    c512m_gf16_add_at(m, ri, ci, v);
}

static force_inline void
cm_gf16_subset_zr_pos(const CMGF16* restrict m, const uint32_t* restrict idxs,
                      uint32_t sz, DiagMGF16* restrict pos) {
    c512m_gf16_subset_zr_pos(m, idxs, sz, pos);
}

#define rm_gf16_rnum(m) \
    r512m_gf16_rnum(m)

#define rm_gf16_raddr(m, i) \
    r512m_gf16_raddr(m, i)

#define rcm_gf16_raddr(m, i) \
    rc512m_gf16_raddr(m, i)

#define rcm_gf16_at(m, i, j) \
    rc512m_gf16_at(m, i, j)

#define rm_gf16_at(m, i, j) \
    r512m_gf16_at(m, i, j)

#define rm_gf16_create(rnum) \
    r512m_gf16_create(rnum)

#define rcm_gf16_create() \
    rc512m_gf16_create()

#define rm_gf16_free(m) \
    r512m_gf16_free(m)

#define rcm_gf16_free(m) \
    rc512m_gf16_free(m)

#define rcm_gf16_mixi(a, b, di) \
    rc512m_gf16_mixi(a, b, di)

#define rcm_gf16_mul_naive(p, m, n) \
    rc512m_gf16_mul_naive(p, m, n)

#define rm_gf16_mixi(a, b, di) \
    r512m_gf16_mixi(a, b, di)

#define rm_gf16_fms_diag(a, b, c, d) \
    r512m_gf16_fms_diag(a, b, c, d)

#define rm_gf16_fms(a, b, c) \
    r512m_gf16_fms(a, b, c)

#define rm_gf16_diag_fma(a, b, c, d) \
    r512m_gf16_diag_fma(a, b, c, d)

#define rm_gf16_rand(m) \
    r512m_gf16_rand(m)

#define rm_gf16_addi(m, n) \
    r512m_gf16_addi(m, n)

#define rm_gf16_zero(m) \
    r512m_gf16_zero(m)

#define rcm_gf16_zero(m) \
    rc512m_gf16_zero(m)

#define rm_gf16_gramian(m, p) \
    r512m_gf16_gramian(m, p)

#define rcm_gf16_copy(dst, src) \
    rc512m_gf16_copy(dst, src)

#define rcm_gf16_identity(m) \
    rc512m_gf16_identity(m)

#define rcm_gf16_gj(m, inv, di) \
    rc512m_gf16_gj(m, inv, di)

#define rcm_gf16_zero_subset_rc(m, di) \
    rc512m_gf16_zero_subset_rc(m, di)

#define rcm_gf16_is_symmetric(m) \
    rc512m_gf16_is_symmetric(m)

#define rm_gf16_zc_pos(m, out) \
    r512m_gf16_zc_pos(m, out)

#define rcm_gf16_memsize() \
    rc512m_gf16_memsize()

#define rm_gf16_memsize(rnum) \
    r512m_gf16_memsize(rnum)

#define cm_gf16_memsize(cnum) \
    c512m_gf16_memsize(cnum)

#elif BLK_LANCZOS_BLOCK_SIZE == 256

#include "r256m_gf16.h"
#include "c256m_gf16.h"
#include "rc256m_gf16.h"
#include "grp256_gf16.h"

typedef R256MGF16 RMGF16;
typedef RC256MGF16 RCMGF16;
typedef C256MGF16 CMGF16;
typedef uint256_t DiagMGF16;
typedef Grp256GF16 RowGF16;

/* ========================================================================
 * function prototypes
 * ======================================================================== */

static force_inline gf16_t
row_gf16_at(const RowGF16* row, uint32_t i) {
    return grp256_gf16_at(row, i);
}

static force_inline void
row_gf16_set_at(RowGF16* row, uint32_t i, gf16_t v) {
    grp256_gf16_set_at(row, i, v);
}

static force_inline void
row_gf16_fmaddi_scalar(RowGF16* restrict a, const RowGF16* restrict b,
                       gf16_t c) {
    grp256_gf16_fmaddi_scalar(a, b, c);
}

static force_inline bool
diagm_gf16_is_not_full_rank(const DiagMGF16* d) {
    return !uint256_t_is_max(d);
}

static force_inline bool
diagm_gf16_nonzero(const DiagMGF16* d) {
    return uint256_t_is_not_zero(d);
}

static force_inline bool
diagm_gf16_is_zero(const DiagMGF16* d) {
    return uint256_t_is_zero(d);
}

static force_inline void
diagm_gf16_negate(DiagMGF16* restrict out, const DiagMGF16* restrict d) {
    uint256_t_neg(out, d);
}

static force_inline void
diagm_gf16_andn(DiagMGF16* restrict dst, const DiagMGF16* restrict a,
                const DiagMGF16* restrict b) {
    uint256_t_andn(dst, a, b);
}

static force_inline bool
diagm_gf16_at(const DiagMGF16* m, uint32_t i) {
    return uint256_t_at(m, i);
}

static force_inline uint32_t
diagm_gf16_nzc(const DiagMGF16* m) {
    return uint256_t_popcount(m);
}

static force_inline uint32_t
diagm_gf16_zc(const DiagMGF16* m) {
    return 256 - uint256_t_popcount(m);
}

#define cm_gf16_create(cnum) \
    c256m_gf16_create(cnum)

#define cm_gf16_free(m) \
    c256m_gf16_free(m)

#define cm_gf16_zero(m) \
    c256m_gf16_zero(m)

#define cm_gf16_at(m, i, j) \
    c256m_gf16_at(m, i, j)

static force_inline void
cm_gf16_add_at(CMGF16* m, uint32_t ri, uint32_t ci, gf16_t v) {
    c256m_gf16_add_at(m, ri, ci, v);
}

static force_inline void
cm_gf16_subset_zr_pos(const CMGF16* restrict m, const uint32_t* restrict idxs,
                      uint32_t sz, DiagMGF16* restrict pos) {
    c256m_gf16_subset_zr_pos(m, idxs, sz, pos);
}

#define rm_gf16_rnum(m) \
    r256m_gf16_rnum(m)

#define rm_gf16_raddr(m, i) \
    r256m_gf16_raddr(m, i)

#define rcm_gf16_raddr(m, i) \
    rc256m_gf16_raddr(m, i)

#define rcm_gf16_at(m, i, j) \
    rc256m_gf16_at(m, i, j)

#define rm_gf16_at(m, i, j) \
    r256m_gf16_at(m, i, j)

#define rm_gf16_create(rnum) \
    r256m_gf16_create(rnum)

#define rcm_gf16_create() \
    rc256m_gf16_create()

#define rm_gf16_free(m) \
    r256m_gf16_free(m)

#define rcm_gf16_free(m) \
    rc256m_gf16_free(m)

#define rcm_gf16_mixi(a, b, di) \
    rc256m_gf16_mixi(a, b, di)

#define rcm_gf16_mul_naive(p, m, n) \
    rc256m_gf16_mul_naive(p, m, n)

#define rm_gf16_mixi(a, b, di) \
    r256m_gf16_mixi(a, b, di)

#define rm_gf16_fms_diag(a, b, c, d) \
    r256m_gf16_fms_diag(a, b, c, d)

#define rm_gf16_fms(a, b, c) \
    r256m_gf16_fms(a, b, c)

#define rm_gf16_diag_fma(a, b, c, d) \
    r256m_gf16_diag_fma(a, b, c, d)

#define rm_gf16_rand(m) \
    r256m_gf16_rand(m)

#define rm_gf16_addi(m, n) \
    r256m_gf16_addi(m, n)

#define rm_gf16_zero(m) \
    r256m_gf16_zero(m)

#define rcm_gf16_zero(m) \
    rc256m_gf16_zero(m)

#define rm_gf16_gramian(m, p) \
    r256m_gf16_gramian(m, p)

#define rcm_gf16_copy(dst, src) \
    rc256m_gf16_copy(dst, src)

#define rcm_gf16_identity(m) \
    rc256m_gf16_identity(m)

#define rcm_gf16_gj(m, inv, di) \
    rc256m_gf16_gj(m, inv, di)

#define rcm_gf16_zero_subset_rc(m, di) \
    rc256m_gf16_zero_subset_rc(m, di)

#define rcm_gf16_is_symmetric(m) \
    rc256m_gf16_is_symmetric(m)

#define rm_gf16_zc_pos(m, out) \
    r256m_gf16_zc_pos(m, out)

#define rm_gf16_nzc_pos(m, out) \
    r256m_gf16_nzc_pos(m, out)

#define rcm_gf16_memsize() \
    rc256m_gf16_memsize()

#define rm_gf16_memsize(rnum) \
    r256m_gf16_memsize(rnum)

#define cm_gf16_memsize(cnum) \
    c256m_gf16_memsize(cnum)

#elif BLK_LANCZOS_BLOCK_SIZE == 128

#include "r128m_gf16.h"
#include "r128m_gf16_parallel.h"
#include "c128m_gf16.h"
#include "rc128m_gf16.h"
#include "grp128_gf16.h"

typedef R128MGF16 RMGF16;
typedef RC128MGF16 RCMGF16;
typedef C128MGF16 CMGF16;
typedef uint128_t DiagMGF16;
typedef Grp128GF16 RowGF16;
typedef R128MGF16PArg RMGF16PArg;

/* ========================================================================
 * function prototypes
 * ======================================================================== */

static force_inline gf16_t
row_gf16_at(const RowGF16* row, uint32_t i) {
    return grp128_gf16_at(row, i);
}

static force_inline void
row_gf16_set_at(RowGF16* row, uint32_t i, gf16_t v) {
    grp128_gf16_set_at(row, i, v);
}

static force_inline void
row_gf16_fmaddi_scalar(RowGF16* restrict a, const RowGF16* restrict b,
                       gf16_t c) {
    grp128_gf16_fmaddi_scalar(a, b, c);
}

static force_inline bool
diagm_gf16_is_not_full_rank(const DiagMGF16* d) {
    return !uint128_t_is_max(d);
}

static force_inline bool
diagm_gf16_nonzero(const DiagMGF16* d) {
    return uint128_t_is_not_zero(d);
}

static force_inline bool
diagm_gf16_is_zero(const DiagMGF16* d) {
    return uint128_t_is_zero(d);
}

static force_inline void
diagm_gf16_negate(DiagMGF16* restrict out, const DiagMGF16* restrict d) {
    uint128_t_neg(out, d);
}

static force_inline void
diagm_gf16_andn(DiagMGF16* restrict dst, const DiagMGF16* restrict a,
                const DiagMGF16* restrict b) {
    uint128_t_andn(dst, a, b);
}

static force_inline bool
diagm_gf16_at(const DiagMGF16* m, uint32_t i) {
    return uint128_t_at(m, i);
}

static force_inline uint32_t
diagm_gf16_nzc(const DiagMGF16* m) {
    return uint128_t_popcount(m);
}

static force_inline uint32_t
diagm_gf16_zc(const DiagMGF16* m) {
    return 128 - uint128_t_popcount(m);
}

#define cm_gf16_create(cnum) \
    c128m_gf16_create(cnum)

#define cm_gf16_free(m) \
    c128m_gf16_free(m)

#define cm_gf16_zero(m) \
    c128m_gf16_zero(m)

#define cm_gf16_at(m, i, j) \
    c128m_gf16_at(m, i, j)

static force_inline void
cm_gf16_add_at(CMGF16* m, uint32_t ri, uint32_t ci, gf16_t v) {
    c128m_gf16_add_at(m, ri, ci, v);
}

static force_inline void
cm_gf16_subset_zr_pos(const CMGF16* restrict m, const uint32_t* restrict idxs,
                      uint32_t sz, DiagMGF16* restrict pos) {
    c128m_gf16_subset_zr_pos(m, idxs, sz, pos);
}

#define rm_gf16_rnum(m) \
    r128m_gf16_rnum(m)

#define rm_gf16_raddr(m, i) \
    r128m_gf16_raddr(m, i)

#define rcm_gf16_raddr(m, i) \
    rc128m_gf16_raddr(m, i)

#define rcm_gf16_at(m, i, j) \
    rc128m_gf16_at(m, i, j)

#define rm_gf16_at(m, i, j) \
    r128m_gf16_at(m, i, j)

#define rm_gf16_create(rnum) \
    r128m_gf16_create(rnum)

#define rcm_gf16_create() \
    rc128m_gf16_create()

#define rcm_gf16_arr_create(sz) \
    rc128m_gf16_arr_create(sz)

#define rm_gf16_free(m) \
    r128m_gf16_free(m)

#define rcm_gf16_free(m) \
    rc128m_gf16_free(m)

#define rcm_gf16_arr_free(m) \
    rc128m_gf16_arr_free(m)

#define rcm_gf16_mixi(a, b, di) \
    rc128m_gf16_mixi(a, b, di)

#define rcm_gf16_mul_naive(p, m, n) \
    rc128m_gf16_mul_naive(p, m, n)

#define rm_gf16_mixi(a, b, di) \
    r128m_gf16_mixi(a, b, di)

#define rm_gf16_mixi_parallel(a, b, di, tn, args, tp) \
    r128m_gf16_mixi_parallel(a, b, di, tn, args, tp)

#define rm_gf16_fms_diag(a, b, c, d) \
    r128m_gf16_fms_diag(a, b, c, d)

#define rm_gf16_fms_diag_parallel(a, b, c, d, tn, args, tp) \
    r128m_gf16_fms_diag_parallel(a, b, c, d, tn, args, tp)

#define rm_gf16_fms(a, b, c) \
    r128m_gf16_fms(a, b, c)

#define rm_gf16_fms_parallel(a, b, c, tn, args, tp) \
    r128m_gf16_fms_parallel(a, b, c, tn, args, tp)

#define rm_gf16_diag_fma(a, b, c, d) \
    r128m_gf16_diag_fma(a, b, c, d)

#define rm_gf16_diag_fma_parallel(a, b, c, d, tn, args, tp) \
    r128m_gf16_diag_fma_parallel(a, b, c, d, tn, args, tp)

#define rm_gf16_rand(m) \
    r128m_gf16_rand(m)

#define rm_gf16_addi(m, n) \
    r128m_gf16_addi(m, n)

#define rm_gf16_zero(m) \
    r128m_gf16_zero(m)

#define rcm_gf16_zero(m) \
    rc128m_gf16_zero(m)

#define rm_gf16_gramian(m, p) \
    r128m_gf16_gramian(m, p)

#define rm_gf16_gramian_parallel(m, p, tn, buf, args, tp) \
    r128m_gf16_gramian_parallel(m, p, tn, buf, args, tp)

#define rcm_gf16_copy(dst, src) \
    rc128m_gf16_copy(dst, src)

#define rcm_gf16_identity(m) \
    rc128m_gf16_identity(m)

#define rcm_gf16_gj(m, inv, di) \
    rc128m_gf16_gj(m, inv, di)

#define rcm_gf16_zero_subset_rc(m, di) \
    rc128m_gf16_zero_subset_rc(m, di)

#define rcm_gf16_is_symmetric(m) \
    rc128m_gf16_is_symmetric(m)

#define rm_gf16_zc_pos(m, out) \
    r128m_gf16_zc_pos(m, out)

#define rm_gf16_nzc_pos(m, out) \
    r128m_gf16_nzc_pos(m, out)

#define rcm_gf16_memsize() \
    rc128m_gf16_memsize()

#define rm_gf16_memsize(rnum) \
    r128m_gf16_memsize(rnum)

#define cm_gf16_memsize(cnum) \
    c128m_gf16_memsize(cnum)

#elif BLK_LANCZOS_BLOCK_SIZE == 64

#include "r64m_gf16.h"
#include "r64m_gf16_parallel.h"
#include "c64m_gf16.h"
#include "rc64m_gf16.h"
#include "grp64_gf16.h"

typedef R64MGF16 RMGF16;
typedef RC64MGF16 RCMGF16;
typedef C64MGF16 CMGF16;
typedef uint64_t DiagMGF16;
typedef Grp64GF16 RowGF16;
typedef R64MGF16PArg RMGF16PArg;

/* ========================================================================
 * function prototypes
 * ======================================================================== */

static force_inline gf16_t
row_gf16_at(const RowGF16* row, uint32_t i) {
    return grp64_gf16_at(row, i);
}

static force_inline void
row_gf16_set_at(RowGF16* row, uint32_t i, gf16_t v) {
    grp64_gf16_set_at(row, i, v);
}

static force_inline void
row_gf16_fmaddi_scalar(RowGF16* restrict a, const RowGF16* restrict b,
                       gf16_t c) {
    grp64_gf16_fmaddi_scalar(a, b, c);
}

static force_inline bool
diagm_gf16_is_not_full_rank(const DiagMGF16* d) {
    return ~(*d);
}

static force_inline bool
diagm_gf16_nonzero(const DiagMGF16* d) {
    return *d;
}

static force_inline bool
diagm_gf16_is_zero(const DiagMGF16* d) {
    return !(*d);
}

static force_inline void
diagm_gf16_negate(DiagMGF16* restrict r, const DiagMGF16* restrict d) {
    *r = ~(*d);
}

static force_inline void
diagm_gf16_andn(DiagMGF16* restrict dst, const DiagMGF16* restrict a,
                const DiagMGF16* restrict b) {
    *dst = (*a) & ~(*b);
}

static force_inline bool
diagm_gf16_at(const DiagMGF16* m, uint32_t i) {
    return (*m) & (0x1 << i);
}

static force_inline uint32_t
diagm_gf16_nzc(const DiagMGF16* m) {
    return uint64_popcount(*m);
}

static force_inline uint32_t
diagm_gf16_zc(const DiagMGF16* m) {
    return uint64_popcount(~(*m));
}

#define cm_gf16_create(cnum) \
    c64m_gf16_create(cnum)

#define cm_gf16_free(m) \
    c64m_gf16_free(m)

#define cm_gf16_zero(m) \
    c64m_gf16_zero(m)

#define cm_gf16_at(m, i, j) \
    c64m_gf16_at(m, i, j)

static force_inline void
cm_gf16_add_at(CMGF16* m, uint32_t ri, uint32_t ci, gf16_t v) {
    c64m_gf16_add_at(m, ri, ci, v);
}

static force_inline void
cm_gf16_subset_zr_pos(const CMGF16* restrict m, const uint32_t* restrict idxs,
                      uint32_t sz, DiagMGF16* restrict pos) {
    *pos = c64m_gf16_subset_zr_pos(m, idxs, sz);
}

#define rm_gf16_rnum(m) \
    r64m_gf16_rnum(m)

#define rm_gf16_raddr(m, i) \
    r64m_gf16_raddr(m, i)

#define rcm_gf16_raddr(m, i) \
    rc64m_gf16_raddr(m, i)

#define rcm_gf16_at(m, i, j) \
    rc64m_gf16_at(m, i, j)

#define rm_gf16_at(m, i, j) \
    r64m_gf16_at(m, i, j)

#define rm_gf16_create(rnum) \
    r64m_gf16_create(rnum)

#define rcm_gf16_create() \
    rc64m_gf16_create()

#define rcm_gf16_arr_create(sz) \
    rc64m_gf16_arr_create(sz)

#define rm_gf16_free(m) \
    r64m_gf16_free(m)

#define rcm_gf16_free(m) \
    rc64m_gf16_free(m)

#define rcm_gf16_arr_free(m) \
    rc64m_gf16_arr_free(m)

#define rcm_gf16_mixi(a, b, di) \
    rc64m_gf16_mixi(a, b, *(di))

#define rcm_gf16_mul_naive(p, m, n) \
    rc64m_gf16_mul_naive(p, m, n)

#define rm_gf16_mixi(a, b, di) \
    r64m_gf16_mixi(a, b, *(di))

#define rm_gf16_mixi_parallel(a, b, di, tn, args, tp) \
    r64m_gf16_mixi(a, b, *(di))

#define rm_gf16_fms_diag(a, b, c, d) \
    r64m_gf16_fms_diag(a, b, c, *(d))

#define rm_gf16_fms_diag_parallel(a, b, c, d, tn, args, tp) \
    r64m_gf16_fms_diag(a, b, c, *(d))

#define rm_gf16_fms(a, b, c) \
    r64m_gf16_fms(a, b, c)

#define rm_gf16_fms_parallel(a, b, c, tn, args, tp) \
    r64m_gf16_fms(a, b, c)

#define rm_gf16_diag_fma(a, b, c, d) \
    r64m_gf16_diag_fma(a, b, c, *(d))

#define rm_gf16_diag_fma_parallel(a, b, c, d, tn, args, tp) \
    r64m_gf16_diag_fma(a, b, c, *(d))

#define rm_gf16_rand(m) \
    r64m_gf16_rand(m)

#define rm_gf16_addi(m, n) \
    r64m_gf16_addi(m, n)

#define rm_gf16_zero(m) \
    r64m_gf16_zero(m)

#define rcm_gf16_zero(m) \
    rc64m_gf16_zero(m)

#define rm_gf16_gramian(m, p) \
    r64m_gf16_gramian(m, p)

#define rm_gf16_gramian_parallel(m, p, tn, gp, args, tp) \
    r64m_gf16_gramian(m, p)

#define rcm_gf16_copy(dst, src) \
    rc64m_gf16_copy(dst, src)

#define rcm_gf16_identity(m) \
    rc64m_gf16_identity(m)

#define rcm_gf16_gj(m, inv, di) \
    rc64m_gf16_gj(m, inv, di)

#define rcm_gf16_zero_subset_rc(m, di) \
    rc64m_gf16_zero_subset_rc(m, *di)

#define rcm_gf16_is_symmetric(m) \
    rc64m_gf16_is_symmetric(m)

#define rm_gf16_zc_pos(m, out) do { \
    *(out) = r64m_gf16_zc_pos(m); \
} while(0)

#define rm_gf16_nzc_pos(m, out) do { \
    *(out) = r64m_gf16_nzc_pos(m); \
} while(0)

#define rcm_gf16_memsize() \
    rc64m_gf16_memsize()

#define rm_gf16_memsize(rnum) \
    r64m_gf16_memsize(rnum)

#define cm_gf16_memsize(cnum) \
    c64m_gf16_memsize(cnum)

#else // BLK_LANCZOS_BLOCK_SIZE == 64

#error "Unsupported block size for Lanczos"

#endif

#endif // __MATRIX_GF16_H__
