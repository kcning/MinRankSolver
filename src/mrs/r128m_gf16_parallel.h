#ifndef __R128M_GF16_PARALLEL_H__
#define __R128M_GF16_PARALLEL_H__

#include "r128m_gf16.h"
#include "rc128m_gf16.h"
#include "thpool.h"
#include <stdint.h>

typedef struct {
    R128MGF16* restrict a;
    const R128MGF16* restrict b;
    RC128MGF16* restrict c;
    RC128MGF16* restrict buf;
    const uint128_t* restrict d;
    uint64_t sidx;
    uint64_t eidx;
    void* restrict ptr; // a generic ptr
} R128MGF16PArg;

/* ========================================================================
 * function prototypes
 * ======================================================================== */

/* usage: Given a R128MGF16 matrix m and a container, compute in parallel the
 *      Gramian matrix of m, i.e. m.transpose() * m, and store it into the
 *      container.  Note that the product has dimension 128x128.
 * params:
 *      1) m: ptr to a struct R128MGF16
 *      2) p: ptr to a struct RC128MGF16, container for the result
 *      3) tnum: number of threads to use
 *      4) buf: an array of RC128MGF16 of size tnum used to hold partial
 *          computation. Will be overwritten.
 *      5) args: ptr to an array of struct R128MGF16PArg. Must
 *          have size at least as large as the number of threads to use
 *      6) tp: ptr to a struct Threadpool
 * return: void */
void
r128m_gf16_gramian_parallel(const R128MGF16* restrict m, RC128MGF16* restrict p,
                            uint32_t tnum, RC128MGF16* restrict buf,
                            R128MGF16PArg* restrict args,
                            Threadpool* restrict tp);

/* usage: Given 2 R128MGF16 A and B, and a RC128MGF16 C, compute
 *      A + B * C and store the result back into A in parallel
 * params:
 *      1) a: ptr to struct R128MGF16, storing the matrix A
 *      2) b: ptr to struct R128MGF16, storing the matrix B
 *      3) c: ptr to struct RC128MGF16, storing the matrix C
 *      4) tnum: number of threads to use
 *      5) args: ptr to an array of struct R128MGF16PArg. Must
 *          have size at least as large as the number of threads to use
 *      6) tp: ptr to a struct Threadpool
 * return: void */
void
r128m_gf16_fma_parallel(R128MGF16* restrict a, const R128MGF16* restrict b,
                        const RC128MGF16* restrict c, uint32_t tnum,
                        R128MGF16PArg* restrict args, Threadpool* restrict tp);

/* usage: Given 2 R128MGF16 A and B, and a RC128MGF16 C, compute
 *      A - B * C and store the result back into A in parallel
 * params:
 *      1) a: ptr to struct R128MGF16, storing the matrix A
 *      2) b: ptr to struct R128MGF16, storing the matrix B
 *      3) c: ptr to struct RC128MGF16, storing the matrix C
 *      4) tnum: number of threads to use
 *      5) args: ptr to an array of struct R128MGF16PArg. Must
 *          have size at least as large as the number of threads to use
 *      6) tp: ptr to a struct Threadpool
 * return: void */
void
r128m_gf16_fms_parallel(R128MGF16* restrict a, const R128MGF16* restrict b,
                        const RC128MGF16* restrict c, uint32_t tnum,
                        R128MGF16PArg* restrict args, Threadpool* restrict tp);

/* usage: Given 2 R128MGF16 A and B, a RC128MGF16 C, and a 128x128 diagonal
 *      matrix D with coefficients either 1 and 0, compute A * D + B * C
 *      and store the result back into A in parallel
 * params:
 *      1) a: ptr to struct R128MGF16, storing the matrix A
 *      2) b: ptr to struct R128MGF16, storing the matrix B
 *      3) c: ptr to struct RC128MGF16, storing the matrix C
 *      4) d: ptr to a uint128_t which encodes the diagonal matrix D. If the
 *          LSB is 1, then entry (0, 0) of D is 1. Otherwise 0.
 *      5) tnum: number of threads to use
 *      6) args: ptr to an array of struct R128MGF16PArg. Must
 *          have size at least as large as the number of threads to use
 *      7) tp: ptr to a struct Threadpool
 * return: void */
void
r128m_gf16_diag_fma_parallel(R128MGF16* restrict a, const R128MGF16* restrict b,
                             const RC128MGF16* restrict c,
                             const uint128_t* restrict d, uint32_t tnum,
                             R128MGF16PArg* restrict args,
                             Threadpool* restrict tp);

/* usage: Given 2 R128MGF16 A and B, a RC128MGF16 C, and a 128x128 diagonal
 *      matrix D with coefficients either 1 and 0, compute A + B * C * D
 *      and store the result back into A in parallel
 * params:
 *      1) a: ptr to struct R128MGF16, storing the matrix A
 *      2) b: ptr to struct R128MGF16, storing the matrix B
 *      3) c: ptr to struct RC128MGF16, storing the matrix C
 *      4) d: ptr to a uint128_t which encodes the diagonal matrix D. If the
 *          LSB is 1, then entry (0, 0) of D is 1. Otherwise 0.
 *      5) tnum: number of threads to use
 *      6) args: ptr to an array of struct R128MGF16PArg. Must
 *          have size at least as large as the number of threads to use
 *      7) tp: ptr to a struct Threadpool
 * return: void */
void
r128m_gf16_fma_diag_parallel(R128MGF16* restrict a, const R128MGF16* restrict b,
                             const RC128MGF16* restrict c,
                             const uint128_t* restrict d, uint32_t tnum,
                             R128MGF16PArg* restrict args,
                             Threadpool* restrict tp);

/* usage: Given 2 R128MGF16 A and B, a RC128MGF16 C, and a 128x128 diagonal
 *      matrix D with coefficients either 1 and 0, compute A - B * C * D
 *      and store the result back into A in parallel
 * params:
 *      1) a: ptr to struct R128MGF16, storing the matrix A
 *      2) b: ptr to struct R128MGF16, storing the matrix B
 *      3) c: ptr to struct RC128MGF16, storing the matrix C
 *      4) d: ptr to a uint128_t which encodes the diagonal matrix D. If the
 *          LSB is 1, then entry (0, 0) of D is 1. Otherwise 0.
 *      5) tnum: number of threads to use
 *      6) args: ptr to an array of struct R128MGF16PArg. Must
 *          have size at least as large as the number of threads to use
 *      7) tp: ptr to a struct Threadpool
 * return: void */
void
r128m_gf16_fms_diag_parallel(R128MGF16* restrict a, const R128MGF16* restrict b,
                             const RC128MGF16* restrict c,
                             const uint128_t* restrict d, uint32_t tnum,
                             R128MGF16PArg* restrict args,
                             Threadpool* restrict tp);

/* usage: Given 2 R128MGF16 A and B, replace a subset of columns of A with
 *      corresponding columns of B in parallel
 * params:
 *      1) a: ptr to struct R128MGF16, storing the matrix A
 *      2) b: ptr to struct R128MGF16, storing the matrix B
 *      3) di: ptr to a uint128_t which encodes the diagonal matrix D. If the
 *          LSB is 1, then entry (0, 0) of D is 1. Otherwise 0.
 *      4) tnum: number of threads to use
 *      5) args: ptr to an array of struct R128MGF16PArg. Must
 *          have size at least as large as the number of threads to use
 *      6) tp: ptr to a struct Threadpool
 * return: void */
void
r128m_gf16_mixi_parallel(R128MGF16* restrict a, const R128MGF16* restrict b,
                         const uint128_t* restrict di, uint32_t tnum,
                         R128MGF16PArg* restrict args,
                         Threadpool* restrict tp);


#endif // __R128M_GF16_PARALLEL_H__
