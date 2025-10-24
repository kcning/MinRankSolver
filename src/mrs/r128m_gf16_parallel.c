#include "r128m_gf16_parallel.h"
#include "rc128m_gf16.h"
#include "thpool.h"
#include "util.h"
#include <pthread.h>

/* ========================================================================
 * function implementations
 * ======================================================================== */

#if defined(__AVX512F__)

void
r128m_gf16_gramian_worker_avx512(void* __arg) {
    const R128MGF16PArg* arg = (R128MGF16PArg*) __arg;
    uint32_t i = arg->sidx;
    const Grp128GF16* m_row = r128m_gf16_raddr((R128MGF16*) arg->a, i);
    const __m512i v_1st = _mm512_load_si512(m_row);
    Grp128GF16* dst = rc128m_gf16_raddr(arg->buf, 0);
    for(uint32_t j = 0; j < 128; j += 2, dst += 2) {
        __m512i p0 = grp128_gf16_mul_scalar_bs_avx512(v_1st, m_row, j);
        __m512i p1 = grp128_gf16_mul_scalar_bs_avx512(v_1st, m_row, j + 1);
        _mm512_store_si512(dst, p0);
        _mm512_store_si512(dst + 1, p1);
    }
    ++i; ++m_row;

    for(; i < arg->eidx; ++i, ++m_row) {
        Grp128GF16* dst = rc128m_gf16_raddr(arg->buf, 0);
        __m512i v = _mm512_load_si512(m_row);
        for(uint32_t j = 0; j < 128; j += 2, dst += 2) {
            __m512i p0 = grp128_gf16_mul_scalar_bs_avx512(v, m_row, j);
            __m512i p1 = grp128_gf16_mul_scalar_bs_avx512(v, m_row, j + 1);
            __m512i d0 = _mm512_load_si512(dst);
            __m512i d1 = _mm512_load_si512(dst + 1);
            _mm512_store_si512(dst, _mm512_xor_si512(d0, p0));
            _mm512_store_si512(dst + 1, _mm512_xor_si512(d1, p1));
        }
    }
}

#elif defined(__AVX2__)

void
r128m_gf16_gramian_worker_avx2(void* __arg) {
    const R128MGF16PArg* arg = (R128MGF16PArg*) __arg;
    uint32_t i = arg->sidx;
    const Grp128GF16* m_row = r128m_gf16_raddr((R128MGF16*) arg->a, i);
    __m256i* maddr = (__m256i*) m_row->b;
    const __m256i v0_1st = _mm256_load_si256(maddr);
    const __m256i v1_1st = _mm256_load_si256(maddr + 1);
    __m256i* dst = (__m256i*) rc128m_gf16_raddr(arg->buf, 0);
    for(uint32_t j = 0; j < 128; j += 2, dst += 4) {
        __m256i p0, p1, p2, p3;
        p0 = grp128_gf16_mul_scalar_bs_avx2(&p1, v0_1st, v1_1st, m_row, j);
        p2 = grp128_gf16_mul_scalar_bs_avx2(&p3, v0_1st, v1_1st, m_row, j + 1);
        _mm256_store_si256(dst, p0);
        _mm256_store_si256(dst + 1, p1);
        _mm256_store_si256(dst + 2, p2);
        _mm256_store_si256(dst + 3, p3);
    }
    ++i; ++m_row;

    for(; i < arg->eidx; ++i, ++m_row) {
        maddr = (__m256i*) m_row->b;
        const __m256i v0 = _mm256_load_si256(maddr);
        const __m256i v1 = _mm256_load_si256(maddr + 1);
        dst = (__m256i*) rc128m_gf16_raddr(arg->buf, 0);
        for(uint32_t j = 0; j < 128; j += 2, dst += 4) {
            __m256i p0, p1, p2, p3;
            p0 = grp128_gf16_mul_scalar_bs_avx2(&p1, v0, v1, m_row, j);
            p2 = grp128_gf16_mul_scalar_bs_avx2(&p3, v0, v1, m_row, j + 1);
            __m256i d0 = _mm256_load_si256(dst);
            __m256i d1 = _mm256_load_si256(dst + 1);
            __m256i d2 = _mm256_load_si256(dst + 2);
            __m256i d3 = _mm256_load_si256(dst + 3);
            _mm256_store_si256(dst, _mm256_xor_si256(d0, p0));
            _mm256_store_si256(dst + 1, _mm256_xor_si256(d1, p1));
            _mm256_store_si256(dst + 2, _mm256_xor_si256(d2, p2));
            _mm256_store_si256(dst + 3, _mm256_xor_si256(d3, p3));
        }
    }
}

#else

void
r128m_gf16_gramian_worker_naive(void* __arg) {
    const R128MGF16PArg* arg = (R128MGF16PArg*) __arg;
    rc128m_gf16_zero(arg->buf);
    uint32_t i = arg->sidx;
    const Grp128GF16* m_row = r128m_gf16_raddr((R128MGF16*) arg->a, i);
    for(; i < arg->eidx; ++i, ++m_row) {
        Grp128GF16* dst = rc128m_gf16_raddr(arg->buf, 0);
        for(uint32_t j = 0; j < 128; j += 2, dst += 2) {
            grp128_gf16_fmaddi_scalar_bs(dst, m_row, m_row, j);
            grp128_gf16_fmaddi_scalar_bs(dst + 1, m_row, m_row, j + 1);
        }
    }
}

#endif

static void
r128m_gf16_gramian_worker(void* __arg) {
#if defined(__AVX512F__)
    r128m_gf16_gramian_worker_avx512(__arg);
#elif defined(__AVX2__)
    r128m_gf16_gramian_worker_avx2(__arg);
#else
    r128m_gf16_gramian_worker_naive(__arg);
#endif
    const R128MGF16PArg* arg = (R128MGF16PArg*) __arg;
    pthread_mutex_t* lock = arg->ptr;
    pthread_mutex_lock(lock);
    rc128m_gf16_addi(arg->c, arg->buf);
    pthread_mutex_unlock(lock);
}

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
                            Threadpool* restrict tp) {
    pthread_mutex_t lock;
    pthread_mutexattr_t lock_attr;
    // TODO: check the return values
    pthread_mutexattr_init(&lock_attr);
    pthread_mutexattr_setrobust(&lock_attr, PTHREAD_MUTEX_ROBUST);
    pthread_mutex_init(&lock, &lock_attr);
    pthread_mutexattr_destroy(&lock_attr);

    rc128m_gf16_zero(p);
    uint32_t strip_sz = r128m_gf16_rnum(m) / tnum;
    uint32_t sidx = 0;
    for(uint32_t i = 0; i < (tnum - 1); ++i) {
        args[i].a = (R128MGF16*) m;
        args[i].c = p;
        args[i].ptr = &lock;
        args[i].buf = rc128m_gf16_arr_at(buf, i);
        args[i].sidx = sidx;
        sidx += strip_sz;
        args[i].eidx = sidx;
    }
    args[tnum-1].a = (R128MGF16*) m;
    args[tnum-1].c = p;
    args[tnum-1].ptr = &lock;
    args[tnum-1].buf = rc128m_gf16_arr_at(buf, tnum-1);
    args[tnum-1].sidx = sidx;
    args[tnum-1].eidx = r128m_gf16_rnum(m);

    for(uint32_t i = 0; i < tnum; ++i) {
        thpool_add_job(tp, r128m_gf16_gramian_worker, args + i);
    }
    thpool_wait_jobs(tp);
    pthread_mutex_destroy(&lock);
}

#if defined(__AVX512F__)

static force_inline void
r128m_gf16_fma_worker_avx512(void* __arg) {
    const R128MGF16PArg* arg = (R128MGF16PArg*) __arg;
    Grp128GF16* dst = r128m_gf16_raddr(arg->a, arg->sidx);
    const Grp128GF16* b_row = r128m_gf16_raddr((R128MGF16*)(arg->b), arg->sidx);
    for(uint32_t i = arg->sidx; i < arg->eidx; ++i, ++dst, ++b_row) {
        const Grp128GF16* src = rc128m_gf16_raddr((RC128MGF16*) arg->c, 0);
        __m512i prod = _mm512_load_si512(dst);
        for(uint32_t j = 0; j < 128; j += 2, src += 2) {
            __m512i v0 = _mm512_load_si512(src);
            __m512i v1 = _mm512_load_si512(src + 1);
            __m512i p0 = grp128_gf16_mul_scalar_bs_avx512(v0, b_row, j);
            __m512i p1 = grp128_gf16_mul_scalar_bs_avx512(v1, b_row, j + 1);
            prod = _mm512_xor_si512(prod, p0);
            prod = _mm512_xor_si512(prod, p1);
        }
        _mm512_store_si512(dst, prod);
    }
}

#elif defined(__AVX2__)

static force_inline void
r128m_gf16_fma_worker_avx2(void* __arg) {
    const R128MGF16PArg* arg = (R128MGF16PArg*) __arg;
    __m256i* dst = (__m256i*) r128m_gf16_raddr(arg->a, arg->sidx);
    const Grp128GF16* b_row = r128m_gf16_raddr((R128MGF16*) arg->b, arg->sidx);
    for(uint32_t i = arg->sidx; i < arg->eidx; ++i, ++b_row, dst += 2) {
        const __m256i* src =(__m256i*)rc128m_gf16_raddr((RC128MGF16*)arg->c, 0);
        __m256i prod0 = _mm256_load_si256(dst);
        __m256i prod1 = _mm256_load_si256(dst + 1);
        for(uint32_t j = 0; j < 128; j += 2, src += 4) {
            __m256i v0 = _mm256_load_si256(src);
            __m256i v1 = _mm256_load_si256(src + 1);
            __m256i v2 = _mm256_load_si256(src + 2);
            __m256i v3 = _mm256_load_si256(src + 3);
            __m256i p0, p1, p2, p3;
            p0 = grp128_gf16_mul_scalar_bs_avx2(&p1, v0, v1, b_row, j);
            p2 = grp128_gf16_mul_scalar_bs_avx2(&p3, v2, v3, b_row, j + 1);
            prod0 = _mm256_xor_si256(prod0, p0);
            prod1 = _mm256_xor_si256(prod1, p1);
            prod0 = _mm256_xor_si256(prod0, p2);
            prod1 = _mm256_xor_si256(prod1, p3);
        }
        _mm256_store_si256(dst, prod0);
        _mm256_store_si256(dst + 1, prod1);
    }
}

#else

static force_inline void
r128m_gf16_fma_worker_naive(void* __arg) {
    const R128MGF16PArg* arg = (R128MGF16PArg*) __arg;
    const Grp128GF16* b_row = r128m_gf16_raddr((R128MGF16*) arg->b, arg->sidx);
    Grp128GF16* dst = r128m_gf16_raddr(arg->a, arg->sidx);
    for(uint32_t i = arg->sidx; i < arg->eidx; ++i, ++b_row, ++dst) {
        const Grp128GF16* src = rc128m_gf16_raddr((RC128MGF16*)arg->c, 0);
        for(uint32_t j = 0; j < 128; j += 2, src += 2) {
            grp128_gf16_fmaddi_scalar_bs(dst, src, b_row, j);
            grp128_gf16_fmaddi_scalar_bs(dst, src + 1, b_row, j + 1);
        }
    }
}

#endif

static void
r128m_gf16_fma_worker(void* __arg) {
#if defined(__AVX512F__)
    r128m_gf16_fma_worker_avx512(__arg);
#elif defined(__AVX2__)
    r128m_gf16_fma_worker_avx2(__arg);
#else
    r128m_gf16_fma_worker_naive(__arg);
#endif
}

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
                        R128MGF16PArg* restrict args, Threadpool* restrict tp) {
    uint32_t strip_sz = r128m_gf16_rnum(a) / tnum;
    uint32_t sidx = 0;
    for(uint32_t i = 0; i < (tnum - 1); ++i) {
        args[i].a = a;
        args[i].b = b;
        args[i].c = (RC128MGF16*) c;
        args[i].sidx = sidx;
        sidx += strip_sz;
        args[i].eidx = sidx;
    }
    args[tnum-1].a = a;
    args[tnum-1].b = b;
    args[tnum-1].c = (RC128MGF16*) c;
    args[tnum-1].sidx = sidx;
    args[tnum-1].eidx = r128m_gf16_rnum(a);

    for(uint32_t i = 0; i < tnum; ++i) {
        thpool_add_job(tp, r128m_gf16_fma_worker, args + i);
    }
    thpool_wait_jobs(tp);
}

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
                        R128MGF16PArg* restrict args, Threadpool* restrict tp) {
    r128m_gf16_fma_parallel(a, b, c, tnum, args, tp);
}

#if defined(__AVX512F__)

void
r128m_gf16_diag_fma_worker_avx512(void* __arg) {
    R128MGF16PArg* arg = (R128MGF16PArg*) __arg;
    __m512i vd = _mm512_castsi128_si512(_mm_load_si128((__m128i*)arg->d));
    vd = _mm512_shuffle_i64x2(vd, vd, 0x0); // [mask, mask, mask, mask]

    uint32_t i = arg->sidx;
    const Grp128GF16* b_row = r128m_gf16_raddr((R128MGF16*) arg->b, i);
    Grp128GF16* dst = r128m_gf16_raddr(arg->a, i);
    for(; i < arg->eidx; ++i, ++b_row, ++dst) {
        const Grp128GF16* src = rc128m_gf16_raddr((RC128MGF16*) arg->c, 0);
        __m512i prod = _mm512_and_si512(vd, _mm512_load_si512(dst));
        for(uint32_t j = 0; j < 128; j += 2, src += 2) {
            __m512i v0 = _mm512_load_si512(src);
            __m512i v1 = _mm512_load_si512(src + 1);
            __m512i p0 = grp128_gf16_mul_scalar_bs_avx512(v0, b_row, j);
            __m512i p1 = grp128_gf16_mul_scalar_bs_avx512(v1, b_row, j + 1);
            prod = _mm512_xor_si512(prod, p0);
            prod = _mm512_xor_si512(prod, p1);
        }
        _mm512_store_si512(dst, prod);
    }
}

#elif defined(__AVX2__)

void
r128m_gf16_diag_fma_worker_avx2(void* __arg) {
    R128MGF16PArg* arg = (R128MGF16PArg*) __arg;
    __m256i vd = _mm256_castsi128_si256(_mm_load_si128((__m128i*)arg->d));
    vd = _mm256_permute2x128_si256(vd, vd, 0x0); // [mask, mask]

    uint32_t i = arg->sidx;
    const Grp128GF16* b_row = r128m_gf16_raddr((R128MGF16*) arg->b, i);
    __m256i* dst = (__m256i*) r128m_gf16_raddr(arg->a, i);
    for(; i < arg->eidx; ++i, ++b_row, dst += 2) {
        __m256i prod0 = _mm256_load_si256(dst);
        __m256i prod1 = _mm256_load_si256(dst + 1);
        prod0 = _mm256_and_si256(prod0, vd);
        prod1 = _mm256_and_si256(prod1, vd);

        const __m256i* src =(__m256i*)rc128m_gf16_raddr((RC128MGF16*)arg->c, 0);
        for(uint32_t j = 0; j < 128; j += 2, src += 4) {
            __m256i v0 = _mm256_load_si256(src);
            __m256i v1 = _mm256_load_si256(src + 1);
            __m256i v2 = _mm256_load_si256(src + 2);
            __m256i v3 = _mm256_load_si256(src + 3);
            __m256i p0, p1, p2, p3;
            p0 = grp128_gf16_mul_scalar_bs_avx2(&p1, v0, v1, b_row, j);
            p2 = grp128_gf16_mul_scalar_bs_avx2(&p3, v2, v3, b_row, j + 1);
            prod0 = _mm256_xor_si256(prod0, p0);
            prod1 = _mm256_xor_si256(prod1, p1);
            prod0 = _mm256_xor_si256(prod0, p2);
            prod1 = _mm256_xor_si256(prod1, p3);
        }
        _mm256_store_si256(dst, prod0);
        _mm256_store_si256(dst + 1, prod1);
    }
}

#else

void
r128m_gf16_diag_fma_worker_naive(void* __arg) {
    R128MGF16PArg* arg = (R128MGF16PArg*) __arg;
    uint32_t i = arg->sidx;
    const Grp128GF16* b_row = r128m_gf16_raddr((R128MGF16*) arg->b, i);
    Grp128GF16* dst = r128m_gf16_raddr(arg->a, i);
    for(; i < arg->eidx; ++i, ++b_row, ++dst) {
        grp128_gf16_zero_subset(dst, arg->d);
        const Grp128GF16* src = rc128m_gf16_raddr((RC128MGF16*)arg->c, 0);
        for(uint32_t j = 0; j < 128; j += 2, src += 2) {
            grp128_gf16_fmaddi_scalar_bs(dst, src, b_row, j);
            grp128_gf16_fmaddi_scalar_bs(dst, src + 1, b_row, j + 1);
        }
    }
}

#endif

static void
r128m_gf16_diag_fma_worker(void* __arg) {
#if defined(__AVX512F__)
    r128m_gf16_diag_fma_worker_avx512(__arg);
#elif defined(__AVX2__)
    r128m_gf16_diag_fma_worker_avx2(__arg);
#else
    r128m_gf16_diag_fma_worker_naive(__arg);
#endif
}

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
                             Threadpool* restrict tp) {
    assert(r128m_gf16_rnum(a) == r128m_gf16_rnum(b));
    uint32_t strip_sz = r128m_gf16_rnum(a) / tnum;
    uint32_t sidx = 0;
    for(uint32_t i = 0; i < (tnum - 1); ++i) {
        args[i].a = a;
        args[i].b = b;
        args[i].c = (RC128MGF16*) c;
        args[i].d = d;
        args[i].sidx = sidx;
        sidx += strip_sz;
        args[i].eidx = sidx;
    }
    args[tnum-1].a = a;
    args[tnum-1].b = b;
    args[tnum-1].c = (RC128MGF16*) c;
    args[tnum-1].d = d;
    args[tnum-1].sidx = sidx;
    args[tnum-1].eidx = r128m_gf16_rnum(a);

    for(uint32_t i = 0; i < tnum; ++i) {
        thpool_add_job(tp, r128m_gf16_diag_fma_worker, args + i);
    }
    thpool_wait_jobs(tp);
}

#if defined(__AVX512F__)

void
r128m_gf16_fma_diag_worker_avx512(void* __arg) {
    R128MGF16PArg* arg = (R128MGF16PArg*) __arg;
    __m512i vd = _mm512_castsi128_si512(_mm_load_si128((__m128i*)arg->d));
    vd = _mm512_shuffle_i64x2(vd, vd, 0x0); // [mask, mask, mask, mask]

    uint32_t i = arg->sidx;
    Grp128GF16* dst = r128m_gf16_raddr(arg->a, i);
    const Grp128GF16* b_row = r128m_gf16_raddr((R128MGF16*) arg->b, i);
    for(; i < arg->eidx; ++i, ++b_row, ++dst) {
        const Grp128GF16* src = rc128m_gf16_raddr((RC128MGF16*) arg->c, 0);
        __m512i prod = _mm512_setzero_si512();
        for(uint32_t j = 0; j < 128; j += 2, src += 2) {
            __m512i v0 = _mm512_load_si512(src);
            __m512i v1 = _mm512_load_si512(src + 1);
            __m512i p0 = grp128_gf16_mul_scalar_bs_avx512(v0, b_row, j);
            __m512i p1 = grp128_gf16_mul_scalar_bs_avx512(v1, b_row, j + 1);
            prod = _mm512_xor_si512(prod, p0);
            prod = _mm512_xor_si512(prod, p1);
        }
        prod = _mm512_and_si512(prod, vd);
        __m512i d = _mm512_load_si512(dst);
        _mm512_store_si512(dst, _mm512_xor_si512(prod, d));
    }
}

#elif defined(__AVX2__)

void
r128m_gf16_fma_diag_worker_avx2(void* __arg) {
    R128MGF16PArg* arg = (R128MGF16PArg*) __arg;
    // [mask, U]
    __m256i vd = _mm256_castsi128_si256(_mm_load_si128((__m128i*)arg->d));
    vd = _mm256_permute2x128_si256(vd, vd, 0x0); // [mask, mask]

    uint32_t i = arg->sidx;
    __m256i* dst = (__m256i*) r128m_gf16_raddr(arg->a, i);
    const Grp128GF16* b_row = r128m_gf16_raddr((R128MGF16*) arg->b, i);
    for(; i < arg->eidx; ++i, ++b_row, dst += 2) {
        const __m256i* src = (__m256i*) rc128m_gf16_raddr((RC128MGF16*)arg->c, 0);
        __m256i prod0 = _mm256_setzero_si256();
        __m256i prod1 = _mm256_setzero_si256();
        for(uint32_t j = 0; j < 128; j += 2, src += 4) {
            __m256i v0 = _mm256_load_si256(src);
            __m256i v1 = _mm256_load_si256(src + 1);
            __m256i v2 = _mm256_load_si256(src + 2);
            __m256i v3 = _mm256_load_si256(src + 3);
            __m256i p0, p1, p2, p3;
            p0 = grp128_gf16_mul_scalar_bs_avx2(&p1, v0, v1, b_row, j);
            p2 = grp128_gf16_mul_scalar_bs_avx2(&p3, v2, v3, b_row, j + 1);
            prod0 = _mm256_xor_si256(prod0, p0);
            prod1 = _mm256_xor_si256(prod1, p1);
            prod0 = _mm256_xor_si256(prod0, p2);
            prod1 = _mm256_xor_si256(prod1, p3);
        }
        prod0 = _mm256_and_si256(prod0, vd);
        prod1 = _mm256_and_si256(prod1, vd);
        __m256i d0 = _mm256_load_si256(dst);
        __m256i d1 = _mm256_load_si256(dst + 1);
        _mm256_store_si256(dst, _mm256_xor_si256(d0, prod0));
        _mm256_store_si256(dst + 1, _mm256_xor_si256(d1, prod1));
    }
}

#else

void
r128m_gf16_fma_diag_worker_naive(void* __arg) {
    R128MGF16PArg* arg = (R128MGF16PArg*) __arg;
    uint32_t i = arg->sidx;
    Grp128GF16* dst = r128m_gf16_raddr(arg->a, i);
    const Grp128GF16* b_row = r128m_gf16_raddr((R128MGF16*) arg->b, i);
    for(; i < arg->eidx; ++i, ++b_row, ++dst) {
        const Grp128GF16* src = rc128m_gf16_raddr((RC128MGF16*)arg->c, 0);
        for(uint32_t j = 0; j < 128; j += 2, src += 2) {
            grp128_gf16_fmaddi_scalar_mask_bs(dst, src, b_row, j, arg->d);
            grp128_gf16_fmaddi_scalar_mask_bs(dst, src + 1, b_row, j + 1, arg->d);
        }
    }
}

#endif

static void
r128m_gf16_fma_diag_worker(void* __arg) {
#if defined(__AVX512F__)
    r128m_gf16_fma_diag_worker_avx512(__arg);
#elif defined(__AVX2__)
    r128m_gf16_fma_diag_worker_avx2(__arg);
#else
    r128m_gf16_fma_diag_worker_naive(__arg);
#endif
}

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
                             Threadpool* restrict tp) {
    assert(r128m_gf16_rnum(a) == r128m_gf16_rnum(b));
    uint32_t strip_sz = r128m_gf16_rnum(a) / tnum;
    uint32_t sidx = 0;
    for(uint32_t i = 0; i < (tnum - 1); ++i) {
        args[i].a = a;
        args[i].b = b;
        args[i].c = (RC128MGF16*) c;
        args[i].d = d;
        args[i].sidx = sidx;
        sidx += strip_sz;
        args[i].eidx = sidx;
    }
    args[tnum-1].a = a;
    args[tnum-1].b = b;
    args[tnum-1].c = (RC128MGF16*) c;
    args[tnum-1].d = d;
    args[tnum-1].sidx = sidx;
    args[tnum-1].eidx = r128m_gf16_rnum(a);

    for(uint32_t i = 0; i < tnum; ++i) {
        thpool_add_job(tp, r128m_gf16_fma_diag_worker, args + i);
    }
    thpool_wait_jobs(tp);
}

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
                             Threadpool* restrict tp) {
    r128m_gf16_fma_diag_parallel(a, b, c, d, tnum, args, tp);
}

static void
r128m_gf16_mixi_worker(void* __arg) {
    R128MGF16PArg* arg = (R128MGF16PArg*) __arg;
    uint32_t strip_sz = arg->eidx - arg->sidx;
    uint32_t head = arg->sidx + (strip_sz & ~0x1U);
    uint32_t i = arg->sidx;
    Grp128GF16* dst = r128m_gf16_raddr(arg->a, i);
    const Grp128GF16* src = r128m_gf16_raddr((R128MGF16*)arg->b, i);
#if defined(__AVX2__)
    // [mask, mask]
    __m256i vd = _mm256_castsi128_si256(_mm_load_si128((__m128i*)arg->d));
    vd = _mm256_permute2x128_si256(vd, vd, 0x0); // [mask, mask]
    for(; i < head; i += 2, src += 2, dst += 2) {
        grp128_gf16_mixi_avx2(dst, src, vd);
        grp128_gf16_mixi_avx2(dst + 1, src + 1, vd);
    }

    if(i < arg->eidx)
        grp128_gf16_mixi_avx2(dst, src, vd);
#else
    for(; i < head; i += 2, src += 2, dst += 2) {
        grp128_gf16_mixi(dst, src, arg->d);
        grp128_gf16_mixi(dst + 1, src + 1, arg->d);
    }

    if(i < arg->eidx)
        grp128_gf16_mixi(dst, src, arg->d);
#endif
}

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
                         Threadpool* restrict tp) {
    uint32_t strip_sz = r128m_gf16_rnum(a) / tnum;
    uint32_t sidx = 0;
    for(uint32_t i = 0; i < (tnum - 1); ++i) {
        args[i].a = a;
        args[i].b = b;
        args[i].d = di;
        args[i].sidx = sidx;
        sidx += strip_sz;
        args[i].eidx = sidx;
    }
    args[tnum-1].a = a;
    args[tnum-1].b = b;
    args[tnum-1].d = di;
    args[tnum-1].sidx = sidx;
    args[tnum-1].eidx = r128m_gf16_rnum(a);

    for(uint32_t i = 0; i < tnum; ++i) {
        thpool_add_job(tp, r128m_gf16_mixi_worker, args + i);
    }
    thpool_wait_jobs(tp);
}
