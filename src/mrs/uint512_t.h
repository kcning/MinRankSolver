/* uint512_t.h: header file for struct uint512_t */

#ifndef __BLK_LANCZOS_UINT512_T_H__
#define __BLK_LANCZOS_UINT512_T_H__

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <stdalign.h>
#include <assert.h>

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

#include "util.h"
#include "bitmap_table.h"

/* ========================================================================
 * struct uint512_t definition
 * ======================================================================== */

typedef struct {
    alignas(64) uint64_t s[8];
} uint512_t;

/* ========================================================================
 * functions prototypes
 * ======================================================================== */

/* usage: Given 1 uint512_t a, treat it as uint64_t[8] and
 *      return the specified 64 bits.
 * params:
 *      1) a: ptr to struct uint512_t
 *      2) i: index of the slot; from 0 ~ 7
 * return: the specified 64 bits as a uint64_t */
static force_inline uint64_t
uint512_t_64b_at(const uint512_t* a, uint32_t i) {
    assert(i < 8);
    return a->s[i];
}

/* usage: Given 1 uint512_t a, treat it as uint64_t[8] and set
 *      the specified 64 bits to the given value.
 * params:
 *      1) a: ptr to struct uint512_t
 *      2) i: index of the slot; from 0 ~ 7
 *      3) v: the new value
 * return: void */
static force_inline void
uint512_t_set_64b_at(uint512_t* a, uint32_t i, uint64_t v) {
    assert(i < 8);
    a->s[i] = v;
}

/* usage: Given 1 uint512_t a, treat it as uint64_t[8] and set
 *      the all 8 64-bit integers to the given value.
 * params:
 *      1) a: ptr to struct uint512_t
 *      3) v: the new value
 * return: void */
static force_inline void
uint512_t_set1_64b(uint512_t* a, uint64_t v) {
#if defined(__AVX512F__)
    __m512i av = _mm512_set1_epi64(v);
    _mm512_store_si512(a->s, av);
#else
    a->s[0] = v;
    a->s[1] = v;
    a->s[2] = v;
    a->s[3] = v;
    a->s[4] = v;
    a->s[5] = v;
    a->s[6] = v;
    a->s[7] = v;
#endif
}

/* usage: Given 1 uint512_t a, treat it as uint8_t[64] and return
 *      the specified 8 bits.
 * params:
 *      1) a: ptr to struct uint512_t
 *      2) i: index of the byte; from 0 ~ 63
 * return: the specified 8-bits as a uint8_t */
static force_inline uint8_t
uint512_t_8b_at(const uint512_t* a, uint32_t i) {
    assert(i < 64);
    uint8_t* s_8b = (uint8_t*) a->s;
    return s_8b[i];
}

/* usage: Given 1 uint512_t a, treat it as uint8_t[64] and set
 *      the specified 8 bits to the given value
 * params:
 *      1) a: ptr to struct uint512_t
 *      2) i: index of the byte; from 0 ~ 63
 *      3) v: the new value
 * return: void */
static force_inline void
uint512_t_set_8b_at(const uint512_t* a, uint32_t i, uint8_t v) {
    assert(i < 64);
    uint8_t* s_8b = (uint8_t*) a->s;
    s_8b[i] = v;
}

/* usage: Given 1 uint512_t a, set a to all 0's.
 * params:
 *      1) a: ptr to struct uint512_t
 * return: void */
#define uint512_t_zero(a) do { \
    memset((a), 0x0, sizeof(uint512_t)); \
} while(0)

/* usage: Given 1 uint512_t, set a to all 1's.
 * params:
 *      1) a: ptr to struct uint512_t
 * return: void */
static force_inline void
uint512_t_max(uint512_t* a) {
#if defined(__AVX512F__)
    __m512i v = _mm512_set1_epi64(-1);
    _mm512_store_si512((void*) a->s, v);
#else
    uint512_t_set1_64b(a, UINT64_MAX);
#endif
}

/* usage: Given 1 uint512_t a, return true if all 1's. Otherwise false;
 * params:
 *      1) a: ptr to struct uint512_t
 * return: void */
static force_inline bool
uint512_t_is_max(const uint512_t* const a) {
#if defined(__AVX512F__)
    __m512i v = _mm512_load_si512(a->s);
    __m512i mask = _mm512_set1_epi64(-1);
    __m512i nv = _mm512_xor_si512(v, mask);
    return !_mm512_test_epi64_mask(nv, mask);
#else
    for(uint32_t i = 0; i < 8; ++i) {
        if(a->s[i] != UINT64_MAX)
            return false;
    }
    return true;
#endif
}

/* usage: Given 1 uint512_t a, return false if all 0's. Otherwise true;
 * params:
 *      1) a: ptr to struct uint512_t
 * return: void */
static force_inline bool
uint512_t_is_not_zero(const uint512_t* const a) {
#if defined(__AVX512F__)
    __m512i v = _mm512_load_si512(a->s);
    __m512i mask = _mm512_set1_epi64(-1);
    return _mm512_test_epi64_mask(v, mask);
#else
    return (a->s[ 0] || a->s[ 1] || a->s[ 2] || a->s[ 3] ||
            a->s[ 4] || a->s[ 5] || a->s[ 6] || a->s[ 7]);
#endif
}

/* usage: Given 1 uint512_t a, return true if all 0's. Otherwise false;
 * params:
 *      1) a: ptr to struct uint512_t
 * return: void */
static force_inline bool
uint512_t_is_zero(const uint512_t* const a) {
    return !uint512_t_is_not_zero(a);
}

/* usage: Given 2 uint512_t a and b, check if they are the same.
 * params:
 *      1) a: ptr to struct uint512_t
 *      2) b: ptr to struct uint512_t
 * return: true if they are the same. otherwise false */
static force_inline bool
uint512_t_equal(const uint512_t* const a, const uint512_t* const b) {
    return (0 == memcmp(a, b, sizeof(uint512_t)));
}

/* usage: Given 1 uint512_t a, bitwise negate it.
 * params:
 *      1) a: ptr to struct uint512_t
 * return: void */
#if defined(__AVX512F__)

#define uint512_t_negi(a) do { \
    uint512_t* const aa = (a); \
    __m512i v0 = _mm512_load_si512((void const*) aa->s); \
    _mm512_store_si512((void*) aa->s, ~v0); \
} while(0)

#elif defined(__AVX2__)

#define uint512_t_negi(a) do { \
    uint512_t* const aa = (a); \
    __m256i v0 = _mm256_load_si256((__m256i*) aa->s); \
    __m256i v1 = _mm256_load_si256((__m256i*) &aa->s[4]); \
    __m256i all1s = _mm256_set1_epi32(-1); \
    _mm256_store_si256((__m256i*) aa->s, _mm256_xor_si256(v0, all1s)); \
    _mm256_store_si256((__m256i*) &aa->s[4], _mm256_xor_si256(v1, all1s)); \
} while(0)

#elif defined(__AVX__)

#define uint512_t_negi(a) do { \
    uint512_t* const aa = (a); \
    __m256i v0 = _mm256_load_si256((__m256i*) aa->s); \
    __m256i v1 = _mm256_load_si256((__m256i*) &aa->s[4]); \
    __m256i all1s = _mm256_set1_epi32(-1); \
    v0 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(v0), \
                                           _mm256_castsi256_pd(all1s))); \
    _mm256_store_si256((__m256i*) aa->s, v0); \
    v1 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(v1), \
                                           _mm256_castsi256_pd(all1s))); \
    _mm256_store_si256((__m256i*) &aa->s[4], v1); \
} while(0)

#else

#define uint512_t_negi(a) do { \
    uint512_t* const aa = (a); \
    aa->s[ 0] = ~aa->s[ 0]; \
    aa->s[ 1] = ~aa->s[ 1]; \
    aa->s[ 2] = ~aa->s[ 2]; \
    aa->s[ 3] = ~aa->s[ 3]; \
    aa->s[ 4] = ~aa->s[ 4]; \
    aa->s[ 5] = ~aa->s[ 5]; \
    aa->s[ 6] = ~aa->s[ 6]; \
    aa->s[ 7] = ~aa->s[ 7]; \
} while(0)

#endif

/* usage: Given 1 uint512_t a, return the number of set bits in a
 * params:
 *      1) a: ptr to struct uint512_t
 * return: the number of 1's in the uint512_t */
static force_inline uint64_t
uint512_t_popcount(const uint512_t* const a) {
    // TODO: AVX512
    return (__builtin_popcountll(a->s[ 0]) + __builtin_popcountll(a->s[ 1]) +
            __builtin_popcountll(a->s[ 2]) + __builtin_popcountll(a->s[ 3]) +
            __builtin_popcountll(a->s[ 4]) + __builtin_popcountll(a->s[ 5]) +
            __builtin_popcountll(a->s[ 6]) + __builtin_popcountll(a->s[ 7]));
}

/* usage: Given 1 uint512_t a, set a to a random value.
 * params:
 *      1) a: ptr to struct uint512_t
 * return: void */
static force_inline void
uint512_t_rand(uint512_t* a) {
    int32_t* buf = (int32_t*) (a->s);
    for(uint32_t i = 0; i < 16; ++i) {
        buf[i] = rand();
    }
}

/* usage: Given 2 uint512_t a and b, copy b into a.
 * params:
 *      1) a: ptr to struct uint512_t
 *      2) b: ptr to struct uint512_t
 * return: void */
#define uint512_t_copy(a, b) do { \
    memcpy((a), (b), sizeof(uint512_t)); \
} while(0)

/* usage: Given 2 uint512_t a and b, swap their values
 * params:
 *      1) a: ptr to struct uint512_t
 *      2) b: ptr to struct uint512_t
 * return: void */
static force_inline void
uint512_t_swap(uint512_t* restrict a, uint512_t* restrict b) {
#if defined(__AVX512F__)
    __m512i* s1 = (__m512i*) a->s;
    __m512i* s2 = (__m512i*) b->s;
    __m512i a0 = _mm512_load_si512(s1);
    __m512i b0 = _mm512_load_si512(s2);
    _mm512_store_si512(s1, b0);
    _mm512_store_si512(s2, a0);
#elif defined(__AVX__)
    __m256i* s1 = (__m256i*) a->s;
    __m256i* s2 = (__m256i*) b->s;
    __m256i a0 = _mm256_load_si256(s1);
    __m256i a1 = _mm256_load_si256(s1 + 1);
    __m256i b0 = _mm256_load_si256(s2);
    __m256i b1 = _mm256_load_si256(s2 + 1);
    _mm256_store_si256(s1, b0);
    _mm256_store_si256(s1 + 1, b1);
    _mm256_store_si256(s2, a0);
    _mm256_store_si256(s2 + 1, a1);
#else
    uint64_t t0 = a->s[0];
    uint64_t t1 = a->s[1];
    uint64_t t2 = a->s[2];
    uint64_t t3 = a->s[3];
    uint64_t t4 = a->s[4];
    uint64_t t5 = a->s[5];
    uint64_t t6 = a->s[6];
    uint64_t t7 = a->s[7];
    a->s[0] = b->s[0];
    a->s[1] = b->s[1];
    a->s[2] = b->s[2];
    a->s[3] = b->s[3];
    a->s[4] = b->s[4];
    a->s[5] = b->s[5];
    a->s[6] = b->s[6];
    a->s[7] = b->s[7];
    b->s[0] = t0;
    b->s[1] = t1;
    b->s[2] = t2;
    b->s[3] = t3;
    b->s[4] = t4;
    b->s[5] = t5;
    b->s[6] = t6;
    b->s[7] = t7;
#endif
}

/* usage: Given 2 uint512_t a and b, copy b into a and then negate a bitwise.
 * params:
 *      1) a: ptr to struct uint512_t
 *      2) b: ptr to struct uint512_t
 * return: void */
#if defined(__AVX512F__)

#define uint512_t_neg(a, b) do { \
    uint512_t* const aa = (a); \
    const uint512_t* const bb = (b); \
    __m512i v0 = _mm512_load_si512((void const*) bb->s); \
    _mm512_store_si512((void*) aa->s, ~v0); \
} while(0)

#elif defined(__AVX2__)

#define uint512_t_neg(a, b) do { \
    uint512_t* const aa = (a); \
    const uint512_t* const bb = (b); \
    __m256i v0 = _mm256_load_si256((__m256i*) bb->s); \
    __m256i v1 = _mm256_load_si256((__m256i*) &bb->s[4]); \
    __m256i all1s = _mm256_set1_epi32(-1); \
    _mm256_store_si256((__m256i*) aa->s, _mm256_xor_si256(v0, all1s)); \
    _mm256_store_si256((__m256i*) &aa->s[4], _mm256_xor_si256(v1, all1s)); \
} while(0)

#elif defined(__AVX__)

#define uint512_t_neg(a, b) do { \
    uint512_t* const aa = (a); \
    const uint512_t* const bb = (b); \
    __m256i v0 = _mm256_load_si256((__m256i*) bb->s); \
    __m256i v1 = _mm256_load_si256((__m256i*) &bb->s[4]); \
    __m256i all1s = _mm256_set1_epi32(-1); \
    v0 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(v0), \
                                           _mm256_castsi256_pd(all1s))); \
    _mm256_store_si256((__m256i*) aa->s, v0); \
    v1 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(v1), \
                                           _mm256_castsi256_pd(all1s))); \
    _mm256_store_si256((__m256i*) &aa->s[4], v1); \
} while(0)

#else

#define uint512_t_neg(a, b) do { \
    uint512_t* const aa = (a); \
    const uint512_t* const bb = (b); \
    aa->s[ 0] = ~bb->s[ 0]; \
    aa->s[ 1] = ~bb->s[ 1]; \
    aa->s[ 2] = ~bb->s[ 2]; \
    aa->s[ 3] = ~bb->s[ 3]; \
    aa->s[ 4] = ~bb->s[ 4]; \
    aa->s[ 5] = ~bb->s[ 5]; \
    aa->s[ 6] = ~bb->s[ 6]; \
    aa->s[ 7] = ~bb->s[ 7]; \
} while(0)

#endif

/* usage: Given 2 uint512_t a and b, compute a^b and store the result into p
 * params:
 *      1) p: ptr to struct uint512_t
 *      2) a: ptr to struct uint512_t
 *      3) b: ptr to struct uint512_t
 * return: void */
#if defined(__AVX512F__)

#define uint512_t_xor(p, a, b) do { \
    uint512_t* const pp = (p); \
    const uint512_t* const aa = (a); \
    const uint512_t* const bb = (b); \
    __m512i va0 = _mm512_load_si512((void const*) aa->s); \
    __m512i vb0 = _mm512_load_si512((void const*) bb->s); \
    va0 = _mm512_xor_si512(va0, vb0); \
    _mm512_store_si512((void*) pp->s, va0); \
} while(0)

#elif defined(__AVX2__)

#define uint512_t_xor(p, a, b) do { \
    uint512_t* const pp = (p); \
    const uint512_t* const aa = (a); \
    const uint512_t* const bb = (b); \
    __m256i va0 = _mm256_load_si256((__m256i*) aa->s); \
    __m256i va1 = _mm256_load_si256((__m256i*) &aa->s[4]); \
    __m256i vb0 = _mm256_load_si256((__m256i*) bb->s); \
    __m256i vb1 = _mm256_load_si256((__m256i*) &bb->s[4]); \
    va0 = _mm256_xor_si256(vb0, va0); \
    va1 = _mm256_xor_si256(vb1, va1); \
    _mm256_store_si256((__m256i*) pp->s, va0); \
    _mm256_store_si256((__m256i*) &pp->s[4], va1); \
} while(0)

#elif defined(__AVX__)

#define uint512_t_xor(p, a, b) do { \
    uint512_t* const pp = (p); \
    const uint512_t* const aa = (a); \
    const uint512_t* const bb = (b); \
    __m256i va0 = _mm256_load_si256((__m256i*) aa->s); \
    __m256i va1 = _mm256_load_si256((__m256i*) &aa->s[4]); \
    __m256i vb0 = _mm256_load_si256((__m256i*) bb->s); \
    __m256i vb1 = _mm256_load_si256((__m256i*) &bb->s[4]); \
    va0 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(vb0), \
                                            _mm256_castsi256_pd(va0))); \
    va1 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(vb1), \
                                            _mm256_castsi256_pd(va1))); \
    _mm256_store_si256((__m256i*) pp->s, va0); \
    _mm256_store_si256((__m256i*) &pp->s[4], va1); \
} while(0)

#else

#define uint512_t_xor(p, a, b) do { \
    uint512_t* const pp = (p); \
    const uint512_t* const aa = (a); \
    const uint512_t* const bb = (b); \
    pp->s[ 0] = aa->s[ 0] ^ bb->s[ 0]; \
    pp->s[ 1] = aa->s[ 1] ^ bb->s[ 1]; \
    pp->s[ 2] = aa->s[ 2] ^ bb->s[ 2]; \
    pp->s[ 3] = aa->s[ 3] ^ bb->s[ 3]; \
    pp->s[ 4] = aa->s[ 4] ^ bb->s[ 4]; \
    pp->s[ 5] = aa->s[ 5] ^ bb->s[ 5]; \
    pp->s[ 6] = aa->s[ 6] ^ bb->s[ 6]; \
    pp->s[ 7] = aa->s[ 7] ^ bb->s[ 7]; \
} while(0)

#endif

#if defined(__AVX512F__)

/* usage: Given 1 uint512_t a and 1 __m512i registers, load the content of
 *      a into the register
 * params:
 *      1) a: ptr to struct uint512_t
 *      2) va: register that stores the content of uint512_t a
 * return: void */
#define uint512_t_load_to_reg_avx512(a, va) do { \
    const uint512_t* const aa = (a); \
    va = _mm512_load_si512((void const*) aa->s); \
} while(0)

/* usage: Given 1 uint512_t a and 1 __m512i registers, store the content of
 *      the register into a
 * params:
 *      1) a: ptr to struct uint512_t
 *      2) va: register to store the content of uint512_t a
 * return: void */
#define uint512_t_load_from_reg_avx512(a, va) do { \
    uint512_t* const aa = (a); \
    _mm512_store_si512((void*) aa->s, va); \
} while(0)

/* usage: Given 1 uint512_t a and 1 __m512i registers, store the content of
 *      the register into a with non-temporal hint.
 * params:
 *      1) a: ptr to struct uint512_t
 *      2) va: register to store the content of uint512_t a
 * return: void */
#define uint512_t_load_from_reg_avx512_nt(a, va) do { \
    uint512_t* const aa = (a); \
    _mm512_stream_si512((void*) aa->s, va); \
} while(0)

/* usage: Given 2 uint512_t a and b, where the content of a is stored in 1
 *      __m512i register, compute a^b and store the result back into the
 *      register (in place xor).
 * params:
 *      1) b: ptr to struct uint512_t
 *      2) va: register that store the content of uint512_t a
 * return: void */
#define uint512_t_xori_to_reg_avx512(b, va) do { \
    const uint512_t* const bb = (b); \
    __m512i vb = _mm512_load_si512((void const*) bb->s); \
    va = _mm512_xor_si512(va, vb); \
} while(0)

/* usage: Given 2 uint512_t a and b, where the content of b is stored in 1
 *      __m512i register, compute a^b and store the result back into a
 *      (in place xor).
 * params:
 *      1) a: ptr to struct uint512_t
 *      2) vb: register that store the content of uint512_t b
 * return: void */
#define uint512_t_xori_from_reg_avx512(a, vb) do { \
    uint512_t* const aa = (a); \
    __m512i va = _mm512_load_si512((void const*) aa->s); \
    va = _mm512_xor_si512(va, vb); \
    _mm512_store_si512((void*) aa->s, va); \
} while(0)

#endif

#if defined(__AVX__)

/* usage: Given 1 uint512_t a and 2 __m256i registers, load the content of
 *      a into those 2 registers
 * params:
 *      1) a: ptr to struct uint512_t
 *      2) va0: register to store the 1st 256 bits of uint512_t a
 *      3) va1: register to store the 2nd 256 bits of uint512_t a
 * return: void */
#define uint512_t_load_to_reg(a, va0, va1) do { \
    const uint512_t* const aa = (a); \
    va0 = _mm256_load_si256((__m256i*) aa->s); \
    va1 = _mm256_load_si256((__m256i*) &aa->s[4]); \
} while(0)

/* usage: Given 1 uint512_t a and 2 __m256i registers, store the content of
 *      those 2 registers into a
 * params:
 *      1) a: ptr to struct uint512_t
 *      2) va0: register to store the 1st 256 bits of uint512_t a
 *      3) va1: register to store the 2nd 256 bits of uint512_t a
 * return: void */
#define uint512_t_load_from_reg(a, va0, va1) do { \
    uint512_t* const aa = (a); \
    _mm256_store_si256((__m256i*) aa->s, va0); \
    _mm256_store_si256((__m256i*) &aa->s[4], va1); \
} while(0)

/* usage: Given 1 uint512_t a and 2 __m256i registers, store the content of
 *      those 2 registers into a with non-temporal hint.
 * params:
 *      1) a: ptr to struct uint512_t
 *      2) va0: register to store the 1st 256 bits of uint512_t a
 *      3) va1: register to store the 2nd 256 bits of uint512_t a
 * return: void */
#define uint512_t_load_from_reg_nt(a, va0, va1) do { \
    uint512_t* const aa = (a); \
    _mm256_stream_si256((__m256i*) aa->s, va0); \
    _mm256_stream_si256((__m256i*) &aa->s[4], va1); \
} while(0)

#endif

#if defined(__AVX2__)

/* usage: Given 2 uint512_t a and b, where the content of a is stored in 2
 *      __m256i registers, compute a^b and store the result back into those
 *      registers (in place xor).
 * params:
 *      1) b: ptr to struct uint512_t
 *      2) va0: register that store the 1st 256 bits of uint512_t a
 *      3) va1: register that stores the 2nd 256 bits of uint512_t a
 * return: void */
#define uint512_t_xori_to_reg(b, va0, va1) do { \
    const uint512_t* const bb = (b); \
    __m256i vb0 = _mm256_load_si256((__m256i*) bb->s); \
    __m256i vb1 = _mm256_load_si256((__m256i*) &bb->s[4]); \
    va0 = _mm256_xor_si256(vb0, va0); \
    va1 = _mm256_xor_si256(vb1, va1); \
} while(0)

/* usage: Given 2 uint512_t a and b, where the content of b is stored in 2
 *      __m256i registers, compute a^b and store the result back into a
 *      (in place xor).
 * params:
 *      1) a: ptr to struct uint512_t
 *      2) vb0: register that store the 1st 256 bits of uint512_t b
 *      3) vb1: register that stores the 2nd 256 bits of uint512_t b
 * return: void */
#define uint512_t_xori_from_reg(a, vb0, vb1) do { \
    uint512_t* const aa = (a); \
    __m256i va0 = _mm256_load_si256((__m256i*) aa->s); \
    __m256i va1 = _mm256_load_si256((__m256i*) &aa->s[4]); \
    va0 = _mm256_xor_si256(vb0, va0); \
    va1 = _mm256_xor_si256(vb1, va1); \
    _mm256_store_si256((__m256i*) aa->s, va0); \
    _mm256_store_si256((__m256i*) &aa->s[4], va1); \
} while(0)

#elif defined(__AVX__)

/* usage: Given 2 uint512_t a and b, where the content of a is stored in 2
 *      __m256i registers, compute a^b and store the result back into those
 *      registers (in place xor).
 * params:
 *      1) b: ptr to struct uint512_t
 *      2) va0: register that store the 1st 256 bits of uint512_t a
 *      3) va1: register that stores the 2nd 256 bits of uint512_t a
 * return: void */
#define uint512_t_xori_to_reg(b, va0, va1) do { \
    const uint512_t* const bb = (b); \
    __m256i vb0 = _mm256_load_si256((__m256i*) bb->s); \
    __m256i vb1 = _mm256_load_si256((__m256i*) &bb->s[4]); \
    va0 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(vb0), \
                                            _mm256_castsi256_pd(va0))); \
    va1 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(vb1), \
                                            _mm256_castsi256_pd(va1))); \
} while(0)

/* usage: Given 2 uint512_t a and b, where the content of b is stored in 2
 *      __m256i registers, compute a^b and store the result back into a
 *      (in place xor).
 * params:
 *      1) a: ptr to struct uint512_t
 *      2) vb0: register that store the 1st 256 bits of uint512_t b
 *      3) vb1: register that stores the 2nd 256 bits of uint512_t b
 * return: void */
#define uint512_t_xori_from_reg(a, vb0, vb1) do { \
    uint512_t* const aa = (a); \
    __m256i va0 = _mm256_load_si256((__m256i*) aa->s); \
    __m256i va1 = _mm256_load_si256((__m256i*) &aa->s[4]); \
    va0 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(vb0), \
                                            _mm256_castsi256_pd(va0))); \
    va1 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(vb1), \
                                            _mm256_castsi256_pd(va1))); \
    _mm256_store_si256((__m256i*) aa->s, va0); \
    _mm256_store_si256((__m256i*) &aa->s[4], va1); \
} while(0)

#endif

/* usage: Given 2 uint512_t a and b, compute a^b and store the result into a.
 *      (in place xor)
 * params:
 *      1) a: ptr to struct uint512_t
 *      2) b: ptr to struct uint512_t
 * return: void */
#if defined(__AVX512F__)

#define uint512_t_xori(a, b) do { \
    uint512_t* const aa = (a); \
    const uint512_t* const bb = (b); \
    __m512i va = _mm512_load_si512((void const*) aa->s); \
    __m512i vb = _mm512_load_si512((void const*) bb->s); \
    va = _mm512_xor_si512(va, vb); \
    _mm512_store_si512((void*) aa->s, va); \
} while(0)

#elif defined(__AVX2__)

#define uint512_t_xori(a, b) do { \
    uint512_t* const aa = (a); \
    const uint512_t* const bb = (b); \
    __m256i va0 = _mm256_load_si256((__m256i*) aa->s); \
    __m256i va1 = _mm256_load_si256((__m256i*) &aa->s[4]); \
    __m256i vb0 = _mm256_load_si256((__m256i*) bb->s); \
    __m256i vb1 = _mm256_load_si256((__m256i*) &bb->s[4]); \
    va0 = _mm256_xor_si256(vb0, va0); \
    va1 = _mm256_xor_si256(vb1, va1); \
    _mm256_store_si256((__m256i*) aa->s, va0); \
    _mm256_store_si256((__m256i*) &aa->s[4], va1); \
} while(0)

#elif defined(__AVX__)

#define uint512_t_xori(a, b) do { \
    uint512_t* const aa = (a); \
    const uint512_t* const bb = (b); \
    __m256i va0 = _mm256_load_si256((__m256i*) aa->s); \
    __m256i va1 = _mm256_load_si256((__m256i*) &aa->s[4]); \
    __m256i vb0 = _mm256_load_si256((__m256i*) bb->s); \
    __m256i vb1 = _mm256_load_si256((__m256i*) &bb->s[4]); \
    va0 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(vb0), \
                                            _mm256_castsi256_pd(va0))); \
    va1 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(vb1), \
                                            _mm256_castsi256_pd(va1))); \
    _mm256_store_si256((__m256i*) aa->s, va0); \
    _mm256_store_si256((__m256i*) &aa->s[4], va1); \
} while(0)

#else

#define uint512_t_xori(a, b) do { \
    uint512_t* const aa = (a); \
    const uint512_t* const bb = (b); \
    aa->s[ 0] ^= bb->s[ 0]; \
    aa->s[ 1] ^= bb->s[ 1]; \
    aa->s[ 2] ^= bb->s[ 2]; \
    aa->s[ 3] ^= bb->s[ 3]; \
    aa->s[ 4] ^= bb->s[ 4]; \
    aa->s[ 5] ^= bb->s[ 5]; \
    aa->s[ 6] ^= bb->s[ 6]; \
    aa->s[ 7] ^= bb->s[ 7]; \
} while(0)

#endif

/* usage: Given 3 uint512_t a, b, and c, compute a ^ (b & c) and store the
 *      result back into a. (in place)
 * params:
 *      1) a: ptr to struct uint512_t
 *      2) b: ptr to struct uint512_t
 *      3) c: ptr to struct uint512_t
 * return: void */
#if defined(__AVX512F__)

#define uint512_t_xori_and(a, b, c) do { \
    uint512_t* const aa = (a); \
    const uint512_t* const bb = (b); \
    const uint512_t* const cc = (c); \
    __m512i vc0 = _mm512_load_si512((void const*) cc->s); \
    __m512i vb0 = _mm512_load_si512((void const*) bb->s); \
    vb0 = _mm512_and_si512(vb0, vc0); \
    __m512i va0 = _mm512_load_si512((void const*) aa->s); \
    va0 = _mm512_xor_si512(vb0, va0); \
    _mm512_store_si512((void*) aa->s, va0); \
} while(0)

#elif defined(__AVX2__)

#define uint512_t_xori_and(a, b, c) do { \
    uint512_t* const aa = (a); \
    const uint512_t* const bb = (b); \
    const uint512_t* const cc = (c); \
    __m256i vc0 = _mm256_load_si256((__m256i*) cc->s); \
    __m256i vc1 = _mm256_load_si256((__m256i*) &cc->s[4]); \
    __m256i vb0 = _mm256_load_si256((__m256i*) bb->s); \
    __m256i vb1 = _mm256_load_si256((__m256i*) &bb->s[4]); \
    vb0 = _mm256_and_si256(vb0, vc0); \
    vb1 = _mm256_and_si256(vb1, vc1); \
    __m256i va0 = _mm256_load_si256((__m256i*) aa->s); \
    __m256i va1 = _mm256_load_si256((__m256i*) &aa->s[4]); \
    va0 = _mm256_xor_si256(vb0, va0); \
    va1 = _mm256_xor_si256(vb1, va1); \
    _mm256_store_si256((__m256i*) aa->s, va0); \
    _mm256_store_si256((__m256i*) &aa->s[4], va1); \
} while(0)

#elif defined(__AVX__)

#define uint512_t_xori_and(a, b, c) do { \
    uint512_t* const aa = (a); \
    const uint512_t* const bb = (b); \
    const uint512_t* const cc = (c); \
    __m256i vc0 = _mm256_load_si256((__m256i*) cc->s); \
    __m256i vc1 = _mm256_load_si256((__m256i*) &cc->s[4]); \
    __m256i vb0 = _mm256_load_si256((__m256i*) bb->s); \
    __m256i vb1 = _mm256_load_si256((__m256i*) &bb->s[4]); \
    vb0 = _mm256_castpd_si256(_mm256_and_pd(_mm256_castsi256_pd(vb0), \
                                            _mm256_castsi256_pd(vc0))); \
    vb1 = _mm256_castpd_si256(_mm256_and_pd(_mm256_castsi256_pd(vb1), \
                                            _mm256_castsi256_pd(vc1))); \
    __m256i va0 = _mm256_load_si256((__m256i*) aa->s); \
    __m256i va1 = _mm256_load_si256((__m256i*) &aa->s[4]); \
    va0 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(vb0), \
                                            _mm256_castsi256_pd(va0))); \
    va1 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(vb1), \
                                            _mm256_castsi256_pd(va1))); \
    _mm256_store_si256((__m256i*) aa->s, va0); \
    _mm256_store_si256((__m256i*) &aa->s[4], va1); \
} while(0)

#else

#define uint512_t_xori_and(a, b, c) do { \
    uint512_t* const aa = (a); \
    const uint512_t* const bb = (b); \
    const uint512_t* const cc = (c); \
    aa->s[ 0] ^= bb->s[ 0] & cc->s[ 0]; \
    aa->s[ 1] ^= bb->s[ 1] & cc->s[ 1]; \
    aa->s[ 2] ^= bb->s[ 2] & cc->s[ 2]; \
    aa->s[ 3] ^= bb->s[ 3] & cc->s[ 3]; \
    aa->s[ 4] ^= bb->s[ 4] & cc->s[ 4]; \
    aa->s[ 5] ^= bb->s[ 5] & cc->s[ 5]; \
    aa->s[ 6] ^= bb->s[ 6] & cc->s[ 6]; \
    aa->s[ 7] ^= bb->s[ 7] & cc->s[ 7]; \
} while(0)

#endif

/* usage: Given 3 uint512_t a, b, c, compute (a & c) ^ (b & (~c))
 *      and store the result into p.
 * params:
 *      1) p: ptr to struct uint512_t
 *      2) a: ptr to struct uint512_t
 *      3) b: ptr to struct uint512_t
 *      4) c: ptr to struct uint512_t
 * return: void */
#if defined(__AVX512F__)

#define uint512_t_mix(p, a, b, c) do { \
    uint512_t* const pp = (p); \
    const uint512_t* const aa = (a); \
    const uint512_t* const bb = (b); \
    const uint512_t* const cc = (c); \
    __m512i vc0 = _mm512_load_si512((void const*) cc->s); \
    __m512i va0 = _mm512_load_si512((void const*) aa->s); \
    va0 = _mm512_and_si512(va0, vc0); \
    __m512i vb0 = _mm512_load_si512((void const*) bb->s); \
    vb0 = _mm512_andnot_si512(vc0, vb0); \
    va0 = _mm512_xor_si512(va0, vb0); \
    _mm512_store_si512((void*) pp->s, va0); \
} while(0)

#elif defined(__AVX2__)

#define uint512_t_mix(p, a, b, c) do { \
    uint512_t* const pp = (p); \
    const uint512_t* const aa = (a); \
    const uint512_t* const bb = (b); \
    const uint512_t* const cc = (c); \
    __m256i vc0 = _mm256_load_si256((__m256i*) cc->s); \
    __m256i vc1 = _mm256_load_si256((__m256i*) &cc->s[4]); \
    __m256i va0 = _mm256_load_si256((__m256i*) aa->s); \
    __m256i va1 = _mm256_load_si256((__m256i*) &aa->s[4]); \
    va0 = _mm256_and_si256(va0, vc0); \
    va1 = _mm256_and_si256(va1, vc1); \
    __m256i vb0 = _mm256_load_si256((__m256i*) bb->s); \
    __m256i vb1 = _mm256_load_si256((__m256i*) &bb->s[4]); \
    vb0 = _mm256_andnot_si256(vc0, vb0); \
    vb1 = _mm256_andnot_si256(vc1, vb1); \
    va0 = _mm256_xor_si256(va0, vb0); \
    va1 = _mm256_xor_si256(va1, vb1); \
    _mm256_store_si256((__m256i*) pp->s, va0); \
    _mm256_store_si256((__m256i*) &pp->s[4], va1); \
} while(0)

#elif defined(__AVX__)

#define uint512_t_mix(p, a, b, c) do { \
    uint512_t* const pp = (p); \
    const uint512_t* const aa = (a); \
    const uint512_t* const bb = (b); \
    const uint512_t* const cc = (c); \
    __m256i vc0 = _mm256_load_si256((__m256i*) cc->s); \
    __m256i vc1 = _mm256_load_si256((__m256i*) &cc->s[4]); \
    __m256i va0 = _mm256_load_si256((__m256i*) aa->s); \
    __m256i va1 = _mm256_load_si256((__m256i*) &aa->s[4]); \
    va0 = _mm256_castpd_si256(_mm256_and_pd(_mm256_castsi256_pd(va0), \
                                            _mm256_castsi256_pd(vc0))); \
    va1 = _mm256_castpd_si256(_mm256_and_pd(_mm256_castsi256_pd(va1), \
                                            _mm256_castsi256_pd(vc1))); \
    __m256i vb0 = _mm256_load_si256((__m256i*) bb->s); \
    __m256i vb1 = _mm256_load_si256((__m256i*) &bb->s[4]); \
    vb0 = _mm256_castpd_si256(_mm256_andnot_pd(_mm256_castsi256_pd(vc0), \
                                               _mm256_castsi256_pd(vb0))); \
    vb1 = _mm256_castpd_si256(_mm256_andnot_pd(_mm256_castsi256_pd(vc1), \
                                               _mm256_castsi256_pd(vb1))); \
    va0 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(va0), \
                                            _mm256_castsi256_pd(vb0))); \
    va1 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(va1), \
                                            _mm256_castsi256_pd(vb1))); \
    _mm256_store_si256((__m256i*) pp->s, va0); \
    _mm256_store_si256((__m256i*) &pp->s[4], va1); \
} while(0)

#else

#define uint512_t_mix(p, a, b, c) do { \
    uint512_t* const pp = (p); \
    const uint512_t* const aa = (a); \
    const uint512_t* const bb = (b); \
    const uint512_t* const cc = (c); \
    pp->s[ 0] = aa->s[ 0] & cc->s[ 0]; \
    pp->s[ 1] = aa->s[ 1] & cc->s[ 1]; \
    pp->s[ 2] = aa->s[ 2] & cc->s[ 2]; \
    pp->s[ 3] = aa->s[ 3] & cc->s[ 3]; \
    pp->s[ 4] = aa->s[ 4] & cc->s[ 4]; \
    pp->s[ 5] = aa->s[ 5] & cc->s[ 5]; \
    pp->s[ 6] = aa->s[ 6] & cc->s[ 6]; \
    pp->s[ 7] = aa->s[ 7] & cc->s[ 7]; \
    pp->s[ 0] ^= bb->s[ 0] & ~cc->s[ 0]; \
    pp->s[ 1] ^= bb->s[ 1] & ~cc->s[ 1]; \
    pp->s[ 2] ^= bb->s[ 2] & ~cc->s[ 2]; \
    pp->s[ 3] ^= bb->s[ 3] & ~cc->s[ 3]; \
    pp->s[ 4] ^= bb->s[ 4] & ~cc->s[ 4]; \
    pp->s[ 5] ^= bb->s[ 5] & ~cc->s[ 5]; \
    pp->s[ 6] ^= bb->s[ 6] & ~cc->s[ 6]; \
    pp->s[ 7] ^= bb->s[ 7] & ~cc->s[ 7]; \
} while(0)

#endif

/* usage: Given 3 uint512_t a, b, c, compute (a & c) ^ (b & (~c))
 *      and store the result back into a. (in place)
 * params:
 *      1) a: ptr to struct uint512_t
 *      2) b: ptr to struct uint512_t
 *      3) c: ptr to struct uint512_t
 * return: void */
#if defined(__AVX512F__)

#define uint512_t_mixi(a, b, c) do { \
    uint512_t* const aa = (a); \
    const uint512_t* const bb = (b); \
    const uint512_t* const cc = (c); \
    __m512i vc0 = _mm512_load_si512((void const*) cc->s); \
    __m512i va0 = _mm512_load_si512((void const*) aa->s); \
    va0 = _mm512_and_si512(va0, vc0); \
    __m512i vb0 = _mm512_load_si512((void const*) bb->s); \
    vb0 = _mm512_andnot_si512(vc0, vb0); \
    va0 = _mm512_xor_si512(va0, vb0); \
    _mm512_store_si512((void*) aa->s, va0); \
} while(0)

#elif defined(__AVX2__)

#define uint512_t_mixi(a, b, c) do { \
    uint512_t* const aa = (a); \
    const uint512_t* const bb = (b); \
    const uint512_t* const cc = (c); \
    __m256i vc0 = _mm256_load_si256((__m256i*) cc->s); \
    __m256i vc1 = _mm256_load_si256((__m256i*) &cc->s[4]); \
    __m256i vb0 = _mm256_load_si256((__m256i*) bb->s); \
    __m256i vb1 = _mm256_load_si256((__m256i*) &bb->s[4]); \
    __m256i va0 = _mm256_load_si256((__m256i*) aa->s); \
    __m256i va1 = _mm256_load_si256((__m256i*) &aa->s[4]); \
    va0 = _mm256_and_si256(va0, vc0); \
    va1 = _mm256_and_si256(va1, vc1); \
    vb0 = _mm256_andnot_si256(vc0, vb0); \
    vb1 = _mm256_andnot_si256(vc1, vb1); \
    va0 = _mm256_xor_si256(va0, vb0); \
    va1 = _mm256_xor_si256(va1, vb1); \
    _mm256_store_si256((__m256i*) aa->s, va0); \
    _mm256_store_si256((__m256i*) &aa->s[4], va1); \
} while(0)

#elif defined(__AVX__)

#define uint512_t_mixi(a, b, c) do { \
    uint512_t* const aa = (a); \
    const uint512_t* const bb = (b); \
    const uint512_t* const cc = (c); \
    __m256i vc0 = _mm256_load_si256((__m256i*) cc->s); \
    __m256i vc1 = _mm256_load_si256((__m256i*) &cc->s[4]); \
    __m256i vb0 = _mm256_load_si256((__m256i*) bb->s); \
    __m256i vb1 = _mm256_load_si256((__m256i*) &bb->s[4]); \
    __m256i va0 = _mm256_load_si256((__m256i*) aa->s); \
    __m256i va1 = _mm256_load_si256((__m256i*) &aa->s[4]); \
    va0 = _mm256_castpd_si256(_mm256_and_pd(_mm256_castsi256_pd(va0), \
                                            _mm256_castsi256_pd(vc0))); \
    va1 = _mm256_castpd_si256(_mm256_and_pd(_mm256_castsi256_pd(va1), \
                                            _mm256_castsi256_pd(vc1))); \
    vb0 = _mm256_castpd_si256(_mm256_andnot_pd(_mm256_castsi256_pd(vc0), \
                                               _mm256_castsi256_pd(vb0))); \
    vb1 = _mm256_castpd_si256(_mm256_andnot_pd(_mm256_castsi256_pd(vc1), \
                                               _mm256_castsi256_pd(vb1))); \
    va0 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(va0), \
                                            _mm256_castsi256_pd(vb0))); \
    va1 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(va1), \
                                            _mm256_castsi256_pd(vb1))); \
    _mm256_store_si256((__m256i*) aa->s, va0); \
    _mm256_store_si256((__m256i*) &aa->s[4], va1); \
} while(0)

#else

#define uint512_t_mixi(a, b, c) do { \
    uint512_t* const aa = (a); \
    const uint512_t* const bb = (b); \
    const uint512_t* const cc = (c); \
    aa->s[ 0] &= cc->s[ 0]; \
    aa->s[ 1] &= cc->s[ 1]; \
    aa->s[ 2] &= cc->s[ 2]; \
    aa->s[ 3] &= cc->s[ 3]; \
    aa->s[ 4] &= cc->s[ 4]; \
    aa->s[ 5] &= cc->s[ 5]; \
    aa->s[ 6] &= cc->s[ 6]; \
    aa->s[ 7] &= cc->s[ 7]; \
    aa->s[ 0] ^= bb->s[ 0] & ~cc->s[ 0]; \
    aa->s[ 1] ^= bb->s[ 1] & ~cc->s[ 1]; \
    aa->s[ 2] ^= bb->s[ 2] & ~cc->s[ 2]; \
    aa->s[ 3] ^= bb->s[ 3] & ~cc->s[ 3]; \
    aa->s[ 4] ^= bb->s[ 4] & ~cc->s[ 4]; \
    aa->s[ 5] ^= bb->s[ 5] & ~cc->s[ 5]; \
    aa->s[ 6] ^= bb->s[ 6] & ~cc->s[ 6]; \
    aa->s[ 7] ^= bb->s[ 7] & ~cc->s[ 7]; \
} while(0)

#endif

/* usage: Given 4 uint512_t a, b, c, and d, compute a ^ (b & d) ^ (c & (~d))
 *      and store the result back into a. (in place)
 * params:
 *      1) a: ptr to struct uint512_t
 *      2) b: ptr to struct uint512_t
 *      3) c: ptr to struct uint512_t
 *      4) d: ptr to struct uint512_t
 * return: void */
#if defined(__AVX512F__)

#define uint512_t_xor_mixi(a, b, c, d) do { \
    uint512_t* const aa = (a); \
    const uint512_t* const bb = (b); \
    const uint512_t* const cc = (c); \
    const uint512_t* const dd = (d); \
    __m512i vd0 = _mm512_load_si512((void const*) dd->s); \
    __m512i vc0 = _mm512_load_si512((void const*) cc->s); \
    __m512i vb0 = _mm512_load_si512((void const*) bb->s); \
    __m512i va0 = _mm512_load_si512((void const*) aa->s); \
    vc0 = _mm512_andnot_si512(vd0, vc0); \
    vb0 = _mm512_and_si512(vd0, vb0); \
    vb0 = _mm512_xor_si512(vb0, vc0); \
    va0 = _mm512_xor_si512(va0, vb0); \
    _mm512_store_si512((void*) aa->s, va0); \
} while(0)

#elif defined(__AVX2__)

#define uint512_t_xor_mixi(a, b, c, d) do { \
    uint512_t* const aa = (a); \
    const uint512_t* const bb = (b); \
    const uint512_t* const cc = (c); \
    const uint512_t* const dd = (d); \
    __m256i vd0 = _mm256_load_si256((__m256i*) dd->s); \
    __m256i vd1 = _mm256_load_si256((__m256i*) &dd->s[4]); \
    __m256i vc0 = _mm256_load_si256((__m256i*) cc->s); \
    __m256i vc1 = _mm256_load_si256((__m256i*) &cc->s[4]); \
    __m256i vb0 = _mm256_load_si256((__m256i*) bb->s); \
    __m256i vb1 = _mm256_load_si256((__m256i*) &bb->s[4]); \
    __m256i va0 = _mm256_load_si256((__m256i*) aa->s); \
    __m256i va1 = _mm256_load_si256((__m256i*) &aa->s[4]); \
    vc0 = _mm256_andnot_si256(vd0, vc0); \
    vc1 = _mm256_andnot_si256(vd1, vc1); \
    vb0 = _mm256_and_si256(vd0, vb0); \
    vb1 = _mm256_and_si256(vd1, vb1); \
    vb0 = _mm256_xor_si256(vb0, vc0); \
    vb1 = _mm256_xor_si256(vb1, vc1); \
    va0 = _mm256_xor_si256(va0, vb0); \
    va1 = _mm256_xor_si256(va1, vb1); \
    _mm256_store_si256((__m256i*) aa->s, va0); \
    _mm256_store_si256((__m256i*) &aa->s[4], va1); \
} while(0)

#elif defined(__AVX__)

#define uint512_t_xor_mixi(a, b, c, d) do { \
    uint512_t* const aa = (a); \
    const uint512_t* const bb = (b); \
    const uint512_t* const cc = (c); \
    const uint512_t* const dd = (d); \
    __m256i vd0 = _mm256_load_si256((__m256i*) dd->s); \
    __m256i vd1 = _mm256_load_si256((__m256i*) &dd->s[4]); \
    __m256i vc0 = _mm256_load_si256((__m256i*) cc->s); \
    __m256i vc1 = _mm256_load_si256((__m256i*) &cc->s[4]); \
    __m256i vb0 = _mm256_load_si256((__m256i*) bb->s); \
    __m256i vb1 = _mm256_load_si256((__m256i*) &bb->s[4]); \
    __m256i va0 = _mm256_load_si256((__m256i*) aa->s); \
    __m256i va1 = _mm256_load_si256((__m256i*) &aa->s[4]); \
    vc0 = _mm256_castpd_si256(_mm256_andnot_pd(_mm256_castsi256_pd(vd0), \
                                               _mm256_castsi256_pd(vc0))); \
    vc1 = _mm256_castpd_si256(_mm256_andnot_pd(_mm256_castsi256_pd(vd1), \
                                               _mm256_castsi256_pd(vc1))); \
    vb0 = _mm256_castpd_si256(_mm256_and_pd(_mm256_castsi256_pd(vd0), \
                                            _mm256_castsi256_pd(vb0))); \
    vb1 = _mm256_castpd_si256(_mm256_and_pd(_mm256_castsi256_pd(vd1), \
                                            _mm256_castsi256_pd(vb1))); \
    vb0 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(vb0), \
                                            _mm256_castsi256_pd(vc0))); \
    vb1 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(vb1), \
                                            _mm256_castsi256_pd(vc1))); \
    va0 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(va0), \
                                            _mm256_castsi256_pd(vb0))); \
    va1 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(va1), \
                                            _mm256_castsi256_pd(vb1))); \
    _mm256_store_si256((__m256i*) aa->s, va0); \
    _mm256_store_si256((__m256i*) &aa->s[4], va1); \
} while(0)

#else

#define uint512_t_xor_mixi(a, b, c, d) do { \
    uint512_t* const aa = (a); \
    const uint512_t* const bb = (b); \
    const uint512_t* const cc = (c); \
    const uint512_t* const dd = (d); \
    aa->s[ 0] ^= bb->s[ 0] & dd->s[ 0]; \
    aa->s[ 1] ^= bb->s[ 1] & dd->s[ 1]; \
    aa->s[ 2] ^= bb->s[ 2] & dd->s[ 2]; \
    aa->s[ 3] ^= bb->s[ 3] & dd->s[ 3]; \
    aa->s[ 4] ^= bb->s[ 4] & dd->s[ 4]; \
    aa->s[ 5] ^= bb->s[ 5] & dd->s[ 5]; \
    aa->s[ 6] ^= bb->s[ 6] & dd->s[ 6]; \
    aa->s[ 7] ^= bb->s[ 7] & dd->s[ 7]; \
    aa->s[ 0] ^= cc->s[ 0] & ~dd->s[ 0]; \
    aa->s[ 1] ^= cc->s[ 1] & ~dd->s[ 1]; \
    aa->s[ 2] ^= cc->s[ 2] & ~dd->s[ 2]; \
    aa->s[ 3] ^= cc->s[ 3] & ~dd->s[ 3]; \
    aa->s[ 4] ^= cc->s[ 4] & ~dd->s[ 4]; \
    aa->s[ 5] ^= cc->s[ 5] & ~dd->s[ 5]; \
    aa->s[ 6] ^= cc->s[ 6] & ~dd->s[ 6]; \
    aa->s[ 7] ^= cc->s[ 7] & ~dd->s[ 7]; \
} while(0)

#endif

/* usage: Given 2 uint512_t a and b, compute a&b and store the result into p
 * params:
 *      1) p: ptr to struct uint512_t
 *      2) a: ptr to struct uint512_t
 *      3) b: ptr to struct uint512_t
 * return: void */
#if defined(__AVX512F__)

#define uint512_t_and(p, a, b) do { \
    uint512_t* const pp = (p); \
    const uint512_t* const aa = (a); \
    const uint512_t* const bb = (b); \
    __m512i va0 = _mm512_load_si512((void const*) aa->s); \
    __m512i vb0 = _mm512_load_si512((void const*) bb->s); \
    va0 = _mm512_and_si512(vb0, va0); \
    _mm512_store_si512((void*) pp->s, va0); \
} while(0)

#elif defined(__AVX2__)

#define uint512_t_and(p, a, b) do { \
    uint512_t* const pp = (p); \
    const uint512_t* const aa = (a); \
    const uint512_t* const bb = (b); \
    __m256i va0 = _mm256_load_si256((__m256i*) aa->s); \
    __m256i va1 = _mm256_load_si256((__m256i*) &aa->s[4]); \
    __m256i vb0 = _mm256_load_si256((__m256i*) bb->s); \
    __m256i vb1 = _mm256_load_si256((__m256i*) &bb->s[4]); \
    va0 = _mm256_and_si256(vb0, va0); \
    va1 = _mm256_and_si256(vb1, va1); \
    _mm256_store_si256((__m256i*) pp->s, va0); \
    _mm256_store_si256((__m256i*) &pp->s[4], va1); \
} while(0)

#elif defined(__AVX__)

#define uint512_t_and(p, a, b) do { \
    uint512_t* const pp = (p); \
    const uint512_t* const aa = (a); \
    const uint512_t* const bb = (b); \
    __m256i va0 = _mm256_load_si256((__m256i*) aa->s); \
    __m256i va1 = _mm256_load_si256((__m256i*) &aa->s[4]); \
    __m256i vb0 = _mm256_load_si256((__m256i*) bb->s); \
    __m256i vb1 = _mm256_load_si256((__m256i*) &bb->s[4]); \
    va0 = _mm256_castpd_si256(_mm256_and_pd(_mm256_castsi256_pd(vb0), \
                                            _mm256_castsi256_pd(va0))); \
    va1 = _mm256_castpd_si256(_mm256_and_pd(_mm256_castsi256_pd(vb1), \
                                            _mm256_castsi256_pd(va1))); \
    _mm256_store_si256((__m256i*) pp->s, va0); \
    _mm256_store_si256((__m256i*) &pp->s[4], va1); \
} while(0)

#else

#define uint512_t_and(p, a, b) do { \
    uint512_t* const pp = (p); \
    const uint512_t* const aa = (a); \
    const uint512_t* const bb = (b); \
    pp->s[ 0] = aa->s[ 0] & bb->s[ 0]; \
    pp->s[ 1] = aa->s[ 1] & bb->s[ 1]; \
    pp->s[ 2] = aa->s[ 2] & bb->s[ 2]; \
    pp->s[ 3] = aa->s[ 3] & bb->s[ 3]; \
    pp->s[ 4] = aa->s[ 4] & bb->s[ 4]; \
    pp->s[ 5] = aa->s[ 5] & bb->s[ 5]; \
    pp->s[ 6] = aa->s[ 6] & bb->s[ 6]; \
    pp->s[ 7] = aa->s[ 7] & bb->s[ 7]; \
} while(0)

#endif

/* usage: Given 2 uint512_t a and b, compute a&b and store the result into a.
 *      (in place and)
 * params:
 *      1) a: ptr to struct uint512_t
 *      2) b: ptr to struct uint512_t
 * return: void */
#if defined(__AVX512F__)

#define uint512_t_andi(a, b) do { \
    uint512_t* const aa = (a); \
    const uint512_t* const bb = (b); \
    __m512i va0 = _mm512_load_si512((void const*) aa->s); \
    __m512i vb0 = _mm512_load_si512((void const*) bb->s); \
    va0 = _mm512_and_si512(vb0, va0); \
    _mm512_store_si512((void*) aa->s, va0); \
} while(0)

#elif defined(__AVX2__)

#define uint512_t_andi(a, b) do { \
    uint512_t* const aa = (a); \
    const uint512_t* const bb = (b); \
    __m256i va0 = _mm256_load_si256((__m256i*) aa->s); \
    __m256i va1 = _mm256_load_si256((__m256i*) &aa->s[4]); \
    __m256i vb0 = _mm256_load_si256((__m256i*) bb->s); \
    __m256i vb1 = _mm256_load_si256((__m256i*) &bb->s[4]); \
    va0 = _mm256_and_si256(vb0, va0); \
    va1 = _mm256_and_si256(vb1, va1); \
    _mm256_store_si256((__m256i*) aa->s, va0); \
    _mm256_store_si256((__m256i*) &aa->s[4], va1); \
} while(0)

#elif defined(__AVX__)

#define uint512_t_andi(a, b) do { \
    uint512_t* const aa = (a); \
    const uint512_t* const bb = (b); \
    __m256i va0 = _mm256_load_si256((__m256i*) aa->s); \
    __m256i va1 = _mm256_load_si256((__m256i*) &aa->s[4]); \
    __m256i vb0 = _mm256_load_si256((__m256i*) bb->s); \
    __m256i vb1 = _mm256_load_si256((__m256i*) &bb->s[4]); \
    va0 = _mm256_castpd_si256(_mm256_and_pd(_mm256_castsi256_pd(vb0), \
                                            _mm256_castsi256_pd(va0))); \
    va1 = _mm256_castpd_si256(_mm256_and_pd(_mm256_castsi256_pd(vb1), \
                                            _mm256_castsi256_pd(va1))); \
    _mm256_store_si256((__m256i*) aa->s, va0); \
    _mm256_store_si256((__m256i*) &aa->s[4], va1); \
} while(0)

#else

#define uint512_t_andi(a, b) do { \
    uint512_t* const aa = (a); \
    const uint512_t* const bb = (b); \
    aa->s[ 0] &= bb->s[ 0]; \
    aa->s[ 1] &= bb->s[ 1]; \
    aa->s[ 2] &= bb->s[ 2]; \
    aa->s[ 3] &= bb->s[ 3]; \
    aa->s[ 4] &= bb->s[ 4]; \
    aa->s[ 5] &= bb->s[ 5]; \
    aa->s[ 6] &= bb->s[ 6]; \
    aa->s[ 7] &= bb->s[ 7]; \
} while(0)

#endif

/* usage: Given 2 uint512_t a and b, compute a&~b and store the result into p
 * params:
 *      1) p: ptr to struct uint512_t
 *      2) a: ptr to struct uint512_t
 *      3) b: ptr to struct uint512_t
 * return: void */
#if defined(__AVX512F__)

#define uint512_t_andn(p, a, b) do { \
    uint512_t* const pp = (p); \
    const uint512_t* const aa = (a); \
    const uint512_t* const bb = (b); \
    __m512i va0 = _mm512_load_si512((void const*) aa->s); \
    __m512i vb0 = _mm512_load_si512((void const*) bb->s); \
    va0 = _mm512_andnot_si512(vb0, va0); \
    _mm512_store_si512((void*) pp->s, va0); \
} while(0)

#elif defined(__AVX2__)

#define uint512_t_andn(p, a, b) do { \
    uint512_t* const pp = (p); \
    const uint512_t* const aa = (a); \
    const uint512_t* const bb = (b); \
    __m256i va0 = _mm256_load_si256((__m256i*) aa->s); \
    __m256i va1 = _mm256_load_si256((__m256i*) &aa->s[4]); \
    __m256i vb0 = _mm256_load_si256((__m256i*) bb->s); \
    __m256i vb1 = _mm256_load_si256((__m256i*) &bb->s[4]); \
    va0 = _mm256_andnot_si256(vb0, va0); \
    va1 = _mm256_andnot_si256(vb1, va1); \
    _mm256_store_si256((__m256i*) pp->s, va0); \
    _mm256_store_si256((__m256i*) &pp->s[4], va1); \
} while(0)

#elif defined(__AVX__)

#define uint512_t_andn(p, a, b) do { \
    uint512_t* const pp = (p); \
    const uint512_t* const aa = (a); \
    const uint512_t* const bb = (b); \
    __m256i va0 = _mm256_load_si256((__m256i*) aa->s); \
    __m256i va1 = _mm256_load_si256((__m256i*) &aa->s[4]); \
    __m256i vb0 = _mm256_load_si256((__m256i*) bb->s); \
    __m256i vb1 = _mm256_load_si256((__m256i*) &bb->s[4]); \
    va0 = _mm256_castpd_si256(_mm256_andnot_pd(_mm256_castsi256_pd(vb0), \
                                               _mm256_castsi256_pd(va0))); \
    va1 = _mm256_castpd_si256(_mm256_andnot_pd(_mm256_castsi256_pd(vb1), \
                                               _mm256_castsi256_pd(va1))); \
    _mm256_store_si256((__m256i*) pp->s, va0); \
    _mm256_store_si256((__m256i*) &pp->s[4], va1); \
} while(0)

#else

#define uint512_t_andn(p, a, b) do { \
    uint512_t* const pp = (p); \
    const uint512_t* const aa = (a); \
    const uint512_t* const bb = (b); \
    pp->s[ 0] = aa->s[ 0] & ~bb->s[ 0]; \
    pp->s[ 1] = aa->s[ 1] & ~bb->s[ 1]; \
    pp->s[ 2] = aa->s[ 2] & ~bb->s[ 2]; \
    pp->s[ 3] = aa->s[ 3] & ~bb->s[ 3]; \
    pp->s[ 4] = aa->s[ 4] & ~bb->s[ 4]; \
    pp->s[ 5] = aa->s[ 5] & ~bb->s[ 5]; \
    pp->s[ 6] = aa->s[ 6] & ~bb->s[ 6]; \
    pp->s[ 7] = aa->s[ 7] & ~bb->s[ 7]; \
} while(0)

#endif

/* usage: Given 2 uint512_t a and b, compute a&~b and store the result into a
 *      (in place negate-and).
 * params:
 *      1) a: ptr to struct uint512_t
 *      2) b: ptr to struct uint512_t
 * return: void */
#if defined(__AVX512F__)

#define uint512_t_andni(a, b) do { \
    uint512_t* const aa = (a); \
    const uint512_t* const bb = (b); \
    __m512i va0 = _mm512_load_si512((void const*) aa->s); \
    __m512i vb0 = _mm512_load_si512((void const*) bb->s); \
    va0 = _mm512_andnot_si512(vb0, va0); \
    _mm512_store_si512((void*) aa->s, va0); \
} while(0)

#elif defined(__AVX2__)

#define uint512_t_andni(a, b) do { \
    uint512_t* const aa = (a); \
    const uint512_t* const bb = (b); \
    __m256i va0 = _mm256_load_si256((__m256i*) aa->s); \
    __m256i va1 = _mm256_load_si256((__m256i*) &aa->s[4]); \
    __m256i vb0 = _mm256_load_si256((__m256i*) bb->s); \
    __m256i vb1 = _mm256_load_si256((__m256i*) &bb->s[4]); \
    va0 = _mm256_andnot_si256(vb0, va0); \
    va1 = _mm256_andnot_si256(vb1, va1); \
    _mm256_store_si256((__m256i*) aa->s, va0); \
    _mm256_store_si256((__m256i*) &aa->s[4], va1); \
} while(0)

#elif defined(__AVX__)

#define uint512_t_andni(a, b) do { \
    uint512_t* const aa = (a); \
    const uint512_t* const bb = (b); \
    __m256i va0 = _mm256_load_si256((__m256i*) aa->s); \
    __m256i va1 = _mm256_load_si256((__m256i*) &aa->s[4]); \
    __m256i vb0 = _mm256_load_si256((__m256i*) bb->s); \
    __m256i vb1 = _mm256_load_si256((__m256i*) &bb->s[4]); \
    va0 = _mm256_castpd_si256(_mm256_andnot_pd(_mm256_castsi256_pd(vb0), \
                                               _mm256_castsi256_pd(va0))); \
    va1 = _mm256_castpd_si256(_mm256_andnot_pd(_mm256_castsi256_pd(vb1), \
                                               _mm256_castsi256_pd(va1))); \
    _mm256_store_si256((__m256i*) aa->s, va0); \
    _mm256_store_si256((__m256i*) &aa->s[4], va1); \
} while(0)

#else

#define uint512_t_andni(a, b) do { \
    uint512_t* const aa = (a); \
    const uint512_t* const bb = (b); \
    aa->s[ 0] &= ~bb->s[ 0]; \
    aa->s[ 1] &= ~bb->s[ 1]; \
    aa->s[ 2] &= ~bb->s[ 2]; \
    aa->s[ 3] &= ~bb->s[ 3]; \
    aa->s[ 4] &= ~bb->s[ 4]; \
    aa->s[ 5] &= ~bb->s[ 5]; \
    aa->s[ 6] &= ~bb->s[ 6]; \
    aa->s[ 7] &= ~bb->s[ 7]; \
} while(0)

#endif

/* usage: Given 2 uint512_t a and b, compute a|b and store the result into c.
 * params:
 *      1) c: ptr to struct uint512_t
 *      1) a: ptr to struct uint512_t
 *      2) b: ptr to struct uint512_t
 * return: void */
#if defined(__AVX512F__)

#define uint512_t_or(c, a, b) do { \
    uint512_t* const cc = (c); \
    const uint512_t* const aa = (a); \
    const uint512_t* const bb = (b); \
    __m512i va0 = _mm512_load_si512((void const*) aa->s); \
    __m512i vb0 = _mm512_load_si512((void const*) bb->s); \
    va0 = _mm512_or_si512(vb0, va0); \
    _mm512_store_si512((void*) cc->s, va0); \
} while(0)

#elif defined(__AVX2__)

#define uint512_t_or(c, a, b) do { \
    uint512_t* const cc = (c); \
    const uint512_t* const aa = (a); \
    const uint512_t* const bb = (b); \
    __m256i va0 = _mm256_load_si256((__m256i*) aa->s); \
    __m256i va1 = _mm256_load_si256((__m256i*) &aa->s[4]); \
    __m256i vb0 = _mm256_load_si256((__m256i*) bb->s); \
    __m256i vb1 = _mm256_load_si256((__m256i*) &bb->s[4]); \
    va0 = _mm256_or_si256(va0, vb0); \
    va1 = _mm256_or_si256(va1, vb1); \
    _mm256_store_si256((__m256i*) cc->s, va0); \
    _mm256_store_si256((__m256i*) &cc->s[4], va1); \
} while(0)

#elif defined(__AVX__)

#define uint512_t_or(c, a, b) do { \
    uint512_t* const cc = (c); \
    const uint512_t* const aa = (a); \
    const uint512_t* const bb = (b); \
    __m256i va0 = _mm256_load_si256((__m256i*) aa->s); \
    __m256i va1 = _mm256_load_si256((__m256i*) &aa->s[4]); \
    __m256i vb0 = _mm256_load_si256((__m256i*) bb->s); \
    __m256i vb1 = _mm256_load_si256((__m256i*) &bb->s[4]); \
    va0 = _mm256_castpd_si256(_mm256_or_pd(_mm256_castsi256_pd(va0), \
                                           _mm256_castsi256_pd(vb0))); \
    va1 = _mm256_castpd_si256(_mm256_or_pd(_mm256_castsi256_pd(va1), \
                                           _mm256_castsi256_pd(vb1))); \
    _mm256_store_si256((__m256i*) cc->s, va0); \
    _mm256_store_si256((__m256i*) &cc->s[4], va1); \
} while(0)

#else

#define uint512_t_or(c, a, b) do { \
    uint512_t* const cc = (c); \
    const uint512_t* const aa = (a); \
    const uint512_t* const bb = (b); \
    cc->s[ 0] = aa->s[ 0] | bb->s[ 0]; \
    cc->s[ 1] = aa->s[ 1] | bb->s[ 1]; \
    cc->s[ 2] = aa->s[ 2] | bb->s[ 2]; \
    cc->s[ 3] = aa->s[ 3] | bb->s[ 3]; \
    cc->s[ 4] = aa->s[ 4] | bb->s[ 4]; \
    cc->s[ 5] = aa->s[ 5] | bb->s[ 5]; \
    cc->s[ 6] = aa->s[ 6] | bb->s[ 6]; \
    cc->s[ 7] = aa->s[ 7] | bb->s[ 7]; \
} while(0)

#endif

/* usage: Given 2 uint512_t a and b, compute a|b and store the result into a.
 *      (in place or)
 * params:
 *      1) a: ptr to struct uint512_t
 *      2) b: ptr to struct uint512_t
 * return: void */
#if defined(__AVX512F__)

#define uint512_t_ori(a, b) do { \
    uint512_t* const aa = (a); \
    const uint512_t* const bb = (b); \
    __m512i va0 = _mm512_load_si512((void const*) aa->s); \
    __m512i vb0 = _mm512_load_si512((void const*) bb->s); \
    va0 = _mm512_or_si512(vb0, va0); \
    _mm512_store_si512((void*) aa->s, va0); \
} while(0)

#elif defined(__AVX2__)

#define uint512_t_ori(a, b) do { \
    uint512_t* const aa = (a); \
    const uint512_t* const bb = (b); \
    __m256i va0 = _mm256_load_si256((__m256i*) aa->s); \
    __m256i va1 = _mm256_load_si256((__m256i*) &aa->s[4]); \
    __m256i vb0 = _mm256_load_si256((__m256i*) bb->s); \
    __m256i vb1 = _mm256_load_si256((__m256i*) &bb->s[4]); \
    va0 = _mm256_or_si256(va0, vb0); \
    va1 = _mm256_or_si256(va1, vb1); \
    _mm256_store_si256((__m256i*) aa->s, va0); \
    _mm256_store_si256((__m256i*) &aa->s[4], va1); \
} while(0)

#elif defined(__AVX__)

#define uint512_t_ori(a, b) do { \
    uint512_t* const aa = (a); \
    const uint512_t* const bb = (b); \
    __m256i va0 = _mm256_load_si256((__m256i*) aa->s); \
    __m256i va1 = _mm256_load_si256((__m256i*) &aa->s[4]); \
    __m256i vb0 = _mm256_load_si256((__m256i*) bb->s); \
    __m256i vb1 = _mm256_load_si256((__m256i*) &bb->s[4]); \
    va0 = _mm256_castpd_si256(_mm256_or_pd(_mm256_castsi256_pd(va0), \
                                           _mm256_castsi256_pd(vb0))); \
    va1 = _mm256_castpd_si256(_mm256_or_pd(_mm256_castsi256_pd(va1), \
                                           _mm256_castsi256_pd(vb1))); \
    _mm256_store_si256((__m256i*) aa->s, va0); \
    _mm256_store_si256((__m256i*) &aa->s[4], va1); \
} while(0)


#else

#define uint512_t_ori(a, b) do { \
    uint512_t* const aa = (a); \
    const uint512_t* const bb = (b); \
    aa->s[ 0] |= bb->s[ 0]; \
    aa->s[ 1] |= bb->s[ 1]; \
    aa->s[ 2] |= bb->s[ 2]; \
    aa->s[ 3] |= bb->s[ 3]; \
    aa->s[ 4] |= bb->s[ 4]; \
    aa->s[ 5] |= bb->s[ 5]; \
    aa->s[ 6] |= bb->s[ 6]; \
    aa->s[ 7] |= bb->s[ 7]; \
} while(0)

#endif

/* usage: Given 1 uint512_t a, and an integer i, return the i-th bit.
 *      The LSB is 0-th bit.
 * params:
 *      1) a: ptr to struct uint512_t
 *      2) i: index to the bit, from 0 ~ 511. Do not perform arithmetic here.
 * return: the bit */
static force_inline uint64_t
uint512_t_at(const uint512_t* a, uint32_t i) {
    return (a->s[i >> 6] >> (i & 0x3FULL)) & 0x1ULL;
}

/* usage: Given 1 uint512_t a, an integer i, set the i-th bit to the given
 *      value. The LSB is 0-th bit.
 * params:
 *      1) a: ptr to struct uint512_t
 *      2) i: index to the bit, from 0 ~ 511
 *      3) v: true if the bit is 1, otherwise false
 * return: void */
#define uint512_t_set_at(a, i, v) do { \
    uint512_t* const aa = (a); \
    uint64_t const ii = (i); \
    aa->s[ii >> 6] &= ~(0x1ULL << (ii & 0x3FULL)); \
    aa->s[ii >> 6] |= ((uint64_t) (v)) << (ii & 0x3FULL); \
} while(0)

/* usage: Given 1 uint512_t a, an integer i, toggle the i-th bit.
 *      The LSB is 0-th bit.
 * params:
 *      1) a: ptr to struct uint512_t
 *      2) i: index to the bit, from 0 ~ 511
 * return: void */
#define uint512_t_toggle_at(a, i) do { \
    uint512_t* const aa = (a); \
    uint64_t const ii = (i); \
    aa->s[ii >> 6] ^= 0x1ULL << (ii & 0x3FULL); \
} while(0)

/* usage: Given 1 uint512_t a, an integer i, set the i-th bit to 0.
 *      The LSB is 0-th bit.
 * params:
 *      1) a: ptr to struct uint512_t
 *      2) i: index to the bit, from 0 ~ 511
 * return: void */
#define uint512_t_clear_at(a, i) do { \
    uint64_t const ii = (i); \
    (a)->s[ii >> 6] &= ~(0x1ULL << (ii & 0x3FULL)); \
} while(0)

/* usage: Given 1 uint512_t a, find indices of all set bits in a
 * params:
 *      1) a: ptr to struct uint512_t
 *      2) res: an uint16_t array for storing the indices, must hold at least
 *              512 elements
 * return: the number of set bits */
static inline int
uint512_t_sbpos(const uint512_t* const restrict a,
                uint16_t* const restrict res) {
    uint64_t base = 0x0ULL;
    const uint64_t inc64 = 0x0040004000400040ULL;
    int sbnum = sbidx_in_64b_sz16(res, base, uint512_t_64b_at(a, 0));
    base += inc64;
    sbnum += sbidx_in_64b_sz16(res + sbnum, base, uint512_t_64b_at(a, 1));
    base += inc64;
    sbnum += sbidx_in_64b_sz16(res + sbnum, base, uint512_t_64b_at(a, 2));
    base += inc64;
    sbnum += sbidx_in_64b_sz16(res + sbnum, base, uint512_t_64b_at(a, 3));
    base += inc64;
    sbnum += sbidx_in_64b_sz16(res + sbnum, base, uint512_t_64b_at(a, 4));
    base += inc64;
    sbnum += sbidx_in_64b_sz16(res + sbnum, base, uint512_t_64b_at(a, 5));
    base += inc64;
    sbnum += sbidx_in_64b_sz16(res + sbnum, base, uint512_t_64b_at(a, 6));
    base += inc64;
    sbnum += sbidx_in_64b_sz16(res + sbnum, base, uint512_t_64b_at(a, 7));
    assert(sbnum <= 512);
    return sbnum;
}

#if defined(__AVX512F__)

/* usage: Given 1 uint512_t a whose content is stored in 1 __m512i register,
 *      find indices of all set bits in a
 * params:
 *      1) res: an uint16_t array for storing the indices, must hold at least
 *              512 elements
 *      2) va: register that store the content of uint512_t a
 * return: the number of set bits */
static force_inline int
uint512_t_sbpos_from_reg_avx512(uint16_t* const res, __m512i va) {
    uint64_t base = 0x0ULL;
    const uint64_t inc64 = 0x0040004000400040ULL;
    __m256i v0 = _mm512_extracti64x4_epi64(va, 0);
    __m256i v1 = _mm512_extracti64x4_epi64(va, 1);
    int sbnum = sbidx_in_64b_sz16(res, base, _mm256_extract_epi64(v0, 0));
    base += inc64;
    sbnum += sbidx_in_64b_sz16(res + sbnum, base, _mm256_extract_epi64(v0, 1));
    base += inc64;
    sbnum += sbidx_in_64b_sz16(res + sbnum, base, _mm256_extract_epi64(v0, 2));
    base += inc64;
    sbnum += sbidx_in_64b_sz16(res + sbnum, base, _mm256_extract_epi64(v0, 3));
    base += inc64;
    sbnum += sbidx_in_64b_sz16(res + sbnum, base, _mm256_extract_epi64(v1, 0));
    base += inc64;
    sbnum += sbidx_in_64b_sz16(res + sbnum, base, _mm256_extract_epi64(v1, 1));
    base += inc64;
    sbnum += sbidx_in_64b_sz16(res + sbnum, base, _mm256_extract_epi64(v1, 2));
    base += inc64;
    sbnum += sbidx_in_64b_sz16(res + sbnum, base, _mm256_extract_epi64(v1, 3));
    assert(sbnum <= 512);
    return sbnum;
}

#endif

#if defined(__AVX__)

/* usage: Given 1 uint512_t a whose content is stored in 2 __m256i registers,
 *      find indices of all set bits in a
 * params:
 *      1) res: an uint16_t array for storing the indices, must hold at least
 *              512 elements
 *      2) v0: register that store the 1st 256 bits of uint512_t a
 *      3) v1: register that store the 2nd 256 bits of uint512_t a
 * return: the number of set bits */
static force_inline int
uint512_t_sbpos_from_reg(uint16_t* const res, __m256i v0, __m256i v1) {
    uint64_t base = 0x0ULL;
    const uint64_t inc64 = 0x0040004000400040ULL;
    int sbnum = sbidx_in_64b_sz16(res, base, _mm256_extract_epi64(v0, 0));
    base += inc64;
    sbnum += sbidx_in_64b_sz16(res + sbnum, base, _mm256_extract_epi64(v0, 1));
    base += inc64;
    sbnum += sbidx_in_64b_sz16(res + sbnum, base, _mm256_extract_epi64(v0, 2));
    base += inc64;
    sbnum += sbidx_in_64b_sz16(res + sbnum, base, _mm256_extract_epi64(v0, 3));
    base += inc64;
    sbnum += sbidx_in_64b_sz16(res + sbnum, base, _mm256_extract_epi64(v1, 0));
    base += inc64;
    sbnum += sbidx_in_64b_sz16(res + sbnum, base, _mm256_extract_epi64(v1, 1));
    base += inc64;
    sbnum += sbidx_in_64b_sz16(res + sbnum, base, _mm256_extract_epi64(v1, 2));
    base += inc64;
    sbnum += sbidx_in_64b_sz16(res + sbnum, base, _mm256_extract_epi64(v1, 3));
    assert(sbnum <= 512);
    return sbnum;
}

#endif

#endif /* __BLK_LANCZOS_UINT512_T_H__ */
