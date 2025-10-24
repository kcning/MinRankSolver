/* uint1024_t.h: header file for struct uint1024_t */

#ifndef __BLK_LANCZOS_UINT1024_T_H__
#define __BLK_LANCZOS_UINT1024_T_H__

#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <stdalign.h>

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

#include "bitmap_table.h"

/* ========================================================================
 * struct uint1024_t definition
 * ======================================================================== */

typedef struct {
    alignas(64) uint64_t s[16];
} uint1024_t;

/* ========================================================================
 * functions prototypes
 * ======================================================================== */

/* usage: Given 1 uint1024_t a, treat it as uint64_t[16] and
 *      return the specified 64 bits.
 * params:
 *      1) a: ptr to struct uint1024_t
 *      2) i: index of the slot; from 0 ~ 15
 * return: the specified 64-bit as a uint64_t */
#define uint1024_t_64b_at(a, i) \
    ((a)->s[(i)])

/* usage: Given 1 uint1024_t a, set a to all 0's.
 * params:
 *      1) a: ptr to struct uint1024_t
 * return: void */
#define uint1024_t_zero(a) do { \
    memset((a), 0x0, sizeof(uint1024_t)); \
} while(0)

/* usage: Given 1 uint1024_t a, return false if all 0's. Otherwise true;
 * params:
 *      1) a: ptr to struct uint1024_t
 * return: void */
static inline bool
uint1024_t_is_not_zero(const uint1024_t* const a) {
    return (a->s[ 0] || a->s[ 1] || a->s[ 2] || a->s[ 3] ||
            a->s[ 4] || a->s[ 5] || a->s[ 6] || a->s[ 7] ||
            a->s[ 8] || a->s[ 9] || a->s[10] || a->s[11] ||
            a->s[12] || a->s[13] || a->s[14] || a->s[15] );
}

/* usage: Given 2 uint1024_t a and b, check if they are the same.
 * params:
 *      1) a: ptr to struct uint1024_t
 *      2) b: ptr to struct uint1024_t
 * return: true if they are the same. otherwise false */
static inline bool
uint1024_t_equal(const uint1024_t* const a, const uint1024_t* const b) {
    return (0 == memcmp(a, b, sizeof(uint1024_t)));
}

/* usage: Given 1 uint1024_t a, bitwise negate it.
 * params:
 *      1) a: ptr to struct uint1024_t
 * return: void */
#if defined(__AVX512F__)

#define uint1024_t_negi(a) do { \
    uint1024_t* const aa = (a); \
    __m512i v0 = _mm512_load_si512((void const*) aa->s); \
    __m512i v1 = _mm512_load_si512((void const*) &aa->s[8]); \
    _mm512_store_si512((void*) aa->s, ~v0); \
    _mm512_store_si512((void*) &aa->s[8], ~v1); \
} while(0)

#elif defined(__AVX2__)

#define uint1024_t_negi(a) do { \
    uint1024_t* const aa = (a); \
    __m256i v0 = _mm256_load_si256((__m256i*) aa->s); \
    __m256i v1 = _mm256_load_si256((__m256i*) &aa->s[4]); \
    __m256i v2 = _mm256_load_si256((__m256i*) &aa->s[8]); \
    __m256i v3 = _mm256_load_si256((__m256i*) &aa->s[12]); \
    __m256i all1s = _mm256_set1_epi32(-1); \
    _mm256_store_si256((__m256i*) aa->s, _mm256_xor_si256(v0, all1s)); \
    _mm256_store_si256((__m256i*) &aa->s[4], _mm256_xor_si256(v1, all1s)); \
    _mm256_store_si256((__m256i*) &aa->s[8], _mm256_xor_si256(v2, all1s)); \
    _mm256_store_si256((__m256i*) &aa->s[12], _mm256_xor_si256(v3, all1s)); \
} while(0)

#elif defined(__AVX__)

#define uint1024_t_negi(a) do { \
    uint1024_t* const aa = (a); \
    __m256i v0 = _mm256_load_si256((__m256i*) aa->s); \
    __m256i v1 = _mm256_load_si256((__m256i*) &aa->s[4]); \
    __m256i v2 = _mm256_load_si256((__m256i*) &aa->s[8]); \
    __m256i v3 = _mm256_load_si256((__m256i*) &aa->s[12]); \
    __m256i all1s = _mm256_set1_epi32(-1); \
    v0 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(v0), \
                                           _mm256_castsi256_pd(all1s))); \
    _mm256_store_si256((__m256i*) aa->s, v0); \
    v1 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(v1), \
                                           _mm256_castsi256_pd(all1s))); \
    _mm256_store_si256((__m256i*) &aa->s[4], v1); \
    v2 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(v2), \
                                           _mm256_castsi256_pd(all1s))); \
    _mm256_store_si256((__m256i*) &aa->s[8], v2); \
    v3 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(v3), \
                                           _mm256_castsi256_pd(all1s))); \
    _mm256_store_si256((__m256i*) &aa->s[12], v3); \
} while(0)

#else

#define uint1024_t_negi(a) do { \
    uint1024_t* const aa = (a); \
    aa->s[ 0] = ~aa->s[ 0]; \
    aa->s[ 1] = ~aa->s[ 1]; \
    aa->s[ 2] = ~aa->s[ 2]; \
    aa->s[ 3] = ~aa->s[ 3]; \
    aa->s[ 4] = ~aa->s[ 4]; \
    aa->s[ 5] = ~aa->s[ 5]; \
    aa->s[ 6] = ~aa->s[ 6]; \
    aa->s[ 7] = ~aa->s[ 7]; \
    aa->s[ 8] = ~aa->s[ 8]; \
    aa->s[ 9] = ~aa->s[ 9]; \
    aa->s[10] = ~aa->s[10]; \
    aa->s[11] = ~aa->s[11]; \
    aa->s[12] = ~aa->s[12]; \
    aa->s[13] = ~aa->s[13]; \
    aa->s[14] = ~aa->s[14]; \
    aa->s[15] = ~aa->s[15]; \
} while(0)

#endif

/* usage: Given 1 uint1024_t a, return the number of set bits in a
 * params:
 *      1) a: ptr to struct uint1024_t
 * return: the number of 1's in the uint1024_t */
static inline uint64_t
uint1024_t_popcount(const uint1024_t* const a) {
    return (__builtin_popcountll(a->s[ 0]) + __builtin_popcountll(a->s[ 1]) +
            __builtin_popcountll(a->s[ 2]) + __builtin_popcountll(a->s[ 3]) +
            __builtin_popcountll(a->s[ 4]) + __builtin_popcountll(a->s[ 5]) +
            __builtin_popcountll(a->s[ 6]) + __builtin_popcountll(a->s[ 7]) +
            __builtin_popcountll(a->s[ 8]) + __builtin_popcountll(a->s[ 9]) +
            __builtin_popcountll(a->s[10]) + __builtin_popcountll(a->s[11]) +
            __builtin_popcountll(a->s[12]) + __builtin_popcountll(a->s[13]) +
            __builtin_popcountll(a->s[14]) + __builtin_popcountll(a->s[15]) );
}

/* usage: Given 1 uint1024_t a, set a to a random value.
 * params:
 *      1) a: ptr to struct uint1024_t
 * return: void */
#define uint1024_t_rand(a) do { \
    int* const buf = (int*) ((a)->s); \
    buf[ 0] = rand(); \
    buf[ 1] = rand(); \
    buf[ 2] = rand(); \
    buf[ 3] = rand(); \
    buf[ 4] = rand(); \
    buf[ 5] = rand(); \
    buf[ 6] = rand(); \
    buf[ 7] = rand(); \
    buf[ 8] = rand(); \
    buf[ 9] = rand(); \
    buf[10] = rand(); \
    buf[11] = rand(); \
    buf[12] = rand(); \
    buf[13] = rand(); \
    buf[14] = rand(); \
    buf[15] = rand(); \
    buf[16] = rand(); \
    buf[17] = rand(); \
    buf[18] = rand(); \
    buf[19] = rand(); \
    buf[20] = rand(); \
    buf[21] = rand(); \
    buf[22] = rand(); \
    buf[23] = rand(); \
    buf[24] = rand(); \
    buf[25] = rand(); \
    buf[26] = rand(); \
    buf[27] = rand(); \
    buf[28] = rand(); \
    buf[29] = rand(); \
    buf[30] = rand(); \
    buf[31] = rand(); \
} while(0)

/* usage: Given 2 uint1024_t a and b, copy b into a.
 * params:
 *      1) a: ptr to struct uint1024_t
 *      2) b: ptr to struct uint1024_t
 * return: void */
#define uint1024_t_copy(a, b) do { \
    memcpy((a), (b), sizeof(uint1024_t)); \
} while(0)

/* usage: Given 2 uint1024_t a and b, copy b into a and then negate a bitwise.
 * params:
 *      1) a: ptr to struct uint1024_t
 *      2) b: ptr to struct uint1024_t
 * return: void */
#if defined(__AVX512F__)

#define uint1024_t_neg(a, b) do { \
    uint1024_t* const aa = (a); \
    const uint1024_t* const bb = (b); \
    __m512i v0 = _mm512_load_si512((void const*) bb->s); \
    __m512i v1 = _mm512_load_si512((void const*) &bb->s[8]); \
    _mm512_store_si512((void*) aa->s, ~v0); \
    _mm512_store_si512((void*) &aa->s[8], ~v1); \
} while(0)

#elif defined(__AVX2__)

#define uint1024_t_neg(a, b) do { \
    uint1024_t* const aa = (a); \
    const uint1024_t* const bb = (b); \
    __m256i v0 = _mm256_load_si256((__m256i*) bb->s); \
    __m256i v1 = _mm256_load_si256((__m256i*) &bb->s[4]); \
    __m256i v2 = _mm256_load_si256((__m256i*) &bb->s[8]); \
    __m256i v3 = _mm256_load_si256((__m256i*) &bb->s[12]); \
    __m256i all1s = _mm256_set1_epi32(-1); \
    _mm256_store_si256((__m256i*) aa->s, _mm256_xor_si256(v0, all1s)); \
    _mm256_store_si256((__m256i*) &aa->s[4], _mm256_xor_si256(v1, all1s)); \
    _mm256_store_si256((__m256i*) &aa->s[8], _mm256_xor_si256(v2, all1s)); \
    _mm256_store_si256((__m256i*) &aa->s[12], _mm256_xor_si256(v3, all1s)); \
} while(0)

#elif defined(__AVX__)

#define uint1024_t_neg(a, b) do { \
    uint1024_t* const aa = (a); \
    const uint1024_t* const bb = (b); \
    __m256i v0 = _mm256_load_si256((__m256i*) bb->s); \
    __m256i v1 = _mm256_load_si256((__m256i*) &bb->s[4]); \
    __m256i v2 = _mm256_load_si256((__m256i*) &bb->s[8]); \
    __m256i v3 = _mm256_load_si256((__m256i*) &bb->s[12]); \
    __m256i all1s = _mm256_set1_epi32(-1); \
    v0 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(v0), \
                                           _mm256_castsi256_pd(all1s))); \
    _mm256_store_si256((__m256i*) aa->s, v0); \
    v1 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(v1), \
                                           _mm256_castsi256_pd(all1s))); \
    _mm256_store_si256((__m256i*) &aa->s[4], v1); \
    v2 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(v2), \
                                           _mm256_castsi256_pd(all1s))); \
    _mm256_store_si256((__m256i*) &aa->s[8], v2); \
    v3 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(v3), \
                                           _mm256_castsi256_pd(all1s))); \
    _mm256_store_si256((__m256i*) &aa->s[12], v3); \
} while(0)

#else

#define uint1024_t_neg(a, b) do { \
    uint1024_t* const aa = (a); \
    const uint1024_t* const bb = (b); \
    aa->s[ 0] = ~bb->s[ 0]; \
    aa->s[ 1] = ~bb->s[ 1]; \
    aa->s[ 2] = ~bb->s[ 2]; \
    aa->s[ 3] = ~bb->s[ 3]; \
    aa->s[ 4] = ~bb->s[ 4]; \
    aa->s[ 5] = ~bb->s[ 5]; \
    aa->s[ 6] = ~bb->s[ 6]; \
    aa->s[ 7] = ~bb->s[ 7]; \
    aa->s[ 8] = ~bb->s[ 8]; \
    aa->s[ 9] = ~bb->s[ 9]; \
    aa->s[10] = ~bb->s[10]; \
    aa->s[11] = ~bb->s[11]; \
    aa->s[12] = ~bb->s[12]; \
    aa->s[13] = ~bb->s[13]; \
    aa->s[14] = ~bb->s[14]; \
    aa->s[15] = ~bb->s[15]; \
} while(0)

#endif

/* usage: Given 2 uint1024_t a and b, compute a^b and store the result into p
 * params:
 *      1) p: ptr to struct uint1024_t
 *      2) a: ptr to struct uint1024_t
 *      3) b: ptr to struct uint1024_t
 * return: void */
#if defined(__AVX512F__)

#define uint1024_t_xor(p, a, b) do { \
    uint1024_t* const pp = (p); \
    const uint1024_t* const aa = (a); \
    const uint1024_t* const bb = (b); \
    __m512i va0 = _mm512_load_si512((void const*) aa->s); \
    __m512i va1 = _mm512_load_si512((void const*) &aa->s[8]); \
    __m512i vb0 = _mm512_load_si512((void const*) bb->s); \
    __m512i vb1 = _mm512_load_si512((void const*) &bb->s[8]); \
    va0 = _mm512_xor_si512(va0, vb0); \
    va1 = _mm512_xor_si512(va1, vb1); \
    _mm512_store_si512((void*) pp->s, va0); \
    _mm512_store_si512((void*) &pp->s[8], va1); \
} while(0)

#elif defined(__AVX2__)

#define uint1024_t_xor(p, a, b) do { \
    uint1024_t* const pp = (p); \
    const uint1024_t* const aa = (a); \
    const uint1024_t* const bb = (b); \
    __m256i va0 = _mm256_load_si256((__m256i*) aa->s); \
    __m256i va1 = _mm256_load_si256((__m256i*) &aa->s[4]); \
    __m256i va2 = _mm256_load_si256((__m256i*) &aa->s[8]); \
    __m256i va3 = _mm256_load_si256((__m256i*) &aa->s[12]); \
    __m256i vb0 = _mm256_load_si256((__m256i*) bb->s); \
    __m256i vb1 = _mm256_load_si256((__m256i*) &bb->s[4]); \
    __m256i vb2 = _mm256_load_si256((__m256i*) &bb->s[8]); \
    __m256i vb3 = _mm256_load_si256((__m256i*) &bb->s[12]); \
    va0 = _mm256_xor_si256(vb0, va0); \
    va1 = _mm256_xor_si256(vb1, va1); \
    va2 = _mm256_xor_si256(vb2, va2); \
    va3 = _mm256_xor_si256(vb3, va3); \
    _mm256_store_si256((__m256i*) pp->s, va0); \
    _mm256_store_si256((__m256i*) &pp->s[4], va1); \
    _mm256_store_si256((__m256i*) &pp->s[8], va2); \
    _mm256_store_si256((__m256i*) &pp->s[12], va3); \
} while(0)

#elif defined(__AVX__)

#define uint1024_t_xor(p, a, b) do { \
    uint1024_t* const pp = (p); \
    const uint1024_t* const aa = (a); \
    const uint1024_t* const bb = (b); \
    __m256i va0 = _mm256_load_si256((__m256i*) aa->s); \
    __m256i va1 = _mm256_load_si256((__m256i*) &aa->s[4]); \
    __m256i va2 = _mm256_load_si256((__m256i*) &aa->s[8]); \
    __m256i va3 = _mm256_load_si256((__m256i*) &aa->s[12]); \
    __m256i vb0 = _mm256_load_si256((__m256i*) bb->s); \
    __m256i vb1 = _mm256_load_si256((__m256i*) &bb->s[4]); \
    __m256i vb2 = _mm256_load_si256((__m256i*) &bb->s[8]); \
    __m256i vb3 = _mm256_load_si256((__m256i*) &bb->s[12]); \
    va0 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(vb0), \
                                            _mm256_castsi256_pd(va0))); \
    va1 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(vb1), \
                                            _mm256_castsi256_pd(va1))); \
    va2 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(vb2), \
                                            _mm256_castsi256_pd(va2))); \
    va3 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(vb3), \
                                            _mm256_castsi256_pd(va3))); \
    _mm256_store_si256((__m256i*) pp->s, va0); \
    _mm256_store_si256((__m256i*) &pp->s[4], va1); \
    _mm256_store_si256((__m256i*) &pp->s[8], va2); \
    _mm256_store_si256((__m256i*) &pp->s[12], va3); \
} while(0)

#else

#define uint1024_t_xor(p, a, b) do { \
    uint1024_t* const pp = (p); \
    const uint1024_t* const aa = (a); \
    const uint1024_t* const bb = (b); \
    pp->s[ 0] = aa->s[ 0] ^ bb->s[ 0]; \
    pp->s[ 1] = aa->s[ 1] ^ bb->s[ 1]; \
    pp->s[ 2] = aa->s[ 2] ^ bb->s[ 2]; \
    pp->s[ 3] = aa->s[ 3] ^ bb->s[ 3]; \
    pp->s[ 4] = aa->s[ 4] ^ bb->s[ 4]; \
    pp->s[ 5] = aa->s[ 5] ^ bb->s[ 5]; \
    pp->s[ 6] = aa->s[ 6] ^ bb->s[ 6]; \
    pp->s[ 7] = aa->s[ 7] ^ bb->s[ 7]; \
    pp->s[ 8] = aa->s[ 8] ^ bb->s[ 8]; \
    pp->s[ 9] = aa->s[ 9] ^ bb->s[ 9]; \
    pp->s[10] = aa->s[10] ^ bb->s[10]; \
    pp->s[11] = aa->s[11] ^ bb->s[11]; \
    pp->s[12] = aa->s[12] ^ bb->s[12]; \
    pp->s[13] = aa->s[13] ^ bb->s[13]; \
    pp->s[14] = aa->s[14] ^ bb->s[14]; \
    pp->s[15] = aa->s[15] ^ bb->s[15]; \
} while(0)

#endif

#if defined(__AVX512F__)

/* usage: Given 1 uint1024_t a and 2 __m512i registers, load the content of
 *      a into those 2 registers
 * params:
 *      1) a: ptr to struct uint1024_t
 *      2) va0: register to store the 1st 512 bits of uint1024_t a
 *      3) va1: register to store the 2nd 512 bits of uint1024_t a
 * return: void */
#define uint1024_t_load_to_reg_avx512(a, va0, va1) do { \
    const uint1024_t* const aa = (a); \
    va0 = _mm512_load_si512((void const*) aa->s); \
    va1 = _mm512_load_si512((void const*) &a->s[8]); \
} while(0)

/* usage: Given 1 uint1024_t a and 2 __m512i registers, store the content of
 *      those 2 registers into a
 * params:
 *      1) a: ptr to struct uint1024_t
 *      2) va0: register to store the 1st 512 bits of uint1024_t a
 *      3) va1: register to store the 2nd 512 bits of uint1024_t a
 * return: void */
#define uint1024_t_load_from_reg_avx512(a, va0, va1) do { \
    uint1024_t* const aa = (a); \
    _mm512_store_si512((void*) aa->s, va0); \
    _mm512_store_si512((void*) &a->s[8], va1); \
} while(0)

/* usage: Given 1 uint1024_t a and 2 __m512i registers, store the content of
 *      those 2 registers into a with non-temporal hint.
 * params:
 *      1) a: ptr to struct uint1024_t
 *      2) va0: register to store the 1st 512 bits of uint1024_t a
 *      3) va1: register to store the 2nd 512 bits of uint1024_t a
 * return: void */
#define uint1024_t_load_from_reg_avx512_nt(a, va0, va1) do { \
    uint1024_t* const aa = (a); \
    _mm512_stream_si512((void*) aa->s, va0); \
    _mm512_stream_si512((void*) &a->s[8], va1); \
} while(0)

/* usage: Given 2 uint1024_t a and b, where the content of a is stored in 2
 *      __m512i registers, compute a^b and store the result back into those
 *      registers (in place xor).
 * params:
 *      1) b: ptr to struct uint1024_t
 *      2) va0: register that store the 1st 512 bits of uint1024_t a
 *      3) va1: register that stores the 2nd 512 bits of uint1024_t a
 * return: void */
#define uint1024_t_xori_to_reg_avx512(b, va0, va1) do { \
    const uint1024_t* const bb = (b); \
    __m512i vb0 = _mm512_load_si512((void const*) bb->s); \
    __m512i vb1 = _mm512_load_si512((void const*) &bb->s[8]); \
    va0 = _mm512_xor_si512(va0, vb0); \
    va1 = _mm512_xor_si512(va1, vb1); \
} while(0)

/* usage: Given 2 uint1024_t a and b, where the content of b is stored in 2
 *      __m512i registers, compute a^b and store the result back into a
 *      (in place xor).
 * params:
 *      1) a: ptr to struct uint1024_t
 *      2) vb0: register that store the 1st 512 bits of uint1024_t b
 *      3) vb1: register that stores the 2nd 512 bits of uint1024_t b
 * return: void */
#define uint1024_t_xori_from_reg_avx512(a, vb0, vb1) do { \
    uint1024_t* const aa = (a); \
    __m512i va0 = _mm512_load_si512((void const*) aa->s); \
    __m512i va1 = _mm512_load_si512((void const*) &aa->s[8]); \
    va0 = _mm512_xor_si512(va0, vb0); \
    va1 = _mm512_xor_si512(va1, vb1); \
    _mm512_store_si512((void*) aa->s, va0); \
    _mm512_store_si512((void*) &aa->s[8], va1); \
} while(0)

#endif

#if defined(__AVX__)

/* usage: Given 1 uint1024_t a and 4 __m256i registers, load the content of
 *      a into those 4 registers
 * params:
 *      1) a: ptr to struct uint1024_t
 *      2) va0: register to store the 1st 256 bits of uint1024_t a
 *      3) va1: register to store the 2nd 256 bits of uint1024_t a
 *      4) va2: register to store the 3rd 256 bits of uint1024_t a
 *      5) va3: register to store the 4th 256 bits of uint1024_t a
 * return: void */
#define uint1024_t_load_to_reg(a, va0, va1, va2, va3) do { \
    const uint1024_t* const aa = (a); \
    va0 = _mm256_load_si256((__m256i*) aa->s); \
    va1 = _mm256_load_si256((__m256i*) &aa->s[4]); \
    va2 = _mm256_load_si256((__m256i*) &aa->s[8]); \
    va3 = _mm256_load_si256((__m256i*) &aa->s[12]); \
} while(0)

/* usage: Given 1 uint1024_t a and 4 __m256i registers, store the content of
 *      those 4 registers into a
 * params:
 *      1) a: ptr to struct uint1024_t
 *      2) va0: register to store the 1st 256 bits of uint1024_t a
 *      3) va1: register to store the 2nd 256 bits of uint1024_t a
 *      4) va2: register to store the 3rd 256 bits of uint1024_t a
 *      5) va3: register to store the 4th 256 bits of uint1024_t a
 * return: void */
#define uint1024_t_load_from_reg(a, va0, va1, va2, va3) do { \
    uint1024_t* const aa = (a); \
    _mm256_store_si256((__m256i*) aa->s, va0); \
    _mm256_store_si256((__m256i*) &aa->s[4], va1); \
    _mm256_store_si256((__m256i*) &aa->s[8], va2); \
    _mm256_store_si256((__m256i*) &aa->s[12], va3); \
} while(0)

/* usage: Given 1 uint1024_t a and 4 __m256i registers, store the content of
 *      those 4 registers into a with non-temporal hint.
 * params:
 *      1) a: ptr to struct uint1024_t
 *      2) va0: register to store the 1st 256 bits of uint1024_t a
 *      3) va1: register to store the 2nd 256 bits of uint1024_t a
 *      4) va2: register to store the 3rd 256 bits of uint1024_t a
 *      5) va3: register to store the 4th 256 bits of uint1024_t a
 * return: void */
#define uint1024_t_load_from_reg_nt(a, va0, va1, va2, va3) do { \
    uint1024_t* const aa = (a); \
    _mm256_stream_si256((__m256i*) aa->s, va0); \
    _mm256_stream_si256((__m256i*) &aa->s[4], va1); \
    _mm256_stream_si256((__m256i*) &aa->s[8], va2); \
    _mm256_stream_si256((__m256i*) &aa->s[12], va3); \
} while(0)

#endif

#if defined(__AVX2__)

/* usage: Given 2 uint1024_t a and b, where the content of a is stored in 4
 *      __m256i registers, compute a^b and store the result back into those
 *      registers (in place xor).
 * params:
 *      1) b: ptr to struct uint1024_t
 *      2) va0: register that store the 1st 256 bits of uint1024_t a
 *      3) va1: register that stores the 2nd 256 bits of uint1024_t a
 *      4) va2: register that stores the 3rd 256 bits of uint1024_t a
 *      5) va3: register that stores the 4th 256 bits of uint1024_t a
 * return: void */
#define uint1024_t_xori_to_reg(b, va0, va1, va2, va3) do { \
    const uint1024_t* const bb = (b); \
    __m256i vb0 = _mm256_load_si256((__m256i*) bb->s); \
    __m256i vb1 = _mm256_load_si256((__m256i*) &bb->s[4]); \
    __m256i vb2 = _mm256_load_si256((__m256i*) &bb->s[8]); \
    __m256i vb3 = _mm256_load_si256((__m256i*) &bb->s[12]); \
    va0 = _mm256_xor_si256(vb0, va0); \
    va1 = _mm256_xor_si256(vb1, va1); \
    va2 = _mm256_xor_si256(vb2, va2); \
    va3 = _mm256_xor_si256(vb3, va3); \
} while(0)

/* usage: Given 2 uint1024_t a and b, where the content of b is stored in 4
 *      __m256i registers, compute a^b and store the result back into a
 *      (in place xor).
 * params:
 *      1) a: ptr to struct uint1024_t
 *      2) vb0: register that store the 1st 256 bits of uint1024_t b
 *      3) vb1: register that stores the 2nd 256 bits of uint1024_t b
 *      4) vb2: register that stores the 3rd 256 bits of uint1024_t b
 *      5) vb3: register that stores the 4th 256 bits of uint1024_t b
 * return: void */
#define uint1024_t_xori_from_reg(a, vb0, vb1, vb2, vb3) do { \
    uint1024_t* const aa = (a); \
    __m256i va0 = _mm256_load_si256((__m256i*) aa->s); \
    __m256i va1 = _mm256_load_si256((__m256i*) &aa->s[4]); \
    __m256i va2 = _mm256_load_si256((__m256i*) &aa->s[8]); \
    __m256i va3 = _mm256_load_si256((__m256i*) &aa->s[12]); \
    va0 = _mm256_xor_si256(vb0, va0); \
    va1 = _mm256_xor_si256(vb1, va1); \
    va2 = _mm256_xor_si256(vb2, va2); \
    va3 = _mm256_xor_si256(vb3, va3); \
    _mm256_store_si256((__m256i*) aa->s, va0); \
    _mm256_store_si256((__m256i*) &aa->s[4], va1); \
    _mm256_store_si256((__m256i*) &aa->s[8], va2); \
    _mm256_store_si256((__m256i*) &aa->s[12], va3); \
} while(0)

#elif defined(__AVX__)

/* usage: Given 2 uint1024_t a and b, where the content of a is stored in 4
 *      __m256i registers, compute a^b and store the result back into those
 *      registers (in place xor).
 * params:
 *      1) b: ptr to struct uint1024_t
 *      2) va0: register that store the 1st 256 bits of uint1024_t a
 *      3) va1: register that stores the 2nd 256 bits of uint1024_t a
 *      4) va2: register that stores the 3rd 256 bits of uint1024_t a
 *      5) va3: register that stores the 4th 256 bits of uint1024_t a
 * return: void */
#define uint1024_t_xori_to_reg(b, va0, va1, va2, va3) do { \
    const uint1024_t* const bb = (b); \
    __m256i vb0 = _mm256_load_si256((__m256i*) bb->s); \
    __m256i vb1 = _mm256_load_si256((__m256i*) &bb->s[4]); \
    __m256i vb2 = _mm256_load_si256((__m256i*) &bb->s[8]); \
    __m256i vb3 = _mm256_load_si256((__m256i*) &bb->s[12]); \
    va0 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(vb0), \
                                            _mm256_castsi256_pd(va0))); \
    va1 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(vb1), \
                                            _mm256_castsi256_pd(va1))); \
    va2 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(vb2), \
                                            _mm256_castsi256_pd(va2))); \
    va3 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(vb3), \
                                            _mm256_castsi256_pd(va3))); \
} while(0)

/* usage: Given 2 uint1024_t a and b, where the content of b is stored in 4
 *      __m256i registers, compute a^b and store the result back into a
 *      (in place xor).
 * params:
 *      1) a: ptr to struct uint1024_t
 *      2) vb0: register that store the 1st 256 bits of uint1024_t b
 *      3) vb1: register that stores the 2nd 256 bits of uint1024_t b
 *      4) vb2: register that stores the 3rd 256 bits of uint1024_t b
 *      5) vb3: register that stores the 4th 256 bits of uint1024_t b
 * return: void */
#define uint1024_t_xori_from_reg(a, vb0, vb1, vb2, vb3) do { \
    uint1024_t* const aa = (a); \
    __m256i va0 = _mm256_load_si256((__m256i*) aa->s); \
    __m256i va1 = _mm256_load_si256((__m256i*) &aa->s[4]); \
    __m256i va2 = _mm256_load_si256((__m256i*) &aa->s[8]); \
    __m256i va3 = _mm256_load_si256((__m256i*) &aa->s[12]); \
    va0 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(vb0), \
                                            _mm256_castsi256_pd(va0))); \
    va1 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(vb1), \
                                            _mm256_castsi256_pd(va1))); \
    va2 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(vb2), \
                                            _mm256_castsi256_pd(va2))); \
    va3 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(vb3), \
                                            _mm256_castsi256_pd(va3))); \
    _mm256_store_si256((__m256i*) aa->s, va0); \
    _mm256_store_si256((__m256i*) &aa->s[4], va1); \
    _mm256_store_si256((__m256i*) &aa->s[8], va2); \
    _mm256_store_si256((__m256i*) &aa->s[12], va3); \
} while(0)

#endif

/* usage: Given 2 uint1024_t a and b, compute a^b and store the result into a.
 *      (in place xor)
 * params:
 *      1) a: ptr to struct uint1024_t
 *      2) b: ptr to struct uint1024_t
 * return: void */
#if defined(__AVX512F__)

#define uint1024_t_xori(a, b) do { \
    uint1024_t* const aa = (a); \
    const uint1024_t* const bb = (b); \
    __m512i va0 = _mm512_load_si512((void const*) aa->s); \
    __m512i va1 = _mm512_load_si512((void const*) &aa->s[8]); \
    __m512i vb0 = _mm512_load_si512((void const*) bb->s); \
    __m512i vb1 = _mm512_load_si512((void const*) &bb->s[8]); \
    va0 = _mm512_xor_si512(va0, vb0); \
    va1 = _mm512_xor_si512(va1, vb1); \
    _mm512_store_si512((void*) aa->s, va0); \
    _mm512_store_si512((void*) &aa->s[8], va1); \
} while(0)

#elif defined(__AVX2__)

#define uint1024_t_xori(a, b) do { \
    uint1024_t* const aa = (a); \
    const uint1024_t* const bb = (b); \
    __m256i va0 = _mm256_load_si256((__m256i*) aa->s); \
    __m256i va1 = _mm256_load_si256((__m256i*) &aa->s[4]); \
    __m256i va2 = _mm256_load_si256((__m256i*) &aa->s[8]); \
    __m256i va3 = _mm256_load_si256((__m256i*) &aa->s[12]); \
    __m256i vb0 = _mm256_load_si256((__m256i*) bb->s); \
    __m256i vb1 = _mm256_load_si256((__m256i*) &bb->s[4]); \
    __m256i vb2 = _mm256_load_si256((__m256i*) &bb->s[8]); \
    __m256i vb3 = _mm256_load_si256((__m256i*) &bb->s[12]); \
    va0 = _mm256_xor_si256(vb0, va0); \
    va1 = _mm256_xor_si256(vb1, va1); \
    va2 = _mm256_xor_si256(vb2, va2); \
    va3 = _mm256_xor_si256(vb3, va3); \
    _mm256_store_si256((__m256i*) aa->s, va0); \
    _mm256_store_si256((__m256i*) &aa->s[4], va1); \
    _mm256_store_si256((__m256i*) &aa->s[8], va2); \
    _mm256_store_si256((__m256i*) &aa->s[12], va3); \
} while(0)

#elif defined(__AVX__)

#define uint1024_t_xori(a, b) do { \
    uint1024_t* const aa = (a); \
    const uint1024_t* const bb = (b); \
    __m256i va0 = _mm256_load_si256((__m256i*) aa->s); \
    __m256i va1 = _mm256_load_si256((__m256i*) &aa->s[4]); \
    __m256i va2 = _mm256_load_si256((__m256i*) &aa->s[8]); \
    __m256i va3 = _mm256_load_si256((__m256i*) &aa->s[12]); \
    __m256i vb0 = _mm256_load_si256((__m256i*) bb->s); \
    __m256i vb1 = _mm256_load_si256((__m256i*) &bb->s[4]); \
    __m256i vb2 = _mm256_load_si256((__m256i*) &bb->s[8]); \
    __m256i vb3 = _mm256_load_si256((__m256i*) &bb->s[12]); \
    va0 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(vb0), \
                                            _mm256_castsi256_pd(va0))); \
    va1 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(vb1), \
                                            _mm256_castsi256_pd(va1))); \
    va2 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(vb2), \
                                            _mm256_castsi256_pd(va2))); \
    va3 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(vb3), \
                                            _mm256_castsi256_pd(va3))); \
    _mm256_store_si256((__m256i*) aa->s, va0); \
    _mm256_store_si256((__m256i*) &aa->s[4], va1); \
    _mm256_store_si256((__m256i*) &aa->s[8], va2); \
    _mm256_store_si256((__m256i*) &aa->s[12], va3); \
} while(0)

#else

#define uint1024_t_xori(a, b) do { \
    uint1024_t* const aa = (a); \
    const uint1024_t* const bb = (b); \
    aa->s[ 0] ^= bb->s[ 0]; \
    aa->s[ 1] ^= bb->s[ 1]; \
    aa->s[ 2] ^= bb->s[ 2]; \
    aa->s[ 3] ^= bb->s[ 3]; \
    aa->s[ 4] ^= bb->s[ 4]; \
    aa->s[ 5] ^= bb->s[ 5]; \
    aa->s[ 6] ^= bb->s[ 6]; \
    aa->s[ 7] ^= bb->s[ 7]; \
    aa->s[ 8] ^= bb->s[ 8]; \
    aa->s[ 9] ^= bb->s[ 9]; \
    aa->s[10] ^= bb->s[10]; \
    aa->s[11] ^= bb->s[11]; \
    aa->s[12] ^= bb->s[12]; \
    aa->s[13] ^= bb->s[13]; \
    aa->s[14] ^= bb->s[14]; \
    aa->s[15] ^= bb->s[15]; \
} while(0)

#endif

/* usage: Given 3 uint1024_t a, b, and c, compute a ^ (b & c) and store the
 *      result back into a. (in place)
 * params:
 *      1) a: ptr to struct uint1024_t
 *      2) b: ptr to struct uint1024_t
 *      3) c: ptr to struct uint1024_t
 * return: void */
#if defined(__AVX512F__)

#define uint1024_t_xori_and(a, b, c) do { \
    uint1024_t* const aa = (a); \
    const uint1024_t* const bb = (b); \
    const uint1024_t* const cc = (c); \
    __m512i vc0 = _mm512_load_si512((void const*) cc->s); \
    __m512i vc1 = _mm512_load_si512((void const*) &cc->s[8]); \
    __m512i vb0 = _mm512_load_si512((void const*) bb->s); \
    __m512i vb1 = _mm512_load_si512((void const*) &bb->s[8]); \
    vb0 = _mm512_and_si512(vb0, vc0); \
    vb1 = _mm512_and_si512(vb1, vc1); \
    __m512i va0 = _mm512_load_si512((void const*) aa->s); \
    __m512i va1 = _mm512_load_si512((void const*) &aa->s[8]); \
    va0 = _mm512_xor_si512(vb0, va0); \
    va1 = _mm512_xor_si512(vb1, va1); \
    _mm512_store_si512((void*) aa->s, va0); \
    _mm512_store_si512((void*) &aa->s[8], va1); \
} while(0)

#elif defined(__AVX2__)

#define uint1024_t_xori_and(a, b, c) do { \
    uint1024_t* const aa = (a); \
    const uint1024_t* const bb = (b); \
    const uint1024_t* const cc = (c); \
    __m256i vc0 = _mm256_load_si256((__m256i*) cc->s); \
    __m256i vc1 = _mm256_load_si256((__m256i*) &cc->s[4]); \
    __m256i vc2 = _mm256_load_si256((__m256i*) &cc->s[8]); \
    __m256i vc3 = _mm256_load_si256((__m256i*) &cc->s[12]); \
    __m256i vb0 = _mm256_load_si256((__m256i*) bb->s); \
    __m256i vb1 = _mm256_load_si256((__m256i*) &bb->s[4]); \
    __m256i vb2 = _mm256_load_si256((__m256i*) &bb->s[8]); \
    __m256i vb3 = _mm256_load_si256((__m256i*) &bb->s[12]); \
    vb0 = _mm256_and_si256(vb0, vc0); \
    vb1 = _mm256_and_si256(vb1, vc1); \
    vb2 = _mm256_and_si256(vb2, vc2); \
    vb3 = _mm256_and_si256(vb3, vc3); \
    __m256i va0 = _mm256_load_si256((__m256i*) aa->s); \
    __m256i va1 = _mm256_load_si256((__m256i*) &aa->s[4]); \
    __m256i va2 = _mm256_load_si256((__m256i*) &aa->s[8]); \
    __m256i va3 = _mm256_load_si256((__m256i*) &aa->s[12]); \
    va0 = _mm256_xor_si256(vb0, va0); \
    va1 = _mm256_xor_si256(vb1, va1); \
    va2 = _mm256_xor_si256(vb2, va2); \
    va3 = _mm256_xor_si256(vb3, va3); \
    _mm256_store_si256((__m256i*) aa->s, va0); \
    _mm256_store_si256((__m256i*) &aa->s[4], va1); \
    _mm256_store_si256((__m256i*) &aa->s[8], va2); \
    _mm256_store_si256((__m256i*) &aa->s[12], va3); \
} while(0)

#elif defined(__AVX__)

#define uint1024_t_xori_and(a, b, c) do { \
    uint1024_t* const aa = (a); \
    const uint1024_t* const bb = (b); \
    const uint1024_t* const cc = (c); \
    __m256i vc0 = _mm256_load_si256((__m256i*) cc->s); \
    __m256i vc1 = _mm256_load_si256((__m256i*) &cc->s[4]); \
    __m256i vc2 = _mm256_load_si256((__m256i*) &cc->s[8]); \
    __m256i vc3 = _mm256_load_si256((__m256i*) &cc->s[12]); \
    __m256i vb0 = _mm256_load_si256((__m256i*) bb->s); \
    __m256i vb1 = _mm256_load_si256((__m256i*) &bb->s[4]); \
    __m256i vb2 = _mm256_load_si256((__m256i*) &bb->s[8]); \
    __m256i vb3 = _mm256_load_si256((__m256i*) &bb->s[12]); \
    vb0 = _mm256_castpd_si256(_mm256_and_pd(_mm256_castsi256_pd(vb0), \
                                            _mm256_castsi256_pd(vc0))); \
    vb1 = _mm256_castpd_si256(_mm256_and_pd(_mm256_castsi256_pd(vb1), \
                                            _mm256_castsi256_pd(vc1))); \
    vb2 = _mm256_castpd_si256(_mm256_and_pd(_mm256_castsi256_pd(vb2), \
                                            _mm256_castsi256_pd(vc2))); \
    vb3 = _mm256_castpd_si256(_mm256_and_pd(_mm256_castsi256_pd(vb3), \
                                            _mm256_castsi256_pd(vc3))); \
    __m256i va0 = _mm256_load_si256((__m256i*) aa->s); \
    __m256i va1 = _mm256_load_si256((__m256i*) &aa->s[4]); \
    __m256i va2 = _mm256_load_si256((__m256i*) &aa->s[8]); \
    __m256i va3 = _mm256_load_si256((__m256i*) &aa->s[12]); \
    va0 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(vb0), \
                                            _mm256_castsi256_pd(va0))); \
    va1 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(vb1), \
                                            _mm256_castsi256_pd(va1))); \
    va2 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(vb2), \
                                            _mm256_castsi256_pd(va2))); \
    va3 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(vb3), \
                                            _mm256_castsi256_pd(va3))); \
    _mm256_store_si256((__m256i*) aa->s, va0); \
    _mm256_store_si256((__m256i*) &aa->s[4], va1); \
    _mm256_store_si256((__m256i*) &aa->s[8], va2); \
    _mm256_store_si256((__m256i*) &aa->s[12], va3); \
} while(0)

#else

#define uint1024_t_xori_and(a, b, c) do { \
    uint1024_t* const aa = (a); \
    const uint1024_t* const bb = (b); \
    const uint1024_t* const cc = (c); \
    aa->s[ 0] ^= bb->s[ 0] & cc->s[ 0]; \
    aa->s[ 1] ^= bb->s[ 1] & cc->s[ 1]; \
    aa->s[ 2] ^= bb->s[ 2] & cc->s[ 2]; \
    aa->s[ 3] ^= bb->s[ 3] & cc->s[ 3]; \
    aa->s[ 4] ^= bb->s[ 4] & cc->s[ 4]; \
    aa->s[ 5] ^= bb->s[ 5] & cc->s[ 5]; \
    aa->s[ 6] ^= bb->s[ 6] & cc->s[ 6]; \
    aa->s[ 7] ^= bb->s[ 7] & cc->s[ 7]; \
    aa->s[ 8] ^= bb->s[ 8] & cc->s[ 8]; \
    aa->s[ 9] ^= bb->s[ 9] & cc->s[ 9]; \
    aa->s[10] ^= bb->s[10] & cc->s[10]; \
    aa->s[11] ^= bb->s[11] & cc->s[11]; \
    aa->s[12] ^= bb->s[12] & cc->s[12]; \
    aa->s[13] ^= bb->s[13] & cc->s[13]; \
    aa->s[14] ^= bb->s[14] & cc->s[14]; \
    aa->s[15] ^= bb->s[15] & cc->s[15]; \
} while(0)

#endif

/* usage: Given 3 uint1024_t a, b, c, compute (a & c) ^ (b & (~c))
 *      and store the result into p.
 * params:
 *      1) p: ptr to struct uint1024_t
 *      2) a: ptr to struct uint1024_t
 *      3) b: ptr to struct uint1024_t
 *      4) c: ptr to struct uint1024_t
 * return: void */
#if defined(__AVX512F__)

#define uint1024_t_mix(p, a, b, c) do { \
    uint1024_t* const pp = (p); \
    const uint1024_t* const aa = (a); \
    const uint1024_t* const bb = (b); \
    const uint1024_t* const cc = (c); \
    __m512i vc0 = _mm512_load_si512((void const*) cc->s); \
    __m512i vc1 = _mm512_load_si512((void const*) &cc->s[8]); \
    __m512i va0 = _mm512_load_si512((void const*) aa->s); \
    __m512i va1 = _mm512_load_si512((void const*) &aa->s[8]); \
    va0 = _mm512_and_si512(va0, vc0); \
    va1 = _mm512_and_si512(va1, vc1); \
    __m512i vb0 = _mm512_load_si512((void const*) bb->s); \
    __m512i vb1 = _mm512_load_si512((void const*) &bb->s[8]); \
    vb0 = _mm512_andnot_si512(vc0, vb0); \
    vb1 = _mm512_andnot_si512(vc1, vb1); \
    va0 = _mm512_xor_si512(va0, vb0); \
    va1 = _mm512_xor_si512(va1, vb1); \
    _mm512_store_si512((void*) pp->s, va0); \
    _mm512_store_si512((void*) &pp->s[8], va1); \
} while(0)

#elif defined(__AVX2__)

#define uint1024_t_mix(p, a, b, c) do { \
    uint1024_t* const pp = (p); \
    const uint1024_t* const aa = (a); \
    const uint1024_t* const bb = (b); \
    const uint1024_t* const cc = (c); \
    __m256i vc0 = _mm256_load_si256((__m256i*) cc->s); \
    __m256i vc1 = _mm256_load_si256((__m256i*) &cc->s[4]); \
    __m256i vc2 = _mm256_load_si256((__m256i*) &cc->s[8]); \
    __m256i vc3 = _mm256_load_si256((__m256i*) &cc->s[12]); \
    __m256i va0 = _mm256_load_si256((__m256i*) aa->s); \
    __m256i va1 = _mm256_load_si256((__m256i*) &aa->s[4]); \
    __m256i va2 = _mm256_load_si256((__m256i*) &aa->s[8]); \
    __m256i va3 = _mm256_load_si256((__m256i*) &aa->s[12]); \
    va0 = _mm256_and_si256(va0, vc0); \
    va1 = _mm256_and_si256(va1, vc1); \
    va2 = _mm256_and_si256(va2, vc2); \
    va3 = _mm256_and_si256(va3, vc3); \
    __m256i vb0 = _mm256_load_si256((__m256i*) bb->s); \
    __m256i vb1 = _mm256_load_si256((__m256i*) &bb->s[4]); \
    __m256i vb2 = _mm256_load_si256((__m256i*) &bb->s[8]); \
    __m256i vb3 = _mm256_load_si256((__m256i*) &bb->s[12]); \
    vb0 = _mm256_andnot_si256(vc0, vb0); \
    vb1 = _mm256_andnot_si256(vc1, vb1); \
    vb2 = _mm256_andnot_si256(vc2, vb2); \
    vb3 = _mm256_andnot_si256(vc3, vb3); \
    va0 = _mm256_xor_si256(va0, vb0); \
    va1 = _mm256_xor_si256(va1, vb1); \
    va2 = _mm256_xor_si256(va2, vb2); \
    va3 = _mm256_xor_si256(va3, vb3); \
    _mm256_store_si256((__m256i*) pp->s, va0); \
    _mm256_store_si256((__m256i*) &pp->s[4], va1); \
    _mm256_store_si256((__m256i*) &pp->s[8], va2); \
    _mm256_store_si256((__m256i*) &pp->s[12], va3); \
} while(0)

#elif defined(__AVX__)

#define uint1024_t_mix(p, a, b, c) do { \
    uint1024_t* const pp = (p); \
    const uint1024_t* const aa = (a); \
    const uint1024_t* const bb = (b); \
    const uint1024_t* const cc = (c); \
    __m256i vc0 = _mm256_load_si256((__m256i*) cc->s); \
    __m256i vc1 = _mm256_load_si256((__m256i*) &cc->s[4]); \
    __m256i vc2 = _mm256_load_si256((__m256i*) &cc->s[8]); \
    __m256i vc3 = _mm256_load_si256((__m256i*) &cc->s[12]); \
    __m256i va0 = _mm256_load_si256((__m256i*) aa->s); \
    __m256i va1 = _mm256_load_si256((__m256i*) &aa->s[4]); \
    __m256i va2 = _mm256_load_si256((__m256i*) &aa->s[8]); \
    __m256i va3 = _mm256_load_si256((__m256i*) &aa->s[12]); \
    va0 = _mm256_castpd_si256(_mm256_and_pd(_mm256_castsi256_pd(va0), \
                                            _mm256_castsi256_pd(vc0))); \
    va1 = _mm256_castpd_si256(_mm256_and_pd(_mm256_castsi256_pd(va1), \
                                            _mm256_castsi256_pd(vc1))); \
    va2 = _mm256_castpd_si256(_mm256_and_pd(_mm256_castsi256_pd(va2), \
                                            _mm256_castsi256_pd(vc2))); \
    va3 = _mm256_castpd_si256(_mm256_and_pd(_mm256_castsi256_pd(va3), \
                                            _mm256_castsi256_pd(vc3))); \
    __m256i vb0 = _mm256_load_si256((__m256i*) bb->s); \
    __m256i vb1 = _mm256_load_si256((__m256i*) &bb->s[4]); \
    __m256i vb2 = _mm256_load_si256((__m256i*) &bb->s[8]); \
    __m256i vb3 = _mm256_load_si256((__m256i*) &bb->s[12]); \
    vb0 = _mm256_castpd_si256(_mm256_andnot_pd(_mm256_castsi256_pd(vc0), \
                                               _mm256_castsi256_pd(vb0))); \
    vb1 = _mm256_castpd_si256(_mm256_andnot_pd(_mm256_castsi256_pd(vc1), \
                                               _mm256_castsi256_pd(vb1))); \
    vb2 = _mm256_castpd_si256(_mm256_andnot_pd(_mm256_castsi256_pd(vc2), \
                                               _mm256_castsi256_pd(vb2))); \
    vb3 = _mm256_castpd_si256(_mm256_andnot_pd(_mm256_castsi256_pd(vc3), \
                                               _mm256_castsi256_pd(vb3))); \
    va0 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(va0), \
                                            _mm256_castsi256_pd(vb0))); \
    va1 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(va1), \
                                            _mm256_castsi256_pd(vb1))); \
    va2 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(va2), \
                                            _mm256_castsi256_pd(vb2))); \
    va3 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(va3), \
                                            _mm256_castsi256_pd(vb3))); \
    _mm256_store_si256((__m256i*) pp->s, va0); \
    _mm256_store_si256((__m256i*) &pp->s[4], va1); \
    _mm256_store_si256((__m256i*) &pp->s[8], va2); \
    _mm256_store_si256((__m256i*) &pp->s[12], va3); \
} while(0)

#else

#define uint1024_t_mix(p, a, b, c) do { \
    uint1024_t* const pp = (p); \
    const uint1024_t* const aa = (a); \
    const uint1024_t* const bb = (b); \
    const uint1024_t* const cc = (c); \
    pp->s[ 0] = aa->s[ 0] & cc->s[ 0]; \
    pp->s[ 1] = aa->s[ 1] & cc->s[ 1]; \
    pp->s[ 2] = aa->s[ 2] & cc->s[ 2]; \
    pp->s[ 3] = aa->s[ 3] & cc->s[ 3]; \
    pp->s[ 4] = aa->s[ 4] & cc->s[ 4]; \
    pp->s[ 5] = aa->s[ 5] & cc->s[ 5]; \
    pp->s[ 6] = aa->s[ 6] & cc->s[ 6]; \
    pp->s[ 7] = aa->s[ 7] & cc->s[ 7]; \
    pp->s[ 8] = aa->s[ 8] & cc->s[ 8]; \
    pp->s[ 9] = aa->s[ 9] & cc->s[ 9]; \
    pp->s[10] = aa->s[10] & cc->s[10]; \
    pp->s[11] = aa->s[11] & cc->s[11]; \
    pp->s[12] = aa->s[12] & cc->s[12]; \
    pp->s[13] = aa->s[13] & cc->s[13]; \
    pp->s[14] = aa->s[14] & cc->s[14]; \
    pp->s[15] = aa->s[15] & cc->s[15]; \
    pp->s[ 0] ^= bb->s[ 0] & ~cc->s[ 0]; \
    pp->s[ 1] ^= bb->s[ 1] & ~cc->s[ 1]; \
    pp->s[ 2] ^= bb->s[ 2] & ~cc->s[ 2]; \
    pp->s[ 3] ^= bb->s[ 3] & ~cc->s[ 3]; \
    pp->s[ 4] ^= bb->s[ 4] & ~cc->s[ 4]; \
    pp->s[ 5] ^= bb->s[ 5] & ~cc->s[ 5]; \
    pp->s[ 6] ^= bb->s[ 6] & ~cc->s[ 6]; \
    pp->s[ 7] ^= bb->s[ 7] & ~cc->s[ 7]; \
    pp->s[ 8] ^= bb->s[ 8] & ~cc->s[ 8]; \
    pp->s[ 9] ^= bb->s[ 9] & ~cc->s[ 9]; \
    pp->s[10] ^= bb->s[10] & ~cc->s[10]; \
    pp->s[11] ^= bb->s[11] & ~cc->s[11]; \
    pp->s[12] ^= bb->s[12] & ~cc->s[12]; \
    pp->s[13] ^= bb->s[13] & ~cc->s[13]; \
    pp->s[14] ^= bb->s[14] & ~cc->s[14]; \
    pp->s[15] ^= bb->s[15] & ~cc->s[15]; \
} while(0)

#endif

/* usage: Given 3 uint1024_t a, b, c, compute (a & c) ^ (b & (~c))
 *      and store the result back into a. (in place)
 * params:
 *      1) a: ptr to struct uint1024_t
 *      2) b: ptr to struct uint1024_t
 *      3) c: ptr to struct uint1024_t
 * return: void */
#if defined(__AVX512F__)

#define uint1024_t_mixi(a, b, c) do { \
    uint1024_t* const aa = (a); \
    const uint1024_t* const bb = (b); \
    const uint1024_t* const cc = (c); \
    __m512i vc0 = _mm512_load_si512((void const*) cc->s); \
    __m512i vc1 = _mm512_load_si512((void const*) &cc->s[8]); \
    __m512i va0 = _mm512_load_si512((void const*) aa->s); \
    __m512i va1 = _mm512_load_si512((void const*) &aa->s[8]); \
    va0 = _mm512_and_si512(va0, vc0); \
    va1 = _mm512_and_si512(va1, vc1); \
    __m512i vb0 = _mm512_load_si512((void const*) bb->s); \
    __m512i vb1 = _mm512_load_si512((void const*) &bb->s[8]); \
    vb0 = _mm512_andnot_si512(vc0, vb0); \
    vb1 = _mm512_andnot_si512(vc1, vb1); \
    va0 = _mm512_xor_si512(va0, vb0); \
    va1 = _mm512_xor_si512(va1, vb1); \
    _mm512_store_si512((void*) aa->s, va0); \
    _mm512_store_si512((void*) &aa->s[8], va1); \
} while(0)

#elif defined(__AVX2__)

#define uint1024_t_mixi(a, b, c) do { \
    uint1024_t* const aa = (a); \
    const uint1024_t* const bb = (b); \
    const uint1024_t* const cc = (c); \
    __m256i vc0 = _mm256_load_si256((__m256i*) cc->s); \
    __m256i vc1 = _mm256_load_si256((__m256i*) &cc->s[4]); \
    __m256i vc2 = _mm256_load_si256((__m256i*) &cc->s[8]); \
    __m256i vc3 = _mm256_load_si256((__m256i*) &cc->s[12]); \
    __m256i va0 = _mm256_load_si256((__m256i*) aa->s); \
    __m256i va1 = _mm256_load_si256((__m256i*) &aa->s[4]); \
    __m256i va2 = _mm256_load_si256((__m256i*) &aa->s[8]); \
    __m256i va3 = _mm256_load_si256((__m256i*) &aa->s[12]); \
    va0 = _mm256_and_si256(va0, vc0); \
    va1 = _mm256_and_si256(va1, vc1); \
    va2 = _mm256_and_si256(va2, vc2); \
    va3 = _mm256_and_si256(va3, vc3); \
    __m256i vb0 = _mm256_load_si256((__m256i*) bb->s); \
    __m256i vb1 = _mm256_load_si256((__m256i*) &bb->s[4]); \
    __m256i vb2 = _mm256_load_si256((__m256i*) &bb->s[8]); \
    __m256i vb3 = _mm256_load_si256((__m256i*) &bb->s[12]); \
    vb0 = _mm256_andnot_si256(vc0, vb0); \
    vb1 = _mm256_andnot_si256(vc1, vb1); \
    vb2 = _mm256_andnot_si256(vc2, vb2); \
    vb3 = _mm256_andnot_si256(vc3, vb3); \
    va0 = _mm256_xor_si256(va0, vb0); \
    va1 = _mm256_xor_si256(va1, vb1); \
    va2 = _mm256_xor_si256(va2, vb2); \
    va3 = _mm256_xor_si256(va3, vb3); \
    _mm256_store_si256((__m256i*) aa->s, va0); \
    _mm256_store_si256((__m256i*) &aa->s[4], va1); \
    _mm256_store_si256((__m256i*) &aa->s[8], va2); \
    _mm256_store_si256((__m256i*) &aa->s[12], va3); \
} while(0)

#elif defined(__AVX__)

#define uint1024_t_mixi(a, b, c) do { \
    uint1024_t* const aa = (a); \
    const uint1024_t* const bb = (b); \
    const uint1024_t* const cc = (c); \
    __m256i vc0 = _mm256_load_si256((__m256i*) cc->s); \
    __m256i vc1 = _mm256_load_si256((__m256i*) &cc->s[4]); \
    __m256i vc2 = _mm256_load_si256((__m256i*) &cc->s[8]); \
    __m256i vc3 = _mm256_load_si256((__m256i*) &cc->s[12]); \
    __m256i va0 = _mm256_load_si256((__m256i*) aa->s); \
    __m256i va1 = _mm256_load_si256((__m256i*) &aa->s[4]); \
    __m256i va2 = _mm256_load_si256((__m256i*) &aa->s[8]); \
    __m256i va3 = _mm256_load_si256((__m256i*) &aa->s[12]); \
    va0 = _mm256_castpd_si256(_mm256_and_pd(_mm256_castsi256_pd(va0), \
                                            _mm256_castsi256_pd(vc0))); \
    va1 = _mm256_castpd_si256(_mm256_and_pd(_mm256_castsi256_pd(va1), \
                                            _mm256_castsi256_pd(vc1))); \
    va2 = _mm256_castpd_si256(_mm256_and_pd(_mm256_castsi256_pd(va2), \
                                            _mm256_castsi256_pd(vc2))); \
    va3 = _mm256_castpd_si256(_mm256_and_pd(_mm256_castsi256_pd(va3), \
                                            _mm256_castsi256_pd(vc3))); \
    __m256i vb0 = _mm256_load_si256((__m256i*) bb->s); \
    __m256i vb1 = _mm256_load_si256((__m256i*) &bb->s[4]); \
    __m256i vb2 = _mm256_load_si256((__m256i*) &bb->s[8]); \
    __m256i vb3 = _mm256_load_si256((__m256i*) &bb->s[12]); \
    vb0 = _mm256_castpd_si256(_mm256_andnot_pd(_mm256_castsi256_pd(vc0), \
                                               _mm256_castsi256_pd(vb0))); \
    vb1 = _mm256_castpd_si256(_mm256_andnot_pd(_mm256_castsi256_pd(vc1), \
                                               _mm256_castsi256_pd(vb1))); \
    vb2 = _mm256_castpd_si256(_mm256_andnot_pd(_mm256_castsi256_pd(vc2), \
                                               _mm256_castsi256_pd(vb2))); \
    vb3 = _mm256_castpd_si256(_mm256_andnot_pd(_mm256_castsi256_pd(vc3), \
                                               _mm256_castsi256_pd(vb3))); \
    va0 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(va0), \
                                            _mm256_castsi256_pd(vb0))); \
    va1 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(va1), \
                                            _mm256_castsi256_pd(vb1))); \
    va2 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(va2), \
                                            _mm256_castsi256_pd(vb2))); \
    va3 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(va3), \
                                            _mm256_castsi256_pd(vb3))); \
    _mm256_store_si256((__m256i*) aa->s, va0); \
    _mm256_store_si256((__m256i*) &aa->s[4], va1); \
    _mm256_store_si256((__m256i*) &aa->s[8], va2); \
    _mm256_store_si256((__m256i*) &aa->s[12], va3); \
} while(0)

#else

#define uint1024_t_mixi(a, b, c) do { \
    uint1024_t* const aa = (a); \
    const uint1024_t* const bb = (b); \
    const uint1024_t* const cc = (c); \
    aa->s[ 0] &= cc->s[ 0]; \
    aa->s[ 1] &= cc->s[ 1]; \
    aa->s[ 2] &= cc->s[ 2]; \
    aa->s[ 3] &= cc->s[ 3]; \
    aa->s[ 4] &= cc->s[ 4]; \
    aa->s[ 5] &= cc->s[ 5]; \
    aa->s[ 6] &= cc->s[ 6]; \
    aa->s[ 7] &= cc->s[ 7]; \
    aa->s[ 8] &= cc->s[ 8]; \
    aa->s[ 9] &= cc->s[ 9]; \
    aa->s[10] &= cc->s[10]; \
    aa->s[11] &= cc->s[11]; \
    aa->s[12] &= cc->s[12]; \
    aa->s[13] &= cc->s[13]; \
    aa->s[14] &= cc->s[14]; \
    aa->s[15] &= cc->s[15]; \
    aa->s[ 0] ^= bb->s[ 0] & ~cc->s[ 0]; \
    aa->s[ 1] ^= bb->s[ 1] & ~cc->s[ 1]; \
    aa->s[ 2] ^= bb->s[ 2] & ~cc->s[ 2]; \
    aa->s[ 3] ^= bb->s[ 3] & ~cc->s[ 3]; \
    aa->s[ 4] ^= bb->s[ 4] & ~cc->s[ 4]; \
    aa->s[ 5] ^= bb->s[ 5] & ~cc->s[ 5]; \
    aa->s[ 6] ^= bb->s[ 6] & ~cc->s[ 6]; \
    aa->s[ 7] ^= bb->s[ 7] & ~cc->s[ 7]; \
    aa->s[ 8] ^= bb->s[ 8] & ~cc->s[ 8]; \
    aa->s[ 9] ^= bb->s[ 9] & ~cc->s[ 9]; \
    aa->s[10] ^= bb->s[10] & ~cc->s[10]; \
    aa->s[11] ^= bb->s[11] & ~cc->s[11]; \
    aa->s[12] ^= bb->s[12] & ~cc->s[12]; \
    aa->s[13] ^= bb->s[13] & ~cc->s[13]; \
    aa->s[14] ^= bb->s[14] & ~cc->s[14]; \
    aa->s[15] ^= bb->s[15] & ~cc->s[15]; \
} while(0)

#endif

/* usage: Given 4 uint1024_t a, b, c, and d, compute a ^ (b & d) ^ (c & (~d))
 *      and store the result back into a. (in place)
 * params:
 *      1) a: ptr to struct uint1024_t
 *      2) b: ptr to struct uint1024_t
 *      3) c: ptr to struct uint1024_t
 *      4) d: ptr to struct uint1024_t
 * return: void */
#if defined(__AVX512F__)

#define uint1024_t_xor_mixi(a, b, c, d) do { \
    uint1024_t* const aa = (a); \
    const uint1024_t* const bb = (b); \
    const uint1024_t* const cc = (c); \
    const uint1024_t* const dd = (d); \
    __m512i vd0 = _mm512_load_si512((void const*) dd->s); \
    __m512i vd1 = _mm512_load_si512((void const*) &dd->s[8]); \
    __m512i vc0 = _mm512_load_si512((void const*) cc->s); \
    __m512i vc1 = _mm512_load_si512((void const*) &cc->s[8]); \
    vc0 = _mm512_andnot_si512(vd0, vc0); \
    vc1 = _mm512_andnot_si512(vd1, vc1); \
    __m512i vb0 = _mm512_load_si512((void const*) bb->s); \
    __m512i vb1 = _mm512_load_si512((void const*) &bb->s[8]); \
    vb0 = _mm512_and_si512(vd0, vb0); \
    vb1 = _mm512_and_si512(vd1, vb1); \
    vb0 = _mm512_xor_si512(vb0, vc0); \
    vb1 = _mm512_xor_si512(vb1, vc1); \
    __m512i va0 = _mm512_load_si512((void const*) aa->s); \
    __m512i va1 = _mm512_load_si512((void const*) &aa->s[8]); \
    va0 = _mm512_xor_si512(va0, vb0); \
    va1 = _mm512_xor_si512(va1, vb1); \
    _mm512_store_si512((void*) aa->s, va0); \
    _mm512_store_si512((void*) &aa->s[8], va1); \
} while(0)

#elif defined(__AVX2__)

#define uint1024_t_xor_mixi(a, b, c, d) do { \
    uint1024_t* const aa = (a); \
    const uint1024_t* const bb = (b); \
    const uint1024_t* const cc = (c); \
    const uint1024_t* const dd = (d); \
    __m256i vd0 = _mm256_load_si256((__m256i*) dd->s); \
    __m256i vd1 = _mm256_load_si256((__m256i*) &dd->s[4]); \
    __m256i vd2 = _mm256_load_si256((__m256i*) &dd->s[8]); \
    __m256i vd3 = _mm256_load_si256((__m256i*) &dd->s[12]); \
    __m256i vc0 = _mm256_load_si256((__m256i*) cc->s); \
    __m256i vc1 = _mm256_load_si256((__m256i*) &cc->s[4]); \
    __m256i vc2 = _mm256_load_si256((__m256i*) &cc->s[8]); \
    __m256i vc3 = _mm256_load_si256((__m256i*) &cc->s[12]); \
    vc0 = _mm256_andnot_si256(vd0, vc0); \
    vc1 = _mm256_andnot_si256(vd1, vc1); \
    vc2 = _mm256_andnot_si256(vd2, vc2); \
    vc3 = _mm256_andnot_si256(vd3, vc3); \
    __m256i vb0 = _mm256_load_si256((__m256i*) bb->s); \
    __m256i vb1 = _mm256_load_si256((__m256i*) &bb->s[4]); \
    __m256i vb2 = _mm256_load_si256((__m256i*) &bb->s[8]); \
    __m256i vb3 = _mm256_load_si256((__m256i*) &bb->s[12]); \
    vb0 = _mm256_and_si256(vd0, vb0); \
    vb1 = _mm256_and_si256(vd1, vb1); \
    vb2 = _mm256_and_si256(vd2, vb2); \
    vb3 = _mm256_and_si256(vd3, vb3); \
    vb0 = _mm256_xor_si256(vb0, vc0); \
    vb1 = _mm256_xor_si256(vb1, vc1); \
    vb2 = _mm256_xor_si256(vb2, vc2); \
    vb3 = _mm256_xor_si256(vb3, vc3); \
    __m256i va0 = _mm256_load_si256((__m256i*) aa->s); \
    __m256i va1 = _mm256_load_si256((__m256i*) &aa->s[4]); \
    __m256i va2 = _mm256_load_si256((__m256i*) &aa->s[8]); \
    __m256i va3 = _mm256_load_si256((__m256i*) &aa->s[12]); \
    va0 = _mm256_xor_si256(va0, vb0); \
    va1 = _mm256_xor_si256(va1, vb1); \
    va2 = _mm256_xor_si256(va2, vb2); \
    va3 = _mm256_xor_si256(va3, vb3); \
    _mm256_store_si256((__m256i*) aa->s, va0); \
    _mm256_store_si256((__m256i*) &aa->s[4], va1); \
    _mm256_store_si256((__m256i*) &aa->s[8], va2); \
    _mm256_store_si256((__m256i*) &aa->s[12], va3); \
} while(0)

#elif defined(__AVX__)

#define uint1024_t_xor_mixi(a, b, c, d) do { \
    uint1024_t* const aa = (a); \
    const uint1024_t* const bb = (b); \
    const uint1024_t* const cc = (c); \
    const uint1024_t* const dd = (d); \
    __m256i vd0 = _mm256_load_si256((__m256i*) dd->s); \
    __m256i vd1 = _mm256_load_si256((__m256i*) &dd->s[4]); \
    __m256i vd2 = _mm256_load_si256((__m256i*) &dd->s[8]); \
    __m256i vd3 = _mm256_load_si256((__m256i*) &dd->s[12]); \
    __m256i vc0 = _mm256_load_si256((__m256i*) cc->s); \
    __m256i vc1 = _mm256_load_si256((__m256i*) &cc->s[4]); \
    __m256i vc2 = _mm256_load_si256((__m256i*) &cc->s[8]); \
    __m256i vc3 = _mm256_load_si256((__m256i*) &cc->s[12]); \
    vc0 = _mm256_castpd_si256(_mm256_andnot_pd(_mm256_castsi256_pd(vd0), \
                                               _mm256_castsi256_pd(vc0))); \
    vc1 = _mm256_castpd_si256(_mm256_andnot_pd(_mm256_castsi256_pd(vd1), \
                                               _mm256_castsi256_pd(vc1))); \
    vc2 = _mm256_castpd_si256(_mm256_andnot_pd(_mm256_castsi256_pd(vd2), \
                                               _mm256_castsi256_pd(vc2))); \
    vc3 = _mm256_castpd_si256(_mm256_andnot_pd(_mm256_castsi256_pd(vd3), \
                                               _mm256_castsi256_pd(vc3))); \
    __m256i vb0 = _mm256_load_si256((__m256i*) bb->s); \
    __m256i vb1 = _mm256_load_si256((__m256i*) &bb->s[4]); \
    __m256i vb2 = _mm256_load_si256((__m256i*) &bb->s[8]); \
    __m256i vb3 = _mm256_load_si256((__m256i*) &bb->s[12]); \
    vb0 = _mm256_castpd_si256(_mm256_and_pd(_mm256_castsi256_pd(vd0), \
                                            _mm256_castsi256_pd(vb0))); \
    vb1 = _mm256_castpd_si256(_mm256_and_pd(_mm256_castsi256_pd(vd1), \
                                            _mm256_castsi256_pd(vb1))); \
    vb2 = _mm256_castpd_si256(_mm256_and_pd(_mm256_castsi256_pd(vd2), \
                                            _mm256_castsi256_pd(vb2))); \
    vb3 = _mm256_castpd_si256(_mm256_and_pd(_mm256_castsi256_pd(vd3), \
                                            _mm256_castsi256_pd(vb3))); \
    vb0 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(vb0), \
                                            _mm256_castsi256_pd(vc0))); \
    vb1 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(vb1), \
                                            _mm256_castsi256_pd(vc1))); \
    vb2 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(vb2), \
                                            _mm256_castsi256_pd(vc2))); \
    vb3 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(vb3), \
                                            _mm256_castsi256_pd(vc3))); \
    __m256i va0 = _mm256_load_si256((__m256i*) aa->s); \
    __m256i va1 = _mm256_load_si256((__m256i*) &aa->s[4]); \
    __m256i va2 = _mm256_load_si256((__m256i*) &aa->s[8]); \
    __m256i va3 = _mm256_load_si256((__m256i*) &aa->s[12]); \
    va0 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(va0), \
                                            _mm256_castsi256_pd(vb0))); \
    va1 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(va1), \
                                            _mm256_castsi256_pd(vb1))); \
    va2 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(va2), \
                                            _mm256_castsi256_pd(vb2))); \
    va3 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(va3), \
                                            _mm256_castsi256_pd(vb3))); \
    _mm256_store_si256((__m256i*) aa->s, va0); \
    _mm256_store_si256((__m256i*) &aa->s[4], va1); \
    _mm256_store_si256((__m256i*) &aa->s[8], va2); \
    _mm256_store_si256((__m256i*) &aa->s[12], va3); \
} while(0)

#else

#define uint1024_t_xor_mixi(a, b, c, d) do { \
    uint1024_t* const aa = (a); \
    const uint1024_t* const bb = (b); \
    const uint1024_t* const cc = (c); \
    const uint1024_t* const dd = (d); \
    aa->s[ 0] ^= bb->s[ 0] & dd->s[ 0]; \
    aa->s[ 1] ^= bb->s[ 1] & dd->s[ 1]; \
    aa->s[ 2] ^= bb->s[ 2] & dd->s[ 2]; \
    aa->s[ 3] ^= bb->s[ 3] & dd->s[ 3]; \
    aa->s[ 4] ^= bb->s[ 4] & dd->s[ 4]; \
    aa->s[ 5] ^= bb->s[ 5] & dd->s[ 5]; \
    aa->s[ 6] ^= bb->s[ 6] & dd->s[ 6]; \
    aa->s[ 7] ^= bb->s[ 7] & dd->s[ 7]; \
    aa->s[ 8] ^= bb->s[ 8] & dd->s[ 8]; \
    aa->s[ 9] ^= bb->s[ 9] & dd->s[ 9]; \
    aa->s[10] ^= bb->s[10] & dd->s[10]; \
    aa->s[11] ^= bb->s[11] & dd->s[11]; \
    aa->s[12] ^= bb->s[12] & dd->s[12]; \
    aa->s[13] ^= bb->s[13] & dd->s[13]; \
    aa->s[14] ^= bb->s[14] & dd->s[14]; \
    aa->s[15] ^= bb->s[15] & dd->s[15]; \
    aa->s[ 0] ^= cc->s[ 0] & ~dd->s[ 0]; \
    aa->s[ 1] ^= cc->s[ 1] & ~dd->s[ 1]; \
    aa->s[ 2] ^= cc->s[ 2] & ~dd->s[ 2]; \
    aa->s[ 3] ^= cc->s[ 3] & ~dd->s[ 3]; \
    aa->s[ 4] ^= cc->s[ 4] & ~dd->s[ 4]; \
    aa->s[ 5] ^= cc->s[ 5] & ~dd->s[ 5]; \
    aa->s[ 6] ^= cc->s[ 6] & ~dd->s[ 6]; \
    aa->s[ 7] ^= cc->s[ 7] & ~dd->s[ 7]; \
    aa->s[ 8] ^= cc->s[ 8] & ~dd->s[ 8]; \
    aa->s[ 9] ^= cc->s[ 9] & ~dd->s[ 9]; \
    aa->s[10] ^= cc->s[10] & ~dd->s[10]; \
    aa->s[11] ^= cc->s[11] & ~dd->s[11]; \
    aa->s[12] ^= cc->s[12] & ~dd->s[12]; \
    aa->s[13] ^= cc->s[13] & ~dd->s[13]; \
    aa->s[14] ^= cc->s[14] & ~dd->s[14]; \
    aa->s[15] ^= cc->s[15] & ~dd->s[15]; \
} while(0)

#endif

/* usage: Given 2 uint1024_t a and b, compute a&b and store the result into p
 * params:
 *      1) p: ptr to struct uint1024_t
 *      2) a: ptr to struct uint1024_t
 *      3) b: ptr to struct uint1024_t
 * return: void */
#if defined(__AVX512F__)

#define uint1024_t_and(p, a, b) do { \
    uint1024_t* const pp = (p); \
    const uint1024_t* const aa = (a); \
    const uint1024_t* const bb = (b); \
    __m512i va0 = _mm512_load_si512((void const*) aa->s); \
    __m512i va1 = _mm512_load_si512((void const*) &aa->s[8]); \
    __m512i vb0 = _mm512_load_si512((void const*) bb->s); \
    __m512i vb1 = _mm512_load_si512((void const*) &bb->s[8]); \
    va0 = _mm512_and_si512(vb0, va0); \
    va1 = _mm512_and_si512(vb1, va1); \
    _mm512_store_si512((void*) pp->s, va0); \
    _mm512_store_si512((void*) &pp->s[8], va1); \
} while(0)

#elif defined(__AVX2__)

#define uint1024_t_and(p, a, b) do { \
    uint1024_t* const pp = (p); \
    const uint1024_t* const aa = (a); \
    const uint1024_t* const bb = (b); \
    __m256i va0 = _mm256_load_si256((__m256i*) aa->s); \
    __m256i va1 = _mm256_load_si256((__m256i*) &aa->s[4]); \
    __m256i va2 = _mm256_load_si256((__m256i*) &aa->s[8]); \
    __m256i va3 = _mm256_load_si256((__m256i*) &aa->s[12]); \
    __m256i vb0 = _mm256_load_si256((__m256i*) bb->s); \
    __m256i vb1 = _mm256_load_si256((__m256i*) &bb->s[4]); \
    __m256i vb2 = _mm256_load_si256((__m256i*) &bb->s[8]); \
    __m256i vb3 = _mm256_load_si256((__m256i*) &bb->s[12]); \
    va0 = _mm256_and_si256(vb0, va0); \
    va1 = _mm256_and_si256(vb1, va1); \
    va2 = _mm256_and_si256(vb2, va2); \
    va3 = _mm256_and_si256(vb3, va3); \
    _mm256_store_si256((__m256i*) pp->s, va0); \
    _mm256_store_si256((__m256i*) &pp->s[4], va1); \
    _mm256_store_si256((__m256i*) &pp->s[8], va2); \
    _mm256_store_si256((__m256i*) &pp->s[12], va3); \
} while(0)

#elif defined(__AVX__)

#define uint1024_t_and(p, a, b) do { \
    uint1024_t* const pp = (p); \
    const uint1024_t* const aa = (a); \
    const uint1024_t* const bb = (b); \
    __m256i va0 = _mm256_load_si256((__m256i*) aa->s); \
    __m256i va1 = _mm256_load_si256((__m256i*) &aa->s[4]); \
    __m256i va2 = _mm256_load_si256((__m256i*) &aa->s[8]); \
    __m256i va3 = _mm256_load_si256((__m256i*) &aa->s[12]); \
    __m256i vb0 = _mm256_load_si256((__m256i*) bb->s); \
    __m256i vb1 = _mm256_load_si256((__m256i*) &bb->s[4]); \
    __m256i vb2 = _mm256_load_si256((__m256i*) &bb->s[8]); \
    __m256i vb3 = _mm256_load_si256((__m256i*) &bb->s[12]); \
    va0 = _mm256_castpd_si256(_mm256_and_pd(_mm256_castsi256_pd(vb0), \
                                            _mm256_castsi256_pd(va0))); \
    va1 = _mm256_castpd_si256(_mm256_and_pd(_mm256_castsi256_pd(vb1), \
                                            _mm256_castsi256_pd(va1))); \
    va2 = _mm256_castpd_si256(_mm256_and_pd(_mm256_castsi256_pd(vb2), \
                                            _mm256_castsi256_pd(va2))); \
    va3 = _mm256_castpd_si256(_mm256_and_pd(_mm256_castsi256_pd(vb3), \
                                            _mm256_castsi256_pd(va3))); \
    _mm256_store_si256((__m256i*) pp->s, va0); \
    _mm256_store_si256((__m256i*) &pp->s[4], va1); \
    _mm256_store_si256((__m256i*) &pp->s[8], va2); \
    _mm256_store_si256((__m256i*) &pp->s[12], va3); \
} while(0)

#else

#define uint1024_t_and(p, a, b) do { \
    uint1024_t* const pp = (p); \
    const uint1024_t* const aa = (a); \
    const uint1024_t* const bb = (b); \
    pp->s[ 0] = aa->s[ 0] & bb->s[ 0]; \
    pp->s[ 1] = aa->s[ 1] & bb->s[ 1]; \
    pp->s[ 2] = aa->s[ 2] & bb->s[ 2]; \
    pp->s[ 3] = aa->s[ 3] & bb->s[ 3]; \
    pp->s[ 4] = aa->s[ 4] & bb->s[ 4]; \
    pp->s[ 5] = aa->s[ 5] & bb->s[ 5]; \
    pp->s[ 6] = aa->s[ 6] & bb->s[ 6]; \
    pp->s[ 7] = aa->s[ 7] & bb->s[ 7]; \
    pp->s[ 8] = aa->s[ 8] & bb->s[ 8]; \
    pp->s[ 9] = aa->s[ 9] & bb->s[ 9]; \
    pp->s[10] = aa->s[10] & bb->s[10]; \
    pp->s[11] = aa->s[11] & bb->s[11]; \
    pp->s[12] = aa->s[12] & bb->s[12]; \
    pp->s[13] = aa->s[13] & bb->s[13]; \
    pp->s[14] = aa->s[14] & bb->s[14]; \
    pp->s[15] = aa->s[15] & bb->s[15]; \
} while(0)

#endif

/* usage: Given 2 uint1024_t a and b, compute a&b and store the result into a.
 *      (in place and)
 * params:
 *      1) a: ptr to struct uint1024_t
 *      2) b: ptr to struct uint1024_t
 * return: void */
#if defined(__AVX512F__)

#define uint1024_t_andi(a, b) do { \
    uint1024_t* const aa = (a); \
    const uint1024_t* const bb = (b); \
    __m512i va0 = _mm512_load_si512((void const*) aa->s); \
    __m512i va1 = _mm512_load_si512((void const*) &aa->s[8]); \
    __m512i vb0 = _mm512_load_si512((void const*) bb->s); \
    __m512i vb1 = _mm512_load_si512((void const*) &bb->s[8]); \
    va0 = _mm512_and_si512(vb0, va0); \
    va1 = _mm512_and_si512(vb1, va1); \
    _mm512_store_si512((void*) aa->s, va0); \
    _mm512_store_si512((void*) &aa->s[8], va1); \
} while(0)

#elif defined(__AVX2__)

#define uint1024_t_andi(a, b) do { \
    uint1024_t* const aa = (a); \
    const uint1024_t* const bb = (b); \
    __m256i va0 = _mm256_load_si256((__m256i*) aa->s); \
    __m256i va1 = _mm256_load_si256((__m256i*) &aa->s[4]); \
    __m256i va2 = _mm256_load_si256((__m256i*) &aa->s[8]); \
    __m256i va3 = _mm256_load_si256((__m256i*) &aa->s[12]); \
    __m256i vb0 = _mm256_load_si256((__m256i*) bb->s); \
    __m256i vb1 = _mm256_load_si256((__m256i*) &bb->s[4]); \
    __m256i vb2 = _mm256_load_si256((__m256i*) &bb->s[8]); \
    __m256i vb3 = _mm256_load_si256((__m256i*) &bb->s[12]); \
    va0 = _mm256_and_si256(vb0, va0); \
    va1 = _mm256_and_si256(vb1, va1); \
    va2 = _mm256_and_si256(vb2, va2); \
    va3 = _mm256_and_si256(vb3, va3); \
    _mm256_store_si256((__m256i*) aa->s, va0); \
    _mm256_store_si256((__m256i*) &aa->s[4], va1); \
    _mm256_store_si256((__m256i*) &aa->s[8], va2); \
    _mm256_store_si256((__m256i*) &aa->s[12], va3); \
} while(0)

#elif defined(__AVX__)

#define uint1024_t_andi(a, b) do { \
    uint1024_t* const aa = (a); \
    const uint1024_t* const bb = (b); \
    __m256i va0 = _mm256_load_si256((__m256i*) aa->s); \
    __m256i va1 = _mm256_load_si256((__m256i*) &aa->s[4]); \
    __m256i va2 = _mm256_load_si256((__m256i*) &aa->s[8]); \
    __m256i va3 = _mm256_load_si256((__m256i*) &aa->s[12]); \
    __m256i vb0 = _mm256_load_si256((__m256i*) bb->s); \
    __m256i vb1 = _mm256_load_si256((__m256i*) &bb->s[4]); \
    __m256i vb2 = _mm256_load_si256((__m256i*) &bb->s[8]); \
    __m256i vb3 = _mm256_load_si256((__m256i*) &bb->s[12]); \
    va0 = _mm256_castpd_si256(_mm256_and_pd(_mm256_castsi256_pd(vb0), \
                                            _mm256_castsi256_pd(va0))); \
    va1 = _mm256_castpd_si256(_mm256_and_pd(_mm256_castsi256_pd(vb1), \
                                            _mm256_castsi256_pd(va1))); \
    va2 = _mm256_castpd_si256(_mm256_and_pd(_mm256_castsi256_pd(vb2), \
                                            _mm256_castsi256_pd(va2))); \
    va3 = _mm256_castpd_si256(_mm256_and_pd(_mm256_castsi256_pd(vb3), \
                                            _mm256_castsi256_pd(va3))); \
    _mm256_store_si256((__m256i*) aa->s, va0); \
    _mm256_store_si256((__m256i*) &aa->s[4], va1); \
    _mm256_store_si256((__m256i*) &aa->s[8], va2); \
    _mm256_store_si256((__m256i*) &aa->s[12], va3); \
} while(0)

#else

#define uint1024_t_andi(a, b) do { \
    uint1024_t* const aa = (a); \
    const uint1024_t* const bb = (b); \
    aa->s[ 0] &= bb->s[ 0]; \
    aa->s[ 1] &= bb->s[ 1]; \
    aa->s[ 2] &= bb->s[ 2]; \
    aa->s[ 3] &= bb->s[ 3]; \
    aa->s[ 4] &= bb->s[ 4]; \
    aa->s[ 5] &= bb->s[ 5]; \
    aa->s[ 6] &= bb->s[ 6]; \
    aa->s[ 7] &= bb->s[ 7]; \
    aa->s[ 8] &= bb->s[ 8]; \
    aa->s[ 9] &= bb->s[ 9]; \
    aa->s[10] &= bb->s[10]; \
    aa->s[11] &= bb->s[11]; \
    aa->s[12] &= bb->s[12]; \
    aa->s[13] &= bb->s[13]; \
    aa->s[14] &= bb->s[14]; \
    aa->s[15] &= bb->s[15]; \
} while(0)

#endif

/* usage: Given 2 uint1024_t a and b, compute a&~b and store the result into p
 * params:
 *      1) p: ptr to struct uint1024_t
 *      2) a: ptr to struct uint1024_t
 *      3) b: ptr to struct uint1024_t
 * return: void */
#if defined(__AVX512F__)

#define uint1024_t_andn(p, a, b) do { \
    uint1024_t* const pp = (p); \
    const uint1024_t* const aa = (a); \
    const uint1024_t* const bb = (b); \
    __m512i va0 = _mm512_load_si512((void const*) aa->s); \
    __m512i va1 = _mm512_load_si512((void const*) &aa->s[8]); \
    __m512i vb0 = _mm512_load_si512((void const*) bb->s); \
    __m512i vb1 = _mm512_load_si512((void const*) &bb->s[8]); \
    va0 = _mm512_andnot_si512(vb0, va0); \
    va1 = _mm512_andnot_si512(vb1, va1); \
    _mm512_store_si512((void*) pp->s, va0); \
    _mm512_store_si512((void*) &pp->s[8], va1); \
} while(0)

#elif defined(__AVX2__)

#define uint1024_t_andn(p, a, b) do { \
    uint1024_t* const pp = (p); \
    const uint1024_t* const aa = (a); \
    const uint1024_t* const bb = (b); \
    __m256i va0 = _mm256_load_si256((__m256i*) aa->s); \
    __m256i va1 = _mm256_load_si256((__m256i*) &aa->s[4]); \
    __m256i va2 = _mm256_load_si256((__m256i*) &aa->s[8]); \
    __m256i va3 = _mm256_load_si256((__m256i*) &aa->s[12]); \
    __m256i vb0 = _mm256_load_si256((__m256i*) bb->s); \
    __m256i vb1 = _mm256_load_si256((__m256i*) &bb->s[4]); \
    __m256i vb2 = _mm256_load_si256((__m256i*) &bb->s[8]); \
    __m256i vb3 = _mm256_load_si256((__m256i*) &bb->s[12]); \
    va0 = _mm256_andnot_si256(vb0, va0); \
    va1 = _mm256_andnot_si256(vb1, va1); \
    va2 = _mm256_andnot_si256(vb2, va2); \
    va3 = _mm256_andnot_si256(vb3, va3); \
    _mm256_store_si256((__m256i*) pp->s, va0); \
    _mm256_store_si256((__m256i*) &pp->s[4], va1); \
    _mm256_store_si256((__m256i*) &pp->s[8], va2); \
    _mm256_store_si256((__m256i*) &pp->s[12], va3); \
} while(0)

#elif defined(__AVX__)

#define uint1024_t_andn(p, a, b) do { \
    uint1024_t* const pp = (p); \
    const uint1024_t* const aa = (a); \
    const uint1024_t* const bb = (b); \
    __m256i va0 = _mm256_load_si256((__m256i*) aa->s); \
    __m256i va1 = _mm256_load_si256((__m256i*) &aa->s[4]); \
    __m256i va2 = _mm256_load_si256((__m256i*) &aa->s[8]); \
    __m256i va3 = _mm256_load_si256((__m256i*) &aa->s[12]); \
    __m256i vb0 = _mm256_load_si256((__m256i*) bb->s); \
    __m256i vb1 = _mm256_load_si256((__m256i*) &bb->s[4]); \
    __m256i vb2 = _mm256_load_si256((__m256i*) &bb->s[8]); \
    __m256i vb3 = _mm256_load_si256((__m256i*) &bb->s[12]); \
    va0 = _mm256_castpd_si256(_mm256_andnot_pd(_mm256_castsi256_pd(vb0), \
                                               _mm256_castsi256_pd(va0))); \
    va1 = _mm256_castpd_si256(_mm256_andnot_pd(_mm256_castsi256_pd(vb1), \
                                               _mm256_castsi256_pd(va1))); \
    va2 = _mm256_castpd_si256(_mm256_andnot_pd(_mm256_castsi256_pd(vb2), \
                                               _mm256_castsi256_pd(va2))); \
    va3 = _mm256_castpd_si256(_mm256_andnot_pd(_mm256_castsi256_pd(vb3), \
                                               _mm256_castsi256_pd(va3))); \
    _mm256_store_si256((__m256i*) pp->s, va0); \
    _mm256_store_si256((__m256i*) &pp->s[4], va1); \
    _mm256_store_si256((__m256i*) &pp->s[8], va2); \
    _mm256_store_si256((__m256i*) &pp->s[12], va3); \
} while(0)

#else

#define uint1024_t_andn(p, a, b) do { \
    uint1024_t* const pp = (p); \
    const uint1024_t* const aa = (a); \
    const uint1024_t* const bb = (b); \
    pp->s[ 0] = aa->s[ 0] & ~bb->s[ 0]; \
    pp->s[ 1] = aa->s[ 1] & ~bb->s[ 1]; \
    pp->s[ 2] = aa->s[ 2] & ~bb->s[ 2]; \
    pp->s[ 3] = aa->s[ 3] & ~bb->s[ 3]; \
    pp->s[ 4] = aa->s[ 4] & ~bb->s[ 4]; \
    pp->s[ 5] = aa->s[ 5] & ~bb->s[ 5]; \
    pp->s[ 6] = aa->s[ 6] & ~bb->s[ 6]; \
    pp->s[ 7] = aa->s[ 7] & ~bb->s[ 7]; \
    pp->s[ 8] = aa->s[ 8] & ~bb->s[ 8]; \
    pp->s[ 9] = aa->s[ 9] & ~bb->s[ 9]; \
    pp->s[10] = aa->s[10] & ~bb->s[10]; \
    pp->s[11] = aa->s[11] & ~bb->s[11]; \
    pp->s[12] = aa->s[12] & ~bb->s[12]; \
    pp->s[13] = aa->s[13] & ~bb->s[13]; \
    pp->s[14] = aa->s[14] & ~bb->s[14]; \
    pp->s[15] = aa->s[15] & ~bb->s[15]; \
} while(0)

#endif

/* usage: Given 2 uint1024_t a and b, compute a&~b and store the result into a
 *      (in place negate-and).
 * params:
 *      1) a: ptr to struct uint1024_t
 *      2) b: ptr to struct uint1024_t
 * return: void */
#if defined(__AVX512F__)

#define uint1024_t_andni(a, b) do { \
    uint1024_t* const aa = (a); \
    const uint1024_t* const bb = (b); \
    __m512i va0 = _mm512_load_si512((void const*) aa->s); \
    __m512i va1 = _mm512_load_si512((void const*) &aa->s[8]); \
    __m512i vb0 = _mm512_load_si512((void const*) bb->s); \
    __m512i vb1 = _mm512_load_si512((void const*) &bb->s[8]); \
    va0 = _mm512_andnot_si512(vb0, va0); \
    va1 = _mm512_andnot_si512(vb1, va1); \
    _mm512_store_si512((void*) aa->s, va0); \
    _mm512_store_si512((void*) &aa->s[8], va1); \
} while(0)

#elif defined(__AVX2__)

#define uint1024_t_andni(a, b) do { \
    uint1024_t* const aa = (a); \
    const uint1024_t* const bb = (b); \
    __m256i va0 = _mm256_load_si256((__m256i*) aa->s); \
    __m256i va1 = _mm256_load_si256((__m256i*) &aa->s[4]); \
    __m256i va2 = _mm256_load_si256((__m256i*) &aa->s[8]); \
    __m256i va3 = _mm256_load_si256((__m256i*) &aa->s[12]); \
    __m256i vb0 = _mm256_load_si256((__m256i*) bb->s); \
    __m256i vb1 = _mm256_load_si256((__m256i*) &bb->s[4]); \
    __m256i vb2 = _mm256_load_si256((__m256i*) &bb->s[8]); \
    __m256i vb3 = _mm256_load_si256((__m256i*) &bb->s[12]); \
    va0 = _mm256_andnot_si256(vb0, va0); \
    va1 = _mm256_andnot_si256(vb1, va1); \
    va2 = _mm256_andnot_si256(vb2, va2); \
    va3 = _mm256_andnot_si256(vb3, va3); \
    _mm256_store_si256((__m256i*) aa->s, va0); \
    _mm256_store_si256((__m256i*) &aa->s[4], va1); \
    _mm256_store_si256((__m256i*) &aa->s[8], va2); \
    _mm256_store_si256((__m256i*) &aa->s[12], va3); \
} while(0)

#elif defined(__AVX__)

#define uint1024_t_andni(a, b) do { \
    uint1024_t* const aa = (a); \
    const uint1024_t* const bb = (b); \
    __m256i va0 = _mm256_load_si256((__m256i*) aa->s); \
    __m256i va1 = _mm256_load_si256((__m256i*) &aa->s[4]); \
    __m256i va2 = _mm256_load_si256((__m256i*) &aa->s[8]); \
    __m256i va3 = _mm256_load_si256((__m256i*) &aa->s[12]); \
    __m256i vb0 = _mm256_load_si256((__m256i*) bb->s); \
    __m256i vb1 = _mm256_load_si256((__m256i*) &bb->s[4]); \
    __m256i vb2 = _mm256_load_si256((__m256i*) &bb->s[8]); \
    __m256i vb3 = _mm256_load_si256((__m256i*) &bb->s[12]); \
    va0 = _mm256_castpd_si256(_mm256_andnot_pd(_mm256_castsi256_pd(vb0), \
                                               _mm256_castsi256_pd(va0))); \
    va1 = _mm256_castpd_si256(_mm256_andnot_pd(_mm256_castsi256_pd(vb1), \
                                               _mm256_castsi256_pd(va1))); \
    va2 = _mm256_castpd_si256(_mm256_andnot_pd(_mm256_castsi256_pd(vb2), \
                                               _mm256_castsi256_pd(va2))); \
    va3 = _mm256_castpd_si256(_mm256_andnot_pd(_mm256_castsi256_pd(vb3), \
                                               _mm256_castsi256_pd(va3))); \
    _mm256_store_si256((__m256i*) aa->s, va0); \
    _mm256_store_si256((__m256i*) &aa->s[4], va1); \
    _mm256_store_si256((__m256i*) &aa->s[8], va2); \
    _mm256_store_si256((__m256i*) &aa->s[12], va3); \
} while(0)

#else

#define uint1024_t_andni(a, b) do { \
    uint1024_t* const aa = (a); \
    const uint1024_t* const bb = (b); \
    aa->s[ 0] &= ~bb->s[ 0]; \
    aa->s[ 1] &= ~bb->s[ 1]; \
    aa->s[ 2] &= ~bb->s[ 2]; \
    aa->s[ 3] &= ~bb->s[ 3]; \
    aa->s[ 4] &= ~bb->s[ 4]; \
    aa->s[ 5] &= ~bb->s[ 5]; \
    aa->s[ 6] &= ~bb->s[ 6]; \
    aa->s[ 7] &= ~bb->s[ 7]; \
    aa->s[ 8] &= ~bb->s[ 8]; \
    aa->s[ 9] &= ~bb->s[ 9]; \
    aa->s[10] &= ~bb->s[10]; \
    aa->s[11] &= ~bb->s[11]; \
    aa->s[12] &= ~bb->s[12]; \
    aa->s[13] &= ~bb->s[13]; \
    aa->s[14] &= ~bb->s[14]; \
    aa->s[15] &= ~bb->s[15]; \
} while(0)

#endif

/* usage: Given 2 uint1024_t a and b, compute a|b and store the result into a.
 *      (in place or)
 * params:
 *      1) a: ptr to struct uint1024_t
 *      2) b: ptr to struct uint1024_t
 * return: void */
#if defined(__AVX512F__)

#define uint1024_t_ori(a, b) do { \
    uint1024_t* const aa = (a); \
    const uint1024_t* const bb = (b); \
    __m512i va0 = _mm512_load_si512((void const*) aa->s); \
    __m512i va1 = _mm512_load_si512((void const*) &aa->s[8]); \
    __m512i vb0 = _mm512_load_si512((void const*) bb->s); \
    __m512i vb1 = _mm512_load_si512((void const*) &bb->s[8]); \
    va0 = _mm512_or_si512(vb0, va0); \
    va1 = _mm512_or_si512(vb1, va1); \
    _mm512_store_si512((void*) aa->s, va0); \
    _mm512_store_si512((void*) &aa->s[8], va1); \
} while(0)

#elif defined(__AVX2__)

#define uint1024_t_ori(a, b) do { \
    uint1024_t* const aa = (a); \
    const uint1024_t* const bb = (b); \
    __m256i va0 = _mm256_load_si256((__m256i*) aa->s); \
    __m256i va1 = _mm256_load_si256((__m256i*) &aa->s[4]); \
    __m256i va2 = _mm256_load_si256((__m256i*) &aa->s[8]); \
    __m256i va3 = _mm256_load_si256((__m256i*) &aa->s[12]); \
    __m256i vb0 = _mm256_load_si256((__m256i*) bb->s); \
    __m256i vb1 = _mm256_load_si256((__m256i*) &bb->s[4]); \
    __m256i vb2 = _mm256_load_si256((__m256i*) &bb->s[8]); \
    __m256i vb3 = _mm256_load_si256((__m256i*) &bb->s[12]); \
    va0 = _mm256_or_si256(va0, vb0); \
    va1 = _mm256_or_si256(va1, vb1); \
    va2 = _mm256_or_si256(va2, vb2); \
    va3 = _mm256_or_si256(va3, vb3); \
    _mm256_store_si256((__m256i*) aa->s, va0); \
    _mm256_store_si256((__m256i*) &aa->s[4], va1); \
    _mm256_store_si256((__m256i*) &aa->s[8], va2); \
    _mm256_store_si256((__m256i*) &aa->s[12], va3); \
} while(0)

#elif defined(__AVX__)

#define uint1024_t_ori(a, b) do { \
    uint1024_t* const aa = (a); \
    const uint1024_t* const bb = (b); \
    __m256i va0 = _mm256_load_si256((__m256i*) aa->s); \
    __m256i va1 = _mm256_load_si256((__m256i*) &aa->s[4]); \
    __m256i va2 = _mm256_load_si256((__m256i*) &aa->s[8]); \
    __m256i va3 = _mm256_load_si256((__m256i*) &aa->s[12]); \
    __m256i vb0 = _mm256_load_si256((__m256i*) bb->s); \
    __m256i vb1 = _mm256_load_si256((__m256i*) &bb->s[4]); \
    __m256i vb2 = _mm256_load_si256((__m256i*) &bb->s[8]); \
    __m256i vb3 = _mm256_load_si256((__m256i*) &bb->s[12]); \
    va0 = _mm256_castpd_si256(_mm256_or_pd(_mm256_castsi256_pd(va0), \
                                           _mm256_castsi256_pd(vb0))); \
    va1 = _mm256_castpd_si256(_mm256_or_pd(_mm256_castsi256_pd(va1), \
                                           _mm256_castsi256_pd(vb1))); \
    va2 = _mm256_castpd_si256(_mm256_or_pd(_mm256_castsi256_pd(va2), \
                                           _mm256_castsi256_pd(vb2))); \
    va3 = _mm256_castpd_si256(_mm256_or_pd(_mm256_castsi256_pd(va3), \
                                           _mm256_castsi256_pd(vb3))); \
    _mm256_store_si256((__m256i*) aa->s, va0); \
    _mm256_store_si256((__m256i*) &aa->s[4], va1); \
    _mm256_store_si256((__m256i*) &aa->s[8], va2); \
    _mm256_store_si256((__m256i*) &aa->s[12], va3); \
} while(0)


#else

#define uint1024_t_ori(a, b) do { \
    uint1024_t* const aa = (a); \
    const uint1024_t* const bb = (b); \
    aa->s[ 0] |= bb->s[ 0]; \
    aa->s[ 1] |= bb->s[ 1]; \
    aa->s[ 2] |= bb->s[ 2]; \
    aa->s[ 3] |= bb->s[ 3]; \
    aa->s[ 4] |= bb->s[ 4]; \
    aa->s[ 5] |= bb->s[ 5]; \
    aa->s[ 6] |= bb->s[ 6]; \
    aa->s[ 7] |= bb->s[ 7]; \
    aa->s[ 8] |= bb->s[ 8]; \
    aa->s[ 9] |= bb->s[ 9]; \
    aa->s[10] |= bb->s[10]; \
    aa->s[11] |= bb->s[11]; \
    aa->s[12] |= bb->s[12]; \
    aa->s[13] |= bb->s[13]; \
    aa->s[14] |= bb->s[14]; \
    aa->s[15] |= bb->s[15]; \
} while(0)

#endif

/* usage: Given 1 uint1024_t a, set a to all 1's.
 * params:
 *      1) a: ptr to struct uint1024_t
 * return: void */
#define uint1024_t_set_max(a) do { \
    memset((a), UINT8_MAX, sizeof(uint1024_t)); \
} while(0)

/* usage: Given 1 uint1024_t a, and an integer i, return the i-th bit.
 *      The LSB is 0-th bit.
 * params:
 *      1) a: ptr to struct uint1024_t
 *      2) i: index to the bit, from 0 ~ 1023. Do not perform arithmetic here.
 * return: the bit */
#define uint1024_t_at(a, i) \
    (((a)->s[(i) >> 6] >> ((i) & 0x3FULL)) & 0x1ULL)

/* usage: Given 1 uint1024_t a, an integer i, set the i-th bit to the given
 *      value. The LSB is 0-th bit.
 * params:
 *      1) a: ptr to struct uint1024_t
 *      2) i: index to the bit, from 0 ~ 1023
 *      3) v: true if the bit is 1, otherwise false
 * return: void */
#define uint1024_t_set_at(a, i, v) do { \
    uint1024_t* const aa = (a); \
    uint64_t const ii = (i); \
    aa->s[ii >> 6] &= ~(0x1ULL << (ii & 0x3FULL)); \
    aa->s[ii >> 6] |= ((uint64_t) (v)) << (ii & 0x3FULL); \
} while(0)

/* usage: Given 1 uint1024_t a, an integer i, toggle the i-th bit.
 *      The LSB is 0-th bit.
 * params:
 *      1) a: ptr to struct uint1024_t
 *      2) i: index to the bit, from 0 ~ 1023
 * return: void */
#define uint1024_t_toggle_at(a, i) do { \
    uint1024_t* const aa = (a); \
    uint64_t const ii = (i); \
    aa->s[ii >> 6] ^= 0x1ULL << (ii & 0x3FULL); \
} while(0)

/* usage: Given 1 uint1024_t a, an integer i, set the i-th bit to 0.
 *      The LSB is 0-th bit.
 * params:
 *      1) a: ptr to struct uint1024_t
 *      2) i: index to the bit, from 0 ~ 1023
 * return: void */
#define uint1024_t_clear_at(a, i) do { \
    uint64_t const ii = (i); \
    (a)->s[ii >> 6] &= ~(0x1ULL << (ii & 0x3FULL)); \
} while(0)

/* usage: Given 1 uint1024_t a, find indices of all set bits in a
 * params:
 *      1) a: ptr to struct uint1024_t
 *      2) res: an uint16_t array for storing the indices, must hold at least
 *              1024 elements
 * return: the number of set bits */
static inline int
uint1024_t_sbpos(const uint1024_t* const restrict a,
                 uint16_t* const restrict res) {
    uint64_t base = 0x0ULL;
    const uint64_t inc64 = 0x0040004000400040ULL;
    int sbnum = sbidx_in_64b_sz16(res, base, uint1024_t_64b_at(a, 0));
    base += inc64;
    sbnum += sbidx_in_64b_sz16(res + sbnum, base, uint1024_t_64b_at(a, 1));
    base += inc64;
    sbnum += sbidx_in_64b_sz16(res + sbnum, base, uint1024_t_64b_at(a, 2));
    base += inc64;
    sbnum += sbidx_in_64b_sz16(res + sbnum, base, uint1024_t_64b_at(a, 3));
    base += inc64;
    sbnum += sbidx_in_64b_sz16(res + sbnum, base, uint1024_t_64b_at(a, 4));
    base += inc64;
    sbnum += sbidx_in_64b_sz16(res + sbnum, base, uint1024_t_64b_at(a, 5));
    base += inc64;
    sbnum += sbidx_in_64b_sz16(res + sbnum, base, uint1024_t_64b_at(a, 6));
    base += inc64;
    sbnum += sbidx_in_64b_sz16(res + sbnum, base, uint1024_t_64b_at(a, 7));
    base += inc64;
    sbnum += sbidx_in_64b_sz16(res + sbnum, base, uint1024_t_64b_at(a, 8));
    base += inc64;
    sbnum += sbidx_in_64b_sz16(res + sbnum, base, uint1024_t_64b_at(a, 9));
    base += inc64;
    sbnum += sbidx_in_64b_sz16(res + sbnum, base, uint1024_t_64b_at(a, 10));
    base += inc64;
    sbnum += sbidx_in_64b_sz16(res + sbnum, base, uint1024_t_64b_at(a, 11));
    base += inc64;
    sbnum += sbidx_in_64b_sz16(res + sbnum, base, uint1024_t_64b_at(a, 12));
    base += inc64;
    sbnum += sbidx_in_64b_sz16(res + sbnum, base, uint1024_t_64b_at(a, 13));
    base += inc64;
    sbnum += sbidx_in_64b_sz16(res + sbnum, base, uint1024_t_64b_at(a, 14));
    base += inc64;
    sbnum += sbidx_in_64b_sz16(res + sbnum, base, uint1024_t_64b_at(a, 15));
    assert(sbnum <= 1024);
    return sbnum;
}

#if defined(__AVX512F__)

/* usage: Given 1 uint1024_t a whose content is stored in 2 __m512i registers,
 *      find indices of all set bits in a
 * params:
 *      1) res: an uint16_t array for storing the indices, must hold at least
 *              1024 elements
 *      2) lv: register that store the lower 512 bits of a
 *      3) hv: register that store the higher 512 bits of a
 * return: the number of set bits */
static inline int
uint1024_t_sbpos_from_reg_avx512(uint16_t* const res, __m512i lv, __m512i hv) {
    uint64_t base = 0x0ULL;
    const uint64_t inc64 = 0x0040004000400040ULL;
    __m256i v0 = _mm512_extracti64x4_epi64(lv, 0);
    __m256i v1 = _mm512_extracti64x4_epi64(lv, 1);
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
    base += inc64;
    v0 = _mm512_extracti64x4_epi64(hv, 0);
    v1 = _mm512_extracti64x4_epi64(hv, 1);
    sbnum += sbidx_in_64b_sz16(res + sbnum, base, _mm256_extract_epi64(v0, 0));
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
    assert(sbnum <= 1024);
    return sbnum;
}

#endif

#if defined(__AVX__)

/* usage: Given 1 uint1024_t a whose content is stored in 4 __m256i registers,
 *      find indices of all set bits in a
 * params:
 *      1) res: an uint16_t array for storing the indices, must hold at least
 *              1024 elements
 *      2) v0: register that store the 1st 256 bits of uint1024_t a
 *      3) v1: register that store the 2nd 256 bits of uint1024_t a
 *      4) v2: register that store the 3rd 256 bits of uint1024_t a
 *      5) v3: register that store the 4th 256 bits of uint1024_t a
 * return: the number of set bits */
static inline int
uint1024_t_sbpos_from_reg(uint16_t* const res, __m256i v0, __m256i v1,
                          __m256i v2, __m256i v3) {
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
    base += inc64;
    sbnum += sbidx_in_64b_sz16(res + sbnum, base, _mm256_extract_epi64(v2, 0));
    base += inc64;
    sbnum += sbidx_in_64b_sz16(res + sbnum, base, _mm256_extract_epi64(v2, 1));
    base += inc64;
    sbnum += sbidx_in_64b_sz16(res + sbnum, base, _mm256_extract_epi64(v2, 2));
    base += inc64;
    sbnum += sbidx_in_64b_sz16(res + sbnum, base, _mm256_extract_epi64(v2, 3));
    base += inc64;
    sbnum += sbidx_in_64b_sz16(res + sbnum, base, _mm256_extract_epi64(v3, 0));
    base += inc64;
    sbnum += sbidx_in_64b_sz16(res + sbnum, base, _mm256_extract_epi64(v3, 1));
    base += inc64;
    sbnum += sbidx_in_64b_sz16(res + sbnum, base, _mm256_extract_epi64(v3, 2));
    base += inc64;
    sbnum += sbidx_in_64b_sz16(res + sbnum, base, _mm256_extract_epi64(v3, 3));
    assert(sbnum <= 1024);
    return sbnum;
}

#endif

#endif /* __BLK_LANCZOS_UINT1024_T_H__ */
