/* uint256_t.h: header file for struct uint256_t */

/* Note that some of the following optimized macros use floating point domain
 * instructions when the integer counterparts are not available. E.g. to xor
 * two 256-bit registers, VXORPD, which might only execute in floating point
 * domain, is used when the AVX2 instruction VPXOR is not available. This might
 * have slightly worse performance on some AMD architectures and Intel
 * architectures before Skylake. */

#ifndef __BLK_LANCZOS_UINT256_T_H__
#define __BLK_LANCZOS_UINT256_T_H__

#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <stdalign.h>

#if defined(__AVX__) || defined(__AVX2__)
#include <immintrin.h>
#endif

#include "util.h"
#include "bitmap_table.h"

/* ========================================================================
 * struct uint256_t definition
 * ======================================================================== */

typedef struct {
    alignas(32) uint64_t s[4];
} uint256_t;

/* ========================================================================
 * functions prototypes
 * ======================================================================== */

/* usage: Given 1 uint256_t a, treat it as uint64_t[4] and
 *      return the specified 64 bits.
 * params:
 *      1) a: ptr to struct uint256_t
 *      2) i: index of the slot; from 0 ~ 3
 * return: the specified 64-bit as a uint64_t */
#define uint256_t_64b_at(a, i) \
    ((a)->s[(i)])

/* usage: Given 1 uint256_t a, treat it as uint64_t[4] and set
 *      the all 4 64-bit integers to the given value.
 * params:
 *      1) a: ptr to struct uint256_t
 *      3) v: the new value
 * return: void */
static force_inline void
uint256_t_set1_64b(uint256_t* a, uint64_t v) {
#if defined(__AVX2__)
    __m256i av = _mm256_set1_epi64x(v);
    _mm256_store_si256((__m256i*)a->s, av);
#else
    a->s[0] = v;
    a->s[1] = v;
    a->s[2] = v;
    a->s[3] = v;
#endif
}

/* usage: Given 1 uint256_t a, set a to all 0's.
 * params:
 *      1) a: ptr to struct uint256_t
 * return: void */
#define uint256_t_zero(a) do { \
    memset((a), 0x0, sizeof(uint256_t)); \
} while(0)

/* usage: Given 1 uint256_t, set a to all 1's.
 * params:
 *      1) a: ptr to struct uint256_t
 * return: void */
static force_inline void
uint256_t_max(uint256_t* a) {
#if defined(__AVX2__)
    __m256i v = _mm256_set1_epi64x(-1);
    _mm256_store_si256((__m256i*) a->s, v);
#else
    a->s[0] = UINT64_MAX;
    a->s[1] = UINT64_MAX;
    a->s[2] = UINT64_MAX;
    a->s[3] = UINT64_MAX;
#endif
}

/* usage: Given 1 uint256_t a, return true if all 1's. Otherwise false;
 * params:
 *      1) a: ptr to struct uint256_t
 * return: void */
static force_inline bool
uint256_t_is_max(const uint256_t* const a) {
#if defined(__AVX2__)
    __m256i v = _mm256_load_si256((__m256i*)a->s);
    __m256i maxv = _mm256_set1_epi64x(-1);
    return _mm256_testc_si256(v, maxv);
#else
    for(uint32_t i = 0; i < 4; ++i) {
        if(a->s[i] != UINT64_MAX)
            return false;
    }
    return true;
#endif
}

/* usage: Given 1 uint256_t a, return true if all 0's. Otherwise false;
 * params:
 *      1) a: ptr to struct uint256_t
 * return: void */
static inline bool
uint256_t_is_zero(const uint256_t* const a) {
#if defined(__AVX2__)
    __m256i v = _mm256_load_si256((__m256i*)a->s);
    __m256i maxv = _mm256_set1_epi64x(-1);
    return _mm256_testz_si256(v, maxv);
#else
    return !(a->s[0] | a->s[1] | a->s[2] | a->s[3]);
#endif
}

/* usage: Given 1 uint256_t a, return false if all 0's. Otherwise true;
 * params:
 *      1) a: ptr to struct uint256_t
 * return: void */
static inline bool
uint256_t_is_not_zero(const uint256_t* const a) {
    return !uint256_t_is_zero(a);
}

/* usage: Given 2 uint256_t a and b, check if they are the same.
 * params:
 *      1) a: ptr to struct uint256_t
 *      2) b: ptr to struct uint256_t
 * return: true if they are the same. otherwise false */
static inline bool
uint256_t_equal(const uint256_t* const a, const uint256_t* const b) {
    return (0 == memcmp(a, b, sizeof(uint256_t)));
}

/* usage: Given 1 uint256_t a, bitwise negate it.
 * params:
 *      1) a: ptr to struct uint256_t
 * return: void */
#if defined(__AVX2__)

#define uint256_t_negi(a) do { \
    uint256_t* const aa = (a); \
    __m256i v = _mm256_load_si256((__m256i*) aa->s); \
    __m256i all1s = _mm256_set1_epi32(-1); \
    _mm256_store_si256((__m256i*) aa->s, _mm256_xor_si256(v, all1s)); \
} while(0)

#elif defined(__AVX__)

#define uint256_t_negi(a) do { \
    uint256_t* const aa = (a); \
    __m256i v = _mm256_load_si256((__m256i*) aa->s); \
    __m256i all1s = _mm256_set1_epi32(-1); \
    v = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(v), \
                                          _mm256_castsi256_pd(all1s))); \
    _mm256_store_si256((__m256i*) aa->s, v); \
} while(0)

#else

#define uint256_t_negi(a) do { \
    uint256_t* const aa = (a); \
    aa->s[0] = ~aa->s[0]; \
    aa->s[1] = ~aa->s[1]; \
    aa->s[2] = ~aa->s[2]; \
    aa->s[3] = ~aa->s[3]; \
} while(0)

#endif

/* usage: Given 1 uint256_t a, return the number of bits that are set to 1 in a
 * params:
 *      1) a: ptr to struct uint256_t
 * return: the number of 1's in the uint256_t */
static inline uint64_t
uint256_t_popcount(const uint256_t* const a) {
    return (__builtin_popcountll(a->s[0]) + __builtin_popcountll(a->s[1]) +
            __builtin_popcountll(a->s[2]) + __builtin_popcountll(a->s[3]));
}

/* usage: Given 1 uint256_t a, set a to a random value.
 * params:
 *      1) a: ptr to struct uint256_t
 * return: void */
#define uint256_t_rand(a) do { \
    int* const buf = (int*) ((a)->s); \
    buf[0] = rand(); \
    buf[1] = rand(); \
    buf[2] = rand(); \
    buf[3] = rand(); \
    buf[4] = rand(); \
    buf[5] = rand(); \
    buf[6] = rand(); \
    buf[7] = rand(); \
} while(0)

/* usage: Given 2 uint256_t a and b, copy b into a.
 * params:
 *      1) a: ptr to struct uint256_t
 *      2) b: ptr to struct uint256_t
 * return: void */
#define uint256_t_copy(a, b) do { \
    memcpy((a), (b), sizeof(uint256_t)); \
} while(0)

/* usage: Given 2 uint256_t a and b, swap their values
 * params:
 *      1) a: ptr to struct uint256_t
 *      2) b: ptr to struct uint256_t
 * return: void */
static force_inline void
uint256_t_swap(uint256_t* restrict a, uint256_t* restrict b) {
#if defined(__AVX__)
    __m256i* s1 = (__m256i*) a->s;
    __m256i* s2 = (__m256i*) b->s;
    __m256i va = _mm256_load_si256(s1);
    __m256i vb = _mm256_load_si256(s2);
    _mm256_store_si256(s1, vb);
    _mm256_store_si256(s2, va);
#else
    uint64_t t0 = a->s[0];
    uint64_t t1 = a->s[1];
    uint64_t t2 = a->s[2];
    uint64_t t3 = a->s[3];
    a->s[0] = b->s[0];
    a->s[1] = b->s[1];
    a->s[2] = b->s[2];
    a->s[3] = b->s[3];
    b->s[0] = t0;
    b->s[1] = t1;
    b->s[2] = t2;
    b->s[3] = t3;
#endif
}

/* usage: Given 2 uint256_t a and b, copy b into a and then negate a bitwise.
 * params:
 *      1) a: ptr to struct uint256_t
 *      2) b: ptr to struct uint256_t
 * return: void */
#if defined(__AVX2__)

#define uint256_t_neg(a, b) do { \
    uint256_t* const aa = (a); \
    const uint256_t* const bb = (b); \
    __m256i v = _mm256_load_si256((__m256i*) bb->s); \
    __m256i all1s = _mm256_set1_epi32(-1); \
    _mm256_store_si256((__m256i*) aa->s, _mm256_xor_si256(v, all1s)); \
} while(0)

#elif defined(__AVX__)

#define uint256_t_neg(a, b) do { \
    uint256_t* const aa = (a); \
    const uint256_t* const bb = (b); \
    __m256i v = _mm256_load_si256((__m256i*) bb->s); \
    __m256i all1s = _mm256_set1_epi32(-1); \
    v = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(v), \
                                          _mm256_castsi256_pd(all1s))); \
    _mm256_store_si256((__m256i*) aa->s, v); \
} while(0)

#else

#define uint256_t_neg(a, b) do { \
    uint256_t* const aa = (a); \
    const uint256_t* const bb = (b); \
    aa->s[0] = ~bb->s[0]; \
    aa->s[1] = ~bb->s[1]; \
    aa->s[2] = ~bb->s[2]; \
    aa->s[3] = ~bb->s[3]; \
} while(0)

#endif

/* usage: Given 2 uint256_t a and b, compute m^n and store the result into p
 * params:
 *      1) p: ptr to struct uint256_t
 *      2) a: ptr to struct uint256_t
 *      3) b: ptr to struct uint256_t
 * return: void */
#ifdef __AVX2__

#define uint256_t_xor(p, a, b) do { \
    uint256_t* const pp = (p); \
    const uint256_t* const aa = (a); \
    const uint256_t* const bb = (b); \
    __m256i va = _mm256_load_si256((__m256i*) aa->s); \
    __m256i vb = _mm256_load_si256((__m256i*) bb->s); \
    va = _mm256_xor_si256(va, vb); \
    _mm256_store_si256((__m256i*) pp->s, va); \
} while(0)

#elif defined(__AVX__)

#define uint256_t_xor(p, a, b) do { \
    uint256_t* const pp = (p); \
    const uint256_t* const aa = (a); \
    const uint256_t* const bb = (b); \
    __m256i va = _mm256_load_si256((__m256i*) aa->s); \
    __m256i vb = _mm256_load_si256((__m256i*) bb->s); \
    va = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(vb), \
                                           _mm256_castsi256_pd(va))); \
    _mm256_store_si256((__m256i*) pp->s, va); \
} while(0)

#else

#define uint256_t_xor(p, a, b) do { \
    uint256_t* const pp = (p); \
    const uint256_t* const aa = (a); \
    const uint256_t* const bb = (b); \
    pp->s[0] = aa->s[0] ^ bb->s[0]; \
    pp->s[1] = aa->s[1] ^ bb->s[1]; \
    pp->s[2] = aa->s[2] ^ bb->s[2]; \
    pp->s[3] = aa->s[3] ^ bb->s[3]; \
} while(0)

#endif

#if defined(__AVX__)

/* usage: Given 1 uint256_t a and 1 __m256i register, load the content of
 *      a into the register
 * params:
 *      1) a: ptr to struct uint256_t
 *      2) va: register to store the content of uint256_t a
 * return: void */
#define uint256_t_load_to_reg(a, va) do { \
    const uint256_t* const aa = (a); \
    va = _mm256_load_si256((__m256i*) aa->s); \
} while(0)

/* usage: Given 1 uint256_t a and 1 __m256i register, store the content of
 *      the register into a
 * params:
 *      1) a: ptr to struct uint256_t
 *      2) va: register to store the content of uint256_t a
 * return: void */
#define uint256_t_load_from_reg(a, va) do { \
    uint256_t* const aa = (a); \
    _mm256_store_si256((__m256i*) aa->s, va); \
} while(0)

/* usage: Given 1 uint256_t a and 1 __m256i register, store the content of
 *      the register into a with non-temporal hint.
 * params:
 *      1) a: ptr to struct uint256_t
 *      2) va: register to store the content of uint256_t a
 * return: void */
#define uint256_t_load_from_reg_nt(a, va) do { \
    uint256_t* const aa = (a); \
    _mm256_stream_si256((__m256i*) aa->s, va); \
} while(0)

#endif

#if defined(__AVX2__)

/* usage: Given 2 uint256_t a and b, where the content of a is stored in 1
 *      __m256i register, compute a^b and store the result back into the
 *      register (in place xor).
 * params:
 *      1) b: ptr to struct uint256_t
 *      2) va: register that store the content of uint256_t a
 * return: void */
#define uint256_t_xori_to_reg(b, va) do { \
    const uint256_t* const bb = (b); \
    __m256i vb = _mm256_load_si256((__m256i*) bb->s); \
    va = _mm256_xor_si256(vb, va); \
} while(0)

/* usage: Given 2 uint256_t a and b, where the content of b is stored in 1
 *      __m256i register, compute a^b and store the result back into a
 *      (in place xor).
 * params:
 *      1) a: ptr to struct uint256_t
 *      2) vb: register that store the content of uint256_t b
 * return: void */
#define uint256_t_xori_from_reg(a, vb) do { \
    uint256_t* const aa = (a); \
    __m256i va = _mm256_load_si256((__m256i*) aa->s); \
    va = _mm256_xor_si256(vb, va); \
    _mm256_store_si256((__m256i*) aa->s, va); \
} while(0)

/* usage: Given 2 uint256_t a and b, where the content of b is stored in 1
 *      __m256i register, compute a^b and store the result back into a
 *      (in place xor) with non-temporal hint.
 * params:
 *      1) a: ptr to struct uint256_t
 *      2) vb: register that store the content of uint256_t b
 * return: void */
#define uint256_t_xori_from_reg_nt(a, vb) do { \
    uint256_t* const aa = (a); \
    __m256i va = _mm256_stream_load_si256((__m256i*) aa->s); \
    va = _mm256_xor_si256(vb, va); \
    _mm256_stream_si256((__m256i*) aa->s, va); \
} while(0)


#elif defined(__AVX__)

/* usage: Given 2 uint256_t a and b, where the content of a is stored in 1
 *      __m256i register, compute a^b and store the result back into the
 *      register (in place xor).
 * params:
 *      1) b: ptr to struct uint256_t
 *      2) va: register that store the content of uint256_t a
 * return: void */
#define uint256_t_xori_to_reg(b, va) do { \
    const uint256_t* const bb = (b); \
    __m256i vb = _mm256_load_si256((__m256i*) bb->s); \
    va = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(vb), \
                                           _mm256_castsi256_pd(va))); \
} while(0)

/* usage: Given 2 uint256_t a and b, where the content of b is stored in 1
 *      __m256i register, compute a^b and store the result back into a
 *      (in place xor).
 * params:
 *      1) a: ptr to struct uint256_t
 *      2) vb: register that store the content of uint256_t b
 * return: void */
#define uint256_t_xori_from_reg(a, vb) do { \
    uint256_t* const aa = (a); \
    __m256i va = _mm256_load_si256((__m256i*) aa->s); \
    va = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(vb), \
                                           _mm256_castsi256_pd(va))); \
    _mm256_store_si256((__m256i*) aa->s, va); \
} while(0)

/* usage: Given 2 uint256_t a and b, where the content of b is stored in 1
 *      __m256i register, compute a^b and store the result back into a
 *      (in place xor) with non-temporal hint.
 * params:
 *      1) a: ptr to struct uint256_t
 *      2) vb: register that store the content of uint256_t b
 * return: void */
#define uint256_t_xori_from_reg_nt(a, vb) do { \
    uint256_t* const aa = (a); \
    __m256i va = _mm256_load_si256((__m256i*) aa->s); \
    va = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(vb), \
                                           _mm256_castsi256_pd(va))); \
    _mm256_stream_si256((__m256i*) aa->s, va); \
} while(0)

#endif

/* usage: Given 2 uint256_t a and b, compute a^b and store the result into a.
 *      (in place xor)
 * params:
 *      1) a: ptr to struct uint256_t
 *      2) b: ptr to struct uint256_t
 * return: void */
#ifdef __AVX2__

#define uint256_t_xori(a, b) do { \
    uint256_t* const aa = (a); \
    __m256i va = _mm256_load_si256((__m256i*) aa->s); \
    __m256i vb = _mm256_load_si256((__m256i*)(b)->s); \
    _mm256_store_si256((__m256i*) aa->s, _mm256_xor_si256(va, vb)); \
} while(0)

#elif defined(__AVX__)

#define uint256_t_xori(a, b) do { \
    uint256_t* const aa = (a); \
    __m256i va = _mm256_load_si256((__m256i*) aa->s); \
    __m256i vb = _mm256_load_si256((__m256i*)(b)->s); \
    va = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(vb), \
                                           _mm256_castsi256_pd(va))); \
    _mm256_store_si256((__m256i*) aa->s, va); \
} while(0)

#else

#define uint256_t_xori(a, b) do { \
    uint256_t* const aa = (a); \
    const uint256_t* const bb = (b); \
    aa->s[0] ^= bb->s[0]; \
    aa->s[1] ^= bb->s[1]; \
    aa->s[2] ^= bb->s[2]; \
    aa->s[3] ^= bb->s[3]; \
} while(0)

#endif

/* usage: Given 3 uint256_t a, b, and c, compute a ^ (b & c) and store the
 *      result back into a. (in place)
 * params:
 *      1) a: ptr to struct uint256_t
 *      2) b: ptr to struct uint256_t
 *      3) c: ptr to struct uint256_t
 * return: void */
#ifdef __AVX2__

#define uint256_t_xori_and(a, b, c) do { \
    uint256_t* const aa = (a); \
    const uint256_t* const bb = (b); \
    const uint256_t* const cc = (c); \
    __m256i vb = _mm256_load_si256((__m256i*) bb->s); \
    __m256i vc = _mm256_load_si256((__m256i*) cc->s); \
    __m256i va = _mm256_load_si256((__m256i*) aa->s); \
    __m256i vd = _mm256_and_si256(vb, vc); \
    va = _mm256_xor_si256(va, vd); \
    _mm256_store_si256((__m256i*) aa->s, va); \
} while(0)

#elif defined(__AVX__)

#define uint256_t_xori_and(a, b, c) do { \
    uint256_t* const aa = (a); \
    const uint256_t* const bb = (b); \
    const uint256_t* const cc = (c); \
    __m256i vb = _mm256_load_si256((__m256i*) bb->s); \
    __m256i vc = _mm256_load_si256((__m256i*) cc->s); \
    __m256i va = _mm256_load_si256((__m256i*) aa->s); \
    __m256d vd = _mm256_and_pd(_mm256_castsi256_pd(vb), \
                               _mm256_castsi256_pd(vc)); \
    va = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(va), vd)); \
    _mm256_store_si256((__m256i*) aa->s, va); \
} while(0)

#else

#define uint256_t_xori_and(a, b, c) do { \
    uint256_t* const aa = (a); \
    const uint256_t* const bb = (b); \
    const uint256_t* const cc = (c); \
    aa->s[0] ^= bb->s[0] & cc->s[0]; \
    aa->s[1] ^= bb->s[1] & cc->s[1]; \
    aa->s[2] ^= bb->s[2] & cc->s[2]; \
    aa->s[3] ^= bb->s[3] & cc->s[3]; \
} while(0)

#endif

/* usage: Given 3 uint256_t a, b, c, compute (a & c) ^ (b & (~c)) and store the
 *      result into p.
 * params:
 *      1) p: ptr to struct uint256_t
 *      2) a: ptr to struct uint256_t
 *      3) b: ptr to struct uint256_t
 *      4) c: ptr to struct uint256_t
 * return: void */
#ifdef __AVX2__

#define uint256_t_mix(p, a, b, c) do { \
    uint256_t* const pp = (p); \
    const uint256_t* const aa = (a); \
    const uint256_t* const bb = (b); \
    const uint256_t* const cc = (c); \
    __m256i vc = _mm256_load_si256((__m256i*) cc->s); \
    __m256i vb = _mm256_load_si256((__m256i*) bb->s); \
    __m256i va = _mm256_load_si256((__m256i*) aa->s); \
    va = _mm256_and_si256(vc, va); \
    vb = _mm256_andnot_si256(vc, vb); \
    va = _mm256_xor_si256(va, vb); \
    _mm256_store_si256((__m256i*) pp->s, va); \
} while(0)

#elif defined(__AVX__)

#define uint256_t_mix(p, a, b, c) do { \
    uint256_t* const pp = (p); \
    const uint256_t* const aa = (a); \
    const uint256_t* const bb = (b); \
    const uint256_t* const cc = (c); \
    __m256i vc = _mm256_load_si256((__m256i*) cc->s); \
    __m256i vb = _mm256_load_si256((__m256i*) bb->s); \
    __m256i va = _mm256_load_si256((__m256i*) aa->s); \
    va = _mm256_castpd_si256(_mm256_and_pd(_mm256_castsi256_pd(vc), \
                                           _mm256_castsi256_pd(va))); \
    vb = _mm256_castpd_si256(_mm256_andnot_pd(_mm256_castsi256_pd(vc), \
                                              _mm256_castsi256_pd(vb))); \
    va = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(va), \
                                           _mm256_castsi256_pd(vb))); \
    _mm256_store_si256((__m256i*) pp->s, va); \
} while(0)

#else

#define uint256_t_mix(p, a, b, c) do { \
    uint256_t* const pp = (p); \
    const uint256_t* const aa = (a); \
    const uint256_t* const bb = (b); \
    const uint256_t* const cc = (c); \
    pp->s[0] = aa->s[0] & cc->s[0]; \
    pp->s[1] = aa->s[1] & cc->s[1]; \
    pp->s[2] = aa->s[2] & cc->s[2]; \
    pp->s[3] = aa->s[3] & cc->s[3]; \
    pp->s[0] ^= bb->s[0] & ~cc->s[0]; \
    pp->s[1] ^= bb->s[1] & ~cc->s[1]; \
    pp->s[2] ^= bb->s[2] & ~cc->s[2]; \
    pp->s[3] ^= bb->s[3] & ~cc->s[3]; \
} while(0)

#endif

/* usage: Given 3 uint256_t a, b, c, compute (a & c) ^ (b & (~c)) and store the
 *      result back into a. (in place)
 * params:
 *      1) a: ptr to struct uint256_t
 *      2) b: ptr to struct uint256_t
 *      3) c: ptr to struct uint256_t
 * return: void */
#ifdef __AVX2__

#define uint256_t_mixi(a, b, c) do { \
    uint256_t* const aa = (a); \
    const uint256_t* const bb = (b); \
    const uint256_t* const cc = (c); \
    __m256i vc = _mm256_load_si256((__m256i*) cc->s); \
    __m256i vb = _mm256_load_si256((__m256i*) bb->s); \
    __m256i va = _mm256_load_si256((__m256i*) aa->s); \
    va = _mm256_and_si256(vc, va); \
    vb = _mm256_andnot_si256(vc, vb); \
    va = _mm256_xor_si256(va, vb); \
    _mm256_store_si256((__m256i*) aa->s, va); \
} while(0)

#elif defined(__AVX__)

#define uint256_t_mixi(a, b, c) do { \
    uint256_t* const aa = (a); \
    const uint256_t* const bb = (b); \
    const uint256_t* const cc = (c); \
    __m256i vc = _mm256_load_si256((__m256i*) cc->s); \
    __m256i vb = _mm256_load_si256((__m256i*) bb->s); \
    __m256i va = _mm256_load_si256((__m256i*) aa->s); \
    va = _mm256_castpd_si256(_mm256_and_pd(_mm256_castsi256_pd(vc), \
                                           _mm256_castsi256_pd(va))); \
    vb = _mm256_castpd_si256(_mm256_andnot_pd(_mm256_castsi256_pd(vc), \
                                              _mm256_castsi256_pd(vb))); \
    va = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(va), \
                                           _mm256_castsi256_pd(vb))); \
    _mm256_store_si256((__m256i*) aa->s, va); \
} while(0)

#else

#define uint256_t_mixi(a, b, c) do { \
    uint256_t* const aa = (a); \
    const uint256_t* const bb = (b); \
    const uint256_t* const cc = (c); \
    aa->s[0] &= cc->s[0]; \
    aa->s[1] &= cc->s[1]; \
    aa->s[2] &= cc->s[2]; \
    aa->s[3] &= cc->s[3]; \
    aa->s[0] ^= bb->s[0] & ~cc->s[0]; \
    aa->s[1] ^= bb->s[1] & ~cc->s[1]; \
    aa->s[2] ^= bb->s[2] & ~cc->s[2]; \
    aa->s[3] ^= bb->s[3] & ~cc->s[3]; \
} while(0)

#endif

/* usage: Given 4 uint256_t a, b, c, and d, compute a ^ (b & d) ^ (c & (~d)) and
 *      store the result back into a. (in place)
 * params:
 *      1) a: ptr to struct uint256_t
 *      2) b: ptr to struct uint256_t
 *      3) c: ptr to struct uint256_t
 *      4) d: ptr to struct uint256_t
 * return: void */
#ifdef __AVX2__

#define uint256_t_xor_mixi(a, b, c, d) do { \
    uint256_t* const aa = (a); \
    const uint256_t* const bb = (b); \
    const uint256_t* const cc = (c); \
    const uint256_t* const dd = (d); \
    __m256i vd = _mm256_load_si256((__m256i*) dd->s); \
    __m256i vc = _mm256_load_si256((__m256i*) cc->s); \
    __m256i vb = _mm256_load_si256((__m256i*) bb->s); \
    __m256i va = _mm256_load_si256((__m256i*) aa->s); \
    vc = _mm256_andnot_si256(vd, vc); \
    vb = _mm256_and_si256(vd, vb); \
    vb = _mm256_xor_si256(vb, vc); \
    va = _mm256_xor_si256(va, vb); \
    _mm256_store_si256((__m256i*) aa->s, va); \
} while(0)

#elif defined(__AVX__)

#define uint256_t_xor_mixi(a, b, c, d) do { \
    uint256_t* const aa = (a); \
    const uint256_t* const bb = (b); \
    const uint256_t* const cc = (c); \
    const uint256_t* const dd = (d); \
    __m256i vd = _mm256_load_si256((__m256i*) dd->s); \
    __m256i vc = _mm256_load_si256((__m256i*) cc->s); \
    vc = _mm256_castpd_si256(_mm256_andnot_pd(_mm256_castsi256_pd(vd), \
                                              _mm256_castsi256_pd(vc))); \
    __m256i vb = _mm256_load_si256((__m256i*) bb->s); \
    __m256i va = _mm256_load_si256((__m256i*) aa->s); \
    vb = _mm256_castpd_si256(_mm256_and_pd(_mm256_castsi256_pd(vd), \
                                           _mm256_castsi256_pd(vb))); \
    vb = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(vb), \
                                           _mm256_castsi256_pd(vc))); \
    va = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(va), \
                                           _mm256_castsi256_pd(vb))); \
    _mm256_store_si256((__m256i*) aa->s, va); \
} while(0)

#else

#define uint256_t_xor_mixi(a, b, c, d) do { \
    uint256_t* const aa = (a); \
    const uint256_t* const bb = (b); \
    const uint256_t* const cc = (c); \
    const uint256_t* const dd = (d); \
    aa->s[0] ^= bb->s[0] & dd->s[0]; \
    aa->s[1] ^= bb->s[1] & dd->s[1]; \
    aa->s[2] ^= bb->s[2] & dd->s[2]; \
    aa->s[3] ^= bb->s[3] & dd->s[3]; \
    aa->s[0] ^= cc->s[0] & ~dd->s[0]; \
    aa->s[1] ^= cc->s[1] & ~dd->s[1]; \
    aa->s[2] ^= cc->s[2] & ~dd->s[2]; \
    aa->s[3] ^= cc->s[3] & ~dd->s[3]; \
} while(0)

#endif

/* usage: Given 2 uint256_t a and b, compute m&n and store the result into p
 * params:
 *      1) p: ptr to struct uint256_t
 *      2) a: ptr to struct uint256_t
 *      3) b: ptr to struct uint256_t
 * return: void */
#ifdef __AVX2__

#define uint256_t_and(p, a, b) do { \
    uint256_t* const pp = (p); \
    const uint256_t* const aa = (a); \
    const uint256_t* const bb = (b); \
    __m256i vb = _mm256_load_si256((__m256i*) bb->s); \
    __m256i va = _mm256_load_si256((__m256i*) aa->s); \
    va = _mm256_and_si256(va, vb); \
    _mm256_store_si256((__m256i*) pp->s, va); \
} while(0)

#elif defined (__AVX__)

#define uint256_t_and(p, a, b) do { \
    uint256_t* const pp = (p); \
    const uint256_t* const aa = (a); \
    const uint256_t* const bb = (b); \
    __m256i vb = _mm256_load_si256((__m256i*) bb->s); \
    __m256i va = _mm256_load_si256((__m256i*) aa->s); \
    va = _mm256_castpd_si256(_mm256_and_pd(_mm256_castsi256_pd(va), \
                                           _mm256_castsi256_pd(vb))); \
    _mm256_store_si256((__m256i*) pp->s, va); \
} while(0)

#else

#define uint256_t_and(p, a, b) do { \
    uint256_t* const pp = (p); \
    const uint256_t* const aa = (a); \
    const uint256_t* const bb = (b); \
    pp->s[0] = aa->s[0] & bb->s[0]; \
    pp->s[1] = aa->s[1] & bb->s[1]; \
    pp->s[2] = aa->s[2] & bb->s[2]; \
    pp->s[3] = aa->s[3] & bb->s[3]; \
} while(0)

#endif

/* usage: Given 2 uint256_t a and b, compute a&b and store the result into a.
 *      (in place and)
 * params:
 *      1) a: ptr to struct uint256_t
 *      2) b: ptr to struct uint256_t
 * return: void */
#ifdef __AVX2__

#define uint256_t_andi(a, b) do { \
    uint256_t* const aa = (a); \
    const uint256_t* const bb = (b); \
    __m256i vb = _mm256_load_si256((__m256i*) bb->s); \
    __m256i va = _mm256_load_si256((__m256i*) aa->s); \
    va = _mm256_and_si256(va, vb); \
    _mm256_store_si256((__m256i*) aa->s, va); \
} while(0)

#elif defined (__AVX__)

#define uint256_t_andi(a, b) do { \
    uint256_t* const aa = (a); \
    const uint256_t* const bb = (b); \
    __m256i vb = _mm256_load_si256((__m256i*) bb->s); \
    __m256i va = _mm256_load_si256((__m256i*) aa->s); \
    va = _mm256_castpd_si256(_mm256_and_pd(_mm256_castsi256_pd(va), \
                                           _mm256_castsi256_pd(vb))); \
    _mm256_store_si256((__m256i*) aa->s, va); \
} while(0)

#else

#define uint256_t_andi(a, b) do { \
    uint256_t* const aa = (a); \
    const uint256_t* const bb = (b); \
    aa->s[0] &= bb->s[0]; \
    aa->s[1] &= bb->s[1]; \
    aa->s[2] &= bb->s[2]; \
    aa->s[3] &= bb->s[3]; \
} while(0)

#endif

/* usage: Given 2 uint256_t a and b, compute a & ~b and store the result into p
 * params:
 *      1) p: ptr to struct uint256_t
 *      2) a: ptr to struct uint256_t
 *      3) b: ptr to struct uint256_t
 * return: void */
#ifdef __AVX2__

#define uint256_t_andn(p, a, b) do { \
    uint256_t* const pp = (p); \
    const uint256_t* const aa = (a); \
    const uint256_t* const bb = (b); \
    __m256i vb = _mm256_load_si256((__m256i*) bb->s); \
    __m256i va = _mm256_load_si256((__m256i*) aa->s); \
    va = _mm256_andnot_si256(vb, va); \
    _mm256_store_si256((__m256i*) pp->s, va); \
} while(0)

#elif defined(__AVX__)

#define uint256_t_andn(p, a, b) do { \
    uint256_t* const pp = (p); \
    const uint256_t* const aa = (a); \
    const uint256_t* const bb = (b); \
    __m256i vb = _mm256_load_si256((__m256i*) bb->s); \
    __m256i va = _mm256_load_si256((__m256i*) aa->s); \
    va = _mm256_castpd_si256(_mm256_andnot_pd(_mm256_castsi256_pd(vb), \
                                              _mm256_castsi256_pd(va))); \
    _mm256_store_si256((__m256i*) pp->s, va); \
} while(0)

#else

#define uint256_t_andn(p, a, b) do { \
    uint256_t* const pp = (p); \
    const uint256_t* const aa = (a); \
    const uint256_t* const bb = (b); \
    pp->s[0] = aa->s[0] & ~bb->s[0]; \
    pp->s[1] = aa->s[1] & ~bb->s[1]; \
    pp->s[2] = aa->s[2] & ~bb->s[2]; \
    pp->s[3] = aa->s[3] & ~bb->s[3]; \
} while(0)

#endif

/* usage: Given 2 uint256_t a and b, compute a & ~b and store the result into a.
 *      (in place negate-and)
 * params:
 *      1) a: ptr to struct uint256_t
 *      2) b: ptr to struct uint256_t
 * return: void */
#ifdef __AVX2__

#define uint256_t_andni(a, b) do { \
    const uint256_t* const aa = (a); \
    const uint256_t* const bb = (b); \
    __m256i vb = _mm256_load_si256((__m256i*) bb->s); \
    __m256i va = _mm256_load_si256((__m256i*) aa->s); \
    va = _mm256_andnot_si256(vb, va); \
    _mm256_store_si256((__m256i*) aa->s, va); \
} while(0)

#elif defined(__AVX__)

#define uint256_t_andni(a, b) do { \
    uint256_t* const aa = (a); \
    const uint256_t* const bb = (b); \
    __m256i vb = _mm256_load_si256((__m256i*) bb->s); \
    __m256i va = _mm256_load_si256((__m256i*) aa->s); \
    va = _mm256_castpd_si256(_mm256_andnot_pd(_mm256_castsi256_pd(vb), \
                                              _mm256_castsi256_pd(va))); \
    _mm256_store_si256((__m256i*) aa->s, va); \
} while(0)

#else

#define uint256_t_andni(a, b) do { \
    uint256_t* const aa = (a); \
    const uint256_t* const bb = (b); \
    aa->s[0] &= ~bb->s[0]; \
    aa->s[1] &= ~bb->s[1]; \
    aa->s[2] &= ~bb->s[2]; \
    aa->s[3] &= ~bb->s[3]; \
} while(0)

#endif

/* usage: Given 2 uint256_t a and b, compute a|b and store the result into c.
 * params:
 *      1) c: ptr to struct uint256_t
 *      1) a: ptr to struct uint256_t
 *      2) b: ptr to struct uint256_t
 * return: void */
#if defined(__AVX2__)

#define uint256_t_or(c, a, b) do { \
    uint256_t* const cc = (c); \
    const uint256_t* const aa = (a); \
    const uint256_t* const bb = (b); \
    __m256i va = _mm256_load_si256((__m256i*) aa->s); \
    __m256i vb = _mm256_load_si256((__m256i*) bb->s); \
    va = _mm256_or_si256(va, vb); \
    _mm256_store_si256((__m256i*) cc->s, va); \
} while(0)

#elif defined(__AVX__)

#define uint256_t_or(c, a, b) do { \
    uint256_t* const cc = (c); \
    const uint256_t* const aa = (a); \
    const uint256_t* const bb = (b); \
    __m256i va = _mm256_load_si256((__m256i*) aa->s); \
    __m256i vb = _mm256_load_si256((__m256i*) bb->s); \
    va = _mm256_castpd_si256(_mm256_or_pd(_mm256_castsi256_pd(va), \
                                          _mm256_castsi256_pd(vb))); \
    _mm256_store_si256((__m256i*) cc->s, va); \
} while(0)

#else

#define uint256_t_or(c, a, b) do { \
    uint256_t* const cc = (c); \
    const uint256_t* const aa = (a); \
    const uint256_t* const bb = (b); \
    cc->s[ 0] = aa->s[ 0] | bb->s[ 0]; \
    cc->s[ 1] = aa->s[ 1] | bb->s[ 1]; \
    cc->s[ 2] = aa->s[ 2] | bb->s[ 2]; \
    cc->s[ 3] = aa->s[ 3] | bb->s[ 3]; \
} while(0)

#endif

/* usage: Given 2 uint256_t a and b, compute a|b and store the result into a.
 *      (in place or)
 * params:
 *      1) a: ptr to struct uint256_t
 *      2) b: ptr to struct uint256_t
 * return: void */
#ifdef __AVX2__

#define uint256_t_ori(a, b) do { \
    uint256_t* const aa = (a); \
    const uint256_t* const bb = (b); \
    __m256i vb = _mm256_load_si256((__m256i*) bb->s); \
    __m256i va = _mm256_load_si256((__m256i*) aa->s); \
    va = _mm256_or_si256(va, vb); \
    _mm256_store_si256((__m256i*) aa->s, va); \
} while(0)

#elif defined(__AVX__)

#define uint256_t_ori(a, b) do { \
    uint256_t* const aa = (a); \
    const uint256_t* const bb = (b); \
    __m256i vb = _mm256_load_si256((__m256i*) bb->s); \
    __m256i va = _mm256_load_si256((__m256i*) aa->s); \
    va = _mm256_castpd_si256(_mm256_or_pd(_mm256_castsi256_pd(vb), \
                                          _mm256_castsi256_pd(va))); \
    _mm256_store_si256((__m256i*) aa->s, va); \
} while(0)

#else

#define uint256_t_ori(a, b) do { \
    uint256_t* const aa = (a); \
    const uint256_t* const bb = (b); \
    aa->s[0] |= bb->s[0]; \
    aa->s[1] |= bb->s[1]; \
    aa->s[2] |= bb->s[2]; \
    aa->s[3] |= bb->s[3]; \
} while(0)

#endif

/* usage: Given 1 uint256_t a, set a to all 1's.
 * params:
 *      1) a: ptr to struct uint256_t
 * return: void */
#define uint256_t_set_max(a) do { \
    memset((a), UINT8_MAX, sizeof(uint256_t)); \
} while(0)

/* usage: Given 1 uint256_t a, and an integer i, return the i-th bit.
 *      The LSB is 0-th bit.
 * params:
 *      1) a: ptr to struct uint256_t
 *      2) i: index to the bit, from 0 ~ 255. Do not perform arithmetic here.
 * return: the bit */
static inline uint64_t
uint256_t_at(const uint256_t* a, uint32_t i) {
    return (a->s[i >> 6] >> (i & 0x3FULL)) & 0x1ULL;
}

/* usage: Given 1 uint256_t a, an integer i, set the i-th bit to the given
 *      value. The LSB is 0-th bit.
 * params:
 *      1) a: ptr to struct uint256_t
 *      2) i: index to the bit, from 0 ~ 255
 *      3) v: true if the bit is 1, otherwise false
 * return: void */
#define uint256_t_set_at(a, i, v) do { \
    uint256_t* const aa = (a); \
    uint64_t const ii = (i); \
    aa->s[ii >> 6] &= ~(0x1ULL << (ii & 0x3FULL)); \
    aa->s[ii >> 6] |= ((uint64_t) (v)) << (ii & 0x3FULL); \
} while(0)

/* usage: Given 1 uint256_t a, an integer i, toggle the i-th bit.
 *      The LSB is 0-th bit.
 * params:
 *      1) a: ptr to struct uint256_t
 *      2) i: index to the bit, from 0 ~ 255
 * return: void */
#define uint256_t_toggle_at(a, i) do { \
    uint256_t* const aa = (a); \
    uint64_t const ii = (i); \
    aa->s[ii >> 6] ^= 0x1ULL << (ii & 0x3FULL); \
} while(0)

/* usage: Given 1 uint256_t a, an integer i, set the i-th bit to 0.
 *      The LSB is 0-th bit.
 * params:
 *      1) a: ptr to struct uint256_t
 *      2) i: index to the bit, from 0 ~ 255
 * return: void */
#define uint256_t_clear_at(a, i) do { \
    uint64_t const ii = (i); \
    (a)->s[ii >> 6] &= ~(0x1ULL << (ii & 0x3FULL)); \
} while(0)

/* usage: Given 1 uint256_t a, an integer i, shift packed 64-bit integers in
 *      a right by i while shitfting in zeros, and store the result into dst
 * params:
 *      1) dst: ptr to struct uint256_t
 *      2) a: ptr to struct uint256_t
 *      3) i: number of bits to shift right, 0 ~ 255
 * return: void */
#if defined(__AVX2__)

#define uint256_t_srl_64b(dst, a, i) do { \
    const uint64_t ii = (i); \
    assert(ii < 256); \
    __m256i va = _mm256_load_si256((__m256i*) a); \
    __m256i vd = _mm256_srli_epi64(va, ii); \
    _mm256_store_si256((__m256i*) dst, vd); \
} while(0)

#else

#define uint256_t_srl_64b(dst, a, i) do { \
    const uint64_t ii = (i); \
    assert(ii < 256); \
    const uint256_t* aa = (a); \
    uint256_t* dd = (dst); \
    dd->s[0] = aa->s[0] >> ii; \
    dd->s[1] = aa->s[1] >> ii; \
    dd->s[2] = aa->s[2] >> ii; \
    dd->s[3] = aa->s[3] >> ii; \
} while(0)

#endif

/* usage: Given 1 uint256_t a, an integer i, shift packed 64-bit integers in
 *      a right by i while shitfting in zeros, and store the result back to a
 * params:
 *      1) a: ptr to struct uint256_t
 *      2) i: number of bits to shift right, 0 ~ 255
 * return: void */
#if defined(__AVX2__)

#define uint256_t_srli_64b(a, i) do { \
    const uint64_t ii = (i); \
    assert(ii < 256); \
    uint256_t* aa = (a); \
    __m256i va = _mm256_load_si256((__m256i*) aa); \
    va = _mm256_srli_epi64(va, ii); \
    _mm256_store_si256((__m256i*) aa, va); \
} while(0)

#else

#define uint256_t_srli_64b(a, i) do { \
    const uint64_t ii = (i); \
    assert(ii < 256); \
    uint256_t* aa = (a); \
    aa->s[0] >>= ii; \
    aa->s[1] >>= ii; \
    aa->s[2] >>= ii; \
    aa->s[3] >>= ii; \
} while(0)

#endif

/* usage: Given 1 uint256_t a, find location of the first set bit in a
 * params:
 *      1) a: ptr to struct uint256_t
 * return: i if i-th bit is the first set bit. If no bits are set, return 0. */
int
uint256_t_ffs(const uint256_t* const a);

/* usage: Given 1 uint256_t a, find the location of first set bit after the
 *      i-th bit (included)
 * params:
 *      1) a: ptr to struct uint256_t
 *      2) i: start searching from i-th bit; from 0 ~ 255
 * return: i if i-th bit is the first set bit. If no bits are set, return 0. */
int
uint256_t_ffs_after(const uint256_t* const a, uint64_t i);

/* usage: Given 1 uint256_t a, find indices of all set bits in a
 * params:
 *      1) a: ptr to struct uint256_t
 *      2) res: an uint8_t array for storing the indices, must hold at least
 *              256 elements
 * return: the number of set bits */
static inline int
uint256_t_sbpos(const uint256_t* const restrict a,
                uint8_t* const restrict res) {
    uint64_t base = 0x0ULL;
    const uint64_t inc64 = 0x4040404040404040ULL;
    int sbnum = sbidx_in_64b_sz8(res, base, uint256_t_64b_at(a, 0));
    base += inc64;
    sbnum += sbidx_in_64b_sz8(res + sbnum, base, uint256_t_64b_at(a, 1));
    base += inc64;
    sbnum += sbidx_in_64b_sz8(res + sbnum, base, uint256_t_64b_at(a, 2));
    base += inc64;
    sbnum += sbidx_in_64b_sz8(res + sbnum, base, uint256_t_64b_at(a, 3));
    assert(sbnum <= 256);
    return sbnum;
}

#if defined(__AVX__)

/* usage: Given 1 uint256_t a, find indices of all set bits in a
 * params:
 *      1) res: an uint8_t array for storing the indices, must hold at least
 *              256 elements
 *      2) a: ptr to struct uint256_t
 * return: the number of set bits */
static inline int
uint256_t_sbpos_from_reg(uint8_t* const res, __m256i va) {
    uint64_t base = 0x0ULL;
    const uint64_t inc64 = 0x4040404040404040ULL;
    int sbnum = sbidx_in_64b_sz8(res, base, _mm256_extract_epi64(va, 0));
    base += inc64;
    sbnum += sbidx_in_64b_sz8(res + sbnum, base, _mm256_extract_epi64(va, 1));
    base += inc64;
    sbnum += sbidx_in_64b_sz8(res + sbnum, base, _mm256_extract_epi64(va, 2));
    base += inc64;
    sbnum += sbidx_in_64b_sz8(res + sbnum, base, _mm256_extract_epi64(va, 3));
    assert(sbnum <= 256);
    return sbnum;
}

#endif

#endif /* __BLK_LANCZOS_UINT256_T_H__ */
