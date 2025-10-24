#ifndef __UINT128_T_H__
#define __UINT128_T_H__

#include <stdint.h>
#include <stdbool.h>
#include <string.h>

#if defined(__AVX__) || defined(__AVX2__)
#include <immintrin.h>
#endif

#include "util.h"
#include "bitmap_table.h"

/* ========================================================================
 * struct uint128_t definition
 * ======================================================================== */

typedef struct {
    uint64_t s[2];
} uint128_t;

/* ========================================================================
 * functions prototypes
 * ======================================================================== */

/* usage: Given 1 uint128_t a, treat it as uint64_t[2] and
 *      return the specified 64 bits.
 * params:
 *      1) a: ptr to struct uint128_t
 *      2) i: index of the slot; from 0 ~ 1
 * return: the specified 64-bit as a uint64_t */
static force_inline uint64_t
uint128_t_64b_at(const uint128_t* a, uint32_t i) {
    assert(i < 2);
    return a->s[i];
}

/* usage: Given 1 uint128_t a, treat it as uint64_t[2] and set
 *      the all 2 64-bit integers to the given value.
 * params:
 *      1) a: ptr to struct uint128_t
 *      3) v: the new value
 * return: void */
static force_inline void
uint128_t_set1_64b(uint128_t* a, uint64_t v) {
    a->s[0] = v;
    a->s[1] = v;
}

/* usage: Given 1 uint128_t a, set a to all 0's.
 * params:
 *      1) a: ptr to struct uint128_t
 * return: void */
static force_inline void
uint128_t_zero(uint128_t* a) {
    memset(a, 0x0, sizeof(uint128_t));
}

/* usage: Given 1 uint128_t, set a to all 1's.
 * params:
 *      1) a: ptr to struct uint128_t
 * return: void */
static force_inline void
uint128_t_max(uint128_t* a) {
    a->s[0] = UINT64_MAX;
    a->s[1] = UINT64_MAX;
}

/* usage: Given 1 uint128_t a, return true if all 1's. Otherwise false;
 * params:
 *      1) a: ptr to struct uint128_t
 * return: void */
static force_inline bool
uint128_t_is_max(const uint128_t* const a) {
    return !(~(a->s[0] & a->s[1]));
}

/* usage: Given 1 uint128_t a, return true if all 0's. Otherwise false;
 * params:
 *      1) a: ptr to struct uint128_t
 * return: void */
static inline bool
uint128_t_is_zero(const uint128_t* const a) {
    return !(a->s[0] | a->s[1]);
}

/* usage: Given 1 uint128_t a, return false if all 0's. Otherwise true;
 * params:
 *      1) a: ptr to struct uint128_t
 * return: void */
static inline bool
uint128_t_is_not_zero(const uint128_t* const a) {
    return !uint128_t_is_zero(a);
}

/* usage: Given 2 uint128_t a and b, check if they are the same.
 * params:
 *      1) a: ptr to struct uint128_t
 *      2) b: ptr to struct uint128_t
 * return: true if they are the same. otherwise false */
static inline bool
uint128_t_equal(const uint128_t* const a, const uint128_t* const b) {
    return (0 == memcmp(a, b, sizeof(uint128_t)));
}

/* usage: Given 1 uint128_t a, return the number of bits that are set to 1 in a
 * params:
 *      1) a: ptr to struct uint128_t
 * return: the number of 1's in the uint128_t */
static inline uint64_t
uint128_t_popcount(const uint128_t* const a) {
    return (__builtin_popcountll(a->s[0]) + __builtin_popcountll(a->s[1]));
}

/* usage: Given 2 uint128_t a and b, copy b into a.
 * params:
 *      1) a: ptr to struct uint128_t
 *      2) b: ptr to struct uint128_t
 * return: void */
static force_inline void
uint128_t_copy(uint128_t* restrict a, const uint128_t* restrict b) {
    memcpy(a, b, sizeof(uint128_t));
}

/* usage: Given 1 uint128_t a, set a to a random value.
 * params:
 *      1) a: ptr to struct uint128_t
 * return: void */
static force_inline void
uint128_t_rand(uint128_t* a) {
    int* buf = (int*) (a->s);
    buf[0] = rand();
    buf[1] = rand();
    buf[2] = rand();
    buf[3] = rand();
}

/* usage: Given 2 uint128_t a and b, swap their values
 * params:
 *      1) a: ptr to struct uint128_t
 *      2) b: ptr to struct uint128_t
 * return: void */
static force_inline void
uint128_t_swap(uint128_t* restrict a, uint128_t* restrict b) {
    uint64_t t0 = a->s[0];
    uint64_t t1 = a->s[1];
    a->s[0] = b->s[0];
    a->s[1] = b->s[1];
    b->s[0] = t0;
    b->s[1] = t1;
}

/* usage: Given 1 uint128_t a, and an integer i, return the i-th bit.
 *      The LSB is 0-th bit.
 * params:
 *      1) a: ptr to struct uint128_t
 *      2) i: index to the bit, from 0 ~ 127
 * return: the bit */
static force_inline uint64_t
uint128_t_at(const uint128_t* a, uint32_t i) {
   return (a->s[i >> 6] >> (i & 0x3FU)) & 0x1U;
}

/* usage: Given 1 uint128_t a, an integer i, set the i-th bit to the given
 *      value. The LSB is 0-th bit.
 * params:
 *      1) a: ptr to struct uint128_t
 *      2) i: index to the bit, from 0 ~ 127
 *      3) v: true if the bit is 1, otherwise false
 * return: void */
static force_inline void
uint128_t_set_at(uint128_t* a, uint32_t i, uint32_t v) {
    assert(v == 0 || v == 1);
    a->s[i >> 6] &= ~(0x1ULL << (i & 0x3FULL));
    a->s[i >> 6] |= ((uint64_t) (v)) << (i & 0x3FULL);
}

/* usage: Given 1 uint128_t a, an integer i, toggle the i-th bit.
 *      The LSB is 0-th bit.
 * params:
 *      1) a: ptr to struct uint128_t
 *      2) i: index to the bit, from 0 ~ 127
 * return: void */
static force_inline void
uint128_t_toggle_at(uint128_t* a, uint32_t i) {
    a->s[i >> 6] ^= 0x1ULL << (i & 0x3FULL);
}

/* usage: Given 1 uint128_t a, an integer i, set the i-th bit to 0.
 *      The LSB is 0-th bit.
 * params:
 *      1) a: ptr to struct uint128_t
 *      2) i: index to the bit, from 0 ~ 127
 * return: void */
static force_inline void
uint128_t_clear_at(uint128_t* a, uint32_t i) {
    a->s[i >> 6] &= ~(0x1ULL << (i & 0x3FULL));
}

/* usage: Given 1 uint128_t a, find indices of all set bits in a
 * params:
 *      1) a: ptr to struct uint128_t
 *      2) res: an uint8_t array for storing the indices, must hold at least
 *              128 elements
 * return: the number of set bits */
static inline int
uint128_t_sbpos(const uint128_t* const restrict a,
                uint8_t* const restrict res) {
    uint64_t base = 0x0ULL;
    const uint64_t inc64 = 0x4040404040404040ULL;
    int sbnum = sbidx_in_64b_sz8(res, base, uint128_t_64b_at(a, 0));
    base += inc64;
    sbnum += sbidx_in_64b_sz8(res + sbnum, base, uint128_t_64b_at(a, 1));
    assert(sbnum <= 128);
    return sbnum;
}

/* usage: Given 2 uint128_t a and b, compute a|b and store the result into c.
 * params:
 *      1) c: ptr to struct uint128_t
 *      1) a: ptr to struct uint128_t
 *      2) b: ptr to struct uint128_t
 * return: void */
static force_inline void
uint128_t_or(uint128_t* restrict c, const uint128_t* restrict a,
             const uint128_t* restrict b) {
    c->s[0] = a->s[0] | b->s[0];
    c->s[1] = a->s[1] | b->s[1];
}

/* usage: Given 1 uint128_t a, bitwise negate it and store the result into
 *      another container.
 * params:
 *      1) out: ptr to struct uint128_t. Container for the result
 *      2) a: ptr to struct uint128_t
 * return: void */
static force_inline void
uint128_t_neg(uint128_t* out, const uint128_t* a) {
    out->s[0] = ~(a->s[0]);
    out->s[1] = ~(a->s[1]);
}

/* usage: Given 1 uint128_t a, bitwise negate it.
 * params:
 *      1) a: ptr to struct uint128_t
 * return: void */
static force_inline void
uint128_t_negi(uint128_t* a) {
    a->s[0] = ~(a->s[0]);
    a->s[1] = ~(a->s[1]);
}

/* usage: Given 2 uint128_t a and b, compute a&b and store the result into a.
 *      (in place and)
 * params:
 *      1) a: ptr to struct uint128_t
 *      2) b: ptr to struct uint128_t
 * return: void */
static force_inline void
uint128_t_andi(uint128_t* a, const uint128_t* b) {
    a->s[0] &= b->s[0];
    a->s[1] &= b->s[1];
}

/* usage: Given 3 uint128_t a, b, c, compute (a & c) ^ (b & (~c)) and store the
 *      result back into a. (in place)
 * params:
 *      1) a: ptr to struct uint128_t
 *      2) b: ptr to struct uint128_t
 *      3) c: ptr to struct uint128_t
 * return: void */
static force_inline void
uint128_t_mixi(uint128_t* restrict a, const uint128_t* restrict b,
               const uint128_t* restrict c) {
    a->s[0] &= c->s[0];
    a->s[1] &= c->s[1];
    a->s[0] ^= b->s[0] & ~c->s[0];
    a->s[1] ^= b->s[1] & ~c->s[1];
}

/* usage: Given 2 uint128_t a and b, compute a^b and store the result into a.
 *      (in place xor)
 * params:
 *      1) a: ptr to struct uint128_t
 *      2) b: ptr to struct uint128_t
 * return: void */
static force_inline void
uint128_t_xori(uint128_t* a, const uint128_t* b) {
    a->s[0] ^= b->s[0];
    a->s[1] ^= b->s[1];
}

/* usage: Given 2 uint128_t a and b, compute a & b and store the result into p
 * params:
 *      1) p: ptr to struct uint128_t
 *      2) a: ptr to struct uint128_t
 *      3) b: ptr to struct uint128_t
 * return: void */
static force_inline void
uint128_t_and(uint128_t* restrict p, const uint128_t* restrict a,
              const uint128_t* restrict b) {
    p->s[0] = a->s[0] & b->s[0];
    p->s[1] = a->s[1] & b->s[1];
}

/* usage: Given 2 uint128_t a and b, compute a & ~b and store the result into p
 * params:
 *      1) p: ptr to struct uint128_t
 *      2) a: ptr to struct uint128_t
 *      3) b: ptr to struct uint128_t
 * return: void */
static force_inline void
uint128_t_andn(uint128_t* restrict p, const uint128_t* restrict a,
               const uint128_t* restrict b) {
    p->s[0] = a->s[0] & ~b->s[0];
    p->s[1] = a->s[1] & ~b->s[1];
}

/* usage: Given 3 uint128_t a, b, and c, compute a ^ (b & c) and store the
 *      result back into a. (in place)
 * params:
 *      1) a: ptr to struct uint128_t
 *      2) b: ptr to struct uint128_t
 *      3) c: ptr to struct uint128_t
 * return: void */
static force_inline void
uint128_t_xori_and(uint128_t* restrict a, const uint128_t* restrict b,
                   const uint128_t* restrict c) {
    a->s[0] ^= b->s[0] & c->s[0];
    a->s[1] ^= b->s[1] & c->s[1];
}

#endif // __UINT128_T_H__
