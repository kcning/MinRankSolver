/* uint64a_avx.c: implementation of bit operations on an array of uint64_t
 * optimized with AVX */

#include <stdint.h>
#include <assert.h>

#include "uint64a.h"
#include "util.h"

#if defined(__AVX__)

#include <immintrin.h>

// NOTE: It's inefficient to extract elements in an __m256i one by one, and
// keep the __m256i around. The test results show that one should extract all 4
// elements together, and that the best time to extract is when there's only 1
// individual column left.
//
// Storing a __m256i onto the stack to extract its 4 elements is slightly
// slower than calling _mm256_extract_epi64() 4 times.
//
// One should keep a small amount of non-AVX instructions around for each
// iteration (in this case checking individual columns) to fill in the gap
// when CPU waits for execution results of AVX instructions.
// 
// Checking individual columns compiles to CMOV, which doesn't disrupt CPU
// pipeline.
//
// It's cheaper to check each lsb immediately after it's computed, because
// it compiles to 1 single JE instruction. Most of the time it doesn't
// jump either because singular systems are rare.

static inline __m256i __attribute__((always_inline))
mm256i_cmpeq(__m256i a, __m256i b) {
    // NOTE: no uint64_t in both a and b is NaN when interpreted as double,
    // since it has either only 1 set bit, or is completely zero.
    __m256d r = _mm256_cmp_pd(_mm256_castsi256_pd(a),
                              _mm256_castsi256_pd(b),_CMP_EQ_OQ);
    return _mm256_castpd_si256(r);
}

static inline __m256i __attribute__((always_inline))
mm256i_and(__m256i a, __m256i b) {
    return _mm256_castpd_si256(_mm256_and_pd(_mm256_castsi256_pd(a),
                                             _mm256_castsi256_pd(b)));
}

static inline __m256i __attribute__((always_inline))
mm256i_xor(__m256i a, __m256i b) {
    return _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(a),
                                             _mm256_castsi256_pd(b)));
}

static inline __m256i __attribute__((always_inline))
uint64a_gj_reduc_mm256(__m256i row, __m256i mask, __m256i reduc) {
    __m256i tmp = mm256i_and(row, mask);
    tmp = mm256i_cmpeq(tmp, mask);
    tmp = mm256i_and(tmp, reduc);
    return mm256i_xor(row, tmp);
}

/* usage: Given an array of uint64_t m, treat the 1st column as the constants,
 *      2nd column as the values for x1, 3rd column for x2, and so on.
 *      Perform Gauss-Jordan elimination on it, and extract a solution
 *      if the system is solvable without modifying the array.
 * params:
 *      1) m: ptr to an array of uint64_t
 *      2) sol: ptr to a uint64_t for storing the solution if the system is
 *              solvable. The LSB stores the value for x1, 2nd LSB for x2, and
 *              so on.
 * return: 0 if the system has a unique solution, 1 if not solvable,
 *      and -1 if underdetermined. */
int64_t
uint64a_gj_v5_avx(const uint64_t m[6], uint64_t* const restrict sol) {
    __m256i c0_3 = _mm256_loadu_si256((__m256i*) (m + 0));
    uint64_t c4 = m[4];
    uint64_t c5 = m[5];

    uint64_t lsb5 = uint64_t_lsb(c5);
    if(unlikely(!lsb5)) {
        return -1; // singular
    }

    uint64_t c5_reduc = c5 ^ lsb5;
    __m256i reduc = _mm256_set1_epi64x(c5_reduc);
    __m256i vmask = _mm256_set1_epi64x(lsb5);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb5) c4 ^= c5_reduc;
    uint64_t mask = ~lsb5;

    uint64_t lsb4 = uint64_t_lsb(c4 & mask);
    if(unlikely(!lsb4)) {
        return -1; // singular
    }

    uint64_t c4_reduc = c4 ^ lsb4;
    reduc = _mm256_set1_epi64x(c4_reduc);
    vmask = _mm256_set1_epi64x(lsb4);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    uint64_t c0 = _mm256_extract_epi64(c0_3, 0);
    uint64_t c1 = _mm256_extract_epi64(c0_3, 1);
    uint64_t c2 = _mm256_extract_epi64(c0_3, 2);
    uint64_t c3 = _mm256_extract_epi64(c0_3, 3);
    mask ^= lsb4;

    uint64_t lsb3 = uint64_t_lsb(c3 & mask);
    if(unlikely(!lsb3)) {
        return -1; // singular
    }

    uint64_t c3_reduc = c3 ^ lsb3;
    if(c0 & lsb3) c0 ^= c3_reduc;
    if(c1 & lsb3) c1 ^= c3_reduc;
    if(c2 & lsb3) c2 ^= c3_reduc;
    mask ^= lsb3;

    uint64_t lsb2 = uint64_t_lsb(c2 & mask);
    if(unlikely(!lsb2)) {
        return -1; // singular
    }

    uint64_t c2_reduc = c2 ^ lsb2;
    if(c0 & lsb2) c0 ^= c2_reduc;
    if(c1 & lsb2) c1 ^= c2_reduc;
    mask ^= lsb2;

    uint64_t lsb1 = uint64_t_lsb(c1 & mask);
    if(unlikely(!lsb1)) {
        return -1; // singular
    }

    if(c0 & lsb1) c0 ^= c1 ^ lsb1;
    mask ^= lsb1;

    if(likely(mask & c0)) { // check system consistency
        return mask & c0; // not solvable
    }

    uint64_t s = 0x0ULL;
    if(c0 & lsb1) s = uint64_t_toggle_at(s, 0);
    if(c0 & lsb2) s = uint64_t_toggle_at(s, 1);
    if(c0 & lsb3) s = uint64_t_toggle_at(s, 2);
    if(c0 & lsb4) s = uint64_t_toggle_at(s, 3);
    if(c0 & lsb5) s = uint64_t_toggle_at(s, 4);
    *sol = s;
    return 0;
}

int64_t
uint64a_gj_v6_avx(const uint64_t m[7], uint64_t* const restrict sol) {
    __m256i c0_3 = _mm256_loadu_si256((__m256i*) (m + 0));
    uint64_t c4 = m[4];
    uint64_t c5 = m[5];
    uint64_t c6 = m[6];

    uint64_t lsb6 = uint64_t_lsb(c6);
    if(unlikely(!lsb6)) {
        return -1; // singular
    }

    uint64_t c6_reduc = c6 ^ lsb6;
    __m256i reduc = _mm256_set1_epi64x(c6_reduc);
    __m256i vmask = _mm256_set1_epi64x(lsb6);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb6) c4 ^= c6_reduc;
    if(c5 & lsb6) c5 ^= c6_reduc;
    uint64_t mask = ~lsb6;

    uint64_t lsb5 = uint64_t_lsb(c5 & mask);
    if(unlikely(!lsb5)) {
        return -1; // singular
    }

    uint64_t c5_reduc = c5 ^ lsb5;
    reduc = _mm256_set1_epi64x(c5_reduc);
    vmask = _mm256_set1_epi64x(lsb5);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb5) c4 ^= c5_reduc;
    uint64_t c0 = _mm256_extract_epi64(c0_3, 0);
    uint64_t c1 = _mm256_extract_epi64(c0_3, 1);
    uint64_t c2 = _mm256_extract_epi64(c0_3, 2);
    uint64_t c3 = _mm256_extract_epi64(c0_3, 3);
    mask ^= lsb5;

    uint64_t lsb4 = uint64_t_lsb(c4 & mask);
    if(unlikely(!lsb4)) {
        return -1; // singular
    }

    uint64_t c4_reduc = c4 ^ lsb4;
    if(c0 & lsb4) c0 ^= c4_reduc;
    if(c1 & lsb4) c1 ^= c4_reduc;
    if(c2 & lsb4) c2 ^= c4_reduc;
    if(c3 & lsb4) c3 ^= c4_reduc;
    mask ^= lsb4;

    uint64_t lsb3 = uint64_t_lsb(c3 & mask);
    if(unlikely(!lsb3)) {
        return -1; // singular
    }

    uint64_t c3_reduc = c3 ^ lsb3;
    if(c0 & lsb3) c0 ^= c3_reduc;
    if(c1 & lsb3) c1 ^= c3_reduc;
    if(c2 & lsb3) c2 ^= c3_reduc;
    mask ^= lsb3;

    uint64_t lsb2 = uint64_t_lsb(c2 & mask);
    if(unlikely(!lsb2)) {
        return -1; // singular
    }

    uint64_t c2_reduc = c2 ^ lsb2;
    if(c0 & lsb2) c0 ^= c2_reduc;
    if(c1 & lsb2) c1 ^= c2_reduc;
    mask ^= lsb2;

    uint64_t lsb1 = uint64_t_lsb(c1 & mask);
    if(unlikely(!lsb1)) {
        return -1; // singular
    }

    if(c0 & lsb1) c0 ^= c1 ^ lsb1;
    mask ^= lsb1;

    if(likely(mask & c0)) { // check system consistency
        return mask & c0; // not solvable
    }

    uint64_t s = 0x0ULL;
    if(c0 & lsb1) s = uint64_t_toggle_at(s, 0);
    if(c0 & lsb2) s = uint64_t_toggle_at(s, 1);
    if(c0 & lsb3) s = uint64_t_toggle_at(s, 2);
    if(c0 & lsb4) s = uint64_t_toggle_at(s, 3);
    if(c0 & lsb5) s = uint64_t_toggle_at(s, 4);
    if(c0 & lsb6) s = uint64_t_toggle_at(s, 5);
    *sol = s;
    return 0;
}

int64_t
uint64a_gj_v7_avx(const uint64_t m[8], uint64_t* const restrict sol) {
    __m256i c0_3 = _mm256_loadu_si256((__m256i*) (m + 0));
    uint64_t c4 = m[4];
    uint64_t c5 = m[5];
    uint64_t c6 = m[6];
    uint64_t c7 = m[7];

    uint64_t lsb7 = uint64_t_lsb(c7);
    if(unlikely(!lsb7)) {
        return -1; // singular
    }

    uint64_t c7_reduc = c7 ^ lsb7;
    __m256i reduc = _mm256_set1_epi64x(c7_reduc);
    __m256i vmask = _mm256_set1_epi64x(lsb7);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb7) c4 ^= c7_reduc;
    if(c5 & lsb7) c5 ^= c7_reduc;
    if(c6 & lsb7) c6 ^= c7_reduc;
    uint64_t mask = ~lsb7;

    uint64_t lsb6 = uint64_t_lsb(c6 & mask);
    if(unlikely(!lsb6)) {
        return -1; // singular
    }

    uint64_t c6_reduc = c6 ^ lsb6;
    reduc = _mm256_set1_epi64x(c6_reduc);
    vmask = _mm256_set1_epi64x(lsb6);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb6) c4 ^= c6_reduc;
    if(c5 & lsb6) c5 ^= c6_reduc;
    mask ^= lsb6;

    uint64_t lsb5 = uint64_t_lsb(c5 & mask);
    if(unlikely(!lsb5)) {
        return -1; // singular
    }

    uint64_t c5_reduc = c5 ^ lsb5;
    reduc = _mm256_set1_epi64x(c5_reduc);
    vmask = _mm256_set1_epi64x(lsb5);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb5) c4 ^= c5_reduc;
    uint64_t c0 = _mm256_extract_epi64(c0_3, 0);
    uint64_t c1 = _mm256_extract_epi64(c0_3, 1);
    uint64_t c2 = _mm256_extract_epi64(c0_3, 2);
    uint64_t c3 = _mm256_extract_epi64(c0_3, 3);
    mask ^= lsb5;

    uint64_t lsb4 = uint64_t_lsb(c4 & mask);
    if(unlikely(!lsb4)) {
        return -1; // singular
    }

    uint64_t c4_reduc = c4 ^ lsb4;
    if(c0 & lsb4) c0 ^= c4_reduc;
    if(c1 & lsb4) c1 ^= c4_reduc;
    if(c2 & lsb4) c2 ^= c4_reduc;
    if(c3 & lsb4) c3 ^= c4_reduc;
    mask ^= lsb4;

    uint64_t lsb3 = uint64_t_lsb(c3 & mask);
    if(unlikely(!lsb3)) {
        return -1; // singular
    }

    uint64_t c3_reduc = c3 ^ lsb3;
    if(c0 & lsb3) c0 ^= c3_reduc;
    if(c1 & lsb3) c1 ^= c3_reduc;
    if(c2 & lsb3) c2 ^= c3_reduc;
    mask ^= lsb3;

    uint64_t lsb2 = uint64_t_lsb(c2 & mask);
    if(unlikely(!lsb2)) {
        return -1; // singular
    }

    uint64_t c2_reduc = c2 ^ lsb2;
    if(c0 & lsb2) c0 ^= c2_reduc;
    if(c1 & lsb2) c1 ^= c2_reduc;
    mask ^= lsb2;

    uint64_t lsb1 = uint64_t_lsb(c1 & mask);
    if(unlikely(!lsb1)) {
        return -1; // singular
    }

    if(c0 & lsb1) c0 ^= c1 ^ lsb1;
    mask ^= lsb1;

    if(likely(mask & c0)) { // check system consistency
        return mask & c0; // not solvable
    }

    uint64_t s = 0x0ULL;
    if(c0 & lsb1) s = uint64_t_toggle_at(s, 0);
    if(c0 & lsb2) s = uint64_t_toggle_at(s, 1);
    if(c0 & lsb3) s = uint64_t_toggle_at(s, 2);
    if(c0 & lsb4) s = uint64_t_toggle_at(s, 3);
    if(c0 & lsb5) s = uint64_t_toggle_at(s, 4);
    if(c0 & lsb6) s = uint64_t_toggle_at(s, 5);
    if(c0 & lsb7) s = uint64_t_toggle_at(s, 6);
    *sol = s;
    return 0;
}

int64_t
uint64a_gj_v8_avx(const uint64_t m[9], uint64_t* const restrict sol) {
    __m256i c0_3 = _mm256_loadu_si256((__m256i*) (m + 0));
    uint64_t c4 = m[4];
    uint64_t c5 = m[5];
    uint64_t c6 = m[6];
    uint64_t c7 = m[7];
    uint64_t c8 = m[8];

    uint64_t lsb8 = uint64_t_lsb(c8);
    if(unlikely(!lsb8)) {
        return -1; // singular
    }

    uint64_t c8_reduc = c8 ^ lsb8;
    __m256i reduc = _mm256_set1_epi64x(c8_reduc);
    __m256i vmask = _mm256_set1_epi64x(lsb8);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb8) c4 ^= c8_reduc;
    if(c5 & lsb8) c5 ^= c8_reduc;
    if(c6 & lsb8) c6 ^= c8_reduc;
    if(c7 & lsb8) c7 ^= c8_reduc;
    uint64_t mask = ~lsb8;

    uint64_t lsb7 = uint64_t_lsb(c7 & mask);
    if(unlikely(!lsb7)) {
        return -1; // singular
    }

    uint64_t c7_reduc = c7 ^ lsb7;
    reduc = _mm256_set1_epi64x(c7_reduc);
    vmask = _mm256_set1_epi64x(lsb7);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb7) c4 ^= c7_reduc;
    if(c5 & lsb7) c5 ^= c7_reduc;
    if(c6 & lsb7) c6 ^= c7_reduc;
    mask ^= lsb7;

    uint64_t lsb6 = uint64_t_lsb(c6 & mask);
    if(unlikely(!lsb6)) {
        return -1; // singular
    }

    uint64_t c6_reduc = c6 ^ lsb6;
    reduc = _mm256_set1_epi64x(c6_reduc);
    vmask = _mm256_set1_epi64x(lsb6);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb6) c4 ^= c6_reduc;
    if(c5 & lsb6) c5 ^= c6_reduc;
    mask ^= lsb6;

    uint64_t lsb5 = uint64_t_lsb(c5 & mask);
    if(unlikely(!lsb5)) {
        return -1; // singular
    }

    uint64_t c5_reduc = c5 ^ lsb5;
    reduc = _mm256_set1_epi64x(c5_reduc);
    vmask = _mm256_set1_epi64x(lsb5);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb5) c4 ^= c5_reduc;
    uint64_t c0 = _mm256_extract_epi64(c0_3, 0);
    uint64_t c1 = _mm256_extract_epi64(c0_3, 1);
    uint64_t c2 = _mm256_extract_epi64(c0_3, 2);
    uint64_t c3 = _mm256_extract_epi64(c0_3, 3);
    mask ^= lsb5;

    uint64_t lsb4 = uint64_t_lsb(c4 & mask);
    if(unlikely(!lsb4)) {
        return -1; // singular
    }

    uint64_t c4_reduc = c4 ^ lsb4;
    if(c0 & lsb4) c0 ^= c4_reduc;
    if(c1 & lsb4) c1 ^= c4_reduc;
    if(c2 & lsb4) c2 ^= c4_reduc;
    if(c3 & lsb4) c3 ^= c4_reduc;
    mask ^= lsb4;

    uint64_t lsb3 = uint64_t_lsb(c3 & mask);
    if(unlikely(!lsb3)) {
        return -1; // singular
    }

    uint64_t c3_reduc = c3 ^ lsb3;
    if(c0 & lsb3) c0 ^= c3_reduc;
    if(c1 & lsb3) c1 ^= c3_reduc;
    if(c2 & lsb3) c2 ^= c3_reduc;
    mask ^= lsb3;

    uint64_t lsb2 = uint64_t_lsb(c2 & mask);
    if(unlikely(!lsb2)) {
        return -1; // singular
    }

    uint64_t c2_reduc = c2 ^ lsb2;
    if(c0 & lsb2) c0 ^= c2_reduc;
    if(c1 & lsb2) c1 ^= c2_reduc;
    mask ^= lsb2;

    uint64_t lsb1 = uint64_t_lsb(c1 & mask);
    if(unlikely(!lsb1)) {
        return -1; // singular
    }

    if(c0 & lsb1) c0 ^= c1 ^ lsb1;
    mask ^= lsb1;

    if(likely(mask & c0)) { // check system consistency
        return mask & c0; // not solvable
    }

    uint64_t s = 0x0ULL;
    if(c0 & lsb1) s = uint64_t_toggle_at(s, 0);
    if(c0 & lsb2) s = uint64_t_toggle_at(s, 1);
    if(c0 & lsb3) s = uint64_t_toggle_at(s, 2);
    if(c0 & lsb4) s = uint64_t_toggle_at(s, 3);
    if(c0 & lsb5) s = uint64_t_toggle_at(s, 4);
    if(c0 & lsb6) s = uint64_t_toggle_at(s, 5);
    if(c0 & lsb7) s = uint64_t_toggle_at(s, 6);
    if(c0 & lsb8) s = uint64_t_toggle_at(s, 7);
    *sol = s;
    return 0;
}

int64_t
uint64a_gj_v9_avx(const uint64_t m[10], uint64_t* const restrict sol) {
    __m256i c0_3 = _mm256_loadu_si256((__m256i*) (m + 0));
    __m256i c4_7 = _mm256_loadu_si256((__m256i*) (m + 4));
    uint64_t c8 = m[8];
    uint64_t c9 = m[9];

    uint64_t lsb9 = uint64_t_lsb(c9);
    if(unlikely(!lsb9)) {
        return -1; // singular
    }

    uint64_t c9_reduc = c9 ^ lsb9;
    __m256i reduc = _mm256_set1_epi64x(c9_reduc);
    __m256i vmask = _mm256_set1_epi64x(lsb9);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb9) c8 ^= c9_reduc;
    uint64_t mask = ~lsb9;

    uint64_t lsb8 = uint64_t_lsb(c8 & mask);
    if(unlikely(!lsb8)) {
        return -1; // singular
    }

    uint64_t c8_reduc = c8 ^ lsb8;
    reduc = _mm256_set1_epi64x(c8_reduc);
    vmask = _mm256_set1_epi64x(lsb8);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    uint64_t c4 = _mm256_extract_epi64(c4_7, 0);
    uint64_t c5 = _mm256_extract_epi64(c4_7, 1);
    uint64_t c6 = _mm256_extract_epi64(c4_7, 2);
    uint64_t c7 = _mm256_extract_epi64(c4_7, 3);
    mask ^= lsb8;

    uint64_t lsb7 = uint64_t_lsb(c7 & mask);
    if(unlikely(!lsb7)) {
        return -1; // singular
    }

    uint64_t c7_reduc = c7 ^ lsb7;
    reduc = _mm256_set1_epi64x(c7_reduc);
    vmask = _mm256_set1_epi64x(lsb7);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb7) c4 ^= c7_reduc;
    if(c5 & lsb7) c5 ^= c7_reduc;
    if(c6 & lsb7) c6 ^= c7_reduc;
    mask ^= lsb7;

    uint64_t lsb6 = uint64_t_lsb(c6 & mask);
    if(unlikely(!lsb6)) {
        return -1; // singular
    }

    uint64_t c6_reduc = c6 ^ lsb6;
    reduc = _mm256_set1_epi64x(c6_reduc);
    vmask = _mm256_set1_epi64x(lsb6);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb6) c4 ^= c6_reduc;
    if(c5 & lsb6) c5 ^= c6_reduc;
    mask ^= lsb6;

    uint64_t lsb5 = uint64_t_lsb(c5 & mask);
    if(unlikely(!lsb5)) {
        return -1; // singular
    }

    uint64_t c5_reduc = c5 ^ lsb5;
    reduc = _mm256_set1_epi64x(c5_reduc);
    vmask = _mm256_set1_epi64x(lsb5);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb5) c4 ^= c5_reduc;
    uint64_t c0 = _mm256_extract_epi64(c0_3, 0);
    uint64_t c1 = _mm256_extract_epi64(c0_3, 1);
    uint64_t c2 = _mm256_extract_epi64(c0_3, 2);
    uint64_t c3 = _mm256_extract_epi64(c0_3, 3);
    mask ^= lsb5;

    uint64_t lsb4 = uint64_t_lsb(c4 & mask);
    if(unlikely(!lsb4)) {
        return -1; // singular
    }

    uint64_t c4_reduc = c4 ^ lsb4;
    if(c0 & lsb4) c0 ^= c4_reduc;
    if(c1 & lsb4) c1 ^= c4_reduc;
    if(c2 & lsb4) c2 ^= c4_reduc;
    if(c3 & lsb4) c3 ^= c4_reduc;
    mask ^= lsb4;

    uint64_t lsb3 = uint64_t_lsb(c3 & mask);
    if(unlikely(!lsb3)) {
        return -1; // singular
    }

    uint64_t c3_reduc = c3 ^ lsb3;
    if(c0 & lsb3) c0 ^= c3_reduc;
    if(c1 & lsb3) c1 ^= c3_reduc;
    if(c2 & lsb3) c2 ^= c3_reduc;
    mask ^= lsb3;

    uint64_t lsb2 = uint64_t_lsb(c2 & mask);
    if(unlikely(!lsb2)) {
        return -1; // singular
    }

    uint64_t c2_reduc = c2 ^ lsb2;
    if(c0 & lsb2) c0 ^= c2_reduc;
    if(c1 & lsb2) c1 ^= c2_reduc;
    mask ^= lsb2;

    uint64_t lsb1 = uint64_t_lsb(c1 & mask);
    if(unlikely(!lsb1)) {
        return -1; // singular
    }

    if(c0 & lsb1) c0 ^= c1 ^ lsb1;
    mask ^= lsb1;

    if(likely(mask & c0)) { // check system consistency
        return mask & c0; // not solvable
    }

    uint64_t s = 0x0ULL;
    if(c0 & lsb1) s = uint64_t_toggle_at(s, 0);
    if(c0 & lsb2) s = uint64_t_toggle_at(s, 1);
    if(c0 & lsb3) s = uint64_t_toggle_at(s, 2);
    if(c0 & lsb4) s = uint64_t_toggle_at(s, 3);
    if(c0 & lsb5) s = uint64_t_toggle_at(s, 4);
    if(c0 & lsb6) s = uint64_t_toggle_at(s, 5);
    if(c0 & lsb7) s = uint64_t_toggle_at(s, 6);
    if(c0 & lsb8) s = uint64_t_toggle_at(s, 7);
    if(c0 & lsb9) s = uint64_t_toggle_at(s, 8);
    *sol = s;
    return 0;
}

int64_t
uint64a_gj_v10_avx(const uint64_t m[11], uint64_t* const restrict sol) {
    __m256i c0_3 = _mm256_loadu_si256((__m256i*) (m + 0));
    __m256i c4_7 = _mm256_loadu_si256((__m256i*) (m + 4));
    uint64_t c8 = m[8];
    uint64_t c9 = m[9];
    uint64_t c10 = m[10];

    uint64_t lsb10 = uint64_t_lsb(c10);
    if(unlikely(!lsb10)) {
        return -1; // singular
    }

    uint64_t c10_reduc = c10 ^ lsb10;
    __m256i reduc = _mm256_set1_epi64x(c10_reduc);
    __m256i vmask = _mm256_set1_epi64x(lsb10);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb10) c8 ^= c10_reduc;
    if(c9 & lsb10) c9 ^= c10_reduc;
    uint64_t mask = ~lsb10;

    uint64_t lsb9 = uint64_t_lsb(c9 & mask);
    if(unlikely(!lsb9)) {
        return -1; // singular
    }

    uint64_t c9_reduc = c9 ^ lsb9;
    reduc = _mm256_set1_epi64x(c9_reduc);
    vmask = _mm256_set1_epi64x(lsb9);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb9) c8 ^= c9_reduc;
    uint64_t c4 = _mm256_extract_epi64(c4_7, 0);
    uint64_t c5 = _mm256_extract_epi64(c4_7, 1);
    uint64_t c6 = _mm256_extract_epi64(c4_7, 2);
    uint64_t c7 = _mm256_extract_epi64(c4_7, 3);
    mask ^= lsb9;

    uint64_t lsb8 = uint64_t_lsb(c8 & mask);
    if(unlikely(!lsb8)) {
        return -1; // singular
    }

    uint64_t c8_reduc = c8 ^ lsb8;
    reduc = _mm256_set1_epi64x(c8_reduc);
    vmask = _mm256_set1_epi64x(lsb8);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb8) c4 ^= c8_reduc;
    if(c5 & lsb8) c5 ^= c8_reduc;
    if(c6 & lsb8) c6 ^= c8_reduc;
    if(c7 & lsb8) c7 ^= c8_reduc;
    mask ^= lsb8;

    uint64_t lsb7 = uint64_t_lsb(c7 & mask);
    if(unlikely(!lsb7)) {
        return -1; // singular
    }

    uint64_t c7_reduc = c7 ^ lsb7;
    reduc = _mm256_set1_epi64x(c7_reduc);
    vmask = _mm256_set1_epi64x(lsb7);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb7) c4 ^= c7_reduc;
    if(c5 & lsb7) c5 ^= c7_reduc;
    if(c6 & lsb7) c6 ^= c7_reduc;
    mask ^= lsb7;

    uint64_t lsb6 = uint64_t_lsb(c6 & mask);
    if(unlikely(!lsb6)) {
        return -1; // singular
    }

    uint64_t c6_reduc = c6 ^ lsb6;
    reduc = _mm256_set1_epi64x(c6_reduc);
    vmask = _mm256_set1_epi64x(lsb6);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb6) c4 ^= c6_reduc;
    if(c5 & lsb6) c5 ^= c6_reduc;
    mask ^= lsb6;

    uint64_t lsb5 = uint64_t_lsb(c5 & mask);
    if(unlikely(!lsb5)) {
        return -1; // singular
    }

    uint64_t c5_reduc = c5 ^ lsb5;
    reduc = _mm256_set1_epi64x(c5_reduc);
    vmask = _mm256_set1_epi64x(lsb5);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb5) c4 ^= c5_reduc;
    uint64_t c0 = _mm256_extract_epi64(c0_3, 0);
    uint64_t c1 = _mm256_extract_epi64(c0_3, 1);
    uint64_t c2 = _mm256_extract_epi64(c0_3, 2);
    uint64_t c3 = _mm256_extract_epi64(c0_3, 3);
    mask ^= lsb5;

    uint64_t lsb4 = uint64_t_lsb(c4 & mask);
    if(unlikely(!lsb4)) {
        return -1; // singular
    }

    uint64_t c4_reduc = c4 ^ lsb4;
    if(c0 & lsb4) c0 ^= c4_reduc;
    if(c1 & lsb4) c1 ^= c4_reduc;
    if(c2 & lsb4) c2 ^= c4_reduc;
    if(c3 & lsb4) c3 ^= c4_reduc;
    mask ^= lsb4;

    uint64_t lsb3 = uint64_t_lsb(c3 & mask);
    if(unlikely(!lsb3)) {
        return -1; // singular
    }

    uint64_t c3_reduc = c3 ^ lsb3;
    if(c0 & lsb3) c0 ^= c3_reduc;
    if(c1 & lsb3) c1 ^= c3_reduc;
    if(c2 & lsb3) c2 ^= c3_reduc;
    mask ^= lsb3;

    uint64_t lsb2 = uint64_t_lsb(c2 & mask);
    if(unlikely(!lsb2)) {
        return -1; // singular
    }

    uint64_t c2_reduc = c2 ^ lsb2;
    if(c0 & lsb2) c0 ^= c2_reduc;
    if(c1 & lsb2) c1 ^= c2_reduc;
    mask ^= lsb2;

    uint64_t lsb1 = uint64_t_lsb(c1 & mask);
    if(unlikely(!lsb1)) {
        return -1; // singular
    }

    if(c0 & lsb1) c0 ^= c1 ^ lsb1;
    mask ^= lsb1;

    if(likely(mask & c0)) { // check system consistency
        return mask & c0; // not solvable
    }

    uint64_t s = 0x0ULL;
    if(c0 & lsb1) s = uint64_t_toggle_at(s, 0);
    if(c0 & lsb2) s = uint64_t_toggle_at(s, 1);
    if(c0 & lsb3) s = uint64_t_toggle_at(s, 2);
    if(c0 & lsb4) s = uint64_t_toggle_at(s, 3);
    if(c0 & lsb5) s = uint64_t_toggle_at(s, 4);
    if(c0 & lsb6) s = uint64_t_toggle_at(s, 5);
    if(c0 & lsb7) s = uint64_t_toggle_at(s, 6);
    if(c0 & lsb8) s = uint64_t_toggle_at(s, 7);
    if(c0 & lsb9) s = uint64_t_toggle_at(s, 8);
    if(c0 & lsb10) s = uint64_t_toggle_at(s, 9);
    *sol = s;
    return 0;
}

int64_t
uint64a_gj_v11_avx(const uint64_t m[12], uint64_t* const restrict sol) {
    __m256i c0_3 = _mm256_loadu_si256((__m256i*) (m + 0));
    __m256i c4_7 = _mm256_loadu_si256((__m256i*) (m + 4));
    uint64_t c8 = m[8];
    uint64_t c9 = m[9];
    uint64_t c10 = m[10];
    uint64_t c11 = m[11];

    uint64_t lsb11 = uint64_t_lsb(c11);
    if(unlikely(!lsb11)) {
        return -1; // singular
    }

    uint64_t c11_reduc = c11 ^ lsb11;
    __m256i reduc = _mm256_set1_epi64x(c11_reduc);
    __m256i vmask = _mm256_set1_epi64x(lsb11);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb11) c8 ^= c11_reduc;
    if(c9 & lsb11) c9 ^= c11_reduc;
    if(c10 & lsb11) c10 ^= c11_reduc;
    uint64_t mask = ~lsb11;

    uint64_t lsb10 = uint64_t_lsb(c10 & mask);
    if(unlikely(!lsb10)) {
        return -1; // singular
    }

    uint64_t c10_reduc = c10 ^ lsb10;
    reduc = _mm256_set1_epi64x(c10_reduc);
    vmask = _mm256_set1_epi64x(lsb10);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb10) c8 ^= c10_reduc;
    if(c9 & lsb10) c9 ^= c10_reduc;
    mask ^= lsb10;

    uint64_t lsb9 = uint64_t_lsb(c9 & mask);
    if(unlikely(!lsb9)) {
        return -1; // singular
    }

    uint64_t c9_reduc = c9 ^ lsb9;
    reduc = _mm256_set1_epi64x(c9_reduc);
    vmask = _mm256_set1_epi64x(lsb9);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb9) c8 ^= c9_reduc;
    uint64_t c4 = _mm256_extract_epi64(c4_7, 0);
    uint64_t c5 = _mm256_extract_epi64(c4_7, 1);
    uint64_t c6 = _mm256_extract_epi64(c4_7, 2);
    uint64_t c7 = _mm256_extract_epi64(c4_7, 3);
    mask ^= lsb9;

    uint64_t lsb8 = uint64_t_lsb(c8 & mask);
    if(unlikely(!lsb8)) {
        return -1; // singular
    }

    uint64_t c8_reduc = c8 ^ lsb8;
    reduc = _mm256_set1_epi64x(c8_reduc);
    vmask = _mm256_set1_epi64x(lsb8);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb8) c4 ^= c8_reduc;
    if(c5 & lsb8) c5 ^= c8_reduc;
    if(c6 & lsb8) c6 ^= c8_reduc;
    if(c7 & lsb8) c7 ^= c8_reduc;
    mask ^= lsb8;

    uint64_t lsb7 = uint64_t_lsb(c7 & mask);
    if(unlikely(!lsb7)) {
        return -1; // singular
    }

    uint64_t c7_reduc = c7 ^ lsb7;
    reduc = _mm256_set1_epi64x(c7_reduc);
    vmask = _mm256_set1_epi64x(lsb7);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb7) c4 ^= c7_reduc;
    if(c5 & lsb7) c5 ^= c7_reduc;
    if(c6 & lsb7) c6 ^= c7_reduc;
    mask ^= lsb7;

    uint64_t lsb6 = uint64_t_lsb(c6 & mask);
    if(unlikely(!lsb6)) {
        return -1; // singular
    }

    uint64_t c6_reduc = c6 ^ lsb6;
    reduc = _mm256_set1_epi64x(c6_reduc);
    vmask = _mm256_set1_epi64x(lsb6);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb6) c4 ^= c6_reduc;
    if(c5 & lsb6) c5 ^= c6_reduc;
    mask ^= lsb6;

    uint64_t lsb5 = uint64_t_lsb(c5 & mask);
    if(unlikely(!lsb5)) {
        return -1; // singular
    }

    uint64_t c5_reduc = c5 ^ lsb5;
    reduc = _mm256_set1_epi64x(c5_reduc);
    vmask = _mm256_set1_epi64x(lsb5);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb5) c4 ^= c5_reduc;
    uint64_t c0 = _mm256_extract_epi64(c0_3, 0);
    uint64_t c1 = _mm256_extract_epi64(c0_3, 1);
    uint64_t c2 = _mm256_extract_epi64(c0_3, 2);
    uint64_t c3 = _mm256_extract_epi64(c0_3, 3);
    mask ^= lsb5;

    uint64_t lsb4 = uint64_t_lsb(c4 & mask);
    if(unlikely(!lsb4)) {
        return -1; // singular
    }

    uint64_t c4_reduc = c4 ^ lsb4;
    if(c0 & lsb4) c0 ^= c4_reduc;
    if(c1 & lsb4) c1 ^= c4_reduc;
    if(c2 & lsb4) c2 ^= c4_reduc;
    if(c3 & lsb4) c3 ^= c4_reduc;
    mask ^= lsb4;

    uint64_t lsb3 = uint64_t_lsb(c3 & mask);
    if(unlikely(!lsb3)) {
        return -1; // singular
    }

    uint64_t c3_reduc = c3 ^ lsb3;
    if(c0 & lsb3) c0 ^= c3_reduc;
    if(c1 & lsb3) c1 ^= c3_reduc;
    if(c2 & lsb3) c2 ^= c3_reduc;
    mask ^= lsb3;

    uint64_t lsb2 = uint64_t_lsb(c2 & mask);
    if(unlikely(!lsb2)) {
        return -1; // singular
    }

    uint64_t c2_reduc = c2 ^ lsb2;
    if(c0 & lsb2) c0 ^= c2_reduc;
    if(c1 & lsb2) c1 ^= c2_reduc;
    mask ^= lsb2;

    uint64_t lsb1 = uint64_t_lsb(c1 & mask);
    if(unlikely(!lsb1)) {
        return -1; // singular
    }

    if(c0 & lsb1) c0 ^= c1 ^ lsb1;
    mask ^= lsb1;

    if(likely(mask & c0)) { // check system consistency
        return mask & c0; // not solvable
    }

    uint64_t s = 0x0ULL;
    if(c0 & lsb1) s = uint64_t_toggle_at(s, 0);
    if(c0 & lsb2) s = uint64_t_toggle_at(s, 1);
    if(c0 & lsb3) s = uint64_t_toggle_at(s, 2);
    if(c0 & lsb4) s = uint64_t_toggle_at(s, 3);
    if(c0 & lsb5) s = uint64_t_toggle_at(s, 4);
    if(c0 & lsb6) s = uint64_t_toggle_at(s, 5);
    if(c0 & lsb7) s = uint64_t_toggle_at(s, 6);
    if(c0 & lsb8) s = uint64_t_toggle_at(s, 7);
    if(c0 & lsb9) s = uint64_t_toggle_at(s, 8);
    if(c0 & lsb10) s = uint64_t_toggle_at(s, 9);
    if(c0 & lsb11) s = uint64_t_toggle_at(s, 10);
    *sol = s;
    return 0;
}

int64_t
uint64a_gj_v12_avx(const uint64_t m[13], uint64_t* const restrict sol) {
    __m256i c0_3 = _mm256_loadu_si256((__m256i*) (m + 0));
    __m256i c4_7 = _mm256_loadu_si256((__m256i*) (m + 4));
    uint64_t c8 = m[8];
    uint64_t c9 = m[9];
    uint64_t c10 = m[10];
    uint64_t c11 = m[11];
    uint64_t c12 = m[12];

    uint64_t lsb12 = uint64_t_lsb(c12);
    if(unlikely(!lsb12)) {
        return -1; // singular
    }

    uint64_t c12_reduc = c12 ^ lsb12;
    __m256i reduc = _mm256_set1_epi64x(c12_reduc);
    __m256i vmask = _mm256_set1_epi64x(lsb12);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb12) c8 ^= c12_reduc;
    if(c9 & lsb12) c9 ^= c12_reduc;
    if(c10 & lsb12) c10 ^= c12_reduc;
    if(c11 & lsb12) c11 ^= c12_reduc;
    uint64_t mask = ~lsb12;

    uint64_t lsb11 = uint64_t_lsb(c11 & mask);
    if(unlikely(!lsb11)) {
        return -1; // singular
    }

    uint64_t c11_reduc = c11 ^ lsb11;
    reduc = _mm256_set1_epi64x(c11_reduc);
    vmask = _mm256_set1_epi64x(lsb11);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb11) c8 ^= c11_reduc;
    if(c9 & lsb11) c9 ^= c11_reduc;
    if(c10 & lsb11) c10 ^= c11_reduc;
    mask ^= lsb11;

    uint64_t lsb10 = uint64_t_lsb(c10 & mask);
    if(unlikely(!lsb10)) {
        return -1; // singular
    }

    uint64_t c10_reduc = c10 ^ lsb10;
    reduc = _mm256_set1_epi64x(c10_reduc);
    vmask = _mm256_set1_epi64x(lsb10);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb10) c8 ^= c10_reduc;
    if(c9 & lsb10) c9 ^= c10_reduc;
    mask ^= lsb10;

    uint64_t lsb9 = uint64_t_lsb(c9 & mask);
    if(unlikely(!lsb9)) {
        return -1; // singular
    }

    uint64_t c9_reduc = c9 ^ lsb9;
    reduc = _mm256_set1_epi64x(c9_reduc);
    vmask = _mm256_set1_epi64x(lsb9);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb9) c8 ^= c9_reduc;
    uint64_t c4 = _mm256_extract_epi64(c4_7, 0);
    uint64_t c5 = _mm256_extract_epi64(c4_7, 1);
    uint64_t c6 = _mm256_extract_epi64(c4_7, 2);
    uint64_t c7 = _mm256_extract_epi64(c4_7, 3);
    mask ^= lsb9;

    uint64_t lsb8 = uint64_t_lsb(c8 & mask);
    if(unlikely(!lsb8)) {
        return -1; // singular
    }

    uint64_t c8_reduc = c8 ^ lsb8;
    reduc = _mm256_set1_epi64x(c8_reduc);
    vmask = _mm256_set1_epi64x(lsb8);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb8) c4 ^= c8_reduc;
    if(c5 & lsb8) c5 ^= c8_reduc;
    if(c6 & lsb8) c6 ^= c8_reduc;
    if(c7 & lsb8) c7 ^= c8_reduc;
    mask ^= lsb8;

    uint64_t lsb7 = uint64_t_lsb(c7 & mask);
    if(unlikely(!lsb7)) {
        return -1; // singular
    }

    uint64_t c7_reduc = c7 ^ lsb7;
    reduc = _mm256_set1_epi64x(c7_reduc);
    vmask = _mm256_set1_epi64x(lsb7);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb7) c4 ^= c7_reduc;
    if(c5 & lsb7) c5 ^= c7_reduc;
    if(c6 & lsb7) c6 ^= c7_reduc;
    mask ^= lsb7;

    uint64_t lsb6 = uint64_t_lsb(c6 & mask);
    if(unlikely(!lsb6)) {
        return -1; // singular
    }

    uint64_t c6_reduc = c6 ^ lsb6;
    reduc = _mm256_set1_epi64x(c6_reduc);
    vmask = _mm256_set1_epi64x(lsb6);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb6) c4 ^= c6_reduc;
    if(c5 & lsb6) c5 ^= c6_reduc;
    mask ^= lsb6;

    uint64_t lsb5 = uint64_t_lsb(c5 & mask);
    if(unlikely(!lsb5)) {
        return -1; // singular
    }

    uint64_t c5_reduc = c5 ^ lsb5;
    reduc = _mm256_set1_epi64x(c5_reduc);
    vmask = _mm256_set1_epi64x(lsb5);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb5) c4 ^= c5_reduc;
    uint64_t c0 = _mm256_extract_epi64(c0_3, 0);
    uint64_t c1 = _mm256_extract_epi64(c0_3, 1);
    uint64_t c2 = _mm256_extract_epi64(c0_3, 2);
    uint64_t c3 = _mm256_extract_epi64(c0_3, 3);
    mask ^= lsb5;

    uint64_t lsb4 = uint64_t_lsb(c4 & mask);
    if(unlikely(!lsb4)) {
        return -1; // singular
    }

    uint64_t c4_reduc = c4 ^ lsb4;
    if(c0 & lsb4) c0 ^= c4_reduc;
    if(c1 & lsb4) c1 ^= c4_reduc;
    if(c2 & lsb4) c2 ^= c4_reduc;
    if(c3 & lsb4) c3 ^= c4_reduc;
    mask ^= lsb4;

    uint64_t lsb3 = uint64_t_lsb(c3 & mask);
    if(unlikely(!lsb3)) {
        return -1; // singular
    }

    uint64_t c3_reduc = c3 ^ lsb3;
    if(c0 & lsb3) c0 ^= c3_reduc;
    if(c1 & lsb3) c1 ^= c3_reduc;
    if(c2 & lsb3) c2 ^= c3_reduc;
    mask ^= lsb3;

    uint64_t lsb2 = uint64_t_lsb(c2 & mask);
    if(unlikely(!lsb2)) {
        return -1; // singular
    }

    uint64_t c2_reduc = c2 ^ lsb2;
    if(c0 & lsb2) c0 ^= c2_reduc;
    if(c1 & lsb2) c1 ^= c2_reduc;
    mask ^= lsb2;

    uint64_t lsb1 = uint64_t_lsb(c1 & mask);
    if(unlikely(!lsb1)) {
        return -1; // singular
    }

    if(c0 & lsb1) c0 ^= c1 ^ lsb1;
    mask ^= lsb1;

    if(likely(mask & c0)) { // check system consistency
        return mask & c0; // not solvable
    }

    uint64_t s = 0x0ULL;
    if(c0 & lsb1) s = uint64_t_toggle_at(s, 0);
    if(c0 & lsb2) s = uint64_t_toggle_at(s, 1);
    if(c0 & lsb3) s = uint64_t_toggle_at(s, 2);
    if(c0 & lsb4) s = uint64_t_toggle_at(s, 3);
    if(c0 & lsb5) s = uint64_t_toggle_at(s, 4);
    if(c0 & lsb6) s = uint64_t_toggle_at(s, 5);
    if(c0 & lsb7) s = uint64_t_toggle_at(s, 6);
    if(c0 & lsb8) s = uint64_t_toggle_at(s, 7);
    if(c0 & lsb9) s = uint64_t_toggle_at(s, 8);
    if(c0 & lsb10) s = uint64_t_toggle_at(s, 9);
    if(c0 & lsb11) s = uint64_t_toggle_at(s, 10);
    if(c0 & lsb12) s = uint64_t_toggle_at(s, 11);
    *sol = s;
    return 0;
}

int64_t
uint64a_gj_v13_avx(const uint64_t m[14], uint64_t* const restrict sol) {
    __m256i c0_3 = _mm256_loadu_si256((__m256i*) (m + 0));
    __m256i c4_7 = _mm256_loadu_si256((__m256i*) (m + 4));
    __m256i c8_11 = _mm256_loadu_si256((__m256i*) (m + 8));
    uint64_t c12 = m[12];
    uint64_t c13 = m[13];

    uint64_t lsb13 = uint64_t_lsb(c13);
    if(unlikely(!lsb13)) {
        return -1; // singular
    }

    uint64_t c13_reduc = c13 ^ lsb13;
    __m256i reduc = _mm256_set1_epi64x(c13_reduc);
    __m256i vmask = _mm256_set1_epi64x(lsb13);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    if(c12 & lsb13) c12 ^= c13_reduc;
    uint64_t mask = ~lsb13;

    uint64_t lsb12 = uint64_t_lsb(c12 & mask);
    if(unlikely(!lsb12)) {
        return -1; // singular
    }

    uint64_t c12_reduc = c12 ^ lsb12;
    reduc = _mm256_set1_epi64x(c12_reduc);
    vmask = _mm256_set1_epi64x(lsb12);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    uint64_t c8 = _mm256_extract_epi64(c8_11, 0);
    uint64_t c9 = _mm256_extract_epi64(c8_11, 1);
    uint64_t c10 = _mm256_extract_epi64(c8_11, 2);
    uint64_t c11 = _mm256_extract_epi64(c8_11, 3);
    mask ^= lsb12;

    uint64_t lsb11 = uint64_t_lsb(c11 & mask);
    if(unlikely(!lsb11)) {
        return -1; // singular
    }

    uint64_t c11_reduc = c11 ^ lsb11;
    reduc = _mm256_set1_epi64x(c11_reduc);
    vmask = _mm256_set1_epi64x(lsb11);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb11) c8 ^= c11_reduc;
    if(c9 & lsb11) c9 ^= c11_reduc;
    if(c10 & lsb11) c10 ^= c11_reduc;
    mask ^= lsb11;

    uint64_t lsb10 = uint64_t_lsb(c10 & mask);
    if(unlikely(!lsb10)) {
        return -1; // singular
    }

    uint64_t c10_reduc = c10 ^ lsb10;
    reduc = _mm256_set1_epi64x(c10_reduc);
    vmask = _mm256_set1_epi64x(lsb10);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb10) c8 ^= c10_reduc;
    if(c9 & lsb10) c9 ^= c10_reduc;
    mask ^= lsb10;

    uint64_t lsb9 = uint64_t_lsb(c9 & mask);
    if(unlikely(!lsb9)) {
        return -1; // singular
    }

    uint64_t c9_reduc = c9 ^ lsb9;
    reduc = _mm256_set1_epi64x(c9_reduc);
    vmask = _mm256_set1_epi64x(lsb9);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb9) c8 ^= c9_reduc;
    uint64_t c4 = _mm256_extract_epi64(c4_7, 0);
    uint64_t c5 = _mm256_extract_epi64(c4_7, 1);
    uint64_t c6 = _mm256_extract_epi64(c4_7, 2);
    uint64_t c7 = _mm256_extract_epi64(c4_7, 3);
    mask ^= lsb9;

    uint64_t lsb8 = uint64_t_lsb(c8 & mask);
    if(unlikely(!lsb8)) {
        return -1; // singular
    }

    uint64_t c8_reduc = c8 ^ lsb8;
    reduc = _mm256_set1_epi64x(c8_reduc);
    vmask = _mm256_set1_epi64x(lsb8);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb8) c4 ^= c8_reduc;
    if(c5 & lsb8) c5 ^= c8_reduc;
    if(c6 & lsb8) c6 ^= c8_reduc;
    if(c7 & lsb8) c7 ^= c8_reduc;
    mask ^= lsb8;

    uint64_t lsb7 = uint64_t_lsb(c7 & mask);
    if(unlikely(!lsb7)) {
        return -1; // singular
    }

    uint64_t c7_reduc = c7 ^ lsb7;
    reduc = _mm256_set1_epi64x(c7_reduc);
    vmask = _mm256_set1_epi64x(lsb7);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb7) c4 ^= c7_reduc;
    if(c5 & lsb7) c5 ^= c7_reduc;
    if(c6 & lsb7) c6 ^= c7_reduc;
    mask ^= lsb7;

    uint64_t lsb6 = uint64_t_lsb(c6 & mask);
    if(unlikely(!lsb6)) {
        return -1; // singular
    }

    uint64_t c6_reduc = c6 ^ lsb6;
    reduc = _mm256_set1_epi64x(c6_reduc);
    vmask = _mm256_set1_epi64x(lsb6);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb6) c4 ^= c6_reduc;
    if(c5 & lsb6) c5 ^= c6_reduc;
    mask ^= lsb6;

    uint64_t lsb5 = uint64_t_lsb(c5 & mask);
    if(unlikely(!lsb5)) {
        return -1; // singular
    }

    uint64_t c5_reduc = c5 ^ lsb5;
    reduc = _mm256_set1_epi64x(c5_reduc);
    vmask = _mm256_set1_epi64x(lsb5);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb5) c4 ^= c5_reduc;
    uint64_t c0 = _mm256_extract_epi64(c0_3, 0);
    uint64_t c1 = _mm256_extract_epi64(c0_3, 1);
    uint64_t c2 = _mm256_extract_epi64(c0_3, 2);
    uint64_t c3 = _mm256_extract_epi64(c0_3, 3);
    mask ^= lsb5;

    uint64_t lsb4 = uint64_t_lsb(c4 & mask);
    if(unlikely(!lsb4)) {
        return -1; // singular
    }

    uint64_t c4_reduc = c4 ^ lsb4;
    if(c0 & lsb4) c0 ^= c4_reduc;
    if(c1 & lsb4) c1 ^= c4_reduc;
    if(c2 & lsb4) c2 ^= c4_reduc;
    if(c3 & lsb4) c3 ^= c4_reduc;
    mask ^= lsb4;

    uint64_t lsb3 = uint64_t_lsb(c3 & mask);
    if(unlikely(!lsb3)) {
        return -1; // singular
    }

    uint64_t c3_reduc = c3 ^ lsb3;
    if(c0 & lsb3) c0 ^= c3_reduc;
    if(c1 & lsb3) c1 ^= c3_reduc;
    if(c2 & lsb3) c2 ^= c3_reduc;
    mask ^= lsb3;

    uint64_t lsb2 = uint64_t_lsb(c2 & mask);
    if(unlikely(!lsb2)) {
        return -1; // singular
    }

    uint64_t c2_reduc = c2 ^ lsb2;
    if(c0 & lsb2) c0 ^= c2_reduc;
    if(c1 & lsb2) c1 ^= c2_reduc;
    mask ^= lsb2;

    uint64_t lsb1 = uint64_t_lsb(c1 & mask);
    if(unlikely(!lsb1)) {
        return -1; // singular
    }

    if(c0 & lsb1) c0 ^= c1 ^ lsb1;
    mask ^= lsb1;

    if(likely(mask & c0)) { // check system consistency
        return mask & c0; // not solvable
    }

    uint64_t s = 0x0ULL;
    if(c0 & lsb1) s = uint64_t_toggle_at(s, 0);
    if(c0 & lsb2) s = uint64_t_toggle_at(s, 1);
    if(c0 & lsb3) s = uint64_t_toggle_at(s, 2);
    if(c0 & lsb4) s = uint64_t_toggle_at(s, 3);
    if(c0 & lsb5) s = uint64_t_toggle_at(s, 4);
    if(c0 & lsb6) s = uint64_t_toggle_at(s, 5);
    if(c0 & lsb7) s = uint64_t_toggle_at(s, 6);
    if(c0 & lsb8) s = uint64_t_toggle_at(s, 7);
    if(c0 & lsb9) s = uint64_t_toggle_at(s, 8);
    if(c0 & lsb10) s = uint64_t_toggle_at(s, 9);
    if(c0 & lsb11) s = uint64_t_toggle_at(s, 10);
    if(c0 & lsb12) s = uint64_t_toggle_at(s, 11);
    if(c0 & lsb13) s = uint64_t_toggle_at(s, 12);
    *sol = s;
    return 0;
}

int64_t
uint64a_gj_v14_avx(const uint64_t m[15], uint64_t* const restrict sol) {
    __m256i c0_3 = _mm256_loadu_si256((__m256i*) (m + 0));
    __m256i c4_7 = _mm256_loadu_si256((__m256i*) (m + 4));
    __m256i c8_11 = _mm256_loadu_si256((__m256i*) (m + 8));
    uint64_t c12 = m[12];
    uint64_t c13 = m[13];
    uint64_t c14 = m[14];

    uint64_t lsb14 = uint64_t_lsb(c14);
    if(unlikely(!lsb14)) {
        return -1; // singular
    }

    uint64_t c14_reduc = c14 ^ lsb14;
    __m256i reduc = _mm256_set1_epi64x(c14_reduc);
    __m256i vmask = _mm256_set1_epi64x(lsb14);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    if(c12 & lsb14) c12 ^= c14_reduc;
    if(c13 & lsb14) c13 ^= c14_reduc;
    uint64_t mask = ~lsb14;

    uint64_t lsb13 = uint64_t_lsb(c13 & mask);
    if(unlikely(!lsb13)) {
        return -1; // singular
    }

    uint64_t c13_reduc = c13 ^ lsb13;
    reduc = _mm256_set1_epi64x(c13_reduc);
    vmask = _mm256_set1_epi64x(lsb13);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    if(c12 & lsb13) c12 ^= c13_reduc;
    uint64_t c8 = _mm256_extract_epi64(c8_11, 0);
    uint64_t c9 = _mm256_extract_epi64(c8_11, 1);
    uint64_t c10 = _mm256_extract_epi64(c8_11, 2);
    uint64_t c11 = _mm256_extract_epi64(c8_11, 3);
    mask ^= lsb13;

    uint64_t lsb12 = uint64_t_lsb(c12 & mask);
    if(unlikely(!lsb12)) {
        return -1; // singular
    }

    uint64_t c12_reduc = c12 ^ lsb12;
    reduc = _mm256_set1_epi64x(c12_reduc);
    vmask = _mm256_set1_epi64x(lsb12);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb12) c8 ^= c12_reduc;
    if(c9 & lsb12) c9 ^= c12_reduc;
    if(c10 & lsb12) c10 ^= c12_reduc;
    if(c11 & lsb12) c11 ^= c12_reduc;
    mask ^= lsb12;

    uint64_t lsb11 = uint64_t_lsb(c11 & mask);
    if(unlikely(!lsb11)) {
        return -1; // singular
    }

    uint64_t c11_reduc = c11 ^ lsb11;
    reduc = _mm256_set1_epi64x(c11_reduc);
    vmask = _mm256_set1_epi64x(lsb11);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb11) c8 ^= c11_reduc;
    if(c9 & lsb11) c9 ^= c11_reduc;
    if(c10 & lsb11) c10 ^= c11_reduc;
    mask ^= lsb11;

    uint64_t lsb10 = uint64_t_lsb(c10 & mask);
    if(unlikely(!lsb10)) {
        return -1; // singular
    }

    uint64_t c10_reduc = c10 ^ lsb10;
    reduc = _mm256_set1_epi64x(c10_reduc);
    vmask = _mm256_set1_epi64x(lsb10);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb10) c8 ^= c10_reduc;
    if(c9 & lsb10) c9 ^= c10_reduc;
    mask ^= lsb10;

    uint64_t lsb9 = uint64_t_lsb(c9 & mask);
    if(unlikely(!lsb9)) {
        return -1; // singular
    }

    uint64_t c9_reduc = c9 ^ lsb9;
    reduc = _mm256_set1_epi64x(c9_reduc);
    vmask = _mm256_set1_epi64x(lsb9);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb9) c8 ^= c9_reduc;
    uint64_t c4 = _mm256_extract_epi64(c4_7, 0);
    uint64_t c5 = _mm256_extract_epi64(c4_7, 1);
    uint64_t c6 = _mm256_extract_epi64(c4_7, 2);
    uint64_t c7 = _mm256_extract_epi64(c4_7, 3);
    mask ^= lsb9;

    uint64_t lsb8 = uint64_t_lsb(c8 & mask);
    if(unlikely(!lsb8)) {
        return -1; // singular
    }

    uint64_t c8_reduc = c8 ^ lsb8;
    reduc = _mm256_set1_epi64x(c8_reduc);
    vmask = _mm256_set1_epi64x(lsb8);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb8) c4 ^= c8_reduc;
    if(c5 & lsb8) c5 ^= c8_reduc;
    if(c6 & lsb8) c6 ^= c8_reduc;
    if(c7 & lsb8) c7 ^= c8_reduc;
    mask ^= lsb8;

    uint64_t lsb7 = uint64_t_lsb(c7 & mask);
    if(unlikely(!lsb7)) {
        return -1; // singular
    }

    uint64_t c7_reduc = c7 ^ lsb7;
    reduc = _mm256_set1_epi64x(c7_reduc);
    vmask = _mm256_set1_epi64x(lsb7);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb7) c4 ^= c7_reduc;
    if(c5 & lsb7) c5 ^= c7_reduc;
    if(c6 & lsb7) c6 ^= c7_reduc;
    mask ^= lsb7;

    uint64_t lsb6 = uint64_t_lsb(c6 & mask);
    if(unlikely(!lsb6)) {
        return -1; // singular
    }

    uint64_t c6_reduc = c6 ^ lsb6;
    reduc = _mm256_set1_epi64x(c6_reduc);
    vmask = _mm256_set1_epi64x(lsb6);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb6) c4 ^= c6_reduc;
    if(c5 & lsb6) c5 ^= c6_reduc;
    mask ^= lsb6;

    uint64_t lsb5 = uint64_t_lsb(c5 & mask);
    if(unlikely(!lsb5)) {
        return -1; // singular
    }

    uint64_t c5_reduc = c5 ^ lsb5;
    reduc = _mm256_set1_epi64x(c5_reduc);
    vmask = _mm256_set1_epi64x(lsb5);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb5) c4 ^= c5_reduc;
    uint64_t c0 = _mm256_extract_epi64(c0_3, 0);
    uint64_t c1 = _mm256_extract_epi64(c0_3, 1);
    uint64_t c2 = _mm256_extract_epi64(c0_3, 2);
    uint64_t c3 = _mm256_extract_epi64(c0_3, 3);
    mask ^= lsb5;

    uint64_t lsb4 = uint64_t_lsb(c4 & mask);
    if(unlikely(!lsb4)) {
        return -1; // singular
    }

    uint64_t c4_reduc = c4 ^ lsb4;
    if(c0 & lsb4) c0 ^= c4_reduc;
    if(c1 & lsb4) c1 ^= c4_reduc;
    if(c2 & lsb4) c2 ^= c4_reduc;
    if(c3 & lsb4) c3 ^= c4_reduc;
    mask ^= lsb4;

    uint64_t lsb3 = uint64_t_lsb(c3 & mask);
    if(unlikely(!lsb3)) {
        return -1; // singular
    }

    uint64_t c3_reduc = c3 ^ lsb3;
    if(c0 & lsb3) c0 ^= c3_reduc;
    if(c1 & lsb3) c1 ^= c3_reduc;
    if(c2 & lsb3) c2 ^= c3_reduc;
    mask ^= lsb3;

    uint64_t lsb2 = uint64_t_lsb(c2 & mask);
    if(unlikely(!lsb2)) {
        return -1; // singular
    }

    uint64_t c2_reduc = c2 ^ lsb2;
    if(c0 & lsb2) c0 ^= c2_reduc;
    if(c1 & lsb2) c1 ^= c2_reduc;
    mask ^= lsb2;

    uint64_t lsb1 = uint64_t_lsb(c1 & mask);
    if(unlikely(!lsb1)) {
        return -1; // singular
    }

    if(c0 & lsb1) c0 ^= c1 ^ lsb1;
    mask ^= lsb1;

    if(likely(mask & c0)) { // check system consistency
        return mask & c0; // not solvable
    }

    uint64_t s = 0x0ULL;
    if(c0 & lsb1) s = uint64_t_toggle_at(s, 0);
    if(c0 & lsb2) s = uint64_t_toggle_at(s, 1);
    if(c0 & lsb3) s = uint64_t_toggle_at(s, 2);
    if(c0 & lsb4) s = uint64_t_toggle_at(s, 3);
    if(c0 & lsb5) s = uint64_t_toggle_at(s, 4);
    if(c0 & lsb6) s = uint64_t_toggle_at(s, 5);
    if(c0 & lsb7) s = uint64_t_toggle_at(s, 6);
    if(c0 & lsb8) s = uint64_t_toggle_at(s, 7);
    if(c0 & lsb9) s = uint64_t_toggle_at(s, 8);
    if(c0 & lsb10) s = uint64_t_toggle_at(s, 9);
    if(c0 & lsb11) s = uint64_t_toggle_at(s, 10);
    if(c0 & lsb12) s = uint64_t_toggle_at(s, 11);
    if(c0 & lsb13) s = uint64_t_toggle_at(s, 12);
    if(c0 & lsb14) s = uint64_t_toggle_at(s, 13);
    *sol = s;
    return 0;
}

int64_t
uint64a_gj_v15_avx(const uint64_t m[16], uint64_t* const restrict sol) {
    __m256i c0_3 = _mm256_loadu_si256((__m256i*) (m + 0));
    __m256i c4_7 = _mm256_loadu_si256((__m256i*) (m + 4));
    __m256i c8_11 = _mm256_loadu_si256((__m256i*) (m + 8));
    uint64_t c12 = m[12];
    uint64_t c13 = m[13];
    uint64_t c14 = m[14];
    uint64_t c15 = m[15];

    uint64_t lsb15 = uint64_t_lsb(c15);
    if(unlikely(!lsb15)) {
        return -1; // singular
    }

    uint64_t c15_reduc = c15 ^ lsb15;
    __m256i reduc = _mm256_set1_epi64x(c15_reduc);
    __m256i vmask = _mm256_set1_epi64x(lsb15);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    if(c12 & lsb15) c12 ^= c15_reduc;
    if(c13 & lsb15) c13 ^= c15_reduc;
    if(c14 & lsb15) c14 ^= c15_reduc;
    uint64_t mask = ~lsb15;

    uint64_t lsb14 = uint64_t_lsb(c14 & mask);
    if(unlikely(!lsb14)) {
        return -1; // singular
    }

    uint64_t c14_reduc = c14 ^ lsb14;
    reduc = _mm256_set1_epi64x(c14_reduc);
    vmask = _mm256_set1_epi64x(lsb14);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    if(c12 & lsb14) c12 ^= c14_reduc;
    if(c13 & lsb14) c13 ^= c14_reduc;
    mask ^= lsb14;

    uint64_t lsb13 = uint64_t_lsb(c13 & mask);
    if(unlikely(!lsb13)) {
        return -1; // singular
    }

    uint64_t c13_reduc = c13 ^ lsb13;
    reduc = _mm256_set1_epi64x(c13_reduc);
    vmask = _mm256_set1_epi64x(lsb13);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    if(c12 & lsb13) c12 ^= c13_reduc;
    uint64_t c8 = _mm256_extract_epi64(c8_11, 0);
    uint64_t c9 = _mm256_extract_epi64(c8_11, 1);
    uint64_t c10 = _mm256_extract_epi64(c8_11, 2);
    uint64_t c11 = _mm256_extract_epi64(c8_11, 3);
    mask ^= lsb13;

    uint64_t lsb12 = uint64_t_lsb(c12 & mask);
    if(unlikely(!lsb12)) {
        return -1; // singular
    }

    uint64_t c12_reduc = c12 ^ lsb12;
    reduc = _mm256_set1_epi64x(c12_reduc);
    vmask = _mm256_set1_epi64x(lsb12);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb12) c8 ^= c12_reduc;
    if(c9 & lsb12) c9 ^= c12_reduc;
    if(c10 & lsb12) c10 ^= c12_reduc;
    if(c11 & lsb12) c11 ^= c12_reduc;
    mask ^= lsb12;

    uint64_t lsb11 = uint64_t_lsb(c11 & mask);
    if(unlikely(!lsb11)) {
        return -1; // singular
    }

    uint64_t c11_reduc = c11 ^ lsb11;
    reduc = _mm256_set1_epi64x(c11_reduc);
    vmask = _mm256_set1_epi64x(lsb11);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb11) c8 ^= c11_reduc;
    if(c9 & lsb11) c9 ^= c11_reduc;
    if(c10 & lsb11) c10 ^= c11_reduc;
    mask ^= lsb11;

    uint64_t lsb10 = uint64_t_lsb(c10 & mask);
    if(unlikely(!lsb10)) {
        return -1; // singular
    }

    uint64_t c10_reduc = c10 ^ lsb10;
    reduc = _mm256_set1_epi64x(c10_reduc);
    vmask = _mm256_set1_epi64x(lsb10);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb10) c8 ^= c10_reduc;
    if(c9 & lsb10) c9 ^= c10_reduc;
    mask ^= lsb10;

    uint64_t lsb9 = uint64_t_lsb(c9 & mask);
    if(unlikely(!lsb9)) {
        return -1; // singular
    }

    uint64_t c9_reduc = c9 ^ lsb9;
    reduc = _mm256_set1_epi64x(c9_reduc);
    vmask = _mm256_set1_epi64x(lsb9);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb9) c8 ^= c9_reduc;
    uint64_t c4 = _mm256_extract_epi64(c4_7, 0);
    uint64_t c5 = _mm256_extract_epi64(c4_7, 1);
    uint64_t c6 = _mm256_extract_epi64(c4_7, 2);
    uint64_t c7 = _mm256_extract_epi64(c4_7, 3);
    mask ^= lsb9;

    uint64_t lsb8 = uint64_t_lsb(c8 & mask);
    if(unlikely(!lsb8)) {
        return -1; // singular
    }

    uint64_t c8_reduc = c8 ^ lsb8;
    reduc = _mm256_set1_epi64x(c8_reduc);
    vmask = _mm256_set1_epi64x(lsb8);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb8) c4 ^= c8_reduc;
    if(c5 & lsb8) c5 ^= c8_reduc;
    if(c6 & lsb8) c6 ^= c8_reduc;
    if(c7 & lsb8) c7 ^= c8_reduc;
    mask ^= lsb8;

    uint64_t lsb7 = uint64_t_lsb(c7 & mask);
    if(unlikely(!lsb7)) {
        return -1; // singular
    }

    uint64_t c7_reduc = c7 ^ lsb7;
    reduc = _mm256_set1_epi64x(c7_reduc);
    vmask = _mm256_set1_epi64x(lsb7);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb7) c4 ^= c7_reduc;
    if(c5 & lsb7) c5 ^= c7_reduc;
    if(c6 & lsb7) c6 ^= c7_reduc;
    mask ^= lsb7;

    uint64_t lsb6 = uint64_t_lsb(c6 & mask);
    if(unlikely(!lsb6)) {
        return -1; // singular
    }

    uint64_t c6_reduc = c6 ^ lsb6;
    reduc = _mm256_set1_epi64x(c6_reduc);
    vmask = _mm256_set1_epi64x(lsb6);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb6) c4 ^= c6_reduc;
    if(c5 & lsb6) c5 ^= c6_reduc;
    mask ^= lsb6;

    uint64_t lsb5 = uint64_t_lsb(c5 & mask);
    if(unlikely(!lsb5)) {
        return -1; // singular
    }

    uint64_t c5_reduc = c5 ^ lsb5;
    reduc = _mm256_set1_epi64x(c5_reduc);
    vmask = _mm256_set1_epi64x(lsb5);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb5) c4 ^= c5_reduc;
    uint64_t c0 = _mm256_extract_epi64(c0_3, 0);
    uint64_t c1 = _mm256_extract_epi64(c0_3, 1);
    uint64_t c2 = _mm256_extract_epi64(c0_3, 2);
    uint64_t c3 = _mm256_extract_epi64(c0_3, 3);
    mask ^= lsb5;

    uint64_t lsb4 = uint64_t_lsb(c4 & mask);
    if(unlikely(!lsb4)) {
        return -1; // singular
    }

    uint64_t c4_reduc = c4 ^ lsb4;
    if(c0 & lsb4) c0 ^= c4_reduc;
    if(c1 & lsb4) c1 ^= c4_reduc;
    if(c2 & lsb4) c2 ^= c4_reduc;
    if(c3 & lsb4) c3 ^= c4_reduc;
    mask ^= lsb4;

    uint64_t lsb3 = uint64_t_lsb(c3 & mask);
    if(unlikely(!lsb3)) {
        return -1; // singular
    }

    uint64_t c3_reduc = c3 ^ lsb3;
    if(c0 & lsb3) c0 ^= c3_reduc;
    if(c1 & lsb3) c1 ^= c3_reduc;
    if(c2 & lsb3) c2 ^= c3_reduc;
    mask ^= lsb3;

    uint64_t lsb2 = uint64_t_lsb(c2 & mask);
    if(unlikely(!lsb2)) {
        return -1; // singular
    }

    uint64_t c2_reduc = c2 ^ lsb2;
    if(c0 & lsb2) c0 ^= c2_reduc;
    if(c1 & lsb2) c1 ^= c2_reduc;
    mask ^= lsb2;

    uint64_t lsb1 = uint64_t_lsb(c1 & mask);
    if(unlikely(!lsb1)) {
        return -1; // singular
    }

    if(c0 & lsb1) c0 ^= c1 ^ lsb1;
    mask ^= lsb1;

    if(likely(mask & c0)) { // check system consistency
        return mask & c0; // not solvable
    }

    uint64_t s = 0x0ULL;
    if(c0 & lsb1) s = uint64_t_toggle_at(s, 0);
    if(c0 & lsb2) s = uint64_t_toggle_at(s, 1);
    if(c0 & lsb3) s = uint64_t_toggle_at(s, 2);
    if(c0 & lsb4) s = uint64_t_toggle_at(s, 3);
    if(c0 & lsb5) s = uint64_t_toggle_at(s, 4);
    if(c0 & lsb6) s = uint64_t_toggle_at(s, 5);
    if(c0 & lsb7) s = uint64_t_toggle_at(s, 6);
    if(c0 & lsb8) s = uint64_t_toggle_at(s, 7);
    if(c0 & lsb9) s = uint64_t_toggle_at(s, 8);
    if(c0 & lsb10) s = uint64_t_toggle_at(s, 9);
    if(c0 & lsb11) s = uint64_t_toggle_at(s, 10);
    if(c0 & lsb12) s = uint64_t_toggle_at(s, 11);
    if(c0 & lsb13) s = uint64_t_toggle_at(s, 12);
    if(c0 & lsb14) s = uint64_t_toggle_at(s, 13);
    if(c0 & lsb15) s = uint64_t_toggle_at(s, 14);
    *sol = s;
    return 0;
}

int64_t
uint64a_gj_v16_avx(const uint64_t m[17], uint64_t* const restrict sol) {
    __m256i c0_3 = _mm256_loadu_si256((__m256i*) (m + 0));
    __m256i c4_7 = _mm256_loadu_si256((__m256i*) (m + 4));
    __m256i c8_11 = _mm256_loadu_si256((__m256i*) (m + 8));
    uint64_t c12 = m[12];
    uint64_t c13 = m[13];
    uint64_t c14 = m[14];
    uint64_t c15 = m[15];
    uint64_t c16 = m[16];

    uint64_t lsb16 = uint64_t_lsb(c16);
    if(unlikely(!lsb16)) {
        return -1; // singular
    }

    uint64_t c16_reduc = c16 ^ lsb16;
    __m256i reduc = _mm256_set1_epi64x(c16_reduc);
    __m256i vmask = _mm256_set1_epi64x(lsb16);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    if(c12 & lsb16) c12 ^= c16_reduc;
    if(c13 & lsb16) c13 ^= c16_reduc;
    if(c14 & lsb16) c14 ^= c16_reduc;
    if(c15 & lsb16) c15 ^= c16_reduc;
    uint64_t mask = ~lsb16;

    uint64_t lsb15 = uint64_t_lsb(c15 & mask);
    if(unlikely(!lsb15)) {
        return -1; // singular
    }

    uint64_t c15_reduc = c15 ^ lsb15;
    reduc = _mm256_set1_epi64x(c15_reduc);
    vmask = _mm256_set1_epi64x(lsb15);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    if(c12 & lsb15) c12 ^= c15_reduc;
    if(c13 & lsb15) c13 ^= c15_reduc;
    if(c14 & lsb15) c14 ^= c15_reduc;
    mask ^= lsb15;

    uint64_t lsb14 = uint64_t_lsb(c14 & mask);
    if(unlikely(!lsb14)) {
        return -1; // singular
    }

    uint64_t c14_reduc = c14 ^ lsb14;
    reduc = _mm256_set1_epi64x(c14_reduc);
    vmask = _mm256_set1_epi64x(lsb14);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    if(c12 & lsb14) c12 ^= c14_reduc;
    if(c13 & lsb14) c13 ^= c14_reduc;
    mask ^= lsb14;

    uint64_t lsb13 = uint64_t_lsb(c13 & mask);
    if(unlikely(!lsb13)) {
        return -1; // singular
    }

    uint64_t c13_reduc = c13 ^ lsb13;
    reduc = _mm256_set1_epi64x(c13_reduc);
    vmask = _mm256_set1_epi64x(lsb13);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    if(c12 & lsb13) c12 ^= c13_reduc;
    uint64_t c8 = _mm256_extract_epi64(c8_11, 0);
    uint64_t c9 = _mm256_extract_epi64(c8_11, 1);
    uint64_t c10 = _mm256_extract_epi64(c8_11, 2);
    uint64_t c11 = _mm256_extract_epi64(c8_11, 3);
    mask ^= lsb13;

    uint64_t lsb12 = uint64_t_lsb(c12 & mask);
    if(unlikely(!lsb12)) {
        return -1; // singular
    }

    uint64_t c12_reduc = c12 ^ lsb12;
    reduc = _mm256_set1_epi64x(c12_reduc);
    vmask = _mm256_set1_epi64x(lsb12);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb12) c8 ^= c12_reduc;
    if(c9 & lsb12) c9 ^= c12_reduc;
    if(c10 & lsb12) c10 ^= c12_reduc;
    if(c11 & lsb12) c11 ^= c12_reduc;
    mask ^= lsb12;

    uint64_t lsb11 = uint64_t_lsb(c11 & mask);
    if(unlikely(!lsb11)) {
        return -1; // singular
    }

    uint64_t c11_reduc = c11 ^ lsb11;
    reduc = _mm256_set1_epi64x(c11_reduc);
    vmask = _mm256_set1_epi64x(lsb11);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb11) c8 ^= c11_reduc;
    if(c9 & lsb11) c9 ^= c11_reduc;
    if(c10 & lsb11) c10 ^= c11_reduc;
    mask ^= lsb11;

    uint64_t lsb10 = uint64_t_lsb(c10 & mask);
    if(unlikely(!lsb10)) {
        return -1; // singular
    }

    uint64_t c10_reduc = c10 ^ lsb10;
    reduc = _mm256_set1_epi64x(c10_reduc);
    vmask = _mm256_set1_epi64x(lsb10);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb10) c8 ^= c10_reduc;
    if(c9 & lsb10) c9 ^= c10_reduc;
    mask ^= lsb10;

    uint64_t lsb9 = uint64_t_lsb(c9 & mask);
    if(unlikely(!lsb9)) {
        return -1; // singular
    }

    uint64_t c9_reduc = c9 ^ lsb9;
    reduc = _mm256_set1_epi64x(c9_reduc);
    vmask = _mm256_set1_epi64x(lsb9);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb9) c8 ^= c9_reduc;
    uint64_t c4 = _mm256_extract_epi64(c4_7, 0);
    uint64_t c5 = _mm256_extract_epi64(c4_7, 1);
    uint64_t c6 = _mm256_extract_epi64(c4_7, 2);
    uint64_t c7 = _mm256_extract_epi64(c4_7, 3);
    mask ^= lsb9;

    uint64_t lsb8 = uint64_t_lsb(c8 & mask);
    if(unlikely(!lsb8)) {
        return -1; // singular
    }

    uint64_t c8_reduc = c8 ^ lsb8;
    reduc = _mm256_set1_epi64x(c8_reduc);
    vmask = _mm256_set1_epi64x(lsb8);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb8) c4 ^= c8_reduc;
    if(c5 & lsb8) c5 ^= c8_reduc;
    if(c6 & lsb8) c6 ^= c8_reduc;
    if(c7 & lsb8) c7 ^= c8_reduc;
    mask ^= lsb8;

    uint64_t lsb7 = uint64_t_lsb(c7 & mask);
    if(unlikely(!lsb7)) {
        return -1; // singular
    }

    uint64_t c7_reduc = c7 ^ lsb7;
    reduc = _mm256_set1_epi64x(c7_reduc);
    vmask = _mm256_set1_epi64x(lsb7);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb7) c4 ^= c7_reduc;
    if(c5 & lsb7) c5 ^= c7_reduc;
    if(c6 & lsb7) c6 ^= c7_reduc;
    mask ^= lsb7;

    uint64_t lsb6 = uint64_t_lsb(c6 & mask);
    if(unlikely(!lsb6)) {
        return -1; // singular
    }

    uint64_t c6_reduc = c6 ^ lsb6;
    reduc = _mm256_set1_epi64x(c6_reduc);
    vmask = _mm256_set1_epi64x(lsb6);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb6) c4 ^= c6_reduc;
    if(c5 & lsb6) c5 ^= c6_reduc;
    mask ^= lsb6;

    uint64_t lsb5 = uint64_t_lsb(c5 & mask);
    if(unlikely(!lsb5)) {
        return -1; // singular
    }

    uint64_t c5_reduc = c5 ^ lsb5;
    reduc = _mm256_set1_epi64x(c5_reduc);
    vmask = _mm256_set1_epi64x(lsb5);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb5) c4 ^= c5_reduc;
    uint64_t c0 = _mm256_extract_epi64(c0_3, 0);
    uint64_t c1 = _mm256_extract_epi64(c0_3, 1);
    uint64_t c2 = _mm256_extract_epi64(c0_3, 2);
    uint64_t c3 = _mm256_extract_epi64(c0_3, 3);
    mask ^= lsb5;

    uint64_t lsb4 = uint64_t_lsb(c4 & mask);
    if(unlikely(!lsb4)) {
        return -1; // singular
    }

    uint64_t c4_reduc = c4 ^ lsb4;
    if(c0 & lsb4) c0 ^= c4_reduc;
    if(c1 & lsb4) c1 ^= c4_reduc;
    if(c2 & lsb4) c2 ^= c4_reduc;
    if(c3 & lsb4) c3 ^= c4_reduc;
    mask ^= lsb4;

    uint64_t lsb3 = uint64_t_lsb(c3 & mask);
    if(unlikely(!lsb3)) {
        return -1; // singular
    }

    uint64_t c3_reduc = c3 ^ lsb3;
    if(c0 & lsb3) c0 ^= c3_reduc;
    if(c1 & lsb3) c1 ^= c3_reduc;
    if(c2 & lsb3) c2 ^= c3_reduc;
    mask ^= lsb3;

    uint64_t lsb2 = uint64_t_lsb(c2 & mask);
    if(unlikely(!lsb2)) {
        return -1; // singular
    }

    uint64_t c2_reduc = c2 ^ lsb2;
    if(c0 & lsb2) c0 ^= c2_reduc;
    if(c1 & lsb2) c1 ^= c2_reduc;
    mask ^= lsb2;

    uint64_t lsb1 = uint64_t_lsb(c1 & mask);
    if(unlikely(!lsb1)) {
        return -1; // singular
    }

    if(c0 & lsb1) c0 ^= c1 ^ lsb1;
    mask ^= lsb1;

    if(likely(mask & c0)) { // check system consistency
        return mask & c0; // not solvable
    }

    uint64_t s = 0x0ULL;
    if(c0 & lsb1) s = uint64_t_toggle_at(s, 0);
    if(c0 & lsb2) s = uint64_t_toggle_at(s, 1);
    if(c0 & lsb3) s = uint64_t_toggle_at(s, 2);
    if(c0 & lsb4) s = uint64_t_toggle_at(s, 3);
    if(c0 & lsb5) s = uint64_t_toggle_at(s, 4);
    if(c0 & lsb6) s = uint64_t_toggle_at(s, 5);
    if(c0 & lsb7) s = uint64_t_toggle_at(s, 6);
    if(c0 & lsb8) s = uint64_t_toggle_at(s, 7);
    if(c0 & lsb9) s = uint64_t_toggle_at(s, 8);
    if(c0 & lsb10) s = uint64_t_toggle_at(s, 9);
    if(c0 & lsb11) s = uint64_t_toggle_at(s, 10);
    if(c0 & lsb12) s = uint64_t_toggle_at(s, 11);
    if(c0 & lsb13) s = uint64_t_toggle_at(s, 12);
    if(c0 & lsb14) s = uint64_t_toggle_at(s, 13);
    if(c0 & lsb15) s = uint64_t_toggle_at(s, 14);
    if(c0 & lsb16) s = uint64_t_toggle_at(s, 15);
    *sol = s;
    return 0;
}

int64_t
uint64a_gj_v17_avx(const uint64_t m[18], uint64_t* const restrict sol) {
    __m256i c0_3 = _mm256_loadu_si256((__m256i*) (m + 0));
    __m256i c4_7 = _mm256_loadu_si256((__m256i*) (m + 4));
    __m256i c8_11 = _mm256_loadu_si256((__m256i*) (m + 8));
    __m256i c12_15 = _mm256_loadu_si256((__m256i*) (m + 12));
    uint64_t c16 = m[16];
    uint64_t c17 = m[17];

    uint64_t lsb17 = uint64_t_lsb(c17);
    if(unlikely(!lsb17)) {
        return -1; // singular
    }

    uint64_t c17_reduc = c17 ^ lsb17;
    __m256i reduc = _mm256_set1_epi64x(c17_reduc);
    __m256i vmask = _mm256_set1_epi64x(lsb17);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    if(c16 & lsb17) c16 ^= c17_reduc;
    uint64_t mask = ~lsb17;

    uint64_t lsb16 = uint64_t_lsb(c16 & mask);
    if(unlikely(!lsb16)) {
        return -1; // singular
    }

    uint64_t c16_reduc = c16 ^ lsb16;
    reduc = _mm256_set1_epi64x(c16_reduc);
    vmask = _mm256_set1_epi64x(lsb16);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    uint64_t c12 = _mm256_extract_epi64(c12_15, 0);
    uint64_t c13 = _mm256_extract_epi64(c12_15, 1);
    uint64_t c14 = _mm256_extract_epi64(c12_15, 2);
    uint64_t c15 = _mm256_extract_epi64(c12_15, 3);
    mask ^= lsb16;

    uint64_t lsb15 = uint64_t_lsb(c15 & mask);
    if(unlikely(!lsb15)) {
        return -1; // singular
    }

    uint64_t c15_reduc = c15 ^ lsb15;
    reduc = _mm256_set1_epi64x(c15_reduc);
    vmask = _mm256_set1_epi64x(lsb15);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    if(c12 & lsb15) c12 ^= c15_reduc;
    if(c13 & lsb15) c13 ^= c15_reduc;
    if(c14 & lsb15) c14 ^= c15_reduc;
    mask ^= lsb15;

    uint64_t lsb14 = uint64_t_lsb(c14 & mask);
    if(unlikely(!lsb14)) {
        return -1; // singular
    }

    uint64_t c14_reduc = c14 ^ lsb14;
    reduc = _mm256_set1_epi64x(c14_reduc);
    vmask = _mm256_set1_epi64x(lsb14);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    if(c12 & lsb14) c12 ^= c14_reduc;
    if(c13 & lsb14) c13 ^= c14_reduc;
    mask ^= lsb14;

    uint64_t lsb13 = uint64_t_lsb(c13 & mask);
    if(unlikely(!lsb13)) {
        return -1; // singular
    }

    uint64_t c13_reduc = c13 ^ lsb13;
    reduc = _mm256_set1_epi64x(c13_reduc);
    vmask = _mm256_set1_epi64x(lsb13);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    if(c12 & lsb13) c12 ^= c13_reduc;
    uint64_t c8 = _mm256_extract_epi64(c8_11, 0);
    uint64_t c9 = _mm256_extract_epi64(c8_11, 1);
    uint64_t c10 = _mm256_extract_epi64(c8_11, 2);
    uint64_t c11 = _mm256_extract_epi64(c8_11, 3);
    mask ^= lsb13;

    uint64_t lsb12 = uint64_t_lsb(c12 & mask);
    if(unlikely(!lsb12)) {
        return -1; // singular
    }

    uint64_t c12_reduc = c12 ^ lsb12;
    reduc = _mm256_set1_epi64x(c12_reduc);
    vmask = _mm256_set1_epi64x(lsb12);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb12) c8 ^= c12_reduc;
    if(c9 & lsb12) c9 ^= c12_reduc;
    if(c10 & lsb12) c10 ^= c12_reduc;
    if(c11 & lsb12) c11 ^= c12_reduc;
    mask ^= lsb12;

    uint64_t lsb11 = uint64_t_lsb(c11 & mask);
    if(unlikely(!lsb11)) {
        return -1; // singular
    }

    uint64_t c11_reduc = c11 ^ lsb11;
    reduc = _mm256_set1_epi64x(c11_reduc);
    vmask = _mm256_set1_epi64x(lsb11);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb11) c8 ^= c11_reduc;
    if(c9 & lsb11) c9 ^= c11_reduc;
    if(c10 & lsb11) c10 ^= c11_reduc;
    mask ^= lsb11;

    uint64_t lsb10 = uint64_t_lsb(c10 & mask);
    if(unlikely(!lsb10)) {
        return -1; // singular
    }

    uint64_t c10_reduc = c10 ^ lsb10;
    reduc = _mm256_set1_epi64x(c10_reduc);
    vmask = _mm256_set1_epi64x(lsb10);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb10) c8 ^= c10_reduc;
    if(c9 & lsb10) c9 ^= c10_reduc;
    mask ^= lsb10;

    uint64_t lsb9 = uint64_t_lsb(c9 & mask);
    if(unlikely(!lsb9)) {
        return -1; // singular
    }

    uint64_t c9_reduc = c9 ^ lsb9;
    reduc = _mm256_set1_epi64x(c9_reduc);
    vmask = _mm256_set1_epi64x(lsb9);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb9) c8 ^= c9_reduc;
    uint64_t c4 = _mm256_extract_epi64(c4_7, 0);
    uint64_t c5 = _mm256_extract_epi64(c4_7, 1);
    uint64_t c6 = _mm256_extract_epi64(c4_7, 2);
    uint64_t c7 = _mm256_extract_epi64(c4_7, 3);
    mask ^= lsb9;

    uint64_t lsb8 = uint64_t_lsb(c8 & mask);
    if(unlikely(!lsb8)) {
        return -1; // singular
    }

    uint64_t c8_reduc = c8 ^ lsb8;
    reduc = _mm256_set1_epi64x(c8_reduc);
    vmask = _mm256_set1_epi64x(lsb8);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb8) c4 ^= c8_reduc;
    if(c5 & lsb8) c5 ^= c8_reduc;
    if(c6 & lsb8) c6 ^= c8_reduc;
    if(c7 & lsb8) c7 ^= c8_reduc;
    mask ^= lsb8;

    uint64_t lsb7 = uint64_t_lsb(c7 & mask);
    if(unlikely(!lsb7)) {
        return -1; // singular
    }

    uint64_t c7_reduc = c7 ^ lsb7;
    reduc = _mm256_set1_epi64x(c7_reduc);
    vmask = _mm256_set1_epi64x(lsb7);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb7) c4 ^= c7_reduc;
    if(c5 & lsb7) c5 ^= c7_reduc;
    if(c6 & lsb7) c6 ^= c7_reduc;
    mask ^= lsb7;

    uint64_t lsb6 = uint64_t_lsb(c6 & mask);
    if(unlikely(!lsb6)) {
        return -1; // singular
    }

    uint64_t c6_reduc = c6 ^ lsb6;
    reduc = _mm256_set1_epi64x(c6_reduc);
    vmask = _mm256_set1_epi64x(lsb6);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb6) c4 ^= c6_reduc;
    if(c5 & lsb6) c5 ^= c6_reduc;
    mask ^= lsb6;

    uint64_t lsb5 = uint64_t_lsb(c5 & mask);
    if(unlikely(!lsb5)) {
        return -1; // singular
    }

    uint64_t c5_reduc = c5 ^ lsb5;
    reduc = _mm256_set1_epi64x(c5_reduc);
    vmask = _mm256_set1_epi64x(lsb5);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb5) c4 ^= c5_reduc;
    uint64_t c0 = _mm256_extract_epi64(c0_3, 0);
    uint64_t c1 = _mm256_extract_epi64(c0_3, 1);
    uint64_t c2 = _mm256_extract_epi64(c0_3, 2);
    uint64_t c3 = _mm256_extract_epi64(c0_3, 3);
    mask ^= lsb5;

    uint64_t lsb4 = uint64_t_lsb(c4 & mask);
    if(unlikely(!lsb4)) {
        return -1; // singular
    }

    uint64_t c4_reduc = c4 ^ lsb4;
    if(c0 & lsb4) c0 ^= c4_reduc;
    if(c1 & lsb4) c1 ^= c4_reduc;
    if(c2 & lsb4) c2 ^= c4_reduc;
    if(c3 & lsb4) c3 ^= c4_reduc;
    mask ^= lsb4;

    uint64_t lsb3 = uint64_t_lsb(c3 & mask);
    if(unlikely(!lsb3)) {
        return -1; // singular
    }

    uint64_t c3_reduc = c3 ^ lsb3;
    if(c0 & lsb3) c0 ^= c3_reduc;
    if(c1 & lsb3) c1 ^= c3_reduc;
    if(c2 & lsb3) c2 ^= c3_reduc;
    mask ^= lsb3;

    uint64_t lsb2 = uint64_t_lsb(c2 & mask);
    if(unlikely(!lsb2)) {
        return -1; // singular
    }

    uint64_t c2_reduc = c2 ^ lsb2;
    if(c0 & lsb2) c0 ^= c2_reduc;
    if(c1 & lsb2) c1 ^= c2_reduc;
    mask ^= lsb2;

    uint64_t lsb1 = uint64_t_lsb(c1 & mask);
    if(unlikely(!lsb1)) {
        return -1; // singular
    }

    if(c0 & lsb1) c0 ^= c1 ^ lsb1;
    mask ^= lsb1;

    if(likely(mask & c0)) { // check system consistency
        return mask & c0; // not solvable
    }

    uint64_t s = 0x0ULL;
    if(c0 & lsb1) s = uint64_t_toggle_at(s, 0);
    if(c0 & lsb2) s = uint64_t_toggle_at(s, 1);
    if(c0 & lsb3) s = uint64_t_toggle_at(s, 2);
    if(c0 & lsb4) s = uint64_t_toggle_at(s, 3);
    if(c0 & lsb5) s = uint64_t_toggle_at(s, 4);
    if(c0 & lsb6) s = uint64_t_toggle_at(s, 5);
    if(c0 & lsb7) s = uint64_t_toggle_at(s, 6);
    if(c0 & lsb8) s = uint64_t_toggle_at(s, 7);
    if(c0 & lsb9) s = uint64_t_toggle_at(s, 8);
    if(c0 & lsb10) s = uint64_t_toggle_at(s, 9);
    if(c0 & lsb11) s = uint64_t_toggle_at(s, 10);
    if(c0 & lsb12) s = uint64_t_toggle_at(s, 11);
    if(c0 & lsb13) s = uint64_t_toggle_at(s, 12);
    if(c0 & lsb14) s = uint64_t_toggle_at(s, 13);
    if(c0 & lsb15) s = uint64_t_toggle_at(s, 14);
    if(c0 & lsb16) s = uint64_t_toggle_at(s, 15);
    if(c0 & lsb17) s = uint64_t_toggle_at(s, 16);
    *sol = s;
    return 0;
}

int64_t
uint64a_gj_v18_avx(const uint64_t m[19], uint64_t* const restrict sol) {
    __m256i c0_3 = _mm256_loadu_si256((__m256i*) (m + 0));
    __m256i c4_7 = _mm256_loadu_si256((__m256i*) (m + 4));
    __m256i c8_11 = _mm256_loadu_si256((__m256i*) (m + 8));
    __m256i c12_15 = _mm256_loadu_si256((__m256i*) (m + 12));
    uint64_t c16 = m[16];
    uint64_t c17 = m[17];
    uint64_t c18 = m[18];

    uint64_t lsb18 = uint64_t_lsb(c18);
    if(unlikely(!lsb18)) {
        return -1; // singular
    }

    uint64_t c18_reduc = c18 ^ lsb18;
    __m256i reduc = _mm256_set1_epi64x(c18_reduc);
    __m256i vmask = _mm256_set1_epi64x(lsb18);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    if(c16 & lsb18) c16 ^= c18_reduc;
    if(c17 & lsb18) c17 ^= c18_reduc;
    uint64_t mask = ~lsb18;

    uint64_t lsb17 = uint64_t_lsb(c17 & mask);
    if(unlikely(!lsb17)) {
        return -1; // singular
    }

    uint64_t c17_reduc = c17 ^ lsb17;
    reduc = _mm256_set1_epi64x(c17_reduc);
    vmask = _mm256_set1_epi64x(lsb17);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    if(c16 & lsb17) c16 ^= c17_reduc;
    uint64_t c12 = _mm256_extract_epi64(c12_15, 0);
    uint64_t c13 = _mm256_extract_epi64(c12_15, 1);
    uint64_t c14 = _mm256_extract_epi64(c12_15, 2);
    uint64_t c15 = _mm256_extract_epi64(c12_15, 3);
    mask ^= lsb17;

    uint64_t lsb16 = uint64_t_lsb(c16 & mask);
    if(unlikely(!lsb16)) {
        return -1; // singular
    }

    uint64_t c16_reduc = c16 ^ lsb16;
    reduc = _mm256_set1_epi64x(c16_reduc);
    vmask = _mm256_set1_epi64x(lsb16);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    if(c12 & lsb16) c12 ^= c16_reduc;
    if(c13 & lsb16) c13 ^= c16_reduc;
    if(c14 & lsb16) c14 ^= c16_reduc;
    if(c15 & lsb16) c15 ^= c16_reduc;
    mask ^= lsb16;

    uint64_t lsb15 = uint64_t_lsb(c15 & mask);
    if(unlikely(!lsb15)) {
        return -1; // singular
    }

    uint64_t c15_reduc = c15 ^ lsb15;
    reduc = _mm256_set1_epi64x(c15_reduc);
    vmask = _mm256_set1_epi64x(lsb15);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    if(c12 & lsb15) c12 ^= c15_reduc;
    if(c13 & lsb15) c13 ^= c15_reduc;
    if(c14 & lsb15) c14 ^= c15_reduc;
    mask ^= lsb15;

    uint64_t lsb14 = uint64_t_lsb(c14 & mask);
    if(unlikely(!lsb14)) {
        return -1; // singular
    }

    uint64_t c14_reduc = c14 ^ lsb14;
    reduc = _mm256_set1_epi64x(c14_reduc);
    vmask = _mm256_set1_epi64x(lsb14);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    if(c12 & lsb14) c12 ^= c14_reduc;
    if(c13 & lsb14) c13 ^= c14_reduc;
    mask ^= lsb14;

    uint64_t lsb13 = uint64_t_lsb(c13 & mask);
    if(unlikely(!lsb13)) {
        return -1; // singular
    }

    uint64_t c13_reduc = c13 ^ lsb13;
    reduc = _mm256_set1_epi64x(c13_reduc);
    vmask = _mm256_set1_epi64x(lsb13);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    if(c12 & lsb13) c12 ^= c13_reduc;
    uint64_t c8 = _mm256_extract_epi64(c8_11, 0);
    uint64_t c9 = _mm256_extract_epi64(c8_11, 1);
    uint64_t c10 = _mm256_extract_epi64(c8_11, 2);
    uint64_t c11 = _mm256_extract_epi64(c8_11, 3);
    mask ^= lsb13;

    uint64_t lsb12 = uint64_t_lsb(c12 & mask);
    if(unlikely(!lsb12)) {
        return -1; // singular
    }

    uint64_t c12_reduc = c12 ^ lsb12;
    reduc = _mm256_set1_epi64x(c12_reduc);
    vmask = _mm256_set1_epi64x(lsb12);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb12) c8 ^= c12_reduc;
    if(c9 & lsb12) c9 ^= c12_reduc;
    if(c10 & lsb12) c10 ^= c12_reduc;
    if(c11 & lsb12) c11 ^= c12_reduc;
    mask ^= lsb12;

    uint64_t lsb11 = uint64_t_lsb(c11 & mask);
    if(unlikely(!lsb11)) {
        return -1; // singular
    }

    uint64_t c11_reduc = c11 ^ lsb11;
    reduc = _mm256_set1_epi64x(c11_reduc);
    vmask = _mm256_set1_epi64x(lsb11);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb11) c8 ^= c11_reduc;
    if(c9 & lsb11) c9 ^= c11_reduc;
    if(c10 & lsb11) c10 ^= c11_reduc;
    mask ^= lsb11;

    uint64_t lsb10 = uint64_t_lsb(c10 & mask);
    if(unlikely(!lsb10)) {
        return -1; // singular
    }

    uint64_t c10_reduc = c10 ^ lsb10;
    reduc = _mm256_set1_epi64x(c10_reduc);
    vmask = _mm256_set1_epi64x(lsb10);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb10) c8 ^= c10_reduc;
    if(c9 & lsb10) c9 ^= c10_reduc;
    mask ^= lsb10;

    uint64_t lsb9 = uint64_t_lsb(c9 & mask);
    if(unlikely(!lsb9)) {
        return -1; // singular
    }

    uint64_t c9_reduc = c9 ^ lsb9;
    reduc = _mm256_set1_epi64x(c9_reduc);
    vmask = _mm256_set1_epi64x(lsb9);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb9) c8 ^= c9_reduc;
    uint64_t c4 = _mm256_extract_epi64(c4_7, 0);
    uint64_t c5 = _mm256_extract_epi64(c4_7, 1);
    uint64_t c6 = _mm256_extract_epi64(c4_7, 2);
    uint64_t c7 = _mm256_extract_epi64(c4_7, 3);
    mask ^= lsb9;

    uint64_t lsb8 = uint64_t_lsb(c8 & mask);
    if(unlikely(!lsb8)) {
        return -1; // singular
    }

    uint64_t c8_reduc = c8 ^ lsb8;
    reduc = _mm256_set1_epi64x(c8_reduc);
    vmask = _mm256_set1_epi64x(lsb8);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb8) c4 ^= c8_reduc;
    if(c5 & lsb8) c5 ^= c8_reduc;
    if(c6 & lsb8) c6 ^= c8_reduc;
    if(c7 & lsb8) c7 ^= c8_reduc;
    mask ^= lsb8;

    uint64_t lsb7 = uint64_t_lsb(c7 & mask);
    if(unlikely(!lsb7)) {
        return -1; // singular
    }

    uint64_t c7_reduc = c7 ^ lsb7;
    reduc = _mm256_set1_epi64x(c7_reduc);
    vmask = _mm256_set1_epi64x(lsb7);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb7) c4 ^= c7_reduc;
    if(c5 & lsb7) c5 ^= c7_reduc;
    if(c6 & lsb7) c6 ^= c7_reduc;
    mask ^= lsb7;

    uint64_t lsb6 = uint64_t_lsb(c6 & mask);
    if(unlikely(!lsb6)) {
        return -1; // singular
    }

    uint64_t c6_reduc = c6 ^ lsb6;
    reduc = _mm256_set1_epi64x(c6_reduc);
    vmask = _mm256_set1_epi64x(lsb6);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb6) c4 ^= c6_reduc;
    if(c5 & lsb6) c5 ^= c6_reduc;
    mask ^= lsb6;

    uint64_t lsb5 = uint64_t_lsb(c5 & mask);
    if(unlikely(!lsb5)) {
        return -1; // singular
    }

    uint64_t c5_reduc = c5 ^ lsb5;
    reduc = _mm256_set1_epi64x(c5_reduc);
    vmask = _mm256_set1_epi64x(lsb5);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb5) c4 ^= c5_reduc;
    uint64_t c0 = _mm256_extract_epi64(c0_3, 0);
    uint64_t c1 = _mm256_extract_epi64(c0_3, 1);
    uint64_t c2 = _mm256_extract_epi64(c0_3, 2);
    uint64_t c3 = _mm256_extract_epi64(c0_3, 3);
    mask ^= lsb5;

    uint64_t lsb4 = uint64_t_lsb(c4 & mask);
    if(unlikely(!lsb4)) {
        return -1; // singular
    }

    uint64_t c4_reduc = c4 ^ lsb4;
    if(c0 & lsb4) c0 ^= c4_reduc;
    if(c1 & lsb4) c1 ^= c4_reduc;
    if(c2 & lsb4) c2 ^= c4_reduc;
    if(c3 & lsb4) c3 ^= c4_reduc;
    mask ^= lsb4;

    uint64_t lsb3 = uint64_t_lsb(c3 & mask);
    if(unlikely(!lsb3)) {
        return -1; // singular
    }

    uint64_t c3_reduc = c3 ^ lsb3;
    if(c0 & lsb3) c0 ^= c3_reduc;
    if(c1 & lsb3) c1 ^= c3_reduc;
    if(c2 & lsb3) c2 ^= c3_reduc;
    mask ^= lsb3;

    uint64_t lsb2 = uint64_t_lsb(c2 & mask);
    if(unlikely(!lsb2)) {
        return -1; // singular
    }

    uint64_t c2_reduc = c2 ^ lsb2;
    if(c0 & lsb2) c0 ^= c2_reduc;
    if(c1 & lsb2) c1 ^= c2_reduc;
    mask ^= lsb2;

    uint64_t lsb1 = uint64_t_lsb(c1 & mask);
    if(unlikely(!lsb1)) {
        return -1; // singular
    }

    if(c0 & lsb1) c0 ^= c1 ^ lsb1;
    mask ^= lsb1;

    if(likely(mask & c0)) { // check system consistency
        return mask & c0; // not solvable
    }

    uint64_t s = 0x0ULL;
    if(c0 & lsb1) s = uint64_t_toggle_at(s, 0);
    if(c0 & lsb2) s = uint64_t_toggle_at(s, 1);
    if(c0 & lsb3) s = uint64_t_toggle_at(s, 2);
    if(c0 & lsb4) s = uint64_t_toggle_at(s, 3);
    if(c0 & lsb5) s = uint64_t_toggle_at(s, 4);
    if(c0 & lsb6) s = uint64_t_toggle_at(s, 5);
    if(c0 & lsb7) s = uint64_t_toggle_at(s, 6);
    if(c0 & lsb8) s = uint64_t_toggle_at(s, 7);
    if(c0 & lsb9) s = uint64_t_toggle_at(s, 8);
    if(c0 & lsb10) s = uint64_t_toggle_at(s, 9);
    if(c0 & lsb11) s = uint64_t_toggle_at(s, 10);
    if(c0 & lsb12) s = uint64_t_toggle_at(s, 11);
    if(c0 & lsb13) s = uint64_t_toggle_at(s, 12);
    if(c0 & lsb14) s = uint64_t_toggle_at(s, 13);
    if(c0 & lsb15) s = uint64_t_toggle_at(s, 14);
    if(c0 & lsb16) s = uint64_t_toggle_at(s, 15);
    if(c0 & lsb17) s = uint64_t_toggle_at(s, 16);
    if(c0 & lsb18) s = uint64_t_toggle_at(s, 17);
    *sol = s;
    return 0;
}

int64_t
uint64a_gj_v19_avx(const uint64_t m[20], uint64_t* const restrict sol) {
    __m256i c0_3 = _mm256_loadu_si256((__m256i*) (m + 0));
    __m256i c4_7 = _mm256_loadu_si256((__m256i*) (m + 4));
    __m256i c8_11 = _mm256_loadu_si256((__m256i*) (m + 8));
    __m256i c12_15 = _mm256_loadu_si256((__m256i*) (m + 12));
    uint64_t c16 = m[16];
    uint64_t c17 = m[17];
    uint64_t c18 = m[18];
    uint64_t c19 = m[19];

    uint64_t lsb19 = uint64_t_lsb(c19);
    if(unlikely(!lsb19)) {
        return -1; // singular
    }

    uint64_t c19_reduc = c19 ^ lsb19;
    __m256i reduc = _mm256_set1_epi64x(c19_reduc);
    __m256i vmask = _mm256_set1_epi64x(lsb19);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    if(c16 & lsb19) c16 ^= c19_reduc;
    if(c17 & lsb19) c17 ^= c19_reduc;
    if(c18 & lsb19) c18 ^= c19_reduc;
    uint64_t mask = ~lsb19;

    uint64_t lsb18 = uint64_t_lsb(c18 & mask);
    if(unlikely(!lsb18)) {
        return -1; // singular
    }

    uint64_t c18_reduc = c18 ^ lsb18;
    reduc = _mm256_set1_epi64x(c18_reduc);
    vmask = _mm256_set1_epi64x(lsb18);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    if(c16 & lsb18) c16 ^= c18_reduc;
    if(c17 & lsb18) c17 ^= c18_reduc;
    mask ^= lsb18;

    uint64_t lsb17 = uint64_t_lsb(c17 & mask);
    if(unlikely(!lsb17)) {
        return -1; // singular
    }

    uint64_t c17_reduc = c17 ^ lsb17;
    reduc = _mm256_set1_epi64x(c17_reduc);
    vmask = _mm256_set1_epi64x(lsb17);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    if(c16 & lsb17) c16 ^= c17_reduc;
    uint64_t c12 = _mm256_extract_epi64(c12_15, 0);
    uint64_t c13 = _mm256_extract_epi64(c12_15, 1);
    uint64_t c14 = _mm256_extract_epi64(c12_15, 2);
    uint64_t c15 = _mm256_extract_epi64(c12_15, 3);
    mask ^= lsb17;

    uint64_t lsb16 = uint64_t_lsb(c16 & mask);
    if(unlikely(!lsb16)) {
        return -1; // singular
    }

    uint64_t c16_reduc = c16 ^ lsb16;
    reduc = _mm256_set1_epi64x(c16_reduc);
    vmask = _mm256_set1_epi64x(lsb16);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    if(c12 & lsb16) c12 ^= c16_reduc;
    if(c13 & lsb16) c13 ^= c16_reduc;
    if(c14 & lsb16) c14 ^= c16_reduc;
    if(c15 & lsb16) c15 ^= c16_reduc;
    mask ^= lsb16;

    uint64_t lsb15 = uint64_t_lsb(c15 & mask);
    if(unlikely(!lsb15)) {
        return -1; // singular
    }

    uint64_t c15_reduc = c15 ^ lsb15;
    reduc = _mm256_set1_epi64x(c15_reduc);
    vmask = _mm256_set1_epi64x(lsb15);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    if(c12 & lsb15) c12 ^= c15_reduc;
    if(c13 & lsb15) c13 ^= c15_reduc;
    if(c14 & lsb15) c14 ^= c15_reduc;
    mask ^= lsb15;

    uint64_t lsb14 = uint64_t_lsb(c14 & mask);
    if(unlikely(!lsb14)) {
        return -1; // singular
    }

    uint64_t c14_reduc = c14 ^ lsb14;
    reduc = _mm256_set1_epi64x(c14_reduc);
    vmask = _mm256_set1_epi64x(lsb14);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    if(c12 & lsb14) c12 ^= c14_reduc;
    if(c13 & lsb14) c13 ^= c14_reduc;
    mask ^= lsb14;

    uint64_t lsb13 = uint64_t_lsb(c13 & mask);
    if(unlikely(!lsb13)) {
        return -1; // singular
    }

    uint64_t c13_reduc = c13 ^ lsb13;
    reduc = _mm256_set1_epi64x(c13_reduc);
    vmask = _mm256_set1_epi64x(lsb13);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    if(c12 & lsb13) c12 ^= c13_reduc;
    uint64_t c8 = _mm256_extract_epi64(c8_11, 0);
    uint64_t c9 = _mm256_extract_epi64(c8_11, 1);
    uint64_t c10 = _mm256_extract_epi64(c8_11, 2);
    uint64_t c11 = _mm256_extract_epi64(c8_11, 3);
    mask ^= lsb13;

    uint64_t lsb12 = uint64_t_lsb(c12 & mask);
    if(unlikely(!lsb12)) {
        return -1; // singular
    }

    uint64_t c12_reduc = c12 ^ lsb12;
    reduc = _mm256_set1_epi64x(c12_reduc);
    vmask = _mm256_set1_epi64x(lsb12);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb12) c8 ^= c12_reduc;
    if(c9 & lsb12) c9 ^= c12_reduc;
    if(c10 & lsb12) c10 ^= c12_reduc;
    if(c11 & lsb12) c11 ^= c12_reduc;
    mask ^= lsb12;

    uint64_t lsb11 = uint64_t_lsb(c11 & mask);
    if(unlikely(!lsb11)) {
        return -1; // singular
    }

    uint64_t c11_reduc = c11 ^ lsb11;
    reduc = _mm256_set1_epi64x(c11_reduc);
    vmask = _mm256_set1_epi64x(lsb11);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb11) c8 ^= c11_reduc;
    if(c9 & lsb11) c9 ^= c11_reduc;
    if(c10 & lsb11) c10 ^= c11_reduc;
    mask ^= lsb11;

    uint64_t lsb10 = uint64_t_lsb(c10 & mask);
    if(unlikely(!lsb10)) {
        return -1; // singular
    }

    uint64_t c10_reduc = c10 ^ lsb10;
    reduc = _mm256_set1_epi64x(c10_reduc);
    vmask = _mm256_set1_epi64x(lsb10);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb10) c8 ^= c10_reduc;
    if(c9 & lsb10) c9 ^= c10_reduc;
    mask ^= lsb10;

    uint64_t lsb9 = uint64_t_lsb(c9 & mask);
    if(unlikely(!lsb9)) {
        return -1; // singular
    }

    uint64_t c9_reduc = c9 ^ lsb9;
    reduc = _mm256_set1_epi64x(c9_reduc);
    vmask = _mm256_set1_epi64x(lsb9);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb9) c8 ^= c9_reduc;
    uint64_t c4 = _mm256_extract_epi64(c4_7, 0);
    uint64_t c5 = _mm256_extract_epi64(c4_7, 1);
    uint64_t c6 = _mm256_extract_epi64(c4_7, 2);
    uint64_t c7 = _mm256_extract_epi64(c4_7, 3);
    mask ^= lsb9;

    uint64_t lsb8 = uint64_t_lsb(c8 & mask);
    if(unlikely(!lsb8)) {
        return -1; // singular
    }

    uint64_t c8_reduc = c8 ^ lsb8;
    reduc = _mm256_set1_epi64x(c8_reduc);
    vmask = _mm256_set1_epi64x(lsb8);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb8) c4 ^= c8_reduc;
    if(c5 & lsb8) c5 ^= c8_reduc;
    if(c6 & lsb8) c6 ^= c8_reduc;
    if(c7 & lsb8) c7 ^= c8_reduc;
    mask ^= lsb8;

    uint64_t lsb7 = uint64_t_lsb(c7 & mask);
    if(unlikely(!lsb7)) {
        return -1; // singular
    }

    uint64_t c7_reduc = c7 ^ lsb7;
    reduc = _mm256_set1_epi64x(c7_reduc);
    vmask = _mm256_set1_epi64x(lsb7);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb7) c4 ^= c7_reduc;
    if(c5 & lsb7) c5 ^= c7_reduc;
    if(c6 & lsb7) c6 ^= c7_reduc;
    mask ^= lsb7;

    uint64_t lsb6 = uint64_t_lsb(c6 & mask);
    if(unlikely(!lsb6)) {
        return -1; // singular
    }

    uint64_t c6_reduc = c6 ^ lsb6;
    reduc = _mm256_set1_epi64x(c6_reduc);
    vmask = _mm256_set1_epi64x(lsb6);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb6) c4 ^= c6_reduc;
    if(c5 & lsb6) c5 ^= c6_reduc;
    mask ^= lsb6;

    uint64_t lsb5 = uint64_t_lsb(c5 & mask);
    if(unlikely(!lsb5)) {
        return -1; // singular
    }

    uint64_t c5_reduc = c5 ^ lsb5;
    reduc = _mm256_set1_epi64x(c5_reduc);
    vmask = _mm256_set1_epi64x(lsb5);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb5) c4 ^= c5_reduc;
    uint64_t c0 = _mm256_extract_epi64(c0_3, 0);
    uint64_t c1 = _mm256_extract_epi64(c0_3, 1);
    uint64_t c2 = _mm256_extract_epi64(c0_3, 2);
    uint64_t c3 = _mm256_extract_epi64(c0_3, 3);
    mask ^= lsb5;

    uint64_t lsb4 = uint64_t_lsb(c4 & mask);
    if(unlikely(!lsb4)) {
        return -1; // singular
    }

    uint64_t c4_reduc = c4 ^ lsb4;
    if(c0 & lsb4) c0 ^= c4_reduc;
    if(c1 & lsb4) c1 ^= c4_reduc;
    if(c2 & lsb4) c2 ^= c4_reduc;
    if(c3 & lsb4) c3 ^= c4_reduc;
    mask ^= lsb4;

    uint64_t lsb3 = uint64_t_lsb(c3 & mask);
    if(unlikely(!lsb3)) {
        return -1; // singular
    }

    uint64_t c3_reduc = c3 ^ lsb3;
    if(c0 & lsb3) c0 ^= c3_reduc;
    if(c1 & lsb3) c1 ^= c3_reduc;
    if(c2 & lsb3) c2 ^= c3_reduc;
    mask ^= lsb3;

    uint64_t lsb2 = uint64_t_lsb(c2 & mask);
    if(unlikely(!lsb2)) {
        return -1; // singular
    }

    uint64_t c2_reduc = c2 ^ lsb2;
    if(c0 & lsb2) c0 ^= c2_reduc;
    if(c1 & lsb2) c1 ^= c2_reduc;
    mask ^= lsb2;

    uint64_t lsb1 = uint64_t_lsb(c1 & mask);
    if(unlikely(!lsb1)) {
        return -1; // singular
    }

    if(c0 & lsb1) c0 ^= c1 ^ lsb1;
    mask ^= lsb1;

    if(likely(mask & c0)) { // check system consistency
        return mask & c0; // not solvable
    }

    uint64_t s = 0x0ULL;
    if(c0 & lsb1) s = uint64_t_toggle_at(s, 0);
    if(c0 & lsb2) s = uint64_t_toggle_at(s, 1);
    if(c0 & lsb3) s = uint64_t_toggle_at(s, 2);
    if(c0 & lsb4) s = uint64_t_toggle_at(s, 3);
    if(c0 & lsb5) s = uint64_t_toggle_at(s, 4);
    if(c0 & lsb6) s = uint64_t_toggle_at(s, 5);
    if(c0 & lsb7) s = uint64_t_toggle_at(s, 6);
    if(c0 & lsb8) s = uint64_t_toggle_at(s, 7);
    if(c0 & lsb9) s = uint64_t_toggle_at(s, 8);
    if(c0 & lsb10) s = uint64_t_toggle_at(s, 9);
    if(c0 & lsb11) s = uint64_t_toggle_at(s, 10);
    if(c0 & lsb12) s = uint64_t_toggle_at(s, 11);
    if(c0 & lsb13) s = uint64_t_toggle_at(s, 12);
    if(c0 & lsb14) s = uint64_t_toggle_at(s, 13);
    if(c0 & lsb15) s = uint64_t_toggle_at(s, 14);
    if(c0 & lsb16) s = uint64_t_toggle_at(s, 15);
    if(c0 & lsb17) s = uint64_t_toggle_at(s, 16);
    if(c0 & lsb18) s = uint64_t_toggle_at(s, 17);
    if(c0 & lsb19) s = uint64_t_toggle_at(s, 18);
    *sol = s;
    return 0;
}

int64_t
uint64a_gj_v20_avx(const uint64_t m[21], uint64_t* const restrict sol) {
    __m256i c0_3 = _mm256_loadu_si256((__m256i*) (m + 0));
    __m256i c4_7 = _mm256_loadu_si256((__m256i*) (m + 4));
    __m256i c8_11 = _mm256_loadu_si256((__m256i*) (m + 8));
    __m256i c12_15 = _mm256_loadu_si256((__m256i*) (m + 12));
    uint64_t c16 = m[16];
    uint64_t c17 = m[17];
    uint64_t c18 = m[18];
    uint64_t c19 = m[19];
    uint64_t c20 = m[20];

    uint64_t lsb20 = uint64_t_lsb(c20);
    if(unlikely(!lsb20)) {
        return -1; // singular
    }

    uint64_t c20_reduc = c20 ^ lsb20;
    __m256i reduc = _mm256_set1_epi64x(c20_reduc);
    __m256i vmask = _mm256_set1_epi64x(lsb20);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    if(c16 & lsb20) c16 ^= c20_reduc;
    if(c17 & lsb20) c17 ^= c20_reduc;
    if(c18 & lsb20) c18 ^= c20_reduc;
    if(c19 & lsb20) c19 ^= c20_reduc;
    uint64_t mask = ~lsb20;

    uint64_t lsb19 = uint64_t_lsb(c19 & mask);
    if(unlikely(!lsb19)) {
        return -1; // singular
    }

    uint64_t c19_reduc = c19 ^ lsb19;
    reduc = _mm256_set1_epi64x(c19_reduc);
    vmask = _mm256_set1_epi64x(lsb19);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    if(c16 & lsb19) c16 ^= c19_reduc;
    if(c17 & lsb19) c17 ^= c19_reduc;
    if(c18 & lsb19) c18 ^= c19_reduc;
    mask ^= lsb19;

    uint64_t lsb18 = uint64_t_lsb(c18 & mask);
    if(unlikely(!lsb18)) {
        return -1; // singular
    }

    uint64_t c18_reduc = c18 ^ lsb18;
    reduc = _mm256_set1_epi64x(c18_reduc);
    vmask = _mm256_set1_epi64x(lsb18);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    if(c16 & lsb18) c16 ^= c18_reduc;
    if(c17 & lsb18) c17 ^= c18_reduc;
    mask ^= lsb18;

    uint64_t lsb17 = uint64_t_lsb(c17 & mask);
    if(unlikely(!lsb17)) {
        return -1; // singular
    }

    uint64_t c17_reduc = c17 ^ lsb17;
    reduc = _mm256_set1_epi64x(c17_reduc);
    vmask = _mm256_set1_epi64x(lsb17);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    if(c16 & lsb17) c16 ^= c17_reduc;
    uint64_t c12 = _mm256_extract_epi64(c12_15, 0);
    uint64_t c13 = _mm256_extract_epi64(c12_15, 1);
    uint64_t c14 = _mm256_extract_epi64(c12_15, 2);
    uint64_t c15 = _mm256_extract_epi64(c12_15, 3);
    mask ^= lsb17;

    uint64_t lsb16 = uint64_t_lsb(c16 & mask);
    if(unlikely(!lsb16)) {
        return -1; // singular
    }

    uint64_t c16_reduc = c16 ^ lsb16;
    reduc = _mm256_set1_epi64x(c16_reduc);
    vmask = _mm256_set1_epi64x(lsb16);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    if(c12 & lsb16) c12 ^= c16_reduc;
    if(c13 & lsb16) c13 ^= c16_reduc;
    if(c14 & lsb16) c14 ^= c16_reduc;
    if(c15 & lsb16) c15 ^= c16_reduc;
    mask ^= lsb16;

    uint64_t lsb15 = uint64_t_lsb(c15 & mask);
    if(unlikely(!lsb15)) {
        return -1; // singular
    }

    uint64_t c15_reduc = c15 ^ lsb15;
    reduc = _mm256_set1_epi64x(c15_reduc);
    vmask = _mm256_set1_epi64x(lsb15);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    if(c12 & lsb15) c12 ^= c15_reduc;
    if(c13 & lsb15) c13 ^= c15_reduc;
    if(c14 & lsb15) c14 ^= c15_reduc;
    mask ^= lsb15;

    uint64_t lsb14 = uint64_t_lsb(c14 & mask);
    if(unlikely(!lsb14)) {
        return -1; // singular
    }

    uint64_t c14_reduc = c14 ^ lsb14;
    reduc = _mm256_set1_epi64x(c14_reduc);
    vmask = _mm256_set1_epi64x(lsb14);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    if(c12 & lsb14) c12 ^= c14_reduc;
    if(c13 & lsb14) c13 ^= c14_reduc;
    mask ^= lsb14;

    uint64_t lsb13 = uint64_t_lsb(c13 & mask);
    if(unlikely(!lsb13)) {
        return -1; // singular
    }

    uint64_t c13_reduc = c13 ^ lsb13;
    reduc = _mm256_set1_epi64x(c13_reduc);
    vmask = _mm256_set1_epi64x(lsb13);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    if(c12 & lsb13) c12 ^= c13_reduc;
    uint64_t c8 = _mm256_extract_epi64(c8_11, 0);
    uint64_t c9 = _mm256_extract_epi64(c8_11, 1);
    uint64_t c10 = _mm256_extract_epi64(c8_11, 2);
    uint64_t c11 = _mm256_extract_epi64(c8_11, 3);
    mask ^= lsb13;

    uint64_t lsb12 = uint64_t_lsb(c12 & mask);
    if(unlikely(!lsb12)) {
        return -1; // singular
    }

    uint64_t c12_reduc = c12 ^ lsb12;
    reduc = _mm256_set1_epi64x(c12_reduc);
    vmask = _mm256_set1_epi64x(lsb12);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb12) c8 ^= c12_reduc;
    if(c9 & lsb12) c9 ^= c12_reduc;
    if(c10 & lsb12) c10 ^= c12_reduc;
    if(c11 & lsb12) c11 ^= c12_reduc;
    mask ^= lsb12;

    uint64_t lsb11 = uint64_t_lsb(c11 & mask);
    if(unlikely(!lsb11)) {
        return -1; // singular
    }

    uint64_t c11_reduc = c11 ^ lsb11;
    reduc = _mm256_set1_epi64x(c11_reduc);
    vmask = _mm256_set1_epi64x(lsb11);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb11) c8 ^= c11_reduc;
    if(c9 & lsb11) c9 ^= c11_reduc;
    if(c10 & lsb11) c10 ^= c11_reduc;
    mask ^= lsb11;

    uint64_t lsb10 = uint64_t_lsb(c10 & mask);
    if(unlikely(!lsb10)) {
        return -1; // singular
    }

    uint64_t c10_reduc = c10 ^ lsb10;
    reduc = _mm256_set1_epi64x(c10_reduc);
    vmask = _mm256_set1_epi64x(lsb10);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb10) c8 ^= c10_reduc;
    if(c9 & lsb10) c9 ^= c10_reduc;
    mask ^= lsb10;

    uint64_t lsb9 = uint64_t_lsb(c9 & mask);
    if(unlikely(!lsb9)) {
        return -1; // singular
    }

    uint64_t c9_reduc = c9 ^ lsb9;
    reduc = _mm256_set1_epi64x(c9_reduc);
    vmask = _mm256_set1_epi64x(lsb9);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb9) c8 ^= c9_reduc;
    uint64_t c4 = _mm256_extract_epi64(c4_7, 0);
    uint64_t c5 = _mm256_extract_epi64(c4_7, 1);
    uint64_t c6 = _mm256_extract_epi64(c4_7, 2);
    uint64_t c7 = _mm256_extract_epi64(c4_7, 3);
    mask ^= lsb9;

    uint64_t lsb8 = uint64_t_lsb(c8 & mask);
    if(unlikely(!lsb8)) {
        return -1; // singular
    }

    uint64_t c8_reduc = c8 ^ lsb8;
    reduc = _mm256_set1_epi64x(c8_reduc);
    vmask = _mm256_set1_epi64x(lsb8);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb8) c4 ^= c8_reduc;
    if(c5 & lsb8) c5 ^= c8_reduc;
    if(c6 & lsb8) c6 ^= c8_reduc;
    if(c7 & lsb8) c7 ^= c8_reduc;
    mask ^= lsb8;

    uint64_t lsb7 = uint64_t_lsb(c7 & mask);
    if(unlikely(!lsb7)) {
        return -1; // singular
    }

    uint64_t c7_reduc = c7 ^ lsb7;
    reduc = _mm256_set1_epi64x(c7_reduc);
    vmask = _mm256_set1_epi64x(lsb7);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb7) c4 ^= c7_reduc;
    if(c5 & lsb7) c5 ^= c7_reduc;
    if(c6 & lsb7) c6 ^= c7_reduc;
    mask ^= lsb7;

    uint64_t lsb6 = uint64_t_lsb(c6 & mask);
    if(unlikely(!lsb6)) {
        return -1; // singular
    }

    uint64_t c6_reduc = c6 ^ lsb6;
    reduc = _mm256_set1_epi64x(c6_reduc);
    vmask = _mm256_set1_epi64x(lsb6);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb6) c4 ^= c6_reduc;
    if(c5 & lsb6) c5 ^= c6_reduc;
    mask ^= lsb6;

    uint64_t lsb5 = uint64_t_lsb(c5 & mask);
    if(unlikely(!lsb5)) {
        return -1; // singular
    }

    uint64_t c5_reduc = c5 ^ lsb5;
    reduc = _mm256_set1_epi64x(c5_reduc);
    vmask = _mm256_set1_epi64x(lsb5);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb5) c4 ^= c5_reduc;
    uint64_t c0 = _mm256_extract_epi64(c0_3, 0);
    uint64_t c1 = _mm256_extract_epi64(c0_3, 1);
    uint64_t c2 = _mm256_extract_epi64(c0_3, 2);
    uint64_t c3 = _mm256_extract_epi64(c0_3, 3);
    mask ^= lsb5;

    uint64_t lsb4 = uint64_t_lsb(c4 & mask);
    if(unlikely(!lsb4)) {
        return -1; // singular
    }

    uint64_t c4_reduc = c4 ^ lsb4;
    if(c0 & lsb4) c0 ^= c4_reduc;
    if(c1 & lsb4) c1 ^= c4_reduc;
    if(c2 & lsb4) c2 ^= c4_reduc;
    if(c3 & lsb4) c3 ^= c4_reduc;
    mask ^= lsb4;

    uint64_t lsb3 = uint64_t_lsb(c3 & mask);
    if(unlikely(!lsb3)) {
        return -1; // singular
    }

    uint64_t c3_reduc = c3 ^ lsb3;
    if(c0 & lsb3) c0 ^= c3_reduc;
    if(c1 & lsb3) c1 ^= c3_reduc;
    if(c2 & lsb3) c2 ^= c3_reduc;
    mask ^= lsb3;

    uint64_t lsb2 = uint64_t_lsb(c2 & mask);
    if(unlikely(!lsb2)) {
        return -1; // singular
    }

    uint64_t c2_reduc = c2 ^ lsb2;
    if(c0 & lsb2) c0 ^= c2_reduc;
    if(c1 & lsb2) c1 ^= c2_reduc;
    mask ^= lsb2;

    uint64_t lsb1 = uint64_t_lsb(c1 & mask);
    if(unlikely(!lsb1)) {
        return -1; // singular
    }

    if(c0 & lsb1) c0 ^= c1 ^ lsb1;
    mask ^= lsb1;

    if(likely(mask & c0)) { // check system consistency
        return mask & c0; // not solvable
    }

    uint64_t s = 0x0ULL;
    if(c0 & lsb1) s = uint64_t_toggle_at(s, 0);
    if(c0 & lsb2) s = uint64_t_toggle_at(s, 1);
    if(c0 & lsb3) s = uint64_t_toggle_at(s, 2);
    if(c0 & lsb4) s = uint64_t_toggle_at(s, 3);
    if(c0 & lsb5) s = uint64_t_toggle_at(s, 4);
    if(c0 & lsb6) s = uint64_t_toggle_at(s, 5);
    if(c0 & lsb7) s = uint64_t_toggle_at(s, 6);
    if(c0 & lsb8) s = uint64_t_toggle_at(s, 7);
    if(c0 & lsb9) s = uint64_t_toggle_at(s, 8);
    if(c0 & lsb10) s = uint64_t_toggle_at(s, 9);
    if(c0 & lsb11) s = uint64_t_toggle_at(s, 10);
    if(c0 & lsb12) s = uint64_t_toggle_at(s, 11);
    if(c0 & lsb13) s = uint64_t_toggle_at(s, 12);
    if(c0 & lsb14) s = uint64_t_toggle_at(s, 13);
    if(c0 & lsb15) s = uint64_t_toggle_at(s, 14);
    if(c0 & lsb16) s = uint64_t_toggle_at(s, 15);
    if(c0 & lsb17) s = uint64_t_toggle_at(s, 16);
    if(c0 & lsb18) s = uint64_t_toggle_at(s, 17);
    if(c0 & lsb19) s = uint64_t_toggle_at(s, 18);
    if(c0 & lsb20) s = uint64_t_toggle_at(s, 19);
    *sol = s;
    return 0;
}

int64_t
uint64a_gj_v21_avx(const uint64_t m[22], uint64_t* const restrict sol) {
    __m256i c0_3 = _mm256_loadu_si256((__m256i*) (m + 0));
    __m256i c4_7 = _mm256_loadu_si256((__m256i*) (m + 4));
    __m256i c8_11 = _mm256_loadu_si256((__m256i*) (m + 8));
    __m256i c12_15 = _mm256_loadu_si256((__m256i*) (m + 12));
    __m256i c16_19 = _mm256_loadu_si256((__m256i*) (m + 16));
    uint64_t c20 = m[20];
    uint64_t c21 = m[21];

    uint64_t lsb21 = uint64_t_lsb(c21);
    if(unlikely(!lsb21)) {
        return -1; // singular
    }

    uint64_t c21_reduc = c21 ^ lsb21;
    __m256i reduc = _mm256_set1_epi64x(c21_reduc);
    __m256i vmask = _mm256_set1_epi64x(lsb21);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    if(c20 & lsb21) c20 ^= c21_reduc;
    uint64_t mask = ~lsb21;

    uint64_t lsb20 = uint64_t_lsb(c20 & mask);
    if(unlikely(!lsb20)) {
        return -1; // singular
    }

    uint64_t c20_reduc = c20 ^ lsb20;
    reduc = _mm256_set1_epi64x(c20_reduc);
    vmask = _mm256_set1_epi64x(lsb20);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    uint64_t c16 = _mm256_extract_epi64(c16_19, 0);
    uint64_t c17 = _mm256_extract_epi64(c16_19, 1);
    uint64_t c18 = _mm256_extract_epi64(c16_19, 2);
    uint64_t c19 = _mm256_extract_epi64(c16_19, 3);
    mask ^= lsb20;

    uint64_t lsb19 = uint64_t_lsb(c19 & mask);
    if(unlikely(!lsb19)) {
        return -1; // singular
    }

    uint64_t c19_reduc = c19 ^ lsb19;
    reduc = _mm256_set1_epi64x(c19_reduc);
    vmask = _mm256_set1_epi64x(lsb19);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    if(c16 & lsb19) c16 ^= c19_reduc;
    if(c17 & lsb19) c17 ^= c19_reduc;
    if(c18 & lsb19) c18 ^= c19_reduc;
    mask ^= lsb19;

    uint64_t lsb18 = uint64_t_lsb(c18 & mask);
    if(unlikely(!lsb18)) {
        return -1; // singular
    }

    uint64_t c18_reduc = c18 ^ lsb18;
    reduc = _mm256_set1_epi64x(c18_reduc);
    vmask = _mm256_set1_epi64x(lsb18);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    if(c16 & lsb18) c16 ^= c18_reduc;
    if(c17 & lsb18) c17 ^= c18_reduc;
    mask ^= lsb18;

    uint64_t lsb17 = uint64_t_lsb(c17 & mask);
    if(unlikely(!lsb17)) {
        return -1; // singular
    }

    uint64_t c17_reduc = c17 ^ lsb17;
    reduc = _mm256_set1_epi64x(c17_reduc);
    vmask = _mm256_set1_epi64x(lsb17);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    if(c16 & lsb17) c16 ^= c17_reduc;
    uint64_t c12 = _mm256_extract_epi64(c12_15, 0);
    uint64_t c13 = _mm256_extract_epi64(c12_15, 1);
    uint64_t c14 = _mm256_extract_epi64(c12_15, 2);
    uint64_t c15 = _mm256_extract_epi64(c12_15, 3);
    mask ^= lsb17;

    uint64_t lsb16 = uint64_t_lsb(c16 & mask);
    if(unlikely(!lsb16)) {
        return -1; // singular
    }

    uint64_t c16_reduc = c16 ^ lsb16;
    reduc = _mm256_set1_epi64x(c16_reduc);
    vmask = _mm256_set1_epi64x(lsb16);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    if(c12 & lsb16) c12 ^= c16_reduc;
    if(c13 & lsb16) c13 ^= c16_reduc;
    if(c14 & lsb16) c14 ^= c16_reduc;
    if(c15 & lsb16) c15 ^= c16_reduc;
    mask ^= lsb16;

    uint64_t lsb15 = uint64_t_lsb(c15 & mask);
    if(unlikely(!lsb15)) {
        return -1; // singular
    }

    uint64_t c15_reduc = c15 ^ lsb15;
    reduc = _mm256_set1_epi64x(c15_reduc);
    vmask = _mm256_set1_epi64x(lsb15);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    if(c12 & lsb15) c12 ^= c15_reduc;
    if(c13 & lsb15) c13 ^= c15_reduc;
    if(c14 & lsb15) c14 ^= c15_reduc;
    mask ^= lsb15;

    uint64_t lsb14 = uint64_t_lsb(c14 & mask);
    if(unlikely(!lsb14)) {
        return -1; // singular
    }

    uint64_t c14_reduc = c14 ^ lsb14;
    reduc = _mm256_set1_epi64x(c14_reduc);
    vmask = _mm256_set1_epi64x(lsb14);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    if(c12 & lsb14) c12 ^= c14_reduc;
    if(c13 & lsb14) c13 ^= c14_reduc;
    mask ^= lsb14;

    uint64_t lsb13 = uint64_t_lsb(c13 & mask);
    if(unlikely(!lsb13)) {
        return -1; // singular
    }

    uint64_t c13_reduc = c13 ^ lsb13;
    reduc = _mm256_set1_epi64x(c13_reduc);
    vmask = _mm256_set1_epi64x(lsb13);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    if(c12 & lsb13) c12 ^= c13_reduc;
    uint64_t c8 = _mm256_extract_epi64(c8_11, 0);
    uint64_t c9 = _mm256_extract_epi64(c8_11, 1);
    uint64_t c10 = _mm256_extract_epi64(c8_11, 2);
    uint64_t c11 = _mm256_extract_epi64(c8_11, 3);
    mask ^= lsb13;

    uint64_t lsb12 = uint64_t_lsb(c12 & mask);
    if(unlikely(!lsb12)) {
        return -1; // singular
    }

    uint64_t c12_reduc = c12 ^ lsb12;
    reduc = _mm256_set1_epi64x(c12_reduc);
    vmask = _mm256_set1_epi64x(lsb12);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb12) c8 ^= c12_reduc;
    if(c9 & lsb12) c9 ^= c12_reduc;
    if(c10 & lsb12) c10 ^= c12_reduc;
    if(c11 & lsb12) c11 ^= c12_reduc;
    mask ^= lsb12;

    uint64_t lsb11 = uint64_t_lsb(c11 & mask);
    if(unlikely(!lsb11)) {
        return -1; // singular
    }

    uint64_t c11_reduc = c11 ^ lsb11;
    reduc = _mm256_set1_epi64x(c11_reduc);
    vmask = _mm256_set1_epi64x(lsb11);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb11) c8 ^= c11_reduc;
    if(c9 & lsb11) c9 ^= c11_reduc;
    if(c10 & lsb11) c10 ^= c11_reduc;
    mask ^= lsb11;

    uint64_t lsb10 = uint64_t_lsb(c10 & mask);
    if(unlikely(!lsb10)) {
        return -1; // singular
    }

    uint64_t c10_reduc = c10 ^ lsb10;
    reduc = _mm256_set1_epi64x(c10_reduc);
    vmask = _mm256_set1_epi64x(lsb10);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb10) c8 ^= c10_reduc;
    if(c9 & lsb10) c9 ^= c10_reduc;
    mask ^= lsb10;

    uint64_t lsb9 = uint64_t_lsb(c9 & mask);
    if(unlikely(!lsb9)) {
        return -1; // singular
    }

    uint64_t c9_reduc = c9 ^ lsb9;
    reduc = _mm256_set1_epi64x(c9_reduc);
    vmask = _mm256_set1_epi64x(lsb9);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb9) c8 ^= c9_reduc;
    uint64_t c4 = _mm256_extract_epi64(c4_7, 0);
    uint64_t c5 = _mm256_extract_epi64(c4_7, 1);
    uint64_t c6 = _mm256_extract_epi64(c4_7, 2);
    uint64_t c7 = _mm256_extract_epi64(c4_7, 3);
    mask ^= lsb9;

    uint64_t lsb8 = uint64_t_lsb(c8 & mask);
    if(unlikely(!lsb8)) {
        return -1; // singular
    }

    uint64_t c8_reduc = c8 ^ lsb8;
    reduc = _mm256_set1_epi64x(c8_reduc);
    vmask = _mm256_set1_epi64x(lsb8);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb8) c4 ^= c8_reduc;
    if(c5 & lsb8) c5 ^= c8_reduc;
    if(c6 & lsb8) c6 ^= c8_reduc;
    if(c7 & lsb8) c7 ^= c8_reduc;
    mask ^= lsb8;

    uint64_t lsb7 = uint64_t_lsb(c7 & mask);
    if(unlikely(!lsb7)) {
        return -1; // singular
    }

    uint64_t c7_reduc = c7 ^ lsb7;
    reduc = _mm256_set1_epi64x(c7_reduc);
    vmask = _mm256_set1_epi64x(lsb7);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb7) c4 ^= c7_reduc;
    if(c5 & lsb7) c5 ^= c7_reduc;
    if(c6 & lsb7) c6 ^= c7_reduc;
    mask ^= lsb7;

    uint64_t lsb6 = uint64_t_lsb(c6 & mask);
    if(unlikely(!lsb6)) {
        return -1; // singular
    }

    uint64_t c6_reduc = c6 ^ lsb6;
    reduc = _mm256_set1_epi64x(c6_reduc);
    vmask = _mm256_set1_epi64x(lsb6);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb6) c4 ^= c6_reduc;
    if(c5 & lsb6) c5 ^= c6_reduc;
    mask ^= lsb6;

    uint64_t lsb5 = uint64_t_lsb(c5 & mask);
    if(unlikely(!lsb5)) {
        return -1; // singular
    }

    uint64_t c5_reduc = c5 ^ lsb5;
    reduc = _mm256_set1_epi64x(c5_reduc);
    vmask = _mm256_set1_epi64x(lsb5);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb5) c4 ^= c5_reduc;
    uint64_t c0 = _mm256_extract_epi64(c0_3, 0);
    uint64_t c1 = _mm256_extract_epi64(c0_3, 1);
    uint64_t c2 = _mm256_extract_epi64(c0_3, 2);
    uint64_t c3 = _mm256_extract_epi64(c0_3, 3);
    mask ^= lsb5;

    uint64_t lsb4 = uint64_t_lsb(c4 & mask);
    if(unlikely(!lsb4)) {
        return -1; // singular
    }

    uint64_t c4_reduc = c4 ^ lsb4;
    if(c0 & lsb4) c0 ^= c4_reduc;
    if(c1 & lsb4) c1 ^= c4_reduc;
    if(c2 & lsb4) c2 ^= c4_reduc;
    if(c3 & lsb4) c3 ^= c4_reduc;
    mask ^= lsb4;

    uint64_t lsb3 = uint64_t_lsb(c3 & mask);
    if(unlikely(!lsb3)) {
        return -1; // singular
    }

    uint64_t c3_reduc = c3 ^ lsb3;
    if(c0 & lsb3) c0 ^= c3_reduc;
    if(c1 & lsb3) c1 ^= c3_reduc;
    if(c2 & lsb3) c2 ^= c3_reduc;
    mask ^= lsb3;

    uint64_t lsb2 = uint64_t_lsb(c2 & mask);
    if(unlikely(!lsb2)) {
        return -1; // singular
    }

    uint64_t c2_reduc = c2 ^ lsb2;
    if(c0 & lsb2) c0 ^= c2_reduc;
    if(c1 & lsb2) c1 ^= c2_reduc;
    mask ^= lsb2;

    uint64_t lsb1 = uint64_t_lsb(c1 & mask);
    if(unlikely(!lsb1)) {
        return -1; // singular
    }

    if(c0 & lsb1) c0 ^= c1 ^ lsb1;
    mask ^= lsb1;

    if(likely(mask & c0)) { // check system consistency
        return mask & c0; // not solvable
    }

    uint64_t s = 0x0ULL;
    if(c0 & lsb1) s = uint64_t_toggle_at(s, 0);
    if(c0 & lsb2) s = uint64_t_toggle_at(s, 1);
    if(c0 & lsb3) s = uint64_t_toggle_at(s, 2);
    if(c0 & lsb4) s = uint64_t_toggle_at(s, 3);
    if(c0 & lsb5) s = uint64_t_toggle_at(s, 4);
    if(c0 & lsb6) s = uint64_t_toggle_at(s, 5);
    if(c0 & lsb7) s = uint64_t_toggle_at(s, 6);
    if(c0 & lsb8) s = uint64_t_toggle_at(s, 7);
    if(c0 & lsb9) s = uint64_t_toggle_at(s, 8);
    if(c0 & lsb10) s = uint64_t_toggle_at(s, 9);
    if(c0 & lsb11) s = uint64_t_toggle_at(s, 10);
    if(c0 & lsb12) s = uint64_t_toggle_at(s, 11);
    if(c0 & lsb13) s = uint64_t_toggle_at(s, 12);
    if(c0 & lsb14) s = uint64_t_toggle_at(s, 13);
    if(c0 & lsb15) s = uint64_t_toggle_at(s, 14);
    if(c0 & lsb16) s = uint64_t_toggle_at(s, 15);
    if(c0 & lsb17) s = uint64_t_toggle_at(s, 16);
    if(c0 & lsb18) s = uint64_t_toggle_at(s, 17);
    if(c0 & lsb19) s = uint64_t_toggle_at(s, 18);
    if(c0 & lsb20) s = uint64_t_toggle_at(s, 19);
    if(c0 & lsb21) s = uint64_t_toggle_at(s, 20);
    *sol = s;
    return 0;
}

int64_t
uint64a_gj_v22_avx(const uint64_t m[23], uint64_t* const restrict sol) {
    __m256i c0_3 = _mm256_loadu_si256((__m256i*) (m + 0));
    __m256i c4_7 = _mm256_loadu_si256((__m256i*) (m + 4));
    __m256i c8_11 = _mm256_loadu_si256((__m256i*) (m + 8));
    __m256i c12_15 = _mm256_loadu_si256((__m256i*) (m + 12));
    __m256i c16_19 = _mm256_loadu_si256((__m256i*) (m + 16));
    uint64_t c20 = m[20];
    uint64_t c21 = m[21];
    uint64_t c22 = m[22];

    uint64_t lsb22 = uint64_t_lsb(c22);
    if(unlikely(!lsb22)) {
        return -1; // singular
    }

    uint64_t c22_reduc = c22 ^ lsb22;
    __m256i reduc = _mm256_set1_epi64x(c22_reduc);
    __m256i vmask = _mm256_set1_epi64x(lsb22);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    if(c20 & lsb22) c20 ^= c22_reduc;
    if(c21 & lsb22) c21 ^= c22_reduc;
    uint64_t mask = ~lsb22;

    uint64_t lsb21 = uint64_t_lsb(c21 & mask);
    if(unlikely(!lsb21)) {
        return -1; // singular
    }

    uint64_t c21_reduc = c21 ^ lsb21;
    reduc = _mm256_set1_epi64x(c21_reduc);
    vmask = _mm256_set1_epi64x(lsb21);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    if(c20 & lsb21) c20 ^= c21_reduc;
    uint64_t c16 = _mm256_extract_epi64(c16_19, 0);
    uint64_t c17 = _mm256_extract_epi64(c16_19, 1);
    uint64_t c18 = _mm256_extract_epi64(c16_19, 2);
    uint64_t c19 = _mm256_extract_epi64(c16_19, 3);
    mask ^= lsb21;

    uint64_t lsb20 = uint64_t_lsb(c20 & mask);
    if(unlikely(!lsb20)) {
        return -1; // singular
    }

    uint64_t c20_reduc = c20 ^ lsb20;
    reduc = _mm256_set1_epi64x(c20_reduc);
    vmask = _mm256_set1_epi64x(lsb20);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    if(c16 & lsb20) c16 ^= c20_reduc;
    if(c17 & lsb20) c17 ^= c20_reduc;
    if(c18 & lsb20) c18 ^= c20_reduc;
    if(c19 & lsb20) c19 ^= c20_reduc;
    mask ^= lsb20;

    uint64_t lsb19 = uint64_t_lsb(c19 & mask);
    if(unlikely(!lsb19)) {
        return -1; // singular
    }

    uint64_t c19_reduc = c19 ^ lsb19;
    reduc = _mm256_set1_epi64x(c19_reduc);
    vmask = _mm256_set1_epi64x(lsb19);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    if(c16 & lsb19) c16 ^= c19_reduc;
    if(c17 & lsb19) c17 ^= c19_reduc;
    if(c18 & lsb19) c18 ^= c19_reduc;
    mask ^= lsb19;

    uint64_t lsb18 = uint64_t_lsb(c18 & mask);
    if(unlikely(!lsb18)) {
        return -1; // singular
    }

    uint64_t c18_reduc = c18 ^ lsb18;
    reduc = _mm256_set1_epi64x(c18_reduc);
    vmask = _mm256_set1_epi64x(lsb18);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    if(c16 & lsb18) c16 ^= c18_reduc;
    if(c17 & lsb18) c17 ^= c18_reduc;
    mask ^= lsb18;

    uint64_t lsb17 = uint64_t_lsb(c17 & mask);
    if(unlikely(!lsb17)) {
        return -1; // singular
    }

    uint64_t c17_reduc = c17 ^ lsb17;
    reduc = _mm256_set1_epi64x(c17_reduc);
    vmask = _mm256_set1_epi64x(lsb17);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    if(c16 & lsb17) c16 ^= c17_reduc;
    uint64_t c12 = _mm256_extract_epi64(c12_15, 0);
    uint64_t c13 = _mm256_extract_epi64(c12_15, 1);
    uint64_t c14 = _mm256_extract_epi64(c12_15, 2);
    uint64_t c15 = _mm256_extract_epi64(c12_15, 3);
    mask ^= lsb17;

    uint64_t lsb16 = uint64_t_lsb(c16 & mask);
    if(unlikely(!lsb16)) {
        return -1; // singular
    }

    uint64_t c16_reduc = c16 ^ lsb16;
    reduc = _mm256_set1_epi64x(c16_reduc);
    vmask = _mm256_set1_epi64x(lsb16);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    if(c12 & lsb16) c12 ^= c16_reduc;
    if(c13 & lsb16) c13 ^= c16_reduc;
    if(c14 & lsb16) c14 ^= c16_reduc;
    if(c15 & lsb16) c15 ^= c16_reduc;
    mask ^= lsb16;

    uint64_t lsb15 = uint64_t_lsb(c15 & mask);
    if(unlikely(!lsb15)) {
        return -1; // singular
    }

    uint64_t c15_reduc = c15 ^ lsb15;
    reduc = _mm256_set1_epi64x(c15_reduc);
    vmask = _mm256_set1_epi64x(lsb15);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    if(c12 & lsb15) c12 ^= c15_reduc;
    if(c13 & lsb15) c13 ^= c15_reduc;
    if(c14 & lsb15) c14 ^= c15_reduc;
    mask ^= lsb15;

    uint64_t lsb14 = uint64_t_lsb(c14 & mask);
    if(unlikely(!lsb14)) {
        return -1; // singular
    }

    uint64_t c14_reduc = c14 ^ lsb14;
    reduc = _mm256_set1_epi64x(c14_reduc);
    vmask = _mm256_set1_epi64x(lsb14);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    if(c12 & lsb14) c12 ^= c14_reduc;
    if(c13 & lsb14) c13 ^= c14_reduc;
    mask ^= lsb14;

    uint64_t lsb13 = uint64_t_lsb(c13 & mask);
    if(unlikely(!lsb13)) {
        return -1; // singular
    }

    uint64_t c13_reduc = c13 ^ lsb13;
    reduc = _mm256_set1_epi64x(c13_reduc);
    vmask = _mm256_set1_epi64x(lsb13);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    if(c12 & lsb13) c12 ^= c13_reduc;
    uint64_t c8 = _mm256_extract_epi64(c8_11, 0);
    uint64_t c9 = _mm256_extract_epi64(c8_11, 1);
    uint64_t c10 = _mm256_extract_epi64(c8_11, 2);
    uint64_t c11 = _mm256_extract_epi64(c8_11, 3);
    mask ^= lsb13;

    uint64_t lsb12 = uint64_t_lsb(c12 & mask);
    if(unlikely(!lsb12)) {
        return -1; // singular
    }

    uint64_t c12_reduc = c12 ^ lsb12;
    reduc = _mm256_set1_epi64x(c12_reduc);
    vmask = _mm256_set1_epi64x(lsb12);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb12) c8 ^= c12_reduc;
    if(c9 & lsb12) c9 ^= c12_reduc;
    if(c10 & lsb12) c10 ^= c12_reduc;
    if(c11 & lsb12) c11 ^= c12_reduc;
    mask ^= lsb12;

    uint64_t lsb11 = uint64_t_lsb(c11 & mask);
    if(unlikely(!lsb11)) {
        return -1; // singular
    }

    uint64_t c11_reduc = c11 ^ lsb11;
    reduc = _mm256_set1_epi64x(c11_reduc);
    vmask = _mm256_set1_epi64x(lsb11);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb11) c8 ^= c11_reduc;
    if(c9 & lsb11) c9 ^= c11_reduc;
    if(c10 & lsb11) c10 ^= c11_reduc;
    mask ^= lsb11;

    uint64_t lsb10 = uint64_t_lsb(c10 & mask);
    if(unlikely(!lsb10)) {
        return -1; // singular
    }

    uint64_t c10_reduc = c10 ^ lsb10;
    reduc = _mm256_set1_epi64x(c10_reduc);
    vmask = _mm256_set1_epi64x(lsb10);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb10) c8 ^= c10_reduc;
    if(c9 & lsb10) c9 ^= c10_reduc;
    mask ^= lsb10;

    uint64_t lsb9 = uint64_t_lsb(c9 & mask);
    if(unlikely(!lsb9)) {
        return -1; // singular
    }

    uint64_t c9_reduc = c9 ^ lsb9;
    reduc = _mm256_set1_epi64x(c9_reduc);
    vmask = _mm256_set1_epi64x(lsb9);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb9) c8 ^= c9_reduc;
    uint64_t c4 = _mm256_extract_epi64(c4_7, 0);
    uint64_t c5 = _mm256_extract_epi64(c4_7, 1);
    uint64_t c6 = _mm256_extract_epi64(c4_7, 2);
    uint64_t c7 = _mm256_extract_epi64(c4_7, 3);
    mask ^= lsb9;

    uint64_t lsb8 = uint64_t_lsb(c8 & mask);
    if(unlikely(!lsb8)) {
        return -1; // singular
    }

    uint64_t c8_reduc = c8 ^ lsb8;
    reduc = _mm256_set1_epi64x(c8_reduc);
    vmask = _mm256_set1_epi64x(lsb8);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb8) c4 ^= c8_reduc;
    if(c5 & lsb8) c5 ^= c8_reduc;
    if(c6 & lsb8) c6 ^= c8_reduc;
    if(c7 & lsb8) c7 ^= c8_reduc;
    mask ^= lsb8;

    uint64_t lsb7 = uint64_t_lsb(c7 & mask);
    if(unlikely(!lsb7)) {
        return -1; // singular
    }

    uint64_t c7_reduc = c7 ^ lsb7;
    reduc = _mm256_set1_epi64x(c7_reduc);
    vmask = _mm256_set1_epi64x(lsb7);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb7) c4 ^= c7_reduc;
    if(c5 & lsb7) c5 ^= c7_reduc;
    if(c6 & lsb7) c6 ^= c7_reduc;
    mask ^= lsb7;

    uint64_t lsb6 = uint64_t_lsb(c6 & mask);
    if(unlikely(!lsb6)) {
        return -1; // singular
    }

    uint64_t c6_reduc = c6 ^ lsb6;
    reduc = _mm256_set1_epi64x(c6_reduc);
    vmask = _mm256_set1_epi64x(lsb6);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb6) c4 ^= c6_reduc;
    if(c5 & lsb6) c5 ^= c6_reduc;
    mask ^= lsb6;

    uint64_t lsb5 = uint64_t_lsb(c5 & mask);
    if(unlikely(!lsb5)) {
        return -1; // singular
    }

    uint64_t c5_reduc = c5 ^ lsb5;
    reduc = _mm256_set1_epi64x(c5_reduc);
    vmask = _mm256_set1_epi64x(lsb5);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb5) c4 ^= c5_reduc;
    uint64_t c0 = _mm256_extract_epi64(c0_3, 0);
    uint64_t c1 = _mm256_extract_epi64(c0_3, 1);
    uint64_t c2 = _mm256_extract_epi64(c0_3, 2);
    uint64_t c3 = _mm256_extract_epi64(c0_3, 3);
    mask ^= lsb5;

    uint64_t lsb4 = uint64_t_lsb(c4 & mask);
    if(unlikely(!lsb4)) {
        return -1; // singular
    }

    uint64_t c4_reduc = c4 ^ lsb4;
    if(c0 & lsb4) c0 ^= c4_reduc;
    if(c1 & lsb4) c1 ^= c4_reduc;
    if(c2 & lsb4) c2 ^= c4_reduc;
    if(c3 & lsb4) c3 ^= c4_reduc;
    mask ^= lsb4;

    uint64_t lsb3 = uint64_t_lsb(c3 & mask);
    if(unlikely(!lsb3)) {
        return -1; // singular
    }

    uint64_t c3_reduc = c3 ^ lsb3;
    if(c0 & lsb3) c0 ^= c3_reduc;
    if(c1 & lsb3) c1 ^= c3_reduc;
    if(c2 & lsb3) c2 ^= c3_reduc;
    mask ^= lsb3;

    uint64_t lsb2 = uint64_t_lsb(c2 & mask);
    if(unlikely(!lsb2)) {
        return -1; // singular
    }

    uint64_t c2_reduc = c2 ^ lsb2;
    if(c0 & lsb2) c0 ^= c2_reduc;
    if(c1 & lsb2) c1 ^= c2_reduc;
    mask ^= lsb2;

    uint64_t lsb1 = uint64_t_lsb(c1 & mask);
    if(unlikely(!lsb1)) {
        return -1; // singular
    }

    if(c0 & lsb1) c0 ^= c1 ^ lsb1;
    mask ^= lsb1;

    if(likely(mask & c0)) { // check system consistency
        return mask & c0; // not solvable
    }

    uint64_t s = 0x0ULL;
    if(c0 & lsb1) s = uint64_t_toggle_at(s, 0);
    if(c0 & lsb2) s = uint64_t_toggle_at(s, 1);
    if(c0 & lsb3) s = uint64_t_toggle_at(s, 2);
    if(c0 & lsb4) s = uint64_t_toggle_at(s, 3);
    if(c0 & lsb5) s = uint64_t_toggle_at(s, 4);
    if(c0 & lsb6) s = uint64_t_toggle_at(s, 5);
    if(c0 & lsb7) s = uint64_t_toggle_at(s, 6);
    if(c0 & lsb8) s = uint64_t_toggle_at(s, 7);
    if(c0 & lsb9) s = uint64_t_toggle_at(s, 8);
    if(c0 & lsb10) s = uint64_t_toggle_at(s, 9);
    if(c0 & lsb11) s = uint64_t_toggle_at(s, 10);
    if(c0 & lsb12) s = uint64_t_toggle_at(s, 11);
    if(c0 & lsb13) s = uint64_t_toggle_at(s, 12);
    if(c0 & lsb14) s = uint64_t_toggle_at(s, 13);
    if(c0 & lsb15) s = uint64_t_toggle_at(s, 14);
    if(c0 & lsb16) s = uint64_t_toggle_at(s, 15);
    if(c0 & lsb17) s = uint64_t_toggle_at(s, 16);
    if(c0 & lsb18) s = uint64_t_toggle_at(s, 17);
    if(c0 & lsb19) s = uint64_t_toggle_at(s, 18);
    if(c0 & lsb20) s = uint64_t_toggle_at(s, 19);
    if(c0 & lsb21) s = uint64_t_toggle_at(s, 20);
    if(c0 & lsb22) s = uint64_t_toggle_at(s, 21);
    *sol = s;
    return 0;
}

int64_t
uint64a_gj_v23_avx(const uint64_t m[24], uint64_t* const restrict sol) {
    __m256i c0_3 = _mm256_loadu_si256((__m256i*) (m + 0));
    __m256i c4_7 = _mm256_loadu_si256((__m256i*) (m + 4));
    __m256i c8_11 = _mm256_loadu_si256((__m256i*) (m + 8));
    __m256i c12_15 = _mm256_loadu_si256((__m256i*) (m + 12));
    __m256i c16_19 = _mm256_loadu_si256((__m256i*) (m + 16));
    uint64_t c20 = m[20];
    uint64_t c21 = m[21];
    uint64_t c22 = m[22];
    uint64_t c23 = m[23];

    uint64_t lsb23 = uint64_t_lsb(c23);
    if(unlikely(!lsb23)) {
        return -1; // singular
    }

    uint64_t c23_reduc = c23 ^ lsb23;
    __m256i reduc = _mm256_set1_epi64x(c23_reduc);
    __m256i vmask = _mm256_set1_epi64x(lsb23);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    if(c20 & lsb23) c20 ^= c23_reduc;
    if(c21 & lsb23) c21 ^= c23_reduc;
    if(c22 & lsb23) c22 ^= c23_reduc;
    uint64_t mask = ~lsb23;

    uint64_t lsb22 = uint64_t_lsb(c22 & mask);
    if(unlikely(!lsb22)) {
        return -1; // singular
    }

    uint64_t c22_reduc = c22 ^ lsb22;
    reduc = _mm256_set1_epi64x(c22_reduc);
    vmask = _mm256_set1_epi64x(lsb22);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    if(c20 & lsb22) c20 ^= c22_reduc;
    if(c21 & lsb22) c21 ^= c22_reduc;
    mask ^= lsb22;

    uint64_t lsb21 = uint64_t_lsb(c21 & mask);
    if(unlikely(!lsb21)) {
        return -1; // singular
    }

    uint64_t c21_reduc = c21 ^ lsb21;
    reduc = _mm256_set1_epi64x(c21_reduc);
    vmask = _mm256_set1_epi64x(lsb21);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    if(c20 & lsb21) c20 ^= c21_reduc;
    uint64_t c16 = _mm256_extract_epi64(c16_19, 0);
    uint64_t c17 = _mm256_extract_epi64(c16_19, 1);
    uint64_t c18 = _mm256_extract_epi64(c16_19, 2);
    uint64_t c19 = _mm256_extract_epi64(c16_19, 3);
    mask ^= lsb21;

    uint64_t lsb20 = uint64_t_lsb(c20 & mask);
    if(unlikely(!lsb20)) {
        return -1; // singular
    }

    uint64_t c20_reduc = c20 ^ lsb20;
    reduc = _mm256_set1_epi64x(c20_reduc);
    vmask = _mm256_set1_epi64x(lsb20);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    if(c16 & lsb20) c16 ^= c20_reduc;
    if(c17 & lsb20) c17 ^= c20_reduc;
    if(c18 & lsb20) c18 ^= c20_reduc;
    if(c19 & lsb20) c19 ^= c20_reduc;
    mask ^= lsb20;

    uint64_t lsb19 = uint64_t_lsb(c19 & mask);
    if(unlikely(!lsb19)) {
        return -1; // singular
    }

    uint64_t c19_reduc = c19 ^ lsb19;
    reduc = _mm256_set1_epi64x(c19_reduc);
    vmask = _mm256_set1_epi64x(lsb19);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    if(c16 & lsb19) c16 ^= c19_reduc;
    if(c17 & lsb19) c17 ^= c19_reduc;
    if(c18 & lsb19) c18 ^= c19_reduc;
    mask ^= lsb19;

    uint64_t lsb18 = uint64_t_lsb(c18 & mask);
    if(unlikely(!lsb18)) {
        return -1; // singular
    }

    uint64_t c18_reduc = c18 ^ lsb18;
    reduc = _mm256_set1_epi64x(c18_reduc);
    vmask = _mm256_set1_epi64x(lsb18);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    if(c16 & lsb18) c16 ^= c18_reduc;
    if(c17 & lsb18) c17 ^= c18_reduc;
    mask ^= lsb18;

    uint64_t lsb17 = uint64_t_lsb(c17 & mask);
    if(unlikely(!lsb17)) {
        return -1; // singular
    }

    uint64_t c17_reduc = c17 ^ lsb17;
    reduc = _mm256_set1_epi64x(c17_reduc);
    vmask = _mm256_set1_epi64x(lsb17);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    if(c16 & lsb17) c16 ^= c17_reduc;
    uint64_t c12 = _mm256_extract_epi64(c12_15, 0);
    uint64_t c13 = _mm256_extract_epi64(c12_15, 1);
    uint64_t c14 = _mm256_extract_epi64(c12_15, 2);
    uint64_t c15 = _mm256_extract_epi64(c12_15, 3);
    mask ^= lsb17;

    uint64_t lsb16 = uint64_t_lsb(c16 & mask);
    if(unlikely(!lsb16)) {
        return -1; // singular
    }

    uint64_t c16_reduc = c16 ^ lsb16;
    reduc = _mm256_set1_epi64x(c16_reduc);
    vmask = _mm256_set1_epi64x(lsb16);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    if(c12 & lsb16) c12 ^= c16_reduc;
    if(c13 & lsb16) c13 ^= c16_reduc;
    if(c14 & lsb16) c14 ^= c16_reduc;
    if(c15 & lsb16) c15 ^= c16_reduc;
    mask ^= lsb16;

    uint64_t lsb15 = uint64_t_lsb(c15 & mask);
    if(unlikely(!lsb15)) {
        return -1; // singular
    }

    uint64_t c15_reduc = c15 ^ lsb15;
    reduc = _mm256_set1_epi64x(c15_reduc);
    vmask = _mm256_set1_epi64x(lsb15);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    if(c12 & lsb15) c12 ^= c15_reduc;
    if(c13 & lsb15) c13 ^= c15_reduc;
    if(c14 & lsb15) c14 ^= c15_reduc;
    mask ^= lsb15;

    uint64_t lsb14 = uint64_t_lsb(c14 & mask);
    if(unlikely(!lsb14)) {
        return -1; // singular
    }

    uint64_t c14_reduc = c14 ^ lsb14;
    reduc = _mm256_set1_epi64x(c14_reduc);
    vmask = _mm256_set1_epi64x(lsb14);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    if(c12 & lsb14) c12 ^= c14_reduc;
    if(c13 & lsb14) c13 ^= c14_reduc;
    mask ^= lsb14;

    uint64_t lsb13 = uint64_t_lsb(c13 & mask);
    if(unlikely(!lsb13)) {
        return -1; // singular
    }

    uint64_t c13_reduc = c13 ^ lsb13;
    reduc = _mm256_set1_epi64x(c13_reduc);
    vmask = _mm256_set1_epi64x(lsb13);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    if(c12 & lsb13) c12 ^= c13_reduc;
    uint64_t c8 = _mm256_extract_epi64(c8_11, 0);
    uint64_t c9 = _mm256_extract_epi64(c8_11, 1);
    uint64_t c10 = _mm256_extract_epi64(c8_11, 2);
    uint64_t c11 = _mm256_extract_epi64(c8_11, 3);
    mask ^= lsb13;

    uint64_t lsb12 = uint64_t_lsb(c12 & mask);
    if(unlikely(!lsb12)) {
        return -1; // singular
    }

    uint64_t c12_reduc = c12 ^ lsb12;
    reduc = _mm256_set1_epi64x(c12_reduc);
    vmask = _mm256_set1_epi64x(lsb12);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb12) c8 ^= c12_reduc;
    if(c9 & lsb12) c9 ^= c12_reduc;
    if(c10 & lsb12) c10 ^= c12_reduc;
    if(c11 & lsb12) c11 ^= c12_reduc;
    mask ^= lsb12;

    uint64_t lsb11 = uint64_t_lsb(c11 & mask);
    if(unlikely(!lsb11)) {
        return -1; // singular
    }

    uint64_t c11_reduc = c11 ^ lsb11;
    reduc = _mm256_set1_epi64x(c11_reduc);
    vmask = _mm256_set1_epi64x(lsb11);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb11) c8 ^= c11_reduc;
    if(c9 & lsb11) c9 ^= c11_reduc;
    if(c10 & lsb11) c10 ^= c11_reduc;
    mask ^= lsb11;

    uint64_t lsb10 = uint64_t_lsb(c10 & mask);
    if(unlikely(!lsb10)) {
        return -1; // singular
    }

    uint64_t c10_reduc = c10 ^ lsb10;
    reduc = _mm256_set1_epi64x(c10_reduc);
    vmask = _mm256_set1_epi64x(lsb10);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb10) c8 ^= c10_reduc;
    if(c9 & lsb10) c9 ^= c10_reduc;
    mask ^= lsb10;

    uint64_t lsb9 = uint64_t_lsb(c9 & mask);
    if(unlikely(!lsb9)) {
        return -1; // singular
    }

    uint64_t c9_reduc = c9 ^ lsb9;
    reduc = _mm256_set1_epi64x(c9_reduc);
    vmask = _mm256_set1_epi64x(lsb9);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb9) c8 ^= c9_reduc;
    uint64_t c4 = _mm256_extract_epi64(c4_7, 0);
    uint64_t c5 = _mm256_extract_epi64(c4_7, 1);
    uint64_t c6 = _mm256_extract_epi64(c4_7, 2);
    uint64_t c7 = _mm256_extract_epi64(c4_7, 3);
    mask ^= lsb9;

    uint64_t lsb8 = uint64_t_lsb(c8 & mask);
    if(unlikely(!lsb8)) {
        return -1; // singular
    }

    uint64_t c8_reduc = c8 ^ lsb8;
    reduc = _mm256_set1_epi64x(c8_reduc);
    vmask = _mm256_set1_epi64x(lsb8);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb8) c4 ^= c8_reduc;
    if(c5 & lsb8) c5 ^= c8_reduc;
    if(c6 & lsb8) c6 ^= c8_reduc;
    if(c7 & lsb8) c7 ^= c8_reduc;
    mask ^= lsb8;

    uint64_t lsb7 = uint64_t_lsb(c7 & mask);
    if(unlikely(!lsb7)) {
        return -1; // singular
    }

    uint64_t c7_reduc = c7 ^ lsb7;
    reduc = _mm256_set1_epi64x(c7_reduc);
    vmask = _mm256_set1_epi64x(lsb7);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb7) c4 ^= c7_reduc;
    if(c5 & lsb7) c5 ^= c7_reduc;
    if(c6 & lsb7) c6 ^= c7_reduc;
    mask ^= lsb7;

    uint64_t lsb6 = uint64_t_lsb(c6 & mask);
    if(unlikely(!lsb6)) {
        return -1; // singular
    }

    uint64_t c6_reduc = c6 ^ lsb6;
    reduc = _mm256_set1_epi64x(c6_reduc);
    vmask = _mm256_set1_epi64x(lsb6);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb6) c4 ^= c6_reduc;
    if(c5 & lsb6) c5 ^= c6_reduc;
    mask ^= lsb6;

    uint64_t lsb5 = uint64_t_lsb(c5 & mask);
    if(unlikely(!lsb5)) {
        return -1; // singular
    }

    uint64_t c5_reduc = c5 ^ lsb5;
    reduc = _mm256_set1_epi64x(c5_reduc);
    vmask = _mm256_set1_epi64x(lsb5);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb5) c4 ^= c5_reduc;
    uint64_t c0 = _mm256_extract_epi64(c0_3, 0);
    uint64_t c1 = _mm256_extract_epi64(c0_3, 1);
    uint64_t c2 = _mm256_extract_epi64(c0_3, 2);
    uint64_t c3 = _mm256_extract_epi64(c0_3, 3);
    mask ^= lsb5;

    uint64_t lsb4 = uint64_t_lsb(c4 & mask);
    if(unlikely(!lsb4)) {
        return -1; // singular
    }

    uint64_t c4_reduc = c4 ^ lsb4;
    if(c0 & lsb4) c0 ^= c4_reduc;
    if(c1 & lsb4) c1 ^= c4_reduc;
    if(c2 & lsb4) c2 ^= c4_reduc;
    if(c3 & lsb4) c3 ^= c4_reduc;
    mask ^= lsb4;

    uint64_t lsb3 = uint64_t_lsb(c3 & mask);
    if(unlikely(!lsb3)) {
        return -1; // singular
    }

    uint64_t c3_reduc = c3 ^ lsb3;
    if(c0 & lsb3) c0 ^= c3_reduc;
    if(c1 & lsb3) c1 ^= c3_reduc;
    if(c2 & lsb3) c2 ^= c3_reduc;
    mask ^= lsb3;

    uint64_t lsb2 = uint64_t_lsb(c2 & mask);
    if(unlikely(!lsb2)) {
        return -1; // singular
    }

    uint64_t c2_reduc = c2 ^ lsb2;
    if(c0 & lsb2) c0 ^= c2_reduc;
    if(c1 & lsb2) c1 ^= c2_reduc;
    mask ^= lsb2;

    uint64_t lsb1 = uint64_t_lsb(c1 & mask);
    if(unlikely(!lsb1)) {
        return -1; // singular
    }

    if(c0 & lsb1) c0 ^= c1 ^ lsb1;
    mask ^= lsb1;

    if(likely(mask & c0)) { // check system consistency
        return mask & c0; // not solvable
    }

    uint64_t s = 0x0ULL;
    if(c0 & lsb1) s = uint64_t_toggle_at(s, 0);
    if(c0 & lsb2) s = uint64_t_toggle_at(s, 1);
    if(c0 & lsb3) s = uint64_t_toggle_at(s, 2);
    if(c0 & lsb4) s = uint64_t_toggle_at(s, 3);
    if(c0 & lsb5) s = uint64_t_toggle_at(s, 4);
    if(c0 & lsb6) s = uint64_t_toggle_at(s, 5);
    if(c0 & lsb7) s = uint64_t_toggle_at(s, 6);
    if(c0 & lsb8) s = uint64_t_toggle_at(s, 7);
    if(c0 & lsb9) s = uint64_t_toggle_at(s, 8);
    if(c0 & lsb10) s = uint64_t_toggle_at(s, 9);
    if(c0 & lsb11) s = uint64_t_toggle_at(s, 10);
    if(c0 & lsb12) s = uint64_t_toggle_at(s, 11);
    if(c0 & lsb13) s = uint64_t_toggle_at(s, 12);
    if(c0 & lsb14) s = uint64_t_toggle_at(s, 13);
    if(c0 & lsb15) s = uint64_t_toggle_at(s, 14);
    if(c0 & lsb16) s = uint64_t_toggle_at(s, 15);
    if(c0 & lsb17) s = uint64_t_toggle_at(s, 16);
    if(c0 & lsb18) s = uint64_t_toggle_at(s, 17);
    if(c0 & lsb19) s = uint64_t_toggle_at(s, 18);
    if(c0 & lsb20) s = uint64_t_toggle_at(s, 19);
    if(c0 & lsb21) s = uint64_t_toggle_at(s, 20);
    if(c0 & lsb22) s = uint64_t_toggle_at(s, 21);
    if(c0 & lsb23) s = uint64_t_toggle_at(s, 22);
    *sol = s;
    return 0;
}

int64_t
uint64a_gj_v24_avx(const uint64_t m[25], uint64_t* const restrict sol) {
    __m256i c0_3 = _mm256_loadu_si256((__m256i*) (m + 0));
    __m256i c4_7 = _mm256_loadu_si256((__m256i*) (m + 4));
    __m256i c8_11 = _mm256_loadu_si256((__m256i*) (m + 8));
    __m256i c12_15 = _mm256_loadu_si256((__m256i*) (m + 12));
    __m256i c16_19 = _mm256_loadu_si256((__m256i*) (m + 16));
    uint64_t c20 = m[20];
    uint64_t c21 = m[21];
    uint64_t c22 = m[22];
    uint64_t c23 = m[23];
    uint64_t c24 = m[24];

    uint64_t lsb24 = uint64_t_lsb(c24);
    if(unlikely(!lsb24)) {
        return -1; // singular
    }

    uint64_t c24_reduc = c24 ^ lsb24;
    __m256i reduc = _mm256_set1_epi64x(c24_reduc);
    __m256i vmask = _mm256_set1_epi64x(lsb24);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    if(c20 & lsb24) c20 ^= c24_reduc;
    if(c21 & lsb24) c21 ^= c24_reduc;
    if(c22 & lsb24) c22 ^= c24_reduc;
    if(c23 & lsb24) c23 ^= c24_reduc;
    uint64_t mask = ~lsb24;

    uint64_t lsb23 = uint64_t_lsb(c23 & mask);
    if(unlikely(!lsb23)) {
        return -1; // singular
    }

    uint64_t c23_reduc = c23 ^ lsb23;
    reduc = _mm256_set1_epi64x(c23_reduc);
    vmask = _mm256_set1_epi64x(lsb23);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    if(c20 & lsb23) c20 ^= c23_reduc;
    if(c21 & lsb23) c21 ^= c23_reduc;
    if(c22 & lsb23) c22 ^= c23_reduc;
    mask ^= lsb23;

    uint64_t lsb22 = uint64_t_lsb(c22 & mask);
    if(unlikely(!lsb22)) {
        return -1; // singular
    }

    uint64_t c22_reduc = c22 ^ lsb22;
    reduc = _mm256_set1_epi64x(c22_reduc);
    vmask = _mm256_set1_epi64x(lsb22);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    if(c20 & lsb22) c20 ^= c22_reduc;
    if(c21 & lsb22) c21 ^= c22_reduc;
    mask ^= lsb22;

    uint64_t lsb21 = uint64_t_lsb(c21 & mask);
    if(unlikely(!lsb21)) {
        return -1; // singular
    }

    uint64_t c21_reduc = c21 ^ lsb21;
    reduc = _mm256_set1_epi64x(c21_reduc);
    vmask = _mm256_set1_epi64x(lsb21);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    if(c20 & lsb21) c20 ^= c21_reduc;
    uint64_t c16 = _mm256_extract_epi64(c16_19, 0);
    uint64_t c17 = _mm256_extract_epi64(c16_19, 1);
    uint64_t c18 = _mm256_extract_epi64(c16_19, 2);
    uint64_t c19 = _mm256_extract_epi64(c16_19, 3);
    mask ^= lsb21;

    uint64_t lsb20 = uint64_t_lsb(c20 & mask);
    if(unlikely(!lsb20)) {
        return -1; // singular
    }

    uint64_t c20_reduc = c20 ^ lsb20;
    reduc = _mm256_set1_epi64x(c20_reduc);
    vmask = _mm256_set1_epi64x(lsb20);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    if(c16 & lsb20) c16 ^= c20_reduc;
    if(c17 & lsb20) c17 ^= c20_reduc;
    if(c18 & lsb20) c18 ^= c20_reduc;
    if(c19 & lsb20) c19 ^= c20_reduc;
    mask ^= lsb20;

    uint64_t lsb19 = uint64_t_lsb(c19 & mask);
    if(unlikely(!lsb19)) {
        return -1; // singular
    }

    uint64_t c19_reduc = c19 ^ lsb19;
    reduc = _mm256_set1_epi64x(c19_reduc);
    vmask = _mm256_set1_epi64x(lsb19);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    if(c16 & lsb19) c16 ^= c19_reduc;
    if(c17 & lsb19) c17 ^= c19_reduc;
    if(c18 & lsb19) c18 ^= c19_reduc;
    mask ^= lsb19;

    uint64_t lsb18 = uint64_t_lsb(c18 & mask);
    if(unlikely(!lsb18)) {
        return -1; // singular
    }

    uint64_t c18_reduc = c18 ^ lsb18;
    reduc = _mm256_set1_epi64x(c18_reduc);
    vmask = _mm256_set1_epi64x(lsb18);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    if(c16 & lsb18) c16 ^= c18_reduc;
    if(c17 & lsb18) c17 ^= c18_reduc;
    mask ^= lsb18;

    uint64_t lsb17 = uint64_t_lsb(c17 & mask);
    if(unlikely(!lsb17)) {
        return -1; // singular
    }

    uint64_t c17_reduc = c17 ^ lsb17;
    reduc = _mm256_set1_epi64x(c17_reduc);
    vmask = _mm256_set1_epi64x(lsb17);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    if(c16 & lsb17) c16 ^= c17_reduc;
    uint64_t c12 = _mm256_extract_epi64(c12_15, 0);
    uint64_t c13 = _mm256_extract_epi64(c12_15, 1);
    uint64_t c14 = _mm256_extract_epi64(c12_15, 2);
    uint64_t c15 = _mm256_extract_epi64(c12_15, 3);
    mask ^= lsb17;

    uint64_t lsb16 = uint64_t_lsb(c16 & mask);
    if(unlikely(!lsb16)) {
        return -1; // singular
    }

    uint64_t c16_reduc = c16 ^ lsb16;
    reduc = _mm256_set1_epi64x(c16_reduc);
    vmask = _mm256_set1_epi64x(lsb16);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    if(c12 & lsb16) c12 ^= c16_reduc;
    if(c13 & lsb16) c13 ^= c16_reduc;
    if(c14 & lsb16) c14 ^= c16_reduc;
    if(c15 & lsb16) c15 ^= c16_reduc;
    mask ^= lsb16;

    uint64_t lsb15 = uint64_t_lsb(c15 & mask);
    if(unlikely(!lsb15)) {
        return -1; // singular
    }

    uint64_t c15_reduc = c15 ^ lsb15;
    reduc = _mm256_set1_epi64x(c15_reduc);
    vmask = _mm256_set1_epi64x(lsb15);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    if(c12 & lsb15) c12 ^= c15_reduc;
    if(c13 & lsb15) c13 ^= c15_reduc;
    if(c14 & lsb15) c14 ^= c15_reduc;
    mask ^= lsb15;

    uint64_t lsb14 = uint64_t_lsb(c14 & mask);
    if(unlikely(!lsb14)) {
        return -1; // singular
    }

    uint64_t c14_reduc = c14 ^ lsb14;
    reduc = _mm256_set1_epi64x(c14_reduc);
    vmask = _mm256_set1_epi64x(lsb14);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    if(c12 & lsb14) c12 ^= c14_reduc;
    if(c13 & lsb14) c13 ^= c14_reduc;
    mask ^= lsb14;

    uint64_t lsb13 = uint64_t_lsb(c13 & mask);
    if(unlikely(!lsb13)) {
        return -1; // singular
    }

    uint64_t c13_reduc = c13 ^ lsb13;
    reduc = _mm256_set1_epi64x(c13_reduc);
    vmask = _mm256_set1_epi64x(lsb13);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    if(c12 & lsb13) c12 ^= c13_reduc;
    uint64_t c8 = _mm256_extract_epi64(c8_11, 0);
    uint64_t c9 = _mm256_extract_epi64(c8_11, 1);
    uint64_t c10 = _mm256_extract_epi64(c8_11, 2);
    uint64_t c11 = _mm256_extract_epi64(c8_11, 3);
    mask ^= lsb13;

    uint64_t lsb12 = uint64_t_lsb(c12 & mask);
    if(unlikely(!lsb12)) {
        return -1; // singular
    }

    uint64_t c12_reduc = c12 ^ lsb12;
    reduc = _mm256_set1_epi64x(c12_reduc);
    vmask = _mm256_set1_epi64x(lsb12);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb12) c8 ^= c12_reduc;
    if(c9 & lsb12) c9 ^= c12_reduc;
    if(c10 & lsb12) c10 ^= c12_reduc;
    if(c11 & lsb12) c11 ^= c12_reduc;
    mask ^= lsb12;

    uint64_t lsb11 = uint64_t_lsb(c11 & mask);
    if(unlikely(!lsb11)) {
        return -1; // singular
    }

    uint64_t c11_reduc = c11 ^ lsb11;
    reduc = _mm256_set1_epi64x(c11_reduc);
    vmask = _mm256_set1_epi64x(lsb11);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb11) c8 ^= c11_reduc;
    if(c9 & lsb11) c9 ^= c11_reduc;
    if(c10 & lsb11) c10 ^= c11_reduc;
    mask ^= lsb11;

    uint64_t lsb10 = uint64_t_lsb(c10 & mask);
    if(unlikely(!lsb10)) {
        return -1; // singular
    }

    uint64_t c10_reduc = c10 ^ lsb10;
    reduc = _mm256_set1_epi64x(c10_reduc);
    vmask = _mm256_set1_epi64x(lsb10);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb10) c8 ^= c10_reduc;
    if(c9 & lsb10) c9 ^= c10_reduc;
    mask ^= lsb10;

    uint64_t lsb9 = uint64_t_lsb(c9 & mask);
    if(unlikely(!lsb9)) {
        return -1; // singular
    }

    uint64_t c9_reduc = c9 ^ lsb9;
    reduc = _mm256_set1_epi64x(c9_reduc);
    vmask = _mm256_set1_epi64x(lsb9);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb9) c8 ^= c9_reduc;
    uint64_t c4 = _mm256_extract_epi64(c4_7, 0);
    uint64_t c5 = _mm256_extract_epi64(c4_7, 1);
    uint64_t c6 = _mm256_extract_epi64(c4_7, 2);
    uint64_t c7 = _mm256_extract_epi64(c4_7, 3);
    mask ^= lsb9;

    uint64_t lsb8 = uint64_t_lsb(c8 & mask);
    if(unlikely(!lsb8)) {
        return -1; // singular
    }

    uint64_t c8_reduc = c8 ^ lsb8;
    reduc = _mm256_set1_epi64x(c8_reduc);
    vmask = _mm256_set1_epi64x(lsb8);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb8) c4 ^= c8_reduc;
    if(c5 & lsb8) c5 ^= c8_reduc;
    if(c6 & lsb8) c6 ^= c8_reduc;
    if(c7 & lsb8) c7 ^= c8_reduc;
    mask ^= lsb8;

    uint64_t lsb7 = uint64_t_lsb(c7 & mask);
    if(unlikely(!lsb7)) {
        return -1; // singular
    }

    uint64_t c7_reduc = c7 ^ lsb7;
    reduc = _mm256_set1_epi64x(c7_reduc);
    vmask = _mm256_set1_epi64x(lsb7);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb7) c4 ^= c7_reduc;
    if(c5 & lsb7) c5 ^= c7_reduc;
    if(c6 & lsb7) c6 ^= c7_reduc;
    mask ^= lsb7;

    uint64_t lsb6 = uint64_t_lsb(c6 & mask);
    if(unlikely(!lsb6)) {
        return -1; // singular
    }

    uint64_t c6_reduc = c6 ^ lsb6;
    reduc = _mm256_set1_epi64x(c6_reduc);
    vmask = _mm256_set1_epi64x(lsb6);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb6) c4 ^= c6_reduc;
    if(c5 & lsb6) c5 ^= c6_reduc;
    mask ^= lsb6;

    uint64_t lsb5 = uint64_t_lsb(c5 & mask);
    if(unlikely(!lsb5)) {
        return -1; // singular
    }

    uint64_t c5_reduc = c5 ^ lsb5;
    reduc = _mm256_set1_epi64x(c5_reduc);
    vmask = _mm256_set1_epi64x(lsb5);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb5) c4 ^= c5_reduc;
    uint64_t c0 = _mm256_extract_epi64(c0_3, 0);
    uint64_t c1 = _mm256_extract_epi64(c0_3, 1);
    uint64_t c2 = _mm256_extract_epi64(c0_3, 2);
    uint64_t c3 = _mm256_extract_epi64(c0_3, 3);
    mask ^= lsb5;

    uint64_t lsb4 = uint64_t_lsb(c4 & mask);
    if(unlikely(!lsb4)) {
        return -1; // singular
    }

    uint64_t c4_reduc = c4 ^ lsb4;
    if(c0 & lsb4) c0 ^= c4_reduc;
    if(c1 & lsb4) c1 ^= c4_reduc;
    if(c2 & lsb4) c2 ^= c4_reduc;
    if(c3 & lsb4) c3 ^= c4_reduc;
    mask ^= lsb4;

    uint64_t lsb3 = uint64_t_lsb(c3 & mask);
    if(unlikely(!lsb3)) {
        return -1; // singular
    }

    uint64_t c3_reduc = c3 ^ lsb3;
    if(c0 & lsb3) c0 ^= c3_reduc;
    if(c1 & lsb3) c1 ^= c3_reduc;
    if(c2 & lsb3) c2 ^= c3_reduc;
    mask ^= lsb3;

    uint64_t lsb2 = uint64_t_lsb(c2 & mask);
    if(unlikely(!lsb2)) {
        return -1; // singular
    }

    uint64_t c2_reduc = c2 ^ lsb2;
    if(c0 & lsb2) c0 ^= c2_reduc;
    if(c1 & lsb2) c1 ^= c2_reduc;
    mask ^= lsb2;

    uint64_t lsb1 = uint64_t_lsb(c1 & mask);
    if(unlikely(!lsb1)) {
        return -1; // singular
    }

    if(c0 & lsb1) c0 ^= c1 ^ lsb1;
    mask ^= lsb1;

    if(likely(mask & c0)) { // check system consistency
        return mask & c0; // not solvable
    }

    uint64_t s = 0x0ULL;
    if(c0 & lsb1) s = uint64_t_toggle_at(s, 0);
    if(c0 & lsb2) s = uint64_t_toggle_at(s, 1);
    if(c0 & lsb3) s = uint64_t_toggle_at(s, 2);
    if(c0 & lsb4) s = uint64_t_toggle_at(s, 3);
    if(c0 & lsb5) s = uint64_t_toggle_at(s, 4);
    if(c0 & lsb6) s = uint64_t_toggle_at(s, 5);
    if(c0 & lsb7) s = uint64_t_toggle_at(s, 6);
    if(c0 & lsb8) s = uint64_t_toggle_at(s, 7);
    if(c0 & lsb9) s = uint64_t_toggle_at(s, 8);
    if(c0 & lsb10) s = uint64_t_toggle_at(s, 9);
    if(c0 & lsb11) s = uint64_t_toggle_at(s, 10);
    if(c0 & lsb12) s = uint64_t_toggle_at(s, 11);
    if(c0 & lsb13) s = uint64_t_toggle_at(s, 12);
    if(c0 & lsb14) s = uint64_t_toggle_at(s, 13);
    if(c0 & lsb15) s = uint64_t_toggle_at(s, 14);
    if(c0 & lsb16) s = uint64_t_toggle_at(s, 15);
    if(c0 & lsb17) s = uint64_t_toggle_at(s, 16);
    if(c0 & lsb18) s = uint64_t_toggle_at(s, 17);
    if(c0 & lsb19) s = uint64_t_toggle_at(s, 18);
    if(c0 & lsb20) s = uint64_t_toggle_at(s, 19);
    if(c0 & lsb21) s = uint64_t_toggle_at(s, 20);
    if(c0 & lsb22) s = uint64_t_toggle_at(s, 21);
    if(c0 & lsb23) s = uint64_t_toggle_at(s, 22);
    if(c0 & lsb24) s = uint64_t_toggle_at(s, 23);
    *sol = s;
    return 0;
}

int64_t
uint64a_gj_v25_avx(const uint64_t m[26], uint64_t* const restrict sol) {
    __m256i c0_3 = _mm256_loadu_si256((__m256i*) (m + 0));
    __m256i c4_7 = _mm256_loadu_si256((__m256i*) (m + 4));
    __m256i c8_11 = _mm256_loadu_si256((__m256i*) (m + 8));
    __m256i c12_15 = _mm256_loadu_si256((__m256i*) (m + 12));
    __m256i c16_19 = _mm256_loadu_si256((__m256i*) (m + 16));
    __m256i c20_23 = _mm256_loadu_si256((__m256i*) (m + 20));
    uint64_t c24 = m[24];
    uint64_t c25 = m[25];

    uint64_t lsb25 = uint64_t_lsb(c25);
    if(unlikely(!lsb25)) {
        return -1; // singular
    }

    uint64_t c25_reduc = c25 ^ lsb25;
    __m256i reduc = _mm256_set1_epi64x(c25_reduc);
    __m256i vmask = _mm256_set1_epi64x(lsb25);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    c20_23 = uint64a_gj_reduc_mm256(c20_23, vmask, reduc);
    if(c24 & lsb25) c24 ^= c25_reduc;
    uint64_t mask = ~lsb25;

    uint64_t lsb24 = uint64_t_lsb(c24 & mask);
    if(unlikely(!lsb24)) {
        return -1; // singular
    }

    uint64_t c24_reduc = c24 ^ lsb24;
    reduc = _mm256_set1_epi64x(c24_reduc);
    vmask = _mm256_set1_epi64x(lsb24);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    c20_23 = uint64a_gj_reduc_mm256(c20_23, vmask, reduc);
    uint64_t c20 = _mm256_extract_epi64(c20_23, 0);
    uint64_t c21 = _mm256_extract_epi64(c20_23, 1);
    uint64_t c22 = _mm256_extract_epi64(c20_23, 2);
    uint64_t c23 = _mm256_extract_epi64(c20_23, 3);
    mask ^= lsb24;

    uint64_t lsb23 = uint64_t_lsb(c23 & mask);
    if(unlikely(!lsb23)) {
        return -1; // singular
    }

    uint64_t c23_reduc = c23 ^ lsb23;
    reduc = _mm256_set1_epi64x(c23_reduc);
    vmask = _mm256_set1_epi64x(lsb23);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    if(c20 & lsb23) c20 ^= c23_reduc;
    if(c21 & lsb23) c21 ^= c23_reduc;
    if(c22 & lsb23) c22 ^= c23_reduc;
    mask ^= lsb23;

    uint64_t lsb22 = uint64_t_lsb(c22 & mask);
    if(unlikely(!lsb22)) {
        return -1; // singular
    }

    uint64_t c22_reduc = c22 ^ lsb22;
    reduc = _mm256_set1_epi64x(c22_reduc);
    vmask = _mm256_set1_epi64x(lsb22);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    if(c20 & lsb22) c20 ^= c22_reduc;
    if(c21 & lsb22) c21 ^= c22_reduc;
    mask ^= lsb22;

    uint64_t lsb21 = uint64_t_lsb(c21 & mask);
    if(unlikely(!lsb21)) {
        return -1; // singular
    }

    uint64_t c21_reduc = c21 ^ lsb21;
    reduc = _mm256_set1_epi64x(c21_reduc);
    vmask = _mm256_set1_epi64x(lsb21);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    if(c20 & lsb21) c20 ^= c21_reduc;
    uint64_t c16 = _mm256_extract_epi64(c16_19, 0);
    uint64_t c17 = _mm256_extract_epi64(c16_19, 1);
    uint64_t c18 = _mm256_extract_epi64(c16_19, 2);
    uint64_t c19 = _mm256_extract_epi64(c16_19, 3);
    mask ^= lsb21;

    uint64_t lsb20 = uint64_t_lsb(c20 & mask);
    if(unlikely(!lsb20)) {
        return -1; // singular
    }

    uint64_t c20_reduc = c20 ^ lsb20;
    reduc = _mm256_set1_epi64x(c20_reduc);
    vmask = _mm256_set1_epi64x(lsb20);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    if(c16 & lsb20) c16 ^= c20_reduc;
    if(c17 & lsb20) c17 ^= c20_reduc;
    if(c18 & lsb20) c18 ^= c20_reduc;
    if(c19 & lsb20) c19 ^= c20_reduc;
    mask ^= lsb20;

    uint64_t lsb19 = uint64_t_lsb(c19 & mask);
    if(unlikely(!lsb19)) {
        return -1; // singular
    }

    uint64_t c19_reduc = c19 ^ lsb19;
    reduc = _mm256_set1_epi64x(c19_reduc);
    vmask = _mm256_set1_epi64x(lsb19);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    if(c16 & lsb19) c16 ^= c19_reduc;
    if(c17 & lsb19) c17 ^= c19_reduc;
    if(c18 & lsb19) c18 ^= c19_reduc;
    mask ^= lsb19;

    uint64_t lsb18 = uint64_t_lsb(c18 & mask);
    if(unlikely(!lsb18)) {
        return -1; // singular
    }

    uint64_t c18_reduc = c18 ^ lsb18;
    reduc = _mm256_set1_epi64x(c18_reduc);
    vmask = _mm256_set1_epi64x(lsb18);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    if(c16 & lsb18) c16 ^= c18_reduc;
    if(c17 & lsb18) c17 ^= c18_reduc;
    mask ^= lsb18;

    uint64_t lsb17 = uint64_t_lsb(c17 & mask);
    if(unlikely(!lsb17)) {
        return -1; // singular
    }

    uint64_t c17_reduc = c17 ^ lsb17;
    reduc = _mm256_set1_epi64x(c17_reduc);
    vmask = _mm256_set1_epi64x(lsb17);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    if(c16 & lsb17) c16 ^= c17_reduc;
    uint64_t c12 = _mm256_extract_epi64(c12_15, 0);
    uint64_t c13 = _mm256_extract_epi64(c12_15, 1);
    uint64_t c14 = _mm256_extract_epi64(c12_15, 2);
    uint64_t c15 = _mm256_extract_epi64(c12_15, 3);
    mask ^= lsb17;

    uint64_t lsb16 = uint64_t_lsb(c16 & mask);
    if(unlikely(!lsb16)) {
        return -1; // singular
    }

    uint64_t c16_reduc = c16 ^ lsb16;
    reduc = _mm256_set1_epi64x(c16_reduc);
    vmask = _mm256_set1_epi64x(lsb16);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    if(c12 & lsb16) c12 ^= c16_reduc;
    if(c13 & lsb16) c13 ^= c16_reduc;
    if(c14 & lsb16) c14 ^= c16_reduc;
    if(c15 & lsb16) c15 ^= c16_reduc;
    mask ^= lsb16;

    uint64_t lsb15 = uint64_t_lsb(c15 & mask);
    if(unlikely(!lsb15)) {
        return -1; // singular
    }

    uint64_t c15_reduc = c15 ^ lsb15;
    reduc = _mm256_set1_epi64x(c15_reduc);
    vmask = _mm256_set1_epi64x(lsb15);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    if(c12 & lsb15) c12 ^= c15_reduc;
    if(c13 & lsb15) c13 ^= c15_reduc;
    if(c14 & lsb15) c14 ^= c15_reduc;
    mask ^= lsb15;

    uint64_t lsb14 = uint64_t_lsb(c14 & mask);
    if(unlikely(!lsb14)) {
        return -1; // singular
    }

    uint64_t c14_reduc = c14 ^ lsb14;
    reduc = _mm256_set1_epi64x(c14_reduc);
    vmask = _mm256_set1_epi64x(lsb14);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    if(c12 & lsb14) c12 ^= c14_reduc;
    if(c13 & lsb14) c13 ^= c14_reduc;
    mask ^= lsb14;

    uint64_t lsb13 = uint64_t_lsb(c13 & mask);
    if(unlikely(!lsb13)) {
        return -1; // singular
    }

    uint64_t c13_reduc = c13 ^ lsb13;
    reduc = _mm256_set1_epi64x(c13_reduc);
    vmask = _mm256_set1_epi64x(lsb13);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    if(c12 & lsb13) c12 ^= c13_reduc;
    uint64_t c8 = _mm256_extract_epi64(c8_11, 0);
    uint64_t c9 = _mm256_extract_epi64(c8_11, 1);
    uint64_t c10 = _mm256_extract_epi64(c8_11, 2);
    uint64_t c11 = _mm256_extract_epi64(c8_11, 3);
    mask ^= lsb13;

    uint64_t lsb12 = uint64_t_lsb(c12 & mask);
    if(unlikely(!lsb12)) {
        return -1; // singular
    }

    uint64_t c12_reduc = c12 ^ lsb12;
    reduc = _mm256_set1_epi64x(c12_reduc);
    vmask = _mm256_set1_epi64x(lsb12);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb12) c8 ^= c12_reduc;
    if(c9 & lsb12) c9 ^= c12_reduc;
    if(c10 & lsb12) c10 ^= c12_reduc;
    if(c11 & lsb12) c11 ^= c12_reduc;
    mask ^= lsb12;

    uint64_t lsb11 = uint64_t_lsb(c11 & mask);
    if(unlikely(!lsb11)) {
        return -1; // singular
    }

    uint64_t c11_reduc = c11 ^ lsb11;
    reduc = _mm256_set1_epi64x(c11_reduc);
    vmask = _mm256_set1_epi64x(lsb11);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb11) c8 ^= c11_reduc;
    if(c9 & lsb11) c9 ^= c11_reduc;
    if(c10 & lsb11) c10 ^= c11_reduc;
    mask ^= lsb11;

    uint64_t lsb10 = uint64_t_lsb(c10 & mask);
    if(unlikely(!lsb10)) {
        return -1; // singular
    }

    uint64_t c10_reduc = c10 ^ lsb10;
    reduc = _mm256_set1_epi64x(c10_reduc);
    vmask = _mm256_set1_epi64x(lsb10);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb10) c8 ^= c10_reduc;
    if(c9 & lsb10) c9 ^= c10_reduc;
    mask ^= lsb10;

    uint64_t lsb9 = uint64_t_lsb(c9 & mask);
    if(unlikely(!lsb9)) {
        return -1; // singular
    }

    uint64_t c9_reduc = c9 ^ lsb9;
    reduc = _mm256_set1_epi64x(c9_reduc);
    vmask = _mm256_set1_epi64x(lsb9);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb9) c8 ^= c9_reduc;
    uint64_t c4 = _mm256_extract_epi64(c4_7, 0);
    uint64_t c5 = _mm256_extract_epi64(c4_7, 1);
    uint64_t c6 = _mm256_extract_epi64(c4_7, 2);
    uint64_t c7 = _mm256_extract_epi64(c4_7, 3);
    mask ^= lsb9;

    uint64_t lsb8 = uint64_t_lsb(c8 & mask);
    if(unlikely(!lsb8)) {
        return -1; // singular
    }

    uint64_t c8_reduc = c8 ^ lsb8;
    reduc = _mm256_set1_epi64x(c8_reduc);
    vmask = _mm256_set1_epi64x(lsb8);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb8) c4 ^= c8_reduc;
    if(c5 & lsb8) c5 ^= c8_reduc;
    if(c6 & lsb8) c6 ^= c8_reduc;
    if(c7 & lsb8) c7 ^= c8_reduc;
    mask ^= lsb8;

    uint64_t lsb7 = uint64_t_lsb(c7 & mask);
    if(unlikely(!lsb7)) {
        return -1; // singular
    }

    uint64_t c7_reduc = c7 ^ lsb7;
    reduc = _mm256_set1_epi64x(c7_reduc);
    vmask = _mm256_set1_epi64x(lsb7);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb7) c4 ^= c7_reduc;
    if(c5 & lsb7) c5 ^= c7_reduc;
    if(c6 & lsb7) c6 ^= c7_reduc;
    mask ^= lsb7;

    uint64_t lsb6 = uint64_t_lsb(c6 & mask);
    if(unlikely(!lsb6)) {
        return -1; // singular
    }

    uint64_t c6_reduc = c6 ^ lsb6;
    reduc = _mm256_set1_epi64x(c6_reduc);
    vmask = _mm256_set1_epi64x(lsb6);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb6) c4 ^= c6_reduc;
    if(c5 & lsb6) c5 ^= c6_reduc;
    mask ^= lsb6;

    uint64_t lsb5 = uint64_t_lsb(c5 & mask);
    if(unlikely(!lsb5)) {
        return -1; // singular
    }

    uint64_t c5_reduc = c5 ^ lsb5;
    reduc = _mm256_set1_epi64x(c5_reduc);
    vmask = _mm256_set1_epi64x(lsb5);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb5) c4 ^= c5_reduc;
    uint64_t c0 = _mm256_extract_epi64(c0_3, 0);
    uint64_t c1 = _mm256_extract_epi64(c0_3, 1);
    uint64_t c2 = _mm256_extract_epi64(c0_3, 2);
    uint64_t c3 = _mm256_extract_epi64(c0_3, 3);
    mask ^= lsb5;

    uint64_t lsb4 = uint64_t_lsb(c4 & mask);
    if(unlikely(!lsb4)) {
        return -1; // singular
    }

    uint64_t c4_reduc = c4 ^ lsb4;
    if(c0 & lsb4) c0 ^= c4_reduc;
    if(c1 & lsb4) c1 ^= c4_reduc;
    if(c2 & lsb4) c2 ^= c4_reduc;
    if(c3 & lsb4) c3 ^= c4_reduc;
    mask ^= lsb4;

    uint64_t lsb3 = uint64_t_lsb(c3 & mask);
    if(unlikely(!lsb3)) {
        return -1; // singular
    }

    uint64_t c3_reduc = c3 ^ lsb3;
    if(c0 & lsb3) c0 ^= c3_reduc;
    if(c1 & lsb3) c1 ^= c3_reduc;
    if(c2 & lsb3) c2 ^= c3_reduc;
    mask ^= lsb3;

    uint64_t lsb2 = uint64_t_lsb(c2 & mask);
    if(unlikely(!lsb2)) {
        return -1; // singular
    }

    uint64_t c2_reduc = c2 ^ lsb2;
    if(c0 & lsb2) c0 ^= c2_reduc;
    if(c1 & lsb2) c1 ^= c2_reduc;
    mask ^= lsb2;

    uint64_t lsb1 = uint64_t_lsb(c1 & mask);
    if(unlikely(!lsb1)) {
        return -1; // singular
    }

    if(c0 & lsb1) c0 ^= c1 ^ lsb1;
    mask ^= lsb1;

    if(likely(mask & c0)) { // check system consistency
        return mask & c0; // not solvable
    }

    uint64_t s = 0x0ULL;
    if(c0 & lsb1) s = uint64_t_toggle_at(s, 0);
    if(c0 & lsb2) s = uint64_t_toggle_at(s, 1);
    if(c0 & lsb3) s = uint64_t_toggle_at(s, 2);
    if(c0 & lsb4) s = uint64_t_toggle_at(s, 3);
    if(c0 & lsb5) s = uint64_t_toggle_at(s, 4);
    if(c0 & lsb6) s = uint64_t_toggle_at(s, 5);
    if(c0 & lsb7) s = uint64_t_toggle_at(s, 6);
    if(c0 & lsb8) s = uint64_t_toggle_at(s, 7);
    if(c0 & lsb9) s = uint64_t_toggle_at(s, 8);
    if(c0 & lsb10) s = uint64_t_toggle_at(s, 9);
    if(c0 & lsb11) s = uint64_t_toggle_at(s, 10);
    if(c0 & lsb12) s = uint64_t_toggle_at(s, 11);
    if(c0 & lsb13) s = uint64_t_toggle_at(s, 12);
    if(c0 & lsb14) s = uint64_t_toggle_at(s, 13);
    if(c0 & lsb15) s = uint64_t_toggle_at(s, 14);
    if(c0 & lsb16) s = uint64_t_toggle_at(s, 15);
    if(c0 & lsb17) s = uint64_t_toggle_at(s, 16);
    if(c0 & lsb18) s = uint64_t_toggle_at(s, 17);
    if(c0 & lsb19) s = uint64_t_toggle_at(s, 18);
    if(c0 & lsb20) s = uint64_t_toggle_at(s, 19);
    if(c0 & lsb21) s = uint64_t_toggle_at(s, 20);
    if(c0 & lsb22) s = uint64_t_toggle_at(s, 21);
    if(c0 & lsb23) s = uint64_t_toggle_at(s, 22);
    if(c0 & lsb24) s = uint64_t_toggle_at(s, 23);
    if(c0 & lsb25) s = uint64_t_toggle_at(s, 24);
    *sol = s;
    return 0;
}

int64_t
uint64a_gj_v26_avx(const uint64_t m[27], uint64_t* const restrict sol) {
    __m256i c0_3 = _mm256_loadu_si256((__m256i*) (m + 0));
    __m256i c4_7 = _mm256_loadu_si256((__m256i*) (m + 4));
    __m256i c8_11 = _mm256_loadu_si256((__m256i*) (m + 8));
    __m256i c12_15 = _mm256_loadu_si256((__m256i*) (m + 12));
    __m256i c16_19 = _mm256_loadu_si256((__m256i*) (m + 16));
    __m256i c20_23 = _mm256_loadu_si256((__m256i*) (m + 20));
    uint64_t c24 = m[24];
    uint64_t c25 = m[25];
    uint64_t c26 = m[26];

    uint64_t lsb26 = uint64_t_lsb(c26);
    if(unlikely(!lsb26)) {
        return -1; // singular
    }

    uint64_t c26_reduc = c26 ^ lsb26;
    __m256i reduc = _mm256_set1_epi64x(c26_reduc);
    __m256i vmask = _mm256_set1_epi64x(lsb26);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    c20_23 = uint64a_gj_reduc_mm256(c20_23, vmask, reduc);
    if(c24 & lsb26) c24 ^= c26_reduc;
    if(c25 & lsb26) c25 ^= c26_reduc;
    uint64_t mask = ~lsb26;

    uint64_t lsb25 = uint64_t_lsb(c25 & mask);
    if(unlikely(!lsb25)) {
        return -1; // singular
    }

    uint64_t c25_reduc = c25 ^ lsb25;
    reduc = _mm256_set1_epi64x(c25_reduc);
    vmask = _mm256_set1_epi64x(lsb25);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    c20_23 = uint64a_gj_reduc_mm256(c20_23, vmask, reduc);
    if(c24 & lsb25) c24 ^= c25_reduc;
    uint64_t c20 = _mm256_extract_epi64(c20_23, 0);
    uint64_t c21 = _mm256_extract_epi64(c20_23, 1);
    uint64_t c22 = _mm256_extract_epi64(c20_23, 2);
    uint64_t c23 = _mm256_extract_epi64(c20_23, 3);
    mask ^= lsb25;

    uint64_t lsb24 = uint64_t_lsb(c24 & mask);
    if(unlikely(!lsb24)) {
        return -1; // singular
    }

    uint64_t c24_reduc = c24 ^ lsb24;
    reduc = _mm256_set1_epi64x(c24_reduc);
    vmask = _mm256_set1_epi64x(lsb24);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    if(c20 & lsb24) c20 ^= c24_reduc;
    if(c21 & lsb24) c21 ^= c24_reduc;
    if(c22 & lsb24) c22 ^= c24_reduc;
    if(c23 & lsb24) c23 ^= c24_reduc;
    mask ^= lsb24;

    uint64_t lsb23 = uint64_t_lsb(c23 & mask);
    if(unlikely(!lsb23)) {
        return -1; // singular
    }

    uint64_t c23_reduc = c23 ^ lsb23;
    reduc = _mm256_set1_epi64x(c23_reduc);
    vmask = _mm256_set1_epi64x(lsb23);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    if(c20 & lsb23) c20 ^= c23_reduc;
    if(c21 & lsb23) c21 ^= c23_reduc;
    if(c22 & lsb23) c22 ^= c23_reduc;
    mask ^= lsb23;

    uint64_t lsb22 = uint64_t_lsb(c22 & mask);
    if(unlikely(!lsb22)) {
        return -1; // singular
    }

    uint64_t c22_reduc = c22 ^ lsb22;
    reduc = _mm256_set1_epi64x(c22_reduc);
    vmask = _mm256_set1_epi64x(lsb22);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    if(c20 & lsb22) c20 ^= c22_reduc;
    if(c21 & lsb22) c21 ^= c22_reduc;
    mask ^= lsb22;

    uint64_t lsb21 = uint64_t_lsb(c21 & mask);
    if(unlikely(!lsb21)) {
        return -1; // singular
    }

    uint64_t c21_reduc = c21 ^ lsb21;
    reduc = _mm256_set1_epi64x(c21_reduc);
    vmask = _mm256_set1_epi64x(lsb21);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    if(c20 & lsb21) c20 ^= c21_reduc;
    uint64_t c16 = _mm256_extract_epi64(c16_19, 0);
    uint64_t c17 = _mm256_extract_epi64(c16_19, 1);
    uint64_t c18 = _mm256_extract_epi64(c16_19, 2);
    uint64_t c19 = _mm256_extract_epi64(c16_19, 3);
    mask ^= lsb21;

    uint64_t lsb20 = uint64_t_lsb(c20 & mask);
    if(unlikely(!lsb20)) {
        return -1; // singular
    }

    uint64_t c20_reduc = c20 ^ lsb20;
    reduc = _mm256_set1_epi64x(c20_reduc);
    vmask = _mm256_set1_epi64x(lsb20);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    if(c16 & lsb20) c16 ^= c20_reduc;
    if(c17 & lsb20) c17 ^= c20_reduc;
    if(c18 & lsb20) c18 ^= c20_reduc;
    if(c19 & lsb20) c19 ^= c20_reduc;
    mask ^= lsb20;

    uint64_t lsb19 = uint64_t_lsb(c19 & mask);
    if(unlikely(!lsb19)) {
        return -1; // singular
    }

    uint64_t c19_reduc = c19 ^ lsb19;
    reduc = _mm256_set1_epi64x(c19_reduc);
    vmask = _mm256_set1_epi64x(lsb19);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    if(c16 & lsb19) c16 ^= c19_reduc;
    if(c17 & lsb19) c17 ^= c19_reduc;
    if(c18 & lsb19) c18 ^= c19_reduc;
    mask ^= lsb19;

    uint64_t lsb18 = uint64_t_lsb(c18 & mask);
    if(unlikely(!lsb18)) {
        return -1; // singular
    }

    uint64_t c18_reduc = c18 ^ lsb18;
    reduc = _mm256_set1_epi64x(c18_reduc);
    vmask = _mm256_set1_epi64x(lsb18);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    if(c16 & lsb18) c16 ^= c18_reduc;
    if(c17 & lsb18) c17 ^= c18_reduc;
    mask ^= lsb18;

    uint64_t lsb17 = uint64_t_lsb(c17 & mask);
    if(unlikely(!lsb17)) {
        return -1; // singular
    }

    uint64_t c17_reduc = c17 ^ lsb17;
    reduc = _mm256_set1_epi64x(c17_reduc);
    vmask = _mm256_set1_epi64x(lsb17);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    if(c16 & lsb17) c16 ^= c17_reduc;
    uint64_t c12 = _mm256_extract_epi64(c12_15, 0);
    uint64_t c13 = _mm256_extract_epi64(c12_15, 1);
    uint64_t c14 = _mm256_extract_epi64(c12_15, 2);
    uint64_t c15 = _mm256_extract_epi64(c12_15, 3);
    mask ^= lsb17;

    uint64_t lsb16 = uint64_t_lsb(c16 & mask);
    if(unlikely(!lsb16)) {
        return -1; // singular
    }

    uint64_t c16_reduc = c16 ^ lsb16;
    reduc = _mm256_set1_epi64x(c16_reduc);
    vmask = _mm256_set1_epi64x(lsb16);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    if(c12 & lsb16) c12 ^= c16_reduc;
    if(c13 & lsb16) c13 ^= c16_reduc;
    if(c14 & lsb16) c14 ^= c16_reduc;
    if(c15 & lsb16) c15 ^= c16_reduc;
    mask ^= lsb16;

    uint64_t lsb15 = uint64_t_lsb(c15 & mask);
    if(unlikely(!lsb15)) {
        return -1; // singular
    }

    uint64_t c15_reduc = c15 ^ lsb15;
    reduc = _mm256_set1_epi64x(c15_reduc);
    vmask = _mm256_set1_epi64x(lsb15);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    if(c12 & lsb15) c12 ^= c15_reduc;
    if(c13 & lsb15) c13 ^= c15_reduc;
    if(c14 & lsb15) c14 ^= c15_reduc;
    mask ^= lsb15;

    uint64_t lsb14 = uint64_t_lsb(c14 & mask);
    if(unlikely(!lsb14)) {
        return -1; // singular
    }

    uint64_t c14_reduc = c14 ^ lsb14;
    reduc = _mm256_set1_epi64x(c14_reduc);
    vmask = _mm256_set1_epi64x(lsb14);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    if(c12 & lsb14) c12 ^= c14_reduc;
    if(c13 & lsb14) c13 ^= c14_reduc;
    mask ^= lsb14;

    uint64_t lsb13 = uint64_t_lsb(c13 & mask);
    if(unlikely(!lsb13)) {
        return -1; // singular
    }

    uint64_t c13_reduc = c13 ^ lsb13;
    reduc = _mm256_set1_epi64x(c13_reduc);
    vmask = _mm256_set1_epi64x(lsb13);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    if(c12 & lsb13) c12 ^= c13_reduc;
    uint64_t c8 = _mm256_extract_epi64(c8_11, 0);
    uint64_t c9 = _mm256_extract_epi64(c8_11, 1);
    uint64_t c10 = _mm256_extract_epi64(c8_11, 2);
    uint64_t c11 = _mm256_extract_epi64(c8_11, 3);
    mask ^= lsb13;

    uint64_t lsb12 = uint64_t_lsb(c12 & mask);
    if(unlikely(!lsb12)) {
        return -1; // singular
    }

    uint64_t c12_reduc = c12 ^ lsb12;
    reduc = _mm256_set1_epi64x(c12_reduc);
    vmask = _mm256_set1_epi64x(lsb12);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb12) c8 ^= c12_reduc;
    if(c9 & lsb12) c9 ^= c12_reduc;
    if(c10 & lsb12) c10 ^= c12_reduc;
    if(c11 & lsb12) c11 ^= c12_reduc;
    mask ^= lsb12;

    uint64_t lsb11 = uint64_t_lsb(c11 & mask);
    if(unlikely(!lsb11)) {
        return -1; // singular
    }

    uint64_t c11_reduc = c11 ^ lsb11;
    reduc = _mm256_set1_epi64x(c11_reduc);
    vmask = _mm256_set1_epi64x(lsb11);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb11) c8 ^= c11_reduc;
    if(c9 & lsb11) c9 ^= c11_reduc;
    if(c10 & lsb11) c10 ^= c11_reduc;
    mask ^= lsb11;

    uint64_t lsb10 = uint64_t_lsb(c10 & mask);
    if(unlikely(!lsb10)) {
        return -1; // singular
    }

    uint64_t c10_reduc = c10 ^ lsb10;
    reduc = _mm256_set1_epi64x(c10_reduc);
    vmask = _mm256_set1_epi64x(lsb10);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb10) c8 ^= c10_reduc;
    if(c9 & lsb10) c9 ^= c10_reduc;
    mask ^= lsb10;

    uint64_t lsb9 = uint64_t_lsb(c9 & mask);
    if(unlikely(!lsb9)) {
        return -1; // singular
    }

    uint64_t c9_reduc = c9 ^ lsb9;
    reduc = _mm256_set1_epi64x(c9_reduc);
    vmask = _mm256_set1_epi64x(lsb9);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb9) c8 ^= c9_reduc;
    uint64_t c4 = _mm256_extract_epi64(c4_7, 0);
    uint64_t c5 = _mm256_extract_epi64(c4_7, 1);
    uint64_t c6 = _mm256_extract_epi64(c4_7, 2);
    uint64_t c7 = _mm256_extract_epi64(c4_7, 3);
    mask ^= lsb9;

    uint64_t lsb8 = uint64_t_lsb(c8 & mask);
    if(unlikely(!lsb8)) {
        return -1; // singular
    }

    uint64_t c8_reduc = c8 ^ lsb8;
    reduc = _mm256_set1_epi64x(c8_reduc);
    vmask = _mm256_set1_epi64x(lsb8);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb8) c4 ^= c8_reduc;
    if(c5 & lsb8) c5 ^= c8_reduc;
    if(c6 & lsb8) c6 ^= c8_reduc;
    if(c7 & lsb8) c7 ^= c8_reduc;
    mask ^= lsb8;

    uint64_t lsb7 = uint64_t_lsb(c7 & mask);
    if(unlikely(!lsb7)) {
        return -1; // singular
    }

    uint64_t c7_reduc = c7 ^ lsb7;
    reduc = _mm256_set1_epi64x(c7_reduc);
    vmask = _mm256_set1_epi64x(lsb7);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb7) c4 ^= c7_reduc;
    if(c5 & lsb7) c5 ^= c7_reduc;
    if(c6 & lsb7) c6 ^= c7_reduc;
    mask ^= lsb7;

    uint64_t lsb6 = uint64_t_lsb(c6 & mask);
    if(unlikely(!lsb6)) {
        return -1; // singular
    }

    uint64_t c6_reduc = c6 ^ lsb6;
    reduc = _mm256_set1_epi64x(c6_reduc);
    vmask = _mm256_set1_epi64x(lsb6);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb6) c4 ^= c6_reduc;
    if(c5 & lsb6) c5 ^= c6_reduc;
    mask ^= lsb6;

    uint64_t lsb5 = uint64_t_lsb(c5 & mask);
    if(unlikely(!lsb5)) {
        return -1; // singular
    }

    uint64_t c5_reduc = c5 ^ lsb5;
    reduc = _mm256_set1_epi64x(c5_reduc);
    vmask = _mm256_set1_epi64x(lsb5);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb5) c4 ^= c5_reduc;
    uint64_t c0 = _mm256_extract_epi64(c0_3, 0);
    uint64_t c1 = _mm256_extract_epi64(c0_3, 1);
    uint64_t c2 = _mm256_extract_epi64(c0_3, 2);
    uint64_t c3 = _mm256_extract_epi64(c0_3, 3);
    mask ^= lsb5;

    uint64_t lsb4 = uint64_t_lsb(c4 & mask);
    if(unlikely(!lsb4)) {
        return -1; // singular
    }

    uint64_t c4_reduc = c4 ^ lsb4;
    if(c0 & lsb4) c0 ^= c4_reduc;
    if(c1 & lsb4) c1 ^= c4_reduc;
    if(c2 & lsb4) c2 ^= c4_reduc;
    if(c3 & lsb4) c3 ^= c4_reduc;
    mask ^= lsb4;

    uint64_t lsb3 = uint64_t_lsb(c3 & mask);
    if(unlikely(!lsb3)) {
        return -1; // singular
    }

    uint64_t c3_reduc = c3 ^ lsb3;
    if(c0 & lsb3) c0 ^= c3_reduc;
    if(c1 & lsb3) c1 ^= c3_reduc;
    if(c2 & lsb3) c2 ^= c3_reduc;
    mask ^= lsb3;

    uint64_t lsb2 = uint64_t_lsb(c2 & mask);
    if(unlikely(!lsb2)) {
        return -1; // singular
    }

    uint64_t c2_reduc = c2 ^ lsb2;
    if(c0 & lsb2) c0 ^= c2_reduc;
    if(c1 & lsb2) c1 ^= c2_reduc;
    mask ^= lsb2;

    uint64_t lsb1 = uint64_t_lsb(c1 & mask);
    if(unlikely(!lsb1)) {
        return -1; // singular
    }

    if(c0 & lsb1) c0 ^= c1 ^ lsb1;
    mask ^= lsb1;

    if(likely(mask & c0)) { // check system consistency
        return mask & c0; // not solvable
    }

    uint64_t s = 0x0ULL;
    if(c0 & lsb1) s = uint64_t_toggle_at(s, 0);
    if(c0 & lsb2) s = uint64_t_toggle_at(s, 1);
    if(c0 & lsb3) s = uint64_t_toggle_at(s, 2);
    if(c0 & lsb4) s = uint64_t_toggle_at(s, 3);
    if(c0 & lsb5) s = uint64_t_toggle_at(s, 4);
    if(c0 & lsb6) s = uint64_t_toggle_at(s, 5);
    if(c0 & lsb7) s = uint64_t_toggle_at(s, 6);
    if(c0 & lsb8) s = uint64_t_toggle_at(s, 7);
    if(c0 & lsb9) s = uint64_t_toggle_at(s, 8);
    if(c0 & lsb10) s = uint64_t_toggle_at(s, 9);
    if(c0 & lsb11) s = uint64_t_toggle_at(s, 10);
    if(c0 & lsb12) s = uint64_t_toggle_at(s, 11);
    if(c0 & lsb13) s = uint64_t_toggle_at(s, 12);
    if(c0 & lsb14) s = uint64_t_toggle_at(s, 13);
    if(c0 & lsb15) s = uint64_t_toggle_at(s, 14);
    if(c0 & lsb16) s = uint64_t_toggle_at(s, 15);
    if(c0 & lsb17) s = uint64_t_toggle_at(s, 16);
    if(c0 & lsb18) s = uint64_t_toggle_at(s, 17);
    if(c0 & lsb19) s = uint64_t_toggle_at(s, 18);
    if(c0 & lsb20) s = uint64_t_toggle_at(s, 19);
    if(c0 & lsb21) s = uint64_t_toggle_at(s, 20);
    if(c0 & lsb22) s = uint64_t_toggle_at(s, 21);
    if(c0 & lsb23) s = uint64_t_toggle_at(s, 22);
    if(c0 & lsb24) s = uint64_t_toggle_at(s, 23);
    if(c0 & lsb25) s = uint64_t_toggle_at(s, 24);
    if(c0 & lsb26) s = uint64_t_toggle_at(s, 25);
    *sol = s;
    return 0;
}

int64_t
uint64a_gj_v27_avx(const uint64_t m[28], uint64_t* const restrict sol) {
    __m256i c0_3 = _mm256_loadu_si256((__m256i*) (m + 0));
    __m256i c4_7 = _mm256_loadu_si256((__m256i*) (m + 4));
    __m256i c8_11 = _mm256_loadu_si256((__m256i*) (m + 8));
    __m256i c12_15 = _mm256_loadu_si256((__m256i*) (m + 12));
    __m256i c16_19 = _mm256_loadu_si256((__m256i*) (m + 16));
    __m256i c20_23 = _mm256_loadu_si256((__m256i*) (m + 20));
    uint64_t c24 = m[24];
    uint64_t c25 = m[25];
    uint64_t c26 = m[26];
    uint64_t c27 = m[27];

    uint64_t lsb27 = uint64_t_lsb(c27);
    if(unlikely(!lsb27)) {
        return -1; // singular
    }

    uint64_t c27_reduc = c27 ^ lsb27;
    __m256i reduc = _mm256_set1_epi64x(c27_reduc);
    __m256i vmask = _mm256_set1_epi64x(lsb27);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    c20_23 = uint64a_gj_reduc_mm256(c20_23, vmask, reduc);
    if(c24 & lsb27) c24 ^= c27_reduc;
    if(c25 & lsb27) c25 ^= c27_reduc;
    if(c26 & lsb27) c26 ^= c27_reduc;
    uint64_t mask = ~lsb27;

    uint64_t lsb26 = uint64_t_lsb(c26 & mask);
    if(unlikely(!lsb26)) {
        return -1; // singular
    }

    uint64_t c26_reduc = c26 ^ lsb26;
    reduc = _mm256_set1_epi64x(c26_reduc);
    vmask = _mm256_set1_epi64x(lsb26);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    c20_23 = uint64a_gj_reduc_mm256(c20_23, vmask, reduc);
    if(c24 & lsb26) c24 ^= c26_reduc;
    if(c25 & lsb26) c25 ^= c26_reduc;
    mask ^= lsb26;

    uint64_t lsb25 = uint64_t_lsb(c25 & mask);
    if(unlikely(!lsb25)) {
        return -1; // singular
    }

    uint64_t c25_reduc = c25 ^ lsb25;
    reduc = _mm256_set1_epi64x(c25_reduc);
    vmask = _mm256_set1_epi64x(lsb25);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    c20_23 = uint64a_gj_reduc_mm256(c20_23, vmask, reduc);
    if(c24 & lsb25) c24 ^= c25_reduc;
    uint64_t c20 = _mm256_extract_epi64(c20_23, 0);
    uint64_t c21 = _mm256_extract_epi64(c20_23, 1);
    uint64_t c22 = _mm256_extract_epi64(c20_23, 2);
    uint64_t c23 = _mm256_extract_epi64(c20_23, 3);
    mask ^= lsb25;

    uint64_t lsb24 = uint64_t_lsb(c24 & mask);
    if(unlikely(!lsb24)) {
        return -1; // singular
    }

    uint64_t c24_reduc = c24 ^ lsb24;
    reduc = _mm256_set1_epi64x(c24_reduc);
    vmask = _mm256_set1_epi64x(lsb24);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    if(c20 & lsb24) c20 ^= c24_reduc;
    if(c21 & lsb24) c21 ^= c24_reduc;
    if(c22 & lsb24) c22 ^= c24_reduc;
    if(c23 & lsb24) c23 ^= c24_reduc;
    mask ^= lsb24;

    uint64_t lsb23 = uint64_t_lsb(c23 & mask);
    if(unlikely(!lsb23)) {
        return -1; // singular
    }

    uint64_t c23_reduc = c23 ^ lsb23;
    reduc = _mm256_set1_epi64x(c23_reduc);
    vmask = _mm256_set1_epi64x(lsb23);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    if(c20 & lsb23) c20 ^= c23_reduc;
    if(c21 & lsb23) c21 ^= c23_reduc;
    if(c22 & lsb23) c22 ^= c23_reduc;
    mask ^= lsb23;

    uint64_t lsb22 = uint64_t_lsb(c22 & mask);
    if(unlikely(!lsb22)) {
        return -1; // singular
    }

    uint64_t c22_reduc = c22 ^ lsb22;
    reduc = _mm256_set1_epi64x(c22_reduc);
    vmask = _mm256_set1_epi64x(lsb22);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    if(c20 & lsb22) c20 ^= c22_reduc;
    if(c21 & lsb22) c21 ^= c22_reduc;
    mask ^= lsb22;

    uint64_t lsb21 = uint64_t_lsb(c21 & mask);
    if(unlikely(!lsb21)) {
        return -1; // singular
    }

    uint64_t c21_reduc = c21 ^ lsb21;
    reduc = _mm256_set1_epi64x(c21_reduc);
    vmask = _mm256_set1_epi64x(lsb21);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    if(c20 & lsb21) c20 ^= c21_reduc;
    uint64_t c16 = _mm256_extract_epi64(c16_19, 0);
    uint64_t c17 = _mm256_extract_epi64(c16_19, 1);
    uint64_t c18 = _mm256_extract_epi64(c16_19, 2);
    uint64_t c19 = _mm256_extract_epi64(c16_19, 3);
    mask ^= lsb21;

    uint64_t lsb20 = uint64_t_lsb(c20 & mask);
    if(unlikely(!lsb20)) {
        return -1; // singular
    }

    uint64_t c20_reduc = c20 ^ lsb20;
    reduc = _mm256_set1_epi64x(c20_reduc);
    vmask = _mm256_set1_epi64x(lsb20);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    if(c16 & lsb20) c16 ^= c20_reduc;
    if(c17 & lsb20) c17 ^= c20_reduc;
    if(c18 & lsb20) c18 ^= c20_reduc;
    if(c19 & lsb20) c19 ^= c20_reduc;
    mask ^= lsb20;

    uint64_t lsb19 = uint64_t_lsb(c19 & mask);
    if(unlikely(!lsb19)) {
        return -1; // singular
    }

    uint64_t c19_reduc = c19 ^ lsb19;
    reduc = _mm256_set1_epi64x(c19_reduc);
    vmask = _mm256_set1_epi64x(lsb19);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    if(c16 & lsb19) c16 ^= c19_reduc;
    if(c17 & lsb19) c17 ^= c19_reduc;
    if(c18 & lsb19) c18 ^= c19_reduc;
    mask ^= lsb19;

    uint64_t lsb18 = uint64_t_lsb(c18 & mask);
    if(unlikely(!lsb18)) {
        return -1; // singular
    }

    uint64_t c18_reduc = c18 ^ lsb18;
    reduc = _mm256_set1_epi64x(c18_reduc);
    vmask = _mm256_set1_epi64x(lsb18);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    if(c16 & lsb18) c16 ^= c18_reduc;
    if(c17 & lsb18) c17 ^= c18_reduc;
    mask ^= lsb18;

    uint64_t lsb17 = uint64_t_lsb(c17 & mask);
    if(unlikely(!lsb17)) {
        return -1; // singular
    }

    uint64_t c17_reduc = c17 ^ lsb17;
    reduc = _mm256_set1_epi64x(c17_reduc);
    vmask = _mm256_set1_epi64x(lsb17);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    if(c16 & lsb17) c16 ^= c17_reduc;
    uint64_t c12 = _mm256_extract_epi64(c12_15, 0);
    uint64_t c13 = _mm256_extract_epi64(c12_15, 1);
    uint64_t c14 = _mm256_extract_epi64(c12_15, 2);
    uint64_t c15 = _mm256_extract_epi64(c12_15, 3);
    mask ^= lsb17;

    uint64_t lsb16 = uint64_t_lsb(c16 & mask);
    if(unlikely(!lsb16)) {
        return -1; // singular
    }

    uint64_t c16_reduc = c16 ^ lsb16;
    reduc = _mm256_set1_epi64x(c16_reduc);
    vmask = _mm256_set1_epi64x(lsb16);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    if(c12 & lsb16) c12 ^= c16_reduc;
    if(c13 & lsb16) c13 ^= c16_reduc;
    if(c14 & lsb16) c14 ^= c16_reduc;
    if(c15 & lsb16) c15 ^= c16_reduc;
    mask ^= lsb16;

    uint64_t lsb15 = uint64_t_lsb(c15 & mask);
    if(unlikely(!lsb15)) {
        return -1; // singular
    }

    uint64_t c15_reduc = c15 ^ lsb15;
    reduc = _mm256_set1_epi64x(c15_reduc);
    vmask = _mm256_set1_epi64x(lsb15);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    if(c12 & lsb15) c12 ^= c15_reduc;
    if(c13 & lsb15) c13 ^= c15_reduc;
    if(c14 & lsb15) c14 ^= c15_reduc;
    mask ^= lsb15;

    uint64_t lsb14 = uint64_t_lsb(c14 & mask);
    if(unlikely(!lsb14)) {
        return -1; // singular
    }

    uint64_t c14_reduc = c14 ^ lsb14;
    reduc = _mm256_set1_epi64x(c14_reduc);
    vmask = _mm256_set1_epi64x(lsb14);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    if(c12 & lsb14) c12 ^= c14_reduc;
    if(c13 & lsb14) c13 ^= c14_reduc;
    mask ^= lsb14;

    uint64_t lsb13 = uint64_t_lsb(c13 & mask);
    if(unlikely(!lsb13)) {
        return -1; // singular
    }

    uint64_t c13_reduc = c13 ^ lsb13;
    reduc = _mm256_set1_epi64x(c13_reduc);
    vmask = _mm256_set1_epi64x(lsb13);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    if(c12 & lsb13) c12 ^= c13_reduc;
    uint64_t c8 = _mm256_extract_epi64(c8_11, 0);
    uint64_t c9 = _mm256_extract_epi64(c8_11, 1);
    uint64_t c10 = _mm256_extract_epi64(c8_11, 2);
    uint64_t c11 = _mm256_extract_epi64(c8_11, 3);
    mask ^= lsb13;

    uint64_t lsb12 = uint64_t_lsb(c12 & mask);
    if(unlikely(!lsb12)) {
        return -1; // singular
    }

    uint64_t c12_reduc = c12 ^ lsb12;
    reduc = _mm256_set1_epi64x(c12_reduc);
    vmask = _mm256_set1_epi64x(lsb12);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb12) c8 ^= c12_reduc;
    if(c9 & lsb12) c9 ^= c12_reduc;
    if(c10 & lsb12) c10 ^= c12_reduc;
    if(c11 & lsb12) c11 ^= c12_reduc;
    mask ^= lsb12;

    uint64_t lsb11 = uint64_t_lsb(c11 & mask);
    if(unlikely(!lsb11)) {
        return -1; // singular
    }

    uint64_t c11_reduc = c11 ^ lsb11;
    reduc = _mm256_set1_epi64x(c11_reduc);
    vmask = _mm256_set1_epi64x(lsb11);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb11) c8 ^= c11_reduc;
    if(c9 & lsb11) c9 ^= c11_reduc;
    if(c10 & lsb11) c10 ^= c11_reduc;
    mask ^= lsb11;

    uint64_t lsb10 = uint64_t_lsb(c10 & mask);
    if(unlikely(!lsb10)) {
        return -1; // singular
    }

    uint64_t c10_reduc = c10 ^ lsb10;
    reduc = _mm256_set1_epi64x(c10_reduc);
    vmask = _mm256_set1_epi64x(lsb10);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb10) c8 ^= c10_reduc;
    if(c9 & lsb10) c9 ^= c10_reduc;
    mask ^= lsb10;

    uint64_t lsb9 = uint64_t_lsb(c9 & mask);
    if(unlikely(!lsb9)) {
        return -1; // singular
    }

    uint64_t c9_reduc = c9 ^ lsb9;
    reduc = _mm256_set1_epi64x(c9_reduc);
    vmask = _mm256_set1_epi64x(lsb9);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb9) c8 ^= c9_reduc;
    uint64_t c4 = _mm256_extract_epi64(c4_7, 0);
    uint64_t c5 = _mm256_extract_epi64(c4_7, 1);
    uint64_t c6 = _mm256_extract_epi64(c4_7, 2);
    uint64_t c7 = _mm256_extract_epi64(c4_7, 3);
    mask ^= lsb9;

    uint64_t lsb8 = uint64_t_lsb(c8 & mask);
    if(unlikely(!lsb8)) {
        return -1; // singular
    }

    uint64_t c8_reduc = c8 ^ lsb8;
    reduc = _mm256_set1_epi64x(c8_reduc);
    vmask = _mm256_set1_epi64x(lsb8);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb8) c4 ^= c8_reduc;
    if(c5 & lsb8) c5 ^= c8_reduc;
    if(c6 & lsb8) c6 ^= c8_reduc;
    if(c7 & lsb8) c7 ^= c8_reduc;
    mask ^= lsb8;

    uint64_t lsb7 = uint64_t_lsb(c7 & mask);
    if(unlikely(!lsb7)) {
        return -1; // singular
    }

    uint64_t c7_reduc = c7 ^ lsb7;
    reduc = _mm256_set1_epi64x(c7_reduc);
    vmask = _mm256_set1_epi64x(lsb7);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb7) c4 ^= c7_reduc;
    if(c5 & lsb7) c5 ^= c7_reduc;
    if(c6 & lsb7) c6 ^= c7_reduc;
    mask ^= lsb7;

    uint64_t lsb6 = uint64_t_lsb(c6 & mask);
    if(unlikely(!lsb6)) {
        return -1; // singular
    }

    uint64_t c6_reduc = c6 ^ lsb6;
    reduc = _mm256_set1_epi64x(c6_reduc);
    vmask = _mm256_set1_epi64x(lsb6);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb6) c4 ^= c6_reduc;
    if(c5 & lsb6) c5 ^= c6_reduc;
    mask ^= lsb6;

    uint64_t lsb5 = uint64_t_lsb(c5 & mask);
    if(unlikely(!lsb5)) {
        return -1; // singular
    }

    uint64_t c5_reduc = c5 ^ lsb5;
    reduc = _mm256_set1_epi64x(c5_reduc);
    vmask = _mm256_set1_epi64x(lsb5);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb5) c4 ^= c5_reduc;
    uint64_t c0 = _mm256_extract_epi64(c0_3, 0);
    uint64_t c1 = _mm256_extract_epi64(c0_3, 1);
    uint64_t c2 = _mm256_extract_epi64(c0_3, 2);
    uint64_t c3 = _mm256_extract_epi64(c0_3, 3);
    mask ^= lsb5;

    uint64_t lsb4 = uint64_t_lsb(c4 & mask);
    if(unlikely(!lsb4)) {
        return -1; // singular
    }

    uint64_t c4_reduc = c4 ^ lsb4;
    if(c0 & lsb4) c0 ^= c4_reduc;
    if(c1 & lsb4) c1 ^= c4_reduc;
    if(c2 & lsb4) c2 ^= c4_reduc;
    if(c3 & lsb4) c3 ^= c4_reduc;
    mask ^= lsb4;

    uint64_t lsb3 = uint64_t_lsb(c3 & mask);
    if(unlikely(!lsb3)) {
        return -1; // singular
    }

    uint64_t c3_reduc = c3 ^ lsb3;
    if(c0 & lsb3) c0 ^= c3_reduc;
    if(c1 & lsb3) c1 ^= c3_reduc;
    if(c2 & lsb3) c2 ^= c3_reduc;
    mask ^= lsb3;

    uint64_t lsb2 = uint64_t_lsb(c2 & mask);
    if(unlikely(!lsb2)) {
        return -1; // singular
    }

    uint64_t c2_reduc = c2 ^ lsb2;
    if(c0 & lsb2) c0 ^= c2_reduc;
    if(c1 & lsb2) c1 ^= c2_reduc;
    mask ^= lsb2;

    uint64_t lsb1 = uint64_t_lsb(c1 & mask);
    if(unlikely(!lsb1)) {
        return -1; // singular
    }

    if(c0 & lsb1) c0 ^= c1 ^ lsb1;
    mask ^= lsb1;

    if(likely(mask & c0)) { // check system consistency
        return mask & c0; // not solvable
    }

    uint64_t s = 0x0ULL;
    if(c0 & lsb1) s = uint64_t_toggle_at(s, 0);
    if(c0 & lsb2) s = uint64_t_toggle_at(s, 1);
    if(c0 & lsb3) s = uint64_t_toggle_at(s, 2);
    if(c0 & lsb4) s = uint64_t_toggle_at(s, 3);
    if(c0 & lsb5) s = uint64_t_toggle_at(s, 4);
    if(c0 & lsb6) s = uint64_t_toggle_at(s, 5);
    if(c0 & lsb7) s = uint64_t_toggle_at(s, 6);
    if(c0 & lsb8) s = uint64_t_toggle_at(s, 7);
    if(c0 & lsb9) s = uint64_t_toggle_at(s, 8);
    if(c0 & lsb10) s = uint64_t_toggle_at(s, 9);
    if(c0 & lsb11) s = uint64_t_toggle_at(s, 10);
    if(c0 & lsb12) s = uint64_t_toggle_at(s, 11);
    if(c0 & lsb13) s = uint64_t_toggle_at(s, 12);
    if(c0 & lsb14) s = uint64_t_toggle_at(s, 13);
    if(c0 & lsb15) s = uint64_t_toggle_at(s, 14);
    if(c0 & lsb16) s = uint64_t_toggle_at(s, 15);
    if(c0 & lsb17) s = uint64_t_toggle_at(s, 16);
    if(c0 & lsb18) s = uint64_t_toggle_at(s, 17);
    if(c0 & lsb19) s = uint64_t_toggle_at(s, 18);
    if(c0 & lsb20) s = uint64_t_toggle_at(s, 19);
    if(c0 & lsb21) s = uint64_t_toggle_at(s, 20);
    if(c0 & lsb22) s = uint64_t_toggle_at(s, 21);
    if(c0 & lsb23) s = uint64_t_toggle_at(s, 22);
    if(c0 & lsb24) s = uint64_t_toggle_at(s, 23);
    if(c0 & lsb25) s = uint64_t_toggle_at(s, 24);
    if(c0 & lsb26) s = uint64_t_toggle_at(s, 25);
    if(c0 & lsb27) s = uint64_t_toggle_at(s, 26);
    *sol = s;
    return 0;
}

int64_t
uint64a_gj_v28_avx(const uint64_t m[29], uint64_t* const restrict sol) {
    __m256i c0_3 = _mm256_loadu_si256((__m256i*) (m + 0));
    __m256i c4_7 = _mm256_loadu_si256((__m256i*) (m + 4));
    __m256i c8_11 = _mm256_loadu_si256((__m256i*) (m + 8));
    __m256i c12_15 = _mm256_loadu_si256((__m256i*) (m + 12));
    __m256i c16_19 = _mm256_loadu_si256((__m256i*) (m + 16));
    __m256i c20_23 = _mm256_loadu_si256((__m256i*) (m + 20));
    uint64_t c24 = m[24];
    uint64_t c25 = m[25];
    uint64_t c26 = m[26];
    uint64_t c27 = m[27];
    uint64_t c28 = m[28];

    uint64_t lsb28 = uint64_t_lsb(c28);
    if(unlikely(!lsb28)) {
        return -1; // singular
    }

    uint64_t c28_reduc = c28 ^ lsb28;
    __m256i reduc = _mm256_set1_epi64x(c28_reduc);
    __m256i vmask = _mm256_set1_epi64x(lsb28);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    c20_23 = uint64a_gj_reduc_mm256(c20_23, vmask, reduc);
    if(c24 & lsb28) c24 ^= c28_reduc;
    if(c25 & lsb28) c25 ^= c28_reduc;
    if(c26 & lsb28) c26 ^= c28_reduc;
    if(c27 & lsb28) c27 ^= c28_reduc;
    uint64_t mask = ~lsb28;

    uint64_t lsb27 = uint64_t_lsb(c27 & mask);
    if(unlikely(!lsb27)) {
        return -1; // singular
    }

    uint64_t c27_reduc = c27 ^ lsb27;
    reduc = _mm256_set1_epi64x(c27_reduc);
    vmask = _mm256_set1_epi64x(lsb27);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    c20_23 = uint64a_gj_reduc_mm256(c20_23, vmask, reduc);
    if(c24 & lsb27) c24 ^= c27_reduc;
    if(c25 & lsb27) c25 ^= c27_reduc;
    if(c26 & lsb27) c26 ^= c27_reduc;
    mask ^= lsb27;

    uint64_t lsb26 = uint64_t_lsb(c26 & mask);
    if(unlikely(!lsb26)) {
        return -1; // singular
    }

    uint64_t c26_reduc = c26 ^ lsb26;
    reduc = _mm256_set1_epi64x(c26_reduc);
    vmask = _mm256_set1_epi64x(lsb26);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    c20_23 = uint64a_gj_reduc_mm256(c20_23, vmask, reduc);
    if(c24 & lsb26) c24 ^= c26_reduc;
    if(c25 & lsb26) c25 ^= c26_reduc;
    mask ^= lsb26;

    uint64_t lsb25 = uint64_t_lsb(c25 & mask);
    if(unlikely(!lsb25)) {
        return -1; // singular
    }

    uint64_t c25_reduc = c25 ^ lsb25;
    reduc = _mm256_set1_epi64x(c25_reduc);
    vmask = _mm256_set1_epi64x(lsb25);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    c20_23 = uint64a_gj_reduc_mm256(c20_23, vmask, reduc);
    if(c24 & lsb25) c24 ^= c25_reduc;
    uint64_t c20 = _mm256_extract_epi64(c20_23, 0);
    uint64_t c21 = _mm256_extract_epi64(c20_23, 1);
    uint64_t c22 = _mm256_extract_epi64(c20_23, 2);
    uint64_t c23 = _mm256_extract_epi64(c20_23, 3);
    mask ^= lsb25;

    uint64_t lsb24 = uint64_t_lsb(c24 & mask);
    if(unlikely(!lsb24)) {
        return -1; // singular
    }

    uint64_t c24_reduc = c24 ^ lsb24;
    reduc = _mm256_set1_epi64x(c24_reduc);
    vmask = _mm256_set1_epi64x(lsb24);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    if(c20 & lsb24) c20 ^= c24_reduc;
    if(c21 & lsb24) c21 ^= c24_reduc;
    if(c22 & lsb24) c22 ^= c24_reduc;
    if(c23 & lsb24) c23 ^= c24_reduc;
    mask ^= lsb24;

    uint64_t lsb23 = uint64_t_lsb(c23 & mask);
    if(unlikely(!lsb23)) {
        return -1; // singular
    }

    uint64_t c23_reduc = c23 ^ lsb23;
    reduc = _mm256_set1_epi64x(c23_reduc);
    vmask = _mm256_set1_epi64x(lsb23);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    if(c20 & lsb23) c20 ^= c23_reduc;
    if(c21 & lsb23) c21 ^= c23_reduc;
    if(c22 & lsb23) c22 ^= c23_reduc;
    mask ^= lsb23;

    uint64_t lsb22 = uint64_t_lsb(c22 & mask);
    if(unlikely(!lsb22)) {
        return -1; // singular
    }

    uint64_t c22_reduc = c22 ^ lsb22;
    reduc = _mm256_set1_epi64x(c22_reduc);
    vmask = _mm256_set1_epi64x(lsb22);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    if(c20 & lsb22) c20 ^= c22_reduc;
    if(c21 & lsb22) c21 ^= c22_reduc;
    mask ^= lsb22;

    uint64_t lsb21 = uint64_t_lsb(c21 & mask);
    if(unlikely(!lsb21)) {
        return -1; // singular
    }

    uint64_t c21_reduc = c21 ^ lsb21;
    reduc = _mm256_set1_epi64x(c21_reduc);
    vmask = _mm256_set1_epi64x(lsb21);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    if(c20 & lsb21) c20 ^= c21_reduc;
    uint64_t c16 = _mm256_extract_epi64(c16_19, 0);
    uint64_t c17 = _mm256_extract_epi64(c16_19, 1);
    uint64_t c18 = _mm256_extract_epi64(c16_19, 2);
    uint64_t c19 = _mm256_extract_epi64(c16_19, 3);
    mask ^= lsb21;

    uint64_t lsb20 = uint64_t_lsb(c20 & mask);
    if(unlikely(!lsb20)) {
        return -1; // singular
    }

    uint64_t c20_reduc = c20 ^ lsb20;
    reduc = _mm256_set1_epi64x(c20_reduc);
    vmask = _mm256_set1_epi64x(lsb20);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    if(c16 & lsb20) c16 ^= c20_reduc;
    if(c17 & lsb20) c17 ^= c20_reduc;
    if(c18 & lsb20) c18 ^= c20_reduc;
    if(c19 & lsb20) c19 ^= c20_reduc;
    mask ^= lsb20;

    uint64_t lsb19 = uint64_t_lsb(c19 & mask);
    if(unlikely(!lsb19)) {
        return -1; // singular
    }

    uint64_t c19_reduc = c19 ^ lsb19;
    reduc = _mm256_set1_epi64x(c19_reduc);
    vmask = _mm256_set1_epi64x(lsb19);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    if(c16 & lsb19) c16 ^= c19_reduc;
    if(c17 & lsb19) c17 ^= c19_reduc;
    if(c18 & lsb19) c18 ^= c19_reduc;
    mask ^= lsb19;

    uint64_t lsb18 = uint64_t_lsb(c18 & mask);
    if(unlikely(!lsb18)) {
        return -1; // singular
    }

    uint64_t c18_reduc = c18 ^ lsb18;
    reduc = _mm256_set1_epi64x(c18_reduc);
    vmask = _mm256_set1_epi64x(lsb18);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    if(c16 & lsb18) c16 ^= c18_reduc;
    if(c17 & lsb18) c17 ^= c18_reduc;
    mask ^= lsb18;

    uint64_t lsb17 = uint64_t_lsb(c17 & mask);
    if(unlikely(!lsb17)) {
        return -1; // singular
    }

    uint64_t c17_reduc = c17 ^ lsb17;
    reduc = _mm256_set1_epi64x(c17_reduc);
    vmask = _mm256_set1_epi64x(lsb17);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    if(c16 & lsb17) c16 ^= c17_reduc;
    uint64_t c12 = _mm256_extract_epi64(c12_15, 0);
    uint64_t c13 = _mm256_extract_epi64(c12_15, 1);
    uint64_t c14 = _mm256_extract_epi64(c12_15, 2);
    uint64_t c15 = _mm256_extract_epi64(c12_15, 3);
    mask ^= lsb17;

    uint64_t lsb16 = uint64_t_lsb(c16 & mask);
    if(unlikely(!lsb16)) {
        return -1; // singular
    }

    uint64_t c16_reduc = c16 ^ lsb16;
    reduc = _mm256_set1_epi64x(c16_reduc);
    vmask = _mm256_set1_epi64x(lsb16);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    if(c12 & lsb16) c12 ^= c16_reduc;
    if(c13 & lsb16) c13 ^= c16_reduc;
    if(c14 & lsb16) c14 ^= c16_reduc;
    if(c15 & lsb16) c15 ^= c16_reduc;
    mask ^= lsb16;

    uint64_t lsb15 = uint64_t_lsb(c15 & mask);
    if(unlikely(!lsb15)) {
        return -1; // singular
    }

    uint64_t c15_reduc = c15 ^ lsb15;
    reduc = _mm256_set1_epi64x(c15_reduc);
    vmask = _mm256_set1_epi64x(lsb15);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    if(c12 & lsb15) c12 ^= c15_reduc;
    if(c13 & lsb15) c13 ^= c15_reduc;
    if(c14 & lsb15) c14 ^= c15_reduc;
    mask ^= lsb15;

    uint64_t lsb14 = uint64_t_lsb(c14 & mask);
    if(unlikely(!lsb14)) {
        return -1; // singular
    }

    uint64_t c14_reduc = c14 ^ lsb14;
    reduc = _mm256_set1_epi64x(c14_reduc);
    vmask = _mm256_set1_epi64x(lsb14);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    if(c12 & lsb14) c12 ^= c14_reduc;
    if(c13 & lsb14) c13 ^= c14_reduc;
    mask ^= lsb14;

    uint64_t lsb13 = uint64_t_lsb(c13 & mask);
    if(unlikely(!lsb13)) {
        return -1; // singular
    }

    uint64_t c13_reduc = c13 ^ lsb13;
    reduc = _mm256_set1_epi64x(c13_reduc);
    vmask = _mm256_set1_epi64x(lsb13);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    if(c12 & lsb13) c12 ^= c13_reduc;
    uint64_t c8 = _mm256_extract_epi64(c8_11, 0);
    uint64_t c9 = _mm256_extract_epi64(c8_11, 1);
    uint64_t c10 = _mm256_extract_epi64(c8_11, 2);
    uint64_t c11 = _mm256_extract_epi64(c8_11, 3);
    mask ^= lsb13;

    uint64_t lsb12 = uint64_t_lsb(c12 & mask);
    if(unlikely(!lsb12)) {
        return -1; // singular
    }

    uint64_t c12_reduc = c12 ^ lsb12;
    reduc = _mm256_set1_epi64x(c12_reduc);
    vmask = _mm256_set1_epi64x(lsb12);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb12) c8 ^= c12_reduc;
    if(c9 & lsb12) c9 ^= c12_reduc;
    if(c10 & lsb12) c10 ^= c12_reduc;
    if(c11 & lsb12) c11 ^= c12_reduc;
    mask ^= lsb12;

    uint64_t lsb11 = uint64_t_lsb(c11 & mask);
    if(unlikely(!lsb11)) {
        return -1; // singular
    }

    uint64_t c11_reduc = c11 ^ lsb11;
    reduc = _mm256_set1_epi64x(c11_reduc);
    vmask = _mm256_set1_epi64x(lsb11);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb11) c8 ^= c11_reduc;
    if(c9 & lsb11) c9 ^= c11_reduc;
    if(c10 & lsb11) c10 ^= c11_reduc;
    mask ^= lsb11;

    uint64_t lsb10 = uint64_t_lsb(c10 & mask);
    if(unlikely(!lsb10)) {
        return -1; // singular
    }

    uint64_t c10_reduc = c10 ^ lsb10;
    reduc = _mm256_set1_epi64x(c10_reduc);
    vmask = _mm256_set1_epi64x(lsb10);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb10) c8 ^= c10_reduc;
    if(c9 & lsb10) c9 ^= c10_reduc;
    mask ^= lsb10;

    uint64_t lsb9 = uint64_t_lsb(c9 & mask);
    if(unlikely(!lsb9)) {
        return -1; // singular
    }

    uint64_t c9_reduc = c9 ^ lsb9;
    reduc = _mm256_set1_epi64x(c9_reduc);
    vmask = _mm256_set1_epi64x(lsb9);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb9) c8 ^= c9_reduc;
    uint64_t c4 = _mm256_extract_epi64(c4_7, 0);
    uint64_t c5 = _mm256_extract_epi64(c4_7, 1);
    uint64_t c6 = _mm256_extract_epi64(c4_7, 2);
    uint64_t c7 = _mm256_extract_epi64(c4_7, 3);
    mask ^= lsb9;

    uint64_t lsb8 = uint64_t_lsb(c8 & mask);
    if(unlikely(!lsb8)) {
        return -1; // singular
    }

    uint64_t c8_reduc = c8 ^ lsb8;
    reduc = _mm256_set1_epi64x(c8_reduc);
    vmask = _mm256_set1_epi64x(lsb8);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb8) c4 ^= c8_reduc;
    if(c5 & lsb8) c5 ^= c8_reduc;
    if(c6 & lsb8) c6 ^= c8_reduc;
    if(c7 & lsb8) c7 ^= c8_reduc;
    mask ^= lsb8;

    uint64_t lsb7 = uint64_t_lsb(c7 & mask);
    if(unlikely(!lsb7)) {
        return -1; // singular
    }

    uint64_t c7_reduc = c7 ^ lsb7;
    reduc = _mm256_set1_epi64x(c7_reduc);
    vmask = _mm256_set1_epi64x(lsb7);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb7) c4 ^= c7_reduc;
    if(c5 & lsb7) c5 ^= c7_reduc;
    if(c6 & lsb7) c6 ^= c7_reduc;
    mask ^= lsb7;

    uint64_t lsb6 = uint64_t_lsb(c6 & mask);
    if(unlikely(!lsb6)) {
        return -1; // singular
    }

    uint64_t c6_reduc = c6 ^ lsb6;
    reduc = _mm256_set1_epi64x(c6_reduc);
    vmask = _mm256_set1_epi64x(lsb6);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb6) c4 ^= c6_reduc;
    if(c5 & lsb6) c5 ^= c6_reduc;
    mask ^= lsb6;

    uint64_t lsb5 = uint64_t_lsb(c5 & mask);
    if(unlikely(!lsb5)) {
        return -1; // singular
    }

    uint64_t c5_reduc = c5 ^ lsb5;
    reduc = _mm256_set1_epi64x(c5_reduc);
    vmask = _mm256_set1_epi64x(lsb5);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb5) c4 ^= c5_reduc;
    uint64_t c0 = _mm256_extract_epi64(c0_3, 0);
    uint64_t c1 = _mm256_extract_epi64(c0_3, 1);
    uint64_t c2 = _mm256_extract_epi64(c0_3, 2);
    uint64_t c3 = _mm256_extract_epi64(c0_3, 3);
    mask ^= lsb5;

    uint64_t lsb4 = uint64_t_lsb(c4 & mask);
    if(unlikely(!lsb4)) {
        return -1; // singular
    }

    uint64_t c4_reduc = c4 ^ lsb4;
    if(c0 & lsb4) c0 ^= c4_reduc;
    if(c1 & lsb4) c1 ^= c4_reduc;
    if(c2 & lsb4) c2 ^= c4_reduc;
    if(c3 & lsb4) c3 ^= c4_reduc;
    mask ^= lsb4;

    uint64_t lsb3 = uint64_t_lsb(c3 & mask);
    if(unlikely(!lsb3)) {
        return -1; // singular
    }

    uint64_t c3_reduc = c3 ^ lsb3;
    if(c0 & lsb3) c0 ^= c3_reduc;
    if(c1 & lsb3) c1 ^= c3_reduc;
    if(c2 & lsb3) c2 ^= c3_reduc;
    mask ^= lsb3;

    uint64_t lsb2 = uint64_t_lsb(c2 & mask);
    if(unlikely(!lsb2)) {
        return -1; // singular
    }

    uint64_t c2_reduc = c2 ^ lsb2;
    if(c0 & lsb2) c0 ^= c2_reduc;
    if(c1 & lsb2) c1 ^= c2_reduc;
    mask ^= lsb2;

    uint64_t lsb1 = uint64_t_lsb(c1 & mask);
    if(unlikely(!lsb1)) {
        return -1; // singular
    }

    if(c0 & lsb1) c0 ^= c1 ^ lsb1;
    mask ^= lsb1;

    if(likely(mask & c0)) { // check system consistency
        return mask & c0; // not solvable
    }

    uint64_t s = 0x0ULL;
    if(c0 & lsb1) s = uint64_t_toggle_at(s, 0);
    if(c0 & lsb2) s = uint64_t_toggle_at(s, 1);
    if(c0 & lsb3) s = uint64_t_toggle_at(s, 2);
    if(c0 & lsb4) s = uint64_t_toggle_at(s, 3);
    if(c0 & lsb5) s = uint64_t_toggle_at(s, 4);
    if(c0 & lsb6) s = uint64_t_toggle_at(s, 5);
    if(c0 & lsb7) s = uint64_t_toggle_at(s, 6);
    if(c0 & lsb8) s = uint64_t_toggle_at(s, 7);
    if(c0 & lsb9) s = uint64_t_toggle_at(s, 8);
    if(c0 & lsb10) s = uint64_t_toggle_at(s, 9);
    if(c0 & lsb11) s = uint64_t_toggle_at(s, 10);
    if(c0 & lsb12) s = uint64_t_toggle_at(s, 11);
    if(c0 & lsb13) s = uint64_t_toggle_at(s, 12);
    if(c0 & lsb14) s = uint64_t_toggle_at(s, 13);
    if(c0 & lsb15) s = uint64_t_toggle_at(s, 14);
    if(c0 & lsb16) s = uint64_t_toggle_at(s, 15);
    if(c0 & lsb17) s = uint64_t_toggle_at(s, 16);
    if(c0 & lsb18) s = uint64_t_toggle_at(s, 17);
    if(c0 & lsb19) s = uint64_t_toggle_at(s, 18);
    if(c0 & lsb20) s = uint64_t_toggle_at(s, 19);
    if(c0 & lsb21) s = uint64_t_toggle_at(s, 20);
    if(c0 & lsb22) s = uint64_t_toggle_at(s, 21);
    if(c0 & lsb23) s = uint64_t_toggle_at(s, 22);
    if(c0 & lsb24) s = uint64_t_toggle_at(s, 23);
    if(c0 & lsb25) s = uint64_t_toggle_at(s, 24);
    if(c0 & lsb26) s = uint64_t_toggle_at(s, 25);
    if(c0 & lsb27) s = uint64_t_toggle_at(s, 26);
    if(c0 & lsb28) s = uint64_t_toggle_at(s, 27);
    *sol = s;
    return 0;
}

int64_t
uint64a_gj_v29_avx(const uint64_t m[30], uint64_t* const restrict sol) {
    __m256i c0_3 = _mm256_loadu_si256((__m256i*) (m + 0));
    __m256i c4_7 = _mm256_loadu_si256((__m256i*) (m + 4));
    __m256i c8_11 = _mm256_loadu_si256((__m256i*) (m + 8));
    __m256i c12_15 = _mm256_loadu_si256((__m256i*) (m + 12));
    __m256i c16_19 = _mm256_loadu_si256((__m256i*) (m + 16));
    __m256i c20_23 = _mm256_loadu_si256((__m256i*) (m + 20));
    __m256i c24_27 = _mm256_loadu_si256((__m256i*) (m + 24));
    uint64_t c28 = m[28];
    uint64_t c29 = m[29];

    uint64_t lsb29 = uint64_t_lsb(c29);
    if(unlikely(!lsb29)) {
        return -1; // singular
    }

    uint64_t c29_reduc = c29 ^ lsb29;
    __m256i reduc = _mm256_set1_epi64x(c29_reduc);
    __m256i vmask = _mm256_set1_epi64x(lsb29);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    c20_23 = uint64a_gj_reduc_mm256(c20_23, vmask, reduc);
    c24_27 = uint64a_gj_reduc_mm256(c24_27, vmask, reduc);
    if(c28 & lsb29) c28 ^= c29_reduc;
    uint64_t mask = ~lsb29;

    uint64_t lsb28 = uint64_t_lsb(c28 & mask);
    if(unlikely(!lsb28)) {
        return -1; // singular
    }

    uint64_t c28_reduc = c28 ^ lsb28;
    reduc = _mm256_set1_epi64x(c28_reduc);
    vmask = _mm256_set1_epi64x(lsb28);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    c20_23 = uint64a_gj_reduc_mm256(c20_23, vmask, reduc);
    c24_27 = uint64a_gj_reduc_mm256(c24_27, vmask, reduc);
    uint64_t c24 = _mm256_extract_epi64(c24_27, 0);
    uint64_t c25 = _mm256_extract_epi64(c24_27, 1);
    uint64_t c26 = _mm256_extract_epi64(c24_27, 2);
    uint64_t c27 = _mm256_extract_epi64(c24_27, 3);
    mask ^= lsb28;

    uint64_t lsb27 = uint64_t_lsb(c27 & mask);
    if(unlikely(!lsb27)) {
        return -1; // singular
    }

    uint64_t c27_reduc = c27 ^ lsb27;
    reduc = _mm256_set1_epi64x(c27_reduc);
    vmask = _mm256_set1_epi64x(lsb27);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    c20_23 = uint64a_gj_reduc_mm256(c20_23, vmask, reduc);
    if(c24 & lsb27) c24 ^= c27_reduc;
    if(c25 & lsb27) c25 ^= c27_reduc;
    if(c26 & lsb27) c26 ^= c27_reduc;
    mask ^= lsb27;

    uint64_t lsb26 = uint64_t_lsb(c26 & mask);
    if(unlikely(!lsb26)) {
        return -1; // singular
    }

    uint64_t c26_reduc = c26 ^ lsb26;
    reduc = _mm256_set1_epi64x(c26_reduc);
    vmask = _mm256_set1_epi64x(lsb26);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    c20_23 = uint64a_gj_reduc_mm256(c20_23, vmask, reduc);
    if(c24 & lsb26) c24 ^= c26_reduc;
    if(c25 & lsb26) c25 ^= c26_reduc;
    mask ^= lsb26;

    uint64_t lsb25 = uint64_t_lsb(c25 & mask);
    if(unlikely(!lsb25)) {
        return -1; // singular
    }

    uint64_t c25_reduc = c25 ^ lsb25;
    reduc = _mm256_set1_epi64x(c25_reduc);
    vmask = _mm256_set1_epi64x(lsb25);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    c20_23 = uint64a_gj_reduc_mm256(c20_23, vmask, reduc);
    if(c24 & lsb25) c24 ^= c25_reduc;
    uint64_t c20 = _mm256_extract_epi64(c20_23, 0);
    uint64_t c21 = _mm256_extract_epi64(c20_23, 1);
    uint64_t c22 = _mm256_extract_epi64(c20_23, 2);
    uint64_t c23 = _mm256_extract_epi64(c20_23, 3);
    mask ^= lsb25;

    uint64_t lsb24 = uint64_t_lsb(c24 & mask);
    if(unlikely(!lsb24)) {
        return -1; // singular
    }

    uint64_t c24_reduc = c24 ^ lsb24;
    reduc = _mm256_set1_epi64x(c24_reduc);
    vmask = _mm256_set1_epi64x(lsb24);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    if(c20 & lsb24) c20 ^= c24_reduc;
    if(c21 & lsb24) c21 ^= c24_reduc;
    if(c22 & lsb24) c22 ^= c24_reduc;
    if(c23 & lsb24) c23 ^= c24_reduc;
    mask ^= lsb24;

    uint64_t lsb23 = uint64_t_lsb(c23 & mask);
    if(unlikely(!lsb23)) {
        return -1; // singular
    }

    uint64_t c23_reduc = c23 ^ lsb23;
    reduc = _mm256_set1_epi64x(c23_reduc);
    vmask = _mm256_set1_epi64x(lsb23);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    if(c20 & lsb23) c20 ^= c23_reduc;
    if(c21 & lsb23) c21 ^= c23_reduc;
    if(c22 & lsb23) c22 ^= c23_reduc;
    mask ^= lsb23;

    uint64_t lsb22 = uint64_t_lsb(c22 & mask);
    if(unlikely(!lsb22)) {
        return -1; // singular
    }

    uint64_t c22_reduc = c22 ^ lsb22;
    reduc = _mm256_set1_epi64x(c22_reduc);
    vmask = _mm256_set1_epi64x(lsb22);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    if(c20 & lsb22) c20 ^= c22_reduc;
    if(c21 & lsb22) c21 ^= c22_reduc;
    mask ^= lsb22;

    uint64_t lsb21 = uint64_t_lsb(c21 & mask);
    if(unlikely(!lsb21)) {
        return -1; // singular
    }

    uint64_t c21_reduc = c21 ^ lsb21;
    reduc = _mm256_set1_epi64x(c21_reduc);
    vmask = _mm256_set1_epi64x(lsb21);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    if(c20 & lsb21) c20 ^= c21_reduc;
    uint64_t c16 = _mm256_extract_epi64(c16_19, 0);
    uint64_t c17 = _mm256_extract_epi64(c16_19, 1);
    uint64_t c18 = _mm256_extract_epi64(c16_19, 2);
    uint64_t c19 = _mm256_extract_epi64(c16_19, 3);
    mask ^= lsb21;

    uint64_t lsb20 = uint64_t_lsb(c20 & mask);
    if(unlikely(!lsb20)) {
        return -1; // singular
    }

    uint64_t c20_reduc = c20 ^ lsb20;
    reduc = _mm256_set1_epi64x(c20_reduc);
    vmask = _mm256_set1_epi64x(lsb20);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    if(c16 & lsb20) c16 ^= c20_reduc;
    if(c17 & lsb20) c17 ^= c20_reduc;
    if(c18 & lsb20) c18 ^= c20_reduc;
    if(c19 & lsb20) c19 ^= c20_reduc;
    mask ^= lsb20;

    uint64_t lsb19 = uint64_t_lsb(c19 & mask);
    if(unlikely(!lsb19)) {
        return -1; // singular
    }

    uint64_t c19_reduc = c19 ^ lsb19;
    reduc = _mm256_set1_epi64x(c19_reduc);
    vmask = _mm256_set1_epi64x(lsb19);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    if(c16 & lsb19) c16 ^= c19_reduc;
    if(c17 & lsb19) c17 ^= c19_reduc;
    if(c18 & lsb19) c18 ^= c19_reduc;
    mask ^= lsb19;

    uint64_t lsb18 = uint64_t_lsb(c18 & mask);
    if(unlikely(!lsb18)) {
        return -1; // singular
    }

    uint64_t c18_reduc = c18 ^ lsb18;
    reduc = _mm256_set1_epi64x(c18_reduc);
    vmask = _mm256_set1_epi64x(lsb18);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    if(c16 & lsb18) c16 ^= c18_reduc;
    if(c17 & lsb18) c17 ^= c18_reduc;
    mask ^= lsb18;

    uint64_t lsb17 = uint64_t_lsb(c17 & mask);
    if(unlikely(!lsb17)) {
        return -1; // singular
    }

    uint64_t c17_reduc = c17 ^ lsb17;
    reduc = _mm256_set1_epi64x(c17_reduc);
    vmask = _mm256_set1_epi64x(lsb17);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    if(c16 & lsb17) c16 ^= c17_reduc;
    uint64_t c12 = _mm256_extract_epi64(c12_15, 0);
    uint64_t c13 = _mm256_extract_epi64(c12_15, 1);
    uint64_t c14 = _mm256_extract_epi64(c12_15, 2);
    uint64_t c15 = _mm256_extract_epi64(c12_15, 3);
    mask ^= lsb17;

    uint64_t lsb16 = uint64_t_lsb(c16 & mask);
    if(unlikely(!lsb16)) {
        return -1; // singular
    }

    uint64_t c16_reduc = c16 ^ lsb16;
    reduc = _mm256_set1_epi64x(c16_reduc);
    vmask = _mm256_set1_epi64x(lsb16);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    if(c12 & lsb16) c12 ^= c16_reduc;
    if(c13 & lsb16) c13 ^= c16_reduc;
    if(c14 & lsb16) c14 ^= c16_reduc;
    if(c15 & lsb16) c15 ^= c16_reduc;
    mask ^= lsb16;

    uint64_t lsb15 = uint64_t_lsb(c15 & mask);
    if(unlikely(!lsb15)) {
        return -1; // singular
    }

    uint64_t c15_reduc = c15 ^ lsb15;
    reduc = _mm256_set1_epi64x(c15_reduc);
    vmask = _mm256_set1_epi64x(lsb15);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    if(c12 & lsb15) c12 ^= c15_reduc;
    if(c13 & lsb15) c13 ^= c15_reduc;
    if(c14 & lsb15) c14 ^= c15_reduc;
    mask ^= lsb15;

    uint64_t lsb14 = uint64_t_lsb(c14 & mask);
    if(unlikely(!lsb14)) {
        return -1; // singular
    }

    uint64_t c14_reduc = c14 ^ lsb14;
    reduc = _mm256_set1_epi64x(c14_reduc);
    vmask = _mm256_set1_epi64x(lsb14);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    if(c12 & lsb14) c12 ^= c14_reduc;
    if(c13 & lsb14) c13 ^= c14_reduc;
    mask ^= lsb14;

    uint64_t lsb13 = uint64_t_lsb(c13 & mask);
    if(unlikely(!lsb13)) {
        return -1; // singular
    }

    uint64_t c13_reduc = c13 ^ lsb13;
    reduc = _mm256_set1_epi64x(c13_reduc);
    vmask = _mm256_set1_epi64x(lsb13);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    if(c12 & lsb13) c12 ^= c13_reduc;
    uint64_t c8 = _mm256_extract_epi64(c8_11, 0);
    uint64_t c9 = _mm256_extract_epi64(c8_11, 1);
    uint64_t c10 = _mm256_extract_epi64(c8_11, 2);
    uint64_t c11 = _mm256_extract_epi64(c8_11, 3);
    mask ^= lsb13;

    uint64_t lsb12 = uint64_t_lsb(c12 & mask);
    if(unlikely(!lsb12)) {
        return -1; // singular
    }

    uint64_t c12_reduc = c12 ^ lsb12;
    reduc = _mm256_set1_epi64x(c12_reduc);
    vmask = _mm256_set1_epi64x(lsb12);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb12) c8 ^= c12_reduc;
    if(c9 & lsb12) c9 ^= c12_reduc;
    if(c10 & lsb12) c10 ^= c12_reduc;
    if(c11 & lsb12) c11 ^= c12_reduc;
    mask ^= lsb12;

    uint64_t lsb11 = uint64_t_lsb(c11 & mask);
    if(unlikely(!lsb11)) {
        return -1; // singular
    }

    uint64_t c11_reduc = c11 ^ lsb11;
    reduc = _mm256_set1_epi64x(c11_reduc);
    vmask = _mm256_set1_epi64x(lsb11);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb11) c8 ^= c11_reduc;
    if(c9 & lsb11) c9 ^= c11_reduc;
    if(c10 & lsb11) c10 ^= c11_reduc;
    mask ^= lsb11;

    uint64_t lsb10 = uint64_t_lsb(c10 & mask);
    if(unlikely(!lsb10)) {
        return -1; // singular
    }

    uint64_t c10_reduc = c10 ^ lsb10;
    reduc = _mm256_set1_epi64x(c10_reduc);
    vmask = _mm256_set1_epi64x(lsb10);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb10) c8 ^= c10_reduc;
    if(c9 & lsb10) c9 ^= c10_reduc;
    mask ^= lsb10;

    uint64_t lsb9 = uint64_t_lsb(c9 & mask);
    if(unlikely(!lsb9)) {
        return -1; // singular
    }

    uint64_t c9_reduc = c9 ^ lsb9;
    reduc = _mm256_set1_epi64x(c9_reduc);
    vmask = _mm256_set1_epi64x(lsb9);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb9) c8 ^= c9_reduc;
    uint64_t c4 = _mm256_extract_epi64(c4_7, 0);
    uint64_t c5 = _mm256_extract_epi64(c4_7, 1);
    uint64_t c6 = _mm256_extract_epi64(c4_7, 2);
    uint64_t c7 = _mm256_extract_epi64(c4_7, 3);
    mask ^= lsb9;

    uint64_t lsb8 = uint64_t_lsb(c8 & mask);
    if(unlikely(!lsb8)) {
        return -1; // singular
    }

    uint64_t c8_reduc = c8 ^ lsb8;
    reduc = _mm256_set1_epi64x(c8_reduc);
    vmask = _mm256_set1_epi64x(lsb8);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb8) c4 ^= c8_reduc;
    if(c5 & lsb8) c5 ^= c8_reduc;
    if(c6 & lsb8) c6 ^= c8_reduc;
    if(c7 & lsb8) c7 ^= c8_reduc;
    mask ^= lsb8;

    uint64_t lsb7 = uint64_t_lsb(c7 & mask);
    if(unlikely(!lsb7)) {
        return -1; // singular
    }

    uint64_t c7_reduc = c7 ^ lsb7;
    reduc = _mm256_set1_epi64x(c7_reduc);
    vmask = _mm256_set1_epi64x(lsb7);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb7) c4 ^= c7_reduc;
    if(c5 & lsb7) c5 ^= c7_reduc;
    if(c6 & lsb7) c6 ^= c7_reduc;
    mask ^= lsb7;

    uint64_t lsb6 = uint64_t_lsb(c6 & mask);
    if(unlikely(!lsb6)) {
        return -1; // singular
    }

    uint64_t c6_reduc = c6 ^ lsb6;
    reduc = _mm256_set1_epi64x(c6_reduc);
    vmask = _mm256_set1_epi64x(lsb6);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb6) c4 ^= c6_reduc;
    if(c5 & lsb6) c5 ^= c6_reduc;
    mask ^= lsb6;

    uint64_t lsb5 = uint64_t_lsb(c5 & mask);
    if(unlikely(!lsb5)) {
        return -1; // singular
    }

    uint64_t c5_reduc = c5 ^ lsb5;
    reduc = _mm256_set1_epi64x(c5_reduc);
    vmask = _mm256_set1_epi64x(lsb5);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb5) c4 ^= c5_reduc;
    uint64_t c0 = _mm256_extract_epi64(c0_3, 0);
    uint64_t c1 = _mm256_extract_epi64(c0_3, 1);
    uint64_t c2 = _mm256_extract_epi64(c0_3, 2);
    uint64_t c3 = _mm256_extract_epi64(c0_3, 3);
    mask ^= lsb5;

    uint64_t lsb4 = uint64_t_lsb(c4 & mask);
    if(unlikely(!lsb4)) {
        return -1; // singular
    }

    uint64_t c4_reduc = c4 ^ lsb4;
    if(c0 & lsb4) c0 ^= c4_reduc;
    if(c1 & lsb4) c1 ^= c4_reduc;
    if(c2 & lsb4) c2 ^= c4_reduc;
    if(c3 & lsb4) c3 ^= c4_reduc;
    mask ^= lsb4;

    uint64_t lsb3 = uint64_t_lsb(c3 & mask);
    if(unlikely(!lsb3)) {
        return -1; // singular
    }

    uint64_t c3_reduc = c3 ^ lsb3;
    if(c0 & lsb3) c0 ^= c3_reduc;
    if(c1 & lsb3) c1 ^= c3_reduc;
    if(c2 & lsb3) c2 ^= c3_reduc;
    mask ^= lsb3;

    uint64_t lsb2 = uint64_t_lsb(c2 & mask);
    if(unlikely(!lsb2)) {
        return -1; // singular
    }

    uint64_t c2_reduc = c2 ^ lsb2;
    if(c0 & lsb2) c0 ^= c2_reduc;
    if(c1 & lsb2) c1 ^= c2_reduc;
    mask ^= lsb2;

    uint64_t lsb1 = uint64_t_lsb(c1 & mask);
    if(unlikely(!lsb1)) {
        return -1; // singular
    }

    if(c0 & lsb1) c0 ^= c1 ^ lsb1;
    mask ^= lsb1;

    if(likely(mask & c0)) { // check system consistency
        return mask & c0; // not solvable
    }

    uint64_t s = 0x0ULL;
    if(c0 & lsb1) s = uint64_t_toggle_at(s, 0);
    if(c0 & lsb2) s = uint64_t_toggle_at(s, 1);
    if(c0 & lsb3) s = uint64_t_toggle_at(s, 2);
    if(c0 & lsb4) s = uint64_t_toggle_at(s, 3);
    if(c0 & lsb5) s = uint64_t_toggle_at(s, 4);
    if(c0 & lsb6) s = uint64_t_toggle_at(s, 5);
    if(c0 & lsb7) s = uint64_t_toggle_at(s, 6);
    if(c0 & lsb8) s = uint64_t_toggle_at(s, 7);
    if(c0 & lsb9) s = uint64_t_toggle_at(s, 8);
    if(c0 & lsb10) s = uint64_t_toggle_at(s, 9);
    if(c0 & lsb11) s = uint64_t_toggle_at(s, 10);
    if(c0 & lsb12) s = uint64_t_toggle_at(s, 11);
    if(c0 & lsb13) s = uint64_t_toggle_at(s, 12);
    if(c0 & lsb14) s = uint64_t_toggle_at(s, 13);
    if(c0 & lsb15) s = uint64_t_toggle_at(s, 14);
    if(c0 & lsb16) s = uint64_t_toggle_at(s, 15);
    if(c0 & lsb17) s = uint64_t_toggle_at(s, 16);
    if(c0 & lsb18) s = uint64_t_toggle_at(s, 17);
    if(c0 & lsb19) s = uint64_t_toggle_at(s, 18);
    if(c0 & lsb20) s = uint64_t_toggle_at(s, 19);
    if(c0 & lsb21) s = uint64_t_toggle_at(s, 20);
    if(c0 & lsb22) s = uint64_t_toggle_at(s, 21);
    if(c0 & lsb23) s = uint64_t_toggle_at(s, 22);
    if(c0 & lsb24) s = uint64_t_toggle_at(s, 23);
    if(c0 & lsb25) s = uint64_t_toggle_at(s, 24);
    if(c0 & lsb26) s = uint64_t_toggle_at(s, 25);
    if(c0 & lsb27) s = uint64_t_toggle_at(s, 26);
    if(c0 & lsb28) s = uint64_t_toggle_at(s, 27);
    if(c0 & lsb29) s = uint64_t_toggle_at(s, 28);
    *sol = s;
    return 0;
}

int64_t
uint64a_gj_v30_avx(const uint64_t m[31], uint64_t* const restrict sol) {
    __m256i c0_3 = _mm256_loadu_si256((__m256i*) (m + 0));
    __m256i c4_7 = _mm256_loadu_si256((__m256i*) (m + 4));
    __m256i c8_11 = _mm256_loadu_si256((__m256i*) (m + 8));
    __m256i c12_15 = _mm256_loadu_si256((__m256i*) (m + 12));
    __m256i c16_19 = _mm256_loadu_si256((__m256i*) (m + 16));
    __m256i c20_23 = _mm256_loadu_si256((__m256i*) (m + 20));
    __m256i c24_27 = _mm256_loadu_si256((__m256i*) (m + 24));
    uint64_t c28 = m[28];
    uint64_t c29 = m[29];
    uint64_t c30 = m[30];

    uint64_t lsb30 = uint64_t_lsb(c30);
    if(unlikely(!lsb30)) {
        return -1; // singular
    }

    uint64_t c30_reduc = c30 ^ lsb30;
    __m256i reduc = _mm256_set1_epi64x(c30_reduc);
    __m256i vmask = _mm256_set1_epi64x(lsb30);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    c20_23 = uint64a_gj_reduc_mm256(c20_23, vmask, reduc);
    c24_27 = uint64a_gj_reduc_mm256(c24_27, vmask, reduc);
    if(c28 & lsb30) c28 ^= c30_reduc;
    if(c29 & lsb30) c29 ^= c30_reduc;
    uint64_t mask = ~lsb30;

    uint64_t lsb29 = uint64_t_lsb(c29 & mask);
    if(unlikely(!lsb29)) {
        return -1; // singular
    }

    uint64_t c29_reduc = c29 ^ lsb29;
    reduc = _mm256_set1_epi64x(c29_reduc);
    vmask = _mm256_set1_epi64x(lsb29);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    c20_23 = uint64a_gj_reduc_mm256(c20_23, vmask, reduc);
    c24_27 = uint64a_gj_reduc_mm256(c24_27, vmask, reduc);
    if(c28 & lsb29) c28 ^= c29_reduc;
    uint64_t c24 = _mm256_extract_epi64(c24_27, 0);
    uint64_t c25 = _mm256_extract_epi64(c24_27, 1);
    uint64_t c26 = _mm256_extract_epi64(c24_27, 2);
    uint64_t c27 = _mm256_extract_epi64(c24_27, 3);
    mask ^= lsb29;

    uint64_t lsb28 = uint64_t_lsb(c28 & mask);
    if(unlikely(!lsb28)) {
        return -1; // singular
    }

    uint64_t c28_reduc = c28 ^ lsb28;
    reduc = _mm256_set1_epi64x(c28_reduc);
    vmask = _mm256_set1_epi64x(lsb28);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    c20_23 = uint64a_gj_reduc_mm256(c20_23, vmask, reduc);
    if(c24 & lsb28) c24 ^= c28_reduc;
    if(c25 & lsb28) c25 ^= c28_reduc;
    if(c26 & lsb28) c26 ^= c28_reduc;
    if(c27 & lsb28) c27 ^= c28_reduc;
    mask ^= lsb28;

    uint64_t lsb27 = uint64_t_lsb(c27 & mask);
    if(unlikely(!lsb27)) {
        return -1; // singular
    }

    uint64_t c27_reduc = c27 ^ lsb27;
    reduc = _mm256_set1_epi64x(c27_reduc);
    vmask = _mm256_set1_epi64x(lsb27);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    c20_23 = uint64a_gj_reduc_mm256(c20_23, vmask, reduc);
    if(c24 & lsb27) c24 ^= c27_reduc;
    if(c25 & lsb27) c25 ^= c27_reduc;
    if(c26 & lsb27) c26 ^= c27_reduc;
    mask ^= lsb27;

    uint64_t lsb26 = uint64_t_lsb(c26 & mask);
    if(unlikely(!lsb26)) {
        return -1; // singular
    }

    uint64_t c26_reduc = c26 ^ lsb26;
    reduc = _mm256_set1_epi64x(c26_reduc);
    vmask = _mm256_set1_epi64x(lsb26);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    c20_23 = uint64a_gj_reduc_mm256(c20_23, vmask, reduc);
    if(c24 & lsb26) c24 ^= c26_reduc;
    if(c25 & lsb26) c25 ^= c26_reduc;
    mask ^= lsb26;

    uint64_t lsb25 = uint64_t_lsb(c25 & mask);
    if(unlikely(!lsb25)) {
        return -1; // singular
    }

    uint64_t c25_reduc = c25 ^ lsb25;
    reduc = _mm256_set1_epi64x(c25_reduc);
    vmask = _mm256_set1_epi64x(lsb25);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    c20_23 = uint64a_gj_reduc_mm256(c20_23, vmask, reduc);
    if(c24 & lsb25) c24 ^= c25_reduc;
    uint64_t c20 = _mm256_extract_epi64(c20_23, 0);
    uint64_t c21 = _mm256_extract_epi64(c20_23, 1);
    uint64_t c22 = _mm256_extract_epi64(c20_23, 2);
    uint64_t c23 = _mm256_extract_epi64(c20_23, 3);
    mask ^= lsb25;

    uint64_t lsb24 = uint64_t_lsb(c24 & mask);
    if(unlikely(!lsb24)) {
        return -1; // singular
    }

    uint64_t c24_reduc = c24 ^ lsb24;
    reduc = _mm256_set1_epi64x(c24_reduc);
    vmask = _mm256_set1_epi64x(lsb24);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    if(c20 & lsb24) c20 ^= c24_reduc;
    if(c21 & lsb24) c21 ^= c24_reduc;
    if(c22 & lsb24) c22 ^= c24_reduc;
    if(c23 & lsb24) c23 ^= c24_reduc;
    mask ^= lsb24;

    uint64_t lsb23 = uint64_t_lsb(c23 & mask);
    if(unlikely(!lsb23)) {
        return -1; // singular
    }

    uint64_t c23_reduc = c23 ^ lsb23;
    reduc = _mm256_set1_epi64x(c23_reduc);
    vmask = _mm256_set1_epi64x(lsb23);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    if(c20 & lsb23) c20 ^= c23_reduc;
    if(c21 & lsb23) c21 ^= c23_reduc;
    if(c22 & lsb23) c22 ^= c23_reduc;
    mask ^= lsb23;

    uint64_t lsb22 = uint64_t_lsb(c22 & mask);
    if(unlikely(!lsb22)) {
        return -1; // singular
    }

    uint64_t c22_reduc = c22 ^ lsb22;
    reduc = _mm256_set1_epi64x(c22_reduc);
    vmask = _mm256_set1_epi64x(lsb22);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    if(c20 & lsb22) c20 ^= c22_reduc;
    if(c21 & lsb22) c21 ^= c22_reduc;
    mask ^= lsb22;

    uint64_t lsb21 = uint64_t_lsb(c21 & mask);
    if(unlikely(!lsb21)) {
        return -1; // singular
    }

    uint64_t c21_reduc = c21 ^ lsb21;
    reduc = _mm256_set1_epi64x(c21_reduc);
    vmask = _mm256_set1_epi64x(lsb21);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    if(c20 & lsb21) c20 ^= c21_reduc;
    uint64_t c16 = _mm256_extract_epi64(c16_19, 0);
    uint64_t c17 = _mm256_extract_epi64(c16_19, 1);
    uint64_t c18 = _mm256_extract_epi64(c16_19, 2);
    uint64_t c19 = _mm256_extract_epi64(c16_19, 3);
    mask ^= lsb21;

    uint64_t lsb20 = uint64_t_lsb(c20 & mask);
    if(unlikely(!lsb20)) {
        return -1; // singular
    }

    uint64_t c20_reduc = c20 ^ lsb20;
    reduc = _mm256_set1_epi64x(c20_reduc);
    vmask = _mm256_set1_epi64x(lsb20);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    if(c16 & lsb20) c16 ^= c20_reduc;
    if(c17 & lsb20) c17 ^= c20_reduc;
    if(c18 & lsb20) c18 ^= c20_reduc;
    if(c19 & lsb20) c19 ^= c20_reduc;
    mask ^= lsb20;

    uint64_t lsb19 = uint64_t_lsb(c19 & mask);
    if(unlikely(!lsb19)) {
        return -1; // singular
    }

    uint64_t c19_reduc = c19 ^ lsb19;
    reduc = _mm256_set1_epi64x(c19_reduc);
    vmask = _mm256_set1_epi64x(lsb19);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    if(c16 & lsb19) c16 ^= c19_reduc;
    if(c17 & lsb19) c17 ^= c19_reduc;
    if(c18 & lsb19) c18 ^= c19_reduc;
    mask ^= lsb19;

    uint64_t lsb18 = uint64_t_lsb(c18 & mask);
    if(unlikely(!lsb18)) {
        return -1; // singular
    }

    uint64_t c18_reduc = c18 ^ lsb18;
    reduc = _mm256_set1_epi64x(c18_reduc);
    vmask = _mm256_set1_epi64x(lsb18);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    if(c16 & lsb18) c16 ^= c18_reduc;
    if(c17 & lsb18) c17 ^= c18_reduc;
    mask ^= lsb18;

    uint64_t lsb17 = uint64_t_lsb(c17 & mask);
    if(unlikely(!lsb17)) {
        return -1; // singular
    }

    uint64_t c17_reduc = c17 ^ lsb17;
    reduc = _mm256_set1_epi64x(c17_reduc);
    vmask = _mm256_set1_epi64x(lsb17);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    if(c16 & lsb17) c16 ^= c17_reduc;
    uint64_t c12 = _mm256_extract_epi64(c12_15, 0);
    uint64_t c13 = _mm256_extract_epi64(c12_15, 1);
    uint64_t c14 = _mm256_extract_epi64(c12_15, 2);
    uint64_t c15 = _mm256_extract_epi64(c12_15, 3);
    mask ^= lsb17;

    uint64_t lsb16 = uint64_t_lsb(c16 & mask);
    if(unlikely(!lsb16)) {
        return -1; // singular
    }

    uint64_t c16_reduc = c16 ^ lsb16;
    reduc = _mm256_set1_epi64x(c16_reduc);
    vmask = _mm256_set1_epi64x(lsb16);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    if(c12 & lsb16) c12 ^= c16_reduc;
    if(c13 & lsb16) c13 ^= c16_reduc;
    if(c14 & lsb16) c14 ^= c16_reduc;
    if(c15 & lsb16) c15 ^= c16_reduc;
    mask ^= lsb16;

    uint64_t lsb15 = uint64_t_lsb(c15 & mask);
    if(unlikely(!lsb15)) {
        return -1; // singular
    }

    uint64_t c15_reduc = c15 ^ lsb15;
    reduc = _mm256_set1_epi64x(c15_reduc);
    vmask = _mm256_set1_epi64x(lsb15);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    if(c12 & lsb15) c12 ^= c15_reduc;
    if(c13 & lsb15) c13 ^= c15_reduc;
    if(c14 & lsb15) c14 ^= c15_reduc;
    mask ^= lsb15;

    uint64_t lsb14 = uint64_t_lsb(c14 & mask);
    if(unlikely(!lsb14)) {
        return -1; // singular
    }

    uint64_t c14_reduc = c14 ^ lsb14;
    reduc = _mm256_set1_epi64x(c14_reduc);
    vmask = _mm256_set1_epi64x(lsb14);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    if(c12 & lsb14) c12 ^= c14_reduc;
    if(c13 & lsb14) c13 ^= c14_reduc;
    mask ^= lsb14;

    uint64_t lsb13 = uint64_t_lsb(c13 & mask);
    if(unlikely(!lsb13)) {
        return -1; // singular
    }

    uint64_t c13_reduc = c13 ^ lsb13;
    reduc = _mm256_set1_epi64x(c13_reduc);
    vmask = _mm256_set1_epi64x(lsb13);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    if(c12 & lsb13) c12 ^= c13_reduc;
    uint64_t c8 = _mm256_extract_epi64(c8_11, 0);
    uint64_t c9 = _mm256_extract_epi64(c8_11, 1);
    uint64_t c10 = _mm256_extract_epi64(c8_11, 2);
    uint64_t c11 = _mm256_extract_epi64(c8_11, 3);
    mask ^= lsb13;

    uint64_t lsb12 = uint64_t_lsb(c12 & mask);
    if(unlikely(!lsb12)) {
        return -1; // singular
    }

    uint64_t c12_reduc = c12 ^ lsb12;
    reduc = _mm256_set1_epi64x(c12_reduc);
    vmask = _mm256_set1_epi64x(lsb12);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb12) c8 ^= c12_reduc;
    if(c9 & lsb12) c9 ^= c12_reduc;
    if(c10 & lsb12) c10 ^= c12_reduc;
    if(c11 & lsb12) c11 ^= c12_reduc;
    mask ^= lsb12;

    uint64_t lsb11 = uint64_t_lsb(c11 & mask);
    if(unlikely(!lsb11)) {
        return -1; // singular
    }

    uint64_t c11_reduc = c11 ^ lsb11;
    reduc = _mm256_set1_epi64x(c11_reduc);
    vmask = _mm256_set1_epi64x(lsb11);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb11) c8 ^= c11_reduc;
    if(c9 & lsb11) c9 ^= c11_reduc;
    if(c10 & lsb11) c10 ^= c11_reduc;
    mask ^= lsb11;

    uint64_t lsb10 = uint64_t_lsb(c10 & mask);
    if(unlikely(!lsb10)) {
        return -1; // singular
    }

    uint64_t c10_reduc = c10 ^ lsb10;
    reduc = _mm256_set1_epi64x(c10_reduc);
    vmask = _mm256_set1_epi64x(lsb10);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb10) c8 ^= c10_reduc;
    if(c9 & lsb10) c9 ^= c10_reduc;
    mask ^= lsb10;

    uint64_t lsb9 = uint64_t_lsb(c9 & mask);
    if(unlikely(!lsb9)) {
        return -1; // singular
    }

    uint64_t c9_reduc = c9 ^ lsb9;
    reduc = _mm256_set1_epi64x(c9_reduc);
    vmask = _mm256_set1_epi64x(lsb9);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb9) c8 ^= c9_reduc;
    uint64_t c4 = _mm256_extract_epi64(c4_7, 0);
    uint64_t c5 = _mm256_extract_epi64(c4_7, 1);
    uint64_t c6 = _mm256_extract_epi64(c4_7, 2);
    uint64_t c7 = _mm256_extract_epi64(c4_7, 3);
    mask ^= lsb9;

    uint64_t lsb8 = uint64_t_lsb(c8 & mask);
    if(unlikely(!lsb8)) {
        return -1; // singular
    }

    uint64_t c8_reduc = c8 ^ lsb8;
    reduc = _mm256_set1_epi64x(c8_reduc);
    vmask = _mm256_set1_epi64x(lsb8);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb8) c4 ^= c8_reduc;
    if(c5 & lsb8) c5 ^= c8_reduc;
    if(c6 & lsb8) c6 ^= c8_reduc;
    if(c7 & lsb8) c7 ^= c8_reduc;
    mask ^= lsb8;

    uint64_t lsb7 = uint64_t_lsb(c7 & mask);
    if(unlikely(!lsb7)) {
        return -1; // singular
    }

    uint64_t c7_reduc = c7 ^ lsb7;
    reduc = _mm256_set1_epi64x(c7_reduc);
    vmask = _mm256_set1_epi64x(lsb7);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb7) c4 ^= c7_reduc;
    if(c5 & lsb7) c5 ^= c7_reduc;
    if(c6 & lsb7) c6 ^= c7_reduc;
    mask ^= lsb7;

    uint64_t lsb6 = uint64_t_lsb(c6 & mask);
    if(unlikely(!lsb6)) {
        return -1; // singular
    }

    uint64_t c6_reduc = c6 ^ lsb6;
    reduc = _mm256_set1_epi64x(c6_reduc);
    vmask = _mm256_set1_epi64x(lsb6);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb6) c4 ^= c6_reduc;
    if(c5 & lsb6) c5 ^= c6_reduc;
    mask ^= lsb6;

    uint64_t lsb5 = uint64_t_lsb(c5 & mask);
    if(unlikely(!lsb5)) {
        return -1; // singular
    }

    uint64_t c5_reduc = c5 ^ lsb5;
    reduc = _mm256_set1_epi64x(c5_reduc);
    vmask = _mm256_set1_epi64x(lsb5);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb5) c4 ^= c5_reduc;
    uint64_t c0 = _mm256_extract_epi64(c0_3, 0);
    uint64_t c1 = _mm256_extract_epi64(c0_3, 1);
    uint64_t c2 = _mm256_extract_epi64(c0_3, 2);
    uint64_t c3 = _mm256_extract_epi64(c0_3, 3);
    mask ^= lsb5;

    uint64_t lsb4 = uint64_t_lsb(c4 & mask);
    if(unlikely(!lsb4)) {
        return -1; // singular
    }

    uint64_t c4_reduc = c4 ^ lsb4;
    if(c0 & lsb4) c0 ^= c4_reduc;
    if(c1 & lsb4) c1 ^= c4_reduc;
    if(c2 & lsb4) c2 ^= c4_reduc;
    if(c3 & lsb4) c3 ^= c4_reduc;
    mask ^= lsb4;

    uint64_t lsb3 = uint64_t_lsb(c3 & mask);
    if(unlikely(!lsb3)) {
        return -1; // singular
    }

    uint64_t c3_reduc = c3 ^ lsb3;
    if(c0 & lsb3) c0 ^= c3_reduc;
    if(c1 & lsb3) c1 ^= c3_reduc;
    if(c2 & lsb3) c2 ^= c3_reduc;
    mask ^= lsb3;

    uint64_t lsb2 = uint64_t_lsb(c2 & mask);
    if(unlikely(!lsb2)) {
        return -1; // singular
    }

    uint64_t c2_reduc = c2 ^ lsb2;
    if(c0 & lsb2) c0 ^= c2_reduc;
    if(c1 & lsb2) c1 ^= c2_reduc;
    mask ^= lsb2;

    uint64_t lsb1 = uint64_t_lsb(c1 & mask);
    if(unlikely(!lsb1)) {
        return -1; // singular
    }

    if(c0 & lsb1) c0 ^= c1 ^ lsb1;
    mask ^= lsb1;

    if(likely(mask & c0)) { // check system consistency
        return mask & c0; // not solvable
    }

    uint64_t s = 0x0ULL;
    if(c0 & lsb1) s = uint64_t_toggle_at(s, 0);
    if(c0 & lsb2) s = uint64_t_toggle_at(s, 1);
    if(c0 & lsb3) s = uint64_t_toggle_at(s, 2);
    if(c0 & lsb4) s = uint64_t_toggle_at(s, 3);
    if(c0 & lsb5) s = uint64_t_toggle_at(s, 4);
    if(c0 & lsb6) s = uint64_t_toggle_at(s, 5);
    if(c0 & lsb7) s = uint64_t_toggle_at(s, 6);
    if(c0 & lsb8) s = uint64_t_toggle_at(s, 7);
    if(c0 & lsb9) s = uint64_t_toggle_at(s, 8);
    if(c0 & lsb10) s = uint64_t_toggle_at(s, 9);
    if(c0 & lsb11) s = uint64_t_toggle_at(s, 10);
    if(c0 & lsb12) s = uint64_t_toggle_at(s, 11);
    if(c0 & lsb13) s = uint64_t_toggle_at(s, 12);
    if(c0 & lsb14) s = uint64_t_toggle_at(s, 13);
    if(c0 & lsb15) s = uint64_t_toggle_at(s, 14);
    if(c0 & lsb16) s = uint64_t_toggle_at(s, 15);
    if(c0 & lsb17) s = uint64_t_toggle_at(s, 16);
    if(c0 & lsb18) s = uint64_t_toggle_at(s, 17);
    if(c0 & lsb19) s = uint64_t_toggle_at(s, 18);
    if(c0 & lsb20) s = uint64_t_toggle_at(s, 19);
    if(c0 & lsb21) s = uint64_t_toggle_at(s, 20);
    if(c0 & lsb22) s = uint64_t_toggle_at(s, 21);
    if(c0 & lsb23) s = uint64_t_toggle_at(s, 22);
    if(c0 & lsb24) s = uint64_t_toggle_at(s, 23);
    if(c0 & lsb25) s = uint64_t_toggle_at(s, 24);
    if(c0 & lsb26) s = uint64_t_toggle_at(s, 25);
    if(c0 & lsb27) s = uint64_t_toggle_at(s, 26);
    if(c0 & lsb28) s = uint64_t_toggle_at(s, 27);
    if(c0 & lsb29) s = uint64_t_toggle_at(s, 28);
    if(c0 & lsb30) s = uint64_t_toggle_at(s, 29);
    *sol = s;
    return 0;
}

int64_t
uint64a_gj_v31_avx(const uint64_t m[32], uint64_t* const restrict sol) {
    __m256i c0_3 = _mm256_loadu_si256((__m256i*) (m + 0));
    __m256i c4_7 = _mm256_loadu_si256((__m256i*) (m + 4));
    __m256i c8_11 = _mm256_loadu_si256((__m256i*) (m + 8));
    __m256i c12_15 = _mm256_loadu_si256((__m256i*) (m + 12));
    __m256i c16_19 = _mm256_loadu_si256((__m256i*) (m + 16));
    __m256i c20_23 = _mm256_loadu_si256((__m256i*) (m + 20));
    __m256i c24_27 = _mm256_loadu_si256((__m256i*) (m + 24));
    uint64_t c28 = m[28];
    uint64_t c29 = m[29];
    uint64_t c30 = m[30];
    uint64_t c31 = m[31];

    uint64_t lsb31 = uint64_t_lsb(c31);
    if(unlikely(!lsb31)) {
        return -1; // singular
    }

    uint64_t c31_reduc = c31 ^ lsb31;
    __m256i reduc = _mm256_set1_epi64x(c31_reduc);
    __m256i vmask = _mm256_set1_epi64x(lsb31);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    c20_23 = uint64a_gj_reduc_mm256(c20_23, vmask, reduc);
    c24_27 = uint64a_gj_reduc_mm256(c24_27, vmask, reduc);
    if(c28 & lsb31) c28 ^= c31_reduc;
    if(c29 & lsb31) c29 ^= c31_reduc;
    if(c30 & lsb31) c30 ^= c31_reduc;
    uint64_t mask = ~lsb31;

    uint64_t lsb30 = uint64_t_lsb(c30 & mask);
    if(unlikely(!lsb30)) {
        return -1; // singular
    }

    uint64_t c30_reduc = c30 ^ lsb30;
    reduc = _mm256_set1_epi64x(c30_reduc);
    vmask = _mm256_set1_epi64x(lsb30);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    c20_23 = uint64a_gj_reduc_mm256(c20_23, vmask, reduc);
    c24_27 = uint64a_gj_reduc_mm256(c24_27, vmask, reduc);
    if(c28 & lsb30) c28 ^= c30_reduc;
    if(c29 & lsb30) c29 ^= c30_reduc;
    mask ^= lsb30;

    uint64_t lsb29 = uint64_t_lsb(c29 & mask);
    if(unlikely(!lsb29)) {
        return -1; // singular
    }

    uint64_t c29_reduc = c29 ^ lsb29;
    reduc = _mm256_set1_epi64x(c29_reduc);
    vmask = _mm256_set1_epi64x(lsb29);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    c20_23 = uint64a_gj_reduc_mm256(c20_23, vmask, reduc);
    c24_27 = uint64a_gj_reduc_mm256(c24_27, vmask, reduc);
    if(c28 & lsb29) c28 ^= c29_reduc;
    uint64_t c24 = _mm256_extract_epi64(c24_27, 0);
    uint64_t c25 = _mm256_extract_epi64(c24_27, 1);
    uint64_t c26 = _mm256_extract_epi64(c24_27, 2);
    uint64_t c27 = _mm256_extract_epi64(c24_27, 3);
    mask ^= lsb29;

    uint64_t lsb28 = uint64_t_lsb(c28 & mask);
    if(unlikely(!lsb28)) {
        return -1; // singular
    }

    uint64_t c28_reduc = c28 ^ lsb28;
    reduc = _mm256_set1_epi64x(c28_reduc);
    vmask = _mm256_set1_epi64x(lsb28);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    c20_23 = uint64a_gj_reduc_mm256(c20_23, vmask, reduc);
    if(c24 & lsb28) c24 ^= c28_reduc;
    if(c25 & lsb28) c25 ^= c28_reduc;
    if(c26 & lsb28) c26 ^= c28_reduc;
    if(c27 & lsb28) c27 ^= c28_reduc;
    mask ^= lsb28;

    uint64_t lsb27 = uint64_t_lsb(c27 & mask);
    if(unlikely(!lsb27)) {
        return -1; // singular
    }

    uint64_t c27_reduc = c27 ^ lsb27;
    reduc = _mm256_set1_epi64x(c27_reduc);
    vmask = _mm256_set1_epi64x(lsb27);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    c20_23 = uint64a_gj_reduc_mm256(c20_23, vmask, reduc);
    if(c24 & lsb27) c24 ^= c27_reduc;
    if(c25 & lsb27) c25 ^= c27_reduc;
    if(c26 & lsb27) c26 ^= c27_reduc;
    mask ^= lsb27;

    uint64_t lsb26 = uint64_t_lsb(c26 & mask);
    if(unlikely(!lsb26)) {
        return -1; // singular
    }

    uint64_t c26_reduc = c26 ^ lsb26;
    reduc = _mm256_set1_epi64x(c26_reduc);
    vmask = _mm256_set1_epi64x(lsb26);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    c20_23 = uint64a_gj_reduc_mm256(c20_23, vmask, reduc);
    if(c24 & lsb26) c24 ^= c26_reduc;
    if(c25 & lsb26) c25 ^= c26_reduc;
    mask ^= lsb26;

    uint64_t lsb25 = uint64_t_lsb(c25 & mask);
    if(unlikely(!lsb25)) {
        return -1; // singular
    }

    uint64_t c25_reduc = c25 ^ lsb25;
    reduc = _mm256_set1_epi64x(c25_reduc);
    vmask = _mm256_set1_epi64x(lsb25);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    c20_23 = uint64a_gj_reduc_mm256(c20_23, vmask, reduc);
    if(c24 & lsb25) c24 ^= c25_reduc;
    uint64_t c20 = _mm256_extract_epi64(c20_23, 0);
    uint64_t c21 = _mm256_extract_epi64(c20_23, 1);
    uint64_t c22 = _mm256_extract_epi64(c20_23, 2);
    uint64_t c23 = _mm256_extract_epi64(c20_23, 3);
    mask ^= lsb25;

    uint64_t lsb24 = uint64_t_lsb(c24 & mask);
    if(unlikely(!lsb24)) {
        return -1; // singular
    }

    uint64_t c24_reduc = c24 ^ lsb24;
    reduc = _mm256_set1_epi64x(c24_reduc);
    vmask = _mm256_set1_epi64x(lsb24);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    if(c20 & lsb24) c20 ^= c24_reduc;
    if(c21 & lsb24) c21 ^= c24_reduc;
    if(c22 & lsb24) c22 ^= c24_reduc;
    if(c23 & lsb24) c23 ^= c24_reduc;
    mask ^= lsb24;

    uint64_t lsb23 = uint64_t_lsb(c23 & mask);
    if(unlikely(!lsb23)) {
        return -1; // singular
    }

    uint64_t c23_reduc = c23 ^ lsb23;
    reduc = _mm256_set1_epi64x(c23_reduc);
    vmask = _mm256_set1_epi64x(lsb23);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    if(c20 & lsb23) c20 ^= c23_reduc;
    if(c21 & lsb23) c21 ^= c23_reduc;
    if(c22 & lsb23) c22 ^= c23_reduc;
    mask ^= lsb23;

    uint64_t lsb22 = uint64_t_lsb(c22 & mask);
    if(unlikely(!lsb22)) {
        return -1; // singular
    }

    uint64_t c22_reduc = c22 ^ lsb22;
    reduc = _mm256_set1_epi64x(c22_reduc);
    vmask = _mm256_set1_epi64x(lsb22);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    if(c20 & lsb22) c20 ^= c22_reduc;
    if(c21 & lsb22) c21 ^= c22_reduc;
    mask ^= lsb22;

    uint64_t lsb21 = uint64_t_lsb(c21 & mask);
    if(unlikely(!lsb21)) {
        return -1; // singular
    }

    uint64_t c21_reduc = c21 ^ lsb21;
    reduc = _mm256_set1_epi64x(c21_reduc);
    vmask = _mm256_set1_epi64x(lsb21);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    if(c20 & lsb21) c20 ^= c21_reduc;
    uint64_t c16 = _mm256_extract_epi64(c16_19, 0);
    uint64_t c17 = _mm256_extract_epi64(c16_19, 1);
    uint64_t c18 = _mm256_extract_epi64(c16_19, 2);
    uint64_t c19 = _mm256_extract_epi64(c16_19, 3);
    mask ^= lsb21;

    uint64_t lsb20 = uint64_t_lsb(c20 & mask);
    if(unlikely(!lsb20)) {
        return -1; // singular
    }

    uint64_t c20_reduc = c20 ^ lsb20;
    reduc = _mm256_set1_epi64x(c20_reduc);
    vmask = _mm256_set1_epi64x(lsb20);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    if(c16 & lsb20) c16 ^= c20_reduc;
    if(c17 & lsb20) c17 ^= c20_reduc;
    if(c18 & lsb20) c18 ^= c20_reduc;
    if(c19 & lsb20) c19 ^= c20_reduc;
    mask ^= lsb20;

    uint64_t lsb19 = uint64_t_lsb(c19 & mask);
    if(unlikely(!lsb19)) {
        return -1; // singular
    }

    uint64_t c19_reduc = c19 ^ lsb19;
    reduc = _mm256_set1_epi64x(c19_reduc);
    vmask = _mm256_set1_epi64x(lsb19);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    if(c16 & lsb19) c16 ^= c19_reduc;
    if(c17 & lsb19) c17 ^= c19_reduc;
    if(c18 & lsb19) c18 ^= c19_reduc;
    mask ^= lsb19;

    uint64_t lsb18 = uint64_t_lsb(c18 & mask);
    if(unlikely(!lsb18)) {
        return -1; // singular
    }

    uint64_t c18_reduc = c18 ^ lsb18;
    reduc = _mm256_set1_epi64x(c18_reduc);
    vmask = _mm256_set1_epi64x(lsb18);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    if(c16 & lsb18) c16 ^= c18_reduc;
    if(c17 & lsb18) c17 ^= c18_reduc;
    mask ^= lsb18;

    uint64_t lsb17 = uint64_t_lsb(c17 & mask);
    if(unlikely(!lsb17)) {
        return -1; // singular
    }

    uint64_t c17_reduc = c17 ^ lsb17;
    reduc = _mm256_set1_epi64x(c17_reduc);
    vmask = _mm256_set1_epi64x(lsb17);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    if(c16 & lsb17) c16 ^= c17_reduc;
    uint64_t c12 = _mm256_extract_epi64(c12_15, 0);
    uint64_t c13 = _mm256_extract_epi64(c12_15, 1);
    uint64_t c14 = _mm256_extract_epi64(c12_15, 2);
    uint64_t c15 = _mm256_extract_epi64(c12_15, 3);
    mask ^= lsb17;

    uint64_t lsb16 = uint64_t_lsb(c16 & mask);
    if(unlikely(!lsb16)) {
        return -1; // singular
    }

    uint64_t c16_reduc = c16 ^ lsb16;
    reduc = _mm256_set1_epi64x(c16_reduc);
    vmask = _mm256_set1_epi64x(lsb16);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    if(c12 & lsb16) c12 ^= c16_reduc;
    if(c13 & lsb16) c13 ^= c16_reduc;
    if(c14 & lsb16) c14 ^= c16_reduc;
    if(c15 & lsb16) c15 ^= c16_reduc;
    mask ^= lsb16;

    uint64_t lsb15 = uint64_t_lsb(c15 & mask);
    if(unlikely(!lsb15)) {
        return -1; // singular
    }

    uint64_t c15_reduc = c15 ^ lsb15;
    reduc = _mm256_set1_epi64x(c15_reduc);
    vmask = _mm256_set1_epi64x(lsb15);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    if(c12 & lsb15) c12 ^= c15_reduc;
    if(c13 & lsb15) c13 ^= c15_reduc;
    if(c14 & lsb15) c14 ^= c15_reduc;
    mask ^= lsb15;

    uint64_t lsb14 = uint64_t_lsb(c14 & mask);
    if(unlikely(!lsb14)) {
        return -1; // singular
    }

    uint64_t c14_reduc = c14 ^ lsb14;
    reduc = _mm256_set1_epi64x(c14_reduc);
    vmask = _mm256_set1_epi64x(lsb14);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    if(c12 & lsb14) c12 ^= c14_reduc;
    if(c13 & lsb14) c13 ^= c14_reduc;
    mask ^= lsb14;

    uint64_t lsb13 = uint64_t_lsb(c13 & mask);
    if(unlikely(!lsb13)) {
        return -1; // singular
    }

    uint64_t c13_reduc = c13 ^ lsb13;
    reduc = _mm256_set1_epi64x(c13_reduc);
    vmask = _mm256_set1_epi64x(lsb13);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    if(c12 & lsb13) c12 ^= c13_reduc;
    uint64_t c8 = _mm256_extract_epi64(c8_11, 0);
    uint64_t c9 = _mm256_extract_epi64(c8_11, 1);
    uint64_t c10 = _mm256_extract_epi64(c8_11, 2);
    uint64_t c11 = _mm256_extract_epi64(c8_11, 3);
    mask ^= lsb13;

    uint64_t lsb12 = uint64_t_lsb(c12 & mask);
    if(unlikely(!lsb12)) {
        return -1; // singular
    }

    uint64_t c12_reduc = c12 ^ lsb12;
    reduc = _mm256_set1_epi64x(c12_reduc);
    vmask = _mm256_set1_epi64x(lsb12);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb12) c8 ^= c12_reduc;
    if(c9 & lsb12) c9 ^= c12_reduc;
    if(c10 & lsb12) c10 ^= c12_reduc;
    if(c11 & lsb12) c11 ^= c12_reduc;
    mask ^= lsb12;

    uint64_t lsb11 = uint64_t_lsb(c11 & mask);
    if(unlikely(!lsb11)) {
        return -1; // singular
    }

    uint64_t c11_reduc = c11 ^ lsb11;
    reduc = _mm256_set1_epi64x(c11_reduc);
    vmask = _mm256_set1_epi64x(lsb11);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb11) c8 ^= c11_reduc;
    if(c9 & lsb11) c9 ^= c11_reduc;
    if(c10 & lsb11) c10 ^= c11_reduc;
    mask ^= lsb11;

    uint64_t lsb10 = uint64_t_lsb(c10 & mask);
    if(unlikely(!lsb10)) {
        return -1; // singular
    }

    uint64_t c10_reduc = c10 ^ lsb10;
    reduc = _mm256_set1_epi64x(c10_reduc);
    vmask = _mm256_set1_epi64x(lsb10);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb10) c8 ^= c10_reduc;
    if(c9 & lsb10) c9 ^= c10_reduc;
    mask ^= lsb10;

    uint64_t lsb9 = uint64_t_lsb(c9 & mask);
    if(unlikely(!lsb9)) {
        return -1; // singular
    }

    uint64_t c9_reduc = c9 ^ lsb9;
    reduc = _mm256_set1_epi64x(c9_reduc);
    vmask = _mm256_set1_epi64x(lsb9);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb9) c8 ^= c9_reduc;
    uint64_t c4 = _mm256_extract_epi64(c4_7, 0);
    uint64_t c5 = _mm256_extract_epi64(c4_7, 1);
    uint64_t c6 = _mm256_extract_epi64(c4_7, 2);
    uint64_t c7 = _mm256_extract_epi64(c4_7, 3);
    mask ^= lsb9;

    uint64_t lsb8 = uint64_t_lsb(c8 & mask);
    if(unlikely(!lsb8)) {
        return -1; // singular
    }

    uint64_t c8_reduc = c8 ^ lsb8;
    reduc = _mm256_set1_epi64x(c8_reduc);
    vmask = _mm256_set1_epi64x(lsb8);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb8) c4 ^= c8_reduc;
    if(c5 & lsb8) c5 ^= c8_reduc;
    if(c6 & lsb8) c6 ^= c8_reduc;
    if(c7 & lsb8) c7 ^= c8_reduc;
    mask ^= lsb8;

    uint64_t lsb7 = uint64_t_lsb(c7 & mask);
    if(unlikely(!lsb7)) {
        return -1; // singular
    }

    uint64_t c7_reduc = c7 ^ lsb7;
    reduc = _mm256_set1_epi64x(c7_reduc);
    vmask = _mm256_set1_epi64x(lsb7);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb7) c4 ^= c7_reduc;
    if(c5 & lsb7) c5 ^= c7_reduc;
    if(c6 & lsb7) c6 ^= c7_reduc;
    mask ^= lsb7;

    uint64_t lsb6 = uint64_t_lsb(c6 & mask);
    if(unlikely(!lsb6)) {
        return -1; // singular
    }

    uint64_t c6_reduc = c6 ^ lsb6;
    reduc = _mm256_set1_epi64x(c6_reduc);
    vmask = _mm256_set1_epi64x(lsb6);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb6) c4 ^= c6_reduc;
    if(c5 & lsb6) c5 ^= c6_reduc;
    mask ^= lsb6;

    uint64_t lsb5 = uint64_t_lsb(c5 & mask);
    if(unlikely(!lsb5)) {
        return -1; // singular
    }

    uint64_t c5_reduc = c5 ^ lsb5;
    reduc = _mm256_set1_epi64x(c5_reduc);
    vmask = _mm256_set1_epi64x(lsb5);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb5) c4 ^= c5_reduc;
    uint64_t c0 = _mm256_extract_epi64(c0_3, 0);
    uint64_t c1 = _mm256_extract_epi64(c0_3, 1);
    uint64_t c2 = _mm256_extract_epi64(c0_3, 2);
    uint64_t c3 = _mm256_extract_epi64(c0_3, 3);
    mask ^= lsb5;

    uint64_t lsb4 = uint64_t_lsb(c4 & mask);
    if(unlikely(!lsb4)) {
        return -1; // singular
    }

    uint64_t c4_reduc = c4 ^ lsb4;
    if(c0 & lsb4) c0 ^= c4_reduc;
    if(c1 & lsb4) c1 ^= c4_reduc;
    if(c2 & lsb4) c2 ^= c4_reduc;
    if(c3 & lsb4) c3 ^= c4_reduc;
    mask ^= lsb4;

    uint64_t lsb3 = uint64_t_lsb(c3 & mask);
    if(unlikely(!lsb3)) {
        return -1; // singular
    }

    uint64_t c3_reduc = c3 ^ lsb3;
    if(c0 & lsb3) c0 ^= c3_reduc;
    if(c1 & lsb3) c1 ^= c3_reduc;
    if(c2 & lsb3) c2 ^= c3_reduc;
    mask ^= lsb3;

    uint64_t lsb2 = uint64_t_lsb(c2 & mask);
    if(unlikely(!lsb2)) {
        return -1; // singular
    }

    uint64_t c2_reduc = c2 ^ lsb2;
    if(c0 & lsb2) c0 ^= c2_reduc;
    if(c1 & lsb2) c1 ^= c2_reduc;
    mask ^= lsb2;

    uint64_t lsb1 = uint64_t_lsb(c1 & mask);
    if(unlikely(!lsb1)) {
        return -1; // singular
    }

    if(c0 & lsb1) c0 ^= c1 ^ lsb1;
    mask ^= lsb1;

    if(likely(mask & c0)) { // check system consistency
        return mask & c0; // not solvable
    }

    uint64_t s = 0x0ULL;
    if(c0 & lsb1) s = uint64_t_toggle_at(s, 0);
    if(c0 & lsb2) s = uint64_t_toggle_at(s, 1);
    if(c0 & lsb3) s = uint64_t_toggle_at(s, 2);
    if(c0 & lsb4) s = uint64_t_toggle_at(s, 3);
    if(c0 & lsb5) s = uint64_t_toggle_at(s, 4);
    if(c0 & lsb6) s = uint64_t_toggle_at(s, 5);
    if(c0 & lsb7) s = uint64_t_toggle_at(s, 6);
    if(c0 & lsb8) s = uint64_t_toggle_at(s, 7);
    if(c0 & lsb9) s = uint64_t_toggle_at(s, 8);
    if(c0 & lsb10) s = uint64_t_toggle_at(s, 9);
    if(c0 & lsb11) s = uint64_t_toggle_at(s, 10);
    if(c0 & lsb12) s = uint64_t_toggle_at(s, 11);
    if(c0 & lsb13) s = uint64_t_toggle_at(s, 12);
    if(c0 & lsb14) s = uint64_t_toggle_at(s, 13);
    if(c0 & lsb15) s = uint64_t_toggle_at(s, 14);
    if(c0 & lsb16) s = uint64_t_toggle_at(s, 15);
    if(c0 & lsb17) s = uint64_t_toggle_at(s, 16);
    if(c0 & lsb18) s = uint64_t_toggle_at(s, 17);
    if(c0 & lsb19) s = uint64_t_toggle_at(s, 18);
    if(c0 & lsb20) s = uint64_t_toggle_at(s, 19);
    if(c0 & lsb21) s = uint64_t_toggle_at(s, 20);
    if(c0 & lsb22) s = uint64_t_toggle_at(s, 21);
    if(c0 & lsb23) s = uint64_t_toggle_at(s, 22);
    if(c0 & lsb24) s = uint64_t_toggle_at(s, 23);
    if(c0 & lsb25) s = uint64_t_toggle_at(s, 24);
    if(c0 & lsb26) s = uint64_t_toggle_at(s, 25);
    if(c0 & lsb27) s = uint64_t_toggle_at(s, 26);
    if(c0 & lsb28) s = uint64_t_toggle_at(s, 27);
    if(c0 & lsb29) s = uint64_t_toggle_at(s, 28);
    if(c0 & lsb30) s = uint64_t_toggle_at(s, 29);
    if(c0 & lsb31) s = uint64_t_toggle_at(s, 30);
    *sol = s;
    return 0;
}

int64_t
uint64a_gj_v32_avx(const uint64_t m[33], uint64_t* const restrict sol) {
    __m256i c0_3 = _mm256_loadu_si256((__m256i*) (m + 0));
    __m256i c4_7 = _mm256_loadu_si256((__m256i*) (m + 4));
    __m256i c8_11 = _mm256_loadu_si256((__m256i*) (m + 8));
    __m256i c12_15 = _mm256_loadu_si256((__m256i*) (m + 12));
    __m256i c16_19 = _mm256_loadu_si256((__m256i*) (m + 16));
    __m256i c20_23 = _mm256_loadu_si256((__m256i*) (m + 20));
    __m256i c24_27 = _mm256_loadu_si256((__m256i*) (m + 24));
    uint64_t c28 = m[28];
    uint64_t c29 = m[29];
    uint64_t c30 = m[30];
    uint64_t c31 = m[31];
    uint64_t c32 = m[32];

    uint64_t lsb32 = uint64_t_lsb(c32);
    if(unlikely(!lsb32)) {
        return -1; // singular
    }

    uint64_t c32_reduc = c32 ^ lsb32;
    __m256i reduc = _mm256_set1_epi64x(c32_reduc);
    __m256i vmask = _mm256_set1_epi64x(lsb32);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    c20_23 = uint64a_gj_reduc_mm256(c20_23, vmask, reduc);
    c24_27 = uint64a_gj_reduc_mm256(c24_27, vmask, reduc);
    if(c28 & lsb32) c28 ^= c32_reduc;
    if(c29 & lsb32) c29 ^= c32_reduc;
    if(c30 & lsb32) c30 ^= c32_reduc;
    if(c31 & lsb32) c31 ^= c32_reduc;
    uint64_t mask = ~lsb32;

    uint64_t lsb31 = uint64_t_lsb(c31 & mask);
    if(unlikely(!lsb31)) {
        return -1; // singular
    }

    uint64_t c31_reduc = c31 ^ lsb31;
    reduc = _mm256_set1_epi64x(c31_reduc);
    vmask = _mm256_set1_epi64x(lsb31);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    c20_23 = uint64a_gj_reduc_mm256(c20_23, vmask, reduc);
    c24_27 = uint64a_gj_reduc_mm256(c24_27, vmask, reduc);
    if(c28 & lsb31) c28 ^= c31_reduc;
    if(c29 & lsb31) c29 ^= c31_reduc;
    if(c30 & lsb31) c30 ^= c31_reduc;
    mask ^= lsb31;

    uint64_t lsb30 = uint64_t_lsb(c30 & mask);
    if(unlikely(!lsb30)) {
        return -1; // singular
    }

    uint64_t c30_reduc = c30 ^ lsb30;
    reduc = _mm256_set1_epi64x(c30_reduc);
    vmask = _mm256_set1_epi64x(lsb30);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    c20_23 = uint64a_gj_reduc_mm256(c20_23, vmask, reduc);
    c24_27 = uint64a_gj_reduc_mm256(c24_27, vmask, reduc);
    if(c28 & lsb30) c28 ^= c30_reduc;
    if(c29 & lsb30) c29 ^= c30_reduc;
    mask ^= lsb30;

    uint64_t lsb29 = uint64_t_lsb(c29 & mask);
    if(unlikely(!lsb29)) {
        return -1; // singular
    }

    uint64_t c29_reduc = c29 ^ lsb29;
    reduc = _mm256_set1_epi64x(c29_reduc);
    vmask = _mm256_set1_epi64x(lsb29);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    c20_23 = uint64a_gj_reduc_mm256(c20_23, vmask, reduc);
    c24_27 = uint64a_gj_reduc_mm256(c24_27, vmask, reduc);
    if(c28 & lsb29) c28 ^= c29_reduc;
    uint64_t c24 = _mm256_extract_epi64(c24_27, 0);
    uint64_t c25 = _mm256_extract_epi64(c24_27, 1);
    uint64_t c26 = _mm256_extract_epi64(c24_27, 2);
    uint64_t c27 = _mm256_extract_epi64(c24_27, 3);
    mask ^= lsb29;

    uint64_t lsb28 = uint64_t_lsb(c28 & mask);
    if(unlikely(!lsb28)) {
        return -1; // singular
    }

    uint64_t c28_reduc = c28 ^ lsb28;
    reduc = _mm256_set1_epi64x(c28_reduc);
    vmask = _mm256_set1_epi64x(lsb28);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    c20_23 = uint64a_gj_reduc_mm256(c20_23, vmask, reduc);
    if(c24 & lsb28) c24 ^= c28_reduc;
    if(c25 & lsb28) c25 ^= c28_reduc;
    if(c26 & lsb28) c26 ^= c28_reduc;
    if(c27 & lsb28) c27 ^= c28_reduc;
    mask ^= lsb28;

    uint64_t lsb27 = uint64_t_lsb(c27 & mask);
    if(unlikely(!lsb27)) {
        return -1; // singular
    }

    uint64_t c27_reduc = c27 ^ lsb27;
    reduc = _mm256_set1_epi64x(c27_reduc);
    vmask = _mm256_set1_epi64x(lsb27);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    c20_23 = uint64a_gj_reduc_mm256(c20_23, vmask, reduc);
    if(c24 & lsb27) c24 ^= c27_reduc;
    if(c25 & lsb27) c25 ^= c27_reduc;
    if(c26 & lsb27) c26 ^= c27_reduc;
    mask ^= lsb27;

    uint64_t lsb26 = uint64_t_lsb(c26 & mask);
    if(unlikely(!lsb26)) {
        return -1; // singular
    }

    uint64_t c26_reduc = c26 ^ lsb26;
    reduc = _mm256_set1_epi64x(c26_reduc);
    vmask = _mm256_set1_epi64x(lsb26);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    c20_23 = uint64a_gj_reduc_mm256(c20_23, vmask, reduc);
    if(c24 & lsb26) c24 ^= c26_reduc;
    if(c25 & lsb26) c25 ^= c26_reduc;
    mask ^= lsb26;

    uint64_t lsb25 = uint64_t_lsb(c25 & mask);
    if(unlikely(!lsb25)) {
        return -1; // singular
    }

    uint64_t c25_reduc = c25 ^ lsb25;
    reduc = _mm256_set1_epi64x(c25_reduc);
    vmask = _mm256_set1_epi64x(lsb25);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    c20_23 = uint64a_gj_reduc_mm256(c20_23, vmask, reduc);
    if(c24 & lsb25) c24 ^= c25_reduc;
    uint64_t c20 = _mm256_extract_epi64(c20_23, 0);
    uint64_t c21 = _mm256_extract_epi64(c20_23, 1);
    uint64_t c22 = _mm256_extract_epi64(c20_23, 2);
    uint64_t c23 = _mm256_extract_epi64(c20_23, 3);
    mask ^= lsb25;

    uint64_t lsb24 = uint64_t_lsb(c24 & mask);
    if(unlikely(!lsb24)) {
        return -1; // singular
    }

    uint64_t c24_reduc = c24 ^ lsb24;
    reduc = _mm256_set1_epi64x(c24_reduc);
    vmask = _mm256_set1_epi64x(lsb24);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    if(c20 & lsb24) c20 ^= c24_reduc;
    if(c21 & lsb24) c21 ^= c24_reduc;
    if(c22 & lsb24) c22 ^= c24_reduc;
    if(c23 & lsb24) c23 ^= c24_reduc;
    mask ^= lsb24;

    uint64_t lsb23 = uint64_t_lsb(c23 & mask);
    if(unlikely(!lsb23)) {
        return -1; // singular
    }

    uint64_t c23_reduc = c23 ^ lsb23;
    reduc = _mm256_set1_epi64x(c23_reduc);
    vmask = _mm256_set1_epi64x(lsb23);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    if(c20 & lsb23) c20 ^= c23_reduc;
    if(c21 & lsb23) c21 ^= c23_reduc;
    if(c22 & lsb23) c22 ^= c23_reduc;
    mask ^= lsb23;

    uint64_t lsb22 = uint64_t_lsb(c22 & mask);
    if(unlikely(!lsb22)) {
        return -1; // singular
    }

    uint64_t c22_reduc = c22 ^ lsb22;
    reduc = _mm256_set1_epi64x(c22_reduc);
    vmask = _mm256_set1_epi64x(lsb22);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    if(c20 & lsb22) c20 ^= c22_reduc;
    if(c21 & lsb22) c21 ^= c22_reduc;
    mask ^= lsb22;

    uint64_t lsb21 = uint64_t_lsb(c21 & mask);
    if(unlikely(!lsb21)) {
        return -1; // singular
    }

    uint64_t c21_reduc = c21 ^ lsb21;
    reduc = _mm256_set1_epi64x(c21_reduc);
    vmask = _mm256_set1_epi64x(lsb21);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    c16_19 = uint64a_gj_reduc_mm256(c16_19, vmask, reduc);
    if(c20 & lsb21) c20 ^= c21_reduc;
    uint64_t c16 = _mm256_extract_epi64(c16_19, 0);
    uint64_t c17 = _mm256_extract_epi64(c16_19, 1);
    uint64_t c18 = _mm256_extract_epi64(c16_19, 2);
    uint64_t c19 = _mm256_extract_epi64(c16_19, 3);
    mask ^= lsb21;

    uint64_t lsb20 = uint64_t_lsb(c20 & mask);
    if(unlikely(!lsb20)) {
        return -1; // singular
    }

    uint64_t c20_reduc = c20 ^ lsb20;
    reduc = _mm256_set1_epi64x(c20_reduc);
    vmask = _mm256_set1_epi64x(lsb20);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    if(c16 & lsb20) c16 ^= c20_reduc;
    if(c17 & lsb20) c17 ^= c20_reduc;
    if(c18 & lsb20) c18 ^= c20_reduc;
    if(c19 & lsb20) c19 ^= c20_reduc;
    mask ^= lsb20;

    uint64_t lsb19 = uint64_t_lsb(c19 & mask);
    if(unlikely(!lsb19)) {
        return -1; // singular
    }

    uint64_t c19_reduc = c19 ^ lsb19;
    reduc = _mm256_set1_epi64x(c19_reduc);
    vmask = _mm256_set1_epi64x(lsb19);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    if(c16 & lsb19) c16 ^= c19_reduc;
    if(c17 & lsb19) c17 ^= c19_reduc;
    if(c18 & lsb19) c18 ^= c19_reduc;
    mask ^= lsb19;

    uint64_t lsb18 = uint64_t_lsb(c18 & mask);
    if(unlikely(!lsb18)) {
        return -1; // singular
    }

    uint64_t c18_reduc = c18 ^ lsb18;
    reduc = _mm256_set1_epi64x(c18_reduc);
    vmask = _mm256_set1_epi64x(lsb18);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    if(c16 & lsb18) c16 ^= c18_reduc;
    if(c17 & lsb18) c17 ^= c18_reduc;
    mask ^= lsb18;

    uint64_t lsb17 = uint64_t_lsb(c17 & mask);
    if(unlikely(!lsb17)) {
        return -1; // singular
    }

    uint64_t c17_reduc = c17 ^ lsb17;
    reduc = _mm256_set1_epi64x(c17_reduc);
    vmask = _mm256_set1_epi64x(lsb17);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    c12_15 = uint64a_gj_reduc_mm256(c12_15, vmask, reduc);
    if(c16 & lsb17) c16 ^= c17_reduc;
    uint64_t c12 = _mm256_extract_epi64(c12_15, 0);
    uint64_t c13 = _mm256_extract_epi64(c12_15, 1);
    uint64_t c14 = _mm256_extract_epi64(c12_15, 2);
    uint64_t c15 = _mm256_extract_epi64(c12_15, 3);
    mask ^= lsb17;

    uint64_t lsb16 = uint64_t_lsb(c16 & mask);
    if(unlikely(!lsb16)) {
        return -1; // singular
    }

    uint64_t c16_reduc = c16 ^ lsb16;
    reduc = _mm256_set1_epi64x(c16_reduc);
    vmask = _mm256_set1_epi64x(lsb16);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    if(c12 & lsb16) c12 ^= c16_reduc;
    if(c13 & lsb16) c13 ^= c16_reduc;
    if(c14 & lsb16) c14 ^= c16_reduc;
    if(c15 & lsb16) c15 ^= c16_reduc;
    mask ^= lsb16;

    uint64_t lsb15 = uint64_t_lsb(c15 & mask);
    if(unlikely(!lsb15)) {
        return -1; // singular
    }

    uint64_t c15_reduc = c15 ^ lsb15;
    reduc = _mm256_set1_epi64x(c15_reduc);
    vmask = _mm256_set1_epi64x(lsb15);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    if(c12 & lsb15) c12 ^= c15_reduc;
    if(c13 & lsb15) c13 ^= c15_reduc;
    if(c14 & lsb15) c14 ^= c15_reduc;
    mask ^= lsb15;

    uint64_t lsb14 = uint64_t_lsb(c14 & mask);
    if(unlikely(!lsb14)) {
        return -1; // singular
    }

    uint64_t c14_reduc = c14 ^ lsb14;
    reduc = _mm256_set1_epi64x(c14_reduc);
    vmask = _mm256_set1_epi64x(lsb14);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    if(c12 & lsb14) c12 ^= c14_reduc;
    if(c13 & lsb14) c13 ^= c14_reduc;
    mask ^= lsb14;

    uint64_t lsb13 = uint64_t_lsb(c13 & mask);
    if(unlikely(!lsb13)) {
        return -1; // singular
    }

    uint64_t c13_reduc = c13 ^ lsb13;
    reduc = _mm256_set1_epi64x(c13_reduc);
    vmask = _mm256_set1_epi64x(lsb13);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    c8_11 = uint64a_gj_reduc_mm256(c8_11, vmask, reduc);
    if(c12 & lsb13) c12 ^= c13_reduc;
    uint64_t c8 = _mm256_extract_epi64(c8_11, 0);
    uint64_t c9 = _mm256_extract_epi64(c8_11, 1);
    uint64_t c10 = _mm256_extract_epi64(c8_11, 2);
    uint64_t c11 = _mm256_extract_epi64(c8_11, 3);
    mask ^= lsb13;

    uint64_t lsb12 = uint64_t_lsb(c12 & mask);
    if(unlikely(!lsb12)) {
        return -1; // singular
    }

    uint64_t c12_reduc = c12 ^ lsb12;
    reduc = _mm256_set1_epi64x(c12_reduc);
    vmask = _mm256_set1_epi64x(lsb12);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb12) c8 ^= c12_reduc;
    if(c9 & lsb12) c9 ^= c12_reduc;
    if(c10 & lsb12) c10 ^= c12_reduc;
    if(c11 & lsb12) c11 ^= c12_reduc;
    mask ^= lsb12;

    uint64_t lsb11 = uint64_t_lsb(c11 & mask);
    if(unlikely(!lsb11)) {
        return -1; // singular
    }

    uint64_t c11_reduc = c11 ^ lsb11;
    reduc = _mm256_set1_epi64x(c11_reduc);
    vmask = _mm256_set1_epi64x(lsb11);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb11) c8 ^= c11_reduc;
    if(c9 & lsb11) c9 ^= c11_reduc;
    if(c10 & lsb11) c10 ^= c11_reduc;
    mask ^= lsb11;

    uint64_t lsb10 = uint64_t_lsb(c10 & mask);
    if(unlikely(!lsb10)) {
        return -1; // singular
    }

    uint64_t c10_reduc = c10 ^ lsb10;
    reduc = _mm256_set1_epi64x(c10_reduc);
    vmask = _mm256_set1_epi64x(lsb10);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb10) c8 ^= c10_reduc;
    if(c9 & lsb10) c9 ^= c10_reduc;
    mask ^= lsb10;

    uint64_t lsb9 = uint64_t_lsb(c9 & mask);
    if(unlikely(!lsb9)) {
        return -1; // singular
    }

    uint64_t c9_reduc = c9 ^ lsb9;
    reduc = _mm256_set1_epi64x(c9_reduc);
    vmask = _mm256_set1_epi64x(lsb9);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    c4_7 = uint64a_gj_reduc_mm256(c4_7, vmask, reduc);
    if(c8 & lsb9) c8 ^= c9_reduc;
    uint64_t c4 = _mm256_extract_epi64(c4_7, 0);
    uint64_t c5 = _mm256_extract_epi64(c4_7, 1);
    uint64_t c6 = _mm256_extract_epi64(c4_7, 2);
    uint64_t c7 = _mm256_extract_epi64(c4_7, 3);
    mask ^= lsb9;

    uint64_t lsb8 = uint64_t_lsb(c8 & mask);
    if(unlikely(!lsb8)) {
        return -1; // singular
    }

    uint64_t c8_reduc = c8 ^ lsb8;
    reduc = _mm256_set1_epi64x(c8_reduc);
    vmask = _mm256_set1_epi64x(lsb8);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb8) c4 ^= c8_reduc;
    if(c5 & lsb8) c5 ^= c8_reduc;
    if(c6 & lsb8) c6 ^= c8_reduc;
    if(c7 & lsb8) c7 ^= c8_reduc;
    mask ^= lsb8;

    uint64_t lsb7 = uint64_t_lsb(c7 & mask);
    if(unlikely(!lsb7)) {
        return -1; // singular
    }

    uint64_t c7_reduc = c7 ^ lsb7;
    reduc = _mm256_set1_epi64x(c7_reduc);
    vmask = _mm256_set1_epi64x(lsb7);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb7) c4 ^= c7_reduc;
    if(c5 & lsb7) c5 ^= c7_reduc;
    if(c6 & lsb7) c6 ^= c7_reduc;
    mask ^= lsb7;

    uint64_t lsb6 = uint64_t_lsb(c6 & mask);
    if(unlikely(!lsb6)) {
        return -1; // singular
    }

    uint64_t c6_reduc = c6 ^ lsb6;
    reduc = _mm256_set1_epi64x(c6_reduc);
    vmask = _mm256_set1_epi64x(lsb6);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb6) c4 ^= c6_reduc;
    if(c5 & lsb6) c5 ^= c6_reduc;
    mask ^= lsb6;

    uint64_t lsb5 = uint64_t_lsb(c5 & mask);
    if(unlikely(!lsb5)) {
        return -1; // singular
    }

    uint64_t c5_reduc = c5 ^ lsb5;
    reduc = _mm256_set1_epi64x(c5_reduc);
    vmask = _mm256_set1_epi64x(lsb5);
    c0_3 = uint64a_gj_reduc_mm256(c0_3, vmask, reduc);
    if(c4 & lsb5) c4 ^= c5_reduc;
    uint64_t c0 = _mm256_extract_epi64(c0_3, 0);
    uint64_t c1 = _mm256_extract_epi64(c0_3, 1);
    uint64_t c2 = _mm256_extract_epi64(c0_3, 2);
    uint64_t c3 = _mm256_extract_epi64(c0_3, 3);
    mask ^= lsb5;

    uint64_t lsb4 = uint64_t_lsb(c4 & mask);
    if(unlikely(!lsb4)) {
        return -1; // singular
    }

    uint64_t c4_reduc = c4 ^ lsb4;
    if(c0 & lsb4) c0 ^= c4_reduc;
    if(c1 & lsb4) c1 ^= c4_reduc;
    if(c2 & lsb4) c2 ^= c4_reduc;
    if(c3 & lsb4) c3 ^= c4_reduc;
    mask ^= lsb4;

    uint64_t lsb3 = uint64_t_lsb(c3 & mask);
    if(unlikely(!lsb3)) {
        return -1; // singular
    }

    uint64_t c3_reduc = c3 ^ lsb3;
    if(c0 & lsb3) c0 ^= c3_reduc;
    if(c1 & lsb3) c1 ^= c3_reduc;
    if(c2 & lsb3) c2 ^= c3_reduc;
    mask ^= lsb3;

    uint64_t lsb2 = uint64_t_lsb(c2 & mask);
    if(unlikely(!lsb2)) {
        return -1; // singular
    }

    uint64_t c2_reduc = c2 ^ lsb2;
    if(c0 & lsb2) c0 ^= c2_reduc;
    if(c1 & lsb2) c1 ^= c2_reduc;
    mask ^= lsb2;

    uint64_t lsb1 = uint64_t_lsb(c1 & mask);
    if(unlikely(!lsb1)) {
        return -1; // singular
    }

    if(c0 & lsb1) c0 ^= c1 ^ lsb1;
    mask ^= lsb1;

    if(likely(mask & c0)) { // check system consistency
        return mask & c0; // not solvable
    }

    uint64_t s = 0x0ULL;
    if(c0 & lsb1) s = uint64_t_toggle_at(s, 0);
    if(c0 & lsb2) s = uint64_t_toggle_at(s, 1);
    if(c0 & lsb3) s = uint64_t_toggle_at(s, 2);
    if(c0 & lsb4) s = uint64_t_toggle_at(s, 3);
    if(c0 & lsb5) s = uint64_t_toggle_at(s, 4);
    if(c0 & lsb6) s = uint64_t_toggle_at(s, 5);
    if(c0 & lsb7) s = uint64_t_toggle_at(s, 6);
    if(c0 & lsb8) s = uint64_t_toggle_at(s, 7);
    if(c0 & lsb9) s = uint64_t_toggle_at(s, 8);
    if(c0 & lsb10) s = uint64_t_toggle_at(s, 9);
    if(c0 & lsb11) s = uint64_t_toggle_at(s, 10);
    if(c0 & lsb12) s = uint64_t_toggle_at(s, 11);
    if(c0 & lsb13) s = uint64_t_toggle_at(s, 12);
    if(c0 & lsb14) s = uint64_t_toggle_at(s, 13);
    if(c0 & lsb15) s = uint64_t_toggle_at(s, 14);
    if(c0 & lsb16) s = uint64_t_toggle_at(s, 15);
    if(c0 & lsb17) s = uint64_t_toggle_at(s, 16);
    if(c0 & lsb18) s = uint64_t_toggle_at(s, 17);
    if(c0 & lsb19) s = uint64_t_toggle_at(s, 18);
    if(c0 & lsb20) s = uint64_t_toggle_at(s, 19);
    if(c0 & lsb21) s = uint64_t_toggle_at(s, 20);
    if(c0 & lsb22) s = uint64_t_toggle_at(s, 21);
    if(c0 & lsb23) s = uint64_t_toggle_at(s, 22);
    if(c0 & lsb24) s = uint64_t_toggle_at(s, 23);
    if(c0 & lsb25) s = uint64_t_toggle_at(s, 24);
    if(c0 & lsb26) s = uint64_t_toggle_at(s, 25);
    if(c0 & lsb27) s = uint64_t_toggle_at(s, 26);
    if(c0 & lsb28) s = uint64_t_toggle_at(s, 27);
    if(c0 & lsb29) s = uint64_t_toggle_at(s, 28);
    if(c0 & lsb30) s = uint64_t_toggle_at(s, 29);
    if(c0 & lsb31) s = uint64_t_toggle_at(s, 30);
    if(c0 & lsb32) s = uint64_t_toggle_at(s, 31);
    *sol = s;
    return 0;
}

#endif // __AVX__
