/* uint64a.h: header file for bit operations on an array of uint64_t */

#ifndef __BLK_LANCZOS_UINT64A_H__
#define __BLK_LANCZOS_UINT64A_H__

#include <stdint.h>
#include <assert.h>

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__BMI__)
#include <immintrin.h>
#endif

#include "util.h"
#include "bitmap_table.h"

/* usage: Given 1 uint64_t a, find indices of all set bits in a
 * params:
 *      1) a: uint64_t
 *      2) res: an uint8_t array for storing the indices, must hold at least
 *              64 elements
 * return: the number of set bits */
static inline uint64_t
uint64_t_sbpos(uint64_t a, uint8_t* res) {
    return sbidx_in_64b_sz8(res, 0x0ULL, a);
}

/* usage: Given 1 uint64_t a, extract its lowest set bit. If a is 0, return 0
 * params:
 *      1) a: uint64_t
 * return: uint64_t holding the lowest set bit. 0 if a is 0. */
static inline uint64_t pure_func
uint64_t_lsb(uint64_t a) {
#if defined(__BMI__)
    return _blsi_u64(a);
#else
    uint64_t na = -a;
    return (a & na);
#endif
}

/* usage: Given an index i, and an array of uint64_t, set every elements
 *      in the array to 0x1ULL << i
 * params:
 *      2) dst: an array of uint64_t; storage for the result
 *      1) i: uint64_t; offset to shift left to
 * return: void */
static inline void
uint64a_mask_gen_4s(uint64_t dst[4], uint64_t i) {
#if defined(__AVX2__)
    const __m256i v = _mm256_set1_epi64x(0x1ULL);
    __m256i res = _mm256_slli_epi64(v, i);
    _mm256_storeu_si256((__m256i*) dst, res);
#else
    uint64_t v = 0x1ULL << i;
    dst[0] = v; dst[1] = v; dst[2] = v; dst[3] = v;
#endif
}

/* usage: Given 1 uint64_t a, clear its lowest set bit. If a is 0, return 0
 * params:
 *      1) a: uint64_t
 * return: the resultant uint64_t */
static inline uint64_t pure_func
uint64_t_clear_lsb(uint64_t a) {
#if defined(__BMI__)
    return _blsr_u64(a);
#else
    return (a & (a - 1ULL));
#endif
}

/* usage: Given 1 uint64_t a, an index i, toggle the i-th bit in a
 * params:
 *      1) a: uint64_t
 *      2) i: index, 0 ~ 63
 * return: the resultant uint64_t */
static inline uint64_t pure_func
uint64_t_toggle_at(uint64_t a, uint64_t i) {
    assert(i < 64);
    return a ^ (0x1ULL << i);
}

/* usage: Given a array of uint64_t, populate it with random values.
 * params:
 *      1) a: ptr to uint64_t
 *      2) size: size of the array
 * return: void */
static inline void
uint64a_rand(uint64_t* const a, size_t size) {
    for(uint32_t i = 0; i < size; ++i) {
        int* const buf = (int*) (a + i);
        buf[0] = rand();
        buf[1] = rand();
    }
}

static inline void
uint64a_xori_512b(uint64_t* const restrict a,
                  const uint64_t* const restrict b) {
#if defined(__AVX512F__)
    __m512i va = _mm512_load_si512((void const*) a);
    __m512i vb = _mm512_load_si512((void const*) b);
    _mm512_store_si512((void*) a, _mm512_xor_si512(va, vb));
#elif defined(__AVX2__)
    __m256i va0 = _mm256_load_si256((__m256i*) (a + 0));
    __m256i va1 = _mm256_load_si256((__m256i*) (a + 4));
    __m256i vb0 = _mm256_load_si256((__m256i*) (b + 0));
    __m256i vb1 = _mm256_load_si256((__m256i*) (b + 4));
    _mm256_store_si256((__m256i*) (a + 0), _mm256_xor_si256(va0, vb0));
    _mm256_store_si256((__m256i*) (a + 4), _mm256_xor_si256(va1, vb1));
#elif defined(__AVX__)
    __m256i va0 = _mm256_load_si256((__m256i*) (a + 0));
    __m256i va1 = _mm256_load_si256((__m256i*) (a + 4));
    __m256i vb0 = _mm256_load_si256((__m256i*) (b + 0));
    __m256i vb1 = _mm256_load_si256((__m256i*) (b + 4));
    va0 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(vb0),
                                            _mm256_castsi256_pd(va0)));
    va1 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(vb1),
                                            _mm256_castsi256_pd(va1)));
    _mm256_store_si256((__m256i*) (a + 0), va0);
    _mm256_store_si256((__m256i*) (a + 4), va1);
#else
    a[0] ^= b[0];
    a[1] ^= b[1];
    a[2] ^= b[2];
    a[3] ^= b[3];
    a[4] ^= b[4];
    a[5] ^= b[5];
    a[6] ^= b[6];
    a[7] ^= b[7];
#endif
}

static inline void
uint64a_xori_512b_unalign(uint64_t* const restrict a,
                          const uint64_t* const restrict b) {
#if defined(__AVX512F__)
    __m512i va = _mm512_loadu_si512((void const*) a);
    __m512i vb = _mm512_loadu_si512((void const*) b);
    _mm512_storeu_si512((void*) a, _mm512_xor_si512(va, vb));
#elif defined(__AVX2__)
    __m256i va0 = _mm256_loadu_si256((__m256i*) (a + 0));
    __m256i va1 = _mm256_loadu_si256((__m256i*) (a + 4));
    __m256i vb0 = _mm256_loadu_si256((__m256i*) (b + 0));
    __m256i vb1 = _mm256_loadu_si256((__m256i*) (b + 4));
    _mm256_storeu_si256((__m256i*) (a + 0), _mm256_xor_si256(va0, vb0));
    _mm256_storeu_si256((__m256i*) (a + 4), _mm256_xor_si256(va1, vb1));
#elif defined(__AVX__)
    __m256i va0 = _mm256_loadu_si256((__m256i*) (a + 0));
    __m256i va1 = _mm256_loadu_si256((__m256i*) (a + 4));
    __m256i vb0 = _mm256_loadu_si256((__m256i*) (b + 0));
    __m256i vb1 = _mm256_loadu_si256((__m256i*) (b + 4));
    va0 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(vb0),
                                            _mm256_castsi256_pd(va0)));
    va1 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(vb1),
                                            _mm256_castsi256_pd(va1)));
    _mm256_storeu_si256((__m256i*) (a + 0), va0);
    _mm256_storeu_si256((__m256i*) (a + 4), va1);
#else
    a[0] ^= b[0];
    a[1] ^= b[1];
    a[2] ^= b[2];
    a[3] ^= b[3];
    a[4] ^= b[4];
    a[5] ^= b[5];
    a[6] ^= b[6];
    a[7] ^= b[7];
#endif
}

static inline void
uint64a_xori_256b(uint64_t* const restrict a,
                  const uint64_t* const restrict b) {
#if defined(__AVX2__)
    __m256i va0 = _mm256_load_si256((__m256i*) (a + 0));
    __m256i vb0 = _mm256_load_si256((__m256i*) (b + 0));
    _mm256_store_si256((__m256i*) (a + 0), _mm256_xor_si256(va0, vb0));
#elif defined(__AVX__)
    __m256i va0 = _mm256_load_si256((__m256i*) (a + 0));
    __m256i vb0 = _mm256_load_si256((__m256i*) (b + 0));
    va0 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(vb0),
                                            _mm256_castsi256_pd(va0)));
    _mm256_store_si256((__m256i*) (a + 0), va0);
#else
    a[0] ^= b[0];
    a[1] ^= b[1];
    a[2] ^= b[2];
    a[3] ^= b[3];
#endif
}

static inline void
uint64a_xori_256b_unalign(uint64_t* const restrict a,
                          const uint64_t* const restrict b) {
#if defined(__AVX2__)
    __m256i va0 = _mm256_loadu_si256((__m256i*) (a + 0));
    __m256i vb0 = _mm256_loadu_si256((__m256i*) (b + 0));
    _mm256_storeu_si256((__m256i*) (a + 0), _mm256_xor_si256(va0, vb0));
#elif defined(__AVX__)
    __m256i va0 = _mm256_loadu_si256((__m256i*) (a + 0));
    __m256i vb0 = _mm256_loadu_si256((__m256i*) (b + 0));
    va0 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(vb0),
                                            _mm256_castsi256_pd(va0)));
    _mm256_storeu_si256((__m256i*) (a + 0), va0);
#else
    a[0] ^= b[0];
    a[1] ^= b[1];
    a[2] ^= b[2];
    a[3] ^= b[3];
#endif
}

/* usage: Given 2 ptrs to uint64_t a and b, compute a^b and store the result
 *      into a (in place xor). Note that the ptrs must hold addresses with
 *      alignment 64.
 * params:
 *      1) a: ptr to uint64_t
 *      2) b: ptr to uint64_t
 *      3) slot_num: number of uint64_t used to store bits of a; must
 *              be the same for b
 * return: void */
static inline void
uint64a_xori(uint64_t* const restrict a, const uint64_t* const restrict b,
             uint64_t slot_num) {
#if defined(__AVX512F__)
    uint64_t head = slot_num & ~0x1FULL; // multiple of 32
    for(uint64_t i = 0; i < head; i += 16) {
        uint64a_xori_512b(a + i +  0, b + i +  0);
        uint64a_xori_512b(a + i +  8, b + i +  8);
    }
#elif defined(__AVX__)
    uint64_t head = slot_num & ~0xFULL; // multiple of 16
    for(uint64_t i = 0; i < head; i += 16) {
        uint64a_xori_256b(a + i +  0, b + i +  0);
        uint64a_xori_256b(a + i +  4, b + i +  4);
        uint64a_xori_256b(a + i +  8, b + i +  8);
        uint64a_xori_256b(a + i + 12, b + i + 12);
    }
#else
    uint64_t head = slot_num & ~0x3ULL; // multiple of 4
    for(uint64_t i = 0; i < head; i += 4) {
        a[i + 0] ^= b[i + 0];
        a[i + 1] ^= b[i + 1];
        a[i + 2] ^= b[i + 2];
        a[i + 3] ^= b[i + 3];
    }
#endif
    for(uint64_t i = head; i < slot_num; ++i) {
        a[i] ^= b[i];
    }
}

/* usage: Given 2 ptrs to uint64_t a and b, compute a^b and store the result
 *      into a (in place xor).  The addresses stored by the ptrs do not need to
 *      have special alignment.
 * params:
 *      1) a: ptr to uint64_t
 *      2) b: ptr to uint64_t
 *      3) slot_num: number of uint64_t used to store bits of a; must
 *              be the same for b
 * return: void */
#if defined(__AVX512F__)

#define uint64a_xori_unalign(a, b, slot_num) do { \
    uint64_t* aa = (a); \
    const uint64_t* bb = (b); \
    const uint64_t ss = (slot_num); \
    const uint64_t head = ss & ~0xFULL; \
    for(uint64_t i = 0; i < head; i += 16) { \
        uint64a_xori_512b_unalign(aa + i +  0, bb + i +  0); \
        uint64a_xori_512b_unalign(aa + i +  8, bb + i +  8); \
    } \
    for(uint64_t i = head; i < ss; ++i) { \
        aa[i] ^= bb[i]; \
    } \
} while(0)

#elif defined(__AVX__)

#define uint64a_xori_unalign(a, b, slot_num) do { \
    uint64_t* aa = (a); \
    const uint64_t* bb = (b); \
    const uint64_t ss = (slot_num); \
    const uint64_t head = ss & ~0xFULL; \
    for(uint64_t i = 0; i < head; i += 16) { \
        uint64a_xori_256b_unalign(aa + i +  0, bb + i +  0); \
        uint64a_xori_256b_unalign(aa + i +  4, bb + i +  4); \
        uint64a_xori_256b_unalign(aa + i +  8, bb + i +  8); \
        uint64a_xori_256b_unalign(aa + i + 12, bb + i + 12); \
    } \
    for(uint64_t i = head; i < ss; ++i) { \
        aa[i] ^= bb[i]; \
    } \
} while(0)

#else

#define uint64a_xori_unalign(a, b, slot_num) do { \
    uint64_t* aa = (a); \
    const uint64_t* bb = (b); \
    const uint64_t ss = (slot_num); \
    const uint64_t head = ss & ~0x3ULL; \
    for(uint64_t i = 0; i < head; i += 4) { \
        aa[i + 0] ^= bb[i + 0]; \
        aa[i + 1] ^= bb[i + 1]; \
        aa[i + 2] ^= bb[i + 2]; \
        aa[i + 3] ^= bb[i + 3]; \
    } \
    for(uint64_t i = head; i < ss; ++i) { \
        aa[i] ^= bb[i]; \
    } \
} while(0)

#endif

/* usage: Given 2 ptrs to uint64_t a and b of size x, compute a^b and store
 *      the result into a (in place xor).  The addresses stored by the ptrs do
 *      not need to have special alignment.
 * params:
 *      1) a: ptr to uint64_t
 *      2) b: ptr to uint64_t
 *      3) slot_num: number of uint64_t used to store bits of a; must
 *              be the same for b
 * return: void */
static inline void
uint64a_xori_1s_unalign(uint64_t* const restrict a,
                        const uint64_t* const restrict b) {
    a[0] ^= b[0];
}

static inline void
uint64a_xori_2s_unalign(uint64_t* const restrict a,
                        const uint64_t* const restrict b) {
    a[0] ^= b[0];
    a[1] ^= b[1];
}

static inline void
uint64a_xori_3s_unalign(uint64_t* const restrict a,
                        const uint64_t* const restrict b) {
    a[0] ^= b[0];
    a[1] ^= b[1];
    a[2] ^= b[2];
}

static inline void
uint64a_xori_4s_unalign(uint64_t* const restrict a,
                        const uint64_t* const restrict b) {
    uint64a_xori_256b_unalign(a, b);
}

static inline void
uint64a_xori_5s_unalign(uint64_t* const restrict a,
                        const uint64_t* const restrict b) {
    uint64a_xori_4s_unalign(a, b);
    a[4] ^= b[4];
}

static inline void
uint64a_xori_6s_unalign(uint64_t* const restrict a,
                        const uint64_t* const restrict b) {
    uint64a_xori_4s_unalign(a, b);
    a[4] ^= b[4];
    a[5] ^= b[5];
}

static inline void
uint64a_xori_7s_unalign(uint64_t* const restrict a,
                        const uint64_t* const restrict b) {
    uint64a_xori_4s_unalign(a, b);
    a[4] ^= b[4];
    a[5] ^= b[5];
    a[6] ^= b[6];
}

static inline void
uint64a_xori_8s_unalign(uint64_t* const restrict a,
                         const uint64_t* const restrict b) {
#if defined(__AVX512F__)
    uint64a_xori_512b_unalign(a + 0, b + 0);
#else
    uint64a_xori_256b_unalign(a + 0, b + 0);
    uint64a_xori_256b_unalign(a + 4, b + 4);
#endif
}

static inline void
uint64a_xori_9s_unalign(uint64_t* const restrict a,
                        const uint64_t* const restrict b) {
    uint64a_xori_8s_unalign(a, b);
    a[8] ^= b[8];
}

static inline void
uint64a_xori_10s_unalign(uint64_t* const restrict a,
                         const uint64_t* const restrict b) {
    uint64a_xori_8s_unalign(a, b);
    a[8] ^= b[8];
    a[9] ^= b[9];
}

static inline void
uint64a_xori_11s_unalign(uint64_t* const restrict a,
                         const uint64_t* const restrict b) {
    uint64a_xori_8s_unalign(a, b);
    a[ 8] ^= b[ 8];
    a[ 9] ^= b[ 9];
    a[10] ^= b[10];
}

static inline void
uint64a_xori_12s_unalign(uint64_t* const restrict a,
                         const uint64_t* const restrict b) {
    uint64a_xori_8s_unalign(a, b);
    uint64a_xori_4s_unalign(a + 8, b + 8);
}

static inline void
uint64a_xori_13s_unalign(uint64_t* const restrict a,
                         const uint64_t* const restrict b) {
    uint64a_xori_8s_unalign(a, b);
    uint64a_xori_4s_unalign(a + 8, b + 8);
    a[12] ^= b[12];
}

static inline void
uint64a_xori_14s_unalign(uint64_t* const restrict a,
                         const uint64_t* const restrict b) {
    uint64a_xori_8s_unalign(a, b);
    uint64a_xori_4s_unalign(a + 8, b + 8);
    a[12] ^= b[12];
    a[13] ^= b[13];
}

static inline void
uint64a_xori_15s_unalign(uint64_t* const restrict a,
                         const uint64_t* const restrict b) {
    uint64a_xori_8s_unalign(a, b);
    uint64a_xori_4s_unalign(a + 8, b + 8);
    a[12] ^= b[12];
    a[13] ^= b[13];
    a[14] ^= b[14];
}

static inline void
uint64a_xori_16s_unalign(uint64_t* const restrict a,
                         const uint64_t* const restrict b) {
#if defined(__AVX512F__)
    uint64a_xori_512b_unalign(a + 0, b + 0);
    uint64a_xori_512b_unalign(a + 8, b + 8);
#else
    uint64a_xori_256b_unalign(a +  0, b +  0);
    uint64a_xori_256b_unalign(a +  4, b +  4);
    uint64a_xori_256b_unalign(a +  8, b +  8);
    uint64a_xori_256b_unalign(a + 12, b + 12);
#endif
}

static inline void
uint64a_xori_17s_unalign(uint64_t* const restrict a,
                         const uint64_t* const restrict b) {
    uint64a_xori_16s_unalign(a, b);
    a[16] ^= b[16];
}

static inline void
uint64a_xori_18s_unalign(uint64_t* const restrict a,
                         const uint64_t* const restrict b) {
    uint64a_xori_16s_unalign(a, b);
    a[16] ^= b[16];
    a[17] ^= b[17];
}

static inline void
uint64a_xori_19s_unalign(uint64_t* const restrict a,
                         const uint64_t* const restrict b) {
    uint64a_xori_16s_unalign(a, b);
    a[16] ^= b[16];
    a[17] ^= b[17];
    a[18] ^= b[18];
}

static inline void
uint64a_xori_20s_unalign(uint64_t* const restrict a,
                         const uint64_t* const restrict b) {
    uint64a_xori_16s_unalign(a, b);
    uint64a_xori_4s_unalign(a + 16, b + 16);
}

static inline void
uint64a_xori_21s_unalign(uint64_t* const restrict a,
                         const uint64_t* const restrict b) {
    uint64a_xori_16s_unalign(a, b);
    uint64a_xori_4s_unalign(a + 16, b + 16);
    a[20] ^= b[20];
}

static inline void
uint64a_xori_22s_unalign(uint64_t* const restrict a,
                         const uint64_t* const restrict b) {
    uint64a_xori_16s_unalign(a, b);
    uint64a_xori_4s_unalign(a + 16, b + 16);
    a[20] ^= b[20];
    a[21] ^= b[21];
}

static inline void
uint64a_xori_23s_unalign(uint64_t* const restrict a,
                         const uint64_t* const restrict b) {
    uint64a_xori_16s_unalign(a, b);
    uint64a_xori_4s_unalign(a + 16, b + 16);
    a[20] ^= b[20];
    a[21] ^= b[21];
    a[22] ^= b[22];
}

static inline void
uint64a_xori_24s_unalign(uint64_t* const restrict a,
                         const uint64_t* const restrict b) {
    uint64a_xori_16s_unalign(a, b);
    uint64a_xori_8s_unalign(a + 16, b + 16);
}

static inline void
uint64a_xori_25s_unalign(uint64_t* const restrict a,
                         const uint64_t* const restrict b) {
    uint64a_xori_16s_unalign(a, b);
    uint64a_xori_8s_unalign(a + 16, b + 16);
    a[24] ^= b[24];
}

static inline void
uint64a_xori_26s_unalign(uint64_t* const restrict a,
                         const uint64_t* const restrict b) {
    uint64a_xori_16s_unalign(a, b);
    uint64a_xori_8s_unalign(a + 16, b + 16);
    a[24] ^= b[24];
    a[25] ^= b[25];
}

static inline void
uint64a_xori_27s_unalign(uint64_t* const restrict a,
                         const uint64_t* const restrict b) {
    uint64a_xori_16s_unalign(a, b);
    uint64a_xori_8s_unalign(a + 16, b + 16);
    a[24] ^= b[24];
    a[25] ^= b[25];
    a[26] ^= b[26];
}

static inline void
uint64a_xori_28s_unalign(uint64_t* const restrict a,
                         const uint64_t* const restrict b) {
    uint64a_xori_16s_unalign(a, b);
    uint64a_xori_8s_unalign(a + 16, b + 16);
    uint64a_xori_4s_unalign(a + 24, b + 24);
}

static inline void
uint64a_xori_29s_unalign(uint64_t* const restrict a,
                         const uint64_t* const restrict b) {
    uint64a_xori_16s_unalign(a, b);
    uint64a_xori_8s_unalign(a + 16, b + 16);
    uint64a_xori_4s_unalign(a + 24, b + 24);
    a[28] ^= b[28];
}

static inline void
uint64a_xori_30s_unalign(uint64_t* const restrict a,
                         const uint64_t* const restrict b) {
    uint64a_xori_16s_unalign(a, b);
    uint64a_xori_8s_unalign(a + 16, b + 16);
    uint64a_xori_4s_unalign(a + 24, b + 24);
    a[28] ^= b[28];
    a[29] ^= b[29];
}

static inline void
uint64a_xori_31s_unalign(uint64_t* const restrict a,
                         const uint64_t* const restrict b) {
    uint64a_xori_16s_unalign(a, b);
    uint64a_xori_8s_unalign(a + 16, b + 16);
    uint64a_xori_4s_unalign(a + 24, b + 24);
    a[28] ^= b[28];
    a[29] ^= b[29];
    a[30] ^= b[30];
}

static inline void
uint64a_xori_32s_unalign(uint64_t* const restrict a,
                         const uint64_t* const restrict b) {
#if defined(__AVX512F__)
    uint64a_xori_512b_unalign(a +  0, b +  0);
    uint64a_xori_512b_unalign(a +  8, b +  8);
    uint64a_xori_512b_unalign(a + 16, b + 16);
    uint64a_xori_512b_unalign(a + 24, b + 24);
#else
    uint64a_xori_256b_unalign(a +  0, b +  0);
    uint64a_xori_256b_unalign(a +  4, b +  4);
    uint64a_xori_256b_unalign(a +  8, b +  8);
    uint64a_xori_256b_unalign(a + 12, b + 12);
    uint64a_xori_256b_unalign(a + 16, b + 16);
    uint64a_xori_256b_unalign(a + 20, b + 20);
    uint64a_xori_256b_unalign(a + 24, b + 24);
    uint64a_xori_256b_unalign(a + 28, b + 28);
#endif
}

static inline void
uint64a_xori_33s_unalign(uint64_t* const restrict a,
                         const uint64_t* const restrict b) {
    uint64a_xori_32s_unalign(a, b);
    a[32] ^= b[32];
}

static inline void
uint64a_andi_512b(uint64_t* const restrict a,
                  const uint64_t* const restrict b) {
#if defined(__AVX512F__)
    __m512i va = _mm512_load_si512((void const*) a);
    __m512i vb = _mm512_load_si512((void const*) b);
    _mm512_store_si512((void*) a, _mm512_and_si512(va, vb));
#elif defined(__AVX2__)
    __m256i va0 = _mm256_load_si256((__m256i*) (a + 0));
    __m256i va1 = _mm256_load_si256((__m256i*) (a + 4));
    __m256i vb0 = _mm256_load_si256((__m256i*) (b + 0));
    __m256i vb1 = _mm256_load_si256((__m256i*) (b + 4));
    _mm256_store_si256((__m256i*) (a + 0), _mm256_and_si256(va0, vb0));
    _mm256_store_si256((__m256i*) (a + 4), _mm256_and_si256(va1, vb1));
#elif defined(__AVX__)
    __m256i va0 = _mm256_load_si256((__m256i*) (a + 0));
    __m256i va1 = _mm256_load_si256((__m256i*) (a + 4));
    __m256i vb0 = _mm256_load_si256((__m256i*) (b + 0));
    __m256i vb1 = _mm256_load_si256((__m256i*) (b + 4));
    va0 = _mm256_castpd_si256(_mm256_and_pd(_mm256_castsi256_pd(vb0),
                                            _mm256_castsi256_pd(va0)));
    va1 = _mm256_castpd_si256(_mm256_and_pd(_mm256_castsi256_pd(vb1),
                                            _mm256_castsi256_pd(va1)));
    _mm256_store_si256((__m256i*) (a + 0), va0);
    _mm256_store_si256((__m256i*) (a + 4), va1);
#else
    a[0] &= b[0];
    a[1] &= b[1];
    a[2] &= b[2];
    a[3] &= b[3];
    a[4] &= b[4];
    a[5] &= b[5];
    a[6] &= b[6];
    a[7] &= b[7];
#endif
}

static inline void
uint64a_ori_512b(uint64_t* const restrict a,
                 const uint64_t* const restrict b) {
#if defined(__AVX512F__)
    __m512i va = _mm512_load_si512((void const*) a);
    __m512i vb = _mm512_load_si512((void const*) b);
    _mm512_store_si512((void*) a, _mm512_or_si512(va, vb));
#elif defined(__AVX2__)
    __m256i va0 = _mm256_load_si256((__m256i*) (a + 0));
    __m256i va1 = _mm256_load_si256((__m256i*) (a + 4));
    __m256i vb0 = _mm256_load_si256((__m256i*) (b + 0));
    __m256i vb1 = _mm256_load_si256((__m256i*) (b + 4));
    _mm256_store_si256((__m256i*) (a + 0), _mm256_or_si256(va0, vb0));
    _mm256_store_si256((__m256i*) (a + 4), _mm256_or_si256(va1, vb1));
#elif defined(__AVX__)
    __m256i va0 = _mm256_load_si256((__m256i*) (a + 0));
    __m256i va1 = _mm256_load_si256((__m256i*) (a + 4));
    __m256i vb0 = _mm256_load_si256((__m256i*) (b + 0));
    __m256i vb1 = _mm256_load_si256((__m256i*) (b + 4));
    va0 = _mm256_castpd_si256(_mm256_or_pd(_mm256_castsi256_pd(vb0),
                                            _mm256_castsi256_pd(va0)));
    va1 = _mm256_castpd_si256(_mm256_or_pd(_mm256_castsi256_pd(vb1),
                                            _mm256_castsi256_pd(va1)));
    _mm256_store_si256((__m256i*) (a + 0), va0);
    _mm256_store_si256((__m256i*) (a + 4), va1);
#else
    a[0] |= b[0];
    a[1] |= b[1];
    a[2] |= b[2];
    a[3] |= b[3];
    a[4] |= b[4];
    a[5] |= b[5];
    a[6] |= b[6];
    a[7] |= b[7];
#endif
}

static inline void
uint64a_ori_512b_unalign(uint64_t* const restrict a,
                         const uint64_t* const restrict b) {
#if defined(__AVX512F__)
    __m512i va = _mm512_loadu_si512((void const*) a);
    __m512i vb = _mm512_loadu_si512((void const*) b);
    _mm512_storeu_si512((void*) a, _mm512_or_si512(va, vb));
#elif defined(__AVX2__)
    __m256i va0 = _mm256_loadu_si256((__m256i*) (a + 0));
    __m256i va1 = _mm256_loadu_si256((__m256i*) (a + 4));
    __m256i vb0 = _mm256_loadu_si256((__m256i*) (b + 0));
    __m256i vb1 = _mm256_loadu_si256((__m256i*) (b + 4));
    _mm256_storeu_si256((__m256i*) (a + 0), _mm256_or_si256(va0, vb0));
    _mm256_storeu_si256((__m256i*) (a + 4), _mm256_or_si256(va1, vb1));
#elif defined(__AVX__)
    __m256i va0 = _mm256_loadu_si256((__m256i*) (a + 0));
    __m256i va1 = _mm256_loadu_si256((__m256i*) (a + 4));
    __m256i vb0 = _mm256_loadu_si256((__m256i*) (b + 0));
    __m256i vb1 = _mm256_loadu_si256((__m256i*) (b + 4));
    va0 = _mm256_castpd_si256(_mm256_or_pd(_mm256_castsi256_pd(vb0),
                                            _mm256_castsi256_pd(va0)));
    va1 = _mm256_castpd_si256(_mm256_or_pd(_mm256_castsi256_pd(vb1),
                                            _mm256_castsi256_pd(va1)));
    _mm256_storeu_si256((__m256i*) (a + 0), va0);
    _mm256_storeu_si256((__m256i*) (a + 4), va1);
#else
    a[0] |= b[0];
    a[1] |= b[1];
    a[2] |= b[2];
    a[3] |= b[3];
    a[4] |= b[4];
    a[5] |= b[5];
    a[6] |= b[6];
    a[7] |= b[7];
#endif
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
uint64a_gj_v1_generic(const uint64_t m[2], uint64_t* const restrict sol);
int64_t
uint64a_gj_v2_generic(const uint64_t m[3], uint64_t* const restrict sol);
int64_t
uint64a_gj_v3_generic(const uint64_t m[4], uint64_t* const restrict sol);
int64_t
uint64a_gj_v4_generic(const uint64_t m[5], uint64_t* const restrict sol);
int64_t
uint64a_gj_v5_generic(const uint64_t m[6], uint64_t* const restrict sol);
int64_t
uint64a_gj_v6_generic(const uint64_t m[7], uint64_t* const restrict sol);
int64_t
uint64a_gj_v7_generic(const uint64_t m[8], uint64_t* const restrict sol);
int64_t
uint64a_gj_v8_generic(const uint64_t m[9], uint64_t* const restrict sol);
int64_t
uint64a_gj_v9_generic(const uint64_t m[10], uint64_t* const restrict sol);
int64_t
uint64a_gj_v10_generic(const uint64_t m[11], uint64_t* const restrict sol);
int64_t
uint64a_gj_v11_generic(const uint64_t m[12], uint64_t* const restrict sol);
int64_t
uint64a_gj_v12_generic(const uint64_t m[13], uint64_t* const restrict sol);
int64_t
uint64a_gj_v13_generic(const uint64_t m[14], uint64_t* const restrict sol);
int64_t
uint64a_gj_v14_generic(const uint64_t m[15], uint64_t* const restrict sol);
int64_t
uint64a_gj_v15_generic(const uint64_t m[16], uint64_t* const restrict sol);
int64_t
uint64a_gj_v16_generic(const uint64_t m[17], uint64_t* const restrict sol);
int64_t
uint64a_gj_v17_generic(const uint64_t m[18], uint64_t* const restrict sol);
int64_t
uint64a_gj_v18_generic(const uint64_t m[19], uint64_t* const restrict sol);
int64_t
uint64a_gj_v19_generic(const uint64_t m[20], uint64_t* const restrict sol);
int64_t
uint64a_gj_v20_generic(const uint64_t m[21], uint64_t* const restrict sol);
int64_t
uint64a_gj_v21_generic(const uint64_t m[22], uint64_t* const restrict sol);
int64_t
uint64a_gj_v22_generic(const uint64_t m[23], uint64_t* const restrict sol);
int64_t
uint64a_gj_v23_generic(const uint64_t m[24], uint64_t* const restrict sol);
int64_t
uint64a_gj_v24_generic(const uint64_t m[25], uint64_t* const restrict sol);
int64_t
uint64a_gj_v25_generic(const uint64_t m[26], uint64_t* const restrict sol);
int64_t
uint64a_gj_v26_generic(const uint64_t m[27], uint64_t* const restrict sol);
int64_t
uint64a_gj_v27_generic(const uint64_t m[28], uint64_t* const restrict sol);
int64_t
uint64a_gj_v28_generic(const uint64_t m[29], uint64_t* const restrict sol);
int64_t
uint64a_gj_v29_generic(const uint64_t m[30], uint64_t* const restrict sol);
int64_t
uint64a_gj_v30_generic(const uint64_t m[31], uint64_t* const restrict sol);
int64_t
uint64a_gj_v31_generic(const uint64_t m[32], uint64_t* const restrict sol);
int64_t
uint64a_gj_v32_generic(const uint64_t m[33], uint64_t* const restrict sol);

#if defined(__AVX512F__)
int64_t
uint64a_gj_v9_avx512(const uint64_t m[10], uint64_t* const restrict sol);
int64_t
uint64a_gj_v10_avx512(const uint64_t m[11], uint64_t* const restrict sol);
int64_t
uint64a_gj_v11_avx512(const uint64_t m[12], uint64_t* const restrict sol);
int64_t
uint64a_gj_v12_avx512(const uint64_t m[13], uint64_t* const restrict sol);
int64_t
uint64a_gj_v13_avx512(const uint64_t m[14], uint64_t* const restrict sol);
int64_t
uint64a_gj_v14_avx512(const uint64_t m[15], uint64_t* const restrict sol);
int64_t
uint64a_gj_v15_avx512(const uint64_t m[16], uint64_t* const restrict sol);
int64_t
uint64a_gj_v16_avx512(const uint64_t m[17], uint64_t* const restrict sol);
int64_t
uint64a_gj_v17_avx512(const uint64_t m[18], uint64_t* const restrict sol);
int64_t
uint64a_gj_v18_avx512(const uint64_t m[19], uint64_t* const restrict sol);
int64_t
uint64a_gj_v19_avx512(const uint64_t m[20], uint64_t* const restrict sol);
int64_t
uint64a_gj_v20_avx512(const uint64_t m[21], uint64_t* const restrict sol);
int64_t
uint64a_gj_v21_avx512(const uint64_t m[22], uint64_t* const restrict sol);
int64_t
uint64a_gj_v22_avx512(const uint64_t m[23], uint64_t* const restrict sol);
int64_t
uint64a_gj_v23_avx512(const uint64_t m[24], uint64_t* const restrict sol);
int64_t
uint64a_gj_v24_avx512(const uint64_t m[25], uint64_t* const restrict sol);
int64_t
uint64a_gj_v25_avx512(const uint64_t m[26], uint64_t* const restrict sol);
int64_t
uint64a_gj_v26_avx512(const uint64_t m[27], uint64_t* const restrict sol);
int64_t
uint64a_gj_v27_avx512(const uint64_t m[28], uint64_t* const restrict sol);
int64_t
uint64a_gj_v28_avx512(const uint64_t m[29], uint64_t* const restrict sol);
int64_t
uint64a_gj_v29_avx512(const uint64_t m[30], uint64_t* const restrict sol);
int64_t
uint64a_gj_v30_avx512(const uint64_t m[31], uint64_t* const restrict sol);
int64_t
uint64a_gj_v31_avx512(const uint64_t m[32], uint64_t* const restrict sol);
int64_t
uint64a_gj_v32_avx512(const uint64_t m[33], uint64_t* const restrict sol);
#endif

#if defined(__AVX2__)
int64_t
uint64a_gj_v5_avx2(const uint64_t m[6], uint64_t* const restrict sol);
int64_t
uint64a_gj_v6_avx2(const uint64_t m[7], uint64_t* const restrict sol);
int64_t
uint64a_gj_v7_avx2(const uint64_t m[8], uint64_t* const restrict sol);
int64_t
uint64a_gj_v8_avx2(const uint64_t m[9], uint64_t* const restrict sol);
int64_t
uint64a_gj_v9_avx2(const uint64_t m[10], uint64_t* const restrict sol);
int64_t
uint64a_gj_v10_avx2(const uint64_t m[11], uint64_t* const restrict sol);
int64_t
uint64a_gj_v11_avx2(const uint64_t m[12], uint64_t* const restrict sol);
int64_t
uint64a_gj_v12_avx2(const uint64_t m[13], uint64_t* const restrict sol);
int64_t
uint64a_gj_v13_avx2(const uint64_t m[14], uint64_t* const restrict sol);
int64_t
uint64a_gj_v14_avx2(const uint64_t m[15], uint64_t* const restrict sol);
int64_t
uint64a_gj_v15_avx2(const uint64_t m[16], uint64_t* const restrict sol);
int64_t
uint64a_gj_v16_avx2(const uint64_t m[17], uint64_t* const restrict sol);
int64_t
uint64a_gj_v17_avx2(const uint64_t m[18], uint64_t* const restrict sol);
int64_t
uint64a_gj_v18_avx2(const uint64_t m[19], uint64_t* const restrict sol);
int64_t
uint64a_gj_v19_avx2(const uint64_t m[20], uint64_t* const restrict sol);
int64_t
uint64a_gj_v20_avx2(const uint64_t m[21], uint64_t* const restrict sol);
int64_t
uint64a_gj_v21_avx2(const uint64_t m[22], uint64_t* const restrict sol);
int64_t
uint64a_gj_v22_avx2(const uint64_t m[23], uint64_t* const restrict sol);
int64_t
uint64a_gj_v23_avx2(const uint64_t m[24], uint64_t* const restrict sol);
int64_t
uint64a_gj_v24_avx2(const uint64_t m[25], uint64_t* const restrict sol);
int64_t
uint64a_gj_v25_avx2(const uint64_t m[26], uint64_t* const restrict sol);
int64_t
uint64a_gj_v26_avx2(const uint64_t m[27], uint64_t* const restrict sol);
int64_t
uint64a_gj_v27_avx2(const uint64_t m[28], uint64_t* const restrict sol);
int64_t
uint64a_gj_v28_avx2(const uint64_t m[29], uint64_t* const restrict sol);
int64_t
uint64a_gj_v29_avx2(const uint64_t m[30], uint64_t* const restrict sol);
int64_t
uint64a_gj_v30_avx2(const uint64_t m[31], uint64_t* const restrict sol);
int64_t
uint64a_gj_v31_avx2(const uint64_t m[32], uint64_t* const restrict sol);
int64_t
uint64a_gj_v32_avx2(const uint64_t m[33], uint64_t* const restrict sol);
#endif

#if defined(__AVX__)
int64_t
uint64a_gj_v5_avx(const uint64_t m[6], uint64_t* const restrict sol);
int64_t
uint64a_gj_v6_avx(const uint64_t m[7], uint64_t* const restrict sol);
int64_t
uint64a_gj_v7_avx(const uint64_t m[8], uint64_t* const restrict sol);
int64_t
uint64a_gj_v8_avx(const uint64_t m[9], uint64_t* const restrict sol);
int64_t
uint64a_gj_v9_avx(const uint64_t m[10], uint64_t* const restrict sol);
int64_t
uint64a_gj_v10_avx(const uint64_t m[11], uint64_t* const restrict sol);
int64_t
uint64a_gj_v11_avx(const uint64_t m[12], uint64_t* const restrict sol);
int64_t
uint64a_gj_v12_avx(const uint64_t m[13], uint64_t* const restrict sol);
int64_t
uint64a_gj_v13_avx(const uint64_t m[14], uint64_t* const restrict sol);
int64_t
uint64a_gj_v14_avx(const uint64_t m[15], uint64_t* const restrict sol);
int64_t
uint64a_gj_v15_avx(const uint64_t m[16], uint64_t* const restrict sol);
int64_t
uint64a_gj_v16_avx(const uint64_t m[17], uint64_t* const restrict sol);
int64_t
uint64a_gj_v17_avx(const uint64_t m[18], uint64_t* const restrict sol);
int64_t
uint64a_gj_v18_avx(const uint64_t m[19], uint64_t* const restrict sol);
int64_t
uint64a_gj_v19_avx(const uint64_t m[20], uint64_t* const restrict sol);
int64_t
uint64a_gj_v20_avx(const uint64_t m[21], uint64_t* const restrict sol);
int64_t
uint64a_gj_v21_avx(const uint64_t m[22], uint64_t* const restrict sol);
int64_t
uint64a_gj_v22_avx(const uint64_t m[23], uint64_t* const restrict sol);
int64_t
uint64a_gj_v23_avx(const uint64_t m[24], uint64_t* const restrict sol);
int64_t
uint64a_gj_v24_avx(const uint64_t m[25], uint64_t* const restrict sol);
int64_t
uint64a_gj_v25_avx(const uint64_t m[26], uint64_t* const restrict sol);
int64_t
uint64a_gj_v26_avx(const uint64_t m[27], uint64_t* const restrict sol);
int64_t
uint64a_gj_v27_avx(const uint64_t m[28], uint64_t* const restrict sol);
int64_t
uint64a_gj_v28_avx(const uint64_t m[29], uint64_t* const restrict sol);
int64_t
uint64a_gj_v29_avx(const uint64_t m[30], uint64_t* const restrict sol);
int64_t
uint64a_gj_v30_avx(const uint64_t m[31], uint64_t* const restrict sol);
int64_t
uint64a_gj_v31_avx(const uint64_t m[32], uint64_t* const restrict sol);
int64_t
uint64a_gj_v32_avx(const uint64_t m[33], uint64_t* const restrict sol);
#endif

#define uint64a_gj_v1 uint64a_gj_v1_generic
#define uint64a_gj_v2 uint64a_gj_v2_generic
#define uint64a_gj_v3 uint64a_gj_v3_generic
#define uint64a_gj_v4 uint64a_gj_v4_generic

// number of columns is large enough for optimization
#if defined(__AVX512F__)

#define uint64a_gj_v5 uint64a_gj_v5_avx2
#define uint64a_gj_v6 uint64a_gj_v6_avx2
#define uint64a_gj_v7 uint64a_gj_v7_avx2
#define uint64a_gj_v8 uint64a_gj_v8_avx2

#define uint64a_gj_v9 uint64a_gj_v9_avx512
#define uint64a_gj_v10 uint64a_gj_v10_avx512
#define uint64a_gj_v11 uint64a_gj_v11_avx512
#define uint64a_gj_v12 uint64a_gj_v12_avx512
#define uint64a_gj_v13 uint64a_gj_v13_avx512
#define uint64a_gj_v14 uint64a_gj_v14_avx512
#define uint64a_gj_v15 uint64a_gj_v15_avx512
#define uint64a_gj_v16 uint64a_gj_v16_avx512
#define uint64a_gj_v17 uint64a_gj_v17_avx512
#define uint64a_gj_v18 uint64a_gj_v18_avx512
#define uint64a_gj_v19 uint64a_gj_v19_avx512
#define uint64a_gj_v20 uint64a_gj_v20_avx512
#define uint64a_gj_v21 uint64a_gj_v21_avx512
#define uint64a_gj_v22 uint64a_gj_v22_avx512
#define uint64a_gj_v23 uint64a_gj_v23_avx512
#define uint64a_gj_v24 uint64a_gj_v24_avx512
#define uint64a_gj_v25 uint64a_gj_v25_avx512
#define uint64a_gj_v26 uint64a_gj_v26_avx512
#define uint64a_gj_v27 uint64a_gj_v27_avx512
#define uint64a_gj_v28 uint64a_gj_v28_avx512
#define uint64a_gj_v29 uint64a_gj_v29_avx512
#define uint64a_gj_v30 uint64a_gj_v30_avx512
#define uint64a_gj_v31 uint64a_gj_v31_avx512
#define uint64a_gj_v32 uint64a_gj_v32_avx512

#elif defined(__AVX2__)

#define uint64a_gj_v5 uint64a_gj_v5_avx2
#define uint64a_gj_v6 uint64a_gj_v6_avx2
#define uint64a_gj_v7 uint64a_gj_v7_avx2
#define uint64a_gj_v8 uint64a_gj_v8_avx2
#define uint64a_gj_v9 uint64a_gj_v9_avx2
#define uint64a_gj_v10 uint64a_gj_v10_avx2
#define uint64a_gj_v11 uint64a_gj_v11_avx2
#define uint64a_gj_v12 uint64a_gj_v12_avx2
#define uint64a_gj_v13 uint64a_gj_v13_avx2
#define uint64a_gj_v14 uint64a_gj_v14_avx2
#define uint64a_gj_v15 uint64a_gj_v15_avx2
#define uint64a_gj_v16 uint64a_gj_v16_avx2
#define uint64a_gj_v17 uint64a_gj_v17_avx2
#define uint64a_gj_v18 uint64a_gj_v18_avx2
#define uint64a_gj_v19 uint64a_gj_v19_avx2
#define uint64a_gj_v20 uint64a_gj_v20_avx2
#define uint64a_gj_v21 uint64a_gj_v21_avx2
#define uint64a_gj_v22 uint64a_gj_v22_avx2
#define uint64a_gj_v23 uint64a_gj_v23_avx2
#define uint64a_gj_v24 uint64a_gj_v24_avx2
#define uint64a_gj_v25 uint64a_gj_v25_avx2
#define uint64a_gj_v26 uint64a_gj_v26_avx2
#define uint64a_gj_v27 uint64a_gj_v27_avx2
#define uint64a_gj_v28 uint64a_gj_v28_avx2
#define uint64a_gj_v29 uint64a_gj_v29_avx2
#define uint64a_gj_v30 uint64a_gj_v30_avx2
#define uint64a_gj_v31 uint64a_gj_v31_avx2
#define uint64a_gj_v32 uint64a_gj_v32_avx2

#elif defined(__AVX__)

#define uint64a_gj_v5 uint64a_gj_v5_avx
#define uint64a_gj_v6 uint64a_gj_v6_avx
#define uint64a_gj_v7 uint64a_gj_v7_avx
#define uint64a_gj_v8 uint64a_gj_v8_avx
#define uint64a_gj_v9 uint64a_gj_v9_avx
#define uint64a_gj_v10 uint64a_gj_v10_avx
#define uint64a_gj_v11 uint64a_gj_v11_avx
#define uint64a_gj_v12 uint64a_gj_v12_avx
#define uint64a_gj_v13 uint64a_gj_v13_avx
#define uint64a_gj_v14 uint64a_gj_v14_avx
#define uint64a_gj_v15 uint64a_gj_v15_avx
#define uint64a_gj_v16 uint64a_gj_v16_avx
#define uint64a_gj_v17 uint64a_gj_v17_avx
#define uint64a_gj_v18 uint64a_gj_v18_avx
#define uint64a_gj_v19 uint64a_gj_v19_avx
#define uint64a_gj_v20 uint64a_gj_v20_avx
#define uint64a_gj_v21 uint64a_gj_v21_avx
#define uint64a_gj_v22 uint64a_gj_v22_avx
#define uint64a_gj_v23 uint64a_gj_v23_avx
#define uint64a_gj_v24 uint64a_gj_v24_avx
#define uint64a_gj_v25 uint64a_gj_v25_avx
#define uint64a_gj_v26 uint64a_gj_v26_avx
#define uint64a_gj_v27 uint64a_gj_v27_avx
#define uint64a_gj_v28 uint64a_gj_v28_avx
#define uint64a_gj_v29 uint64a_gj_v29_avx
#define uint64a_gj_v30 uint64a_gj_v30_avx
#define uint64a_gj_v31 uint64a_gj_v31_avx
#define uint64a_gj_v32 uint64a_gj_v32_avx

#else

#define uint64a_gj_v5 uint64a_gj_v5_generic
#define uint64a_gj_v6 uint64a_gj_v6_generic
#define uint64a_gj_v7 uint64a_gj_v7_generic
#define uint64a_gj_v8 uint64a_gj_v8_generic
#define uint64a_gj_v9 uint64a_gj_v9_generic
#define uint64a_gj_v10 uint64a_gj_v10_generic
#define uint64a_gj_v11 uint64a_gj_v11_generic
#define uint64a_gj_v12 uint64a_gj_v12_generic
#define uint64a_gj_v13 uint64a_gj_v13_generic
#define uint64a_gj_v14 uint64a_gj_v14_generic
#define uint64a_gj_v15 uint64a_gj_v15_generic
#define uint64a_gj_v16 uint64a_gj_v16_generic
#define uint64a_gj_v17 uint64a_gj_v17_generic
#define uint64a_gj_v18 uint64a_gj_v18_generic
#define uint64a_gj_v19 uint64a_gj_v19_generic
#define uint64a_gj_v20 uint64a_gj_v20_generic
#define uint64a_gj_v21 uint64a_gj_v21_generic
#define uint64a_gj_v22 uint64a_gj_v22_generic
#define uint64a_gj_v23 uint64a_gj_v23_generic
#define uint64a_gj_v24 uint64a_gj_v24_generic
#define uint64a_gj_v25 uint64a_gj_v25_generic
#define uint64a_gj_v26 uint64a_gj_v26_generic
#define uint64a_gj_v27 uint64a_gj_v27_generic
#define uint64a_gj_v28 uint64a_gj_v28_generic
#define uint64a_gj_v29 uint64a_gj_v29_generic
#define uint64a_gj_v30 uint64a_gj_v30_generic
#define uint64a_gj_v31 uint64a_gj_v31_generic
#define uint64a_gj_v32 uint64a_gj_v32_generic

#endif


#endif  // __BLK_LANCZOS_UINT64A_H__
