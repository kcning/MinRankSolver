#include "gf16.h"
#include "util.h"
#include <string.h> // memset

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#include "bitmap_table.h"
#endif

// pick x^4 + x + 1 as the irreducible polynomial
#define GF16_POLY       (0x13U)     // 0b00010011 = 0x13
#define GF16_POLY_SLI2  (0x4CU)     // 0b01001100 = 0x4c

// NOTE: 0 has no inverse
gf16_t gf16_t_inv_table[16] = {0x0, 0x1, 0x9, 0xE, 0xD, 0xB, 0x7, 0x6, 0xF, 0x2, 0xC, 0x5, 0xA, 0x4, 0x3, 0x8};

gf16_t
gf16_t_reduc_7b(uint8_t v) {
    assert( !(v & 0x80U) );
    if(v & 0x40U) // 7-th bit
        v ^= GF16_POLY << (6 - 4);
    if(v & 0x20U) // 6-th bit
        v ^= GF16_POLY << (5 - 4);
    if(v & 0x10U) // 5-th bit
        v ^= GF16_POLY << (4 - 4);
    return v;
}

void
gf16_t_arr_reduc(gf16_t* arr, uint32_t sz) {
    // TODO: optimize
    for(uint32_t i = 0; i < sz; ++i) {
        arr[i] = gf16_t_reduc_7b(arr[i]);
    }
}

gf16_t
gf16_t_reduc_32b(uint32_t v) {
    for(uint32_t i = 31; i >= 4; --i) {
        uint32_t mask = 0x1U << (i - 4);
        if(v & mask)
            v ^= GF16_POLY << (i - 4);
    }
    return v;
}

gf16_t
gf16_t_add(gf16_t a, gf16_t b) {
    return a ^ b;
}

gf16_t
gf16_t_sub(gf16_t a, gf16_t b) {
    return gf16_t_add(a, b);
}

gf16_t
gf16_t_mul(gf16_t a, gf16_t b) {
    if(a == 0 || b == 0)
        return 0;

    if(a == 1)
        return b;

    if(b == 1)
        return a;

    gf16_t p = 0;
    for(uint32_t i = 0; i < 4; ++i) {
        if(b & 0x1U)
            p ^= a;

        a <<= 1;
        b >>= 1;
    }

    return gf16_t_reduc_7b(p); // p has at most 7 bits
}

gf16_t
gf16_t_square(gf16_t a) {
    return gf16_t_mul(a, a);
}

// TODO: efficient doubling
// https://crypto.stackexchange.com/questions/63158/regular-and-efficient-doubling-in-gf2n

gf16_t
gf16_t_inv_by_table(gf16_t a) {
    return gf16_t_inv_table[a];
}

gf16_t
gf16_t_inv_by_squaring(gf16_t a) {
    // compute a^(16-2)
    if(a == 0)
        return 0;

    if(a == 1)
        return 1;

    gf16_t p2 = gf16_t_square(a);
    gf16_t p4 = gf16_t_square(p2);
    gf16_t p8 = gf16_t_square(p4);
    gf16_t p12 = gf16_t_mul(p8, p4);
    gf16_t p14 = gf16_t_mul(p12, p2);
    return p14;
}

gf16_t
gf16_t_inv(gf16_t a) {
    // NOTE: lookup table seems faster than squaring
    return gf16_t_inv_by_table(a);
    //return gf16_t_inv_by_squaring(a);
}

#if defined(__AVX512F__) && defined(__AVX512BW__)

static force_inline void
mm512_create_mask_from_64b_avx512(__m512i* restrict m, uint64_t d) {
    __m512i mask = _mm512_movm_epi8((__mmask64) d);
    *m = mask;
}

static force_inline void
gf16_t_arr_zero_64b_avx512(gf16_t* a, uint64_t mask) {
    __m512i v = _mm512_loadu_si512(a);
    __m512i m; mm512_create_mask_from_64b_avx512(&m, mask);
    v = _mm512_and_si512(v, m);
    _mm512_storeu_si512(a, v);
}

#elif defined(__AVX2__)

static inline uint64_t
uint64_t_dup_8b(uint8_t b) {
    uint64_t v_quater = ((uint64_t) b) | (((uint64_t) b) << 8);
    uint64_t v_half = v_quater | (v_quater << 16);
    return v_half | (v_half << 32);
}

static force_inline __m256i
mm256_8b_to_32b(uint32_t d) {
    uint8_t d0 = d;
    uint8_t d1 = d >> 8;
    uint8_t d2 = d >> 16;
    uint8_t d3 = d >> 24;
    // NOTE: _mm256_setr_epi8 is slower
    uint64_t v0 = uint64_t_dup_8b(d0);
    uint64_t v1 = uint64_t_dup_8b(d1);
    uint64_t v2 = uint64_t_dup_8b(d2);
    uint64_t v3 = uint64_t_dup_8b(d3);
    return _mm256_setr_epi64x(v0, v1, v2, v3);
}

static force_inline void
mm256_create_mask_from_64b_avx2(__m256i* restrict m0, __m256i* restrict m1,
                                uint64_t d) {
    //const __m256i shuffle_mask = _mm256_loadu_si256((__m256i*) avx2_shuffle_mask);
    const __m256i shuffle_mask = _mm256_set1_epi64x(0x8040201008040201ULL);
    // mask for lower half
    __m256i mask0 = _mm256_and_si256(mm256_8b_to_32b(d), shuffle_mask);
    *m0 = _mm256_cmpeq_epi8(mask0, shuffle_mask);
    // NOTE: _mm256_cmpgt_epi8(mask0, full_zero_vector) doesn't work because
    // each byte is interpreted as a signed 8-bit integer. When the MSB of
    // a byte is set, the integer is treated as negative and thus < 0.

    // mask for upper half
    __m256i mask1 = _mm256_and_si256(mm256_8b_to_32b(d >> 32), shuffle_mask);
    *m1 = _mm256_cmpeq_epi8(mask1, shuffle_mask);
}

static force_inline void
gf16_t_arr_zero_64b_avx2(gf16_t* a, uint64_t mask) {
    __m256i va0 = _mm256_loadu_si256((__m256i*) a);
    __m256i va1 = _mm256_loadu_si256((__m256i*) (a + 32));
    __m256i m0, m1;
    mm256_create_mask_from_64b_avx2(&m0, &m1, mask);
    va0 = _mm256_and_si256(va0, m0);
    va1 = _mm256_and_si256(va1, m1);
    _mm256_storeu_si256((__m256i*) a, va0);
    _mm256_storeu_si256((__m256i*) (a + 32), va1);
}

force_inline void
gf16_t_arr_mask_from_64b_reg_avx2(__m256i* restrict high,
                                  __m256i* restrict low, uint64_t mask) {
    mm256_create_mask_from_64b_avx2(low, high, mask);
}

#endif

void
gf16_t_arr_mask_from_64b(gf16_t* a, uint64_t mask) {
#if defined(__AVX512F__) && defined(__AVX512BW__)
    __m512i m; mm512_create_mask_from_64b_avx512(&m, mask);
    _mm512_storeu_si512(a, m);
#elif defined(__AVX2__)
    __m256i m0, m1;
    mm256_create_mask_from_64b_avx2(&m0, &m1, mask);
    _mm256_storeu_si256((__m256i*) a, m0);
    _mm256_storeu_si256((__m256i*) (a + 32), m1);
#else
    memset(a, 0x0, sizeof(gf16_t) * 64);

    if(unlikely(!mask)) {
        return;
    }

    for(uint32_t i = 0; i < 64; ++i) {
        if(mask & 0x1)
            a[i] = 0xFF;
        mask >>= 1;
    }
#endif
}

void
gf16_t_arr_zero_64b(gf16_t* a, uint64_t mask) {
    if(unlikely(!mask)) {
        memset(a, 0x0, sizeof(gf16_t) * 64);
        return;
    }
#if defined(__AVX512F__) && defined(__AVX512BW__)
    gf16_t_arr_zero_64b_avx512(a, mask);
#elif defined(__AVX2__)
    gf16_t_arr_zero_64b_avx2(a, mask);
#else
    for(uint32_t i = 0; i < 64; ++i) {
        if( !(mask & 0x1) )
            a[i] = 0;
        mask >>= 1;
    }
#endif
}

void
gf16_t_arr_shl(gf16_t* arr, uint32_t sz, uint32_t offset) {
    assert(offset <= 4);
    // TODO: optimize
    for(uint32_t i = 0; i < sz; ++i) {
        arr[i] <<= offset;
    }
}

void
gf16_t_arr_shr(gf16_t* arr, uint32_t sz, uint32_t offset) {
    // TODO: optimize
    assert(offset <= 8);
    for(uint32_t i = 0; i < sz; ++i) {
        arr[i] >>= offset;
    }
}

void
gf16_t_arr_addi(gf16_t* restrict a, gf16_t* restrict b, uint32_t sz) {
    // TODO: optimize
    for(uint32_t i = 0; i < sz; ++i) {
        a[i] ^= b[i];
    }
}

#if defined(__AVX512F__) && defined(__AVX512BW__)

force_inline void
gf16_t_arr_andi_64_reg_avx512(gf16_t* a, __m512i mask) {
    __m512i v = _mm512_loadu_si512(a);
    _mm512_storeu_si512(a, _mm512_and_si512(v, mask));
}

force_inline void
gf16_t_arr_xori_64_reg_avx512(gf16_t* a, __m512i mask) {
    __m512i v = _mm512_loadu_si512(a);
    _mm512_storeu_si512(a, _mm512_xor_si512(v, mask));
}

#elif defined(__AVX2__)

force_inline void
gf16_t_arr_andi_64_reg_avx2(gf16_t* a, __m256i high, __m256i low) {
    __m256i a0 = _mm256_loadu_si256((__m256i*) a);
    __m256i a1 = _mm256_loadu_si256((__m256i*) (a + 32));
    a0 = _mm256_and_si256(a0, low);
    a1 = _mm256_and_si256(a1, high);
    _mm256_storeu_si256((__m256i*) a, a0);
    _mm256_storeu_si256((__m256i*) (a + 32), a1);
}

force_inline void
gf16_t_arr_xori_64_reg_avx2(gf16_t* a, __m256i high, __m256i low) {
    __m256i a0 = _mm256_loadu_si256((__m256i*) a);
    __m256i a1 = _mm256_loadu_si256((__m256i*) (a + 32));
    a0 = _mm256_xor_si256(a0, low);
    a1 = _mm256_xor_si256(a1, high);
    _mm256_storeu_si256((__m256i*) a, a0);
    _mm256_storeu_si256((__m256i*) (a + 32), a1);
}

#endif

void
gf16_t_arr_andi_64(gf16_t* restrict a, gf16_t* restrict b) {
#if defined(__AVX512F__) && defined(__AVX512BW__)
    __m512i vb = _mm512_loadu_si512(b);
    gf16_t_arr_andi_64_reg_avx512(a, vb);
#elif defined(__AVX2__)
    __m256i b0 = _mm256_loadu_si256((__m256i*) b);
    __m256i b1 = _mm256_loadu_si256((__m256i*) (b + 32));
    gf16_t_arr_andi_64_reg_avx2(a, b1, b0);
#else
    for(uint32_t i = 0; i < 64; ++i) {
        a[i] &= b[i];
    }
#endif
}

void
gf16_t_arr_addi_64(gf16_t* restrict a, gf16_t* restrict b) {
#if defined(__AVX512F__) && defined(__AVX512BW__)
    __m512i vb = _mm512_loadu_si512(b);
    gf16_t_arr_xori_64_reg_avx512(a, vb);
#elif defined(__AVX2__)
    __m256i b0 = _mm256_loadu_si256((__m256i*) b);
    __m256i b1 = _mm256_loadu_si256((__m256i*) (b + 32));
    gf16_t_arr_xori_64_reg_avx2(a, b1, b0);
#else
    for(uint32_t i = 0; i < 64; ++i) {
        a[i] ^= b[i];
    }
#endif
}

void
gf16_t_arr_mul_scalar(gf16_t* restrict res, gf16_t* restrict arr,
                      uint32_t sz, gf16_t x) {
    // Each element in the array takes 8 bit, but has at most 4 bits.
    // After the multiplication, each element has at most 7 bits. Therefore,
    // we can perform reduction after the multiplication.
    memset(res, 0x0, sizeof(gf16_t) * sz); // res = 0
    for(uint32_t i = 0; i < 4; ++i) {
        if(x & 0x1U) {
            gf16_t_arr_addi(res, arr, sz);
        }

        // the array is modified in place, which we restore in the end
        gf16_t_arr_shl(arr, sz, 1);
        x >>= 1;
    }
    gf16_t_arr_shr(arr, sz, 4); // restore the input array
    gf16_t_arr_reduc(res, sz);
}

#if defined(__AVX512F__) && defined(__AVX512BW__)

static force_inline void
gf16_t_arr_muli_scalar64_reg_avx512(__m512i* restrict dst,
                                    const gf16_t* restrict arr, gf16_t x) {
    __m512i v = _mm512_loadu_si512(arr);
    uint8_t sbidxs[4];
    uint32_t sbnum = sbidx_in_4b(sbidxs, x);
    assert(sbnum >= 1);
    __m512i p = _mm512_slli_epi16(v, sbidxs[0]);
    for(uint32_t i = 1; i < sbnum; ++i)
        p = _mm512_xor_si512(p, _mm512_slli_epi16(v, sbidxs[i]));

    // 7-th bit
    __m512i v_gf_7b = _mm512_set1_epi32(0x4C4C4C4CU);
    // TODO: compare the cycle count
//    __m512i threshold = _mm512_set1_epi32(0x40404040U);
//    __mmask64 mask = _mm512_cmpge_epu8_mask(p, threshold);
//    p = _mm512_mask_blend_epi8(mask, p, _mm512_xor_si512(p, v_gf_7b));
//    __m512i v_gf_6b = _mm512_srli_epi64(v_gf_7b, 1);
//    mask = _mm512_cmpge_epu8_mask(p, _mm512_srli_epi32(threshold, 1));
//    p = _mm512_mask_blend_epi8(mask, p, _mm512_xor_si512(p, v_gf_6b));
//    __m512i v_gf_5b = _mm512_srli_epi64(v_gf_7b, 2);
//    mask = _mm512_cmpge_epu8_mask(p, _mm512_srli_epi32(threshold, 2));
//    p = _mm512_mask_blend_epi8(mask, p, _mm512_xor_si512(p, v_gf_5b));


    __mmask64 mask = _mm512_cmpgt_epu8_mask(p, _mm512_set1_epi32(0x3F3F3F3FU));
    p = _mm512_mask_blend_epi8(mask, p, _mm512_xor_si512(p, v_gf_7b));
    // 6-th bit
    __m512i v_gf_6b = _mm512_srli_epi64(v_gf_7b, 1);
    mask = _mm512_cmpgt_epu8_mask(p, _mm512_set1_epi32(0x1F1F1F1FU));
    p = _mm512_mask_blend_epi8(mask, p, _mm512_xor_si512(p, v_gf_6b));
    // 5-th bit
    __m512i v_gf_5b = _mm512_srli_epi64(v_gf_7b, 2);
    mask = _mm512_cmpgt_epu8_mask(p, _mm512_set1_epi32(0x0F0F0F0FU));
    p = _mm512_mask_blend_epi8(mask, p, _mm512_xor_si512(p, v_gf_5b));

    *dst = p;
}

static force_inline void
gf16_t_arr_muli_scalar64_avx512(gf16_t* arr, gf16_t x) {
    __m512i v;
    gf16_t_arr_muli_scalar64_reg_avx512(&v, arr, x);
    _mm512_storeu_si512(arr, v);
}

#elif defined(__AVX2__)

static force_inline void
gf16_t_arr_muli_scalar64_reg_avx2(__m256i* restrict high, __m256i* restrict low,
                                  const gf16_t* restrict arr, gf16_t x) {
    __m256i v0 = _mm256_loadu_si256((__m256i*) arr);
    __m256i v1 = _mm256_loadu_si256((__m256i*) (arr + 32));

    __m256i vp0 = _mm256_setzero_si256();
    __m256i vp1 = _mm256_setzero_si256();
    uint8_t sbidxs[4];
    uint32_t sbnum = sbidx_in_4b(sbidxs, x);
    assert(sbnum >= 1);
    /*
    uint32_t i = 0;
    do {
        uint8_t idx = sbidxs[i];
        vp0 = _mm256_xor_si256(vp0, _mm256_slli_epi16(v0, idx));
        vp1 = _mm256_xor_si256(vp1, _mm256_slli_epi16(v1, idx));
    } while(++i < sbnum);
    */
    // NOTE: for-loop is faster than do-while
    for(uint32_t i = 0; i < sbnum; ++i) {
        uint8_t idx = sbidxs[i];
        vp0 = _mm256_xor_si256(vp0, _mm256_slli_epi16(v0, idx));
        vp1 = _mm256_xor_si256(vp1, _mm256_slli_epi16(v1, idx));
    }

    /*
    __m256i vp0 = _mm256_slli_epi16(v0, sbidxs[0]);
    __m256i vp1 = _mm256_slli_epi16(v1, sbidxs[0]);

    for(uint32_t i = 1; i < sbnum; ++i) {
        uint8_t idx = sbidxs[i];
        vp0 = _mm256_xor_si256(vp0, _mm256_slli_epi16(v0, idx));
        vp1 = _mm256_xor_si256(vp1, _mm256_slli_epi16(v1, idx));
    }
    */
    /*
    vp0 = _mmV256_slli_epi16(v0, sbidxs[0]);
    vp1 = _mm256_slli_epi16(v1, sbidxs[0]);

    if(likely(sbnum >= 2)) {
        vp0 = _mm256_xor_si256(vp0, _mm256_slli_epi16(v0, sbidxs[1]));
        vp1 = _mm256_xor_si256(vp1, _mm256_slli_epi16(v1, sbidxs[1]));
    }
    if(sbnum >= 3) {
        vp0 = _mm256_xor_si256(vp0, _mm256_slli_epi16(v0, sbidxs[2]));
        vp1 = _mm256_xor_si256(vp1, _mm256_slli_epi16(v1, sbidxs[2]));
    }
    if(unlikely(sbnum >= 4)) {
        vp0 = _mm256_xor_si256(vp0, _mm256_slli_epi16(v0, 3));
        vp1 = _mm256_xor_si256(vp1, _mm256_slli_epi16(v1, 3));
    }
    */

    /*
    if(unlikely(sbnum == 4)) {
        vp0 = _mm256_xor_si256(vp0, _mm256_slli_epi16(v0, 3)); // must be the 4-th bit
        vp1 = _mm256_xor_si256(vp1, _mm256_slli_epi16(v1, 3));
    }
    if(unlikely(sbnum >= 3)) {
        vp0 = _mm256_xor_si256(vp0, _mm256_slli_epi16(v0, sbidxs[2]));
        vp1 = _mm256_xor_si256(vp1, _mm256_slli_epi16(v1, sbidxs[2]));
    }
    if(likely(sbnum >= 2)) {
        vp0 = _mm256_xor_si256(vp0, _mm256_slli_epi16(v0, sbidxs[1]));
        vp1 = _mm256_xor_si256(vp1, _mm256_slli_epi16(v1, sbidxs[1]));
    }

    vp0 = _mm256_xor_si256(vp0, _mm256_slli_epi16(v0, sbidxs[0]));
    vp1 = _mm256_xor_si256(vp1, _mm256_slli_epi16(v1, sbidxs[0]));
    */

    /*
    switch(sbnum) {
        case 4:
            vp0 = _mm256_xor_si256(vp0, _mm256_slli_epi16(v0, 3)); // must be the 4-th bit
            vp1 = _mm256_xor_si256(vp1, _mm256_slli_epi16(v1, 3));
            // fall through
        case 3:
            vp0 = _mm256_xor_si256(vp0, _mm256_slli_epi16(v0, sbidxs[2]));
            vp1 = _mm256_xor_si256(vp1(v1, sbidxs[2]));
            // fall through
        case 2:
            vp0 = _mm256_xor_si256(vp0, _mm256_slli_epi16(v0, sbidxs[1]));
            vp1 = _mm256_xor_si256(vp1, _mm256_slli_epi16(v1, sbidxs[1]));
            // fall through
        case 1:
            vp0 = _mm256_xor_si256(vp0, _mm256_slli_epi16(v0, sbidxs[0]));
            vp1 = _mm256_xor_si256(vp1, _mm256_slli_epi16(v1, sbidxs[0]));
            // fall through
    }
    */

    // NOTE: each bit is set with 50% prob, which causes bad speculation
    // and thus poorer pipe retiring rate. _Do_ not use this code segment.
    /*
    for(uint32_t i = 0; i < 4; ++i) {
        if(x & 0x1U) {
            vp0 = _mm256_xor_si256(vp0, v0);
            vp1 = _mm256_xor_si256(vp1, v1);
        }
        v0 = _mm256_slli_epi16(v0, 1);
        v1 = _mm256_slli_epi16(v1, 1);
        x >>= 1;
    }
    */

    // reduction on vectorized p
    // TODO: compare broadcast and loading from memory
    //__m256i v_gf16p = _mm256_loadu_si256((__m256i*)gf16_poly_vec);
    __m256i v_gf16p = _mm256_set1_epi32(0x4C4C4C4CU);
    // NOTE: _mm256_set1_epi32 is better than _mm256_set1_epi8 in terms of uop fuse
    // 7-th bit
    __m256i threshold = _mm256_set1_epi32(0x3F3F3F3FU);
    // NOTE: __mm256_cmpgt_epi8 works here because the MSB of each
    // byte is guaranteed to be zero
    __m256i mask0 = _mm256_cmpgt_epi8(vp0, threshold);
    __m256i mask1 = _mm256_cmpgt_epi8(vp1, threshold);
    vp0 = _mm256_xor_si256(vp0, _mm256_and_si256(v_gf16p, mask0));
    vp1 = _mm256_xor_si256(vp1, _mm256_and_si256(v_gf16p, mask1));
    v_gf16p = _mm256_srli_epi16(v_gf16p, 1);
    // 6-th bit
    threshold = _mm256_set1_epi32(0x1F1F1F1FU);
    mask0 = _mm256_cmpgt_epi8(vp0, threshold);
    mask1 = _mm256_cmpgt_epi8(vp1, threshold);
    vp0 = _mm256_xor_si256(vp0, _mm256_and_si256(v_gf16p, mask0));
    vp1 = _mm256_xor_si256(vp1, _mm256_and_si256(v_gf16p, mask1));
    v_gf16p = _mm256_srli_epi16(v_gf16p, 1);
    // 5-th bit
    threshold = _mm256_set1_epi32(0x0F0F0F0FU);
    mask0 = _mm256_cmpgt_epi8(vp0, threshold);
    mask1 = _mm256_cmpgt_epi8(vp1, threshold);
    *low = _mm256_xor_si256(vp0, _mm256_and_si256(v_gf16p, mask0));
    *high = _mm256_xor_si256(vp1, _mm256_and_si256(v_gf16p, mask1));
}


static force_inline void
gf16_t_arr_muli_scalar64_avx2(gf16_t* arr, gf16_t x) {
    __m256i high, low;
    gf16_t_arr_muli_scalar64_reg_avx2(&high, &low, arr, x);
    _mm256_storeu_si256((__m256i*) arr, low);
    _mm256_storeu_si256((__m256i*) (arr + 32), high);
}

#endif

void
gf16_t_arr_muli_scalar64(gf16_t* arr, gf16_t x) {
    if(unlikely(0 == x)) {
        memset(arr, 0x0, sizeof(gf16_t) * 64);
        return;
    }
#if defined(__AVX512F__) && defined(__AVX512BW__)
    gf16_t_arr_muli_scalar64_avx512(arr, x);
#elif defined(__AVX2__)
    gf16_t_arr_muli_scalar64_avx2(arr, x);
#else
    for(uint32_t i = 0; i < 64; ++i) {
        arr[i] = gf16_t_mul(arr[i], x);
    }
#endif
}

void
gf16_t_arr_muli_scalar(gf16_t* arr, uint32_t sz, gf16_t x) {
    // TODO: optimize
    for(uint32_t i = 0; i < sz; ++i) {
        arr[i] = gf16_t_mul(arr[i], x);
    }
}

#if defined(__AVX512F__) && defined(__AVX512BW__)

static force_inline void
gf16_t_arr_fmaddi_scalar64_avx512(gf16_t* restrict a, const gf16_t* restrict b,
                                  gf16_t c) {
    __m512i vp; gf16_t_arr_muli_scalar64_reg_avx512(&vp, b, c);
    __m512i va = _mm512_loadu_si512(a);
    va = _mm512_xor_si512(va, vp);
    _mm512_storeu_si512(a, va);
}

#elif defined(__AVX2__)

static force_inline void
gf16_t_arr_fmaddi_scalar64_avx2(gf16_t* restrict a, const gf16_t* restrict b,
                                gf16_t c) {
    __m256i va0 = _mm256_loadu_si256((__m256i*) a);
    __m256i va1 = _mm256_loadu_si256((__m256i*) (a + 32));

    __m256i vp0, vp1;
    gf16_t_arr_muli_scalar64_reg_avx2(&vp1, &vp0, b, c);

    va0 = _mm256_xor_si256(va0, vp0);
    va1 = _mm256_xor_si256(va1, vp1);
    _mm256_storeu_si256((__m256i*) a, va0);
    _mm256_storeu_si256((__m256i*) (a + 32), va1);
}

#endif

void
gf16_t_arr_fmaddi_scalar64(gf16_t* restrict a, const gf16_t* restrict b, gf16_t c) {
    // compute a += b * c; c is a scalar while a and b are processed element-wise
    if(unlikely(c == 0))
        return;
#if defined(__AVX512F__) && defined(__AVX512BW__)
    gf16_t_arr_fmaddi_scalar64_avx512(a, b, c);
#elif defined(__AVX2__)
    gf16_t_arr_fmaddi_scalar64_avx2(a, b, c);
#else
    for(uint32_t i = 0; i < 64; ++i) {
        gf16_t bc = gf16_t_mul(b[i], c);
        a[i] = gf16_t_add(a[i], bc);
    }
#endif
}

void
gf16_t_arr_fmaddi_scalar64_x2(gf16_t* restrict a, const gf16_t* restrict b0,
                              const gf16_t* restrict b1, gf16_t c0, gf16_t c1) {
#if defined(__AVX512F__) && defined(__AVX512BW__)
    gf16_t_arr_fmaddi_scalar64_avx512(a, b0, c0);
    gf16_t_arr_fmaddi_scalar64_avx512(a, b1, c1);
#elif defined(__AVX2__)
    __m256i va0 = _mm256_loadu_si256((__m256i*) a);
    __m256i va1 = _mm256_loadu_si256((__m256i*) (a + 32));
    __m256i vp0, vp1, vp2, vp3;
    gf16_t_arr_muli_scalar64_reg_avx2(&vp1, &vp0, b0, c0);
    gf16_t_arr_muli_scalar64_reg_avx2(&vp3, &vp2, b1, c1);
    vp0 = _mm256_xor_si256(vp0, vp2);
    vp1 = _mm256_xor_si256(vp1, vp3);
    va0 = _mm256_xor_si256(va0, vp0);
    va1 = _mm256_xor_si256(va1, vp1);
    _mm256_storeu_si256((__m256i*) a, va0);
    _mm256_storeu_si256((__m256i*) (a + 32), va1);
#else
    for(uint32_t i = 0; i < 64; ++i) {
        gf16_t bc = gf16_t_mul(b0[i], c0);
        a[i] = gf16_t_add(a[i], bc);
    }
    for(uint32_t i = 0; i < 64; ++i) {
        gf16_t bc = gf16_t_mul(b1[i], c1);
        a[i] = gf16_t_add(a[i], bc);
    }
#endif
}

void
gf16_t_arr_fmaddi_scalar(gf16_t* restrict a, const gf16_t* restrict b,
                         uint32_t sz, gf16_t c) {
    // compute a += b * c; c is a scalar while a and b are processed element-wise
    if(unlikely(c == 0))
        return;
    for(uint32_t i = 0; i < sz; ++i) {
        gf16_t bc = gf16_t_mul(b[i], c);
        a[i] = gf16_t_add(a[i], bc);
    }
}

#if defined(__AVX512F__) && defined(__AVX512BW__)
void
gf16_t_arr_fmaddi_scalar_mask64_avx512(gf16_t* restrict a, const gf16_t* restrict b,
                                       gf16_t c, uint64_t d) {
    __m512i v = _mm512_loadu_si512(a);
    __m512i p; gf16_t_arr_muli_scalar64_reg_avx512(&p, b, c);
    __m512i mask; mm512_create_mask_from_64b_avx512(&mask, d);
    v = _mm512_xor_si512(v, _mm512_and_si512(p, mask));
    _mm512_storeu_si512(a, v);
}

#elif defined(__AVX2__)

void
gf16_t_arr_fmaddi_scalar_mask64_avx2(gf16_t* restrict a, const gf16_t* restrict b,
                                     gf16_t c, uint64_t d) {
    __m256i va0 = _mm256_loadu_si256((__m256i*) a);
    __m256i va1 = _mm256_loadu_si256((__m256i*) (a + 32));

    __m256i vp0, vp1;
    gf16_t_arr_muli_scalar64_reg_avx2(&vp1, &vp0, b, c);

    /*
    //const __m256i shuffle_mask = _mm256_loadu_si256((__m256i*) avx2_shuffle_mask);
    const __m256i shuffle_mask = _mm256_set1_epi64x(0x8040201008040201ULL);
    // mask for lower half
    __m256i mask0 = _mm256_and_si256(mm256_8b_to_32b(d), shuffle_mask);
    mask0 = _mm256_cmpeq_epi8(mask0, shuffle_mask);
    // NOTE: _mm256_cmpgt_epi8(mask0, full_zero_vector) doesn't work because
    // each byte is interpreted as a signed 8-bit integer. When the MSB of
    // a byte is set, the integer is treated as negative and thus < 0.

    // mask for upper half
    __m256i mask1 = _mm256_and_si256(mm256_8b_to_32b(d >> 32), shuffle_mask);
    mask1 = _mm256_cmpeq_epi8(mask1, shuffle_mask);
    */
    __m256i mask0, mask1;
    mm256_create_mask_from_64b_avx2(&mask0, &mask1, d);

    // mask part of vp with d before xor'ing
    va0 = _mm256_xor_si256(va0, _mm256_and_si256(vp0, mask0));
    va1 = _mm256_xor_si256(va1, _mm256_and_si256(vp1, mask1));

    _mm256_storeu_si256((__m256i*) a, va0);
    _mm256_storeu_si256((__m256i*) (a + 32), va1);
}

#endif

void
gf16_t_arr_fmaddi_scalar_mask64(gf16_t* restrict a, const gf16_t* restrict b,
                                gf16_t c, uint64_t d) {
    // both a and b are gf16_t arrays with 64 elements, d is a 64-bit integer that
    // specifies which elements to keep, and which to zero.
    // compute a += b * c; c is a scalar while a and b are processed element-wise
    // if the i-th bit of d is 1, then the i-th element of b*c is added to the
    // i-th element of a.
    // if 0, then the i-th element of a is untouched.
    if(c == 0)
        return;
#if defined(__AVX512F__) && defined(__AVX512BW__)
    gf16_t_arr_fmaddi_scalar_mask64_avx512(a, b, c, d);
#elif defined(__AVX2__)
    gf16_t_arr_fmaddi_scalar_mask64_avx2(a, b, c, d);
#else
    for(uint32_t i = 0; i < 64; ++i) {
        if(d & 0x1) {
            gf16_t bc = gf16_t_mul(b[i], c);
            a[i] = gf16_t_add(a[i], bc);
        }
        d >>= 1;
    }
#endif
}

void
gf16_t_arr_fmaddi_scalar_mask64_ref(gf16_t* restrict a, const gf16_t* restrict b,
                                    gf16_t c, uint64_t d) {
    // both a and b are gf16_t arrays with 64 elements, d is a 64-bit integer that
    // specifies which elements to keep, and which to zero.
    // compute a += b * c; c is a scalar while a and b are processed element-wise
    // if the i-th bit of d is 1, then the i-th element of b*c is added to the
    // i-th element of a.
    // if 0, then the i-th element of a is untouched.
    if(c == 0)
        return;

    for(uint32_t i = 0; i < 64; ++i) {
        if(d & 0x1) {
            gf16_t bc = gf16_t_mul(b[i], c);
            a[i] = gf16_t_add(a[i], bc);
        }
        d >>= 1;
    }
}

void
gf16_t_arr_fmsubi_scalar64(gf16_t* restrict a, const gf16_t* restrict b, gf16_t c) {
    gf16_t_arr_fmaddi_scalar64(a, b, c); // in GF16, add is the same as sub
}

void
gf16_t_arr_fmsubi_scalar(gf16_t* restrict a, const gf16_t* restrict b,
                         uint32_t sz, gf16_t c) {
    // compute a -= b * c; c is a scalar while a and b are processed element-wise
    if(c == 0)
        return;
    // TODO: optimize
    for(uint32_t i = 0; i < sz; ++i) {
        gf16_t bc = gf16_t_mul(b[i], c);
        a[i] = gf16_t_sub(a[i], bc);
    }
}

void
gf16_t_arr_fmsubi_scalar_mask64_ref(gf16_t* restrict a, const gf16_t* restrict b,
                                    gf16_t c, uint64_t d) {
    // both a and b are gf16_t arrays with 64 elements, d is a 64-bit integer that
    // specifies which elements to keep, and which to zero.
    // compute a -= b * c; c is a scalar while a and b are processed element-wise
    // if the i-th bit of d is zero, then the i-th element of b*c is added to the
    // i-th element of a
    // if 1, then the i-th element of a is untouched.
    if(c == 0)
        return;
    for(uint32_t i = 0; i < 64; ++i) {
        if(d & 0x1) {
            gf16_t bc = gf16_t_mul(b[i], c);
            a[i] = gf16_t_sub(a[i], bc);
        }
        d >>= 1;
    }
}

void
gf16_t_arr_fmsubi_scalar_mask64(gf16_t* restrict a, const gf16_t* restrict b,
                                gf16_t c, uint64_t d) {
    // in GF16, polynomial addition is the same as sub
    gf16_t_arr_fmaddi_scalar_mask64(a, b, c, d);
}

uint64_t
gf16_t_arr_nzc(const gf16_t* a, uint64_t sz) {
    uint64_t c = 0;
    for(uint64_t i = 0; i < sz; ++i) {
        if(a[i])
            ++c;
    }
    return c;
}

uint32_t
gf16_t_arr_zc(const gf16_t* a, uint32_t sz) {
    uint32_t c = 0;
    for(uint32_t i = 0; i < sz; ++i) {
        if(!a[i])
            ++c;
    }
    return c;
}

void
gf16_t_arr_reduc_64(gf16_t* arr) {
#if defined(__AVX512F__) && defined(__AVX512BW__)
    __m512i v = _mm512_loadu_si512(arr);
    const __m512i mask = _mm512_set1_epi32(0x0F0F0F0FU);
    _mm512_storeu_si512(arr, _mm512_and_si512(v, mask));
#elif defined(__AVX2__)
    __m256i v0 = _mm256_loadu_si256((__m256i*) arr);
    __m256i v1 = _mm256_loadu_si256((__m256i*) (arr + 32));
    const __m256i mask = _mm256_set1_epi32(0x0F0F0F0FU);
    v0 = _mm256_and_si256(v0, mask);
    v1 = _mm256_and_si256(v1, mask);
    _mm256_storeu_si256((__m256i*) arr, v0);
    _mm256_storeu_si256((__m256i*) (arr + 32), v1);
#else
    for(uint32_t j = 0; j < 64; ++j) {
        arr[j] = arr[j] % (GF16_MAX + 1);
    }
#endif
}
