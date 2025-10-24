/* bitmap_table.h: map byte(s) into an array of the indices of set bits */

#ifndef __BLK_LANCZOS_BITMAP_TABLE_H__
#define __BLK_LANCZOS_BITMAP_TABLE_H__

#include <stdint.h>
#include <stdalign.h>
#include "util.h"

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

#ifndef BLK_LANCZOS_BITMAP_SIZE

#define BLK_LANCZOS_BITMAP_SIZE 8

#endif

/* ========================================================================
 * global lookup tables
 * ======================================================================== */

extern const uint8_t g_b8_sbnum[256];
extern alignas(4) const uint8_t g_b4_sbpos[16][4];
extern alignas(8) const uint8_t g_b8_sbpos[256][8];
extern alignas(16) const uint16_t g_b8_sbpos_16[256][8];
extern alignas(16) const uint8_t g_b16_sbpos_8[65536][16];
extern alignas(32) const uint16_t g_b16_sbpos_16[65536][16];

/* ========================================================================
 * functions prototypes
 * ======================================================================== */

/* usage: Given a byte, return the number of set bits in it
 * params:
 *      1) byte: a uint8_t
 * return: the number of set bits */
#ifdef BLK_LANCZOS_POPCNT_NO_LOOKUP

// POPCNT has higher latency and less throughput than MOV on most architectures
#define popcnt_8b(byte) \
    (__builtin_popcount((uint8_t)byte))

#else

#define popcnt_8b(byte) \
    (g_b8_sbnum[(uint8_t)byte])

#endif

// TODO: generate lookup table for 16-bit
/* usage: Given a uint16_t, return the number of set bits in it
 * params:
 *      1) hword: a uint16_t
 * return: the number of set bits */
#define popcnt_16b(hword) \
    (__builtin_popcount((uint16_t)hword))

/* usage: Given a uint64_t, return the number of set bits in it
 * params:
 *      1) b64: a uint64_t
 * return: the number of set bits */
#define popcnt_64b(b64) \
    (__builtin_popcountll(b64))

/* usage: Given 4-bit, store the indices of set bits as an array of uint8_t.
 * params:
 *      1) out: an array of uint8_t for storing the indices; must hold at
 *              least 4 elements
 *      2) b: the 4 bits
 * return: the number of set bits */
static inline uint64_t
sbidx_in_4b(uint8_t* out, uint8_t b) {
    assert(!(b & ~0xF));
    uint32_t idx = *((uint32_t*) g_b4_sbpos[b]);
    uint8_t sbn = popcnt_8b(b);
    *((uint32_t*)out) = idx;
    return sbn;
}

/* usage: Given 16-bit, store the indices of set bits as an array of uint8_t.
 *      The indices must be smaller than 256.
 * params:
 *      1) out: an array of uint8_t for storing the indices; must hold at
 *              least 16 elements
 *      2) offset: a uint64_t that stores the offset for all indices. The
 *              lowest 8-bit is the offset and should be propagated to other 56
 *              bits. E.g. To set offset to 8, 0x0808080808080808ULL should be
 *              passed
 *      3) l8b: the lower 8 bits of the 16-bit
 *      4) h8b: the higher 8 bits of the 16-bit
 * return: the number of set bits */
static inline uint64_t
sbidx_in_16b_sz8(uint8_t* out, uint64_t offset, uint8_t l8b, uint8_t h8b) {
    const uint64_t inc8  = 0x0808080808080808ULL;
    uint64_t idx_low = *((uint64_t*) g_b8_sbpos[l8b]);
    uint64_t idx_high = *((uint64_t*) g_b8_sbpos[h8b]);
    uint8_t sbn_low = popcnt_8b(l8b);
    uint8_t sbn_high = popcnt_8b(h8b);
    *((uint64_t*)out) = idx_low + offset;
    *((uint64_t*) (out + sbn_low)) = idx_high + offset + inc8;
    return sbn_low + sbn_high;
}

/* usage: Given 32-bit, store the indices of set bits as an array of uint8_t.
 *      The indices must be smaller than 256.
 * params:
 *      1) out: an array of uint8_t for storing the indices; must hold at
 *              least 32 elements
 *      2) offset: a uint64_t that stores the offset for all indices. The
 *              lowest 8-bit is the offset and should be propagated to other 56
 *              bits. E.g. To set offset to 8, 0x0808080808080808ULL should be
 *              passed
 *      3) l16b: the lower 16 bits of the 32-bit
 *      4) h16b: the higher 16 bits of the 32-bit
 * return: the number of set bits */
static inline uint64_t
sbidx_in_32b_sz8(uint8_t* out, uint64_t offset, uint16_t l16b, uint16_t h16b) {
    const uint64_t inc16 = 0x1010101010101010ULL;
#if defined(__AVX2__)
    __m256i vidx = _mm256_inserti128_si256(
        _mm256_castsi128_si256(_mm_load_si128((__m128i*) g_b16_sbpos_8[l16b])),
        _mm_load_si128((__m128i*) g_b16_sbpos_8[h16b]), 1);

    uint8_t sbn_low = popcnt_16b(l16b);
    uint8_t sbn_high = popcnt_16b(h16b);
    __m256i voff = _mm256_set_epi64x(offset + inc16, offset + inc16,
                                     offset, offset);
    vidx = _mm256_add_epi16(voff, vidx);
    _mm_storeu_si128((__m128i*) (out), _mm256_castsi256_si128(vidx));
    _mm_storeu_si128((__m128i*) (out + sbn_low), _mm256_extracti128_si256(vidx, 1));
#else
    uint64_t lidx_0 = ((uint64_t*) g_b16_sbpos_8[l16b])[0];
    uint64_t lidx_1 = ((uint64_t*) g_b16_sbpos_8[l16b])[1];
    uint64_t hidx_0 = ((uint64_t*) g_b16_sbpos_8[h16b])[0];
    uint64_t hidx_1 = ((uint64_t*) g_b16_sbpos_8[h16b])[1];
    uint8_t sbn_low = popcnt_16b(l16b);
    uint8_t sbn_high = popcnt_16b(h16b);

    lidx_0 += offset;
    lidx_1 += offset;
    offset += inc16;

    hidx_0 += offset;
    hidx_1 += offset;
    ((uint64_t*)out)[0] = lidx_0;
    ((uint64_t*)out)[1] = lidx_1;
    ((uint64_t*)(out + sbn_low))[0] = hidx_0;
    ((uint64_t*)(out + sbn_low))[1] = hidx_1;
#endif
    return sbn_low + sbn_high;
}

/* usage: Given a uint64_t, store the indices of set bits an array of uint8_t.
 *      The indices must be smaller than 256.
 * params:
 *      1) out: an array of uint8_t for storing the indices; must hold at
 *              least 64 elements
 *      2) offset: a uint64_t that stores the offset for all indices. The
 *              lowest 8-bit is the offset and should be propagated to other 56
 *              bits. E.g. To set offset to 8, 0x0808080808080808ULL should be
 *              passed
 *      3) b64: a uint64_t
 * return: the number of set bits */
static inline uint64_t
sbidx_in_64b_sz8(uint8_t* out, uint64_t offset, uint64_t b64) {
    if(unlikely(!b64)) {
        return 0;
    }

    uint64_t sbnum = 0;

#if BLK_LANCZOS_BITMAP_SIZE == 16
    const uint64_t inc32 = 0x2020202020202020ULL;
    sbnum += sbidx_in_32b_sz8(out, offset,
                              (uint16_t) b64, (uint16_t) (b64 >> 16));
    b64 >>= 32;
    sbnum += sbidx_in_32b_sz8(out + sbnum, offset + inc32,
                              (uint16_t) b64, (uint16_t) (b64 >> 16));
#else
    const uint64_t inc16 = 0x1010101010101010ULL;
    sbnum += sbidx_in_16b_sz8(out, offset,
                              (uint8_t) b64, (uint8_t) (b64 >> 8));
    b64 >>= 16;
    offset += inc16;
    sbnum += sbidx_in_16b_sz8(out + sbnum, offset,
                              (uint8_t) b64, (uint8_t) (b64 >> 8));
    b64 >>= 16;
    offset += inc16;
    sbnum += sbidx_in_16b_sz8(out + sbnum, offset,
                              (uint8_t) b64, (uint8_t) (b64 >> 8));
    b64 >>= 16;
    offset += inc16;
    sbnum += sbidx_in_16b_sz8(out + sbnum, offset,
                              (uint8_t) b64, (uint8_t) (b64 >> 8));
#endif

    return sbnum;
}

/* usage: Given a non-zero uint64_t, store the indices of set bits an array
 *      of uint8_t. The indices must be smaller than 256.
 * params:
 *      1) out: an array of uint8_t for storing the indices; must hold at
 *              least 64 elements
 *      2) b64: a uint64_t; must not be zero
 * return: the number of set bits */
static inline uint64_t
sbidx_in_64b_sz8_nz(uint8_t* out, uint64_t b64) {
    assert(b64);
    uint64_t sbnum = 0;
    uint64_t offset = 0ULL;

#if BLK_LANCZOS_BITMAP_SIZE == 16
    const uint64_t inc32 = 0x2020202020202020ULL;
    sbnum += sbidx_in_32b_sz8(out, offset,
                              (uint16_t) b64, (uint16_t) (b64 >> 16));
    b64 >>= 32;
    sbnum += sbidx_in_32b_sz8(out + sbnum, offset + inc32,
                              (uint16_t) b64, (uint16_t) (b64 >> 16));
#else
    const uint64_t inc16 = 0x1010101010101010ULL;
    sbnum += sbidx_in_16b_sz8(out, offset,
                              (uint8_t) b64, (uint8_t) (b64 >> 8));
    b64 >>= 16;
    offset += inc16;
    sbnum += sbidx_in_16b_sz8(out + sbnum, offset,
                              (uint8_t) b64, (uint8_t) (b64 >> 8));
    b64 >>= 16;
    offset += inc16;
    sbnum += sbidx_in_16b_sz8(out + sbnum, offset,
                              (uint8_t) b64, (uint8_t) (b64 >> 8));
    b64 >>= 16;
    offset += inc16;
    sbnum += sbidx_in_16b_sz8(out + sbnum, offset,
                              (uint8_t) b64, (uint8_t) (b64 >> 8));
#endif

    return sbnum;
}

/* usage: Given 16-bit, store the indices of set bits as an array of uint16_t.
 *      The indices must be smaller than 65536.
 * params:
 *      1) out: an array of uint16_t for storing the indices; must hold at
 *              least 16 elements
 *      2) offset: a uint64_t that stores the offset for all indices. The
 *              lowest 16-bit is the offset and should be propagated to other
 *              48 bits. E.g. To set offset to 8, 0x0008000800080008ULL should
 *              be passed
 *      3) l8b: the lower 8 bits of the 16-bit
 *      4) h8b: the higher 8 bits of the 16-bit
 * return: the number of set bits */
static inline uint64_t
sbidx_in_16b_sz16(uint16_t* out, uint64_t offset, uint8_t l8b, uint8_t h8b) {
    const uint64_t inc8  = 0x0008000800080008ULL;
#if defined(__AVX2__)
    __m256i vidx = _mm256_inserti128_si256(
        _mm256_castsi128_si256(_mm_load_si128((__m128i*) g_b8_sbpos_16[l8b])),
        _mm_load_si128((__m128i*) g_b8_sbpos_16[h8b]), 1);
    uint8_t sbn_low = popcnt_8b(l8b);
    uint8_t sbn_high = popcnt_8b(h8b);
    __m256i voff = _mm256_set_epi64x(offset + inc8, offset + inc8,
                                     offset, offset);
    vidx = _mm256_add_epi16(voff, vidx);
    _mm_storeu_si128((__m128i*) (out), _mm256_castsi256_si128(vidx));
    _mm_storeu_si128((__m128i*) (out + sbn_low), _mm256_extracti128_si256(vidx, 1));
#else
    const uint64_t* ptr = (uint64_t*) g_b8_sbpos_16[l8b];
    uint64_t lidx_0 = ptr[0];
    uint64_t lidx_1 = ptr[1];
    ptr = (uint64_t*) g_b8_sbpos_16[h8b];
    uint64_t hidx_0 = ptr[0];
    uint64_t hidx_1 = ptr[1];
    uint8_t sbn_low = popcnt_8b(l8b);
    uint8_t sbn_high = popcnt_8b(h8b);
    ((uint64_t*)out)[0] = lidx_0 + offset;
    ((uint64_t*)out)[1] = lidx_1 + offset;
    ((uint64_t*)(out + sbn_low))[0] = hidx_0 + offset + inc8;
    ((uint64_t*)(out + sbn_low))[1] = hidx_1 + offset + inc8;
#endif
    return sbn_low + sbn_high;
}

/* usage: Given 32-bit, store the indices of set bits as an array of uint16_t.
 *      The indices must be smaller than 65536.
 * params:
 *      1) out: an array of uint16_t for storing the indices; must hold at
 *              least 32 elements
 *      2) offset: a uint64_t that stores the offset for all indices. The
 *              lowest 16-bit is the offset and should be propagated to other
 *              48 bits. E.g. To set offset to 8, 0x0008000800080008ULL should
 *              be passed
 *      3) l16b: the lower 16 bits of the 32-bit
 *      4) h16b: the higher 16 bits of the 32-bit
 * return: the number of set bits */
static inline uint64_t
sbidx_in_32b_sz16(uint16_t* out, uint64_t offset,
                  uint16_t l16b, uint16_t h16b) {
#if defined(__AVX2__)
    __m256i lidx = _mm256_load_si256((__m256i const*) g_b16_sbpos_16[l16b]);
    __m256i hidx = _mm256_load_si256((__m256i const*) g_b16_sbpos_16[h16b]);
    uint8_t sbn_low = popcnt_16b(l16b);
    uint8_t sbn_high = popcnt_16b(h16b);

    __m256i voff = _mm256_set1_epi64x(offset);
    lidx = _mm256_add_epi16(voff, lidx);
    _mm256_storeu_si256((__m256i*)out, lidx);

    const __m256i inc16 = _mm256_set1_epi16(16);
    voff = _mm256_add_epi16(voff, inc16);
    hidx = _mm256_add_epi16(voff, hidx);
    _mm256_storeu_si256((__m256i*)(out + sbn_low), hidx);
#else
    const uint64_t inc16 = 0x0010001000100010ULL;

    uint64_t* ptr = (uint64_t*) g_b16_sbpos_16[l16b];
    uint64_t lidx_0 = ptr[0];
    uint64_t lidx_1 = ptr[1];
    uint64_t lidx_2 = ptr[2];
    uint64_t lidx_3 = ptr[3];
    uint8_t sbn_low = popcnt_16b(l16b);

    ptr = (uint64_t*) g_b16_sbpos_16[h16b];
    uint64_t hidx_0 = ptr[0];
    uint64_t hidx_1 = ptr[1];
    uint64_t hidx_2 = ptr[2];
    uint64_t hidx_3 = ptr[3];
    uint8_t sbn_high = popcnt_16b(h16b);

    lidx_0 += offset;
    lidx_1 += offset;
    lidx_2 += offset;
    lidx_3 += offset;
    offset += inc16;

    hidx_0 += offset;
    hidx_1 += offset;
    hidx_2 += offset;
    hidx_3 += offset;

    ((uint64_t*)out)[0] = lidx_0;
    ((uint64_t*)out)[1] = lidx_1;
    ((uint64_t*)out)[2] = lidx_2;
    ((uint64_t*)out)[3] = lidx_3;

    ((uint64_t*)(out + sbn_low))[0] = hidx_0;
    ((uint64_t*)(out + sbn_low))[1] = hidx_1;
    ((uint64_t*)(out + sbn_low))[2] = hidx_2;
    ((uint64_t*)(out + sbn_low))[3] = hidx_3;
#endif
    return sbn_low + sbn_high;
}

#if defined(__AVX512BW__)

static inline __m512i
sbidx_in_64b_calc(uint64_t b64) {
    const uint64_t msk_1 = 0xffffffff00000000ULL;
    const uint64_t msk_2 = 0xffff0000ffff0000ULL;
    const uint64_t msk_3 = 0xff00ff00ff00ff00ULL;
    const uint64_t msk_4 = 0xf0f0f0f0f0f0f0f0ULL;
    const uint64_t msk_5 = 0xccccccccccccccccULL;
    const uint64_t msk_6 = 0xaaaaaaaaaaaaaaaaULL;
    const __m512i v1_bit = _mm512_set1_epi8(1);
    const __m512i v2_bit = _mm512_set1_epi8(2);
    const __m512i v4_bit = _mm512_set1_epi8(4);
    const __m512i v8_bit = _mm512_set1_epi8(8);
    const __m512i v16_bit = _mm512_set1_epi8(16);
    const __m512i v32_bit = _mm512_set1_epi8(32);

    const uint64_t v1 = _pext_u64(msk_1, b64);
    const uint64_t v2 = _pext_u64(msk_2, b64);
    const uint64_t v3 = _pext_u64(msk_3, b64);
    const uint64_t v4 = _pext_u64(msk_4, b64);
    const uint64_t v5 = _pext_u64(msk_5, b64);
    const uint64_t v6 = _pext_u64(msk_6, b64);

    __m512i vec;
    vec = _mm512_maskz_add_epi8(v1, v32_bit, _mm512_set1_epi8(0));
    vec = _mm512_mask_add_epi8(vec, v2, v16_bit, vec);
    vec = _mm512_mask_add_epi8(vec, v3, v8_bit, vec);
    vec = _mm512_mask_add_epi8(vec, v4, v4_bit, vec);
    vec = _mm512_mask_add_epi8(vec, v5, v2_bit, vec);
    vec = _mm512_mask_add_epi8(vec, v6, v1_bit, vec);

    return vec;
}

/* usage: Given a uint64_t, store the indices of set bits an array uint16_t.
 *      The indices must be smaller than 65535.
 * params:
 *      1) out: an array of uint16_t for storing the indices; must hold at
 *              least 64 elements
 *      2) offset: a uint64_t that stores the offset for all indices. The
 *              lowest 16-bit is the offset and should be propagated to other
 *              48 bits. E.g. To set offset to 8, 0x0008000800080008ULL should
 *              be passed
 *      3) b64: a uint64_t
 * return: the number of set bits */
static inline uint64_t
sbidx_in_64b_sz16_avx512(uint16_t* out, uint64_t offset, uint64_t b64) {
    __m512i vec = sbidx_in_64b_calc(b64);

    // expand the indices from 8-bit to 16-bit
    __m512i val1 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(vec, 0));
    __m512i val2 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(vec, 1));

    // add offset to the indices
    const __m512i voff = _mm512_set1_epi64(offset);
    val1 = _mm512_add_epi16(val1, voff);
    val2 = _mm512_add_epi16(val2, voff);

    // store the indices
    _mm512_storeu_si512((__m512i*) out, val1);
    _mm512_storeu_si512(((__m512i*) out)+1, val2);

    const uint64_t sbnum = popcnt_64b(b64);
    return sbnum;
}

/* usage: Given a uint64_t, store the indices of set bits an array uint32_t.
 *      The indices must be smaller than 2^32.
 * params:
 *      1) out: an array of uint32_t for storing the indices; must hold at
 *              least 64 elements
 *      2) offset: a uint64_t that stores the offset for all indices. The
 *              lowest 32-bit is the offset and should be propagated to other
 *              32 bits. E.g. To set offset to 8, 0x0000000800000008ULL should
 *              be passed
 *      3) b64: a uint64_t
 * return: the number of set bits */
static inline uint64_t
sbidx_in_64b_sz32_avx512(uint32_t* out, uint64_t offset, uint64_t b64) {
    __m512i vec = sbidx_in_64b_calc(b64);

    // expand the indices from 8-bit to 32-bit
    __m512i val1 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(vec, 0));
    __m512i val2 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(vec, 1));
    __m512i val3 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(vec, 2));
    __m512i val4 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(vec, 3));

    // add offset to the indices
    const __m512i voff = _mm512_set1_epi64(offset);
    val1 = _mm512_add_epi32(val1, voff);
    val2 = _mm512_add_epi32(val2, voff);
    val3 = _mm512_add_epi32(val3, voff);
    val4 = _mm512_add_epi32(val4, voff);

    // store the indices
    _mm512_storeu_si512((__m512i*) out, val1);
    _mm512_storeu_si512(((__m512i*) out)+1, val2);
    _mm512_storeu_si512(((__m512i*) out)+2, val3);
    _mm512_storeu_si512(((__m512i*) out)+3, val4);

    const uint64_t sbnum = popcnt_64b(b64);
    return sbnum;
}

#endif

/* usage: Given a uint64_t, store the indices of set bits an array of uint16_t.
 *      The indices must be smaller than 65536.
 * params:
 *      1) out: an array of uint16_t for storing the indices; must hold at
 *              least 64 elements
 *      2) offset: a uint64_t that stores the offset for all indices. The
 *              lowest 16-bit is the offset and should be propagated to other
 *              48 bits. E.g. To set offset to 8, 0x0008000800080008ULL should
 *              be passed
 *      3) b64: a uint64_t
 * return: the number of set bits */
static inline uint64_t
sbidx_in_64b_sz16(uint16_t* out, uint64_t offset, uint64_t b64) {
    if(unlikely(!b64)) {
        return 0;
    }

    uint64_t sbnum = 0;
#if defined(__AVX512BW__)
    sbnum = sbidx_in_64b_sz16_avx512(out, offset, b64);

#elif BLK_LANCZOS_BITMAP_SIZE == 16
    const uint64_t inc32 = 0x0020002000200020ULL;
    sbnum += sbidx_in_32b_sz16(out, offset,
                               (uint16_t) b64, (uint16_t) (b64 >> 16));
    b64 >>= 32;
    sbnum += sbidx_in_32b_sz16(out + sbnum, offset + inc32,
                               (uint16_t) b64, (uint16_t) (b64 >> 16));
#else
    const uint64_t inc16 = 0x0010001000100010ULL;
    sbnum += sbidx_in_16b_sz16(out, offset,
                               (uint8_t) b64, (uint8_t) (b64 >> 8));
    b64 >>= 16;
    offset += inc16;
    sbnum += sbidx_in_16b_sz16(out + sbnum, offset,
                               (uint8_t) b64, (uint8_t) (b64 >> 8));
    b64 >>= 16;
    offset += inc16;
    sbnum += sbidx_in_16b_sz16(out + sbnum, offset,
                               (uint8_t) b64, (uint8_t) (b64 >> 8));
    b64 >>= 16;
    offset += inc16;
    sbnum += sbidx_in_16b_sz16(out + sbnum, offset,
                               (uint8_t) b64, (uint8_t) (b64 >> 8));
#endif

    return sbnum;
}

static inline void
store_sz16_as_sz32(uint32_t* out, uint64_t idx, uint64_t offset) {
    // expand indices from 16-bits to 32-bits
    uint64_t tmp0 = ((uint16_t) (idx)) | ((idx & 0xFFFF0000ULL) << 16);
    ((uint64_t*)out)[0] = tmp0 + offset;
    uint64_t tmp1 = ((uint16_t) (idx >> 32)) | ((idx & 0xFFFF000000000000ULL) >> 16);
    ((uint64_t*)out)[1] = tmp1 + offset;
}

/* usage: Given 16-bit, store the indices of set bits as an array of uint32_t.
 *      The indices must be smaller than 2^32.
 * params:
 *      1) out: an array of uint32_t for storing the indices; must hold at
 *              least 16 elements
 *      2) offset: a uint64_t that stores the offset for all indices. The
 *              lowest 32-bit is the offset and should be propagated to other
 *              32 bits. E.g. To set offset to 8, 0x0000000800000008ULL should
 *              be passed
 *      3) l8b: the lower 8 bits of the 16-bit
 *      4) h8b: the higher 8 bits of the 16-bit
 * return: the number of set bits */
static inline uint64_t
sbidx_in_16b_sz32(uint32_t* out, uint64_t offset, uint8_t l8b, uint8_t h8b) {
#if defined(__AVX2__)
    __m256i vidx = _mm256_inserti128_si256(
        _mm256_castsi128_si256(_mm_load_si128((__m128i*) g_b8_sbpos_16[l8b])),
        _mm_load_si128((__m128i*) g_b8_sbpos_16[h8b]), 1);
    uint8_t sbn_low = popcnt_8b(l8b);
    uint8_t sbn_high = popcnt_8b(h8b);
    __m256i lidx = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(vidx, 0));
    __m256i hidx = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(vidx, 1));
    __m256i voff = _mm256_set1_epi64x(offset);
    lidx = _mm256_add_epi32(voff, lidx);
    _mm256_storeu_si256((__m256i*) out, lidx);

    const __m256i inc8 = _mm256_set1_epi32(8);
    voff = _mm256_add_epi32(voff, inc8);
    hidx = _mm256_add_epi32(voff, hidx);
    _mm256_storeu_si256((__m256i*) (out + sbn_low), hidx);
#else
    const uint64_t inc8  = 0x0000000800000008ULL;
    const uint64_t* ptr = (uint64_t*) g_b8_sbpos_16[l8b];
    uint64_t lidx_0 = ptr[0];
    uint64_t lidx_1 = ptr[1];
    ptr = (uint64_t*) g_b8_sbpos_16[h8b];
    uint64_t hidx_0 = ptr[0];
    uint64_t hidx_1 = ptr[1];
    uint8_t sbn_low = popcnt_8b(l8b);
    uint8_t sbn_high = popcnt_8b(h8b);
    store_sz16_as_sz32(out, lidx_0, offset);
    store_sz16_as_sz32(out + 4, lidx_1, offset);
    offset += inc8;
    out += sbn_low;
    store_sz16_as_sz32(out, hidx_0, offset);
    store_sz16_as_sz32(out + 4, hidx_1, offset);
#endif
    return sbn_low + sbn_high;
}

/* usage: Given 32-bit, store the indices of set bits as an array of uint32_t.
 *      The indices must be smaller than 2^32.
 * params:
 *      1) out: an array of uint32_t for storing the indices; must hold at
 *              least 32 elements
 *      2) offset: a uint64_t that stores the offset for all indices. The
 *              lowest 32-bit is the offset and should be propagated to other
 *              32 bits. E.g. To set offset to 8, 0x0000000800000008ULL should
 *              be passed
 *      3) l16b: the lower 16 bits of the 32-bit
 *      4) h16b: the higher 16 bits of the 32-bit
 * return: the number of set bits */
static inline uint64_t
sbidx_in_32b_sz32(uint32_t* out, uint64_t offset,
                  uint16_t l16b, uint16_t h16b) {
#if defined(__AVX2__)
    __m256i lidx = _mm256_load_si256((__m256i const*) g_b16_sbpos_16[l16b]);
    __m256i hidx = _mm256_load_si256((__m256i const*) g_b16_sbpos_16[h16b]);
    uint8_t sbn_low = popcnt_16b(l16b);
    uint8_t sbn_high = popcnt_16b(h16b);
    __m256i voff = _mm256_set1_epi64x(offset);

    __m256i lidx_0 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(lidx, 0));
    __m256i lidx_1 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(lidx, 1));
    lidx_0 = _mm256_add_epi16(voff, lidx_0);
    lidx_1 = _mm256_add_epi16(voff, lidx_1);
    _mm256_storeu_si256((__m256i*)out, lidx_0);
    _mm256_storeu_si256((__m256i*)(out + 8), lidx_1);

    const __m256i inc16 = _mm256_set1_epi32(16);
    voff = _mm256_add_epi32(voff, inc16);

    __m256i hidx_0 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(hidx, 0));
    __m256i hidx_1 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(hidx, 1));
    hidx_0 = _mm256_add_epi16(voff, hidx_0);
    hidx_1 = _mm256_add_epi16(voff, hidx_1);
    _mm256_storeu_si256((__m256i*)(out + sbn_low), hidx_0);
    _mm256_storeu_si256((__m256i*)(out + sbn_low + 8), hidx_1);
#else
    const uint64_t inc16 = 0x0000001000000010ULL;

    uint64_t* ptr = (uint64_t*) g_b16_sbpos_16[l16b];
    uint64_t lidx_0 = ptr[0];
    uint64_t lidx_1 = ptr[1];
    uint64_t lidx_2 = ptr[2];
    uint64_t lidx_3 = ptr[3];
    uint8_t sbn_low = popcnt_16b(l16b);

    ptr = (uint64_t*) g_b16_sbpos_16[h16b];
    uint64_t hidx_0 = ptr[0];
    uint64_t hidx_1 = ptr[1];
    uint64_t hidx_2 = ptr[2];
    uint64_t hidx_3 = ptr[3];
    uint8_t sbn_high = popcnt_16b(h16b);

    store_sz16_as_sz32(out, lidx_0, offset);
    store_sz16_as_sz32(out + 4, lidx_1, offset);
    store_sz16_as_sz32(out + 8, lidx_2, offset);
    store_sz16_as_sz32(out + 12, lidx_3, offset);
    offset += inc16;
    out += sbn_low;

    store_sz16_as_sz32(out, hidx_0, offset);
    store_sz16_as_sz32(out + 4, hidx_1, offset);
    store_sz16_as_sz32(out + 8, hidx_2, offset);
    store_sz16_as_sz32(out + 12, hidx_3, offset);
#endif
    return sbn_low + sbn_high;
}

/* usage: Given a uint64_t, store the indices of set bits in an array of
 *      uint32_t.  The indices must be smaller than 2^32.
 * params:
 *      1) out: an array of uint32_t for storing the indices; must hold at
 *              least 64 elements
 *      2) offset: a uint64_t that stores the offset for all indices. The
 *              lowest 32-bit is the offset and should be propagated to other
 *              32 bits. E.g. To set offset to 8, 0x0000000800000008ULL should
 *              be passed
 *      3) b64: a uint64_t
 * return: the number of set bits */
static inline uint64_t
sbidx_in_64b_sz32(uint32_t* out, uint64_t offset, uint64_t b64) {
    if(unlikely(!b64)) {
        return 0;
    }

    uint64_t sbnum = 0;
#if defined(__AVX512BW__)
    sbnum = sbidx_in_64b_sz32_avx512(out, offset, b64);

#elif BLK_LANCZOS_BITMAP_SIZE == 16
    const uint64_t inc32 = 0x0000002000000020ULL;
    sbnum += sbidx_in_32b_sz32(out, offset,
                               (uint16_t) b64, (uint16_t) (b64 >> 16));
    b64 >>= 32;
    sbnum += sbidx_in_32b_sz32(out + sbnum, offset + inc32,
                               (uint16_t) b64, (uint16_t) (b64 >> 16));
#else
    const uint64_t inc16 = 0x0000001000000010ULL;
    sbnum += sbidx_in_16b_sz32(out, offset,
                               (uint8_t) b64, (uint8_t) (b64 >> 8));
    b64 >>= 16;
    offset += inc16;
    sbnum += sbidx_in_16b_sz32(out + sbnum, offset,
                               (uint8_t) b64, (uint8_t) (b64 >> 8));
    b64 >>= 16;
    offset += inc16;
    sbnum += sbidx_in_16b_sz32(out + sbnum, offset,
                               (uint8_t) b64, (uint8_t) (b64 >> 8));
    b64 >>= 16;
    offset += inc16;
    sbnum += sbidx_in_16b_sz32(out + sbnum, offset,
                               (uint8_t) b64, (uint8_t) (b64 >> 8));
#endif

    return sbnum;
}

/* usage: Given an array of uint64_t, store the indices of set bits in an array
 *      of uint32_t.  The indices must be smaller than 2^32.
 * params:
 *      1) out: an array of uint32_t for storing the indices; must hold at
 *              least as many elements as the size of the uint64_t array
 *      2) arr: an array of uint64_t
 *      3) size: size of the array
 * return: the number of set bits */
static inline uint64_t
sbidx_in_64arr(uint32_t* const restrict out, const uint64_t* restrict arr,
               size_t size) {
    uint64_t base = 0x0ULL;
    const uint64_t inc64 = 0x0000004000000040ULL;
    uint64_t sbnum = 0;
    uint64_t head = size & ~0x3ULL; // multiple of 4
    for(uint64_t i = 0; i < head; i += 4) {
        sbnum += sbidx_in_64b_sz32(out + sbnum, base, arr[i+0]);
        base += inc64;
        sbnum += sbidx_in_64b_sz32(out + sbnum, base, arr[i+1]);
        base += inc64;
        sbnum += sbidx_in_64b_sz32(out + sbnum, base, arr[i+2]);
        base += inc64;
        sbnum += sbidx_in_64b_sz32(out + sbnum, base, arr[i+3]);
        base += inc64;
    }
    for(uint64_t i = head; i < size; ++i) {
        sbnum += sbidx_in_64b_sz32(out + sbnum, base, arr[i]);
        base += inc64;
    }
    return sbnum;
}

#endif /* __BLK_LANCZOS_BITMAP_TABLE_H__ */
