#include "grp64_gf16.h"
#include "gf16.h"
#include "util.h"
#include <assert.h>

/* ========================================================================
 * function implementations
 * ======================================================================== */

/* usage: given a struct Grp64GF16, set all of its elements to zero
 * params:
 *      1) g: ptr to struct Grp64GF16
 * return: void */
void
grp64_gf16_zero(Grp64GF16* g) {
#if defined(__AVX__)
    __m256i v = _mm256_setzero_si256();
    _mm256_store_si256((__m256i*) (g->b), v);
#else
    g->b[0] = 0;
    g->b[1] = 0;
    g->b[2] = 0;
    g->b[3] = 0;
#endif
}

/* usage: given a struct Grp64GF16, find its zero elements
 * params:
 *      1) g: ptr to struct Grp64GF16
 * return: a uint64_t that encoding the location of zero elements. If the i-th
 *      element is zero, then the i-th bit is set, and so on */
uint64_t
grp64_gf16_zpos(const Grp64GF16* g) {
    uint64_t t = g->b[0] | g->b[1] | g->b[2] | g->b[3];
    return ~t;
}

/* usage: given a struct Grp64GF16, find its non-zero elements
 * params:
 *      1) g: ptr to struct Grp64GF16
 * return: a uint64_t that encoding the location of non-zero elements. If the
 *      i-th element is non-zero, then the i-th bit is set, and so on */
uint64_t
grp64_gf16_nzpos(const Grp64GF16* g) {
    return ~grp64_gf16_zpos(g);
}

/* usage: create a copy of a struct Grp64GF16
 * params:
 *      1) dst: ptr to struct Grp64GF16 for holding the copy
 *      2) src: ptr to struct Grp64GF16. The original
 * return: void */
void
grp64_gf16_copy(Grp64GF16* restrict dst, const Grp64GF16* restrict src) {
#if defined(__AVX__)
    __m256i v = _mm256_load_si256((__m256i*) (src->b));
    _mm256_store_si256((__m256i*) (dst->b), v);
#else
    dst->b[0] = src->b[0];
    dst->b[1] = src->b[1];
    dst->b[2] = src->b[2];
    dst->b[3] = src->b[3];
#endif
}

/* usage: given a struct Grp64GF16, randomize its elements
 * params:
 *      1) g: ptr to struct Grp64GF16
 * return: void */
void
grp64_gf16_rand(Grp64GF16* g) {
    g->b[0] = uint64_rand();
    g->b[1] = uint64_rand();
    g->b[2] = uint64_rand();
    g->b[3] = uint64_rand();
}

/* usage: given a struct Grp64GF16, set a subset of its elements to zero
 * params:
 *      1) g: ptr to struct Grp64GF16
 *      2) mask: a 64-bit integer that encodes which elements to zero.
 *          If the i-th bit is 0, then the i-th element will be zeroed.
 *          If 1, then the i-th element is kept.
 * return: void */
void
grp64_gf16_zero_subset(Grp64GF16* g, uint64_t mask) {
#if defined(__AVX2__)
    __m256i m = _mm256_set1_epi64x(mask);
    __m256i v = _mm256_load_si256((__m256i*) (g->b));
    v = _mm256_and_si256(v, m);
    _mm256_store_si256((__m256i*) (g->b), v);
#else
    g->b[0] &= mask;
    g->b[1] &= mask;
    g->b[2] &= mask;
    g->b[3] &= mask;
#endif
}

/* usage: given a struct Grp64GF16, set its i-th element to zero
 * params:
 *      1) g: ptr to struct Grp64GF16
 *      2) i: index of the element. 0 ~ 63
 * return: void */
void
grp64_gf16_zero_at(Grp64GF16* g, uint32_t i) {
    uint64_t mask = ~(0x1ULL << i);
    grp64_gf16_zero_subset(g, mask);
}

/* usage: given a struct Grp64GF16, return its i-th element
 * params:
 *      1) g: ptr to struct Grp64GF16
 *      2) i: index of the element. 0 ~ 63
 * return: the i-th element as a gf16_t */
gf16_t
grp64_gf16_at(const Grp64GF16* g, uint32_t i) {
    assert(i < 64);
//#if defined(__AVX2__)
//    // FIXME: this is not as efficient as the normal version below
//    __m256i v = _mm256_load_si256((__m256i*) (g->b));
//    __m256i mask = _mm256_set1_epi64x(0x1ULL);
//    v = _mm256_and_si256(_mm256_srli_epi64(v, i), mask);
//    __m256i sll_offsets = _mm256_set_epi64x(0x3ULL, 0x2ULL, 0x1ULL, 0x0ULL);
//    v = _mm256_sllv_epi64(v, sll_offsets);
//    uint64_t b0 = _mm256_extract_epi64(v, 0);
//    uint64_t b1 = _mm256_extract_epi64(v, 1);
//    uint64_t b2 = _mm256_extract_epi64(v, 2);
//    uint64_t b3 = _mm256_extract_epi64(v, 3);
//    return b0 | b1 | b2 | b3;
//#else
    uint64_t b0 = (g->b[0] >> i) & 0x1ULL;
    uint64_t b1 = (g->b[1] >> i) & 0x1ULL;
    uint64_t b2 = (g->b[2] >> i) & 0x1ULL;
    uint64_t b3 = (g->b[3] >> i) & 0x1ULL;
    return b0 | (b1 << 1) | (b2 << 2) | (b3 << 3);
//#endif
}

/* usage: given a struct Grp64GF16, add a value to its i-th element
 * params:
 *      1) g: ptr to struct Grp64GF16
 *      2) i: index of the element. 0 ~ 63
 *      3) v: the value to add
 * return: void */
void
grp64_gf16_add_at(Grp64GF16* g, uint32_t i, gf16_t v) {
    assert(i < 64);
    assert(v <= GF16_MAX);
    uint64_t b0 = v & 0x1ULL;
    uint64_t b1 = (v >> 1) & 0x1ULL;
    uint64_t b2 = (v >> 2) & 0x1ULL;
    uint64_t b3 = (v >> 3) & 0x1ULL;
    g->b[0] ^= b0 << i;
    g->b[1] ^= b1 << i;
    g->b[2] ^= b2 << i;
    g->b[3] ^= b3 << i;
}

/* usage: given a struct Grp64GF16, set its i-th element
 * params:
 *      1) g: ptr to struct Grp64GF16
 *      2) i: index of the element. 0 ~ 63
 *      3) v: value of the i-th element
 * return: void */
void
grp64_gf16_set_at(Grp64GF16* g, uint32_t i, gf16_t v) {
    assert(i < 64);
    assert(v <= GF16_MAX);
    uint64_t b0 = v & 0x1ULL;
    uint64_t b1 = (v >> 1) & 0x1ULL;
    uint64_t b2 = (v >> 2) & 0x1ULL;
    uint64_t b3 = v >> 3;

    grp64_gf16_zero_at(g, i);
    g->b[0] |= b0 << i;
    g->b[1] |= b1 << i;
    g->b[2] |= b2 << i;
    g->b[3] |= b3 << i;
}

/* usage: given a struct Grp64GF16 a and b, replace a subset of elements of
 *      a with the corresponding elements of b
 * params:
 *      1) a: ptr to struct Grp64GF16
 *      2) b: ptr to struct Grp64GF16
 *      3) mask: a 64-bit integer that encodes which elements of a to keep. If
 *          the i-th bit is set, then the i-th element of a is kept. If 0, then
 *          the i-th element is replaced by that of b.
 * return: void */
void
grp64_gf16_mixi(Grp64GF16* restrict a, const Grp64GF16* restrict b,
                uint64_t mask) {
#if defined(__AVX2__)
    __m256i av = _mm256_load_si256((__m256i*) (a->b));
    __m256i bv = _mm256_load_si256((__m256i*) (b->b));
    __m256i m = _mm256_set1_epi64x(mask);
    av = _mm256_and_si256(av, m);
    bv = _mm256_andnot_si256(m, bv);
    _mm256_store_si256((__m256i*)(a->b), _mm256_xor_si256(av, bv));
#else
    uint64_t nd = ~mask;
    a->b[0] = (a->b[0] & mask) ^ (b->b[0] & nd);
    a->b[1] = (a->b[1] & mask) ^ (b->b[1] & nd);
    a->b[2] = (a->b[2] & mask) ^ (b->b[2] & nd);
    a->b[3] = (a->b[3] & mask) ^ (b->b[3] & nd);
#endif
}

/* usage: given 2 size-2 arrays of of struct Grp64GF16 a and b, replace a
 *      subset of elements of a with the corresponding elements of b
 * params:
 *      1) a: ptr to an array of size 2 of struct Grp64GF16
 *      2) b: ptr to an array of size 2 of  struct Grp64GF16
 *      3) mask: a 64-bit integer that encodes which elements of a to keep. If
 *          the i-th bit is set, then the i-th element of a is kept. If 0, then
 *          the i-th element is replaced by that of b.
 * return: void */
void
grp64_gf16_mixi_x2(Grp64GF16* restrict a, const Grp64GF16* restrict b,
                   uint64_t mask) {
#if defined(__AVX512F__)
    __m512i va = _mm512_loadu_si512(a);
    __m512i vb = _mm512_loadu_si512(b);
    __m512i vm = _mm512_set1_epi64(mask);
    va = _mm512_and_si512(va, vm);
    vb = _mm512_andnot_si512(vm, vb);
    _mm512_storeu_si512(a, _mm512_xor_si512(va, vb));
#else
    grp64_gf16_mixi(a, b, mask);
    grp64_gf16_mixi(a + 1, b + 1, mask);
#endif
}

/* usuage: given a struct Grp64GF16, return the index of the 1st non-zero
 *      element.
 * params:
 *      1) g: ptr to struct Grp64GF16
 * return: index of the 1st non-zero element. UINT32_MAX if all elements are
 *      zero */
uint32_t
grp64_gf16_1st_nz_idx(const Grp64GF16* g) {
    uint64_t v = g->b[0] | g->b[1] | g->b[2] | g->b[3];
    if(unlikely(v == 0))
        return UINT32_MAX;

    return uint64_t_ctz(v);
}

/* usage: given 2 struct Grp64GF16 a and b, compute a + b and store the result
 *      back into a.
 * params:
 *      1) a: ptr to struct Grp64GF16; point to a
 *      2) b: ptr to struct Grp64GF16. point to b
 * return: void */
void
grp64_gf16_addi(Grp64GF16* restrict a, const Grp64GF16* restrict b) {
#if defined(__AVX2__)
    __m256i av = _mm256_load_si256((__m256i*) (a->b));
    __m256i bv = _mm256_load_si256((__m256i*) (b->b));
    _mm256_store_si256((__m256i*)(a->b), _mm256_xor_si256(av, bv));
#else
    a->b[0] ^= b->b[0];
    a->b[1] ^= b->b[1];
    a->b[2] ^= b->b[2];
    a->b[3] ^= b->b[3];
#endif
}

/* usage: given 2 size-2 arrays of struct Grp64GF16 a and b, compute a + b and
 *      store the result back into a.
 * params:
 *      1) a: ptr to struct Grp64GF16; point to a
 *      2) b: ptr to struct Grp64GF16. point to b
 * return: void */
void
grp64_gf16_addi_x2(Grp64GF16* restrict a, const Grp64GF16* restrict b) {
#if defined(__AVX512F__)
    __m512i va = _mm512_loadu_si512(a);
    __m512i vb = _mm512_loadu_si512(b);
    _mm512_storeu_si512(a, _mm512_xor_si512(va, vb));
#else
    grp64_gf16_addi(a, b);
    grp64_gf16_addi(a + 1, b + 1);
#endif
}

/* usage: given 2 struct Grp64GF16 a and b, compute a - b and store the result
 *      back into a.
 * params:
 *      1) a: ptr to struct Grp64GF16; point to a
 *      2) b: ptr to struct Grp64GF16. point to b
 * return: void */
void
grp64_gf16_subi(Grp64GF16* restrict a, const Grp64GF16* restrict b) {
    grp64_gf16_addi(a, b);
}

#if defined(__AVX512F__)

static force_inline __m512i
grp64_gf16_mul_scalar_reg_avx512_no_split(const __m512i v, const __mmask8 m0,
                                          const __mmask8 m1, const __mmask8 m2,
                                          const __mmask8 m3) {
    // format for v:
    //     <-      for c0     -><-        for c1     ->
    // (low) v0 | v1 | v2 | v3 || v0' | v1' | v2' | v3' (high)

    // (low) v3 | v0 | v1 | v2 || v3' | v0' | v1' | v2' (high)
    __m512i vsl1 = _mm512_permutex_epi64(v, 0x93); // 0b10010011
    // (low) v2 | v3 | v0 | v1 || v2' | v3' | v0' | v1' (high)
    __m512i vsl2 = _mm512_permutex_epi64(v, 0x4E); // 0b01001110
    // (low) v1 | v2 | v3 | v0 || v1' | v2' | v3' | v0' (high)
    __m512i vsl3 = _mm512_permutex_epi64(v, 0x39); // 0b10010011

    __m512i zv = _mm512_setzero_si512();
    // (low) b0 | b1 | b2 | b3 || b0' | b1' | b2' | b3' (high)
    __m512i b03 = _mm512_mask_blend_epi64((__mmask8) m0, zv, v);
    // (low) b4 | b1 | b2 | b3 || b4' | b1' | b2' | b3' (high)
    __m512i b14 = _mm512_mask_blend_epi64((__mmask8) m1, zv, vsl1);
    // (low) b4 | b5 | b2 | b3 || b4' | b5' | b2' | b3' (high)
    __m512i b25 = _mm512_mask_blend_epi64((__mmask8) m2, zv, vsl2);
    // (low) b4 | b5 | b6 | b3 || b4' | b5' | b6' | b3' (high)
    __m512i b36 = _mm512_mask_blend_epi64((__mmask8) m3, zv, vsl3);

    // (low)  0 | b4 | b5 | b6 ||   0 | b4' | b5' | b6' (high)
    __m512i t0 = _mm512_mask_blend_epi64((__mmask8) (m3 & 0xEE), zv, v);
    // (low)  0 | b4 | b5 |  0 ||   0 | b4' | b5' |  0 (high)
    __m512i t1 = _mm512_mask_blend_epi64((__mmask8) (m2 & 0x66), zv, vsl3);
    // (low)  0 | b4 |  0 |  0 ||   0 | b4' |   0 |  0 (high)
    __m512i t2 = _mm512_mask_blend_epi64((__mmask8) (m1 & 0x22), zv, vsl2);

    t0 = _mm512_xor_epi64(t0, t1);
    t0 = _mm512_xor_epi64(t0, t2);

    __m512i res = _mm512_xor_si512(b03, b14);
    res = _mm512_xor_si512(res, b25);
    res = _mm512_xor_si512(res, b36);
    res = _mm512_xor_si512(res, t0);
    return res;
}

static force_inline void
grp64_gf16_mul_scalar_reg_avx512(__m256i* restrict v0, __m256i* restrict v1,
                                 const __m512i v, const __mmask8 m0,
                                 const __mmask8 m1, const __mmask8 m2,
                                 const __mmask8 m3) {
    __m512i res = grp64_gf16_mul_scalar_reg_avx512_no_split(v, m0, m1, m2, m3);
    *v1 = _mm512_extracti64x4_epi64(res, 1);
    *v0 = _mm512_extracti64x4_epi64(res, 0);
    //*v0 = _mm512_castsi512_si256(res);
}

static const __mmask8 g_2bits_to_mmask[4] = { 0x00, 0x0F, 0xF0, 0xFF };

static force_inline __mmask8
mmask_from_2b(uint8_t b) {
    assert(b < 0x4);
    return g_2bits_to_mmask[b];
}

void
grp64_gf16_mul_scalar_from_bs_adj_avx512(__m256i* restrict v0,
                                         __m256i* restrict v1,
                                         const Grp64GF16* restrict src,
                                         const Grp64GF16* restrict g,
                                         uint32_t i) {
    __m512i v = _mm512_load_si512(src);
    __mmask8 m0 = mmask_from_2b((g->b[0] >> i) & 0x3U);
    __mmask8 m1 = mmask_from_2b((g->b[1] >> i) & 0x3U);
    __mmask8 m2 = mmask_from_2b((g->b[2] >> i) & 0x3U);
    __mmask8 m3 = mmask_from_2b((g->b[3] >> i) & 0x3U);
    grp64_gf16_mul_scalar_reg_avx512(v0, v1, v, m0, m1, m2, m3);
}

__m512i
grp64_gf16_mul_scalar_from_bs_adj_avx512_no_split(const Grp64GF16* restrict src,
                                                  const Grp64GF16* restrict g,
                                                  uint32_t i) {
    __m512i v = _mm512_load_si512(src);
    __mmask8 m0 = mmask_from_2b((g->b[0] >> i) & 0x3U);
    __mmask8 m1 = mmask_from_2b((g->b[1] >> i) & 0x3U);
    __mmask8 m2 = mmask_from_2b((g->b[2] >> i) & 0x3U);
    __mmask8 m3 = mmask_from_2b((g->b[3] >> i) & 0x3U);
    return grp64_gf16_mul_scalar_reg_avx512_no_split(v, m0, m1, m2, m3);
}

void
grp64_gf16_muli_scalar_from_bs_2x1_avx512(__m256i* restrict v0,
                                          __m256i* restrict v1,
                                          const Grp64GF16* restrict src0,
                                          const Grp64GF16* restrict src1,
                                          const Grp64GF16* restrict g,
                                          uint32_t i) {
    __m512i vlo = _mm512_castsi256_si512(_mm256_load_si256((__m256i*) src0));
    __m512i v = _mm512_inserti32x8(vlo, _mm256_load_si256((__m256i*) src1), 1);
    uint8_t m0 = uint8_extend_from_lsb((g->b[0] >> i) & 0x1U); // LSB
    uint8_t m1 = uint8_extend_from_lsb((g->b[1] >> i) & 0x1U); // 2nd LSB
    uint8_t m2 = uint8_extend_from_lsb((g->b[2] >> i) & 0x1U); // 3rd LSB
    uint8_t m3 = uint8_extend_from_lsb((g->b[3] >> i) & 0x1U); // 4th LSB
    grp64_gf16_mul_scalar_reg_avx512(v0, v1, v, m0, m1, m2, m3);
}

static force_inline void
grp64_gf16_mul_scalar_reg_from_coeff_2x2(__m256i* restrict v0,
                                         __m256i* restrict v1,
                                         const Grp64GF16* src0,
                                         const Grp64GF16* src1,
                                         gf16_t c0, gf16_t c1) {
    __m512i vlo = _mm512_castsi256_si512(_mm256_load_si256((__m256i*) src0));
    __m512i v = _mm512_inserti32x8(vlo, _mm256_load_si256((__m256i*) src1), 1);
    __mmask8 m0 = mmask_from_2b( (c0 & 0x1U) | ((c1 & 0x1U) << 1 ));
    __mmask8 m1 = mmask_from_2b( ((c0 & 0x2U) >> 1) | (c1 & 0x2U) );
    __mmask8 m2 = mmask_from_2b( ((c0 & 0x4U) >> 2) | ((c1 & 0x4U) >> 1) );
    __mmask8 m3 = mmask_from_2b( ((c0 & 0x8U) >> 3) | ((c1 & 0x8U) >> 2) );
    grp64_gf16_mul_scalar_reg_avx512(v0, v1, v, m0, m1, m2, m3);
}

static force_inline void
grp64_gf16_mul_scalar_reg_from_coeff_2x1(__m256i* restrict v0,
                                         __m256i* restrict v1,
                                         const Grp64GF16* src0,
                                         const Grp64GF16* src1, gf16_t c) {
    __m512i vlo = _mm512_castsi256_si512(_mm256_load_si256((__m256i*) src0));
    __m512i v = _mm512_inserti32x8(vlo, _mm256_load_si256((__m256i*) src1), 1);
    uint8_t m0 = uint8_extend_from_lsb(c & 0x1U); // LSB
    uint8_t m1 = uint8_extend_from_lsb((c >> 1) & 0x1U); // 2nd LSB
    uint8_t m2 = uint8_extend_from_lsb((c >> 2) & 0x1U); // 3rd LSB
    uint8_t m3 = uint8_extend_from_lsb(c >> 3); // 4th LSB
    grp64_gf16_mul_scalar_reg_avx512(v0, v1, v, m0, m1, m2, m3);
}

static force_inline void
grp64_gf16_mul_scalar_reg_from_coeff_1x2(__m256i* restrict v0,
                                         __m256i* restrict v1,
                                         const Grp64GF16* src,
                                         gf16_t c0, gf16_t c1) {
    __m512i v = _mm512_broadcast_i64x4(_mm256_load_si256((__m256i*)src));
    __mmask8 m0 = mmask_from_2b( (c0 & 0x1U) | ((c1 & 0x1U) << 1 ));
    __mmask8 m1 = mmask_from_2b( ((c0 & 0x2U) >> 1) | (c1 & 0x2U) );
    __mmask8 m2 = mmask_from_2b( ((c0 & 0x4U) >> 2) | ((c1 & 0x4U) >> 1) );
    __mmask8 m3 = mmask_from_2b( ((c0 & 0x8U) >> 3) | ((c1 & 0x8U) >> 2) );
    grp64_gf16_mul_scalar_reg_avx512(v0, v1, v, m0, m1, m2, m3);
}

__m512i
grp64_gf16_mul_scalar_from_bs_1x2_avx512(const Grp64GF16* src,
                                         const Grp64GF16* g,
                                         uint32_t i) {
    __m512i v = _mm512_broadcast_i64x4(_mm256_load_si256((__m256i*)src));
    __mmask8 m0 = mmask_from_2b((g->b[0] >> i) & 0x3U);
    __mmask8 m1 = mmask_from_2b((g->b[1] >> i) & 0x3U);
    __mmask8 m2 = mmask_from_2b((g->b[2] >> i) & 0x3U);
    __mmask8 m3 = mmask_from_2b((g->b[3] >> i) & 0x3U);
    return grp64_gf16_mul_scalar_reg_avx512_no_split(v, m0, m1, m2, m3);
}

#endif

#if defined(__AVX2__)

static force_inline __m256i
grp64_gf16_mul_scalar_reg_avx2(const __m256i v, __m256i m0, __m256i m1,
                               __m256i m2, __m256i m3) {
    // format of v:
    // (low) v0 | v1 | v2 | v3 (high)
    // (low) v3 | v0 | v1 | v2 (high)
    __m256i vsl1 = _mm256_permute4x64_epi64(v, 0x93); //[2,1,0,3] = 0b10010011
    // (low) v2 | v3 | v0 | v1 (high)
    __m256i vsl2 = _mm256_permute4x64_epi64(v, 0x4E); //[1,0,3,2] = 0b01001110
    // (low) v1 | v2 | v3 | v0 (high)
    __m256i vsl3 = _mm256_permute4x64_epi64(v, 0x39); //[0,3,2,1] = 0b00111001

    // (low) b0 | b1 | b2 | b3 (high)
    __m256i b03 = _mm256_and_si256(v, m0); // LSB
    // (low) b4 | b1 | b2 | b3 (high)
    __m256i b14 = _mm256_and_si256(vsl1, m1); // 2nd LSB
    // (low) b4 | b5 | b2 | b3 (high)
    __m256i b25 = _mm256_and_si256(vsl2, m2); // 3rd LSB
    // (low) b4 | b5 | b6 | b3 (high)
    __m256i b36 = _mm256_and_si256(vsl3, m3); // 4th LSB

    // (low) b3 | b4 | b5 | b6 (high)
    __m256i tmp0 = _mm256_and_si256(v, m3);
    // (low) b3 | b4 | b5 | b2 (high)
    __m256i tmp1 = _mm256_and_si256(vsl3, m2);
    // (low) b3 | b4 | b1 | b2 (high)
    __m256i tmp2 = _mm256_and_si256(vsl2, m1);

    __m256i all0 = _mm256_setzero_si256();
    // (low) 0 | b4 | b5 | b6 (high)
    tmp0 = _mm256_blend_epi32(all0, tmp0, 0xFC);
    // (low) 0 | b4 | b5 | 0 (high)
    tmp1 = _mm256_blend_epi32(all0, tmp1, 0x3C);
    // (low) 0 | b4 | 0 | 0 (high)
    tmp2 = _mm256_blend_epi32(all0, tmp2, 0x0C);

    __m256i tmp3 = _mm256_xor_si256(tmp0, tmp1);
    tmp3 = _mm256_xor_si256(tmp3, tmp2);

    // (low) b0 ^ b4 | b1 ^ b5 | b2 ^ b6 | b3 (high)
    __m256i res = _mm256_xor_si256(b03, b14);
    res = _mm256_xor_si256(res, b25);
    res = _mm256_xor_si256(res, b36);
    res = _mm256_xor_si256(res, tmp3);

    return res;
}

/* usage: given struct Grp64GF16 a, g, and an index i, extract the i-th element
 *      c in g as the multipler, then compute a * c. Note that both the value
 *      of a and the product are stored as a 256-bit YMM register.
 * params:
 *      1) v: a 256-bit YMM register which stores the struct Grp64GF16 a
 *      2) g: ptr to struct Grp64GF16. point to g
 *      3) i: the index
 * return: the product as a 256-bit YMM register */
__m256i
grp64_gf16_mul_scalar_from_bs_avx2(const __m256i v, const Grp64GF16* g, uint32_t i) {
    __m256i vg = _mm256_load_si256((__m256i*) g);
    __m256i lsb_extractor = _mm256_set1_epi64x(0x1ULL);
    vg = _mm256_and_si256(_mm256_srli_epi64(vg, i), lsb_extractor);
    vg = _mm256_cmpeq_epi64(vg, lsb_extractor);
    __m256i m0 = _mm256_permute4x64_epi64(vg, 0x00); // [0, 0, 0, 0]
    __m256i m1 = _mm256_permute4x64_epi64(vg, 0x55); // [1, 1, 1, 1]
    __m256i m2 = _mm256_permute4x64_epi64(vg, 0xAA); // [2, 2, 2, 2]
    __m256i m3 = _mm256_permute4x64_epi64(vg, 0xFF); // [3, 3, 3, 3]
    return grp64_gf16_mul_scalar_reg_avx2(v, m0, m1, m2, m3);
}

static force_inline __m256i
grp64_gf16_mul_scalar_from_coeff_avx2(const __m256i v, gf16_t c) {
    __m256i cv = _mm256_set1_epi64x(c);
    __m256i lsb_extractor = _mm256_set1_epi64x(0x1ULL);
    __m256i m0 = _mm256_and_si256(cv, lsb_extractor);
    __m256i m1 = _mm256_and_si256(_mm256_srli_epi64(cv, 1), lsb_extractor);
    __m256i m2 = _mm256_and_si256(_mm256_srli_epi64(cv, 2), lsb_extractor);
    __m256i m3 = _mm256_and_si256(_mm256_srli_epi64(cv, 3), lsb_extractor);
    m0 = _mm256_cmpeq_epi64(m0, lsb_extractor);
    m1 = _mm256_cmpeq_epi64(m1, lsb_extractor);
    m2 = _mm256_cmpeq_epi64(m2, lsb_extractor);
    m3 = _mm256_cmpeq_epi64(m3, lsb_extractor);
    return grp64_gf16_mul_scalar_reg_avx2(v, m0, m1, m2, m3);
}

#endif

static force_inline void
grp64_gf16_mul_scalar_reg(uint64_t out[4], const Grp64GF16* src,
                          uint64_t m0, uint64_t m1, uint64_t m2, uint64_t m3) {
    uint64_t b0 = 0; uint64_t b1 = 0; uint64_t b2 = 0; uint64_t b3 = 0;
    uint64_t b4 = 0; uint64_t b5 = 0; uint64_t b6 = 0;
    // LSB
    b0 ^= src->b[0] & m0;
    b1 ^= src->b[1] & m0;
    b2 ^= src->b[2] & m0;
    b3 ^= src->b[3] & m0;
    // 2nd LSB
    b1 ^= src->b[0] & m1;
    b2 ^= src->b[1] & m1;
    b3 ^= src->b[2] & m1;
    b4 ^= src->b[3] & m1;
    // 3rd LSB
    b2 ^= src->b[0] & m2;
    b3 ^= src->b[1] & m2;
    b4 ^= src->b[2] & m2;
    b5 ^= src->b[3] & m2;
    // 4th LSB
    b3 ^= src->b[0] & m3;
    b4 ^= src->b[1] & m3;
    b5 ^= src->b[2] & m3;
    b6 ^= src->b[3] & m3;
    // reduction with irreducible polynomial x^4 + x + 1 (0b10011)
    // 7-th bit
    b3 ^= b6;
    b2 ^= b6;
    // 6-th bit
    b2 ^= b5;
    b1 ^= b5;
    // 5-th bit
    b1 ^= b4;
    b0 ^= b4;

    out[0] = b0;
    out[1] = b1;
    out[2] = b2;
    out[3] = b3;
}

static force_inline void
grp64_gf16_mul_scalar_from_coeff(uint64_t out[4], const Grp64GF16* src,
                                 gf16_t c) {
    //uint64_t mask0 = uint64_extend_nz(c & 0x1ULL); // LSB
    //uint64_t mask1 = uint64_extend_nz(c & 0x2ULL); // 2nd LSB
    //uint64_t mask2 = uint64_extend_nz(c & 0x4ULL); // 3rd LSB
    //uint64_t mask3 = uint64_extend_nz(c & 0x8ULL); // 4th LSB
    uint64_t mask0 = uint64_extend_from_lsb(c & 0x1ULL); // LSB
    uint64_t mask1 = uint64_extend_from_lsb((c >> 1) & 0x1ULL); // 2nd LSB
    uint64_t mask2 = uint64_extend_from_lsb((c >> 2) & 0x1ULL); // 3rd LSB
    uint64_t mask3 = uint64_extend_from_lsb(c >> 3); // 4th LSB
    grp64_gf16_mul_scalar_reg(out, src, mask0, mask1, mask2, mask3);
}

static force_inline void
grp64_gf16_mul_scalar_from_bs(uint64_t out[4], const Grp64GF16* src,
                              const Grp64GF16* g, uint32_t i) {
    uint64_t b0 = (g->b[0] >> i) & 0x1ULL;
    uint64_t b1 = (g->b[1] >> i) & 0x1ULL;
    uint64_t b2 = (g->b[2] >> i) & 0x1ULL;
    uint64_t b3 = (g->b[3] >> i) & 0x1ULL;
    uint64_t mask0 = uint64_extend_from_lsb(b0); // LSB
    uint64_t mask1 = uint64_extend_from_lsb(b1); // 2nd LSB
    uint64_t mask2 = uint64_extend_from_lsb(b2); // 3rd LSB
    uint64_t mask3 = uint64_extend_from_lsb(b3); // 4th LSB
    grp64_gf16_mul_scalar_reg(out, src, mask0, mask1, mask2, mask3);
}

/* usage: given a struct Grp64GF16 and a gf16_t scalar c, multiply 64 elements
 *      in the struct with c and store the result into a given container
 * params:
 *      1) dst: ptr to struct Grp64GF16; storage for the result
 *      2) src: ptr to struct Grp64GF16. the source
 *      3) c: the scalar multiplier
 * return: void */
void
grp64_gf16_mul_scalar(Grp64GF16* restrict dst, const Grp64GF16* restrict src,
                      gf16_t c) {
#if defined(__AVX2__)
    __m256i v = _mm256_load_si256((__m256i*) src->b);
    __m256i res = grp64_gf16_mul_scalar_from_coeff_avx2(v, c);
    _mm256_store_si256((__m256i*) dst->b, res);
#else
    grp64_gf16_mul_scalar_from_coeff(dst->b, src, c);
#endif
}

/* usage: given a struct Grp64GF16 and a gf16_t scalar c, multiply 64 elements
 *      in the struct with c and store the result back into the struct Grp64GF16
 * params:
 *      1) src: ptr to struct Grp64GF16. the source
 *      2) c: the scalar multiplier
 * return: void */
void
grp64_gf16_muli_scalar(Grp64GF16* restrict src, gf16_t c) {
#if defined(__AVX2__)
    __m256i v = _mm256_load_si256((__m256i*) src->b);
    __m256i res = grp64_gf16_mul_scalar_from_coeff_avx2(v, c);
    _mm256_store_si256((__m256i*) src->b, res);
#else
    grp64_gf16_mul_scalar_from_coeff(src->b, src, c);
#endif
}

/* usage: given 2 struct Grp64GF16 and a gf16_t scalar c, multiply 64 elements
 *      in both struct with c and store the result back respectively
 * params:
 *      1) s0: ptr to struct Grp64GF16. the 1st source
 *      2) s1: ptr to struct Grp64GF16. the 2nd source
 *      2) c: the scalar multiplier
 * return: void */
void
grp64_gf16_muli_scalar_2x1(Grp64GF16* restrict s0, Grp64GF16* restrict s1,
                           gf16_t c) {
#if defined(__AVX512F__)
    __m256i cs0, cs1;
    grp64_gf16_mul_scalar_reg_from_coeff_2x1(&cs0, &cs1, s0, s1, c);
    _mm256_store_si256((__m256i*) s0->b, cs0);
    _mm256_store_si256((__m256i*) s1->b, cs1);
#else
    grp64_gf16_muli_scalar(s0, c);
    grp64_gf16_muli_scalar(s1, c);
#endif
}

/* usage: given 2 struct Grp64GF16 a and b, and a gf16_t scalar c, compute
 *      a + b * c, and store the result back into a.
 * params:
 *      1) a: ptr to struct Grp64GF16; point to a
 *      2) b: ptr to struct Grp64GF16. point to b
 *      3) c: the scalar multiplier
 * return: void */
void
grp64_gf16_fmaddi_scalar(Grp64GF16* restrict a, const Grp64GF16* restrict b,
                         gf16_t c) {
#if defined(__AVX2__)
    __m256i v = _mm256_load_si256((__m256i*) b->b);
    __m256i res = grp64_gf16_mul_scalar_from_coeff_avx2(v, c);
    __m256i va = _mm256_load_si256((__m256i*) a->b);
    _mm256_store_si256((__m256i*) a->b, _mm256_xor_si256(va, res));
#else
    Grp64GF16 tmp;
    grp64_gf16_mul_scalar_from_coeff(tmp.b, b, c);
    grp64_gf16_addi(a, &tmp);
#endif
}

/* usage: given 3 struct Grp64GF16 a, b, g, and index i, extract the i-th
 *      gf16_t element c from g, compute a + b * c, and store the result back
 *      into a.
 * params:
 *      1) a: ptr to struct Grp64GF16; point to a
 *      2) b: ptr to struct Grp64GF16. point to b
 *      3) g: ptr to struct Grp64GF16. point to g
 *      4) i: the index
 * return: void */
void
grp64_gf16_fmaddi_scalar_bs(Grp64GF16* restrict a, const Grp64GF16* restrict b,
                            const Grp64GF16* restrict g, uint32_t i) {
#if defined(__AVX2__)
    __m256i v = _mm256_load_si256((__m256i*) b->b);
    __m256i res = grp64_gf16_mul_scalar_from_bs_avx2(v, g, i);
    __m256i va = _mm256_load_si256((__m256i*) a->b);
    _mm256_store_si256((__m256i*) a->b, _mm256_xor_si256(va, res));
#else
    Grp64GF16 tmp; grp64_gf16_mul_scalar_from_bs(tmp.b, b, g, i);
    grp64_gf16_addi(a, &tmp);
#endif
}

/* usage: given 3 struct Grp64GF16 a, b, c, and 2 gf16_t scalar m0 m1,
 *      compute a + c * m0 and b + c * m1, and store the result back into a
 *      and b respectively
 * params:
 *      1) a: ptr to struct Grp64GF16; point to a
 *      2) b: ptr to struct Grp64GF16. point to b
 *      3) c: ptr to struct Grp64GF16. point to c
 *      4) m0: the 1st scalar multiplier
 *      5) m1: the 2nd scalar multiplier
 * return: void */
void
grp64_gf16_fmaddi_scalar_2x1(Grp64GF16* restrict a, Grp64GF16* restrict b,
                             const Grp64GF16* restrict c, gf16_t m0, gf16_t m1) {
#if defined(__AVX512F__)
    __m256i cm0, cm1;
    grp64_gf16_mul_scalar_reg_from_coeff_1x2(&cm0, &cm1, c, m0, m1);
    __m256i va = _mm256_load_si256((__m256i*) a->b);
    __m256i vb = _mm256_load_si256((__m256i*) b->b);
    _mm256_store_si256((__m256i*) a->b, _mm256_xor_si256(va, cm0));
    _mm256_store_si256((__m256i*) b->b, _mm256_xor_si256(vb, cm1));
#else
    grp64_gf16_fmaddi_scalar(a, c, m0);
    grp64_gf16_fmaddi_scalar(b, c, m1);
#endif
}

/* usage: given 3 struct Grp64GF16 a, b, c, and 2 gf16_t scalar m0 m1,
 *      compute a + b * m0 + c * m1, and store the result back into a
 * params:
 *      1) a: ptr to struct Grp64GF16; point to a
 *      2) b: ptr to struct Grp64GF16. point to b
 *      3) c: ptr to struct Grp64GF16. point to c
 *      4) m0: the 1st scalar multiplier
 *      5) m1: the 2nd scalar multiplier
 * return: void */
void
grp64_gf16_fmaddi_scalar_1x2(Grp64GF16* restrict a, const Grp64GF16* restrict b,
                             const Grp64GF16* restrict c, gf16_t m0, gf16_t m1) {
#if defined(__AVX512F__)
    __m256i bm0, dm1;
    grp64_gf16_mul_scalar_reg_from_coeff_2x2(&bm0, &dm1, b, c, m0, m1);
    __m256i va = _mm256_load_si256((__m256i*) a->b);
    bm0 = _mm256_xor_si256(bm0, dm1);
    _mm256_store_si256((__m256i*) a->b, _mm256_xor_si256(va, bm0));
#else
    grp64_gf16_fmaddi_scalar(a, b, m0);
    grp64_gf16_fmaddi_scalar(a, c, m1);
#endif
}

/* usage: given 2 struct Grp64GF16 a and b, and a gf16_t scalar c, compute
 *      a - b * c, and store the result back into a.
 * params:
 *      1) a: ptr to struct Grp64GF16; point to a
 *      2) b: ptr to struct Grp64GF16. point to b
 *      3) c: the scalar multiplier */
void
grp64_gf16_fmsubi_scalar(Grp64GF16* restrict a, const Grp64GF16* restrict b,
                         gf16_t c) {
    grp64_gf16_fmaddi_scalar(a, b, c);
}

/* usage: given 3 struct Grp64GF16 a, b, g, and index i, extract the i-th
 *      gf16_t element c from g, compute a - b * c, and store the result back
 *      into a.
 * params:
 *      1) a: ptr to struct Grp64GF16; point to a
 *      2) b: ptr to struct Grp64GF16. point to b
 *      3) g: ptr to struct Grp64GF16. point to g
 *      4) i: the index
 * return: void */
void
grp64_gf16_fmsubi_scalar_bs(Grp64GF16* restrict a, const Grp64GF16* restrict b,
                            const Grp64GF16* restrict g, uint32_t i) {
    grp64_gf16_fmaddi_scalar_bs(a, b, g, i);
}

/* usage: given 2 struct Grp64GF16 a and b, a gf16_t scalar c, and a uint64_t
 *      d that encodes which elements to mask, compute b * c. and add the
 *      result to a based on the mask d. If the i-th bit of d is 1, then
 *      the i-th element of b*c is added to a. If 0, then the i-th element
 *      of a is untouched.
 * params:
 *      1) a: ptr to struct Grp64GF16; point to a
 *      2) b: ptr to struct Grp64GF16. point to b
 *      3) c: the scalar multiplier
 *      4) d: the mask
 * return: void */
void
grp64_gf16_fmaddi_scalar_mask(Grp64GF16* restrict a, const Grp64GF16* restrict b,
                              gf16_t c, uint64_t d) {
#if defined(__AVX2__)
    __m256i v = _mm256_load_si256((__m256i*) b->b);
    __m256i res = grp64_gf16_mul_scalar_from_coeff_avx2(v, c);
    res = _mm256_and_si256(res, _mm256_set1_epi64x(d));
    __m256i va = _mm256_load_si256((__m256i*) a->b);
    _mm256_store_si256((__m256i*) a->b, _mm256_xor_si256(va, res));
#else
    Grp64GF16 tmp;
    grp64_gf16_mul_scalar_from_coeff(tmp.b, b, c);
    grp64_gf16_zero_subset(&tmp, d);
    grp64_gf16_addi(a, &tmp);
#endif
}

/* usage: given 3 Grp64GF16 a, b, g, an index i, and a uint64_t d that encodes
 *      which elements to mask, extract the i-th gf16_t elements c from g, then
 *      compute b * c. The result is added back to a based on the mask d. If
 *      the j-th bit of d is 1, then the j-th element of b * c is added to a.
 *      If 0, then the j-th element of a is untouched.
 * params:
 *      1) a: ptr to struct Grp64GF16; point to a
 *      2) b: ptr to struct Grp64GF16. point to b
 *      3) g: ptr to struct Grp64GF16. point to g
 *      4) i: the index of which element in g to use as multiplier
 *      5) d: the mask
 * return: void */
void
grp64_gf16_fmaddi_scalar_mask_bs(Grp64GF16* restrict a,
                                 const Grp64GF16* restrict b,
                                 const Grp64GF16* restrict g,
                                 uint32_t i, uint64_t d) {
#if defined(__AVX2__)
    __m256i v = _mm256_load_si256((__m256i*) b->b);
    __m256i res = grp64_gf16_mul_scalar_from_bs_avx2(v, g, i);
    res = _mm256_and_si256(res, _mm256_set1_epi64x(d));
    __m256i va = _mm256_load_si256((__m256i*) a->b);
    _mm256_store_si256((__m256i*) a->b, _mm256_xor_si256(va, res));
#else
    Grp64GF16 tmp;
    grp64_gf16_mul_scalar_from_bs(tmp.b, b, g, i);
    grp64_gf16_zero_subset(&tmp, d);
    grp64_gf16_addi(a, &tmp);
#endif
}
