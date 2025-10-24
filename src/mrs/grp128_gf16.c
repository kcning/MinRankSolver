#include "grp128_gf16.h"
#include "util.h"
#include <assert.h>
#include "uint512_t.h"

/* ========================================================================
 * function implementations
 * ======================================================================== */

/* usage: given a struct Grp128GF16, set all of its elements to zero
 * params:
 *      1) g: ptr to struct Grp128GF16
 * return: void */
void
grp128_gf16_zero(Grp128GF16* g) {
#if defined(__AVX512F__)
    __m512i v = _mm512_setzero_si512();
    _mm512_store_si512(g, v);
#elif defined(__AVX__)
    __m256i* dst = (__m256i*) g->b;
    __m256i v = _mm256_setzero_si256();
    _mm256_store_si256(dst, v);
    _mm256_store_si256(dst+1, v);
#else
    memset(g->b, 0x0, sizeof(uint128_t) * 4);
#endif
}

/* usage: given a struct Grp128GF16, find its non-zero elements
 * params:
 *      1) ptr to a uint128_t. Upon return, it encodes the location of non-zero
 *          elements. If the i-th element is non-zero, then the i-th bit is set
 *          and so on.
 *      2) g: ptr to struct Grp128GF16
 * return: void */
void
grp128_gf16_nzpos(uint128_t* restrict out, const Grp128GF16* restrict g) {
    uint128_t t0; uint128_t_or(&t0, g->b, g->b + 1);
    uint128_t t1; uint128_t_or(&t1, g->b + 2, g->b + 3);
    uint128_t_or(out, &t0, &t1);
}

/* usage: given a struct Grp128GF16, find its zero elements
 * params:
 *      1) ptr to a uint128_t. Upon return, it encodes the location of zero
 *          elements. If the i-th element is zero, then the i-th bit is set and
 *          so on.
 *      2) g: ptr to struct Grp128GF16
 * return: void */
void
grp128_gf16_zpos(uint128_t* restrict out, const Grp128GF16* restrict g) {
    grp128_gf16_nzpos(out, g);
    uint128_t_negi(out);
}

/* usage: create a copy of a struct Grp128GF16
 * params:
 *      1) dst: ptr to struct Grp128GF16 for holding the copy
 *      2) src: ptr to struct Grp128GF16. The original
 * return: void */
void
grp128_gf16_copy(Grp128GF16* restrict dst, const Grp128GF16* restrict src) {
#if defined(__AVX512F__)
    __m512i v = _mm512_load_si512(src);
    _mm512_store_si512(dst, v);
#elif defined(__AVX__)
    __m256i* s = (__m256i*) src->b;
    __m256i* d = (__m256i*) dst->b;
    __m256i v0 = _mm256_load_si256(s);
    __m256i v1 = _mm256_load_si256(s + 1);
    _mm256_store_si256(d, v0);
    _mm256_store_si256(d + 1, v1);
#else
    memcpy(dst->b, src->b, sizeof(uint128_t) * 4);
#endif
}

/* usage: given a struct Grp128GF16, randomize its elements
 * params:
 *      1) g: ptr to struct Grp128GF16
 * return: void */
void
grp128_gf16_rand(Grp128GF16* g) {
    uint512_t_rand((uint512_t*) g->b );
}

#if defined(__AVX2__)

/* usage: given a struct grp128gf16, set a subset of its elements to zero
 * params:
 *      1) g: ptr to struct grp128gf16
 *      2) m: a 256-bit register that encodes which elements to zero.
 *          if the i-th bit is 0, then the i-th element will be zeroed.
 *          if 1, then the i-th element is kept. The upper 128-bit
 *          needs to be duplicate of the lower 128 bits. I.e. m[255:128]
 *          = m[127:0]
 * return: void */
void
grp128_gf16_zero_subset_avx2(Grp128GF16* restrict g, const __m256i m) {
#if defined(__AVX512F__)
    __m512i v = _mm512_load_si512(g->b);
    // [mask, mask, U, U]
    __m512i vm = _mm512_castsi256_si512(m);
    vm = _mm512_shuffle_i64x2(vm, vm, 0x0); // [mask, mask, mask, mask]
    _mm512_store_si512(g->b, _mm512_and_si512(v, vm));
#else
    __m256i* s = (__m256i*) g->b;
    __m256i v0 = _mm256_load_si256(s);
    __m256i v1 = _mm256_load_si256(s + 1);
    v0 = _mm256_and_si256(v0, m);
    v1 = _mm256_and_si256(v1, m);
    _mm256_store_si256(s, v0);
    _mm256_store_si256(s + 1, v1);
#endif
}

#endif

/* usage: given a struct Grp128GF16, set a subset of its elements to zero
 * params:
 *      1) g: ptr to struct Grp128GF16
 *      2) mask: ptr to a 128-bit integer that encodes which elements to zero.
 *          If the i-th bit is 0, then the i-th element will be zeroed.
 *          If 1, then the i-th element is kept.
 * return: void */
void
grp128_gf16_zero_subset(Grp128GF16* restrict g, const uint128_t* restrict mask) {
#if defined(__AVX512F__)
    __m512i v = _mm512_load_si512(g->b);
    // [mask, U, U, U]
    __m512i vm = _mm512_castsi128_si512(_mm_load_si128((__m128i*)mask));
    vm = _mm512_shuffle_i64x2(vm, vm, 0x0); // [mask, mask, mask, mask]
    _mm512_store_si512(g->b, _mm512_and_si512(v, vm));
#elif defined(__AVX2__)
    __m256i* s = (__m256i*) g->b;
    __m256i v0 = _mm256_load_si256(s);
    __m256i v1 = _mm256_load_si256(s + 1);
    // [mask, U]
    __m256i vm = _mm256_castsi128_si256(_mm_load_si128((__m128i*)mask));
    vm = _mm256_permute2x128_si256(vm, vm, 0x0); // [mask, mask]
    v0 = _mm256_and_si256(v0, vm);
    v1 = _mm256_and_si256(v1, vm);
    _mm256_store_si256(s, v0);
    _mm256_store_si256(s + 1, v1);
#else
    uint128_t_andi(g->b, mask);
    uint128_t_andi(g->b + 1, mask);
    uint128_t_andi(g->b + 2, mask);
    uint128_t_andi(g->b + 3, mask);
#endif
}

/* usage: given a struct Grp128GF16, set its i-th element to zero
 * params:
 *      1) g: ptr to struct Grp128GF16
 *      2) i: index of the element. 0 ~ 511
 * return: void */
void
grp128_gf16_zero_at(Grp128GF16* g, uint32_t i) {
    assert(i < 128);
    uint128_t mask; uint128_t_max(&mask);
    uint128_t_toggle_at(&mask, i);
    grp128_gf16_zero_subset(g, &mask);
}

/* usage: given a struct Grp128GF16, return its i-th element
 * params:
 *      1) g: ptr to struct Grp128GF16
 *      2) i: index of the element. 0 ~ 511
 * return: the i-th element as a gf16_t */
gf16_t
grp128_gf16_at(const Grp128GF16* g, uint32_t i) {
    assert(i < 128);
    // TODO: try to load only 8 bits each instead of 64 bits
    uint64_t b0 = uint128_t_at(g->b, i);
    uint64_t b1 = uint128_t_at(g->b + 1, i);
    uint64_t b2 = uint128_t_at(g->b + 2, i);
    uint64_t b3 = uint128_t_at(g->b + 3, i);
    return b0 | (b1 << 1) | (b2 << 2) | (b3 << 3);
}

/* usage: given a struct Grp128GF16, add a value to its i-th element
 * params:
 *      1) g: ptr to struct Grp128GF16
 *      2) i: index of the element. 0 ~ 511
 *      3) v: the value to add
 * return: void */
void
grp128_gf16_add_at(Grp128GF16* g, uint32_t i, gf16_t v) {
    assert(i < 128);
    assert(v <= GF16_MAX);
    uint64_t b0 = v & 0x1ULL;
    uint64_t b1 = (v >> 1) & 0x1ULL;
    uint64_t b2 = (v >> 2) & 0x1ULL;
    uint64_t b3 = (v >> 3) & 0x1ULL;
    if(b0)
        uint128_t_toggle_at(g->b, i);
    if(b1)
        uint128_t_toggle_at(g->b + 1, i);
    if(b2)
        uint128_t_toggle_at(g->b + 2, i);
    if(b3)
        uint128_t_toggle_at(g->b + 3, i);
}

/* usage: given a struct Grp128GF16, set its i-th element
 * params:
 *      1) g: ptr to struct Grp128GF16
 *      2) i: index of the element. 0 ~ 63
 *      3) v: value of the i-th element
 * return: void */
void
grp128_gf16_set_at(Grp128GF16* g, uint32_t i, gf16_t v) {
    assert(i < 128);
    assert(v <= GF16_MAX);
    uint64_t b0 = v & 0x1ULL;
    uint64_t b1 = (v >> 1) & 0x1ULL;
    uint64_t b2 = (v >> 2) & 0x1ULL;
    uint64_t b3 = (v >> 3) & 0x1ULL;
    uint128_t_set_at(g->b, i, b0);
    uint128_t_set_at(g->b + 1, i, b1);
    uint128_t_set_at(g->b + 2, i, b2);
    uint128_t_set_at(g->b + 3, i, b3);
}

/* usage: given a struct Grp128GF16 a and b, replace a subset of elements of
 *      a with the corresponding elements of b
 * params:
 *      1) a: ptr to struct Grp128GF16
 *      2) b: ptr to struct Grp128GF16
 *      3) mask: ptr to a uint128_t which encodes which elements of a to keep.
 *          If the i-th bit is set, then the i-th element of a is kept. If 0,
 *          then the i-th element is replaced by that of b.
 * return: void */
void
grp128_gf16_mixi(Grp128GF16* restrict a, const Grp128GF16* restrict b,
                 const uint128_t* restrict mask) {
#if defined(__AVX512F__)
    // [mask, U, U, U]
    __m512i vm = _mm512_castsi128_si512(_mm_load_si128((__m128i*)mask));
    vm = _mm512_shuffle_i64x2(vm, vm, 0x0); // [mask, mask, mask, mask]
    __m512i va = _mm512_load_si512(a->b);
    __m512i vb = _mm512_load_si512(b->b);
    va = _mm512_and_si512(va, vm);
    vb = _mm512_andnot_si512(vm, vb);
    _mm512_store_si512(a->b, _mm512_xor_si512(va, vb));
#elif defined(__AVX2__)
    __m256i* sa = (__m256i*) a->b;
    __m256i* sb = (__m256i*) b->b;
    // [mask, mask]
    __m256i vm = _mm256_castsi128_si256(_mm_load_si128((__m128i*)mask));
    vm = _mm256_permute2x128_si256(vm, vm, 0x0); // [mask, mask]
    __m256i va0 = _mm256_load_si256(sa);
    __m256i va1 = _mm256_load_si256(sa + 1);
    __m256i vb0 = _mm256_load_si256(sb);
    __m256i vb1 = _mm256_load_si256(sb + 1);
    va0 = _mm256_and_si256(va0, vm);
    va1 = _mm256_and_si256(va1, vm);
    vb0 = _mm256_andnot_si256(vm, vb0);
    vb1 = _mm256_andnot_si256(vm, vb1);
    _mm256_store_si256(sa, _mm256_xor_si256(va0, vb0));
    _mm256_store_si256(sa + 1, _mm256_xor_si256(va1, vb1));
#else
    uint128_t_mixi(a->b, b->b, mask);
    uint128_t_mixi(a->b + 1, b->b + 1, mask);
    uint128_t_mixi(a->b + 2, b->b + 2, mask);
    uint128_t_mixi(a->b + 3, b->b + 3, mask);
#endif
}

#if defined(__AVX2__)

/* usage: given a struct Grp128GF16 a and b, replace a subset of elements of
 *      a with the corresponding elements of b
 * params:
 *      1) a: ptr to struct Grp128GF16
 *      2) b: ptr to struct Grp128GF16
 *      3) m: 256-bit register which encodes which elements of a to keep.
 *          If the i-th bit is set, then the i-th element of a is kept. If 0,
 *          then the i-th element is replaced by that of b. Only the lower
 *          128 bits need to valid. The upper 256 bits can be garbage.
 * return: void */
void
grp128_gf16_mixi_avx2(Grp128GF16* restrict a, const Grp128GF16* restrict b,
                      const __m256i m) {
#if defined(__AVX512F__)
    // [mask, mask, U, U]
    __m512i vm = _mm512_castsi256_si512(m);
    vm = _mm512_shuffle_i64x2(vm, vm, 0x0); // [mask, mask, mask, mask]
    __m512i va = _mm512_load_si512(a->b);
    __m512i vb = _mm512_load_si512(b->b);
    va = _mm512_and_si512(va, vm);
    vb = _mm512_andnot_si512(vm, vb);
    _mm512_store_si512(a->b, _mm512_xor_si512(va, vb));
#else
    __m256i* sa = (__m256i*) a->b;
    __m256i* sb = (__m256i*) b->b;
    __m256i va0 = _mm256_load_si256(sa);
    __m256i va1 = _mm256_load_si256(sa + 1);
    __m256i vb0 = _mm256_load_si256(sb);
    __m256i vb1 = _mm256_load_si256(sb + 1);
    va0 = _mm256_and_si256(va0, m);
    va1 = _mm256_and_si256(va1, m);
    vb0 = _mm256_andnot_si256(m, vb0);
    vb1 = _mm256_andnot_si256(m, vb1);
    _mm256_store_si256(sa, _mm256_xor_si256(va0, vb0));
    _mm256_store_si256(sa + 1, _mm256_xor_si256(va1, vb1));
#endif
}

#endif

#if defined(__AVX512F__)

static force_inline __m512i
grp128_gf16_add_reg_avx512(const Grp128GF16* restrict a,
                           const Grp128GF16* restrict b) {
    __m512i va = _mm512_load_si512(a);
    __m512i vb = _mm512_load_si512(b);
    return _mm512_xor_si512(va, vb);
}

#elif defined(__AVX2__)

static force_inline void
grp128_gf16_add_reg_avx2(__m256i* restrict v0, __m256i* restrict v1,
                         const Grp128GF16* restrict a,
                         const Grp128GF16* restrict b) {
    __m256i* s0 = (__m256i*) a->b;
    __m256i* s1 = (__m256i*) b->b;
    __m256i va0 = _mm256_load_si256(s0);
    __m256i va1 = _mm256_load_si256(s0 + 1);
    __m256i vb0 = _mm256_load_si256(s1);
    __m256i vb1 = _mm256_load_si256(s1 + 1);
    *v0 = _mm256_xor_si256(va0, vb0);
    *v1 = _mm256_xor_si256(va1, vb1);
}

#endif

/* usage: given 2 struct Grp128GF16 a and b, compute a + b and store the result
 *      back into a.
 * params:
 *      1) a: ptr to struct Grp128GF16; point to a
 *      2) b: ptr to struct Grp128GF16. point to b
 * return: void */
void
grp128_gf16_addi(Grp128GF16* restrict a, const Grp128GF16* restrict b) {
#if defined(__AVX512F__)
    __m512i res = grp128_gf16_add_reg_avx512(a, b);
    _mm512_store_si512(a, res);
#elif defined(__AVX2__)
    __m256i v0, v1; grp128_gf16_add_reg_avx2(&v0, &v1, a, b);
    __m256i* s0 = (__m256i*) a->b;
    _mm256_store_si256(s0, v0);
    _mm256_store_si256(s0 + 1, v1);
#else
    uint128_t_xori(a->b, b->b);
    uint128_t_xori(a->b + 1, b->b + 1);
    uint128_t_xori(a->b + 2, b->b + 2);
    uint128_t_xori(a->b + 3, b->b + 3);
#endif
}

/* usage: given 2 struct Grp128GF16 a and b, compute a - b and store the result
 *      back into a.
 * params:
 *      1) a: ptr to struct Grp128GF16; point to a
 *      2) b: ptr to struct Grp128GF16. point to b
 * return: void */
void
grp128_gf16_subi(Grp128GF16* restrict a, const Grp128GF16* restrict b) {
    grp128_gf16_addi(a, b);
}

#if defined(__AVX512F__)

static force_inline __m512i
grp128_gf16_scalar_reg_avx512(const __m512i v, __mmask8 m0, __mmask8 m1,
                              __mmask8 m2, __mmask8 m3) {
    __m512i zv = _mm512_setzero_si512();
    __m512i s0 = _mm512_mask_blend_epi64(m0, zv, v);
    __m512i s1 = _mm512_mask_blend_epi64(m1, zv, v);
    __m512i s2 = _mm512_mask_blend_epi64(m2, zv, v);
    __m512i s3 = _mm512_mask_blend_epi64(m3, zv, v);
    s1 = _mm512_shuffle_i64x2(s1, s1, 0x93); // 0b10010011
    s2 = _mm512_shuffle_i64x2(s2, s2, 0x4E); // 0b01001110
    s3 = _mm512_shuffle_i64x2(s3, s3, 0x39); // 0b00111001
    __m512i s4 = _mm512_mask_xor_epi64(s3, 0xF, s3, s2); // 0b00001111
    s0 = _mm512_xor_si512(s0, s1);
    s2 = _mm512_xor_si512(s2, s3);
    s0 = _mm512_xor_si512(s0, s2);
    s4 = _mm512_mask_xor_epi64(s4, 0x3, s4, s1); // 0b00000011
    s4 = _mm512_shuffle_i64x2(s4, s4, 0x93); // 0b10010011
    s0 = _mm512_mask_xor_epi64(s0, 0xFC, s0, s4); // 0b11111100
    return s0;
}

static force_inline __m512i
grp128_gf16_mul_scalar_const_avx512(const Grp128GF16* src, gf16_t c) {
    uint8_t m0 = uint8_extend_from_lsb(c & 0x1U); // LSB
    uint8_t m1 = uint8_extend_from_lsb((c >> 1) & 0x1U); // 2nd LSB
    uint8_t m2 = uint8_extend_from_lsb((c >> 2) & 0x1U); // 3rd LSB
    uint8_t m3 = uint8_extend_from_lsb(c >> 3); // 4th LSB
    __m512i v = _mm512_load_si512(src->b);
    return grp128_gf16_scalar_reg_avx512(v, m0, m1, m2, m3);
}

__m512i
grp128_gf16_mul_scalar_bs_avx512(const __m512i v, const Grp128GF16* g,
                                 uint32_t i) {
    // NOTE: slower
//    __m512i vg = _mm512_load_si512(g);
//    __m512i lsb_ext = _mm512_set1_epi64(0x1);
//    vg = _mm512_srli_epi64(vg, i & 0x3F); // i % 64
//    vg = _mm512_and_si512(vg, lsb_ext);
//    __mmask8 res = _mm512_cmpeq_epi64_mask(lsb_ext, vg);
//    res >>= i >> 6; // i / 64
//    m0 = uint8_extend_from_lsb(res & 0x1);
//    m1 = uint8_extend_from_lsb( (res >> 2) & 0x1);
//    m2 = uint8_extend_from_lsb( (res >> 4) & 0x1);
//    m3 = uint8_extend_from_lsb( (res >> 6) & 0x1);
    uint8_t m0 = uint8_extend_from_lsb(uint128_t_at(g->b + 0, i));
    uint8_t m1 = uint8_extend_from_lsb(uint128_t_at(g->b + 1, i));
    uint8_t m2 = uint8_extend_from_lsb(uint128_t_at(g->b + 2, i));
    uint8_t m3 = uint8_extend_from_lsb(uint128_t_at(g->b + 3, i));
    return grp128_gf16_scalar_reg_avx512(v, m0, m1, m2, m3);
}

#elif defined(__AVX2__)

static force_inline __m256i
grp128_gf16_mul_scalar_reg_avx2(__m256i* restrict v1,
                                const __m256i s01, const __m256i s23,
                                const __m256i m0, const __m256i m1,
                                const __m256i m2, const __m256i m3) {
    // LSB
    __m256i b01 = _mm256_and_si256(s01, m0);
    __m256i b23 = _mm256_and_si256(s23, m0);
    // 2nd LSB
    __m256i b12 = _mm256_and_si256(s01, m1);
    __m256i b34 = _mm256_and_si256(s23, m1);
    // 3rd LSB
    b23 = _mm256_xor_si256(b23, _mm256_and_si256(s01, m2));
    __m256i b45 = _mm256_and_si256(s23, m2);
    // 4th LSB
    b34 = _mm256_xor_si256(b34, _mm256_and_si256(s01, m3));
    __m256i b56 = _mm256_and_si256(s23, m3);

    b01 = _mm256_xor_si256(b01, b45);
    b23 = _mm256_xor_si256(b23, b56);
    __m256i bz3 = _mm256_permute2x128_si256(b34, b34, 0x8);//0b00001000,[0,b3]
    __m256i b4z = _mm256_permute2x128_si256(b34, b34, 0x81);//0b10000001,[b4,0]
    b01 = _mm256_xor_si256(b01, b4z);
    b23 = _mm256_xor_si256(b23, bz3);

    b12 = _mm256_xor_si256(b12, b56);
    b12 = _mm256_xor_si256(b12, b45);
    b12 = _mm256_xor_si256(b12, b4z);

    __m256i bz1 = _mm256_permute2x128_si256(b12, b12, 0x8); // [0, b1]
    __m256i b2z = _mm256_permute2x128_si256(b12, b12, 0x81); // [b2, 0]
    b01 = _mm256_xor_si256(b01, bz1);
    b23 = _mm256_xor_si256(b23, b2z);

    *v1 = b23;
    return b01;
}

static force_inline __m256i
grp128_gf16_mul_scalar_const_avx2(__m256i* restrict v1,
                                  const Grp128GF16* src, gf16_t c) {
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
    __m256i* s = (__m256i*) src->b;
    __m256i s01 = _mm256_load_si256(s);
    __m256i s23 = _mm256_load_si256(s + 1);
    return grp128_gf16_mul_scalar_reg_avx2(v1, s01, s23, m0, m1, m2, m3);
}

__m256i
grp128_gf16_mul_scalar_bs_avx2(__m256i* restrict v1,
                               const __m256i s01, const __m256i s23,
                               const Grp128GF16* g, uint32_t i) {
    uint64_t b0 = uint64_extend_from_lsb(uint128_t_at(g->b + 0, i));
    uint64_t b1 = uint64_extend_from_lsb(uint128_t_at(g->b + 1, i));
    uint64_t b2 = uint64_extend_from_lsb(uint128_t_at(g->b + 2, i));
    uint64_t b3 = uint64_extend_from_lsb(uint128_t_at(g->b + 3, i));
    __m256i m0 = _mm256_set1_epi64x(b0);
    __m256i m1 = _mm256_set1_epi64x(b1);
    __m256i m2 = _mm256_set1_epi64x(b2);
    __m256i m3 = _mm256_set1_epi64x(b3);
    return grp128_gf16_mul_scalar_reg_avx2(v1, s01, s23, m0, m1, m2, m3);
}

#endif

static force_inline void
grp128_gf16_mul_scalar_reg(uint128_t out[4], const Grp128GF16* restrict src,
                           const uint128_t* restrict m0,
                           const uint128_t* restrict m1,
                           const uint128_t* restrict m2,
                           const uint128_t* restrict m3) {
    uint128_t b0, b1, b2, b3, b4, b5, b6;
    // LSB
    uint128_t_and(&b0, src->b, m0);
    uint128_t_and(&b1, src->b + 1, m0);
    uint128_t_and(&b2, src->b + 2, m0);
    uint128_t_and(&b3, src->b + 3, m0);
    // 2nd LSB
    uint128_t_xori_and(&b1, src->b, m1);
    uint128_t_xori_and(&b2, src->b + 1, m1);
    uint128_t_xori_and(&b3, src->b + 2, m1);
    uint128_t_and(&b4, src->b + 3, m1);
    // 3rd LSB
    uint128_t_xori_and(&b2, src->b, m2);
    uint128_t_xori_and(&b3, src->b + 1, m2);
    uint128_t_xori_and(&b4, src->b + 2, m2);
    uint128_t_and(&b5, src->b + 3, m2);
    // 4th LSB
    uint128_t_xori_and(&b3, src->b, m3);
    uint128_t_xori_and(&b4, src->b + 1, m3);
    uint128_t_xori_and(&b5, src->b + 2, m3);
    uint128_t_and(&b6, src->b + 3, m3);

    // reduction with irreducible polynomial x^4 + x + 1 (0b10011)
    // 7-th bit
    uint128_t_xori(&b3, &b6);
    uint128_t_xori(&b2, &b6);
    // 6-th bit
    uint128_t_xori(&b2, &b5);
    uint128_t_xori(&b1, &b5);
    // 5-th bit
    uint128_t_xori(&b1, &b4);
    uint128_t_xori(&b0, &b4);

    out[0] = b0;
    out[1] = b1;
    out[2] = b2;
    out[3] = b3;
}

static force_inline void
grp128_gf16_mul_scalar_const(uint128_t out[4], const Grp128GF16* src, gf16_t c) {
    uint64_t mask0 = uint64_extend_from_lsb(c & 0x1ULL); // LSB
    uint64_t mask1 = uint64_extend_from_lsb((c >> 1) & 0x1ULL); // 2nd LSB
    uint64_t mask2 = uint64_extend_from_lsb((c >> 2) & 0x1ULL); // 3rd LSB
    uint64_t mask3 = uint64_extend_from_lsb(c >> 3); // 4th LSB
    uint128_t m0, m1, m2, m3;
    uint128_t_set1_64b(&m0, mask0);
    uint128_t_set1_64b(&m1, mask1);
    uint128_t_set1_64b(&m2, mask2);
    uint128_t_set1_64b(&m3, mask3);
    grp128_gf16_mul_scalar_reg(out, src, &m0, &m1, &m2, &m3);
}

static force_inline void
grp128_gf16_mul_scalar_bs(uint128_t out[4], const Grp128GF16* src,
                          const Grp128GF16* g, uint32_t i) {
    uint64_t b0 = uint64_extend_from_lsb(uint128_t_at(g->b + 0, i));
    uint64_t b1 = uint64_extend_from_lsb(uint128_t_at(g->b + 1, i));
    uint64_t b2 = uint64_extend_from_lsb(uint128_t_at(g->b + 2, i));
    uint64_t b3 = uint64_extend_from_lsb(uint128_t_at(g->b + 3, i));
    uint128_t m0, m1, m2, m3;
    uint128_t_set1_64b(&m0, b0);
    uint128_t_set1_64b(&m1, b1);
    uint128_t_set1_64b(&m2, b2);
    uint128_t_set1_64b(&m3, b3);
    grp128_gf16_mul_scalar_reg(out, src, &m0, &m1, &m2, &m3);
}

/* usage: given a struct Grp128GF16 and a gf16_t scalar c, multiply 128 elements
 *      in the struct with c and store the result into a given container
 * params:
 *      1) dst: ptr to struct Grp128GF16; storage for the result
 *      2) src: ptr to struct Grp128GF16. the source
 *      3) c: the scalar multiplier
 * return: void */
void
grp128_gf16_mul_scalar(Grp128GF16* restrict dst, const Grp128GF16* restrict src,
                       gf16_t c) {
#if defined(__AVX512F__)
    __m512i res = grp128_gf16_mul_scalar_const_avx512(src, c);
    _mm512_store_si512(dst->b, res);
#elif defined(__AVX2__)
    __m256i v1;
    __m256i v0 = grp128_gf16_mul_scalar_const_avx2(&v1, src, c);
    __m256i* out = (__m256i*) dst->b;
    _mm256_store_si256(out, v0);
    _mm256_store_si256(out + 1, v1);
#else
    grp128_gf16_mul_scalar_const(dst->b, src, c);
#endif
}

/* usage: given a struct Grp128GF16 and a gf16_t scalar c, multiply 128
 *      elements in the struct with c and store the result back into the struct
 *      Grp128GF16
 * params:
 *      1) src: ptr to struct Grp128GF16. the source
 *      2) c: the scalar multiplier
 * return: void */
void
grp128_gf16_muli_scalar(Grp128GF16* restrict src, gf16_t c) {
#if defined(__AVX512F__)
    __m512i res = grp128_gf16_mul_scalar_const_avx512(src, c);
    _mm512_store_si512(src->b, res);
#elif defined(__AVX2__)
    __m256i v1;
    __m256i v0 = grp128_gf16_mul_scalar_const_avx2(&v1, src, c);
    __m256i* out = (__m256i*) src->b;
    _mm256_store_si256(out, v0);
    _mm256_store_si256(out + 1, v1);
#else
    grp128_gf16_mul_scalar_const(src->b, src, c);
#endif
}

/* usage: given 2 struct Grp128GF16 a and b, and a gf16_t scalar c, compute
 *      a + b * c, and store the result back into a.
 * params:
 *      1) a: ptr to struct Grp128GF16; point to a
 *      2) b: ptr to struct Grp128GF16. point to b
 *      3) c: the scalar multiplier
 * return: void */
void
grp128_gf16_fmaddi_scalar(Grp128GF16* restrict a, const Grp128GF16* restrict b,
                          gf16_t c) {
#if defined(__AVX512F__)
    __m512i res = grp128_gf16_mul_scalar_const_avx512(b, c);
    __m512i va = _mm512_load_si512(a->b);
    _mm512_store_si512(a->b, _mm512_xor_si512(va, res));
#elif defined(__AVX2__)
    __m256i v1;
    __m256i v0 = grp128_gf16_mul_scalar_const_avx2(&v1, b, c);
    __m256i* s0 = (__m256i*) a->b;
    __m256i va0 = _mm256_load_si256(s0);
    __m256i va1 = _mm256_load_si256(s0 + 1);
    _mm256_store_si256(s0, _mm256_xor_si256(va0, v0));
    _mm256_store_si256(s0 + 1, _mm256_xor_si256(va1, v1));
#else
    Grp128GF16 tmp;
    grp128_gf16_mul_scalar_const(tmp.b, b, c);
    grp128_gf16_addi(a, &tmp);
#endif
}

/* usage: given 3 struct Grp128GF16 a, b, g, and index i, extract the i-th
 *      gf16_t element c from g, compute a + b * c, and store the result back
 *      into a.
 * params:
 *      1) a: ptr to struct Grp128GF16; point to a
 *      2) b: ptr to struct Grp128GF16. point to b
 *      3) g: ptr to struct Grp128GF16. point to g
 *      4) i: the index
 * return: void */
void
grp128_gf16_fmaddi_scalar_bs(Grp128GF16* restrict a,
                             const Grp128GF16* restrict b,
                             const Grp128GF16* restrict g, uint32_t i) {
#if defined(__AVX512F__)
    __m512i v = _mm512_load_si512(b->b);
    __m512i res = grp128_gf16_mul_scalar_bs_avx512(v, g, i);
    __m512i va = _mm512_load_si512(a->b);
    _mm512_store_si512(a->b, _mm512_xor_si512(va, res));
#elif defined(__AVX2__)
    __m256i* s = (__m256i*) b->b;
    __m256i s01 = _mm256_load_si256(s);
    __m256i s23 = _mm256_load_si256(s + 1);
    __m256i v1;
    __m256i v0 = grp128_gf16_mul_scalar_bs_avx2(&v1, s01, s23, g, i);
    __m256i* s0 = (__m256i*) a->b;
    __m256i va0 = _mm256_load_si256(s0);
    __m256i va1 = _mm256_load_si256(s0 + 1);
    _mm256_store_si256(s0, _mm256_xor_si256(va0, v0));
    _mm256_store_si256(s0 + 1, _mm256_xor_si256(va1, v1));
#else
    Grp128GF16 tmp;
    grp128_gf16_mul_scalar_bs(tmp.b, b, g, i);
    grp128_gf16_addi(a, &tmp);
#endif
}

/* usage: given 2 struct Grp128GF16 a and b, and a gf16_t scalar c, compute
 *      a - b * c, and store the result back into a.
 * params:
 *      1) a: ptr to struct Grp128GF16; point to a
 *      2) b: ptr to struct Grp128GF16. point to b
 *      3) c: the scalar multiplier */
void
grp128_gf16_fmsubi_scalar(Grp128GF16* restrict a, const Grp128GF16* restrict b,
                          gf16_t c) {
    grp128_gf16_fmaddi_scalar(a, b, c);
}

/* usage: given 3 struct Grp128GF16 a, b, g, and index i, extract the i-th
 *      gf16_t element c from g, compute a + b * c, and store the result back
 *      into a.
 * params:
 *      1) a: ptr to struct Grp128GF16; point to a
 *      2) b: ptr to struct Grp128GF16. point to b
 *      3) g: ptr to struct Grp128GF16. point to g
 *      4) i: the index
 * return: void */
void
grp128_gf16_fmsubi_scalar_bs(Grp128GF16* restrict a,
                             const Grp128GF16* restrict b,
                             const Grp128GF16* restrict g, uint32_t i) {
    grp128_gf16_fmaddi_scalar_bs(a, b, g, i);
}

/* usage: given 2 struct Grp128GF16 a and b, a gf16_t scalar c, and a uint128_t
 *      d that encodes which elements to mask, compute b * c. and add the
 *      result to a based on the mask d. If the i-th bit of d is 1, then
 *      the i-th element of b*c is added to a. If 0, then the i-th element
 *      of a is untouched.
 * params:
 *      1) a: ptr to struct Grp128GF16; point to a
 *      2) b: ptr to struct Grp128GF16. point to b
 *      3) c: the scalar multiplier
 *      4) d: ptr to uint128_t. the mask
 * return: void */
void
grp128_gf16_fmaddi_scalar_mask(Grp128GF16* restrict a,
                               const Grp128GF16* restrict b, gf16_t c,
                               const uint128_t* restrict d) {
#if defined(__AVX512F__)
    __m512i res = grp128_gf16_mul_scalar_const_avx512(b, c);
    __m512i vm = _mm512_castsi128_si512(_mm_load_si128((__m128i*)d));
    vm = _mm512_shuffle_i64x2(vm, vm, 0x0); // [mask, mask, mask, mask]
    res = _mm512_and_si512(res, vm);
    __m512i va = _mm512_load_si512(a->b);
    _mm512_store_si512(a->b, _mm512_xor_si512(va, res));
#elif defined(__AVX2__)
    __m256i v1;
    __m256i v0 = grp128_gf16_mul_scalar_const_avx2(&v1, b, c);
    __m256i vm = _mm256_castsi128_si256(_mm_load_si128((__m128i*)d));
    vm = _mm256_permute2x128_si256(vm, vm, 0x0); // [mask, mask]
    v0 = _mm256_and_si256(v0, vm);
    v1 = _mm256_and_si256(v1, vm);
    __m256i* s0 = (__m256i*) a->b;
    __m256i va0 = _mm256_load_si256(s0);
    __m256i va1 = _mm256_load_si256(s0 + 1);
    _mm256_store_si256(s0, _mm256_xor_si256(va0, v0));
    _mm256_store_si256(s0 + 1, _mm256_xor_si256(va1, v1));
#else
    Grp128GF16 tmp;
    grp128_gf16_mul_scalar_const(tmp.b, b, c);
    grp128_gf16_zero_subset(&tmp, d);
    grp128_gf16_addi(a, &tmp);
#endif
}

/* usage: given 3 Grp128GF16 a, b, g, an index i, and a uint128_t d that encodes
 *      which elements to mask, extract the i-th gf16_t elements c from g, then
 *      compute b * c. The result is added back to a based on the mask d. If
 *      the j-th bit of d is 1, then the j-th element of b * c is added to a.
 *      If 0, then the j-th element of a is untouched.
 * params:
 *      1) a: ptr to struct Grp128GF16; point to a
 *      2) b: ptr to struct Grp128GF16. point to b
 *      3) g: ptr to struct Grp128GF16. point to g
 *      4) i: the index of which element in g to use as multiplier
 *      5) d: the mask
 * return: void */
void
grp128_gf16_fmaddi_scalar_mask_bs(Grp128GF16* restrict a,
                                  const Grp128GF16* restrict b,
                                  const Grp128GF16* restrict g, uint32_t i,
                                  const uint128_t* restrict d) {
#if defined(__AVX512F__)
    __m512i v = _mm512_load_si512(b->b);
    __m512i res = grp128_gf16_mul_scalar_bs_avx512(v, g, i);
    __m512i vm = _mm512_castsi128_si512(_mm_load_si128((__m128i*)d));
    vm = _mm512_shuffle_i64x2(vm, vm, 0x0); // [mask, mask, mask, mask]
    res = _mm512_and_si512(res, vm);
    __m512i va = _mm512_load_si512(a->b);
    _mm512_store_si512(a->b, _mm512_xor_si512(va, res));
#elif defined(__AVX2__)
    __m256i* s = (__m256i*) b->b;
    __m256i s01 = _mm256_load_si256(s);
    __m256i s23 = _mm256_load_si256(s + 1);
    __m256i v1;
    __m256i v0 = grp128_gf16_mul_scalar_bs_avx2(&v1, s01, s23, g, i);
    __m256i vm = _mm256_castsi128_si256(_mm_load_si128((__m128i*)d));
    vm = _mm256_permute2x128_si256(vm, vm, 0x0); // [mask, mask]
    v0 = _mm256_and_si256(v0, vm);
    v1 = _mm256_and_si256(v1, vm);
    __m256i* s0 = (__m256i*) a->b;
    __m256i va0 = _mm256_load_si256(s0);
    __m256i va1 = _mm256_load_si256(s0 + 1);
    _mm256_store_si256(s0, _mm256_xor_si256(va0, v0));
    _mm256_store_si256(s0 + 1, _mm256_xor_si256(va1, v1));
#else
    Grp128GF16 tmp;
    grp128_gf16_mul_scalar_bs(tmp.b, b, g, i);
    grp128_gf16_zero_subset(&tmp, d);
    grp128_gf16_addi(a, &tmp);
#endif
}
