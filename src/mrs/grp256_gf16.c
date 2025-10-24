#include "grp256_gf16.h"
#include "util.h"
#include <assert.h>

/* ========================================================================
 * function implementations
 * ======================================================================== */

/* usage: given a struct Grp256GF16, set all of its elements to zero
 * params:
 *      1) g: ptr to struct Grp256GF16
 * return: void */
void
grp256_gf16_zero(Grp256GF16* g) {
    memset(g->b, 0x0, sizeof(uint256_t) * 4);
}

/* usage: given a struct Grp256GF16, find its non-zero elements
 * params:
 *      1) ptr to a uint256_t. Upon return, it encodes the location of non-zero
 *          elements. If the i-th element is non-zero, then the i-th bit is set
 *          and so on.
 *      2) g: ptr to struct Grp256GF16
 * return: void */
void
grp256_gf16_nzpos(uint256_t* restrict out, const Grp256GF16* restrict g) {
    uint256_t t0; uint256_t_or(&t0, g->b, g->b + 1);
    uint256_t t1; uint256_t_or(&t1, g->b + 2, g->b + 3);
    uint256_t_or(out, &t0, &t1);
}

/* usage: given a struct Grp256GF16, find its zero elements
 * params:
 *      1) ptr to a uint256_t. Upon return, it encodes the location of zero
 *          elements. If the i-th element is zero, then the i-th bit is set and
 *          so on.
 *      2) g: ptr to struct Grp256GF16
 * return: void */
void
grp256_gf16_zpos(uint256_t* restrict out, const Grp256GF16* restrict g) {
    grp256_gf16_nzpos(out, g);
    uint256_t_negi(out);
}

/* usage: create a copy of a struct Grp256GF16
 * params:
 *      1) dst: ptr to struct Grp256GF16 for holding the copy
 *      2) src: ptr to struct Grp256GF16. The original
 * return: void */
void
grp256_gf16_copy(Grp256GF16* restrict dst, const Grp256GF16* restrict src) {
    memcpy(dst->b, src->b, sizeof(uint256_t) * 4);
}

/* usage: given a struct Grp256GF16, randomize its elements
 * params:
 *      1) g: ptr to struct Grp256GF16
 * return: void */
void
grp256_gf16_rand(Grp256GF16* g) {
    uint256_t_rand(g->b);
    uint256_t_rand(g->b + 1);
    uint256_t_rand(g->b + 2);
    uint256_t_rand(g->b + 3);
}

/* usage: given a struct Grp256GF16, set a subset of its elements to zero
 * params:
 *      1) g: ptr to struct Grp256GF16
 *      2) mask: ptr to a 256-bit integer that encodes which elements to zero.
 *          If the i-th bit is 0, then the i-th element will be zeroed.
 *          If 1, then the i-th element is kept.
 * return: void */
void
grp256_gf16_zero_subset(Grp256GF16* restrict g, const uint256_t* restrict mask) {
    uint256_t_andi(g->b, mask);
    uint256_t_andi(g->b + 1, mask);
    uint256_t_andi(g->b + 2, mask);
    uint256_t_andi(g->b + 3, mask);
}

/* usage: given a struct Grp256GF16, set its i-th element to zero
 * params:
 *      1) g: ptr to struct Grp256GF16
 *      2) i: index of the element. 0 ~ 511
 * return: void */
void
grp256_gf16_zero_at(Grp256GF16* g, uint32_t i) {
    assert(i < 256);
    uint256_t mask; uint256_t_max(&mask);
    uint256_t_toggle_at(&mask, i);
    grp256_gf16_zero_subset(g, &mask);
}

/* usage: given a struct Grp256GF16, return its i-th element
 * params:
 *      1) g: ptr to struct Grp256GF16
 *      2) i: index of the element. 0 ~ 511
 * return: the i-th element as a gf16_t */
gf16_t
grp256_gf16_at(const Grp256GF16* g, uint32_t i) {
    assert(i < 256);
    // TODO: try to load only 8 bits each instead of 64 bits
    uint64_t b0 = uint256_t_at(g->b, i);
    uint64_t b1 = uint256_t_at(g->b + 1, i);
    uint64_t b2 = uint256_t_at(g->b + 2, i);
    uint64_t b3 = uint256_t_at(g->b + 3, i);
    return b0 | (b1 << 1) | (b2 << 2) | (b3 << 3);
}

/* usage: given a struct Grp256GF16, add a value to its i-th element
 * params:
 *      1) g: ptr to struct Grp256GF16
 *      2) i: index of the element. 0 ~ 511
 *      3) v: the value to add
 * return: void */
void
grp256_gf16_add_at(Grp256GF16* g, uint32_t i, gf16_t v) {
    assert(i < 256);
    assert(v <= GF16_MAX);
    uint64_t b0 = v & 0x1ULL;
    uint64_t b1 = (v >> 1) & 0x1ULL;
    uint64_t b2 = (v >> 2) & 0x1ULL;
    uint64_t b3 = (v >> 3) & 0x1ULL;
    if(b0)
        uint256_t_toggle_at(g->b, i);
    if(b1)
        uint256_t_toggle_at(g->b + 1, i);
    if(b2)
        uint256_t_toggle_at(g->b + 2, i);
    if(b3)
        uint256_t_toggle_at(g->b + 3, i);
}

/* usage: given a struct Grp256GF16, set its i-th element
 * params:
 *      1) g: ptr to struct Grp256GF16
 *      2) i: index of the element. 0 ~ 63
 *      3) v: value of the i-th element
 * return: void */
void
grp256_gf16_set_at(Grp256GF16* g, uint32_t i, gf16_t v) {
    assert(i < 256);
    assert(v <= GF16_MAX);
    uint64_t b0 = v & 0x1ULL;
    uint64_t b1 = (v >> 1) & 0x1ULL;
    uint64_t b2 = (v >> 2) & 0x1ULL;
    uint64_t b3 = (v >> 3) & 0x1ULL;
    uint256_t_set_at(g->b, i, b0);
    uint256_t_set_at(g->b + 1, i, b1);
    uint256_t_set_at(g->b + 2, i, b2);
    uint256_t_set_at(g->b + 3, i, b3);
}

/* usage: given a struct Grp256GF16 a and b, replace a subset of elements of
 *      a with the corresponding elements of b
 * params:
 *      1) a: ptr to struct Grp256GF16
 *      2) b: ptr to struct Grp256GF16
 *      3) mask: ptr to a uint256_t which encodes which elements of a to keep.
 *          If the i-th bit is set, then the i-th element of a is kept. If 0,
 *          then the i-th element is replaced by that of b.
 * return: void */
void
grp256_gf16_mixi(Grp256GF16* restrict a, const Grp256GF16* restrict b,
                 const uint256_t* restrict mask) {
    uint256_t_mixi(a->b, b->b, mask);
    uint256_t_mixi(a->b + 1, b->b + 1, mask);
    uint256_t_mixi(a->b + 2, b->b + 2, mask);
    uint256_t_mixi(a->b + 3, b->b + 3, mask);
}

/* usage: given 2 struct Grp256GF16 a and b, compute a + b and store the result
 *      back into a.
 * params:
 *      1) a: ptr to struct Grp256GF16; point to a
 *      2) b: ptr to struct Grp256GF16. point to b
 * return: void */
void
grp256_gf16_addi(Grp256GF16* restrict a, const Grp256GF16* restrict b) {
    uint256_t_xori(a->b, b->b);
    uint256_t_xori(a->b + 1, b->b + 1);
    uint256_t_xori(a->b + 2, b->b + 2);
    uint256_t_xori(a->b + 3, b->b + 3);
}

/* usage: given 2 struct Grp256GF16 a and b, compute a - b and store the result
 *      back into a.
 * params:
 *      1) a: ptr to struct Grp256GF16; point to a
 *      2) b: ptr to struct Grp256GF16. point to b
 * return: void */
void
grp256_gf16_subi(Grp256GF16* restrict a, const Grp256GF16* restrict b) {
    grp256_gf16_addi(a, b);
}

#if defined(__AVX2__)

static force_inline void
grp256_gf16_mul_scalar_reg_avx2(uint256_t out[4], const Grp256GF16* src,
                                gf16_t c) {
    const __m256i* s = (__m256i*) (src->b);
    __m256i v0 = _mm256_load_si256(s + 0);
    __m256i v1 = _mm256_load_si256(s + 1);
    __m256i v2 = _mm256_load_si256(s + 2);
    __m256i v3 = _mm256_load_si256(s + 3);
    __m256i m0 = _mm256_set1_epi64x(uint64_extend_from_lsb(c & 0x1ULL));
    __m256i m1 = _mm256_set1_epi64x(uint64_extend_from_lsb((c & 0x2ULL) >> 1));
    __m256i m2 = _mm256_set1_epi64x(uint64_extend_from_lsb((c & 0x4ULL) >> 2));
    __m256i m3 = _mm256_set1_epi64x(uint64_extend_from_lsb((c & 0x8ULL) >> 3));
    // LSB
    __m256i b0 = _mm256_and_si256(v0, m0);
    __m256i b1 = _mm256_and_si256(v1, m0);
    __m256i b2 = _mm256_and_si256(v2, m0);
    __m256i b3 = _mm256_and_si256(v3, m0);
    // 2nd LSB
    b1 = _mm256_xor_si256(b1, _mm256_and_si256(v0, m1));
    b2 = _mm256_xor_si256(b2, _mm256_and_si256(v1, m1));
    b3 = _mm256_xor_si256(b3, _mm256_and_si256(v2, m1));
    __m256i b4 = _mm256_and_si256(v3, m1);
    // 3rd LSB
    b2 = _mm256_xor_si256(b2, _mm256_and_si256(v0, m2));
    b3 = _mm256_xor_si256(b3, _mm256_and_si256(v1, m2));
    b4 = _mm256_xor_si256(b4, _mm256_and_si256(v2, m2));
    __m256i b5 = _mm256_and_si256(v3, m2);
    // 4th LSB
    b3 = _mm256_xor_si256(b3, _mm256_and_si256(v0, m3));
    b4 = _mm256_xor_si256(b4, _mm256_and_si256(v1, m3));
    b5 = _mm256_xor_si256(b5, _mm256_and_si256(v2, m3));
    __m256i b6 = _mm256_and_si256(v3, m3);
    // reduction with irreducible polynomial x^4 + x + 1 (0b10011)
    // 7-th bit
    b3 = _mm256_xor_si256(b3, b6);
    b2 = _mm256_xor_si256(b2, b6);
    // 6-th bit
    b2 = _mm256_xor_si256(b2, b5);
    b1 = _mm256_xor_si256(b1, b5);
    // 5-th bit
    b1 = _mm256_xor_si256(b1, b4);
    b0 = _mm256_xor_si256(b0, b4);

    __m256i* dst = (__m256i*) out;
    _mm256_store_si256(dst + 0, b0);
    _mm256_store_si256(dst + 1, b1);
    _mm256_store_si256(dst + 2, b2);
    _mm256_store_si256(dst + 3, b3);
}

#endif

static force_inline void
grp256_gf16_mul_scalar_reg(uint256_t out[4], const Grp256GF16* src, gf16_t c) {
    uint64_t mask0 = uint64_extend_from_lsb(c & 0x1ULL); // LSB
    uint64_t mask1 = uint64_extend_from_lsb((c & 0x2ULL) >> 1); // 2nd LSB
    uint64_t mask2 = uint64_extend_from_lsb((c & 0x4ULL) >> 2); // 3rd LSB
    uint64_t mask3 = uint64_extend_from_lsb((c & 0x8ULL) >> 3); // 4th LSB
    uint256_t m0, m1, m2, m3;
    uint256_t_set1_64b(&m0, mask0);
    uint256_t_set1_64b(&m1, mask1);
    uint256_t_set1_64b(&m2, mask2);
    uint256_t_set1_64b(&m3, mask3);

    uint256_t b0, b1, b2, b3, b4, b5, b6;
    // LSB
    uint256_t_and(&b0, src->b, &m0);
    uint256_t_and(&b1, src->b + 1, &m0);
    uint256_t_and(&b2, src->b + 2, &m0);
    uint256_t_and(&b3, src->b + 3, &m0);
    // 2nd LSB
    uint256_t_xori_and(&b1, src->b, &m1);
    uint256_t_xori_and(&b2, src->b + 1, &m1);
    uint256_t_xori_and(&b3, src->b + 2, &m1);
    uint256_t_and(&b4, src->b + 3, &m1);
    // 3rd LSB
    uint256_t_xori_and(&b2, src->b, &m2);
    uint256_t_xori_and(&b3, src->b + 1, &m2);
    uint256_t_xori_and(&b4, src->b + 2, &m2);
    uint256_t_and(&b5, src->b + 3, &m2);
    // 4th LSB
    uint256_t_xori_and(&b3, src->b, &m3);
    uint256_t_xori_and(&b4, src->b + 1, &m3);
    uint256_t_xori_and(&b5, src->b + 2, &m3);
    uint256_t_and(&b6, src->b + 3, &m3);

    // reduction with irreducible polynomial x^4 + x + 1 (0b10011)
    // 7-th bit
    uint256_t_xori(&b3, &b6);
    uint256_t_xori(&b2, &b6);
    // 6-th bit
    uint256_t_xori(&b2, &b5);
    uint256_t_xori(&b1, &b5);
    // 5-th bit
    uint256_t_xori(&b1, &b4);
    uint256_t_xori(&b0, &b4);

    out[0] = b0;
    out[1] = b1;
    out[2] = b2;
    out[3] = b3;
}

/* usage: given a struct Grp256GF16 and a gf16_t scalar c, multiply 256 elements
 *      in the struct with c and store the result into a given container
 * params:
 *      1) dst: ptr to struct Grp256GF16; storage for the result
 *      2) src: ptr to struct Grp256GF16. the source
 *      3) c: the scalar multiplier
 * return: void */
void
grp256_gf16_mul_scalar(Grp256GF16* restrict dst, const Grp256GF16* restrict src,
                       gf16_t c) {
    assert(c <= GF16_MAX);
    if(c == 0) {
        grp256_gf16_zero(dst);
        return;
    }

    if(c == 1) {
        grp256_gf16_copy(dst, src);
        return;
    }

#if defined(__AVX2__)
    grp256_gf16_mul_scalar_reg_avx2(dst->b, src, c);
#else
    grp256_gf16_mul_scalar_reg(dst->b, src, c);
#endif
}

/* usage: given a struct Grp256GF16 and a gf16_t scalar c, multiply 256
 *      elements in the struct with c and store the result back into the struct
 *      Grp256GF16
 * params:
 *      1) src: ptr to struct Grp256GF16. the source
 *      2) c: the scalar multiplier
 * return: void */
void
grp256_gf16_muli_scalar(Grp256GF16* restrict src, gf16_t c) {
    if(c == 0) {
        grp256_gf16_zero(src);
        return;
    }

    if(c == 1)
        return; // do nothing
#if defined(__AVX2__)
    grp256_gf16_mul_scalar_reg_avx2(src->b, src, c);
#else
    grp256_gf16_mul_scalar_reg(src->b, src, c);
#endif
}

/* usage: given 2 struct Grp256GF16 a and b, and a gf16_t scalar c, compute
 *      a + b * c, and store the result back into a.
 * params:
 *      1) a: ptr to struct Grp256GF16; point to a
 *      2) b: ptr to struct Grp256GF16. point to b
 *      3) c: the scalar multiplier
 * return: void */
void
grp256_gf16_fmaddi_scalar(Grp256GF16* restrict a, const Grp256GF16* restrict b,
                          gf16_t c) {
    if(c == 0)
        return;
    if(c == 1) {
        grp256_gf16_addi(a, b);
        return;
    }

    Grp256GF16 tmp;
#if defined(__AVX2__)
    grp256_gf16_mul_scalar_reg_avx2(tmp.b, b, c);
#else
    grp256_gf16_mul_scalar_reg(tmp.b, b, c);
#endif
    grp256_gf16_addi(a, &tmp);
}

/* usage: given 2 struct Grp256GF16 a and b, and a gf16_t scalar c, compute
 *      a - b * c, and store the result back into a.
 * params:
 *      1) a: ptr to struct Grp256GF16; point to a
 *      2) b: ptr to struct Grp256GF16. point to b
 *      3) c: the scalar multiplier */
void
grp256_gf16_fmsubi_scalar(Grp256GF16* restrict a, const Grp256GF16* restrict b,
                          gf16_t c) {
    grp256_gf16_fmaddi_scalar(a, b, c);
}

/* usage: given 2 struct Grp256GF16 a and b, a gf16_t scalar c, and a uint256_t
 *      d that encodes which elements to mask, compute b * c. and add the
 *      result to a based on the mask d. If the i-th bit of d is 1, then
 *      the i-th element of b*c is added to a. If 0, then the i-th element
 *      of a is untouched.
 * params:
 *      1) a: ptr to struct Grp256GF16; point to a
 *      2) b: ptr to struct Grp256GF16. point to b
 *      3) c: the scalar multiplier
 *      4) d: ptr to uint256_t. the mask
 * return: void */
void
grp256_gf16_fmaddi_scalar_mask(Grp256GF16* restrict a,
                               const Grp256GF16* restrict b, gf16_t c,
                               const uint256_t* restrict d) {
    if(c == 0)
        return;
    Grp256GF16 tmp;
#if defined(__AVX2__)
    grp256_gf16_mul_scalar_reg_avx2(tmp.b, b, c);
#else
    grp256_gf16_mul_scalar_reg(tmp.b, b, c);
#endif
    grp256_gf16_zero_subset(&tmp, d);
    grp256_gf16_addi(a, &tmp);
}
