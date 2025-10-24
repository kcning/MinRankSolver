#ifndef __GRP128_GF16_H__
#define __GRP128_GF16_H__

#include "gf16.h"
#include "uint128_t.h"
#include <stdalign.h>

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

// 128 elements in GF16 grouped together

typedef struct Grp128GF16 Grp128GF16;

/* ========================================================================
 * struct Grp128GF16 definition
 * ======================================================================== */

struct Grp128GF16 {
    // Each element in GF(16) requires 4 bits. The 128 elements are stored in
    // bitsliced format. I.e. the first bits of all 128 elements are stored
    // together as 1 uint128_t b[0]. The second bits of all elements are stored
    // in b[1], and so on.
    alignas(64) uint128_t b[4];
};

/* ========================================================================
 * function prototypes
 * ======================================================================== */

/* usage: given a struct Grp128GF16, set all of its elements to zero
 * params:
 *      1) g: ptr to struct Grp128GF16
 * return: void */
void
grp128_gf16_zero(Grp128GF16* g);

/* usage: given a struct Grp128GF16, find its non-zero elements
 * params:
 *      1) ptr to a uint128_t. Upon return, it encodes the location of non-zero
 *          elements. If the i-th element is non-zero, then the i-th bit is set
 *          and so on.
 *      2) g: ptr to struct Grp128GF16
 * return: void */
void
grp128_gf16_nzpos(uint128_t* restrict out, const Grp128GF16* restrict g);

/* usage: given a struct Grp128GF16, find its zero elements
 * params:
 *      1) ptr to a uint128_t. Upon return, it encodes the location of zero
 *          elements. If the i-th element is zero, then the i-th bit is set and
 *          so on.
 *      2) g: ptr to struct Grp128GF16
 * return: void */
void
grp128_gf16_zpos(uint128_t* restrict out, const Grp128GF16* restrict g);

/* usage: create a copy of a struct Grp128GF16
 * params:
 *      1) dst: ptr to struct Grp128GF16 for holding the copy
 *      2) src: ptr to struct Grp128GF16. The original
 * return: void */
void
grp128_gf16_copy(Grp128GF16* restrict dst, const Grp128GF16* restrict src);

/* usage: given a struct Grp128GF16, randomize its elements
 * params:
 *      1) g: ptr to struct Grp128GF16
 * return: void */
void
grp128_gf16_rand(Grp128GF16* g);

/* usage: given a struct Grp128GF16, set a subset of its elements to zero
 * params:
 *      1) g: ptr to struct Grp128GF16
 *      2) mask: ptr to a 128-bit integer that encodes which elements to zero.
 *          If the i-th bit is 0, then the i-th element will be zeroed.
 *          If 1, then the i-th element is kept.
 * return: void */
void
grp128_gf16_zero_subset(Grp128GF16* restrict g, const uint128_t* restrict mask);

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
grp128_gf16_zero_subset_avx2(Grp128GF16* restrict g, const __m256i m);

#endif

/* usage: given a struct Grp128GF16, set its i-th element to zero
 * params:
 *      1) g: ptr to struct Grp128GF16
 *      2) i: index of the element. 0 ~ 511
 * return: void */
void
grp128_gf16_zero_at(Grp128GF16* g, uint32_t i);

/* usage: given a struct Grp128GF16, return its i-th element
 * params:
 *      1) g: ptr to struct Grp128GF16
 *      2) i: index of the element. 0 ~ 511
 * return: the i-th element as a gf16_t */
gf16_t
grp128_gf16_at(const Grp128GF16* g, uint32_t i);

/* usage: given a struct Grp128GF16, add a value to its i-th element
 * params:
 *      1) g: ptr to struct Grp128GF16
 *      2) i: index of the element. 0 ~ 511
 *      3) v: the value to add
 * return: void */
void
grp128_gf16_add_at(Grp128GF16* g, uint32_t i, gf16_t v);

/* usage: given a struct Grp128GF16, set its i-th element
 * params:
 *      1) g: ptr to struct Grp128GF16
 *      2) i: index of the element. 0 ~ 63
 *      3) v: value of the i-th element
 * return: void */
void
grp128_gf16_set_at(Grp128GF16* g, uint32_t i, gf16_t v);

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
                 const uint128_t* restrict mask);

#if defined(__AVX2__)

/* usage: given a struct Grp128GF16 a and b, replace a subset of elements of
 *      a with the corresponding elements of b
 * params:
 *      1) a: ptr to struct Grp128GF16
 *      2) b: ptr to struct Grp128GF16
 *      3) m: a 256-bit register that encodes which elements to zero.
 *          if the i-th bit is 0, then the i-th element will be zeroed.
 *          if 1, then the i-th element is kept. The upper 128-bit
 *          needs to be duplicate of the lower 128 bits. I.e. m[255:128]
 *          = m[127:0]
 * return: void */
void
grp128_gf16_mixi_avx2(Grp128GF16* restrict a, const Grp128GF16* restrict b,
                      const __m256i m);
#endif

/* usage: given 2 struct Grp128GF16 a and b, compute a + b and store the result
 *      back into a.
 * params:
 *      1) a: ptr to struct Grp128GF16; point to a
 *      2) b: ptr to struct Grp128GF16. point to b
 * return: void */
void
grp128_gf16_addi(Grp128GF16* restrict a, const Grp128GF16* restrict b);

/* usage: given 2 struct Grp128GF16 a and b, compute a - b and store the result
 *      back into a.
 * params:
 *      1) a: ptr to struct Grp128GF16; point to a
 *      2) b: ptr to struct Grp128GF16. point to b
 * return: void */
void
grp128_gf16_subi(Grp128GF16* restrict a, const Grp128GF16* restrict b);

/* usage: given a struct Grp128GF16 and a gf16_t scalar c, multiply 128 elements
 *      in the struct with c and store the result into a given container
 * params:
 *      1) dst: ptr to struct Grp128GF16; storage for the result
 *      2) src: ptr to struct Grp128GF16. the source
 *      3) c: the scalar multiplier
 * return: void */
void
grp128_gf16_mul_scalar(Grp128GF16* restrict dst, const Grp128GF16* restrict src,
                       gf16_t c);

/* usage: given a struct Grp128GF16 and a gf16_t scalar c, multiply 128
 *      elements in the struct with c and store the result back into the struct
 *      Grp128GF16
 * params:
 *      1) src: ptr to struct Grp128GF16. the source
 *      2) c: the scalar multiplier
 * return: void */
void
grp128_gf16_muli_scalar(Grp128GF16* restrict src, gf16_t c);

/* usage: given 2 struct Grp128GF16 a and b, and a gf16_t scalar c, compute
 *      a + b * c, and store the result back into a.
 * params:
 *      1) a: ptr to struct Grp128GF16; point to a
 *      2) b: ptr to struct Grp128GF16. point to b
 *      3) c: the scalar multiplier
 * return: void */
void
grp128_gf16_fmaddi_scalar(Grp128GF16* restrict a, const Grp128GF16* restrict b,
                          gf16_t c);

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
                             const Grp128GF16* restrict g, uint32_t i);

#if defined(__AVX512F__)

__m512i
grp128_gf16_mul_scalar_bs_avx512(const __m512i v, const Grp128GF16* g,
                                 uint32_t i);

#elif defined(__AVX2__)

__m256i
grp128_gf16_mul_scalar_bs_avx2(__m256i* restrict v1,
                               const __m256i s01, const __m256i s23,
                               const Grp128GF16* g, uint32_t i);

#endif

/* usage: given 2 struct Grp128GF16 a and b, and a gf16_t scalar c, compute
 *      a - b * c, and store the result back into a.
 * params:
 *      1) a: ptr to struct Grp128GF16; point to a
 *      2) b: ptr to struct Grp128GF16. point to b
 *      3) c: the scalar multiplier */
void
grp128_gf16_fmsubi_scalar(Grp128GF16* restrict a, const Grp128GF16* restrict b,
                          gf16_t c);

/* usage: given 3 struct Grp128GF16 a, b, g, and index i, extract the i-th
 *      gf16_t element c from g, compute a - b * c, and store the result back
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
                             const Grp128GF16* restrict g, uint32_t i);

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
                               const uint128_t* restrict d);

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
                                  const uint128_t* restrict d);

#endif // __GRP128_GF16_H__
