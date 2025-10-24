#ifndef __GRP64_GF16_H__
#define __GRP64_GF16_H__

#include "gf16.h"
#include <stdint.h>
#include <stdalign.h>

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

// 64 elements in GF16 grouped together

typedef struct Grp64GF16 Grp64GF16;

/* ========================================================================
 * struct Grp64GF16 definition
 * ======================================================================== */

struct Grp64GF16 {
    // Each element in GF(16) requires 4 bits. The 64 elements are stored in
    // bitsliced format. I.e. the first bits of all 64 elements are stored
    // together as 1 uint64_t b[0]. The second bits of all elements are stored
    // in b[1], and so on.
    alignas(32) uint64_t b[4];
};

/* ========================================================================
 * function prototypes
 * ======================================================================== */

/* usage: given a struct Grp64GF16, set all of its elements to zero
 * params:
 *      1) g: ptr to struct Grp64GF16
 * return: void */
void
grp64_gf16_zero(Grp64GF16* g);

/* usage: given a struct Grp64GF16, find its zero elements
 * params:
 *      1) g: ptr to struct Grp64GF16
 * return: a uint64_t that encoding the location of zero elements. If the i-th
 *      element is zero, then the i-th bit is set, and so on */
uint64_t
grp64_gf16_zpos(const Grp64GF16* g);

/* usage: given a struct Grp64GF16, find its non-zero elements
 * params:
 *      1) g: ptr to struct Grp64GF16
 * return: a uint64_t that encoding the location of non-zero elements. If the
 *      i-th element is non-zero, then the i-th bit is set, and so on */
uint64_t
grp64_gf16_nzpos(const Grp64GF16* g);

/* usage: create a copy of a struct Grp64GF16
 * params:
 *      1) dst: ptr to struct Grp64GF16 for holding the copy
 *      2) src: ptr to struct Grp64GF16. The original
 * return: void */
void
grp64_gf16_copy(Grp64GF16* restrict dst, const Grp64GF16* restrict src);

/* usage: given a struct Grp64GF16, randomize its elements
 * params:
 *      1) g: ptr to struct Grp64GF16
 * return: void */
void
grp64_gf16_rand(Grp64GF16* g);

/* usage: given a struct Grp64GF16, set a subset of its elements to zero
 * params:
 *      1) g: ptr to struct Grp64GF16
 *      2) mask: a 64-bit integer that encodes which elements to zero.
 *          If the i-th bit is 0, then the i-th element will be zeroed.
 *          If 1, then the i-th element is kept.
 * return: void */
void
grp64_gf16_zero_subset(Grp64GF16* g, uint64_t mask);

/* usage: given a struct Grp64GF16, set its i-th element to zero
 * params:
 *      1) g: ptr to struct Grp64GF16
 *      2) i: index of the element. 0 ~ 63
 * return: void */
void
grp64_gf16_zero_at(Grp64GF16* g, uint32_t i);

/* usage: given a struct Grp64GF16, return its i-th element
 * params:
 *      1) g: ptr to struct Grp64GF16
 *      2) i: index of the element. 0 ~ 63
 * return: the i-th element as a gf16_t */
gf16_t
grp64_gf16_at(const Grp64GF16* g, uint32_t i);

/* usage: given a struct Grp64GF16, add a value to its i-th element
 * params:
 *      1) g: ptr to struct Grp64GF16
 *      2) i: index of the element. 0 ~ 63
 *      3) v: the value to add
 * return:void */
void
grp64_gf16_add_at(Grp64GF16* g, uint32_t i, gf16_t v);

/* usage: given a struct Grp64GF16, set its i-th element
 * params:
 *      1) g: ptr to struct Grp64GF16
 *      2) i: index of the element. 0 ~ 63
 *      3) v: value of the i-th element
 * return: void */
void
grp64_gf16_set_at(Grp64GF16* g, uint32_t i, gf16_t v);

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
                uint64_t mask);

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
                   uint64_t mask);

/* usuage: given a struct Grp64GF16, return the index of the 1st non-zero
 *      element.
 * params:
 *      1) g: ptr to struct Grp64GF16
 * return: index of the 1st non-zero element. UINT32_MAX if all elements are
 *      zero */
uint32_t
grp64_gf16_1st_nz_idx(const Grp64GF16* g);

/* usage: given 2 struct Grp64GF16 a and b, compute a + b and store the result
 *      back into a.
 * params:
 *      1) a: ptr to struct Grp64GF16; point to a
 *      2) b: ptr to struct Grp64GF16. point to b
 * return: void */
void
grp64_gf16_addi(Grp64GF16* restrict a, const Grp64GF16* restrict b);

/* usage: given 2 size-2 arrays of struct Grp64GF16 a and b, compute a + b and
 *      store the result back into a.
 * params:
 *      1) a: ptr to struct Grp64GF16; point to a
 *      2) b: ptr to struct Grp64GF16. point to b
 * return: void */
void
grp64_gf16_addi_x2(Grp64GF16* restrict a, const Grp64GF16* restrict b);

/* usage: given 2 struct Grp64GF16 a and b, compute a - b and store the result
 *      back into a.
 * params:
 *      1) a: ptr to struct Grp64GF16; point to a
 *      2) b: ptr to struct Grp64GF16. point to b
 * return: void */
void
grp64_gf16_subi(Grp64GF16* restrict a, const Grp64GF16* restrict b);

/* usage: given a struct Grp64GF16 and a gf16_t scalar c, multiply 64 elements
 *      in the struct with c and store the result into a given container
 * params:
 *      1) dst: ptr to struct Grp64GF16; storage for the result
 *      2) src: ptr to struct Grp64GF16. the source
 *      3) c: the scalar multiplier
 * return: void */
void
grp64_gf16_mul_scalar(Grp64GF16* restrict dst, const Grp64GF16* restrict src,
                      gf16_t c);

/* usage: given a struct Grp64GF16 and a gf16_t scalar c, multiply 64 elements
 *      in the struct with c and store the result back into the struct Grp64GF16
 * params:
 *      1) src: ptr to struct Grp64GF16. the source
 *      2) c: the scalar multiplier
 * return: void */
void
grp64_gf16_muli_scalar(Grp64GF16* restrict src, gf16_t c);

#if defined(__AVX512F__)

void
grp64_gf16_mul_scalar_from_bs_adj_avx512(__m256i* restrict v0,
                                         __m256i* restrict v1,
                                         const Grp64GF16* restrict s,
                                         const Grp64GF16* restrict g,
                                         uint32_t i);
__m512i
grp64_gf16_mul_scalar_from_bs_adj_avx512_no_split(const Grp64GF16* restrict src,
                                                  const Grp64GF16* restrict g,
                                                  uint32_t i);
void
grp64_gf16_muli_scalar_from_bs_2x1_avx512(__m256i* restrict v0,
                                          __m256i* restrict v1,
                                          const Grp64GF16* restrict src0,
                                          const Grp64GF16* restrict src1,
                                          const Grp64GF16* restrict g,
                                          uint32_t i);
__m512i
grp64_gf16_mul_scalar_from_bs_1x2_avx512(const Grp64GF16* src,
                                         const Grp64GF16* g,
                                         uint32_t i);

#endif

#if defined(__AVX2__)

/* usage: given struct Grp64GF16 a, g, and an index i, extract the i-th element
 *      c in g as the multipler, then compute a * c. Note that both the value
 *      of a and the product are stored as a 256-bit YMM register.
 * params:
 *      1) v: a 256-bit YMM register which stores the struct Grp64GF16 a
 *      2) g: ptr to struct Grp64GF16. point to g
 *      3) i: the index
 * return: the product as a 256-bit YMM register */
__m256i
grp64_gf16_mul_scalar_from_bs_avx2(const __m256i v, const Grp64GF16* g, uint32_t i);

#endif

/* usage: given 2 struct Grp64GF16 and a gf16_t scalar c, multiply 64 elements
 *      in both struct with c and store the result back respectively
 * params:
 *      1) s0: ptr to struct Grp64GF16. the 1st source
 *      2) s1: ptr to struct Grp64GF16. the 2nd source
 *      2) c: the scalar multiplier
 * return: void */
void
grp64_gf16_muli_scalar_2x1(Grp64GF16* restrict s0, Grp64GF16* restrict s1,
                           gf16_t c);

/* usage: given 2 struct Grp64GF16 a and b, and a gf16_t scalar c, compute
 *      a + b * c, and store the result back into a.
 * params:
 *      1) a: ptr to struct Grp64GF16; point to a
 *      2) b: ptr to struct Grp64GF16. point to b
 *      3) c: the scalar multiplier
 * return: void */
void
grp64_gf16_fmaddi_scalar(Grp64GF16* restrict a, const Grp64GF16* restrict b,
                         gf16_t c);

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
                            const Grp64GF16* restrict g, uint32_t i);

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
                             const Grp64GF16* restrict c, gf16_t m0, gf16_t m1);

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
                             const Grp64GF16* restrict c, gf16_t m0, gf16_t m1);

/* usage: given 2 struct Grp64GF16 a and b, and a gf16_t scalar c, compute
 *      a - b * c, and store the result back into a.
 * params:
 *      1) a: ptr to struct Grp64GF16; point to a
 *      2) b: ptr to struct Grp64GF16. point to b
 *      3) c: the scalar multiplier
 * return: void */
void
grp64_gf16_fmsubi_scalar(Grp64GF16* restrict a, const Grp64GF16* restrict b,
                         gf16_t c);

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
                            const Grp64GF16* restrict g, uint32_t i);

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
                              gf16_t c, uint64_t d);

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
                                 uint32_t i, uint64_t d);

#endif // __GRP64_GF16_H__
