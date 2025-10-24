#ifndef __GRP512_GF16_H__
#define __GRP512_GF16_H__

#include "gf16.h"
#include "uint512_t.h"

// 512 elements in GF16 grouped together

typedef struct Grp512GF16 Grp512GF16;

/* ========================================================================
 * struct Grp512GF16 definition
 * ======================================================================== */

struct Grp512GF16 {
    // Each element in GF(16) requires 4 bits. The 512 elements are stored in
    // bitsliced format. I.e. the first bits of all 512 elements are stored
    // together as 1 uint512_t b[0]. The second bits of all elements are stored
    // in b[1], and so on.
    // NOTE: alignment follows that of uint512_t
    uint512_t b[4];
};

/* ========================================================================
 * function prototypes
 * ======================================================================== */

/* usage: given a struct Grp512GF16, set all of its elements to zero
 * params:
 *      1) g: ptr to struct Grp512GF16
 * return: void */
void
grp512_gf16_zero(Grp512GF16* g);

/* usage: given a struct Grp512GF16, find its non-zero elements
 * params:
 *      1) ptr to a uint512_t. Upon return, it encodes the location of non-zero
 *          elements. If the i-th element is non-zero, then the i-th bit is set
 *          and so on.
 *      2) g: ptr to struct Grp512GF16
 * return: void */
void
grp512_gf16_nzpos(uint512_t* restrict out, const Grp512GF16* restrict g);

/* usage: given a struct Grp512GF16, find its zero elements
 * params:
 *      1) ptr to a uint512_t. Upon return, it encodes the location of zero
 *          elements. If the i-th element is zero, then the i-th bit is set and
 *          so on.
 *      2) g: ptr to struct Grp512GF16
 * return: void */
void
grp512_gf16_zpos(uint512_t* restrict out, const Grp512GF16* restrict g);

/* usage: create a copy of a struct Grp512GF16
 * params:
 *      1) dst: ptr to struct Grp512GF16 for holding the copy
 *      2) src: ptr to struct Grp512GF16. The original
 * return: void */
void
grp512_gf16_copy(Grp512GF16* restrict dst, const Grp512GF16* restrict src);

/* usage: given a struct Grp512GF16, randomize its elements
 * params:
 *      1) g: ptr to struct Grp512GF16
 * return: void */
void
grp512_gf16_rand(Grp512GF16* g);

/* usage: given a struct Grp512GF16, set a subset of its elements to zero
 * params:
 *      1) g: ptr to struct Grp512GF16
 *      2) mask: ptr to a 512-bit integer that encodes which elements to zero.
 *          If the i-th bit is 0, then the i-th element will be zeroed.
 *          If 1, then the i-th element is kept.
 * return: void */
void
grp512_gf16_zero_subset(Grp512GF16* restrict g, const uint512_t* restrict mask);

/* usage: given a struct Grp512GF16, set its i-th element to zero
 * params:
 *      1) g: ptr to struct Grp512GF16
 *      2) i: index of the element. 0 ~ 511
 * return: void */
void
grp512_gf16_zero_at(Grp512GF16* g, uint32_t i);

/* usage: given a struct Grp512GF16, return its i-th element
 * params:
 *      1) g: ptr to struct Grp512GF16
 *      2) i: index of the element. 0 ~ 511
 * return: the i-th element as a gf16_t */
gf16_t
grp512_gf16_at(const Grp512GF16* g, uint32_t i);

/* usage: given a struct Grp512GF16, add a value to its i-th element
 * params:
 *      1) g: ptr to struct Grp512GF16
 *      2) i: index of the element. 0 ~ 511
 *      3) v: the value to add
 * return: void */
void
grp512_gf16_add_at(Grp512GF16* g, uint32_t i, gf16_t v);

/* usage: given a struct Grp512GF16, set its i-th element
 * params:
 *      1) g: ptr to struct Grp512GF16
 *      2) i: index of the element. 0 ~ 63
 *      3) v: value of the i-th element
 * return: void */
void
grp512_gf16_set_at(Grp512GF16* g, uint32_t i, gf16_t v);

/* usage: given a struct Grp512GF16 a and b, replace a subset of elements of
 *      a with the corresponding elements of b
 * params:
 *      1) a: ptr to struct Grp512GF16
 *      2) b: ptr to struct Grp512GF16
 *      3) mask: ptr to a uint512_t which encodes which elements of a to keep.
 *          If the i-th bit is set, then the i-th element of a is kept. If 0,
 *          then the i-th element is replaced by that of b.
 * return: void */
void
grp512_gf16_mixi(Grp512GF16* restrict a, const Grp512GF16* restrict b,
                 const uint512_t* restrict mask);

/* usage: given 2 struct Grp512GF16 a and b, compute a + b and store the result
 *      back into a.
 * params:
 *      1) a: ptr to struct Grp512GF16; point to a
 *      2) b: ptr to struct Grp512GF16. point to b
 * return: void */
void
grp512_gf16_addi(Grp512GF16* restrict a, const Grp512GF16* restrict b);

/* usage: given 2 struct Grp512GF16 a and b, compute a - b and store the result
 *      back into a.
 * params:
 *      1) a: ptr to struct Grp512GF16; point to a
 *      2) b: ptr to struct Grp512GF16. point to b
 * return: void */
void
grp512_gf16_subi(Grp512GF16* restrict a, const Grp512GF16* restrict b);

/* usage: given a struct Grp512GF16 and a gf16_t scalar c, multiply 512 elements
 *      in the struct with c and store the result into a given container
 * params:
 *      1) dst: ptr to struct Grp512GF16; storage for the result
 *      2) src: ptr to struct Grp512GF16. the source
 *      3) c: the scalar multiplier
 * return: void */
void
grp512_gf16_mul_scalar(Grp512GF16* restrict dst, const Grp512GF16* restrict src,
                       gf16_t c);

/* usage: given a struct Grp512GF16 and a gf16_t scalar c, multiply 512
 *      elements in the struct with c and store the result back into the struct
 *      Grp512GF16
 * params:
 *      1) src: ptr to struct Grp512GF16. the source
 *      2) c: the scalar multiplier
 * return: void */
void
grp512_gf16_muli_scalar(Grp512GF16* restrict src, gf16_t c);

/* usage: given 2 struct Grp512GF16 a and b, and a gf16_t scalar c, compute
 *      a + b * c, and store the result back into a.
 * params:
 *      1) a: ptr to struct Grp512GF16; point to a
 *      2) b: ptr to struct Grp512GF16. point to b
 *      3) c: the scalar multiplier
 * return: void */
void
grp512_gf16_fmaddi_scalar(Grp512GF16* restrict a, const Grp512GF16* restrict b,
                          gf16_t c);

/* usage: given 2 struct Grp512GF16 a and b, and a gf16_t scalar c, compute
 *      a - b * c, and store the result back into a.
 * params:
 *      1) a: ptr to struct Grp512GF16; point to a
 *      2) b: ptr to struct Grp512GF16. point to b
 *      3) c: the scalar multiplier */
void
grp512_gf16_fmsubi_scalar(Grp512GF16* restrict a, const Grp512GF16* restrict b,
                          gf16_t c);

/* usage: given 2 struct Grp512GF16 a and b, a gf16_t scalar c, and a uint512_t
 *      d that encodes which elements to mask, compute b * c. and add the
 *      result to a based on the mask d. If the i-th bit of d is 1, then
 *      the i-th element of b*c is added to a. If 0, then the i-th element
 *      of a is untouched.
 * params:
 *      1) a: ptr to struct Grp512GF16; point to a
 *      2) b: ptr to struct Grp512GF16. point to b
 *      3) c: the scalar multiplier
 *      4) d: ptr to uint512_t. the mask
 * return: void */
void
grp512_gf16_fmaddi_scalar_mask(Grp512GF16* restrict a,
                               const Grp512GF16* restrict b, gf16_t c,
                               const uint512_t* restrict d);


#endif // __GRP512_GF16_H__
