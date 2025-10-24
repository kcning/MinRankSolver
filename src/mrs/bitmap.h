/* bitmap.h: header file for struct Bitmap */

#ifndef __BLK_LANCZOS_BITMAP_H__
#define __BLK_LANCZOS_BITMAP_H__

#include <stdint.h>
#include <stdbool.h>
#include <assert.h>

#include "util.h"
#include "uint64a.h"

/* ========================================================================
 * struct Bitmap definition
 * ======================================================================== */

typedef struct {
    uint64_t size; // number of bits
    uint64_t snum; // number of slots used to store the bits
    uint64_t* restrict s;
} Bitmap;

/* ========================================================================
 * functions prototypes
 * ======================================================================== */

/* usage: Given the number of bits to store, create a Bitmap
 * params:
 *      1) size: number of bits to store
 * return: ptr to Bitmap, on error return NULL */
Bitmap*
bitmap_create(const uint64_t size);

/* usage: Release a struct Bitmap
 * params:
 *      1) b: ptr to a struct Bitmap
 * return: void */
void
bitmap_free(Bitmap* const b);

/* usage: Given the number of bits to store in the Bitmap, compute the number
 *      of slots (uint64_t) needed
 * params:
 *      1) size: the number of bits to store
 * return: the number of uint64_t needed as a uint64_t */
static inline uint64_t pure_func
bitmap_calc_slot_num(uint64_t size) {
    return (size + 63) / 64;
}

/* usage: Given a Bitmap, return its size
 * params:
 *      1) b: ptr to struct Bitmap
 * return: number of bits that the Bitmap can hold */
#define bitmap_size(b) \
    ((b)->size)

/* usage: Given a Bitmap, return the number of slots that are used to store
 *      the bits internally
 * params:
 *      1) b: ptr to struct Bitmap
 * return: number of slots */
#define bitmap_snum(b) \
    ((b)->snum)

/* usage: Given a Bitmap and index i, return the i-th slot
 * params:
 *      1) b: ptr to struct Bitmap
 *      2) i: index of the slot
 * return: the i-th slot as a uint64_t */
#define bitmap_slot_at(b, i) \
    ((b)->s[(i)])

/* usage: Given 1 Bitmap and an index i, return the i-th entry in the bitmap
 * params:
 *      1) a: ptr to struct Bitmap
 *      2) i: index to the entry, from 0 ~ size of the bitmap-1. Do not perform
 *              arithmetic here
 * return: the entry */
static inline uint8_t
bitmap_at(const Bitmap* b, uint64_t i) {
    uint8_t* bb = (uint8_t*) b->s;
    return ((bb[i >> 3]) >> (i & 0x7UL)) & 0x1U;
}

/* usage: Given 1 Bitmap and an index i, toggle the i-th entry in the bitmap
 * params:
 *      1) a: ptr to struct Bitmap
 *      2) i: index to the entry, from 0 ~ size of the bitmap-1. Do not perform
 *              arithmetic here
 * return: void */
#define bitmap_toggle_at(a, i) do { \
    Bitmap* const aa = (a); \
    uint64_t const ii = (i); \
    ((uint8_t*)aa->s)[ii >> 3] ^= 0x1U << (ii & 0x7U); \
} while(0)

/* usage: Given 1 Bitmap, an index i, and value v, set the i-th entry in
 *      the Bitmap to value v
 * params:
 *      1) a: ptr to struct Bitmap
 *      2) i: index to the entry, from 0 ~ size of the bitmap-1. Do not perform
 *              arithmetic here.
 *      3) v: true for 1, false for 0
 * return: void */
static inline void
bitmap_set_at(Bitmap* b, uint64_t i, uint8_t v) {
    assert(v == 0x0 || v == 0x1);
    uint8_t* bb = (uint8_t*) b->s;
    uint64_t slot_idx = i >> 3;
    uint64_t offset = i & 0x7UL;
    bb[slot_idx] &= ~(0x1U << offset);
    bb[slot_idx] |= v << offset;
}

/* usage: Given 1 Bitmap, an index i, set the i-th entry in the Bitmap to true
 * params:
 *      1) a: ptr to struct Bitmap
 *      2) i: index to the entry, from 0 ~ size of the bitmap-1. Do not perform
 *              arithmetic here.
 * return: void */
static inline void
bitmap_set_true_at(Bitmap* b, uint64_t i) {
    uint8_t* bb = (uint8_t*) b->s;
    uint64_t slot_idx = i >> 3;
    uint64_t offset = i & 0x7UL;
    bb[slot_idx] |= 0x1U << offset;
}

/* usage: Given 2 ptrs to uint64_t a and b, treat a and b as Bitmaps to
 *      compute a^b and store the result into a (in place xor). Note
 *      that the ptrs must hold addresses with alignment 64.
 * params:
 *      1) a: ptr to uint64_t
 *      2) b: ptr to uint64_t
 *      3) slot_num: number of uint64_t used to store bits of a; must
 *              be the same for b
 * return: void */
static inline void
bitmap_slice_xori(uint64_t* const restrict a, const uint64_t* const restrict b,
                  uint64_t slot_num) {
    uint64a_xori(a, b, slot_num);
}

/* usage: Given 2 ptrs to uint64_t a and b, treat a and b as Bitmaps to
 *      compute a^b and store the result into a (in place xor).
 *      The addresses stored by the ptrs do not need to have special alignment
 * params:
 *      1) a: ptr to uint64_t
 *      2) b: ptr to uint64_t
 *      3) slot_num: number of uint64_t used to store bits of a; must
 *              be the same for b
 * return: void */
static inline void
bitmap_slice_xori_unalign(uint64_t* const restrict a,
                          const uint64_t* const restrict b, uint64_t slot_num) {
    uint64a_xori_unalign(a, b, slot_num);
}

/* usage: Given 1 ptr to uint64_t a and 1 Bitmap b, treat a as a Bitmap to
 *      compute a^b and store the result into a (in place xor).
 *      The addresses stored by the ptrs do not need to have special alignment
 * params:
 *      1) a: ptr to uint64_t, must hold as many uint64_t as the Bitmap b
 *      2) b: ptr to Bitmap
 * return: void */
static inline void
bitmap_uint64_xori_unalign(uint64_t* const restrict a,
                           const Bitmap* const restrict b) {
    bitmap_slice_xori_unalign(a, b->s, bitmap_snum(b));
}

/* usage: Given 2 Bitmap a and b, compute a^b and store the result into a.
 *      (in place xor)
 * params:
 *      1) a: ptr to struct Bitmap
 *      2) b: ptr to struct Bitmap
 * return: void */
static inline void
bitmap_xori(Bitmap* const restrict a, const Bitmap* const restrict b) {
    assert(bitmap_size(a) == bitmap_size(b));
    assert(bitmap_snum(a) == bitmap_snum(b));
    bitmap_slice_xori(a->s, b->s, bitmap_snum(b));
}

/* usage: Given 2 Bitmap a and b, compute a^b and store the result into a.
 *      (in place xor) The Bitmaps do not need to have special alignment.
 * params:
 *      1) a: ptr to struct Bitmap
 *      2) b: ptr to struct Bitmap
 * return: void */
static inline void
bitmap_xori_unalign(Bitmap* const restrict a, const Bitmap* const restrict b) {
    assert(bitmap_size(a) == bitmap_size(b));
    assert(bitmap_snum(a) == bitmap_snum(b));
    bitmap_slice_xori_unalign(a->s, b->s, bitmap_snum(b));
}

/* usage: Given 2 Bitmap a and b, compute a&b and store the result into a.
 *      (in place and)
 * params:
 *      1) a: ptr to struct Bitmap
 *      2) b: ptr to struct Bitmap
 * return: void */
void
bitmap_andi(Bitmap* const restrict a, const Bitmap* const restrict b);

/* usage: Given 2 ptrs to uint64_t a and b, treat a and b as Bitmaps to
 *      compute a|b and store the result into a (in place or). Note
 *      that the ptrs must hold addresses with alignment 64.
 * params:
 *      1) a: ptr to uint64_t
 *      2) b: ptr to uint64_t
 *      3) slot_num: number of uint64_t used to store bits of a; must
 *              be the same for b
 * return: void */
void
bitmap_slice_ori(uint64_t* const restrict a, const uint64_t* const restrict b,
                 uint64_t slot_num);

/* usage: Given 2 ptrs to uint64_t a and b, treat a and b as Bitmaps to
 *      compute a|b and store the result into a (in place or).
 *      The addresses stored by the ptrs do not need to have special alignment
 * params:
 *      1) a: ptr to uint64_t
 *      2) b: ptr to uint64_t
 *      3) slot_num: number of uint64_t used to store bits of a; must
 *              be the same for b
 * return: void */
void
bitmap_slice_ori_unalign(uint64_t* const restrict a,
                         const uint64_t* const restrict b, uint64_t slot_num);

/* usage: Given 2 Bitmap a and b, compute a|b and store the result into a.
 *      (in place or)
 * params:
 *      1) a: ptr to struct Bitmap
 *      2) b: ptr to struct Bitmap
 * return: void */
void
bitmap_ori(Bitmap* const restrict a, const Bitmap* const restrict b);

/* usage: Given 2 Bitmap a and b, compute a|b and store the result into a.
 *      (in place or) The Bitmaps do not need to have special alignment.
 * params:
 *      1) a: ptr to struct Bitmap
 *      2) b: ptr to struct Bitmap
 * return: void */
void
bitmap_ori_unalign(Bitmap* const restrict a, const Bitmap* const restrict b);

/* usage: Given 1 Bitmap, negate its content and store the result back into it
 *      (in place)
 * params:
 *      1) a: ptr to struct Bitmap
 * return: void */
void
bitmap_negi(Bitmap* const a);

/* usage: Given 1 Bitmap, return the number of bits that are set to 1
 * params:
 *      1) a: ptr to struct Bitmap
 * return: the number of 1's in the slot as a uint64_t */
uint64_t
bitmap_popcnt(const Bitmap* const a);

/* usage: Given 1 Bitmap, return the number of bits that are set to 1 up
 *      to (but not including) the i-th bit
 * params:
 *      1) a: ptr to struct Bitmap
 *      2) i: index of the bit to stop counting
 * return: the number of 1's up to (but not including) the i-th bit */
uint64_t
bitmap_popcnt_upto(const Bitmap* const a, uint64_t i);

/* usage: Given 1 Bitmap, return the number of bits that are set to 0
 * params:
 *      1) a: ptr to struct Bitmap
 * return: the number of 0's in the slot as a uint64_t */
static inline uint64_t
bitmap_zcnt(const Bitmap* const a) {
    return bitmap_size(a) - bitmap_popcnt(a);
}

/* usage: Given 1 Bitmap, return the number of trailing zeros from LSB
 * params:
 *      1) a: ptr to struct Bitmap
 * return: the number of trailing zeros as a uint64_t; UINT64_MAX if the
 *      Bitmap is zero */
uint64_t
bitmap_ctz(const Bitmap* const a);

/* usage: Given 2 Bitmap a and b, return the number of trailing zeros from LSB
 *      in a&b
 * params:
 *      1) a: ptr to struct Bitmap
 *      2) b: ptr to struct Bitmap
 * return: the number of trailing zeros as a uint64_t; UINT64_MAX if a & b
 *      Bitmap is zero */
uint64_t
bitmap_and_ctz(const Bitmap* const restrict a, const Bitmap* const restrict b);

/* usage: Given a Bitmap, return the last slot
 * param:
 *      1) b: ptr to struct Bitmap
 * return: masked last slot as a uint64_t */
static inline uint64_t
bitmap_last_slot(const Bitmap* const a) {
    uint64_t remain = bitmap_size(a) & 0x3FULL;
    uint64_t mask = (remain) ? ((0x1ULL << remain)-1) : UINT64_MAX;
    return (a->s[bitmap_snum(a)-1] & mask);
}

/* usage: Given a Bitmap and index i, return a pointer to the i-th slot
 * params:
 *      1) b: ptr to struct Bitmap
 *      2) i: index of the slot
 * return: ptr to the i-th slot as a ptr to uint64_t */
static inline uint64_t*
bitmap_slot_addr(Bitmap* b, uint64_t i) {
    return b->s + i;
}

/* usage: Given a Bitmap, set it to completely zero
 * params:
 *      1) b: ptr to struct Bitmap
 * return: void */
static inline void
bitmap_zero(Bitmap* b) {
    memset(b->s, 0x0, sizeof(uint64_t) * bitmap_snum(b));
}

/* usage: Given a Bitmap, set it to completely 1's
 * params:
 *      1) b: ptr to struct Bitmap
 * return: void */
static inline void
bitmap_set_max(Bitmap* b) {
    memset(b->s, UINT8_MAX, sizeof(uint64_t) * bitmap_snum(b));
}

/* usage: Given a Bitmap, check if it's completely zero
 * params:
 *      1) b: ptr to struct Bitmap
 * return: true if so, false otherwise */
bool
bitmap_is_zero(const Bitmap* const b);

/* usage: Given 2 Bitmap a and b, check if a&b is completely zero
 * params:
 *      1) a: ptr to struct Bitmap
 *      2) b: ptr to struct Bitmap
 * return: true if so, false otherwise */
bool
bitmap_and_is_zero(const Bitmap* const a, const Bitmap* const b);

/* usage: Given 2 Bitmap a and b, copy b to a
 * params:
 *      1) a: ptr to struct Bitmap
 *      2) b: ptr to struct Bitmap
 * return: void */
#define bitmap_copy(a, b) do { \
    Bitmap* const aa = (a); \
    const Bitmap* const bb = (b); \
    assert(bitmap_size(aa) == bitmap_size(bb)); \
    assert(bitmap_snum(aa) == bitmap_snum(bb)); \
    memcpy(aa->s, bb->s, sizeof(uint64_t) * bitmap_snum(aa)); \
} while(0)

/* usage: Given 1 Bitmap b and an array of uint64_t, copy the content of b
 *      into the array
 * params:
 *      1) a: an array of uint64_t, which must be as large as the number of
 *              slots used to hold the content of the Bitmap
 *      2) b: ptr to struct Bitmap
 * return: void */
#define bitmap_dump(a, b) do { \
    const Bitmap* const bb = (b); \
    memcpy((a), bb->s, sizeof(uint64_t) * bitmap_snum(bb)); \
} while(0)

/* usage: Given a Bitmap, populate it with random bits
 * params:
 *      1) b: ptr to struct Bitmap
 * return: void */
void
bitmap_rand(Bitmap* b);

/* usage: Given 1 Bitmap, find indices of all set bits in it
 * params:
 *      1) b: ptr to struct Bitmap whose size must be smaller than 2^32
 *      2) res: an uint32_t array for storing the indices, must hold at least
 *              as many elements as the size of the bitmap rounded up to a
 *              multiple of 64
 * return: the number of set bits */
int
bitmap_sbidx(const Bitmap* const restrict b, uint32_t* const restrict res);

/* usage: Given 1 Bitmap and an array of indices, compute the result of
 *      bitwise-and of the selected bits
 * params:
 *      1) b: ptr to struct Bitmap whose size is smaller than 2^32
 *      2) idx: an array of uint32_t that selects the bits
 *      3) size: size of idx
 * return: result of the computation */
bool
bitmap_bitwise_and(const Bitmap* const restrict b,
                   const uint32_t* const restrict idx, uint64_t size);

/* usage: Given 1 Bitmap and a uint64_t, populate the bitmap, starting
 *      from the given offset, with the bits in the uint64_t
 * params:
 *      1) b: ptr to struct Bitmap
 *      2) v: a uint64_t
 *      3) num: number of valid bits in v, starting from LSB
 *      4) offset: offset starting from which to fill in the bits
 * return: void */
void
bitmap_fill(Bitmap* const b, uint64_t v, uint64_t num, uint64_t offset);

#endif /* __BLK_LANCZOS_BITMAP_H__ */
