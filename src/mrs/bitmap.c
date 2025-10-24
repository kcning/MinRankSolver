/* bitmap.c: implementation of bitmap.h */

#include "bitmap.h"
#include "bitmap_table.h"
#include "uint64a.h"

#include <stdlib.h>
#include <assert.h>

/* usage: Given the number of bits to store, create a Bitmap
 * params:
 *      1) size: number of bits to store
 * return: ptr to Bitmap, on error return NULL */
Bitmap*
bitmap_create(const uint64_t size) {
    uint64_t snum = bitmap_calc_slot_num(size);
    Bitmap* b = malloc(sizeof(Bitmap));

    if(!b) {
        return NULL;
    }

    b->s = malloc(sizeof(uint64_t) * snum);
    if(!b->s) {
        free(b);
        return NULL;
    }

    b->snum = snum;
    b->size = size;
    return b;
}

/* usage: Release a struct Bitmap
 * params:
 *      1) b: ptr to a struct Bitmap
 * return: void */
void
bitmap_free(Bitmap* const b) {
    if(!b) {
        return;
    }

    free(b->s);
    free(b);
}

/* usage: Given a Bitmap, populate it with random bits
 * params:
 *      1) b: ptr to struct Bitmap
 * return: void */
void
bitmap_rand(Bitmap* b) {
    for(uint64_t i = 0; i < bitmap_snum(b); ++i) {
        uint32_t* ptr = (uint32_t*) bitmap_slot_addr(b, i);
        ptr[0] = rand();
        ptr[1] = rand();
    }
}

/* usage: Given a Bitmap, check if it's completely zero
 * params:
 *      1) b: ptr to struct Bitmap
 * return: true if so, false otherwise */
bool
bitmap_is_zero(const Bitmap* const b) {
    for(uint64_t i = 0; i < bitmap_snum(b)-1; ++i) { // TODO: unroll this loop
        if(bitmap_slot_at(b, i)) {
            return false;
        }
    }
    if(bitmap_last_slot(b)) {
        return false;
    }
    return true;
}

/* usage: Given 2 Bitmap a and b, check if a&b is completely zero
 * params:
 *      1) a: ptr to struct Bitmap
 *      2) b: ptr to struct Bitmap
 * return: true if so, false otherwise */
bool
bitmap_and_is_zero(const Bitmap* const a, const Bitmap* const b) {
    assert(bitmap_size(a) == bitmap_size(b));
    assert(bitmap_snum(a) == bitmap_snum(b));
    for(uint64_t i = 0; i < bitmap_snum(a)-1; ++i) { // TODO: unroll this loop
        if(bitmap_slot_at(a, i) & bitmap_slot_at(b, i)) {
            return false;
        }
    }
    if(bitmap_last_slot(a) & bitmap_last_slot(b)) {
        return false;
    }
    return true;
}

static inline void
bitmap_xori_512b(uint64_t* const restrict a,
                 const uint64_t* const restrict b) {
    uint64a_xori_512b(a, b);
}

static inline void
bitmap_xori_512b_unalign(uint64_t* const restrict a,
                         const uint64_t* const restrict b) {
    uint64a_xori_512b_unalign(a, b);
}

static inline void
bitmap_andi_512b(uint64_t* const restrict a,
                 const uint64_t* const restrict b) {
    uint64a_andi_512b(a, b);
}

/* usage: Given 2 Bitmap a and b, compute a&b and store the result into a.
 *      (in place and)
 * params:
 *      1) a: ptr to struct Bitmap
 *      2) b: ptr to struct Bitmap
 * return: void */
void
bitmap_andi(Bitmap* const restrict a, const Bitmap* const restrict b) {
    assert(bitmap_size(a) == bitmap_size(b));
    assert(bitmap_snum(a) == bitmap_snum(b));
    uint64_t head = bitmap_snum(a) & ~0x1FULL; // multiple of 32
    for(uint64_t i = 0; i < head; i += 32) {
        bitmap_andi_512b(a->s + i +  0, b->s + i +  0);
        bitmap_andi_512b(a->s + i +  8, b->s + i +  8);
        bitmap_andi_512b(a->s + i + 16, b->s + i + 16);
        bitmap_andi_512b(a->s + i + 24, b->s + i + 24);
    }
    for(uint64_t i = head; i < bitmap_snum(a); ++i) {
        a->s[i] &= b->s[i];
    }
}

static inline void
bitmap_ori_512b(uint64_t* const restrict a,
                const uint64_t* const restrict b) {
    uint64a_ori_512b(a, b);
}

static inline void
bitmap_ori_512b_unalign(uint64_t* const restrict a,
                        const uint64_t* const restrict b) {
    uint64a_ori_512b_unalign(a, b);
}

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
                 uint64_t slot_num) {
    uint64_t head = slot_num & ~0x1FULL; // multiple of 32
    for(uint64_t i = 0; i < head; i += 32) {
        bitmap_ori_512b(a + i +  0, b + i +  0);
        bitmap_ori_512b(a + i +  8, b + i +  8);
        bitmap_ori_512b(a + i + 16, b + i + 16);
        bitmap_ori_512b(a + i + 24, b + i + 24);
    }
    for(uint64_t i = head; i < slot_num; ++i) {
        a[i] |= b[i];
    }
}

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
                         const uint64_t* const restrict b, uint64_t slot_num) {
    uint64_t head = slot_num & ~0x1FULL; // multiple of 32
    for(uint64_t i = 0; i < head; i += 32) {
        bitmap_ori_512b_unalign(a + i +  0, b + i +  0);
        bitmap_ori_512b_unalign(a + i +  8, b + i +  8);
        bitmap_ori_512b_unalign(a + i + 16, b + i + 16);
        bitmap_ori_512b_unalign(a + i + 24, b + i + 24);
    }
    for(uint64_t i = head; i < slot_num; ++i) {
        a[i] |= b[i];
    }
}

/* usage: Given 2 Bitmap a and b, compute a|b and store the result into a.
 *      (in place or)
 * params:
 *      1) a: ptr to struct Bitmap
 *      2) b: ptr to struct Bitmap
 * return: void */
void
bitmap_ori(Bitmap* const restrict a, const Bitmap* const restrict b) {
    assert(bitmap_size(a) == bitmap_size(b));
    assert(bitmap_snum(a) == bitmap_snum(b));
    bitmap_slice_ori(a->s, b->s, bitmap_snum(b));
}

/* usage: Given 2 Bitmap a and b, compute a|b and store the result into a.
 *      (in place or) The Bitmaps do not need to have special alignment.
 * params:
 *      1) a: ptr to struct Bitmap
 *      2) b: ptr to struct Bitmap
 * return: void */
void
bitmap_ori_unalign(Bitmap* const restrict a, const Bitmap* const restrict b) {
    assert(bitmap_size(a) == bitmap_size(b));
    assert(bitmap_snum(a) == bitmap_snum(b));
    bitmap_slice_ori_unalign(a->s, b->s, bitmap_snum(b));
}

/* usage: Given 1 Bitmap, negate its content and store the result back into it
 *      (in place)
 * params:
 *      1) a: ptr to struct Bitmap
 * return: void */
void
bitmap_negi(Bitmap* const a) {
    // TODO: optimize this
    uint64_t head = bitmap_snum(a) & ~0x3ULL; // multiple of 4
    for(uint64_t i = 0; i < head; i += 4) {
        a->s[i + 0] = ~(a->s[i + 0]);
        a->s[i + 1] = ~(a->s[i + 1]);
        a->s[i + 2] = ~(a->s[i + 2]);
        a->s[i + 3] = ~(a->s[i + 3]);
    }
    for(uint64_t i = head; i < bitmap_snum(a); ++i) {
        a->s[i] = ~(a->s[i]);
    }
}

/* usage: Given 1 Bitmap, return the number of bits that are set to 1
 * params:
 *      1) a: ptr to struct Bitmap
 * return: the number of 1's in the slot as a uint64_t */
uint64_t
bitmap_popcnt(const Bitmap* const a) {
    uint64_t sum = 0;
    uint64_t head = (bitmap_snum(a)-1) & ~0x3ULL; // multiple of 4
    for(uint64_t i = 0; i < head; i += 4) {
        uint64_t s0 = popcnt_64b(a->s[i + 0]);
        uint64_t s1 = popcnt_64b(a->s[i + 1]);
        uint64_t s2 = popcnt_64b(a->s[i + 2]);
        uint64_t s3 = popcnt_64b(a->s[i + 3]);
        sum += s0 + s1 + s2 + s3;
    }
    for(uint64_t i = head; i < bitmap_snum(a)-1; ++i) {
        sum += popcnt_64b(a->s[i]);
    }
    sum += popcnt_64b(bitmap_last_slot(a));
    return sum;
}

/* usage: Given 1 Bitmap, return the number of bits that are set to 1 up
 *      to (but not including) the i-th bit
 * params:
 *      1) a: ptr to struct Bitmap
 *      2) i: index of the bit to stop counting
 * return: the number of 1's up to (but not including) the i-th bit */
uint64_t
bitmap_popcnt_upto(const Bitmap* const a, uint64_t i) {
    uint64_t slot_idx = i >> 6;
    uint64_t offset = i & 0x3FUL;
    uint64_t c = 0;
    for(uint64_t j = 0; (j + 1) < slot_idx; ++j) {
        c += popcnt_64b(a->s[j]);
    }

    uint64_t last_slot = a->s[slot_idx];
    uint64_t mask = (0x1UL << offset) - 1;
    c += popcnt_64b(last_slot & mask);
    return c;
}

/* usage: Given 1 Bitmap, return the number of trailing zeros from LSB
 * params:
 *      1) a: ptr to struct Bitmap
 * return: the number of trailing zeros as a uint64_t; UINT64_MAX if the
 *      Bitmap is zero */
uint64_t
bitmap_ctz(const Bitmap* const a) {
    for(uint64_t i = 0; i < bitmap_snum(a)-1; ++i) {
        if(bitmap_slot_at(a, i)) {
            return (i << 6) + uint64_t_ctz(bitmap_slot_at(a, i));
        }
    }
    if(bitmap_last_slot(a)) {
        return ((bitmap_snum(a)-1) << 6) + uint64_t_ctz(bitmap_last_slot(a));
    }
    return UINT64_MAX;
}

/* usage: Given 2 Bitmap a and b, return the number of trailing zeros from LSB
 *      in a&b
 * params:
 *      1) a: ptr to struct Bitmap
 *      2) b: ptr to struct Bitmap
 * return: the number of trailing zeros as a uint64_t; UINT64_MAX if a & b
 *      Bitmap is zero */
uint64_t
bitmap_and_ctz(const Bitmap* const restrict a, const Bitmap* const restrict b) {
    assert(bitmap_size(a) == bitmap_size(b));
    for(uint64_t i = 0; i < bitmap_snum(a)-1; ++i) {
        uint64_t slot = bitmap_slot_at(a, i) & bitmap_slot_at(b, i);
        if(slot) {
            return (i << 6) + uint64_t_ctz(slot);
        }
    }
    uint64_t slot = bitmap_last_slot(a) & bitmap_last_slot(b);
    if(slot) {
        return ((bitmap_snum(a)-1) << 6) + uint64_t_ctz(slot);
    }
    return UINT64_MAX;
}

/* usage: Given 1 Bitmap, find indices of all set bits in it
 * params:
 *      1) b: ptr to struct Bitmap whose size must be smaller than 2^32
 *      2) res: an uint32_t array for storing the indices, must hold at least
 *              as many elements as the size of the bitmap rounded up to a
 *              multiple of 64
 * return: the number of set bits */
int
bitmap_sbidx(const Bitmap* const restrict b, uint32_t* const restrict res) {
    uint64_t base = 0x0ULL;
    const uint64_t inc64 = 0x0000004000000040ULL;
    uint64_t sbnum = 0;
    uint64_t head = (bitmap_snum(b)-1) & ~0x3ULL; // multiple of 4
    for(uint64_t i = 0; i < head; i += 4) {
        sbnum += sbidx_in_64b_sz32(res + sbnum, base, bitmap_slot_at(b, i+0));
        base += inc64;
        sbnum += sbidx_in_64b_sz32(res + sbnum, base, bitmap_slot_at(b, i+1));
        base += inc64;
        sbnum += sbidx_in_64b_sz32(res + sbnum, base, bitmap_slot_at(b, i+2));
        base += inc64;
        sbnum += sbidx_in_64b_sz32(res + sbnum, base, bitmap_slot_at(b, i+3));
        base += inc64;
    }
    for(uint64_t i = head; i < bitmap_snum(b)-1; ++i) {
        sbnum += sbidx_in_64b_sz32(res + sbnum, base, bitmap_slot_at(b, i));
        base += inc64;
    }
    sbnum += sbidx_in_64b_sz32(res + sbnum, base, bitmap_last_slot(b));
    assert(sbnum <= bitmap_size(b));
    return sbnum;
}

/* usage: Given 1 Bitmap and an array of indices, compute the result of
 *      bitwise-and of the selected bits
 * params:
 *      1) b: ptr to struct Bitmap whose size is smaller than 2^32
 *      2) idx: an array of uint32_t that selects the bits
 *      3) size: size of idx
 * return: result of the computation */
bool
bitmap_bitwise_and(const Bitmap* const restrict b,
                   const uint32_t* const restrict idx, uint64_t size) {
    for(uint64_t i = 0; i < size; ++i) {
        if(!bitmap_at(b, idx[i])) {
            return false;
        }
    }
    return true;
}

/* usage: Given 1 Bitmap and a uint64_t, populate the bitmap, starting
 *      from the given offset, with the bits in the uint64_t
 * params:
 *      1) b: ptr to struct Bitmap
 *      2) v: a uint64_t
 *      3) num: number of valid bits in v, starting from LSB
 *      4) offset: offset starting from which to fill in the bits
 * return: void */
void
bitmap_fill(Bitmap* const b, uint64_t v, uint64_t num, uint64_t offset) {
    assert( (offset + num) <= bitmap_size(b));
    for(uint64_t i = 0; i < num; ++i) {
        bitmap_set_at(b, offset + i, v & 0x1ULL);
        v >>= 1;
    }
}
