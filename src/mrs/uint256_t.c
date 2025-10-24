/* uint256_t.c: implementation of struct uint256_t */

#include "uint256_t.h"

#include <string.h> // ffsll
#include <assert.h>


/* usage: Given 1 uint256_t a, find location of the first set bit in a,
 *      starting from the given slot
 * params:
 *      1) a: ptr to struct uint256_t
 *      2) idx: the index of the slot to start searching; from 0 ~ 3
 * return: i if i-th bit is the first set bit. If no bits are set, return 0. */
static inline int
uint256_t_ffs_from_slot(const uint256_t* const a, const uint64_t idx) {
    for(uint64_t i = idx; i < 4; ++i) {
        if(a->s[i])
            return (i << 6) + ffsll(a->s[i]);
    }
    return 0;
}

/* usage: Given 1 uint256_t a, find location of the first set bit in a
 * params:
 *      1) a: ptr to struct uint256_t
 * return: i if i-th bit is the first set bit. If no bits are set, return 0. */
int
uint256_t_ffs(const uint256_t* const a) {
    return uint256_t_ffs_from_slot(a, 0);
}

/* usage: Given 1 uint256_t a, find the location of first set bit after the
 *      i-th bit (included)
 * params:
 *      1) a: ptr to struct uint256_t
 *      2) i: start searching from i-th bit; from 0 ~ 255
 * return: i if i-th bit is the first set bit. If no bits are set, return 0. */
int
uint256_t_ffs_after(const uint256_t* const a, uint64_t i) {
    assert(i < 256);
    uint64_t start_slot_idx = i >> 6;
    // clear all bits up to the given index
    uint64_t r = ((a->s[start_slot_idx]) >> (i & 0x3FULL)) << (i & 0x3FULL);
    if(r) {
        return (start_slot_idx << 6) + ffsll(r);
    }
    return uint256_t_ffs_from_slot(a, start_slot_idx+1);
}
