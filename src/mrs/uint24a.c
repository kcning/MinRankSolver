/* uint24a.c: implementation of struct uint24a */

#include "uint24a.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

#define UINT24A_ALIGNMENT   (4ULL)

struct uint24a {
    uint32_t snum; /* number of slots, 3x the number of elements plus padding */
    uint8_t blk[];
};

/* ========================================================================
 * function implementations
 * ======================================================================== */

/* usage:Given a uint24a and an index i, return the address to the i-th uint24
 * params:
 *      1) arr: ptr to struct uint24a
 *      2) i: the index of the element whose addr to return
 * return: the address as a uint8_t ptr */
uint8_t*
uint24a_addr(uint24a* arr, uint64_t i) {
    return arr->blk + 3 * i;
}

/* usage: Given the number of uint24_t to store in the array, compute the
 *      number of uint8_t needed for storage
 * params:
 *      n: number of uint24_t to store
 * return: number of uint8_t needed */
uint64_t
uint24a_calc_slotnum(const uint64_t n) {
    static_assert(4 == UINT24A_ALIGNMENT, "uint24a does not have alignment 4");
    return ((n * 3ULL + 3LL) / 4ULL) << 2; // round up to multiple of 4
}

/* usage: Given the number of uint24_t to store in the array, compute the
 *      number of bytes needed for storage
 * params:
 *      n: number of uint24_t to store
 * return: number of bytes needed */
uint64_t
uint24a_memsize(const uint64_t n) {
    const uint64_t snum = uint24a_calc_slotnum(n);
    static_assert(sizeof(uint24a) == 4, "uint24a does not have size 4");
    return sizeof(uint24a) + sizeof(uint8_t) * snum;
}

/* usage: Given the number of uint24_t to store in the array (i.e. the array
 *      size), create the array.
 * params:
 *      1) n: number of uint24_t to store
 * return: a ptr to struct uint24a. On error, return NULL. */
uint24a*
uint24a_create(const uint64_t n) {
    const uint64_t snum = uint24a_calc_slotnum(n);
    uint24a* a = aligned_alloc(UINT24A_ALIGNMENT, uint24a_memsize(n));
    if(!a) {
        return NULL;
    }
    a->snum = snum;
    return a;
}

/* usage: Release a struct uint24a
 * params:
 *      1) a: ptr to struct uint24a
 * return: void */
void
uint24a_free(uint24a* a) {
    free(a);
}

/* usage: set a struct uint24a to full zero
 * params:
 *      1) arr: ptr to struct uint24a
 * return: void */
void
uint24a_zero(uint24a* a) {
    memset(a->blk, 0x0, sizeof(uint8_t) * (a->snum + 1));
}

/* usage: set a struct uint24a to full 1's
 * params:
 *      1) arr: ptr to struct uint24a
 * return: void */
void
uint24a_max(uint24a* a) {
    memset(a->blk, UINT8_MAX, sizeof(uint8_t) * (a->snum + 1));
}

/* Given the start addr of a uint24, return its value */
static inline uint32_t
load_uint24(const uint8_t* const e) {
    const uint32_t* const src = (uint32_t*) e;
    return *src & UINT24_MAX;
}

/* Given the start addr of a uint24, store the input v */
static inline void
store_uint24(uint8_t* const e, const uint32_t v) {
    assert(v <= UINT24_MAX);
    // keep the lowest 8 bits of the next uint24 
    *((uint32_t*) e) = v | ( ((uint32_t) e[3]) << 24);
}

/* usage: Given a uint24a and an index i, return the i-th uint24 as a uint32_t
 * params:
 *      1) arr: ptr to struct uint24a
 *      2) i: the index of the element
 * return: the i-th element */
uint32_t
uint24a_at(const uint24a* const arr, const uint64_t i) {
    assert(3 * i < arr->snum);
    return load_uint24(uint24a_addr( (uint24a*) arr, i));
}

/* usage: Given a uint24a and an index i, return i-th ~ (i+63)-th uint24 as
 *      an array of uint32_t
 * params:
 *      1) arr: ptr to struct uint24a
 *      2) i: the starting index
 *      3) dst: container for the results; must hold at least 64 elements
 * return: void */
void
uint24a_at_grp64(const uint24a* const restrict arr, const uint64_t i,
                 uint32_t* const restrict dst) {
    assert(3 * (i + 64) <= arr->snum);

    // TODO: optimize with AVX2 and AVX512, respectively
    uint32_t* src = (uint32_t*) uint24a_addr( (uint24a*) arr, i);
    for(uint64_t c = 0; c < 64; c += 4, src += 3) {
        uint32_t v0 = src[0];
        uint32_t v1 = src[1];
        uint32_t v2 = src[2];

        dst[c + 0] = v0 & UINT24_MAX;
        dst[c + 1] = ((v0 >> 24) | (v1 << 8)) & UINT24_MAX;
        dst[c + 2] = ((v1 >> 16) | (v2 << 16)) & UINT24_MAX;
        dst[c + 3] = v2 >> 8;
    }
}

/* usage: Given a uint24a and an index i, store the input v as the i-th uint24
 * params:
 *      1) arr: ptr to struct uint24a
 *      2) i: the index of the element
 *      3) v: a uint32_t integer less than 2^24
 * return: void */
void
uint24a_set_at(uint24a* const arr, const uint64_t i, const uint32_t v) {
    assert(3 * i < arr->snum);
    store_uint24(uint24a_addr(arr, i), v);
}

/* usage: Given a slice of a uint24a, and the number of elements in the slide
 *      set all elements in the slice to zero
 * params:
 *      1) s: a uint8_t ptr to the start of the slice
 *      2) n: number of elements in the slice
 * return: the void */
void
uint24a_slice_zero(uint8_t* s, uint64_t n) {
    memset(s, 0x0, sizeof(uint8_t) * 3 * n);
}

/* usage: Given a slice of a uint24a, and an index i, return the address to
 *      the i-th uint24 in the slice
 * params:
 *      1) s: a uint8_t ptr to the start of the slice
 *      2) i: the index of the element
 * return: the address as a uint8_t ptr */
uint8_t*
uint24a_slice_addr(uint8_t* s, uint64_t i) {
    return s + 3 * i;
}

/* usage: Given a slice of a uint24a, and an index i, return the i-th uint24
 *      in the slice as a uint32_t
 * params:
 *      1) s: a uint8_t ptr to the start of the slice
 *      2) i: the index of the element
 * return: the i-th element */
uint32_t
uint24a_slice_at(const uint8_t* s, uint64_t i) {
    return load_uint24(uint24a_slice_addr((uint8_t*) s, i));
}

/* usage: Given a slice of a uint24a and an index i, store the input v as the
 *      i-th uint24 in the slice.
 * params:
 *      1) s: a uint8_t ptr to the start of the slice
 *      2) i: the index of the element
 *      3) v: a uint32_t integer less than 2^24
 * return: void */
void
uint24a_slice_set_at(uint8_t* s, uint64_t i, uint32_t v) {
    store_uint24(uint24a_slice_addr(s, i), v);
}

/* usage: Given a slice of uint24a and an index i, return i-th ~ (i+4)-th uint24
 *      in the slice as an array of uint32_t
 * params:
 *      1) s: a uint8_t ptr to the start of the slice
 *      2) i: the starting index
 *      3) dst: container for the results; must hold at least 4 elements
 * return: void */
void
uint24a_slice_at_grp4(const uint8_t* s, uint64_t i, uint32_t* dst) {
    dst[0] = uint24a_slice_at(s, i + 0);
    dst[1] = uint24a_slice_at(s, i + 1);
    dst[2] = uint24a_slice_at(s, i + 2);
    dst[3] = uint24a_slice_at(s, i + 3);
}

/* usage: Given a slice of uint24a and an index i, return i-th ~ (i+63)-th uint24
 *      in the slice as an array of uint32_t
 * params:
 *      1) s: a uint8_t ptr to the start of the slice
 *      2) i: the starting index
 *      3) dst: container for the results; must hold at least 64 elements
 * return: void */
void
uint24a_slice_at_grp64(const uint8_t* const restrict s, const uint64_t i,
                       uint32_t* const restrict dst) {
    // TODO: optimize with AVX2 and AVX512, respectively
    uint32_t* src = (uint32_t*) uint24a_slice_addr((uint8_t*) s, i);
    for(uint64_t c = 0; c < 64; c += 4, src += 3) {
        uint32_t v0 = src[0];
        uint32_t v1 = src[1];
        uint32_t v2 = src[2];

        dst[c + 0] = v0 & UINT24_MAX;
        dst[c + 1] = ((v0 >> 24) | (v1 << 8)) & UINT24_MAX;
        dst[c + 2] = ((v1 >> 16) | (v2 << 16)) & UINT24_MAX;
        dst[c + 3] = v2 >> 8;
    }
}
