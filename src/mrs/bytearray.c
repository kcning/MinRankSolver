#include "bytearray.h"

#include <stdlib.h>


/* ========================================================================
 * struct ByteArray definition
 * ======================================================================== */

struct ByteArray {
    uint64_t size;          // number of bytes
    uint64_t snum;          // number of slots
    uint8_t* restrict s;   // memory block storing the bytes
};

/* ========================================================================
 * function implementations
 * ======================================================================== */

/* usage: Given the number of bytes to store, create a ByteArray
 * params:
 *      1) size: number of bytes to store
 * return: ptr to ByteArray, on error return NULL */
ByteArray*
bytearray_create(const uint64_t size) {
    if(size == 0)
        return NULL;

    ByteArray* b = malloc(sizeof(ByteArray));
    if(!b)
        return NULL;

    uint64_t snum = bytearray_calc_slot_num(size);
    b->s = aligned_alloc(64, sizeof(uint64_t) * snum);
    if(!b->s) {
        free(b);
        return NULL;
    }

    b->snum = snum;
    b->size = size;
    return b;
}

/* usage: Given the number of bytes to store, and a memory block, create a
 *      ByteArray
 * params:
 *      1) size: number of bytes to store
 *      2) mem: ptr to a memory block that can hold at least
 *          'bytearray_calc_slot_num(size)' uint64_t. The address
 *          must be aligned to a multiple of 64.
 * return: ptr to ByteArray, on error return NULL */
ByteArray*
bytearray_create_from_mem(const uint64_t size, uint8_t* mem) {
    if((size == 0) || (!mem))
        return NULL;

    ByteArray* b = malloc(sizeof(ByteArray));
    if(!b)
        return NULL;

    b->s = mem;
    b->snum = bytearray_calc_slot_num(size);
    b->size= size;

    return b;
}

/* usage: Release a struct ByteArray
 * params:
 *      1) b: ptr to a struct ByteArray
 *      2) free_block: if true, release the memory blocking storing the bytes
 * return: void */
void
bytearray_free(ByteArray* const b, bool free_block) {
    if(!b) {
        return;
    }

    if(free_block)
        free(b->s);
    free(b);
}

/* usage: Given a ByteArray, return its size
 * params:
 *      1) b: ptr to struct ByteArray
 * return: number of bytes that the ByteArray can hold */
uint64_t pure_func
bytearray_size(const ByteArray* b) {
    return b->size;
}

/* usage: Given a ByteArray, return the number of slots that are used to store
 *      the bytes internally
 * params:
 *      1) b: ptr to struct ByteArray
 * return: number of slots */
uint64_t pure_func
bytearray_snum(const ByteArray* b) {
    return b->snum;
}

/* usage: Given a ByteArray, return its internal memory block
 * params:
 *      1) b: ptr to struct ByteArray
 * return: ptr to its internal memory block */
uint8_t*
bytearray_memblk(ByteArray* b) {
    return b->s;
}

/* usage: Given a ByteArray, clear its internal memory block
 * params:
 *      1) b: ptr to struct ByteArray
 * return: void */
void
bytearray_zero(ByteArray* b) {
    memset(bytearray_memblk(b), 0x0, sizeof(uint64_t) * bytearray_snum(b));
}

/* usage: Given a ByteArray, return addr to its i-th byte
 * params:
 *  1) b: ptr to struct ByteArray
 *  2) idx: index of the target byte
 * return: ptr to the target byte */
const uint8_t*
bytearray_addr_at(const ByteArray* b, uint64_t idx) {
    return b->s + idx;
}

/* usage: Given a ByteArray, return its i-th byte
 * params:
 *      1) b: ptr to struct ByteArray
 *      2) idx: index of the target byte
 * return: the target byte */
uint8_t
bytearray_at(const ByteArray* b, uint64_t idx) {
    assert(idx < b->size);
    return b->s[idx];
}

/* usage: Given a ByteArray, set its i-th byte
 * params:
 *      1) b: ptr to struct ByteArray
 *      2) idx: index of the target byte
 *      3) v: the new value for the byte
 * return: void */
void
bytearray_set_at(const ByteArray* b, uint64_t idx, uint8_t v) {
    assert(idx < b->size);
    b->s[idx] = v;
}

/* usage: Given a ByteArray, count the number of zero bytes
 * params:
 *      1) b: ptr to struct ByteArray
 * return: number of bytes that are zero */
uint32_t
bytearray_cz(const ByteArray* b) {
    uint32_t c = 0;
    for(uint32_t i = 0; i < bytearray_size(b); ++i) {
        if(0 == bytearray_at(b, i))
            ++c;
    }

    return c;
}
