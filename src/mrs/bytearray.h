#ifndef __BYTEARRAY_H__
#define __BYTEARRAY_H__

#include "util.h"

#include <stdint.h>
#include <stdbool.h>

typedef struct ByteArray ByteArray;

/* ========================================================================
 * functions prototypes
 * ======================================================================== */

/* usage: Given the number of bytes to store in the ByteArray, compute the
 *      number of slots (uint64_t) needed
 * params:
 *      1) size: number of bytes to store
 * return: the number of uint64_t needed as a uint64_t */
static inline uint64_t pure_func
bytearray_calc_slot_num(uint64_t size) {
    uint64_t num_64b = (size + 63) / 64; // round up the closet multiple of 64
    return num_64b * 64ULL / sizeof(uint64_t);
}

/* usage: Given the number of bytes to store, create a ByteArray
 * params:
 *      1) size: number of bytes to store
 * return: ptr to ByteArray, on error return NULL */
ByteArray*
bytearray_create(const uint64_t size);

/* usage: Given the number of bytes to store, and a memory block, create a
 *      ByteArray
 * params:
 *      1) size: number of bytes to store
 *      2) mem: ptr to a memory block that can hold at least
 *          'bytearray_calc_slot_num(size)' uint64_t. The address
 *          must be aligned to a multiple of 64.
 * return: ptr to ByteArray, on error return NULL */
ByteArray*
bytearray_create_from_mem(const uint64_t size, uint8_t* mem);

/* usage: Release a struct ByteArray
 * params:
 *      1) b: ptr to a struct ByteArray
 *      2) free_block: if true, release the memory blocking storing the bytes
 * return: void */
void
bytearray_free(ByteArray* const b, bool free_block);

/* usage: Given a ByteArray, return its size
 * params:
 *      1) b: ptr to struct ByteArray
 * return: number of bytes that the ByteArray can hold */
uint64_t pure_func
bytearray_size(const ByteArray* b);

/* usage: Given a ByteArray, return the number of slots that are used to store
 *      the bytes internally
 * params:
 *      1) b: ptr to struct ByteArray
 * return: number of slots */
uint64_t pure_func
bytearray_snum(const ByteArray* b);

/* usage: Given a ByteArray, return its internal memory block
 * params:
 *      1) b: ptr to struct ByteArray
 * return: ptr to its internal memory block */
uint8_t*
bytearray_memblk(ByteArray* b);

/* usage: Given a ByteArray, clear its internal memory block
 * params:
 *      1) b: ptr to struct ByteArray
 * return: void */
void
bytearray_zero(ByteArray* b);

/* usage: Given a ByteArray, return addr to its i-th byte
 * params:
 *  1) b: ptr to struct ByteArray
 *  2) idx: index of the target byte
 * return: ptr to the target byte */
const uint8_t*
bytearray_addr_at(const ByteArray* b, uint64_t idx);

/* usage: Given a ByteArray, return its i-th byte
 * params:
 *      1) b: ptr to struct ByteArray
 *      2) idx: index of the target byte
 * return: the target byte */
uint8_t
bytearray_at(const ByteArray* b, uint64_t idx);

/* usage: Given a ByteArray, set its i-th byte
 * params:
 *      1) b: ptr to struct ByteArray
 *      2) idx: index of the target byte
 *      3) v: the new value for the byte
 * return: void */
void
bytearray_set_at(const ByteArray* b, uint64_t idx, uint8_t v);

/* usage: Given a ByteArray, count the number of zero bytes
 * params:
 *      1) b: ptr to struct ByteArray
 * return: number of bytes that are zero */
uint32_t
bytearray_cz(const ByteArray* b);

#endif // __BYTEARRAY_H__
