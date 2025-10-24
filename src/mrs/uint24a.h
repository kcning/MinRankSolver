/* uint24a.h: header file for struct uint24a */

#ifndef __BLK_LANCZOS_UINT24A_H__
#define __BLK_LANCZOS_UINT24A_H__

#include <stdint.h>

/* ========================================================================
 * struct uint24a definition
 * ======================================================================== */

#define UINT24_MAX 0xffffffULL
#define UINT24_MIN 0x0ULL

typedef struct uint24a uint24a;

/* ========================================================================
 * function prototypes
 * ======================================================================== */

/* usage: Given the number of uint24_t to store in the array, compute the
 *      number of uint8_t needed for storage
 * params:
 *      n: number of uint24_t to store
 * return: number of uint8_t needed */
uint64_t
uint24a_calc_slotnum(const uint64_t n);

/* usage: Given the number of uint24_t to store in the array, compute the
 *      number of bytes needed for storage
 * params:
 *      n: number of uint24_t to store
 * return: number of bytes needed */
uint64_t
uint24a_memsize(const uint64_t n);

/* usage: Given the number of uint24_t to store in the array (i.e. the array
 *      size), create the array.
 * params:
 *      1) n: number of uint24_t to store
 * return: a ptr to struct uint24a. On error, return NULL. */
uint24a*
uint24a_create(const uint64_t n);

/* usage: Release a struct uint24a
 * params:
 *      1) a: ptr to struct uint24a
 * return: void */
void
uint24a_free(uint24a* a);

/* usage: set a struct uint24a to full zero
 * params:
 *      1) arr: ptr to struct uint24a
 * return: void */
void
uint24a_zero(uint24a* a);

/* usage: set a struct uint24a to full 1's
 * params:
 *      1) arr: ptr to struct uint24a
 * return: void */
void
uint24a_max(uint24a* a);

/* usage:Given a uint24a and an index i, return the address to the i-th uint24
 * params:
 *      1) arr: ptr to struct uint24a
 *      2) i: the index of the element whose addr to return
 * return: the address as a uint8_t ptr */
uint8_t*
uint24a_addr(uint24a* arr, uint64_t i);

/* usage: Given a uint24a and an index i, return the i-th uint24 as a uint32_t
 * params:
 *      1) arr: ptr to struct uint24a
 *      2) i: the index of the element
 * return: the i-th element */
uint32_t
uint24a_at(const uint24a* const arr, const uint64_t i);

/* usage: Given a uint24a and an index i, return i-th ~ (i+63)-th uint24 as
 *      an array of uint32_t
 * params:
 *      1) arr: ptr to struct uint24a
 *      2) i: the starting index
 *      3) dst: container for the results; must hold at least 64 elements
 * return: void */
void
uint24a_at_grp64(const uint24a* const restrict arr, const uint64_t i,
                 uint32_t* const restrict dst);

/* usage: Given a uint24a and an index i, store the input v as the i-th uint24
 * params:
 *      1) arr: ptr to struct uint24a
 *      2) i: the index of the element
 *      3) v: a uint32_t integer less than 2^24
 * return: void */
void
uint24a_set_at(uint24a* const arr, const uint64_t i, const uint32_t v);

/* usage: Given a slice of a uint24a, and the number of elements in the slide
 *      set all elements in the slice to zero
 * params:
 *      1) s: a uint8_t ptr to the start of the slice
 *      2) n: number of elements in the slice
 * return: the void */
void
uint24a_slice_zero(uint8_t* s, uint64_t n);

/* usage: Given a slice of a uint24a, and an index i, return the address to
 *      the i-th uint24 in the slice
 * params:
 *      1) s: a uint8_t ptr to the start of the slice
 *      2) i: the index of the element
 * return: the address as a uint8_t ptr */
uint8_t*
uint24a_slice_addr(uint8_t* s, uint64_t i);

/* usage: Given a slice of a uint24a, and an index i, return the i-th uint24
 *      in the slice as a uint32_t
 * params:
 *      1) s: a uint8_t ptr to the start of the slice
 *      2) i: the index of the element
 * return: the i-th element */
uint32_t
uint24a_slice_at(const uint8_t* s, uint64_t i);

/* usage: Given a slice of a uint24a and an index i, store the input v as the
 *      i-th uint24 in the slice.
 * params:
 *      1) s: a uint8_t ptr to the start of the slice
 *      2) i: the index of the element
 *      3) v: a uint32_t integer less than 2^24
 * return: void */
void
uint24a_slice_set_at(uint8_t* s, uint64_t i, uint32_t v);

/* usage: Given a slice of uint24a and an index i, return i-th ~ (i+4)-th uint24
 *      in the slice as an array of uint32_t
 * params:
 *      1) s: a uint8_t ptr to the start of the slice
 *      2) i: the starting index
 *      3) dst: container for the results; must hold at least 4 elements
 * return: void */
void
uint24a_slice_at_grp4(const uint8_t* s, uint64_t i, uint32_t* dst);

/* usage: Given a slice of uint24a and an index i, return i-th ~ (i+63)-th uint24
 *      in the slice as an array of uint32_t
 * params:
 *      1) s: a uint8_t ptr to the start of the slice
 *      2) i: the starting index
 *      3) dst: container for the results; must hold at least 64 elements
 * return: void */
void
uint24a_slice_at_grp64(const uint8_t* const restrict s, const uint64_t i,
                       uint32_t* const restrict dst);

#endif /* __BLK_LANCZOS_UINT24A_H__ */
