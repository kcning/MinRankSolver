/* hmap.h: header file for struct Hmap, a customized hash map */

#ifndef __BLK_LANCZOS_HMAP_H
#define __BLK_LANCZOS_HMAP_H

#include <stdint.h>
#include <stdbool.h>

// TODO: test and adjust the hash length to the minimal necessary one.
// For 2^s inputs chosen independently of the hash, and w-bit of hash
// output, the probability of collision p <= 2^(2 * s - w - 1)
// See https://crypto.stackexchange.com/questions/39641/what-are-the-odds-of-collisions-for-a-hash-function-with-256-bit-output/39644#39644
#define HMAP_HASH_LEN 8

typedef struct HmapEntry HmapEntry;
typedef struct Hmap Hmap;

/* ========================================================================
 * function prototypes
 * ======================================================================== */

/* usage: Given a ptr to a HmapEntry, return a ptr to its hash value
 * params:
 *      1) e: ptr to a struct HmapEntry
 * return: a uint8_t to the hash value of the entry */
const uint8_t*
hentry_hash(const HmapEntry* e);

/* usage: Given a ptr to a HmapEntry, return a ptr to the data stored in the
 *      entry
 * params:
 *      1) e: ptr to a struct HmapEntry
 * return: a void ptr to the data stored in the entry */
const void*
hentry_data(const HmapEntry* e);

/* usage: Given a ptr to a Hmap, return the max number of entries
 * params:
 *      1) h: ptr to struct Hmap
 * return: the number as a uint64_t */
uint64_t
hmap_size(const Hmap* h);

/* usage: Given a ptr to a Hmap, return the current number of stored entries
 * params:
 *      1) h: ptr to struct Hmap
 * return: the number as a uint64_t */
uint64_t
hmap_cur_size(const Hmap* h);

/* usage: Given a ptr to a Hmap, and index i, return a pointer to the
 *      i-th slot in the internal array for storing HmapEntry's.
 *      Note that if return HmapEntry's hash value is full zero, then the entry
 *      is not used.
 * params:
 *      1) h: ptr to struct Hmap
 *      2) i: the index
 * return: ptr to HmapEntry */
const HmapEntry*
hmap_entry_at(const Hmap* h, uint64_t i);

/* usage: Given the number of hashes to store, create a Hmap
 * params:
 *      1) size: number of hashes to store
 * return: ptr to Hmap, on error return NULL */
Hmap*
hmap_create(uint64_t size);

/* usage: Release a struct Hmap
 * params:
 *      1) h: ptr to a struct Hmap
 * return: void */
void
hmap_free(Hmap* const h);

/* usage: Reset a struct Hmap
 * params;
 *      1) h: ptr to a struct Hmap
 * return: void */
void
hmap_reset(Hmap* const h);

/* return code of hmap_insert */
enum HmapInsertCode {
    HMAP_INSERT_SUC = 0,
    HMAP_INSERT_DUP = 1,
    HMAP_INSERT_FULL = -1,
};

/* usage: Given a ptr to Hmap, and a generic ptr (void*) to a data, and its
 *      hash value, store the data as an entry in the Hmap. If the bin for the
 *      given data is full, or if an entry with the same hash value exists,
 *      this function returns without storing the data.
 * params:
 *      1) h: ptr to a struct Hmap
 *      2) k: ptr to a hash value of the data
 *      3) v: ptr to the data
 * return: HMAP_INSERT_SUC if the entry is stored, HMAP_INSERT_DUP if the an
 *      entry with the same hash value exists, and HMAP_INSERT_FULL if the bin
 *      is full */
int
hmap_insert(Hmap* const restrict h, const uint8_t k[HMAP_HASH_LEN],
            const void* const restrict v);

#define HMAP_GET_NO_MATCH  (0xDEADBEEF01234567ULL)

/* usage: Given a ptr to Hmap, return the data associated with the given
 *      key
 * params:
 *      1) h: ptr to struct Hmap
 *      2) k: a uint8_t array of size HMAP_HASH_LEN that stores the key
 * return: on success, return the data. If no data is associated with the given
 *      key, return HMAP_GET_NO_MATCH */
void*
hmap_get(const Hmap* restrict h, const uint8_t k[HMAP_HASH_LEN]);

/* usage: Given a ptr to Hmap, and a function ptr, call the function ptr
 *      with every entry in the Hmap as its parameter
 * params:
 *      1) h: ptr to a struct Hmap
 *      2) f: function pointer with the signature void (*)(HmapEntry*, void*)
 *      3) args: a void ptr used to pass extra parameters into f, and store
 *              return values from f
 * return: void */
void
hmap_for_each(Hmap* const restrict h, void(*f)(HmapEntry*, void*),
              void* restrict args);

/* usage: Given a ptr to Hmap dst, and an array of ptrs of Hmaps, combine the
 *      array of Hmaps into one, and store the result into dst
 * params:
 *      1) dst: ptr to a struct Hmap
 *      2) arr: an array of ptrs to Hmap
 *      3) size: size the the array
 *      4) reset: if true, reset dst before combining
 * return: void */
void
hmap_combine(Hmap* const restrict dst, const Hmap* arr[], uint64_t size,
#ifdef BLK_LANCZOS_COLLECT_STATS
             bool reset, uint64_t* const restrict stats);
#else
             bool reset);
#endif

#endif // __BLK_LANCZOS_HMAP_H
