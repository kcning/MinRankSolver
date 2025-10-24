/* hmap.c: implementation of hmap.h */

/* This hash map is optimized for deduplication, therefore it does not
 * implement functions for removing entries. With the guarantee of not removing
 * any entry, operations can be implemented more efficiently. */

#include "hmap.h"
#include "util.h"
#include "blake2s.h"

#include <stdlib.h>
#include <string.h>
#include <assert.h>

/* ========================================================================
 * struct Hmap definition
 * ======================================================================== */

#define HMAP_BIN_NUM (0x1ULL << 16)

struct HmapEntry {
    // TODO: save space by only storing part of the hash not used for indexing
    // into the bin
    uint8_t s[HMAP_HASH_LEN];
    const void* ptr; // should not be used to modify the stored data
};

struct Hmap {
    uint64_t size; // max number of entries that can be stored
    uint64_t bsize; // max number of entries that a bin can hold
    uint64_t cur_size; // current number of stored entries
    HmapEntry hs[];
};

/* ========================================================================
 * function implementations
 * ======================================================================== */

/* usage: Given a ptr to a HmapEntry, return a ptr to its hash value
 * params:
 *      1) e: ptr to a struct HmapEntry
 * return: a uint8_t to the hash value of the entry */
const uint8_t*
hentry_hash(const HmapEntry* e) {
    return e->s;
}

/* usage: Given a ptr to a HmapEntry, return a ptr to the data stored in the
 *      entry
 * params:
 *      1) e: ptr to a struct HmapEntry
 * return: a void ptr to the data stored in the entry */
const void*
hentry_data(const HmapEntry* e) {
    return e->ptr;
}

/* usage: Given a ptr to a HmapEntry and an index i, return the i-th byte of
 *      its hash value
 * params:
 *      1) e: ptr to a struct HmapEntry
 *      2) idx: the index, which is between 0 ~ HMAP_HASH_LEN-1
 * return: the i-th byte of its hash value as a uint8_t */
static inline uint8_t
hentry_byte_at(const HmapEntry* const e, const uint64_t idx) {
    assert(idx < HMAP_HASH_LEN);
    return hentry_hash(e)[idx];
}

/* usage: Given a ptr to a hash value, compute the index of its bin
 * params:
 *      1) hv: ptr to a a hash value, which is an uint8_t array of at least
 *              HMAP_HASH_LEN elements
 * return: the bin index as a uint64_t number */
static inline uint64_t
hash_bin_idx(const uint8_t* const hv) {
    // TODO: fine tune the algorithm of computing the bin index
    //const size_t digest_sz = 2;
    //uint8_t digest[digest_sz];
    //blake2s(digest, hv, NULL, digest_sz, HMAP_HASH_LEN, 0);
    //uint64_t idx = *((uint32_t*) digest);
    //uint64_t idx = hv[0] ^ hv[1] ^ hv[2] ^ hv[3];
    uint64_t idx = *((uint16_t*) hv);
    assert(idx < HMAP_BIN_NUM);
    return idx;
}

/* usage: Given a ptr to a HmapEntry, compute the index of its bin
 * params:
 *      1) e: ptr to a struct HmapEntry
 * return: the bin index as a uint64_t number */
static inline uint64_t
hentry_bin_idx(const HmapEntry* const e) {
    return hash_bin_idx(hentry_hash(e));
}

/* usage: Given a ptr to a Hmap, return the max number of entries
 * params:
 *      1) h: ptr to struct Hmap
 * return: the number as a uint64_t */
uint64_t
hmap_size(const Hmap* h) {
    return h->size;
}

/* usage: Given a ptr to a Hmap, return the max number of entries in a bin
 * params:
 *      1) h: ptr to struct Hmap
 * return: the number as a uint64_t */
static inline uint64_t
hmap_bsize(const Hmap* h) {
    return h->bsize;
}

/* usage: Given a ptr to a Hmap, return the current number of stored entries
 * params:
 *      1) h: ptr to struct Hmap
 * return: the number as a uint64_t */
uint64_t
hmap_cur_size(const Hmap* h) {
    return h->cur_size;
}

/* usage: Given a ptr to a Hmap, and index i, return a pointer to the
 *      i-th slot in the internal array for storing HmapEntry's.
 *      Note that if return HmapEntry's hash value is full zero, then the entry
 *      is not used.
 * params:
 *      1) h: ptr to struct Hmap
 *      2) i: the index
 * return: ptr to HmapEntry */
const HmapEntry*
hmap_entry_at(const Hmap* h, uint64_t i) {
    return h->hs + i;
}

/* usage: Given the number of hashes to store, create a Hmap
 * params:
 *      1) size: number of hashes to store
 * return: ptr to Hmap, on error return NULL */
Hmap*
hmap_create(uint64_t size) {
    // round up to the nearest multiple of HMAP_BIN_NUM
    size = round_up_multiple(size, HMAP_BIN_NUM);

    // init to full zero
    Hmap* h = calloc(1, sizeof(Hmap) + sizeof(HmapEntry) * size);
    if(!h) {
        return NULL;
    }

    h->size = size;
    h->bsize = size / HMAP_BIN_NUM;
    h->cur_size = 0;

    return h;
}

/* usage: Release a struct Hmap
 * params:
 *      1) h: ptr to a struct Hmap
 * return: void */
void
hmap_free(Hmap* const h) {
    free(h);
}

/* usage: Reset a struct Hmap
 * params;
 *      1) h: ptr to a struct Hmap
 * return: void */
void
hmap_reset(Hmap* const h) {
    assert(h);
    memset(h->hs, 0x0, sizeof(HmapEntry) * hmap_size(h));
    h->cur_size = 0;
}

/* usage: Given a ptr to a Hmap, and an index of the bin, return
 *      a ptr to an array of HmapEntry, whose length is HMAP_BIN_NUM.
 * params:
 *      1) h: ptr to struct Hmap
 *      2) idx: index of the bin; must be 0 ~ HMAP_BIN_NUM-1
 * return: a ptr of HmapEntry that points to the start of the bin */
static inline HmapEntry*
hmap_get_bin(Hmap* h, uint64_t idx) {
    return h->hs + hmap_bsize(h) * idx;
}

static uint8_t zero[HMAP_HASH_LEN]; // statically init'ed to full zero

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
            const void* const restrict v) {
    assert(h && k);

    HmapEntry* dst = hmap_get_bin(h, hash_bin_idx(k));

    // find the first empty slot in the bin to store the new entry
    for(uint64_t i = 0; i < hmap_bsize(h); ++i, ++dst) {
        if(!memcmp(hentry_hash(dst), zero, sizeof(uint8_t) * HMAP_HASH_LEN)) {
            memcpy(dst->s, k, sizeof(uint8_t) * HMAP_HASH_LEN);
            dst->ptr = v;
            ++h->cur_size;
            return HMAP_INSERT_SUC; // stored
        }

        if(!memcmp(hentry_hash(dst), k, HMAP_HASH_LEN)) { // duplicate
            return HMAP_INSERT_DUP;
        }
    }

    return HMAP_INSERT_FULL; // bin is full
}

/* usage: Given a ptr to Hmap, return the data associated with the given
 *      key
 * params:
 *      1) h: ptr to struct Hmap
 *      2) k: a uint8_t array of size HMAP_HASH_LEN that stores the key
 * return: on success, return the data. If no data is associated with the given
 *      key, return HMAP_GET_NO_MATCH */
void*
hmap_get(const Hmap* restrict h, const uint8_t k[HMAP_HASH_LEN]) {
    const HmapEntry* e = hmap_get_bin((Hmap* )h, hash_bin_idx(k));
    for(uint64_t i = 0; i < hmap_bsize(h); ++i, ++e) {
        if(memcmp(hentry_hash(e), k, sizeof(uint8_t) * HMAP_HASH_LEN)) {
            return (void*) hentry_data(e);
        }

        if(!memcmp(hentry_hash(e), zero, sizeof(uint8_t) * HMAP_HASH_LEN)) {
            break; // end of the bin
        }
    }
    return (void*) HMAP_GET_NO_MATCH;
}

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
              void* restrict args) {
    for(uint64_t i = 0; i < HMAP_BIN_NUM; ++i) {
        HmapEntry* e = hmap_get_bin(h, i);
        for(uint64_t j = 0; j < hmap_bsize(h); ++j, ++e) {
            if(!memcmp(hentry_hash(e), zero, sizeof(uint8_t) * HMAP_HASH_LEN)) {
                break; // end of the bin
            }

            f(e, args); // process the entry with the given function and args
        }
    }
}

/* ========================================================================
 * struct and subroutine for hmap_combine
 * ======================================================================== */

typedef struct {
    Hmap* dst;
#if BLK_LANCZOS_COLLECT_STATS
    uint64_t dup_num;
    uint64_t drop_num;
    uint64_t valid_num;
#endif
} HmapCombineArgs;

static inline void
hmap_combine_f(HmapEntry* e, void* __args) {
    HmapCombineArgs* args = __args;
    Hmap* dst = args->dst;

#ifdef BLK_LANCZOS_COLLECT_STATS
    switch(hmap_insert(dst, hentry_hash(e), hentry_data(e))) {
        case HMAP_INSERT_DUP:
            ++args->dup_num;
            break;
        case HMAP_INSERT_FULL:
            ++args->drop_num;
            break;
        case HMAP_INSERT_SUC:
            ++args->valid_num;
            break;
    }
#else
    hmap_insert(dst, hentry_hash(e), hentry_data(e));
#endif
}

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
             bool reset, uint64_t* const restrict stats) {
#else
             bool reset) {
#endif
    assert(dst && arr);

    if(reset)
        hmap_reset(dst);

    HmapCombineArgs args = {
        .dst = dst,
#ifdef BLK_LANCZOS_COLLECT_STATS
        .dup_num = 0, .drop_num = 0, .valid_num = 0
#endif
    };

    for(uint64_t i = 0; i < size; ++i) {
        assert(arr[i]);
        hmap_for_each((Hmap*) arr[i], hmap_combine_f, &args);
    }

#ifdef BLK_LANCZOS_COLLECT_STATS
    stats[0] = args.valid_num;
    stats[1] = args.dup_num;
    stats[2] = args.drop_num;
#endif
}
