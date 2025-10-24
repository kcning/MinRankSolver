#include "gfa.h"

#include <stdint.h>

/* ========================================================================
 * struct GFA definition
 * ======================================================================== */

// sparse GF array
struct GFA {
    gfa_idx_t size; // number of elements in the array
    gfa_idx_t* restrict e;
};

/* ========================================================================
 * function implementations
 * ======================================================================== */

/* usage: get the number of bits used to store an index in struct GFA
 * return: number of bits */
uint32_t
gfa_size_of_idx(void) {
#if defined(GFA_IDX_SIZE_64)
    return 64 - 8;
#else
    return 32 - 8;
#endif
}

/* usage: get the number of bytes needed to store an element in a struct GFA
 * return: number of bytes */
size_t
gfa_size_of_element(void) {
    return sizeof(gfa_idx_t);
}

/* usage: return the size of struct GFA
 * return: size in bytes */
size_t
gfa_memsize(void) {
    return sizeof(GFA);
}

/* usage: create a struct GFA with a given capacity
 * params:
 *      1) n: number of elements in the GF array
 * return: ptr to struct GFA on success, NULL on error */
GFA*
gfa_create(uint64_t n) {
    GFA* a = malloc(sizeof(GFA));
    if(!a)
        return NULL;

    a->e = malloc(sizeof(gfa_idx_t) * n);
    if(NULL ==  a->e) {
        free(a);
        return NULL;
    }

    a->size = n;
    return a;
}

/* usage: create a struct GFA with a given capacity from a buffer
 * params:
 *      1) n: number of elements in the GF array
 *      2) buf: a gfa_idx_t buffer with size at least n
 * return: ptr to struct GFA on success, NULL on error */
GFA*
gfa_create_from_buf(uint64_t n, gfa_idx_t* buf) {
    if(!buf)
        return NULL;

    GFA* a = malloc(sizeof(GFA));
    if(!a)
        return NULL;
    a->e = buf;
    a->size = n;
    return a;
}

/* usage: Release a struct GFA
 * params:
 *      1) gfa: ptr to struct GFA
 * return: void */
void
gfa_free(GFA* gfa) {
    free(gfa->e);
    free(gfa);
}

/* usage: create an array of struct GFA with a given capacity
 * params:
 *      1) n: number of elements in the GF array
 *      2) len: number of GFA entries in the array
 *      3) buf: a gfa_idx_t buffer with size at least n * len
 * return: ptr to struct GFA on success, NULL on error */
GFA*
gfa_arr_create(uint64_t n, uint64_t len, const gfa_idx_t* buf) {
    GFA* a = malloc(sizeof(GFA) * len);
    if(!a)
        return NULL;

    for(uint64_t i = 0; i < len; ++i) {
        a[i].e = (gfa_idx_t*) buf + n * i;
        a[i].size = n;
    }
    return a;
}

/* usage: create an array of struct GFA where the size and optionally the
 *      content of each GFA is decided by a function
 * params:
 *      1) len: number of GFA entries in the array
 *      2) buf: a gfa_idx_t buffer large enough to hold all elements in the
 *          the array of GFAs
 *      3) arg: a generic pointer which is will be passed to the function
 *          that decides the size of a GFA entry
 *      4) cb: the function that decides the size of a GFA entry. It takes
 *          a uint64_t, a GFA* ptr, and a void* ptr. The 1st parameter is the
 *          index of the GFA entry in the array. The 2nd parameter points to
 *          the the GFA entry being created, while the 3rd parameter is 'arg'.
 *          This function should return the size of the GFA entry, and
 *          optionally initialize the GFA entry
 * return: ptr to struct GFA on success, NULL on error */
GFA*
gfa_arr_create_f(uint64_t len,
                 const gfa_idx_t* restrict buf,
                 void* restrict arg,
                 gfa_idx_t(*cb)(uint64_t, GFA* e, void*)) {
    GFA* a = malloc(sizeof(GFA) * len);
    if(!a)
        return NULL;

    gfa_idx_t offset = 0;
    for(uint64_t i = 0; i < len; ++i) {
        a[i].e = (gfa_idx_t*) buf + offset;
        a[i].size = cb(i, a + i, arg);
        offset += a[i].size;
    }

    return a;
}

/* usage: Release an array of struct GFA
 * params:
 *      1) a: ptr to struct GFA
 * return: void */
void
gfa_arr_free(GFA* a) {
    free(a);
}

/* usage: Given a struct GFA, return the number of elements in it */
gfa_idx_t
gfa_size(const GFA* gfa) {
    return gfa->size;
}

/* usage: Given a struct GFA, set its size to the given value
 * params:
 *      1) a: ptr to struct GFA
 *      2) sz: new size
 * return: void */
void
gfa_set_size(GFA* a, gfa_idx_t sz) {
    a->size = sz;
}

/* usage: Given a struct GFA, increment its size
 * params:
 *      1) a: ptr to struct GFA
 * return: void */
void
gfa_inc_size(GFA* a) {
    ++(a->size);
}

/* usage: Given a struct GFA, return its i-th element
 * params:
 *      1) gfa: ptr to struct GFA
 *      2) i: index of the element in GFA to retrieve
 *      3) idx: ptr to gfa_idx_t, storage for the column index of the element
 * return: value of the element */
gf_t
gfa_at(const GFA* restrict gfa, gfa_idx_t i, gfa_idx_t* restrict idx) {
    assert(i < gfa_size(gfa));
    gfa_idx_t v = gfa->e[i];
    *idx = v >> 8; // upper 24 (or 56) bits store the column index
    return v & 0xFF; // lower 8 bits store the value
}

/* usage: Given a struct GFA, overwrite the i-th element in it with
 *      the given column index and the value
 * params:
 *      1) gfa: ptr to struct GFA
 *      2) i: index of the element in GFA to update
 *      3) idx: column index of the element
 *      4) v: value of the element
 * return: void */
void
gfa_set_at(GFA* gfa, gfa_idx_t i, gfa_idx_t idx, gf_t v) {
    //assert(i < gfa_size(gfa));
    assert(idx <= GFA_IDX_MAX);
    gfa_idx_t r = (idx << 8) | v;
    gfa->e[i] = r;
}

/* usage: Given an array of struct GFA, return its i-th entry
 * params:
 *      1) a: ptr to struct GFA
 *      2) i: index of the entry. Must <= the size of the array
 * return: ptr to the specified struct GFA */
const GFA*
gfa_arr_at(const GFA* a, uint32_t i) {
    return a + i;
}
