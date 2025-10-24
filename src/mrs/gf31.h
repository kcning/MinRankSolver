#ifndef __GF31_H__
#define __GF31_H__


#include <stdint.h>
#include <stdlib.h>
#include <assert.h>

#define GF31_MIN (0)
#define GF31_MAX (30)

typedef uint8_t gf31_t;

/* ========================================================================
 * function prototypes
 * ======================================================================== */

static inline gf31_t
gf31_t_reduc_compiler_optimized(uint32_t x) {
    // just let compiler optimize this
    gf31_t r = x % 31;
    return r;
}

static inline gf31_t
gf31_t_reduc_hand_optimized(uint32_t v) {
    uint32_t q = (v * 138547332ULL) >> 32;
    uint32_t mod = v - ((q << 5) - q); // mod = v - q * 31
    return mod < 31 ? mod : mod - 31;
}

static inline gf31_t
gf31_t_reduc_mqdss(uint32_t x) {
    assert(x <= UINT16_MAX);
    // TODO: this can be further optimized using instruction level parallelism
    gf31_t t = x & 31;
    x >>= 5;
    t += x & 31;
    x >>= 5;
    t += x & 31;
    x >>= 5;
    t += x & 31;
    x >>= 5;
    t = (t >> 5) + (t & 31);
    t = (t >> 5) + (t & 31);
    return (t != 31)*t;
}

static inline gf31_t
gf31_t_reduc_mqdss_opt(uint32_t x) {
    x = (x & 3253763103U) + ((x & 1041204192U) >> 5);
    x += x >> 20;
    x += x >> 10;
    x = (x & 31) + ((x >> 5) & 31);
    return x >= 31 ? x - 31: x;
}

static inline gf31_t
gf31_t_reduc(uint32_t v) {
    //return gf31_t_reduc_hand_optimized(v);
    return gf31_t_reduc_compiler_optimized(v);
}

static inline gf31_t
gf31_t_rand(void) {
    return gf31_t_reduc(rand() & UINT16_MAX);
}

static inline void
gf31_t_arr_rand(gf31_t* buf, const uint64_t size) {
    // TODO: optimize if needed
    for(uint64_t i = 0; i < size; ++i) {
        buf[i] = gf31_t_rand();
    }
}

static inline gf31_t
gf31_t_add(const gf31_t a, const gf31_t b) {
    return gf31_t_reduc(a + b); // not possible to overflow
}

static inline gf31_t
gf31_t_mul(const gf31_t a, const gf31_t b) {
    uint32_t p = ((uint32_t) a) * b;
    return gf31_t_reduc(p);
}

static inline gf31_t
gf31_t_sub(const gf31_t a, const gf31_t b) {
    // TODO: check what happens when the result is negative
    return gf31_t_reduc(a - b);
}

gf31_t
gf31_t_inv_by_table(gf31_t a);

static inline gf31_t
gf31_t_inv_by_squaring(gf31_t a) {
    if(a == 0)
        return 0;

    if(a == 1)
        return 1;

    // raise a to the power of (31-2)
    uint32_t a32b = a;
    uint32_t p2 = gf31_t_reduc(a32b * a32b); // a^2
    uint32_t p4 = gf31_t_reduc(p2 * p2); // a^4
    uint32_t p8 = gf31_t_reduc(p4 * p4); // a^8
    uint32_t p16 = gf31_t_reduc(p8 * p8); // a^16
    uint32_t p24 = gf31_t_reduc(p16 * p8); // a^24
    uint32_t p28 = gf31_t_reduc(p24 * p4); // a^28
    uint32_t p29 = gf31_t_reduc(p28 * a32b); // a^29
    return p29;
}

static inline gf31_t
gf31_t_inv(gf31_t a) {
    //return gf31_t_inv_by_squaring(a);
    return gf31_t_inv_by_table(a);
}

void
gf31_t_arr_muli_scalar(gf31_t* arr, uint32_t sz, gf31_t x);

void
gf31_t_arr_fmaddi_scalar(gf31_t* a, const gf31_t* b, uint32_t sz, gf31_t c);

void
gf31_t_arr_fmaddi_scalar_mask64(gf31_t* a, const gf31_t* b, gf31_t c, uint64_t d);

void
gf31_t_arr_fmsubi_scalar(gf31_t* a, const gf31_t* b, uint32_t sz, gf31_t c);

void
gf31_t_arr_fmsubi_scalar_mask64(gf31_t* a, const gf31_t* b, gf31_t c, uint64_t d);

uint32_t
gf31_t_arr_nzc(const gf31_t* a, uint32_t sz);

uint32_t
gf31_t_arr_zc(const gf31_t* a, uint32_t sz);

// TODO: create versions of functions above that modify a in place

#endif // __GF31_H__
