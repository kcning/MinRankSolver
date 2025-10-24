/* math_util.h: header file for common math funcs */

#ifndef __BLK_LANCZOS_MATH_UTIL_H
#define __BLK_LANCZOS_MATH_UTIL_H

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <assert.h>

#include "util.h"
#include "uint24a.h"

/* usage: Compute binominal coefficient
 * params:
 *      1) n: the size of the set
 *      2) k: the size of the subset
 * return: the binominal coefficient C(n, k) */
uint64_t pure_func
binom(uint32_t n, uint32_t k);

/* usage: Compute the sum of binominal coefficients
 * params:
 *      1) n: the size of the set
 *      2) k: the size of the subset
 * return: the sum of binominal coefficients C(n, i), for i in 0...k */
uint64_t pure_func
sum_binom(uint32_t n, uint32_t k);

/* binomial coefficients */
static inline uint64_t pure_func
binom2(uint64_t n) {
    n = (n * (n - 1)) / 2;
    assert(n < UINT32_MAX);
    return n;
}

static inline uint64_t pure_func
binom3(uint64_t n) {
    n = binom2(n) * (n - 2) / 3;
    assert(n < UINT32_MAX);
    return n;
}

static inline uint64_t pure_func
binom4(uint64_t n) {
    n = binom3(n) * (n - 3) / 4;
    assert(n < UINT32_MAX);
    return n;
}

static inline uint64_t pure_func
binom5(uint64_t n) {
    n = binom4(n) * (n - 4) / 5;
    assert(n < UINT32_MAX);
    return n;
}

/* usage: Given 1 integer, check if it's a power of 2
 * params:
 *      1) i: integer
 * return: true if so, otherwise false */
bool pure_func
is_power_of_2(const uint64_t i);

/* usage: Given 1 integer, round it up to the next power of 2
 * params:
 *      1) i: integer
 * return: the next power of 2 */
uint64_t pure_func
next_power_of_2(uint64_t i);

/* usage: Given an array of uint32_t, compute its average
 * params:
 *      1) a: an array of uint32_t
 *      2) size: number of elements in array a
 * return: the average as double */
double
uint32_t_avg(const uint32_t* const a, const size_t size);

/* usage: Given an array of uint32_t, compute its standard deviation
 * params:
 *      1) a: an array of uint32_t
 *      2) size: number of elements in array a
 *      3) avg: average of the array a
 * return: the standard deviation as double */
double
uint32_t_std(const uint32_t* const a, const size_t size, const double avg);

/* usage: Given an array of uint64_t, compute its average
 * params:
 *      1) a: an array of uint64_t
 *      2) size: number of elements in array a
 * return: the average as double */
double
uint64_t_avg(const uint64_t* const a, const size_t size);

/* usage: Given an array of uint64_t, compute its standard deviation
 * params:
 *      1) a: an array of uint64_t
 *      2) size: number of elements in array a
 *      3) avg: average of the array a
 * return: the standard deviation as double */
double
uint64_t_std(const uint64_t* const a, const size_t size, const double avg);

/* usage: Given an array of uint64_t, sort the array and compute its median
 * params:
 *      1) a: an array of uint64_t
 *      2) size: number of elements in array a
 * return: the median as uint64_t */
uint64_t
uint64_t_med(uint64_t* const a, const size_t size);

/* usage: Compute the column index for the monomial xixj... (at most 5 vars)
 *      according to grlex.
 * params:
 *      1) vnum: number of variables
 *      2) mvnum: number of variables in the monomial, at most 5
 *      3) idxs: a sorted array of ints representing the monomial, e.g. for
 *              x1x3x4, one must pass { 0, 2, 3 }
 * return: the index of the monomial */
static inline uint32_t
midx(const uint32_t vnum, const uint32_t mvnum, const uint32_t idxs[]) {
    assert(mvnum <= vnum);
    uint64_t r = 0;
    switch(mvnum) {
        case 5:
            r += binom5(vnum) - binom5(idxs[4]);
            // fall through
        case 4:
            r += binom4(vnum) - binom4(idxs[3]);
            // fall through
        case 3:
            r += binom3(vnum) - binom3(idxs[2]);
            // fall through
        case 2:
            r += binom2(vnum) - binom2(idxs[1]);
            // fall through
        case 1:
            r += vnum - idxs[0];
    }

#ifdef BLK_LANCZOS_UINT24_MAC
    assert(r < UINT24_MAX);
#else
    assert(r < UINT32_MAX);
#endif
    return r;
}

/* usage: Given the number of samples n, a minimal min, and the range of the
 *      samples r, take n distinct uint32_t random samples min <= x <= r-1
 *      and store them into the given container.
 * params:
 *      1) dst: an uint32_t array for storing the samples. Must hold at least n
 *              elements
 *      2) n: number of samples to take
 *      3) min: uint32_t; the minimal value for all samples
 *      4) r: range of the samples; At most UINT32_MAX+1
 *      5) used: a bool array whose size is at least r. Used as a computation
 *              buffer
 * return: void */
void
uint64_t_min_rsamp(uint32_t* const restrict dst, uint64_t n, uint64_t min,
                   uint64_t r, bool* const restrict used);

/* usage: Given the number of samples n, and the range of the samples r,
 *      take n distinct uint32_t random samples within 0 <= x <= r-1 and store
 *      them into the given container.
 * params:
 *      1) dst: an uint32_t array for storing the samples. Must hold at least n
 *              elements
 *      2) n: number of samples to take
 *      3) r: range of the samples; At most UINT32_MAX+1
 *      4) used: a bool array whose size is at least r. Used as a computation
 *              buffer
 * return: void */
static inline void
uint64_t_rsamp(uint32_t* const restrict dst, uint64_t n, uint64_t r,
               bool* const restrict used) {
    uint64_t_min_rsamp(dst, n, 0, r, used);
}

/* usage: Given 4 uint64_t, find the max amongst them
 * params:
 *      1) i0: the 1st uint64_t
 *      2) i1: the 2nd uint64_t
 *      3) i2: the 3rd uint64_t
 *      4) i4: the 4th uint64_t
 * return: the max as a uint64_t */
static inline uint64_t pure_func
uint64_t_max_of_4(uint64_t i0, uint64_t i1, uint64_t i2, uint64_t i3) {
    uint64_t max = i0;
    if(i1 > max) max = i1;
    if(i2 > max) max = i2;
    if(i3 > max) max = i3;
    return max;
}

/* usage: Given 4 uint64_t, find the min amongst them
 * params:
 *      1) i0: the 1st uint64_t
 *      2) i1: the 2nd uint64_t
 *      3) i2: the 3rd uint64_t
 *      4) i4: the 4th uint64_t
 * return: the min as a uint64_t */
static inline uint64_t pure_func
uint64_t_min_of_4(uint64_t i0, uint64_t i1, uint64_t i2, uint64_t i3) {
    uint64_t min = i0;
    if(i1 < min) min = i1;
    if(i2 < min) min = i2;
    if(i3 < min) min = i3;
    return min;
}

/* usage: Given an array, compute the sum of its elements
 * params:
 *      1) a: ptr to the array. Types can be uint32_t, uint64_t, float, double
 *      2) sz: size of the array
 * return: sum. The return type is the same as the array type */
#define sum_arr(a, sz) _Generic((a), \
    uint32_t *: sum_arr_u32, \
    uint64_t *: sum_arr_u64, \
    const uint32_t *: sum_arr_u32, \
    const uint64_t *: sum_arr_u64, \
    float *: sum_arr_float, \
    const float *: sum_arr_float, \
    double *: sum_arr_double, \
    const double *: sum_arr_double \
)(a, sz)

static inline uint32_t
sum_arr_u32(const uint32_t* a, uint64_t sz) {
    uint32_t sum = 0;
    for(uint64_t i = 0; i < sz; ++i)
        sum += a[i];

    return sum;
}

static inline uint64_t
sum_arr_u64(const uint64_t* a, uint64_t sz) {
    uint64_t sum = 0;
    for(uint64_t i = 0; i < sz; ++i)
        sum += a[i];

    return sum;
}

static inline float
sum_arr_float(const float* a, uint64_t sz) {
    float sum = 0.0;
    for(uint64_t i = 0; i < sz; ++i)
        sum += a[i];

    return sum;
}

static inline double
sum_arr_double(const double* a, uint64_t sz) {
    double sum = 0.0;
    for(uint64_t i = 0; i < sz; ++i)
        sum += a[i];

    return sum;
}

/* usage: given 2 ptrs to uint32_t, compare the integers they point to. Used for qsort
 * params:
 *      1) a: ptr to the 1st uint32_t as a generic ptr
 *      2) b: ptr to the 2nd uint32_t as a generic ptr
 * return: < 0 if *a < *b. 0 if *a == *b. > 0 if *a > *b */
static inline int
uint32_t_cmp(const void* a, const void* b) {
    uint32_t va = *((uint32_t*) a);
    uint32_t vb = *((uint32_t*) b);
    return va - vb;
}

#endif /* __BLK_LANCZOS_MATH_UTIL_H */
