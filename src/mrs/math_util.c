/* math_util.c: implementation of common math funcs */

#include "math_util.h"
#include "util.h"

#include <math.h>
#include <stdlib.h>

/* usage: Compute binominal coefficient
 * params:
 *      1) n: the size of the set
 *      2) k: the size of the subset
 * return: the binominal coefficient C(n, k) */
uint64_t
binom(uint32_t n, uint32_t k) {
    if(n < k) {
        return 0;
    }

    if(n == k || 0 == k) {
        return 1;
    }

    if((n-k) < k) k = n - k;

    uint64_t res = 1;
    uint64_t i = 0;
    for(i = 0; i < k; ++i) {
        res *= n-i;
        res /= i+1;
    }

    return res;
}

/* usage: Compute the sum of binominal coefficients
 * params:
 *      1) n: the size of the set
 *      2) k: the size of the subset
 * return: the sum of binominal coefficients C(n, i), for i in 0...k */
uint64_t
sum_binom(uint32_t n, uint32_t k) {
    uint64_t sum = 0;

    uint32_t i = 0;
    for(i = 0; i <= k; ++i) {
        sum += binom(n, i);
    }

    return sum;
}

/* usage: Given 1 integer, check if it's a power of 2
 * params:
 *      1) i: integer
 * return: true if so, otherwise false */
bool
is_power_of_2(const uint64_t i) {
    return i && (i & (i-1)) == 0;
}

/* usage: Given 1 integer, round it up to the next power of 2
 * params:
 *      1) i: integer
 * return: the next power of 2 */
uint64_t
next_power_of_2(uint64_t i) {
    --i;
    i |= i >> 1;
    i |= i >> 2;
    i |= i >> 4;
    i |= i >> 8;
    i |= i >> 16;
    i |= i >> 32;
    return ++i;
}

/* usage: Given an array of uint32_t, compute its average
 * params:
 *      1) a: an array of uint32_t
 *      2) size: number of elements in array a
 * return: the average as double */
double
uint32_t_avg(const uint32_t* const a, const size_t size) {
    uint64_t sum = 0;
    for(uint64_t i = 0; i < size; ++i) {
        sum += a[i];
    }
    return ((double) sum / size);
}

/* usage: Given an array of uint32_t, compute its standard deviation
 * params:
 *      1) a: an array of uint32_t
 *      2) size: number of elements in array a
 *      3) avg: average of the array a
 * return: the standard deviation as double */
double
uint32_t_std(const uint32_t* const a, const size_t size, const double avg) {
    double var = 0;

    for(uint64_t i = 0; i < size; ++i) {
        var += pow(a[i] - avg, 2);
    }
    return sqrt(var / size);
}

/* usage: Given an array of uint64_t, compute its average
 * params:
 *      1) a: an array of uint64_t
 *      2) size: number of elements in array a
 * return: the average as double */
double
uint64_t_avg(const uint64_t* const a, const size_t size) {
    uint64_t sum = 0;
    for(uint64_t i = 0; i < size; ++i) {
        sum += a[i];
    }
    return ((double) sum / size);
}

/* usage: Given an array of uint64_t, sort the array and compute its median
 * params:
 *      1) a: an array of uint64_t
 *      2) size: number of elements in array a
 * return: the median as uint64_t */
uint64_t
uint64_t_med(uint64_t* const a, const size_t size) {
    qsort(a, size, sizeof(uint64_t), cmp_uint64);
    if(size & 0x1) {
        return a[size / 2];
    } else {
        return (a[size / 2 - 1] + a[size / 2]) / 2;
    }
}

/* usage: Given an array of uint64_t, compute its standard deviation
 * params:
 *      1) a: an array of uint64_t
 *      2) size: number of elements in array a
 *      3) avg: average of the array a
 * return: the standard deviation as double */
double
uint64_t_std(const uint64_t* const a, const size_t size, const double avg) {
    double var = 0;

    for(uint64_t i = 0; i < size; ++i) {
        var += pow(a[i] - avg, 2);
    }
    return sqrt(var / size);
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
                   uint64_t r, bool* const restrict used) {
    assert(r && (r-1) <= UINT32_MAX);
    assert(min < r);
    assert(n <= (r - min));

    r -= min;
    memset(used, 0x0, sizeof(bool) * r);
    uint64_t chosen_num = 0;
    for(uint64_t i = r - n; i < r && chosen_num < n; ++i) {
        uint32_t sample = rand() % (i + 1);

        if(used[sample])
            sample = i;

        assert(!used[sample]);
        dst[chosen_num++] = min + sample;
        used[sample] = true;
    }

    assert(chosen_num == n);
}

