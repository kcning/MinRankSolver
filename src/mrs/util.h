/* util.h: header file for common utilities */

#ifndef __BLK_LANCZOS_UTIL_H__
#define __BLK_LANCZOS_UTIL_H__

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <sys/types.h>  /* getpid() */
#include <unistd.h>
#include <assert.h>

#ifdef __GNUC__

#define likely(x) \
    __builtin_expect(!!(x), 1)

#define unlikely(x) \
    __builtin_expect(!!(x), 0)

#define pure_func __attribute__((const))

#define no_opt __attribute__((optimize("O0")))

#define force_inline inline __attribute__((always_inline))

#else

#define likely(x) \
    (x)

#define unlikely(x) \
    (x)

#define pure_func

#define no_opt

#define force_inline inline

#endif

/* for printf and sscanf specifiers for types in stdint.h */
#define __STDC_FORMAT_MACROS
#include <inttypes.h>

#define KBFLOAT         (1024.0)
#define MBFLOAT         (1024.0 * 1024.0)

/* get timestamp to the precision of miliseconds since the program starts */
double
get_timestamp(void);

/* print msg with timestamp */
#define printf_ts(format, ...) do { \
    flockfile(stdout); \
    printf("%12.2f - ", get_timestamp()); \
    printf(format, ##__VA_ARGS__); \
    fflush(stdout); \
    funlockfile(stdout); \
} while(0)

/* print error msg to stderr */
#define printf_err(format, ...) do { \
    flockfile(stderr); \
    fprintf(stderr, format, ##__VA_ARGS__); \
    fflush(stderr); \
    funlockfile(stderr); \
} while(0)

/* print error msg with timestamp to stderr */
#define printf_err_ts(format, ...) do { \
    flockfile(stderr); \
    fprintf(stderr, "%12.2f - ", get_timestamp()); \
    printf_err(format, ##__VA_ARGS__); \
    funlockfile(stderr); \
} while(0)

/* print error msg with timestamp to stderr then exit */
#define exit_with_msg(format, ...) do { \
    printf_err_ts(format, ##__VA_ARGS__); \
    exit(-1); \
} while (0)

/* print msg with timestamp to stderr if in debug mode */
#ifndef NDEBUG

#define printf_debug(format, ...) \
    printf_err_ts(format, ##__VA_ARGS__)

#else

#define printf_debug(...)

#endif

/* print array to stderr if in debug mode */
#ifndef NDEBUG

#define print_array_debug(ele_format, array, size) do { \
    unsigned int i; \
    fprintf(stderr, "%12.2f - array " #array ": ", get_timestamp()); \
    for(i = 0; i < (size); i++) { \
        fprintf(stderr, ele_format, (array)[i]); \
    } \
    fprintf(stderr, "\n"); \
    fflush(stderr); \
} while (0)

#else

#define print_array_debug(...)

#endif

/* usage: read CPU frequency of the platform. Only Linux is supported.
 * params: void
 * return: CPU frequency and -1 on error */
double
get_cpu_freq(void);

/* usage: read the following info from /proc/$PID/status
 *      VmPeak          peak virtual memory size
 *      VmSize          total program size
 *      VmLck           locked memory size
 *      VmHWM           peak resident set size ("high water mark")
 *      VmRSS           size of memory portions
 *      VmData          size of data, stack, and text segments
 *      VmStk           size of data, stack, and text segments
 *      VmExe           size of text segment
 *      VmLib           size of shared library code
 *      VmPTE           size of page table entries
 *      VmSwap          size of swap usage (the number of referred swapents)
 * params:
 *      1) proc_info: a buffer to hold the info.
 *      2) size: size of the buffer
 * return: 0 if success; non-zero value on error */
int
get_proc_status(char* const proc_info, size_t size);


/* usage: return the number of CPU cores. Only Linux is supported
 * return: number of physical CPU cores */
static inline uint32_t
get_cpu_core_count() {
    return sysconf(_SC_NPROCESSORS_ONLN);
}

/* usage: comparator for qsort and bsearch; compare 2 uint32_t
 * params:
 *      1) a: ptr to uint32_t va
 *      2) b: ptr to uint32_t vb
 * return: 1 if va > vb, 0 if va = vb, -1 if va < vb */
static inline int
cmp_uint(const void* a, const void* b) {
    // NOTE: cannot simply subtracting as follows because they are unsigned
    // (*(uint32_t*) a) - (*(uint32_t*) b);
    const uint32_t va = *(const uint32_t*) a;
    const uint32_t vb = *(const uint32_t*) b;
    if(va < vb) {
        return -1;
    }
    else if(va > vb) {
        return 1;
    } else {
        return 0;
    }
}

/* usage: comparator for qsort and bsearch; compare 2 uint64_t
 * params:
 *      1) a: ptr to uint64_t va
 *      2) b: ptr to uint64_t vb
 * return: 1 if va > vb, 0 if va = vb, -1 if va < vb */
static inline int
cmp_uint64(const void* a, const void* b) {
    // NOTE: cannot simply subtracting as follows because they are unsigned
    // (*(uint64_t*) a) - (*(uint64_t*) b);
    const uint64_t va = *(const uint64_t*) a;
    const uint64_t vb = *(const uint64_t*) b;
    if(va < vb) {
        return -1;
    }
    else if(va > vb) {
        return 1;
    } else {
        return 0;
    }
}

/* usage: Given 2 integers i and x, round i to the next multiple of x
 * params:
 *      1) i: the 1st integer
 *      2) x: the 2nd integer
 * return: the rounded value */
static inline uint64_t pure_func
round_up_multiple(uint64_t i, uint64_t x) {
    return ((i + x - 1) / x) * x;
}

static inline uint64_t pure_func
round_down_multiple_4(uint64_t x) {
    return x & ~0x3ULL;
}

/* usage: return the number of CPU cycles since reset
 * params: void
 * return: the number of cycles as a uint64_t */
static inline uint64_t
rdtsc(void) {
    uint64_t result;
     __asm__ volatile ("rdtsc; shlq $32,%%rdx; orq %%rdx,%%rax"
                       : "=a" (result) : : "%rdx");

    return result;
}

/* usage: measure the overhead of measuring CPU cycles
 * params: void
 * return: the overhead in CPU cycles */
uint64_t
rdtsc_overhead(void);

/* usage: given an integer, convert it into a binary string
 * params:
 *      1) buf: char buffer for storing the output
 *      2) size: length of the binary string (zero padded). The buffer must
 *              hold at least (size+1) chars.  Otherwise the function has
 *              undefined behavior
 *      3) n: the integer
 * return: void */
void
itoa(char* buf, uint64_t size, uint64_t n);

/* usage: find the index of an integer in a sorted integer array
 * params:
 *      1) hay: a sorted integer array
 *      2) sz: size of the integer array
 *      3) needle: target integer
 * return: index of the target integer in the array. sz if not found */
uint32_t
uint32_find_in_arr(const uint32_t* hay, uint32_t sz, uint32_t needle);

/* usage: Given a map from integet set 0 ~ sz-1 to 0 ~ total_range-1, create
 *      a reverse map from 0 ~ total_range-1 to 0 ~ sz-1
 * params:
 *      1) mmap: the map. Should have size sz
 *      2) sz: size of the preimage of the map
 *      3) total_range: size of the image of the map
 * return: a dynamically allocated array storing the reverse map */
uint32_t*
uint32_arr_create_reverse_map(const uint32_t* mmap, uint32_t sz,
                              uint32_t total_range);

/* usage: Given a map from integet set 0 ~ sz-1 to 0 ~ total_range-1, create
 *      a reverse map from 0 ~ total_range-1 to 0 ~ sz-1
 * params:
 *      1) mmap: the map. Should have size sz
 *      2) sz: size of the preimage of the map
 *      3) total_range: size of the image of the map
 * return: a dynamically allocated array storing the reverse map */
uint64_t*
uint64_arr_create_reverse_map(const uint64_t* mmap, uint64_t sz,
                              uint64_t total_range);

/* usage: Given a 64-bit integer, count the number of bits that are 1.
 * params;
 *      v: a 64-bit integer
 * return: number of set bits in v */
static inline uint64_t
uint64_popcount(uint64_t v) {
    return __builtin_popcountll(v);
}

/* usage: Given a string, count the number of integers that appear
 * params:
 *      1) str: ptr to char pointint to the string
 * return: number of integers */
uint32_t
count_int_in_str(const char str[]);

/* usage: extend the LSB to a full 8-bit integer. I.e. 1 becomes
 *      UINT8_MAX, while 0 remains 0.
 * params:
 *      1) b: 8-bit integer holding the LSB. Must be either 1 or 0
 * return: the result */
static inline uint8_t
uint8_extend_from_lsb(uint8_t b) {
    assert(b == 0 || b == 1);
    return ~b + 1U;
}

/* usage: extend the LSB to a full 64-bit integer. I.e. 1 becomes
 *      UINT64_MAX, while 0 remains 0.
 * params:
 *      1) b: 64-bit integer holding the LSB. Must be either 1 or 0
 * return: the result */
static inline uint64_t
uint64_extend_from_lsb(uint64_t b) {
    assert(b == 0 || b == 1);
    return ~b + 1ULL;
}

/* usage: extend any non-zero integer to UINT64_MAX, while 0 remains 0.
 * params:
 *      1) v: 64-bit integer
 * return: the result */
static inline uint64_t
uint64_extend_nz(uint64_t v) {
    // NOTE: should translate to cmov
    return v ? UINT64_MAX : 0;
}

/* usage: generate a random 64-bit integer
 * params: none
 * return: a random 64-bit integer */
uint64_t
uint64_rand(void);

/* usage: Given 1 uint64_t a, return the number of trailing zeros in a from
 *      LSB. If a is 0, the result is undefined
 * params:
 *      1) a: uint64_t
 * return: the number of trailing zeros from LSB. undefined if a is 0 */
#define uint64_t_ctz(a) \
    __builtin_ctzll(a)

static inline uint64_t
uint64_t_at(uint64_t* a, uint32_t i) {
    uint64_t v = *a;
    return (v >> i) & 0x1ULL;
}

#endif /* __BLK_LANCZOS_UTIL_H__ */
