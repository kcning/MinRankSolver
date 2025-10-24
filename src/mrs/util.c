/* util.c: implementation for util.h */

#include "util.h"
#include <ctype.h>
#include <stdlib.h>

/* convert timeval to miliseconds */
#define TIMEVAL2F(stamp) \
    ((stamp).tv_sec * 1000.0 + (stamp).tv_usec / 1000.0)

/* get timestamp to the precision of miliseconds since the program starts */
double
get_timestamp(void) {
    static double __init_stamp = -1;
    static struct timeval __cur_time;

    if(-1 == __init_stamp) {
        gettimeofday(&__cur_time, NULL);
        __init_stamp = TIMEVAL2F(__cur_time);
    }

    gettimeofday(&__cur_time, NULL);
    return ((TIMEVAL2F(__cur_time) - __init_stamp) / 1000.0);
}

#undef TIMEVAL2F

/* usage: read CPU frequency of the platform. Only Linux is supported.
 * params: void
 * return: CPU frequency and -1 on error
 */
double
get_cpu_freq(void) {
    static char* cpu_freq_file = "/sys/devices/system/cpu/cpu0/cpufreq"
        "/cpuinfo_max_freq";  /* read the max freq */
    FILE* fp = NULL;
    char buf[1024];  /* should be less than 1024 bytes */

    if(NULL == (fp = fopen(cpu_freq_file, "r"))) {
        return -1;
    }

    if(NULL == fgets(buf, 1024, fp)) {
        fclose(fp);
        return -1;
    }

    fclose(fp);
    return atof(buf);
}

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
get_proc_status(char* proc_info, size_t size) {
    char buf[1024];
    snprintf(buf, 1024, "/proc/%d/status", getpid());

    FILE* fp = NULL;
    if(NULL == (fp = fopen(buf, "r"))) {
        return -1;
    }

    proc_info[0] = '\0';  /* init */
    size_t space_left = size-1;
    size_t line_len = 0;

    while(NULL != fgets(buf, 1024, fp)) {
        if(0 != strncmp(buf, "Vm", 2)) continue;  /* skip irrelevant lines */

        line_len = strlen(buf);
        if(space_left < line_len) {
            fclose(fp);
            return -2;
        }

        strcat(proc_info, buf);
        space_left -= line_len;
    }

    fclose(fp);
    return 0;
}

/* usage: measure the overhead of measuring CPU cycles
 * params: void
 * return: the overhead in CPU cycles */
uint64_t
rdtsc_overhead(void) {
    uint64_t overhead = UINT64_MAX;

    for(uint64_t i = 0; i < 100000; ++i) {
        uint64_t t0 = rdtsc();
        __asm__ volatile ("");
        uint64_t t1 = rdtsc();
        if((t1 - t0) < overhead)
            overhead = t1 - t0;
    }

    // NOTE: according to Intel documentation, the overhead should be 18 cycles
    // since RDTSC has latency 18, and CPI 2.
    return overhead;
}


/* usage: given an integer, convert it into a binary string
 * params:
 *      1) buf: char buffer for storing the output
 *      2) size: length of the binary string (zero padded). The buffer must
 *              hold at least (size+1) chars.  Otherwise the function has
 *              undefined behavior
 *      3) n: the integer
 * return: void */
void
itoa(char* buf, uint64_t size, uint64_t n) {
    for(uint64_t i = 0; i < size; ++i) {
        buf[i] = (n & 0x1) ? '1' : '0';
        n >>= 1;
    }
    buf[size] = '\0';
}

/* usage: find the index of an integer in a sorted integer array
 * params:
 *      1) hay: a sorted integer array
 *      2) sz: size of the integer array
 *      3) needle: target integer
 * return: index of the target integer in the array. sz if not found */
uint32_t
uint32_find_in_arr(const uint32_t* hay, uint32_t sz, uint32_t needle) {
    for(uint32_t i = 0; i < sz; ++i) {
        if(needle == hay[i])
            return i;
        // this clause only works if the array is sorted
        if(needle < hay[i])
            return sz;
    }

    return sz;
}

/* usage: Given a map from integet set 0 ~ sz-1 to 0 ~ total_range-1, create
 *      a reverse map from 0 ~ total_range-1 to 0 ~ sz-1
 * params:
 *      1) mmap: the map. Should have size sz
 *      2) sz: size of the preimage of the map
 *      3) total_range: size of the image of the map
 * return: a dynamically allocated array storing the reverse map */
uint32_t*
uint32_arr_create_reverse_map(const uint32_t* mmap, uint32_t sz,
                              uint32_t total_range) {
    uint32_t* rmap = malloc(sizeof(uint32_t) * total_range);
    if(!rmap)
        return NULL;

    // UINT32_MAX for missing entries in mmap
    memset(rmap, 0xFF, sizeof(uint32_t) * total_range);
    for(uint32_t i = 0; i < sz; ++i) {
        rmap[mmap[i]] = i;
    }

    return rmap;
}

/* usage: Given a map from integet set 0 ~ sz-1 to 0 ~ total_range-1, create
 *      a reverse map from 0 ~ total_range-1 to 0 ~ sz-1
 * params:
 *      1) mmap: the map. Should have size sz
 *      2) sz: size of the preimage of the map
 *      3) total_range: size of the image of the map
 * return: a dynamically allocated array storing the reverse map */
uint64_t*
uint64_arr_create_reverse_map(const uint64_t* mmap, uint64_t sz,
                              uint64_t total_range) {
    uint64_t* rmap = malloc(sizeof(uint64_t) * total_range);
    if(!rmap)
        return NULL;

    // all 1's for missing entries in mmap
    memset(rmap, 0xFF, sizeof(uint64_t) * total_range);
    for(uint64_t i = 0; i < sz; ++i) {
        rmap[mmap[i]] = i;
    }

    return rmap;
}

/* usage: Given a string, count the number of integers that appear
 * params:
 *      1) str: ptr to char pointint to the string
 * return: number of integers */
uint32_t
count_int_in_str(const char str[]) {
    int32_t count = 0;
    int32_t in_num = 0;

    while(*str) {
        if(isdigit((unsigned char) *str)) {
            if(!in_num)
                in_num = 1;
        } else {
            if(in_num) {
                ++count;
                in_num = 0;
            }
        }
        ++str;
    }

    if(in_num)
        ++count;

    return count;
}

/* usage: generate a random 64-bit integer
 * params: none
 * return: a random 64-bit integer */
uint64_t
uint64_rand(void) {
    uint64_t v = rand();
    return (v << 32) | rand();
}
