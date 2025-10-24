#include "options.h"
#include "math_util.h"
#include "mdeg.h"

#include <linux/limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <string.h>
#include <time.h>
#include <stdbool.h>
#include <errno.h>

#include <sys/sysinfo.h> // get_nprocs

/* ========================================================================
 * struct Options definition
 * ======================================================================== */

#define MAX_FILE_PATH_LEN               (255)
#define MAX_INPUT_STR_LEN               (255)
#define MAX_MDEG_NUM                    (64)

#define OPT_PARSE_ERR_PATH_TOO_LONG     (1)
#define OPT_PARSE_NO_MDEG               (2)
#define OPT_PARSE_INVALID_MDEG          (3)
#define OPT_PARSE_NO_PATH               (4)
#define OPT_PARSE_NO_C                  (5)
#define OPT_PARSE_MDEG_NUM_MAX          (6)
#define OPT_PARSE_MDEG_DIFF_C           (7)
#define OPT_PARSE_INVALID_TNUM          (8)
#define OPT_PARSE_TOO_MANY_MR_FILE      (9)
#define OPT_PARSE_INVALID_NUM           (126)
#define OPT_PARSE_UNKNOWN_ERR           (127)
#define OPT_PARSE_INVALID_OPT           (128)

struct Options {
    uint32_t seed;
    uint32_t tpsize; // thread pool size
    uint32_t c;
    uint64_t mac_nrow; // number of rows to keep in Macaulay matrices
    uint32_t degs_sz;

    char mr_file[MAX_FILE_PATH_LEN+1];
    MDeg* mdeg[MAX_MDEG_NUM];

    bool verbose;
    bool help;
    bool dry;
    bool rand_seed;
    bool has_mr_file;
    bool ks_rand;
};

/* ========================================================================
 * function implementations
 * ======================================================================== */

/* usage: check if the program should be more verbose
 * params:
 *      1) opts: pointer to struct Options
 * return: true if yes, false otherwise */
bool
opt_verbose(const Options* opts) {
    return opts->verbose;
}

/* usage: check if the program should set a new random seed
 * params:
 *      1) opts: pointer to struct Options
 * return: true if yes, false otherwise */
bool
opt_new_randseed(const Options* opts) {
    return opts->rand_seed;
}

/* usage: check if the program is in dry run mode
 * params:
 *      1) opts: pointer to struct Options
 * return: true if yes, false otherwise */
bool
opt_dry(const Options* opts) {
    return opts->dry;
}

/* usage: check if the program should print help message
 * params:
 *      1) opts: pointer to struct Options
 * return: true if yes, false otherwise */
bool
opt_help(const Options* opts) {
    return opts->help;
}

/* usage: return the 32-bit random seed of the program
 * params:
 *      1) opts: pointer to struct Options
 * return: the random seed as an uint32_t */
uint32_t
opt_seed(const Options* opts) {
    return opts->seed;
}

/* usage: return the path to the input MinRank instance
 * params:
 *      1) opts: pointer to struct Options
 * return: a char pointer to the path of input file */
const char*
opt_mr_file(const Options* opts) {
    return opts->mr_file;
}

/* usage: return the number of rows in left matrix of the KS system
 * params:
 *      1) opts: pointer to struct Options
 * return: the number of rows */
uint32_t
opt_c(const Options* opts) {
    return opts->c;
}

/* usage: return the number of multi-degrees for the Macaulay matrix
 * params:
 *      1) opts: pointer to struct Options
 * return: the multi-degree */
uint32_t
opt_mdeg_num(const Options* opts) {
    return opts->degs_sz;
}

/* usage: return the i-th multi-degree of the Macaulay matrix
 * params:
 *      1) opts: pointer to struct Options
 *      2) i: index of the multi-degree
 * return: the multi-degree */
const MDeg*
opt_mdeg(const Options* opts, uint32_t i) {
    assert(i < MAX_MDEG_NUM);
    return opts->mdeg[i];
}

/* usage: return multi-degree(s) of the Macaulay matrix
 * params:
 *      1) opts: pointer to struct Options
 * return: the multi-degree */
const MDeg**
opt_degs(const Options* opts) {
    return (const MDeg**) opts->mdeg;
}

/* usage: check if the program should randomly sample the Kipnis-Shamir
 *      matrix instead of computing it from the input MinRank instance
 *      1) opts: pointer to struct Options
 * return: true if yes, false otherwise */
bool
opt_ks_rand(const Options* opts) {
    return opts->ks_rand;
}

/* usage: return the size of the thread pool
 * params:
 *      1) opts: pointer to struct Options
 * return: the size of the thread pool */
uint32_t
opt_tpsize(const Options* opts) {
    return opts->tpsize;
}

/* usage: return the number of rows to randomly select from the full multi-
 *      degree Macaulay matrix
 * params:
 *      1) opts: pointer to struct Options
 * return: the number of rows */
uint64_t
opt_mac_nrow(const Options* opts) {
    return opts->mac_nrow;
}

/* usage: create and initialize a struct Options
 * params: void
 * return: a pointer to struct Options. On failure, return NULL */
Options*
opt_create(void) {
    Options* opts = calloc(1, sizeof(Options));
    if(!opts) {
        return NULL;
    }

    return opts;
}

/* usage: free a struct Options
 * params:
 *      1) opts: a pointer to struct Options
 * return: void */
void
opt_free(Options* opts) {
    for(uint32_t i = 0; i < opts->degs_sz; ++i)
        mdeg_free(opts->mdeg[i]);
    free(opts);
}

/* for getopt_long */
#define OPT_SEED                1
#define OPT_MR_SYS              2
#define OPT_VERBOSE             3
#define OPT_MAC_MDEG            4
#define OPT_DRY                 5
#define OPT_TPOOL_SIZE          6
#define OPT_MAC_ROW             7
#define OPT_KS_RAND             8

#define OPT_SEED_STR            "seed"
#define OPT_MR_SYS_STR          "minrank"
#define OPT_VERBOSE_STR         "verbose"
#define OPT_DRY_STR             "dry-run"
#define OPT_MAC_MDEG_STR        "mdeg"
#define OPT_TPOOL_SIZE_STR      "thread"
#define OPT_MAC_ROW_STR         "mac-row"
#define OPT_KS_RAND_STR         "ks-rand"
#define OPT_HELP_STR            "help"

static struct option long_opts[] = {
    { OPT_MR_SYS_STR, 1, 0, OPT_MR_SYS },
    { OPT_SEED_STR, 1, 0, OPT_SEED },
    { OPT_VERBOSE_STR, 0, 0, OPT_VERBOSE },
    { OPT_DRY_STR, 0, 0, OPT_DRY },
    { OPT_SEED_STR, 1, 0, OPT_SEED },
    { OPT_DRY_STR, 0, 0, OPT_DRY },
    { OPT_KS_RAND_STR, 0, 0, OPT_KS_RAND },
    { OPT_TPOOL_SIZE_STR, 1, 0, OPT_TPOOL_SIZE },

    { OPT_MAC_MDEG_STR, 1, 0, OPT_MAC_MDEG },
    { OPT_MAC_ROW_STR, 1, 0, OPT_MAC_ROW },

    { OPT_HELP_STR, 0, 0, 'h' },
    { 0, 0, 0, 0 }
};

/* usage: print help message
 * params:
 *      1) name, name of the program
 * return: void */
void
opt_print_usage(const char* const name) {
    // not using macros defined above for each option to keep alignment easier
    printf(
"Usage: %s [OPTIONS] --minrank=FILE --mdeg=DEG\n"
"\n"
"Options:\n"
"\n"
"  --seed=SEED      Use 32-bit SEED to initialize the random number generator.\n"
"                   Default seed is random.\n"
"\n"
"  --minrank=FILE   Read MinRank instance to solve from FILE. FILE must have\n"
"                   the same format as files generated by bin/minrank-gen.sage.\n"
"\n"
"  --mdeg=DEG       Multi-degree of the Macaulay matrix. At least one multi-\n"
"                   degree must be provided. If more than one is provided,\n"
"                   the Macaulay matrix will be defined over the combined multi-\n"
"                   degrees. Currently at most 64 multi-degrees are supported.\n"
"\n"
"  --verbose        Print extra information.\n"
"\n"
"  --thread         Number of threads that should be used. It is recommended\n"
"                   to use as many threads as the number of CPU cores,\n"
"                   which is also the default value.\n"
"\n"
"  --mac-row=NUM    Specify the number of rows to randomly select and keep in\n"
"                   the Macaulay matrix. By default, all rows are kept.\n"
"\n"
"  --ks-rand        Instead of computing the Kipnis-Shamir matrix from the input\n"
"                   MinRank instance, randomly sample it with the same dimension\n"
"\n"
"  --dry-run        Do not actually solve the MinRank instance; Simply check\n"
"                   the sanity of the parameters and then terminate.\n"
"\n"
"Examples:\n"
"\n"
"  %s --verbose --minrank=toy_example.txt --mdeg=2,2,1\n"
"\n"
"  %s --minrank=large_system.txt --mdeg=2,2,2,2,1,1 --mdeg=1,2,2,2,1,2\n"
"\n", name, name, name);
}

/* usage: subroutine of options_parse(): copy input string with strncpy and
 *      make sure it's null-teminated.
 * params:
 *      1) dst: destination to copy to; must be at least n+1 bytes large
 *      2) src: source to copy from
 *      3) n: number of chars to copy form src
 * return: 0 if success; otherwise non-zero value  */
static inline int
safe_strncpy(char* const dst, const char* const src, const size_t n) {
    const size_t len = strnlen(src, n+1);
    if(len > n) {
        return 1;
    }

    strncpy(dst, src, len);
    dst[len] = '\0';
    return 0;
}

static inline MDeg*
opt_parse_mdeg(const char* str) {
    if(MAX_INPUT_STR_LEN == strnlen(str, MAX_INPUT_STR_LEN))
        return NULL;

    uint32_t c = 0;
    const char* substr = strchr(str, ',');
    while(substr != NULL) {
        ++c;
        substr = strchr(substr + 1, ',');
    }

    mdeg_create_static_buf(tmp_buf, c);
    MDeg* mdeg = mdeg_create_from_arr(c, tmp_buf);

    substr = str;
    char* next_substr;
    uint32_t i = 0;
    while(i <= c ) {
        errno = 0;
        uint64_t d = strtol(substr, &next_substr, 0);
        if(errno)
            return NULL;
        if(substr == next_substr && d == 0) // no more integers
            break;
        substr = next_substr + 1;
        assert(d <= UINT32_MAX);
        mdeg_set_deg(mdeg, i++, d);
    }

    if(i != (c + 1))
        return NULL;

    if(*next_substr != '\0')
        return NULL;

    return mdeg_dup(mdeg);
}

/* usage: parse argv and store command-line options into struct Options
 * params:
 *      1) opts: a pointer to struct Options
 *      2) argc: command-line argument counter
 *      3) argv: command-line argument vector
 * return: 0 if success. On error return non-zero value. */
int
opt_parse(Options* const opts, int argc, char** argv) {
    int c, opt_idx;
    while(-1 != (c = getopt_long(argc, argv, "h", long_opts, &opt_idx))) {
        switch(c) {
            case 0:
                if(long_opts[opt_idx].flag == 0) {
                    // do nothing if flag is set
                }
                break;

            case 'h':
                opts->help = true;
                return 0; // no need to parse further

            case OPT_VERBOSE:
                opts->verbose = true;
                break;

            case OPT_DRY:
                opts->dry = true;
                break;

            case OPT_KS_RAND:
                opts->ks_rand = true;
                break;

            case OPT_TPOOL_SIZE:
                errno = 0;
                opts->tpsize = strtol(optarg, NULL, 0);
                if(errno || opt_tpsize(opts) == 0) {
                    return OPT_PARSE_INVALID_NUM;
                }
                break;

            case OPT_SEED:
                errno = 0;
                opts->seed = strtol(optarg, NULL, 0);
                if(errno) {
                    return OPT_PARSE_INVALID_NUM;
                }
                opts->rand_seed = true;
                break;

            case OPT_MR_SYS:
                if(opts->has_mr_file)
                    return OPT_PARSE_TOO_MANY_MR_FILE;

                if(safe_strncpy(opts->mr_file, optarg, MAX_FILE_PATH_LEN))
                    return OPT_PARSE_ERR_PATH_TOO_LONG;

                opts->has_mr_file = true;
                break;

            case OPT_MAC_MDEG:
                if(opts->degs_sz > MAX_MDEG_NUM)
                    return OPT_PARSE_MDEG_NUM_MAX;

                MDeg* d = NULL;
                if(NULL == (d = opt_parse_mdeg(optarg)))
                    return OPT_PARSE_INVALID_MDEG;

                if(opts->degs_sz == 0)
                    opts->c = mdeg_c(d);
                else if (opts->c != mdeg_c(d)) {
                    mdeg_free(d);
                    return OPT_PARSE_MDEG_DIFF_C;
                }

                opts->mdeg[(opts->degs_sz)++] = d;
                break;

            case OPT_MAC_ROW:
                errno = 0;
                opts->mac_nrow = strtol(optarg, NULL, 0);
                if(errno)
                    return OPT_PARSE_INVALID_NUM;
                break;

            case '?':
                // getopt_long already printed an error message
                return OPT_PARSE_INVALID_OPT;

            default:
                return OPT_PARSE_UNKNOWN_ERR;
        }
    }

    // mandatory option
    if(!opts->has_mr_file)
        return OPT_PARSE_NO_PATH;

    if(opts->degs_sz == 0)
        return OPT_PARSE_NO_MDEG;

    // set default thread num
    if(!opt_tpsize(opts))
        opts->tpsize = next_power_of_2(get_nprocs());

    return 0;
}

const char* const opt_parse_path_too_long_str =
    "input path length > 255";
const char* const opt_parse_no_path_str =
    "missing option "OPT_MR_SYS_STR;
const char* const opt_parse_no_mdeg_str =
    "missing option "OPT_MAC_MDEG_STR;
const char* const opt_parse_invalid_mdeg_str =
    "Invalid multi-degree";
const char* const opt_parse_mdeg_num_max_str =
    "too many multi-degrees, max supported number: 64";
const char* const opt_parse_mdeg_diff_c_str =
    "multi-degrees have different number of groups of kernel variables";
const char* const opt_parse_invalid_alg_str =
    "invalid algorithm";
const char* const opt_parse_invalid_fix_str =
    "invalid binary string";
const char* const opt_parse_invalid_num =
    "invalid number";
const char* const opt_parse_invalid_tnum =
    "thread number must be a power of 2";
const char* const opt_parse_invalid_tnum_ssys =
    "thread number for extracting sub-systems must <= thread pool size";
const char* const opt_parse_unknown_str =
    "unknown error";
const char* const opt_parse_invalid_opt_str =
    "invalid option";
const char* const opt_parse_too_many_mr_str =
    "there can be only 1 input MinRank file";

/* usage: Given an error code returned from opt_parse(), return a human
 *      friendly text explanation
 * params:
 *      1) code: error code
 * return: a string pointer to the text explanation. Do not modified the
 *      string it points to. */
const char*
opt_err_code_to_str(int code) {
    switch(code) {
        case OPT_PARSE_ERR_PATH_TOO_LONG:
            return opt_parse_path_too_long_str;
        case OPT_PARSE_NO_MDEG:
            return opt_parse_no_mdeg_str;
        case OPT_PARSE_INVALID_MDEG:
            return opt_parse_invalid_mdeg_str;
        case OPT_PARSE_MDEG_DIFF_C:
            return opt_parse_mdeg_diff_c_str;
        case OPT_PARSE_MDEG_NUM_MAX:
            return opt_parse_mdeg_num_max_str;
        case OPT_PARSE_NO_PATH:
            return opt_parse_no_path_str;
        case OPT_PARSE_INVALID_NUM:
            return opt_parse_invalid_num;
        case OPT_PARSE_INVALID_OPT:
            return opt_parse_invalid_opt_str;
        case OPT_PARSE_INVALID_TNUM:
            return opt_parse_invalid_tnum;
        case OPT_PARSE_TOO_MANY_MR_FILE:
            return opt_parse_too_many_mr_str;
        case OPT_PARSE_UNKNOWN_ERR:
            // fall through
        default:
            return opt_parse_unknown_str;
    }
}
