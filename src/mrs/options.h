#ifndef __BLK_LANCZOS_OPTIONS_H__
#define __BLK_LANCZOS_OPTIONS_H__

#include "mdeg.h"

#include <stdbool.h>
#include <stdint.h>

typedef struct Options Options;

/* ========================================================================
 * function prototypes
 * ======================================================================== */

/* usage: check if the program should be more verbose
 * params:
 *      1) opts: pointer to struct Options
 * return: true if yes, false otherwise */
bool
opt_verbose(const Options* opts);

/* usage: check if the program is in dry run mode
 * params:
 *      1) opts: pointer to struct Options
 * return: true if yes, false otherwise */
bool
opt_dry(const Options* opts);

/* usage: check if the program should print help message
 * params:
 *      1) opts: pointer to struct Options
 * return: true if yes, false otherwise */
bool
opt_help(const Options* opts);

/* usage: check if the program should set a new random seed
 * params:
 *      1) opts: pointer to struct Options
 * return: true if yes, false otherwise */
bool
opt_new_randseed(const Options* opts);

/* usage: return the 32-bit random seed of the program
 * params:
 *      1) opts: pointer to struct Options
 * return: the random seed as an uint32_t */
uint32_t
opt_seed(const Options* opts);

/* usage: return the path to the input MinRank instance
 * params:
 *      1) opts: pointer to struct Options
 * return: a char pointer to the path of input file */
const char*
opt_mr_file(const Options* opts);

/* usage: return the number of multi-degrees for the Macaulay matrix
 * params:
 *      1) opts: pointer to struct Options
 * return: the multi-degree */
uint32_t
opt_mdeg_num(const Options* opts);

/* usage: return the i-th multi-degree of the Macaulay matrix
 * params:
 *      1) opts: pointer to struct Options
 *      2) i: index of the multi-degree
 * return: the multi-degree */
const MDeg*
opt_mdeg(const Options* opts, uint32_t i);

/* usage: return multi-degree(s) of the Macaulay matrix
 * params:
 *      1) opts: pointer to struct Options
 * return: the multi-degree */
const MDeg**
opt_degs(const Options* opts);

/* usage: check if the program should randomly sample the Kipnis-Shamir
 *      matrix instead of computing it from the input MinRank instance
 *      1) opts: pointer to struct Options
 * return: true if yes, false otherwise */
bool
opt_ks_rand(const Options* opts);

/* usage: return the number of rows in left matrix of the KS system
 * params:
 *      1) opts: pointer to struct Options
 * return: the number of rows */
uint32_t
opt_c(const Options* opts);

/* usage: return the size of the thread pool
 * params:
 *      1) opts: pointer to struct Options
 * return: the number of threads */
uint32_t
opt_tpsize(const Options* opts);

/* usage: return the number of rows to randomly select from the full multi-
 *      degree Macaulay matrix
 * params:
 *      1) opts: pointer to struct Options
 * return: the number of rows */
uint64_t
opt_mac_nrow(const Options* opts);

/* usage: create and initialize a struct Options
 * params: void
 * return: a pointer to struct Options. On failure, return NULL */
Options*
opt_create(void);

/* usage: free a struct Options
 * params:
 *      1) opts: a pointer to struct Options
 * return: void */
void
opt_free(Options* opts);

/* usage: parse argv and store command-line options into struct Options
 * params:
 *      1) opts: a pointer to struct Options
 *      2) argc: command-line argument counter
 *      3) argv: command-line argument vector
 * return: 0 if success. On error return a non-zero error code. */
int
opt_parse(Options* const opts, int argc, char** argv);

/* usage: Given an error code returned from opt_parse(), return a human
 *      friendly text explanation
 * params:
 *      1) code: error code
 * return: a string pointer to the text explanation. Do not modified the
 *      string it points to. */
const char*
opt_err_code_to_str(int code);

/* usage: print help message
 * params:
 *      1) name, name of the program
 * return: void */
void
opt_print_usage(const char* const name);

#endif /* __BLK_LANCZOS_OPTIONS_H__ */
