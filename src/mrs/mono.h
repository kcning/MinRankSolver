#ifndef __MONO_H__
#define __MONO_H__

#include <stdint.h>
#include <stdbool.h>

#include "mdeg.h"

typedef struct Mono Mono;

/* ========================================================================
 * function prototypes
 * ======================================================================== */

/* usage: Given the degree, create an uninitialized struct Mono
 * params:
 *      1) d: degree
 * return: ptr to struct Mono. NULL on error */
Mono*
mono_create_container(uint32_t d);

/* usage: Given the degree and an array representing the monomial, create a
 *      struct Mono
 * params:
 *      1) d: degree
 *      2) vars: sorted array representing the monomial. For example
 *          x0x5x9x11 is represented as [0, 5, 9, 11]
 * return: ptr to struct Mono. NULL on error */
Mono*
mono_create(uint32_t d, uint32_t* vars);

/* usage: Given the max degree of the monomial, create an uninitialized
 *      static array that can hold the degrees of the monomials which
 *      can be passed to mono_create_from_arr
 * params:
 *      1) name: name of the array
 *      2) d: max degree
 * return: void */
#define mono_create_static_buf(buf_name, d) \
    uint32_t buf_name[2+(d)]

/* usage: Given the degree and an array representing the monomial, create a
 *      struct Mono from the array without allocating memory. This can
 *      be more efficicient but the caller must ensure the lifespan of the
 *      array is longer than that of the resultant struct Mono
 * params:
 *      1) d: degree
 *      2) vars: sorted array of size at least d+2, where the monomial is
 *          represented by the vars[2] ~ vars[d+1]. Note that vars[0] and
 *          vars[1] will be overwritten by this function
 * return: ptr to struct Mono */
Mono*
mono_create_from_arr(uint32_t d, uint32_t* vars);

/* usage: Given a struct Mono, release it
 * params:
 *      1) m: ptr to struct Mono
 * return: void */
void
mono_free(Mono* m);

/* usage: Copy a monomial from source into the destination as a partial
 *      monomial. The max degree if the destination must be >= the degree of
 *      the source.
 * params:
 *      1) dst: ptr to struct Mono. destination of the copy operation
 *      2) src: ptr to struct Mono. source of the copy operation
 * return: true if sucess, false on error */
bool
mono_copy_partial_from(Mono* dst, const Mono* src);

/* usage: Given a struct Mono, zero its variables
 * params:
 *      1) m: ptr to struct Mono
 * return: void */
void
mono_zero(Mono* m);

/* usage: Given a struct Mono which is represented as an array of variable
 *      indices, sort the array of variable indices into ascending order
 * params:
 *      1) m: ptr to struct Mono
 * return: void */
void
mono_sort(Mono* m);

/* usage: Given a struct Mono, return the max monomial degree it can handle
 * params:
 *      1) m: ptr to struct Mono
 * return: its max degree */
uint32_t
mono_max_deg(const Mono* m);

/* usage: Given a struct Mono, return its degree
 * params:
 *      1) m: ptr to struct Mono
 * return: its degree */
uint32_t
mono_deg(const Mono* m);

/* usage: Given a struct Mono, return its internal buffer used to
 *      store its variables
 * params:
 *      1) m: ptr to struct Mono
 * return: its internal buffer as a ptr to uint32_t */
const uint32_t*
mono_vars(const Mono* m);

/* usage: Given a struct Mono, set its degree
 * params:
 *      1) m: ptr to struct Mono
 *      2) d: new degree
 * return: void */
void
mono_set_deg(Mono* m, uint32_t d);

/* usage: Given a struct Mono, return one of its variable. The variables of a
 *      monomials are stored as indices. For example deg-4 monomial x0x1x9x12 is
 *      stored internally as an sorted array [0, 1, 9, 12] of size 4. With index
 *      0, 1, 2, 3, one will thus receive variable x0, x1, x9, x12, respectively.
 * params:
 *      1) m: ptr to struct Mono
 *      2) i: index of the variable
 * return: index representing a variable. xj is represented by integer j */
uint32_t
mono_var(const Mono* m, uint32_t i);

/* usage: Given a struct Mono, return its largest variable according to grlex
 * params:
 *      1) m: ptr to struct Mono
 * return: index representing a variable. xj is represented by integer j */
uint32_t
mono_last_var(const Mono* m);

/* usage: Given a struct Mono, return one of its variable. The variables of a
 *      monomials are stored as indices. For example deg-4 monomial x0x1x9x12 is
 *      stored internally as an sorted array [0, 1, 9, 12] of size 4. When setting
 *      variable with this function, the caller is responsible to ensure the
 *      array of indices are sorted.
 * params:
 *      1) m: ptr to struct Mono
 *      2) i: index of the variable
 *      3) v: integer representing the new variable. xj is represented by j
 *      4) sort: true if the function should sort the indices. If false, the
 *          caller must ensure that the array of indices remains sorted
 *          after setting the new value
 * return: void */
void
mono_set_var(Mono* m, uint32_t i, uint32_t v, bool sort);

/* usage: Given a multi-degree, pick the first monomial of that multi-degree
 * params:
 *      1) mono: ptr to struct Mono. Container for the first monomial
 *      2) mdeg: ptr to struct MDeg, target multi-degree. Should have the same
 *          degreee as mono
 *      3) k: number of linear variables
 *      4) r: number of kernel variables per row
 * return: void */
void
mono_mdeg_first(Mono* restrict mono, const MDeg* restrict mdeg,
                uint32_t k, uint32_t r);

/* usage: Given a multi-degree, and the current monomial, update the monomial
 *      to the next one
 * params:
 *      1) mono: ptr to struct Mono. Container for the next monomial
 *      2) d: ptr to struct MDeg, target multi-degree. Should have the same
 *          degree as mono
 *      3) k: number of linear variables
 *      4) r: number of kernel variables per row
 * return: true if the current monomial is not the last one, otherwise false */
bool
mono_mdeg_iterate(Mono* restrict mono, const MDeg* restrict d, uint32_t k,
                  uint32_t r);

/* usage: Given a monomial, print it
 * params:
 *      1) mono: ptr to struct Mono
 * return: void */
void
mono_print(const Mono* mono);

/* usage: check if the given monomial is valid for the given target
 *      multi-degree
 * params:
 *      1) m: ptr to struct Mono
 *      2) d: ptr to struct MDeg
 *      3) k: number of linear variables
 *      4) r: number of kernel variables per row
 * return: true or false */
bool
mono_check_mdeg(const Mono* restrict m, const MDeg* restrict d, uint32_t k,
                uint32_t r);

/* usage: given a monomial, find its multi-degree
 * params:
 *      1) d: ptr to struct MDeg
 *      2) m: ptr to struct Mono
 *      3) k: number of linear variables
 *      4) r: number of kernel variables per row
 * return: void */
void
mono_to_mdeg(MDeg* restrict d, const Mono* restrict m, uint32_t k, uint32_t r);

#endif // __MONO_H__
