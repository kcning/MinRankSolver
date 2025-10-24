#include "mono.h"
#include "util.h"
#include "math_util.h"
#include "ks.h"

#include <stdlib.h> // qsort

/* ========================================================================
 * struct Mono definition
 * ======================================================================== */

struct Mono {
    uint32_t deg;     // current degree of the monomial
    uint32_t max_deg; // max monomial degree that this container can handle
    uint32_t vars[];  // sorted array representing the monomial. For example
                      // x0x5x9x11 is represented as [0, 5, 9, 11]
};

/* ========================================================================
 * function implementations
 * ======================================================================== */

/* usage: Given the degree, create an uninitialized struct Mono
 * params:
 *      1) d: degree
 * return: ptr to struct Mono. NULL on error */
Mono*
mono_create_container(uint32_t d) {
    Mono* m = malloc(sizeof(Mono) + sizeof(uint32_t) * d);
    if(!m)
        return NULL;
    m->max_deg = d;
    m->deg = 0;
    return m;
}

/* usage: Given the degree and an array representing the monomial, create a
 *      struct Mono
 * params:
 *      1) d: degree
 *      2) vars: sorted array representing the monomial. For example
 *          x0x5x9x11 is represented as [0, 5, 9, 11]
 * return: ptr to struct Mono. NULL on error */
Mono*
mono_create(uint32_t d, uint32_t* vars) {
    Mono* m = mono_create_container(d);
    if(!m)
        return NULL;

    memcpy(m->vars, vars, sizeof(uint32_t) * d);
    m->deg = d;
    return m;
}

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
mono_create_from_arr(uint32_t d, uint32_t* vars) {
    Mono* m = (Mono*) vars;
    m->max_deg = d;
    m->deg = d;
    return m;
}

/* usage: Given a struct Mono, release it
 * params:
 *      1) m: ptr to struct Mono
 * return: void */
void
mono_free(Mono* m) {
    free(m);
}

/* usage: Copy a monomial from source into the destination as a partial
 *      monomial. The max degree if the destination must be >= the degree of
 *      the source.
 * params:
 *      1) dst: ptr to struct Mono. destination of the copy operation
 *      2) src: ptr to struct Mono. source of the copy operation
 * return: true if sucess, false on error */
bool
mono_copy_partial_from(Mono* dst, const Mono* src) {
    if(mono_max_deg(dst) < mono_deg(src))
        return false;

    memcpy(dst->vars, src->vars, sizeof(uint32_t) * src->deg);
    dst->deg = src->deg;
    return true;
}

/* usage: Given a struct Mono, zero its variables
 * params:
 *      1) m: ptr to struct Mono
 * return: void */
void
mono_zero(Mono* m) {
    memset(m->vars, 0x0, sizeof(uint32_t) * m->max_deg);
    m->deg = 0;
}

/* usage: Given a struct Mono which is represented as an array of variable
 *      indices, sort the array of variable indices into ascending order
 * params:
 *      1) m: ptr to struct Mono
 * return: void */
void
mono_sort(Mono* m) {
    qsort(m->vars, m->deg, sizeof(uint32_t), uint32_t_cmp);
}

/* usage: Given a struct Mono, return the max monomial degree it can handle
 * params:
 *      1) m: ptr to struct Mono
 * return: its max degree */
uint32_t
mono_max_deg(const Mono* m) {
    return m->max_deg;
}

/* usage: Given a struct Mono, return its degree
 * params:
 *      1) m: ptr to struct Mono
 * return: its degree */
uint32_t
mono_deg(const Mono* m) {
    return m->deg;
}

/* usage: Given a struct Mono, return its internal buffer used to
 *      store its variables
 * params:
 *      1) m: ptr to struct Mono
 * return: its internal buffer as a ptr to uint32_t */
const uint32_t*
mono_vars(const Mono* m) {
    return m->vars;
}

/* usage: Given a struct Mono, set its degree
 * params:
 *      1) m: ptr to struct Mono
 *      2) d: new degree
 * return: void */
void
mono_set_deg(Mono* m, uint32_t d) {
    assert(d <= m->max_deg);
    m->deg = d;
}

/* usage: Given a struct Mono, return one of its variable. The variables of a
 *      monomials are stored as indices. For example deg-4 monomial x0x1x9x12 is
 *      stored internally as an sorted array [0, 1, 9, 12] of size 4. With index
 *      0, 1, 2, 3, one will thus receive variable x0, x1, x9, x12, respectively.
 * params:
 *      1) m: ptr to struct Mono
 *      2) i: index of the variable
 * return: index representing a variable. xj is represented by integer j */
uint32_t
mono_var(const Mono* m, uint32_t i) {
    return m->vars[i];
}

/* usage: Given a struct Mono, return its largest variable according to grlex
 * params:
 *      1) m: ptr to struct Mono
 * return: index representing a variable. xj is represented by integer j */
uint32_t
mono_last_var(const Mono* m) {
    return m->vars[mono_deg(m)-1];
}

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
mono_set_var(Mono* m, uint32_t i, uint32_t v, bool sort) {
    assert(i < m->deg);
    m->vars[i] = v;
    if(unlikely(sort))
        mono_sort(m);
}

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
                uint32_t k, uint32_t r) {
    assert(mono_max_deg(mono) >= mdeg_total_deg(mdeg));
    mono_set_deg(mono, mdeg_total_deg(mdeg));
    // NOTE: x^2 or higher degree for a group of vars is possible since we
    // can't use the field equation to reduce the degree
    for(uint32_t i = 0; i < mdeg_lv_deg(mdeg); ++i) // linear vars
        mono_set_var(mono, i, 0, false);

    uint32_t mono_offset = mdeg_lv_deg(mdeg);
    for(uint32_t i = 0; i < mdeg_c(mdeg); ++i) { // for each group of kernel vars
        for(uint32_t j = 0; j < mdeg_kv_deg(mdeg, i); ++j)
            mono_set_var(mono, mono_offset + j, k + r * i, false);
        mono_offset += mdeg_kv_deg(mdeg, i);
    }
}

/* subroutine of mdmac_mdeg_iterate: increment part of a monomial */
static inline bool
mono_mdeg_inc(uint32_t* mono_buf, uint32_t size, uint32_t min, uint32_t max) {
    if(unlikely(size == 0))
        return true;

    uint32_t i = 0;
    while( (i+1) < size && mono_buf[i] == mono_buf[i+1]) {
        mono_buf[i] = min;
        ++i;
    }
    ++mono_buf[i];

    bool carry = false;
    if(unlikely(mono_buf[i] == max)) {
        mono_buf[i] = min;
        carry = true;
    }

    return carry;
}

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
                  uint32_t r) {
    uint32_t* mono_buf = mono->vars;
    bool carry = true;
    uint32_t min = 0;
    uint32_t max = k;
    for(uint32_t i = 0; i <= mdeg_c(d) && carry; ++i) {
        carry = mono_mdeg_inc(mono_buf, mdeg_deg(d, i), min, max);
        mono_buf += mdeg_deg(d, i);
        min = max;
        max += r;
    }

    return !carry;
}

/* usage: Given a monomial, print it
 * params:
 *      1) mono: ptr to struct Mono
 * return: void */
void
mono_print(const Mono* mono) {
    for(uint32_t i = 0; i < mono_deg(mono); ++i) {
        printf("x%d", mono_var(mono, i));
    }
    printf("\n");
}

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
                uint32_t r) {
    uint32_t counter[mdeg_c(d)+1];
    memset(counter, 0x0, sizeof(uint32_t) * (mdeg_c(d) + 1));

    for(uint32_t i = 0; i < mono_deg(m); ++i) {
        uint32_t vidx = mono_var(m, i);
        if(vidx < k) {
            ++counter[0];
        } else {
            uint32_t ri = ks_kernel_var_idx_to_grp_idx(vidx, k, r);
            assert(ri < mdeg_c(d));
            ++counter[1 + ri];
        }
    }

    for(uint32_t i = 0; i <= mdeg_c(d); ++i) {
        if(mdeg_deg(d, i) < counter[i])
            return false;
    }

    return true;
}

/* usage: given a monomial, find its multi-degree
 * params:
 *      1) d: ptr to struct MDeg
 *      2) m: ptr to struct Mono
 *      3) k: number of linear variables
 *      4) r: number of kernel variables per row
 * return: void */
void
mono_to_mdeg(MDeg* restrict d, const Mono* restrict m, uint32_t k, uint32_t r) {
    mdeg_zero(d);
    for(uint32_t i = 0; i < mono_deg(m); ++i) {
        uint32_t vidx = mono_var(m, i);
        if(vidx < k) {
            mdeg_deg_inc(d, 0);
        } else {
            uint32_t ri = ks_kernel_var_idx_to_grp_idx(vidx, k, r);
            assert(ri < mdeg_c(d));
            mdeg_deg_inc(d, 1 + ri);
        }
    }
}
