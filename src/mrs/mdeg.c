#include "mdeg.h"

#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "util.h"
#include "math_util.h"

/* ========================================================================
 * struct MDeg definition
 * ======================================================================== */

struct MDeg {
    uint32_t c;   // number of subgroups of kernel vars
    uint32_t d[]; // multi-degree based on which the multi-degree Macaulay
                  // matrix is computed. It should be array of size c+1
                  /*
                   * For example, for left matrix:
                   *
                   * | 1 0 0 0     0 x1 ... xr |
                   * | 0 1 0 0 ... 0 y1 ... yr |
                   * | 0 0 1 0     0 z1 ... zr |
                   *
                   * (deg(linear variables), deg(x's), deg(y's), deg(z's)) =
                   * (2, 3, 2, 1) is represented as [2, 3, 2, 1] */
};

/* ========================================================================
 * function implementations
 * ======================================================================== */

/* usage: Given a struct MDeg, return the number of subgroups of kernel vars
 * params:
 *      1) d: ptr to struct MDeg
 * return: number of subgroups */
uint32_t
mdeg_c(const MDeg* d) {
    return d->c;
}

/* usage: Given a struct MDeg, expose its internal buffer for storing
 *      degrees as a pointer
 * params:
 *      1) d: ptr to struct Mdeg
 * return: a uint32_t ptr to its internal buffer */
const uint32_t*
mdeg_deg_buffer(const MDeg* d) {
    return d->d;
}

/* usage: Given a struct MDeg, return the selected degree of group of
 *      variables
 * params:
 *      1) d: ptr to struct MDeg
 *      2) i: index of the group. 0 for linear vars, 1 ~ c for kernel vars
 * return: degree of the selected group of vars */
uint32_t
mdeg_deg(const MDeg* d, uint32_t i) {
    return d->d[i];
}

/* usage: Given a struct MDeg, increment the degree of the selected
 *      group of variables
 * params:
 *      1) d: ptr to struct MDeg
 *      2) i: index of the group. 0 for linear vars, 1 ~ c for kernel vars
 * return: void */
void
mdeg_deg_inc(MDeg* d, uint32_t i) {
    ++(d->d[i]);
}

/* usage: Given a struct MDeg, decrement the degree of the selected
 *      group of variables
 * params:
 *      1) d: ptr to struct MDeg
 *      2) i: index of the group. 0 for linear vars, 1 ~ c for kernel vars
 * return: void */
void
mdeg_deg_dec(MDeg* d, uint32_t i) {
    --(d->d[i]);
}

/* usage: Given a struct MDeg, set the degree of the selected group of
 *      variables
 * params:
 *      1) d: ptr to struct MDeg
 *      2) i: index of the group. 0 for linear vars, 1 ~ c for kernel vars
 *      3) v: new degree of linear variables
 * return: degree of the selected group of vars */
void
mdeg_set_deg(MDeg* d, uint32_t i, uint32_t v) {
    d->d[i] = v;
}

/* usage: Given a struct MDeg, set all of its degree to zero
 * params:
 *      1) d: ptr to struct MDeg
 * return: void */
void
mdeg_zero(MDeg* d) {
    memset(d->d, 0x0, sizeof(uint32_t) * (d->c + 1));
}

/* usage: Given a struct MDeg, return total degree of all variables
 * params:
 *      1) d: ptr to struct MDeg
 * return: degree of the all vars */
uint32_t
mdeg_total_deg(const MDeg* d) {
    return sum_arr(d->d, d->c + 1);
}

/* usage: Given a struct MDeg, return the degree of linear variables
 * params:
 *      1) d: ptr to struct MDeg
 * return: degree of linear var */
uint32_t
mdeg_lv_deg(const MDeg* d) {
    return d->d[0];
}

/* usage: Given a struct MDeg, set its degree of linear vars to the
 *      given value
 * params:
 *      1) d: ptr to struct MDeg
 *      2) v: new degree of linear variables
 * return: void */
void
mdeg_set_lv_deg(MDeg* d, uint32_t v) {
    d->d[0] = v;
}

/* usage: Given a struct MDeg, return the degree of kernel variables
 * params:
 *      1) d: ptr to struct MDeg
 *      2) i: index of the subgroup of kernel vars
 * return: degree of the specified subgroup of kernel vars */
uint32_t
mdeg_kv_deg(const MDeg* d, uint32_t i) {
    assert(mdeg_c(d) > i);
    return d->d[1+i];
}

/* usage: Given a struct MDeg, set the degree of selected group of
 *      kernel vars to the given value
 * params:
 *      1) d: ptr to struct MDeg
 *      2) i: index of the subgroup of kernel vars
 *      3) v: new degree for the selected group linear variables
 * return: void */
void
mdeg_set_kv_deg(MDeg* d, uint32_t i, uint32_t v) {
    assert(mdeg_c(d) > i);
    d->d[1+i] = v;
}

/* usage: Given the number of subgroup of kernel vars, and the
 *      degree of linear vars and each subgroup of kernel vars,
 *      create a struct MDeg
 * params:
 *      1) c: number of subgroups of kernel vars
 *      2) ds: a uint32_t array of size (c+1). ds[0] stores
 *          the degree of linear var, ds[1] the degree of the
 *          1st subgroup of kernel vars, and so on.
 * return: ptr to struct MDeg */
MDeg*
mdeg_create(uint32_t c, const uint32_t ds[]) {
    MDeg* m = malloc(sizeof(MDeg) + sizeof(uint32_t) * (c+1));
    if(!m)
        return NULL;
    m->c = c;
    memcpy(m->d, ds, sizeof(uint32_t) * (c+1));

    return m;
}

/* usage: Given the number of subgroup of kernel vars, and an array storing the
 *      degrees of subgroups, create a struct MDeg from the array. The array
 *      will be modified by this function. The caller must ensure the lifespan
 *      of the array is longer than the resultant struct MDeg.
 * params:
 *      1) c: number of subgroups of kernel vars
 *      2) ds: a uint32_t array of size at least (c+2). ds[1] stores
 *          the degree of linear var, ds[2] the degree of the
 *          1st subgroup of kernel vars, and so on.
 * return: ptr to struct MDeg */
MDeg*
mdeg_create_from_arr(uint32_t c, uint32_t* ds) {
    MDeg* m = (MDeg*) ds;
    m->c = c;
    assert(ds[0] == c);
    return m;
}

/* usage: Given the number of subgroup of kernel vars, create a struct MDeg
 *      where each group of vars has zero degree
 * params:
 *      1) c: number of subgroups of kernel vars
 * return: ptr to struct MDeg */
MDeg*
mdeg_create_zero(uint32_t c) {
    MDeg* m = malloc(sizeof(MDeg) + sizeof(uint32_t) * (c+1));
    if(!m)
        return NULL;
    m->c = c;
    mdeg_zero(m);

    return m;
}

/* usage: Release a struct MDeg
 * params:
 *      1) d: ptr to struct MDeg
 * return: void */
void
mdeg_free(MDeg* d) {
    free(d);
}

/* usage: Copy a MDeg from src into dst
 * params:
 *      1) dst: ptr to struct MDeg
 *      2) src: ptr to struct MDeg
 * return: void */
void
mdeg_copy(MDeg* restrict dst, const MDeg* restrict src) {
    assert(mdeg_c(dst) == mdeg_c(src));
    for(uint32_t i = 0; i <= mdeg_c(dst); ++i)
        mdeg_set_deg(dst, i, mdeg_deg(src, i));
}

/* usage: Given a MDeg, create a deep copy of it
 * params:
 *      1) d: ptr to struct MDeg
 * return: ptr to struct MDeg */
MDeg*
mdeg_dup(const MDeg* d) {
    return mdeg_create(d->c, d->d);
}

/* usage: Given a struct MDeg, increment the degree of the group of linear variables
 * params:
 *      1) d: ptr to struct MDeg
 * return: void */
void
mdeg_lv_deg_inc(MDeg* d) {
    ++(d->d[0]);
}

/* usage: Given a struct MDeg, decrement the degree of the group of linear variables
 * params:
 *      1) d: ptr to struct MDeg
 * return: void */
void
mdeg_lv_deg_dec(MDeg* d) {
    --(d->d[0]);
}

/* usage: Given a struct MDeg, increment the degree of the selected group of kernel variables
 * params:
 *      1) d: ptr to struct MDeg
 *      2) i: index of the selected group of kernel variables. 0 for the 1st group.
 * return: void */
void
mdeg_kv_deg_inc(MDeg* d, uint32_t i) {
    ++(d->d[1+i]);
}

/* usage: Given a struct MDeg, decrement the degree of the selected group of kernel variables
 * params:
 *      1) d: ptr to struct MDeg
 *      2) i: index of the selected group of kernel variables. 0 for the 1st group.
 * return: void */
void
mdeg_kv_deg_dec(MDeg* d, uint32_t i) {
    --(d->d[1+i]);
}

/* usage: Given the max multi-degree, and the current multi-degree, raise the
 *      multi-degree to the next one incrementally
 * params:
 *      1) mdeg: the current multi-degree, which will be updated to the next
 *          one after the function returns
 *      2) max_mdeg: the max multi-degree to consider
 * return: true if the current multi-degree is not the max multi-degree. false
 *      otherwise */
bool
mdeg_next(MDeg* restrict mdeg, const MDeg* restrict max_mdeg) {
    assert(mdeg_c(mdeg) == mdeg_c(max_mdeg));
    bool max_d = true;

    for(uint32_t i = 0; i <= mdeg_c(mdeg) && max_d; ++i) {
        assert(mdeg_deg(mdeg, i) <= mdeg_deg(max_mdeg, i));
        if(likely(mdeg_deg(mdeg, i) < mdeg_deg(max_mdeg, i))) {
            mdeg_deg_inc(mdeg, i);
            max_d = false;
        } else
            mdeg_set_deg(mdeg, i, 0);
    }
    return !max_d;
}

/* usage: Given a struct MDeg, print it
 * params:
 *      1): mdeg: ptr to struct MDeg
 * return: void */
void
mdeg_print(const MDeg* mdeg) {
    printf("[");
    for(uint32_t i = 0; i <= mdeg_c(mdeg); ++i) {
        printf("%d, ", mdeg_deg(mdeg, i));
    }
    printf("]\n");
}

/* usage: Given an array of MDeg, find the common multi-degree that defines
 *      the intersection of monomials defined by individual multi-degrees.
 *      The multi-degrees must have the same number of kernel variables (c).
 * params:
 *      1) out: ptr to struct MDeg. Storage for the minimal common multi-degree.
 *          whose number of groups of kernel variables (c) must be the same
 *          as multi-degrees in the array.
 *      2) mdeg_arr: an array of ptrs to struct MDeg
 *      3) sz: size of mdeg_arr
 * return: 0 on success. non-zero value on error */
int32_t
mdeg_find_min_mdeg(MDeg* out, const MDeg** mdeg_arr, uint32_t sz) {
    assert(sz);

    mdeg_copy(out, mdeg_arr[0]);
    const uint32_t c = mdeg_c(out);
    for(uint32_t i = 1; i < sz; ++i) {
        const MDeg* d = mdeg_arr[i];
        if(c != mdeg_c(d)) {
            return 1;
        }

        for(uint32_t j = 0; j <= c; ++j) {
            uint32_t cur_d = mdeg_deg(d, j);
            if(cur_d < mdeg_deg(out, j))
                mdeg_set_deg(out, j, cur_d);
        }
    }

    return 0;
}

/* usage: Given an array of MDeg, find the minimail multi-degree that defines a
 *      set of monomials is that larger or equal to the union of monomials that
 *      all individual multi-degrees.  The multi-degrees must have the same
 *      number of kernel variables (c).
 * params:
 *      1) out: ptr to struct MDeg. Storage for the common multi-degree.
 *          whose number of groups of kernel variables (c) must be the same
 *          as multi-degrees in the array.
 *      2) mdeg_arr: an array of ptrs to struct MDeg
 *      3) sz: size of mdeg_arr
 * return: 0 on success. non-zero value on error */
int32_t
mdeg_find_max_mdeg(MDeg* out, const MDeg** mdeg_arr, uint32_t sz) {
    assert(sz);

    mdeg_copy(out, mdeg_arr[0]);
    const uint32_t c = mdeg_c(out);
    for(uint32_t i = 1; i < sz; ++i) {
        const MDeg* d = mdeg_arr[i];
        if(c != mdeg_c(d)) {
            return 1;
        }

        for(uint32_t j = 0; j <= c; ++j) {
            uint32_t cur_d = mdeg_deg(d, j);
            if(cur_d > mdeg_deg(out, j))
                mdeg_set_deg(out, j, cur_d);
        }
    }

    return 0;
}

/* usage: Given a multi-degree, compute the number of its sub-degrees. For
 *      example (1, 2, 1), its sub-degrees are: (0, 0, 0), (0, 0, 1), (0, 1, 0),
 *      (0, 1, 1), (0, 2, 0), (0, 2, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0),
 *      (1, 1, 1), (1, 2, 0), (1, 2, 1)
 * params:
 *      1) d: ptr to struct MDeg
 * return: number of sub-degrees */
uint32_t
mdeg_num_subdegs(const MDeg* d) {
    uint32_t n = mdeg_lv_deg(d) + 1;
    for(uint32_t i = 0; i < mdeg_c(d); ++i)
        n *= mdeg_kv_deg(d, i) + 1;
    return n;
}

/* usage: Given multi-degrees d1 and d2 where d1 <= d2, compute the number of
 *      sub-degrees defined by d2 but not by d1. For example, let d1 = (1, 1, 1)
 *      d2 = (1, 2, 1), then such sub-degrees are: (0, 2, 0), (0, 2, 1),
 *      (1, 2, 0), (1, 2, 1)
 * params:
 *      1) d1: ptr to struct MDeg
 *      2) d2: ptr to struct MDeg
 * return: number of sub-degrees */
uint32_t
mdeg_num_subdegs_complement(const MDeg* restrict d1, const MDeg* restrict d2) {
    assert(mdeg_c(d1) == mdeg_c(d2));
    assert(mdeg_is_le(d1, d2));
    uint32_t n2 = mdeg_num_subdegs(d2);
    uint32_t n1 = mdeg_num_subdegs(d1);
    assert(n2 >= n1);
    return n2 - n1;
}

/* usage: Given a multi-degree, compute the number of bits necessary to
 *      represent all of its subdegree
 * params:
 *      1) d: ptr to struct MDeg
 *      2) out: an uint8_t array as large as the number of kernel variables
 *          in d plus 1 (i.e. c + 1). Storage for the result. On return,
 *          out[i] stores the number of bits necessary for the i-th degree in d.
 * return: the total number of bits needed */
/*
uint32_t
mdeg_to_int_calc_bits(const MDeg* restrict d, uint8_t* restrict out) {
    uint32_t n = 0;
    for(uint32_t i = 0; i <= mdeg_c(d); ++i) {
        // To get log2(next power of 2 for x), we can compute log2(x-1) + 1
        out[i] = (uint32_t) log2(mdeg_deg(d, i) + 1 - 1) + 1;
        n += out[i];
    }
    return n;
}
*/

static inline bool
do_nothing(MDeg* restrict d, uint64_t idx, void* restrict arg) {
    (void) d; (void) idx; (void) arg;
    return false;
}

/* usage: Given an array of multi-degrees, compute the size of the union
 *      of their sub-degrees
 * params:
 *      1) degs: an array of ptrs to struct MDeg. Those multi-degrees must
 *          have the same number of groups of kernel variables (c)
 *      2) sz: size of degs
 * return: number of sub-degrees in the union */
uint64_t
mdeg_num_subdegs_union(const MDeg** degs, uint32_t sz) {
    assert(sz);
    return mdeg_iter_subdegs_union(degs, sz, do_nothing, NULL);
}

/* usage: Given an array of multi-degrees, iterate over all multi-degrees
 *      in their union and call the callback function on each multi-degree
 * params:
 *      1) degs: an array of ptrs to struct MDeg. Those multi-degrees must
 *          have the same number of groups of kernel variables (c)
 *      2) sz: size of degs
 *      3) cb: the callback function. it takes 3 parameters and returns a
 *          boolean value:
 *              1st param: ptr to mdeg which stores the current sub-degree
 *              2nd param: index of the sub-degree
 *              3rd param: a generic ptr
 *          if the return value is false, then the iteration continues. if true,
 *          then the iteration stops immediately
 *      4) arg: a void ptr used to pass extra parameters to the callback
 *              function, and store its return values
 * return: number of sub-degrees in the union */
uint64_t
mdeg_iter_subdegs_union(const MDeg** degs, uint32_t sz, mdeg_iter_cb_t* cb,
                        void* restrict arg) {
    assert(sz);
    const MDeg* cur_d = degs[0];
    mdeg_create_static_buf(max_deg_buf, mdeg_c(cur_d));
    MDeg* max_d = mdeg_create_from_arr(mdeg_c(cur_d), max_deg_buf);
    mdeg_find_max_mdeg(max_d, degs, sz);

    mdeg_create_static_buf(tmp_deg_buf, mdeg_c(cur_d));
    MDeg* tmp_d = mdeg_create_from_arr(mdeg_c(cur_d), tmp_deg_buf);
    mdeg_zero(tmp_d);
    uint64_t num = 0;
    if(cb(tmp_d, num++, arg)) // (0, 0, .. , 0) is common for all multi-degrees
        return num;

    while(mdeg_next(tmp_d, max_d)) {
        if(mdeg_is_le_any(tmp_d, degs, sz))
            if(cb(tmp_d, num++, arg))
                return num;
    }

    return num;
}

/* usage: Given a multi-degree d, iterate over all multi-degrees <= d
 *      call the callback function on each multi-degree
 * params:
 *      1) d: ptr to struct MDeg
 *      2) cb: the callback function. It takes 3 parameters:
 *          1st param: ptr to MDeg which stores the current sub-degree
 *          2nd param: index of the sub-degree
 *          3rd param: a generic ptr
 *      3) arg: a void ptr used to pass extra parameters to the callback
 *              function, and store its return values
 * return: number of sub-degrees in the union */
uint64_t
mdeg_iter_subdegs(const MDeg* restrict d, mdeg_iter_cb_t * cb,
                  void* restrict arg) {
    return mdeg_iter_subdegs_union((const MDeg**) &d, 1, cb, arg);
}

/* usage: Given 2 multi-degrees d1 and d2, check if d1 is less than or equal
 *      to d2. I.e. any degree for a group of variables in d1 <= that of d2.
 * params:
 *      1) d1: ptr to struct MDeg
 *      2) d2: ptr to struct MDeg
 * return: true if d1 <= d2. Otherwise false */
bool
mdeg_is_le(const MDeg* restrict d1, const MDeg* restrict d2) {
    assert(mdeg_c(d1) == mdeg_c(d2));
    for(uint32_t i = 0; i <= mdeg_c(d1); ++i) {
        if(mdeg_deg(d1, i) > mdeg_deg(d2, i))
            return false;
    }
    return true;
}

/* usage: Given a multi-degree d, and an array of multi-degrees, check if
 *      d is less than or equal to any of the multi-degrees in the array
 * params:
 *      1) d: ptr to struct MDeg
 *      2) degs: an array of ptrs to struct MDeg
 *      3) sz: size of degs
 * return: true or false */
bool
mdeg_is_le_any(const MDeg* restrict d, const MDeg** degs, uint32_t sz) {
    for(uint32_t i = 0; i < sz; ++i) {
        if(mdeg_is_le(d, degs[i]))
            return true;
    }
    return false;
}

/* usage: Given 2 multi-degrees d1 and d2 with the same number of groups of
 *      kernel variables, check if they are the same
 * params:
 *      1) d1: ptr to struct MDeg
 *      2) d2: ptr to struct MDeg
 * return: true if d1 == d2. Otherwise false */
bool
mdeg_is_equal(const MDeg* d1, const MDeg* d2) {
    assert(mdeg_c(d1) == mdeg_c(d2));
    return !memcmp(d1->d, d2->d, sizeof(uint32_t) * (mdeg_c(d1) + 1));
}

/* usage: Given a multi-degree, compute the number of monomials with that
 *      multi-degree
 * params:
 *      1) d: ptr to struct MDeg
 *      2) vnums: number of variables in each group. Must have at least
 *          mdeg_c(d) + 1 elements
 * return: number of monomials with the given multi-degree */
uint64_t
mdeg_mono_num(const MDeg* restrict d, const uint32_t* restrict vnums) {
    uint64_t n = binom(vnums[0] + mdeg_deg(d, 0) - 1, mdeg_deg(d, 0));
    for(uint32_t i = 1; i <= mdeg_c(d); ++i)
        n *= binom(vnums[i] + mdeg_deg(d, i) - 1, mdeg_deg(d, i));
    return n;
}
