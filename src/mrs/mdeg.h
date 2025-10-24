#ifndef __MDEG_H__
#define __MDEG_H__

#include <stdint.h>
#include <stdbool.h>

typedef struct MDeg MDeg;

/* ========================================================================
 * function prototypes
 * ======================================================================== */

/* usage: Given a struct MDeg, return the number of subgroups of kernel vars
 * params:
 *      1) d: ptr to struct MDeg
 * return: number of subgroups */
uint32_t
mdeg_c(const MDeg* d);

/* usage: Given a struct MDeg, expose its internal buffer for storing
 *      degrees as a pointer
 * params:
 *      1) d: ptr to struct Mdeg
 * return: a uint32_t ptr to its internal buffer */
const uint32_t*
mdeg_deg_buffer(const MDeg* d);

/* usage: Given a struct MDeg, return the selected degree of group of
 *      variables
 * params:
 *      1) d: ptr to struct MDeg
 *      2) i: index of the group. 0 for linear vars, 1 ~ c for kernel vars
 * return: degree of the selected group of vars */
uint32_t
mdeg_deg(const MDeg* d, uint32_t i);

/* usage: Given a struct MDeg, increment the degree of the selected
 *      group of variables
 * params:
 *      1) d: ptr to struct MDeg
 *      2) i: index of the group. 0 for linear vars, 1 ~ c for kernel vars
 * return: void */
void
mdeg_deg_inc(MDeg* d, uint32_t i);

/* usage: Given a struct MDeg, decrement the degree of the selected
 *      group of variables
 * params:
 *      1) d: ptr to struct MDeg
 *      2) i: index of the group. 0 for linear vars, 1 ~ c for kernel vars
 * return: void */
void
mdeg_deg_dec(MDeg* d, uint32_t i);

/* usage: Given a struct MDeg, set the degree of the selected group of
 *      variables
 * params:
 *      1) d: ptr to struct MDeg
 *      2) i: index of the group. 0 for linear vars, 1 ~ c for kernel vars
 *      3) v: new degree of linear variables
 * return: degree of the selected group of vars */
void
mdeg_set_deg(MDeg* d, uint32_t i, uint32_t v);

/* usage: Given a struct MDeg, set all of its degree to zero
 * params:
 *      1) d: ptr to struct MDeg
 * return: void */
void
mdeg_zero(MDeg* d);

/* usage: Given a struct MDeg, return total degree of all variables
 * params:
 *      1) d: ptr to struct MDeg
 * return: degree of the all vars */
uint32_t
mdeg_total_deg(const MDeg* d);

/* usage: Given a struct MDeg, return the degree of linear variables
 * params:
 *      1) d: ptr to struct MDeg
 * return: degree of linear var */
uint32_t
mdeg_lv_deg(const MDeg* d);

/* usage: Given a struct MDeg, set its degree of linear vars to the
 *      given value
 * params:
 *      1) d: ptr to struct MDeg
 *      2) v: new degree of linear variables
 * return: void */
void
mdeg_set_lv_deg(MDeg* d, uint32_t v);

/* usage: Given a struct MDeg, return the degree of kernel variables
 * params:
 *      1) d: ptr to struct MDeg
 *      2) i: index of the subgroup of kernel vars
 * return: degree of the specified subgroup of kernel vars */
uint32_t
mdeg_kv_deg(const MDeg* d, uint32_t i);

/* usage: Given a struct MDeg, set the degree of selected group of
 *      kernel vars to the given value
 * params:
 *      1) d: ptr to struct MDeg
 *      2) i: index of the subgroup of kernel vars
 *      3) v: new degree for the selected group linear variables
 * return: void */
void
mdeg_set_kv_deg(MDeg* d, uint32_t i, uint32_t v);

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
mdeg_create(uint32_t c, const uint32_t ds[]);

/* usage: Given the number of subgroups of kernel vars, create an uninitialized
 *      static array that can hold the degrees of linear vars and kernel vars,
 *      which can be passed to mdeg_create_from_arr
 * params:
 *      1) name: name of the array
 *      2) c: number of subgroups of kernel vars
 * return: void */
#define mdeg_create_static_buf(buf_name, c) \
    uint32_t buf_name[2+(c)]

/* usage: Given the number of subgroup of kernel vars, and an array storing the
 *      degrees of subgroups, create a struct MDeg from the array. The array
 *      will be modified by this function. The caller must ensure the lifespan
 *      of the array is longer than the resultant struct MDeg.
 * params:
 *      1) c: number of subgroups of kernel vars
 *      2) ds: a uint32_t array of at least size (c+2). ds[1] stores
 *          the degree of linear var, ds[2] the degree of the
 *          1st subgroup of kernel vars, and so on.
 * return: ptr to struct MDeg */
MDeg*
mdeg_create_from_arr(uint32_t c, uint32_t* ds);

/* usage: Given the number of subgroup of kernel vars, create a struct MDeg
 *      where each group of vars has zero degree
 * params:
 *      1) c: number of subgroups of kernel vars
 * return: ptr to struct MDeg */
MDeg*
mdeg_create_zero(uint32_t c);

/* usage: Release a struct MDeg
 * params:
 *      1) d: ptr to struct MDeg
 * return: void */
void
mdeg_free(MDeg* d);

/* usage: Copy a MDeg from src into dst
 * params:
 *      1) dst: ptr to struct MDeg
 *      2) src: ptr to struct MDeg
 * return: void */
void
mdeg_copy(MDeg* restrict dst, const MDeg* restrict src);

/* usage: Given a MDeg, create a deep copy of it
 * params:
 *      1) d: ptr to struct MDeg
 * return: ptr to struct MDeg */
MDeg*
mdeg_dup(const MDeg* d);

/* usage: Given a struct MDeg, increment the degree of the group of linear variables
 * params:
 *      1) d: ptr to struct MDeg
 * return: void */
void
mdeg_lv_deg_inc(MDeg* d);

/* usage: Given a struct MDeg, decrement the degree of the group of linear variables
 * params:
 *      1) d: ptr to struct MDeg
 * return: void */
void
mdeg_lv_deg_dec(MDeg* d);

/* usage: Given a struct MDeg, increment the degree of the selected group of kernel variables
 * params:
 *      1) d: ptr to struct MDeg
 *      2) i: index of the selected group of kernel variables. 0 for the 1st group.
 * return: void */
void
mdeg_kv_deg_inc(MDeg* d, uint32_t i);

/* usage: Given a struct MDeg, decrement the degree of the selected group of kernel variables
 * params:
 *      1) d: ptr to struct MDeg
 *      2) i: index of the selected group of kernel variables. 0 for the 1st group.
 * return: void */
void
mdeg_kv_deg_dec(MDeg* d, uint32_t i);

/* usage: Given the max multi-degree, and the current multi-degree, raise the
 *      multi-degree to the next one incrementally
 * params:
 *      1) mdeg: the current multi-degree, which will be updated to the next
 *          one after the function returns
 *      2) max_mdeg: the max multi-degree to consider
 * return: true if the current multi-degree is not the max multi-degree. false
 *      otherwise */
bool
mdeg_next(MDeg* restrict mdeg, const MDeg* restrict max_mdeg);

/* usage: Given a struct MDeg, print it
 * params:
 *      1): mdeg: ptr to struct MDeg
 * return: void */
void
mdeg_print(const MDeg* mdeg);

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
mdeg_find_min_mdeg(MDeg* out, const MDeg** mdeg_arr, uint32_t sz);

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
mdeg_find_max_mdeg(MDeg* out, const MDeg** mdeg_arr, uint32_t sz);

/* usage: Given an array of multi-degrees, compute the size of the union
 *      of their sub-degrees
 * params:
 *      1) degs: an array of ptrs to struct MDeg. Those multi-degrees must
 *          have the same number of groups of kernel variables (c)
 *      2) sz: size of degs
 * return: number of sub-degrees in the union */
uint64_t
mdeg_num_subdegs_union(const MDeg** degs, uint32_t sz);

/* usage: Given 2 multi-degrees d1 and d2, check if d1 is less than or equal
 *      to d2. I.e. any degree for a group of variables in d1 <= that of d2.
 * params:
 *      1) d1: ptr to struct MDeg
 *      2) d2: ptr to struct MDeg
 * return: true if d1 <= d2. Otherwise false */
bool
mdeg_is_le(const MDeg* restrict d1, const MDeg* restrict d2);

/* usage: Given a multi-degree d, and an array of multi-degrees, check if
 *      d is less than or equal to any of the multi-degrees in the array
 * params:
 *      1) d: ptr to struct MDeg
 *      2) degs: an array of ptrs to struct MDeg
 *      3) sz: size of degs
 * return: true or false */
bool
mdeg_is_le_any(const MDeg* restrict d, const MDeg** degs, uint32_t sz);

/* usage: Given a multi-degree, compute the number of monomials with that
 *      multi-degree
 * params:
 *      1) d: ptr to struct MDeg
 *      2) vnums: number of variables in each group. Must have at least
 *          mdeg_c(d) + 1 elements
 * return: number of monomials with the given multi-degree */
uint64_t
mdeg_mono_num(const MDeg* restrict d, const uint32_t* restrict vnums);

// callback function
typedef bool (mdeg_iter_cb_t) (MDeg* restrict, uint64_t, void* restrict);

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
                        void* restrict arg);

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
                  void* restrict arg);

/* usage: Given 2 multi-degrees d1 and d2, check if d1 is less than or equal
 *      to d2. I.e. any degree for a group of variables in d1 <= that of d2.
 * params:
 *      1) d1: ptr to struct MDeg
 *      2) d2: ptr to struct MDeg
 * return: true if d1 <= d2. Otherwise false */
bool
mdeg_is_le(const MDeg* restrict d1, const MDeg* restrict d2);

/* usage: Given 2 multi-degrees d1 and d2 with the same number of groups of
 *      kernel variables, check if they are the same
 * params:
 *      1) d1: ptr to struct MDeg
 *      2) d2: ptr to struct MDeg
 * return: true if d1 == d2. Otherwise false */
bool
mdeg_is_equal(const MDeg* d1, const MDeg* d2);


/* usage: Given a multi-degree, check if it's linear. I.e. total degree <= 1
 * params:
 *      1) d: ptr to struct MDeg
 * return: true or false */
static inline bool
mdeg_is_linear(const MDeg* d) {
    if(mdeg_total_deg(d) < 2)
        return true;
    else
        return false;
}

/* usage: Given a multi-degree, check if it's non-linear. I.e. total degree >= 2
 * params:
 *      1) d: ptr to struct MDeg
 * return: true or false */
static inline bool
mdeg_is_nonlinear(const MDeg* d) {
    if(mdeg_total_deg(d) >= 2)
        return true;
    else
        return false;
}

#endif // __MDEG_H__
