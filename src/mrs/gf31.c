#include "gf31.h"

// NOTE: 0 has no inverse
gf31_t gf31_t_inv_table[31] = {
    0, 1, 16, 21, 8, 25,
    26, 9, 4, 7, 28, 17,
    13, 12, 20, 29, 2, 11,
    19, 18, 14, 3, 24, 27,
    22, 5, 6, 23, 10, 15,
    30
};

gf31_t
gf31_t_inv_by_table(gf31_t a) {
    return gf31_t_inv_table[a];
}

void
gf31_t_arr_muli_scalar(gf31_t* arr, uint32_t sz, gf31_t x) {
    // TODO: optimize
    for(uint32_t i = 0; i < sz; ++i) {
        arr[i] = gf31_t_mul(arr[i], x);
    }
}

void
gf31_t_arr_fmaddi_scalar(gf31_t* a, const gf31_t* b, uint32_t sz, gf31_t c) {
    // compute a += b * c; c is a scalar while a and b are processed element-wise
    if(c == 0)
        return;
    // TODO: optimize
    for(uint32_t i = 0; i < sz; ++i) {
        gf31_t bc = gf31_t_mul(b[i], c);
        a[i] = gf31_t_add(a[i], bc);
    }
}

void
gf31_t_arr_fmaddi_scalar_mask64(gf31_t* a, const gf31_t* b, gf31_t c, uint64_t d) {
    // both a and b are arrays with 64 elements, d is a 64-bit integer that
    // specifies which elements to keep, and which to zero.
    // compute a += b * c; c is a scalar while a and b are processed element-wise
    // if the i-th bit of d is zero, then the i-th element of b*c is added to the
    // i-th element of a
    // if 1, then the i-th element of a is untouched.
    if(c == 0)
        return;
    // TODO: optimize
    for(uint32_t i = 0; i < 64; ++i) {
        if(d & 0x1) {
            gf31_t bc = gf31_t_mul(b[i], c);
            a[i] = gf31_t_add(a[i], bc);
        }
        d >>= 1;
    }
}

void
gf31_t_arr_fmsubi_scalar(gf31_t* a, const gf31_t* b, uint32_t sz, gf31_t c) {
    // compute a -= b * c; c is a scalar while a and b are processed element-wise
    if(c == 0)
        return;
    // TODO: optimize
    for(uint32_t i = 0; i < sz; ++i) {
        gf31_t bc = gf31_t_mul(b[i], c);
        a[i] = gf31_t_sub(a[i], bc);
    }
}

void
gf31_t_arr_fmsubi_scalar_mask64(gf31_t* a, const gf31_t* b, gf31_t c, uint64_t d) {
    // both a and b are gf16_t arrays with 64 elements, d is a 64-bit integer that
    // specifies which elements to keep, and which to zero.
    // compute a -= b * c; c is a scalar while a and b are processed element-wise
    // if the i-th bit of d is zero, then the i-th element of b*c is added to the
    // i-th element of a
    // if 1, then the i-th element of a is untouched.
    if(c == 0)
        return;
    // TODO: optimize
    for(uint32_t i = 0; i < 64; ++i) {
        if(d & 0x1) {
            gf31_t bc = gf31_t_mul(b[i], c);
            a[i] = gf31_t_sub(a[i], bc);
        }
        d >>= 1;
    }
}

uint32_t
gf31_t_arr_nzc(const gf31_t* a, uint32_t sz) {
    uint32_t c = 0;
    for(uint32_t i = 0; i < sz; ++i) {
        if(a[i])
            ++c;
    }
    return c;
}

uint32_t
gf31_t_arr_zc(const gf31_t* a, uint32_t sz) {
    uint32_t c = 0;
    for(uint32_t i = 0; i < sz; ++i) {
        if(!a[i])
            ++c;
    }
    return c;
}
