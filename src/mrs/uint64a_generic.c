/* uint64a_generic.c: implementation of bit operations on an array of uint64_t
 * for generic CPUs */

#include <stdint.h>
#include <assert.h>

#include "uint64a.h"
#include "util.h"

// TODO: try to compute solution on the fly to free up registers, and check if
// it's faster
int64_t
uint64a_gj_v1_generic(const uint64_t m[2], uint64_t* const restrict sol) {
    uint64_t const_col = m[0];
    uint64_t c1 = m[1]; // x1
    uint64_t lsb1 = uint64_t_lsb(c1); // initially all eqs are candidates
    if(unlikely(!lsb1)) {
        return -1; // singular
    }

    if(const_col & lsb1) const_col ^= c1 ^ lsb1;
    uint64_t mask = ~lsb1;

    if(likely(mask & const_col)) { // check system consistency
        return mask & const_col; // not solvable
    }

    uint64_t s = 0x0ULL;
    if(const_col & lsb1) s = uint64_t_toggle_at(s, 0);
    *sol = s;
    return 0;
}

int64_t
uint64a_gj_v2_generic(const uint64_t m[3], uint64_t* const restrict sol) {
    uint64_t const_col = m[0];
    uint64_t c1 = m[1]; // x1
    uint64_t c2 = m[2]; // x2
    uint64_t lsb1 = uint64_t_lsb(c1); // initially all eqs are candidates
    if(unlikely(!lsb1)) {
        return -1; // singular
    }

    uint64_t c1_reduc = c1 ^ lsb1; // for row reduction
    if(const_col & lsb1) const_col ^= c1_reduc;
    if(c2 & lsb1) c2 ^= c1_reduc;
    uint64_t mask = ~lsb1;

    uint64_t lsb2 = uint64_t_lsb(c2 & mask); // x2
    if(unlikely(!lsb2)) {
        return -1; // singular
    }

    if(const_col & lsb2) const_col ^= c2 ^ lsb2;
    mask ^= lsb2;

    if(likely(mask & const_col)) { // check system consistency
        return mask & const_col; // not solvable
    }

    uint64_t s = 0x0ULL;
    if(const_col & lsb1) s = uint64_t_toggle_at(s, 0);
    if(const_col & lsb2) s = uint64_t_toggle_at(s, 1);
    *sol = s;
    return 0;
}

int64_t
uint64a_gj_v3_generic(const uint64_t m[4], uint64_t* const restrict sol) {
    uint64_t const_col = m[0];
    uint64_t c1 = m[1]; // x1
    uint64_t c2 = m[2]; // x2
    uint64_t c3 = m[3]; // x3
    uint64_t lsb1 = uint64_t_lsb(c1); // initially all eqs are candidates
    if(unlikely(!lsb1)) {
        return -1; // singular
    }

    uint64_t c1_reduc = c1 ^ lsb1; // for row reduction
    if(const_col & lsb1) const_col ^= c1_reduc;
    if(c2 & lsb1) c2 ^= c1_reduc;
    if(c3 & lsb1) c3 ^= c1_reduc;
    uint64_t mask = ~lsb1;

    uint64_t lsb2 = uint64_t_lsb(c2 & mask); // x2
    if(unlikely(!lsb2)) {
        return -1; // singular
    }

    uint64_t c2_reduc = c2 ^ lsb2;
    if(const_col & lsb2) const_col ^= c2_reduc;
    if(c3 & lsb2) c3 ^= c2_reduc;
    mask ^= lsb2;

    uint64_t lsb3 = uint64_t_lsb(c3 & mask); // x3
    if(unlikely(!lsb3)) {
        return -1; // singular
    }

    if(const_col & lsb3) const_col ^= c3 ^ lsb3;
    mask ^= lsb3;

    if(likely(mask & const_col)) { // check system consistency
        return mask & const_col; // not solvable
    }

    uint64_t s = 0x0ULL;
    if(const_col & lsb1) s = uint64_t_toggle_at(s, 0);
    if(const_col & lsb2) s = uint64_t_toggle_at(s, 1);
    if(const_col & lsb3) s = uint64_t_toggle_at(s, 2);
    *sol = s;
    return 0;
}

int64_t
uint64a_gj_v4_generic(const uint64_t m[5], uint64_t* const restrict sol) {
    uint64_t const_col = m[0];
    uint64_t c1 = m[1]; // x1
    uint64_t c2 = m[2]; // x2
    uint64_t c3 = m[3]; // x3
    uint64_t c4 = m[4]; // x4
    uint64_t lsb1 = uint64_t_lsb(c1); // initially all eqs are candidates
    if(unlikely(!lsb1)) {
        return -1; // singular
    }

    uint64_t c1_reduc = c1 ^ lsb1; // for row reduction
    if(const_col & lsb1) const_col ^= c1_reduc;
    if(c2 & lsb1) c2 ^= c1_reduc;
    if(c3 & lsb1) c3 ^= c1_reduc;
    if(c4 & lsb1) c4 ^= c1_reduc;
    uint64_t mask = ~lsb1;

    uint64_t lsb2 = uint64_t_lsb(c2 & mask); // x2
    if(unlikely(!lsb2)) {
        return -1; // singular
    }

    uint64_t c2_reduc = c2 ^ lsb2;
    if(const_col & lsb2) const_col ^= c2_reduc;
    if(c3 & lsb2) c3 ^= c2_reduc;
    if(c4 & lsb2) c4 ^= c2_reduc;
    mask ^= lsb2;

    uint64_t lsb3 = uint64_t_lsb(c3 & mask); // x3
    if(unlikely(!lsb3)) {
        return -1; // singular
    }

    uint64_t c3_reduc = c3 ^ lsb3;
    if(const_col & lsb3) const_col ^= c3_reduc;
    if(c4 & lsb3) c4 ^= c3_reduc;
    mask ^= lsb3;

    uint64_t lsb4 = uint64_t_lsb(c4 & mask); // x4
    if(unlikely(!lsb4)) {
        return -1; // singular
    }

    if(const_col & lsb4) const_col ^= c4 ^ lsb4;
    mask ^= lsb4;

    if(likely(mask & const_col)) { // check system consistency
        return mask & const_col; // not solvable
    }

    uint64_t s = 0x0ULL;
    if(const_col & lsb1) s = uint64_t_toggle_at(s, 0);
    if(const_col & lsb2) s = uint64_t_toggle_at(s, 1);
    if(const_col & lsb3) s = uint64_t_toggle_at(s, 2);
    if(const_col & lsb4) s = uint64_t_toggle_at(s, 3);
    *sol = s;
    return 0;
}

int64_t
uint64a_gj_v5_generic(const uint64_t m[6], uint64_t* const restrict sol) {
    uint64_t const_col = m[0];
    uint64_t c1 = m[1]; // x1
    uint64_t c2 = m[2]; // x2
    uint64_t c3 = m[3]; // x3
    uint64_t c4 = m[4]; // x4
    uint64_t c5 = m[5]; // x5
    uint64_t lsb1 = uint64_t_lsb(c1); // initially all eqs are candidates
    if(unlikely(!lsb1)) {
        return -1; // singular
    }

    uint64_t c1_reduc = c1 ^ lsb1; // for row reduction
    if(const_col & lsb1) const_col ^= c1_reduc;
    if(c2 & lsb1) c2 ^= c1_reduc;
    if(c3 & lsb1) c3 ^= c1_reduc;
    if(c4 & lsb1) c4 ^= c1_reduc;
    if(c5 & lsb1) c5 ^= c1_reduc;
    uint64_t mask = ~lsb1;

    uint64_t lsb2 = uint64_t_lsb(c2 & mask); // x2
    if(unlikely(!lsb2)) {
        return -1; // singular
    }

    uint64_t c2_reduc = c2 ^ lsb2;
    if(const_col & lsb2) const_col ^= c2_reduc;
    if(c3 & lsb2) c3 ^= c2_reduc;
    if(c4 & lsb2) c4 ^= c2_reduc;
    if(c5 & lsb2) c5 ^= c2_reduc;
    mask ^= lsb2;

    uint64_t lsb3 = uint64_t_lsb(c3 & mask); // x3
    if(unlikely(!lsb3)) {
        return -1; // singular
    }

    uint64_t c3_reduc = c3 ^ lsb3;
    if(const_col & lsb3) const_col ^= c3_reduc;
    if(c4 & lsb3) c4 ^= c3_reduc;
    if(c5 & lsb3) c5 ^= c3_reduc;
    mask ^= lsb3;

    uint64_t lsb4 = uint64_t_lsb(c4 & mask); // x4
    if(unlikely(!lsb4)) {
        return -1; // singular
    }

    uint64_t c4_reduc = c4 ^ lsb4;
    if(const_col & lsb4) const_col ^= c4_reduc;
    if(c5 & lsb4) c5 ^= c4_reduc;
    mask ^= lsb4;

    uint64_t lsb5 = uint64_t_lsb(c5 & mask); // x5
    if(unlikely(!lsb5)) {
        return -1; // singular
    }

    if(const_col & lsb5) const_col ^= c5 ^ lsb5;
    mask ^= lsb5;

    if(likely(mask & const_col)) { // check system consistency
        return mask & const_col; // not solvable
    }

    uint64_t s = 0x0ULL;
    if(const_col & lsb1) s = uint64_t_toggle_at(s, 0);
    if(const_col & lsb2) s = uint64_t_toggle_at(s, 1);
    if(const_col & lsb3) s = uint64_t_toggle_at(s, 2);
    if(const_col & lsb4) s = uint64_t_toggle_at(s, 3);
    if(const_col & lsb5) s = uint64_t_toggle_at(s, 4);
    *sol = s;
    return 0;
}

int64_t
uint64a_gj_v6_generic(const uint64_t m[7], uint64_t* const restrict sol) {
    uint64_t const_col = m[0];
    uint64_t c1 = m[1]; // x1
    uint64_t c2 = m[2]; // x2
    uint64_t c3 = m[3]; // x3
    uint64_t c4 = m[4]; // x4
    uint64_t c5 = m[5]; // x5
    uint64_t c6 = m[6]; // x6
    uint64_t lsb1 = uint64_t_lsb(c1); // initially all eqs are candidates
    if(unlikely(!lsb1)) {
        return -1; // singular
    }

    uint64_t c1_reduc = c1 ^ lsb1; // for row reduction
    if(const_col & lsb1) const_col ^= c1_reduc;
    if(c2 & lsb1) c2 ^= c1_reduc;
    if(c3 & lsb1) c3 ^= c1_reduc;
    if(c4 & lsb1) c4 ^= c1_reduc;
    if(c5 & lsb1) c5 ^= c1_reduc;
    if(c6 & lsb1) c6 ^= c1_reduc;
    uint64_t mask = ~lsb1;

    uint64_t lsb2 = uint64_t_lsb(c2 & mask); // x2
    if(unlikely(!lsb2)) {
        return -1; // singular
    }

    uint64_t c2_reduc = c2 ^ lsb2;
    if(const_col & lsb2) const_col ^= c2_reduc;
    if(c3 & lsb2) c3 ^= c2_reduc;
    if(c4 & lsb2) c4 ^= c2_reduc;
    if(c5 & lsb2) c5 ^= c2_reduc;
    if(c6 & lsb2) c6 ^= c2_reduc;
    mask ^= lsb2;

    uint64_t lsb3 = uint64_t_lsb(c3 & mask); // x3
    if(unlikely(!lsb3)) {
        return -1; // singular
    }

    uint64_t c3_reduc = c3 ^ lsb3;
    if(const_col & lsb3) const_col ^= c3_reduc;
    if(c4 & lsb3) c4 ^= c3_reduc;
    if(c5 & lsb3) c5 ^= c3_reduc;
    if(c6 & lsb3) c6 ^= c3_reduc;
    mask ^= lsb3;

    uint64_t lsb4 = uint64_t_lsb(c4 & mask); // x4
    if(unlikely(!lsb4)) {
        return -1; // singular
    }

    uint64_t c4_reduc = c4 ^ lsb4;
    if(const_col & lsb4) const_col ^= c4_reduc;
    if(c5 & lsb4) c5 ^= c4_reduc;
    if(c6 & lsb4) c6 ^= c4_reduc;
    mask ^= lsb4;

    uint64_t lsb5 = uint64_t_lsb(c5 & mask); // x5
    if(unlikely(!lsb5)) {
        return -1; // singular
    }

    uint64_t c5_reduc = c5 ^ lsb5;
    if(const_col & lsb5) const_col ^= c5_reduc;
    if(c6 & lsb5) c6 ^= c5_reduc;
    mask ^= lsb5;

    uint64_t lsb6 = uint64_t_lsb(c6 & mask); // x6
    if(unlikely(!lsb6)) {
        return -1; // singular
    }

    if(const_col & lsb6) const_col ^= c6 ^ lsb6;
    mask ^= lsb6;

    if(likely(mask & const_col)) { // check system consistency
        return mask & const_col; // not solvable
    }

    uint64_t s = 0x0ULL;
    if(const_col & lsb1) s = uint64_t_toggle_at(s, 0);
    if(const_col & lsb2) s = uint64_t_toggle_at(s, 1);
    if(const_col & lsb3) s = uint64_t_toggle_at(s, 2);
    if(const_col & lsb4) s = uint64_t_toggle_at(s, 3);
    if(const_col & lsb5) s = uint64_t_toggle_at(s, 4);
    if(const_col & lsb6) s = uint64_t_toggle_at(s, 5);
    *sol = s;
    return 0;
}

int64_t
uint64a_gj_v7_generic(const uint64_t m[8], uint64_t* const restrict sol) {
    uint64_t const_col = m[0];
    uint64_t c1 = m[1]; // x1
    uint64_t c2 = m[2]; // x2
    uint64_t c3 = m[3]; // x3
    uint64_t c4 = m[4]; // x4
    uint64_t c5 = m[5]; // x5
    uint64_t c6 = m[6]; // x6
    uint64_t c7 = m[7]; // x7
    uint64_t lsb1 = uint64_t_lsb(c1); // initially all eqs are candidates
    if(unlikely(!lsb1)) {
        return -1; // singular
    }

    uint64_t c1_reduc = c1 ^ lsb1; // for row reduction
    if(const_col & lsb1) const_col ^= c1_reduc;
    if(c2 & lsb1) c2 ^= c1_reduc;
    if(c3 & lsb1) c3 ^= c1_reduc;
    if(c4 & lsb1) c4 ^= c1_reduc;
    if(c5 & lsb1) c5 ^= c1_reduc;
    if(c6 & lsb1) c6 ^= c1_reduc;
    if(c7 & lsb1) c7 ^= c1_reduc;
    uint64_t mask = ~lsb1;

    uint64_t lsb2 = uint64_t_lsb(c2 & mask); // x2
    if(unlikely(!lsb2)) {
        return -1; // singular
    }

    uint64_t c2_reduc = c2 ^ lsb2;
    if(const_col & lsb2) const_col ^= c2_reduc;
    if(c3 & lsb2) c3 ^= c2_reduc;
    if(c4 & lsb2) c4 ^= c2_reduc;
    if(c5 & lsb2) c5 ^= c2_reduc;
    if(c6 & lsb2) c6 ^= c2_reduc;
    if(c7 & lsb2) c7 ^= c2_reduc;
    mask ^= lsb2;

    uint64_t lsb3 = uint64_t_lsb(c3 & mask); // x3
    if(unlikely(!lsb3)) {
        return -1; // singular
    }

    uint64_t c3_reduc = c3 ^ lsb3;
    if(const_col & lsb3) const_col ^= c3_reduc;
    if(c4 & lsb3) c4 ^= c3_reduc;
    if(c5 & lsb3) c5 ^= c3_reduc;
    if(c6 & lsb3) c6 ^= c3_reduc;
    if(c7 & lsb3) c7 ^= c3_reduc;
    mask ^= lsb3;

    uint64_t lsb4 = uint64_t_lsb(c4 & mask); // x4
    if(unlikely(!lsb4)) {
        return -1; // singular
    }

    uint64_t c4_reduc = c4 ^ lsb4;
    if(const_col & lsb4) const_col ^= c4_reduc;
    if(c5 & lsb4) c5 ^= c4_reduc;
    if(c6 & lsb4) c6 ^= c4_reduc;
    if(c7 & lsb4) c7 ^= c4_reduc;
    mask ^= lsb4;

    uint64_t lsb5 = uint64_t_lsb(c5 & mask); // x5
    if(unlikely(!lsb5)) {
        return -1; // singular
    }

    uint64_t c5_reduc = c5 ^ lsb5;
    if(const_col & lsb5) const_col ^= c5_reduc;
    if(c6 & lsb5) c6 ^= c5_reduc;
    if(c7 & lsb5) c7 ^= c5_reduc;
    mask ^= lsb5;

    uint64_t lsb6 = uint64_t_lsb(c6 & mask); // x6
    if(unlikely(!lsb6)) {
        return -1; // singular
    }

    uint64_t c6_reduc = c6 ^ lsb6;
    if(const_col & lsb6) const_col ^= c6_reduc;
    if(c7 & lsb6) c7 ^= c6_reduc;
    mask ^= lsb6;

    uint64_t lsb7 = uint64_t_lsb(c7 & mask); // x7
    if(unlikely(!lsb7)) {
        return -1; // singular
    }

    if(const_col & lsb7) const_col ^= c7 ^ lsb7;
    mask ^= lsb7;

    if(likely(mask & const_col)) { // check system consistency
        return mask & const_col; // not solvable
    }

    uint64_t s = 0x0ULL;
    if(const_col & lsb1) s = uint64_t_toggle_at(s, 0);
    if(const_col & lsb2) s = uint64_t_toggle_at(s, 1);
    if(const_col & lsb3) s = uint64_t_toggle_at(s, 2);
    if(const_col & lsb4) s = uint64_t_toggle_at(s, 3);
    if(const_col & lsb5) s = uint64_t_toggle_at(s, 4);
    if(const_col & lsb6) s = uint64_t_toggle_at(s, 5);
    if(const_col & lsb7) s = uint64_t_toggle_at(s, 6);
    *sol = s;
    return 0;
}

int64_t
uint64a_gj_v8_generic(const uint64_t m[9], uint64_t* const restrict sol) {
    uint64_t const_col = m[0];
    uint64_t c1 = m[1]; // x1
    uint64_t c2 = m[2]; // x2
    uint64_t c3 = m[3]; // x3
    uint64_t c4 = m[4]; // x4
    uint64_t c5 = m[5]; // x5
    uint64_t c6 = m[6]; // x6
    uint64_t c7 = m[7]; // x7
    uint64_t c8 = m[8]; // x8
    uint64_t lsb1 = uint64_t_lsb(c1); // initially all eqs are candidates
    if(unlikely(!lsb1)) {
        return -1; // singular
    }

    uint64_t c1_reduc = c1 ^ lsb1; // for row reduction
    if(const_col & lsb1) const_col ^= c1_reduc;
    if(c2 & lsb1) c2 ^= c1_reduc;
    if(c3 & lsb1) c3 ^= c1_reduc;
    if(c4 & lsb1) c4 ^= c1_reduc;
    if(c5 & lsb1) c5 ^= c1_reduc;
    if(c6 & lsb1) c6 ^= c1_reduc;
    if(c7 & lsb1) c7 ^= c1_reduc;
    if(c8 & lsb1) c8 ^= c1_reduc;
    uint64_t mask = ~lsb1;

    uint64_t lsb2 = uint64_t_lsb(c2 & mask); // x2
    if(unlikely(!lsb2)) {
        return -1; // singular
    }

    uint64_t c2_reduc = c2 ^ lsb2;
    if(const_col & lsb2) const_col ^= c2_reduc;
    if(c3 & lsb2) c3 ^= c2_reduc;
    if(c4 & lsb2) c4 ^= c2_reduc;
    if(c5 & lsb2) c5 ^= c2_reduc;
    if(c6 & lsb2) c6 ^= c2_reduc;
    if(c7 & lsb2) c7 ^= c2_reduc;
    if(c8 & lsb2) c8 ^= c2_reduc;
    mask ^= lsb2;

    uint64_t lsb3 = uint64_t_lsb(c3 & mask); // x3
    if(unlikely(!lsb3)) {
        return -1; // singular
    }

    uint64_t c3_reduc = c3 ^ lsb3;
    if(const_col & lsb3) const_col ^= c3_reduc;
    if(c4 & lsb3) c4 ^= c3_reduc;
    if(c5 & lsb3) c5 ^= c3_reduc;
    if(c6 & lsb3) c6 ^= c3_reduc;
    if(c7 & lsb3) c7 ^= c3_reduc;
    if(c8 & lsb3) c8 ^= c3_reduc;
    mask ^= lsb3;

    uint64_t lsb4 = uint64_t_lsb(c4 & mask); // x4
    if(unlikely(!lsb4)) {
        return -1; // singular
    }

    uint64_t c4_reduc = c4 ^ lsb4;
    if(const_col & lsb4) const_col ^= c4_reduc;
    if(c5 & lsb4) c5 ^= c4_reduc;
    if(c6 & lsb4) c6 ^= c4_reduc;
    if(c7 & lsb4) c7 ^= c4_reduc;
    if(c8 & lsb4) c8 ^= c4_reduc;
    mask ^= lsb4;

    uint64_t lsb5 = uint64_t_lsb(c5 & mask); // x5
    if(unlikely(!lsb5)) {
        return -1; // singular
    }

    uint64_t c5_reduc = c5 ^ lsb5;
    if(const_col & lsb5) const_col ^= c5_reduc;
    if(c6 & lsb5) c6 ^= c5_reduc;
    if(c7 & lsb5) c7 ^= c5_reduc;
    if(c8 & lsb5) c8 ^= c5_reduc;
    mask ^= lsb5;

    uint64_t lsb6 = uint64_t_lsb(c6 & mask); // x6
    if(unlikely(!lsb6)) {
        return -1; // singular
    }

    uint64_t c6_reduc = c6 ^ lsb6;
    if(const_col & lsb6) const_col ^= c6_reduc;
    if(c7 & lsb6) c7 ^= c6_reduc;
    if(c8 & lsb6) c8 ^= c6_reduc;
    mask ^= lsb6;

    uint64_t lsb7 = uint64_t_lsb(c7 & mask); // x7
    if(unlikely(!lsb7)) {
        return -1; // singular
    }

    uint64_t c7_reduc = c7 ^ lsb7;
    if(const_col & lsb7) const_col ^= c7_reduc;
    if(c8 & lsb7) c8 ^= c7_reduc;
    mask ^= lsb7;

    uint64_t lsb8 = uint64_t_lsb(c8 & mask); // x8
    if(unlikely(!lsb8)) {
        return -1; // singular
    }

    if(const_col & lsb8) const_col ^= c8 ^ lsb8;
    mask ^= lsb8;

    if(likely(mask & const_col)) { // check system consistency
        return mask & const_col; // not solvable
    }

    uint64_t s = 0x0ULL;
    if(const_col & lsb1) s = uint64_t_toggle_at(s, 0);
    if(const_col & lsb2) s = uint64_t_toggle_at(s, 1);
    if(const_col & lsb3) s = uint64_t_toggle_at(s, 2);
    if(const_col & lsb4) s = uint64_t_toggle_at(s, 3);
    if(const_col & lsb5) s = uint64_t_toggle_at(s, 4);
    if(const_col & lsb6) s = uint64_t_toggle_at(s, 5);
    if(const_col & lsb7) s = uint64_t_toggle_at(s, 6);
    if(const_col & lsb8) s = uint64_t_toggle_at(s, 7);
    *sol = s;
    return 0;
}

int64_t
uint64a_gj_v9_generic(const uint64_t m[10], uint64_t* const restrict sol) {
    uint64_t const_col = m[0];
    uint64_t c1 = m[1]; // x1
    uint64_t c2 = m[2]; // x2
    uint64_t c3 = m[3]; // x3
    uint64_t c4 = m[4]; // x4
    uint64_t c5 = m[5]; // x5
    uint64_t c6 = m[6]; // x6
    uint64_t c7 = m[7]; // x7
    uint64_t c8 = m[8]; // x8
    uint64_t c9 = m[9]; // x9
    uint64_t lsb1 = uint64_t_lsb(c1); // initially all eqs are candidates
    if(unlikely(!lsb1)) {
        return -1; // singular
    }

    uint64_t c1_reduc = c1 ^ lsb1; // for row reduction
    if(const_col & lsb1) const_col ^= c1_reduc;
    if(c2 & lsb1) c2 ^= c1_reduc;
    if(c3 & lsb1) c3 ^= c1_reduc;
    if(c4 & lsb1) c4 ^= c1_reduc;
    if(c5 & lsb1) c5 ^= c1_reduc;
    if(c6 & lsb1) c6 ^= c1_reduc;
    if(c7 & lsb1) c7 ^= c1_reduc;
    if(c8 & lsb1) c8 ^= c1_reduc;
    if(c9 & lsb1) c9 ^= c1_reduc;
    uint64_t mask = ~lsb1;

    uint64_t lsb2 = uint64_t_lsb(c2 & mask); // x2
    if(unlikely(!lsb2)) {
        return -1; // singular
    }

    uint64_t c2_reduc = c2 ^ lsb2;
    if(const_col & lsb2) const_col ^= c2_reduc;
    if(c3 & lsb2) c3 ^= c2_reduc;
    if(c4 & lsb2) c4 ^= c2_reduc;
    if(c5 & lsb2) c5 ^= c2_reduc;
    if(c6 & lsb2) c6 ^= c2_reduc;
    if(c7 & lsb2) c7 ^= c2_reduc;
    if(c8 & lsb2) c8 ^= c2_reduc;
    if(c9 & lsb2) c9 ^= c2_reduc;
    mask ^= lsb2;

    uint64_t lsb3 = uint64_t_lsb(c3 & mask); // x3
    if(unlikely(!lsb3)) {
        return -1; // singular
    }

    uint64_t c3_reduc = c3 ^ lsb3;
    if(const_col & lsb3) const_col ^= c3_reduc;
    if(c4 & lsb3) c4 ^= c3_reduc;
    if(c5 & lsb3) c5 ^= c3_reduc;
    if(c6 & lsb3) c6 ^= c3_reduc;
    if(c7 & lsb3) c7 ^= c3_reduc;
    if(c8 & lsb3) c8 ^= c3_reduc;
    if(c9 & lsb3) c9 ^= c3_reduc;
    mask ^= lsb3;

    uint64_t lsb4 = uint64_t_lsb(c4 & mask); // x4
    if(unlikely(!lsb4)) {
        return -1; // singular
    }

    uint64_t c4_reduc = c4 ^ lsb4;
    if(const_col & lsb4) const_col ^= c4_reduc;
    if(c5 & lsb4) c5 ^= c4_reduc;
    if(c6 & lsb4) c6 ^= c4_reduc;
    if(c7 & lsb4) c7 ^= c4_reduc;
    if(c8 & lsb4) c8 ^= c4_reduc;
    if(c9 & lsb4) c9 ^= c4_reduc;
    mask ^= lsb4;

    uint64_t lsb5 = uint64_t_lsb(c5 & mask); // x5
    if(unlikely(!lsb5)) {
        return -1; // singular
    }

    uint64_t c5_reduc = c5 ^ lsb5;
    if(const_col & lsb5) const_col ^= c5_reduc;
    if(c6 & lsb5) c6 ^= c5_reduc;
    if(c7 & lsb5) c7 ^= c5_reduc;
    if(c8 & lsb5) c8 ^= c5_reduc;
    if(c9 & lsb5) c9 ^= c5_reduc;
    mask ^= lsb5;

    uint64_t lsb6 = uint64_t_lsb(c6 & mask); // x6
    if(unlikely(!lsb6)) {
        return -1; // singular
    }

    uint64_t c6_reduc = c6 ^ lsb6;
    if(const_col & lsb6) const_col ^= c6_reduc;
    if(c7 & lsb6) c7 ^= c6_reduc;
    if(c8 & lsb6) c8 ^= c6_reduc;
    if(c9 & lsb6) c9 ^= c6_reduc;
    mask ^= lsb6;

    uint64_t lsb7 = uint64_t_lsb(c7 & mask); // x7
    if(unlikely(!lsb7)) {
        return -1; // singular
    }

    uint64_t c7_reduc = c7 ^ lsb7;
    if(const_col & lsb7) const_col ^= c7_reduc;
    if(c8 & lsb7) c8 ^= c7_reduc;
    if(c9 & lsb7) c9 ^= c7_reduc;
    mask ^= lsb7;

    uint64_t lsb8 = uint64_t_lsb(c8 & mask); // x8
    if(unlikely(!lsb8)) {
        return -1; // singular
    }

    uint64_t c8_reduc = c8 ^ lsb8;
    if(const_col & lsb8) const_col ^= c8_reduc;
    if(c9 & lsb8) c9 ^= c8_reduc;
    mask ^= lsb8;

    uint64_t lsb9 = uint64_t_lsb(c9 & mask); // x9
    if(unlikely(!lsb9)) {
        return -1; // singular
    }

    if(const_col & lsb9) const_col ^= c9 ^ lsb9;
    mask ^= lsb9;

    if(likely(mask & const_col)) { // check system consistency
        return mask & const_col; // not solvable
    }

    uint64_t s = 0x0ULL;
    if(const_col & lsb1) s = uint64_t_toggle_at(s, 0);
    if(const_col & lsb2) s = uint64_t_toggle_at(s, 1);
    if(const_col & lsb3) s = uint64_t_toggle_at(s, 2);
    if(const_col & lsb4) s = uint64_t_toggle_at(s, 3);
    if(const_col & lsb5) s = uint64_t_toggle_at(s, 4);
    if(const_col & lsb6) s = uint64_t_toggle_at(s, 5);
    if(const_col & lsb7) s = uint64_t_toggle_at(s, 6);
    if(const_col & lsb8) s = uint64_t_toggle_at(s, 7);
    if(const_col & lsb9) s = uint64_t_toggle_at(s, 8);
    *sol = s;
    return 0;
}

int64_t
uint64a_gj_v10_generic(const uint64_t m[11], uint64_t* const restrict sol) {
    uint64_t const_col = m[0];
    uint64_t c1 = m[1]; // x1
    uint64_t c2 = m[2]; // x2
    uint64_t c3 = m[3]; // x3
    uint64_t c4 = m[4]; // x4
    uint64_t c5 = m[5]; // x5
    uint64_t c6 = m[6]; // x6
    uint64_t c7 = m[7]; // x7
    uint64_t c8 = m[8]; // x8
    uint64_t c9 = m[9]; // x9
    uint64_t c10 = m[10]; // x10
    uint64_t lsb1 = uint64_t_lsb(c1); // initially all eqs are candidates
    if(unlikely(!lsb1)) {
        return -1; // singular
    }

    uint64_t c1_reduc = c1 ^ lsb1; // for row reduction
    if(const_col & lsb1) const_col ^= c1_reduc;
    if(c2 & lsb1) c2 ^= c1_reduc;
    if(c3 & lsb1) c3 ^= c1_reduc;
    if(c4 & lsb1) c4 ^= c1_reduc;
    if(c5 & lsb1) c5 ^= c1_reduc;
    if(c6 & lsb1) c6 ^= c1_reduc;
    if(c7 & lsb1) c7 ^= c1_reduc;
    if(c8 & lsb1) c8 ^= c1_reduc;
    if(c9 & lsb1) c9 ^= c1_reduc;
    if(c10 & lsb1) c10 ^= c1_reduc;
    uint64_t mask = ~lsb1;

    uint64_t lsb2 = uint64_t_lsb(c2 & mask); // x2
    if(unlikely(!lsb2)) {
        return -1; // singular
    }

    uint64_t c2_reduc = c2 ^ lsb2;
    if(const_col & lsb2) const_col ^= c2_reduc;
    if(c3 & lsb2) c3 ^= c2_reduc;
    if(c4 & lsb2) c4 ^= c2_reduc;
    if(c5 & lsb2) c5 ^= c2_reduc;
    if(c6 & lsb2) c6 ^= c2_reduc;
    if(c7 & lsb2) c7 ^= c2_reduc;
    if(c8 & lsb2) c8 ^= c2_reduc;
    if(c9 & lsb2) c9 ^= c2_reduc;
    if(c10 & lsb2) c10 ^= c2_reduc;
    mask ^= lsb2;

    uint64_t lsb3 = uint64_t_lsb(c3 & mask); // x3
    if(unlikely(!lsb3)) {
        return -1; // singular
    }

    uint64_t c3_reduc = c3 ^ lsb3;
    if(const_col & lsb3) const_col ^= c3_reduc;
    if(c4 & lsb3) c4 ^= c3_reduc;
    if(c5 & lsb3) c5 ^= c3_reduc;
    if(c6 & lsb3) c6 ^= c3_reduc;
    if(c7 & lsb3) c7 ^= c3_reduc;
    if(c8 & lsb3) c8 ^= c3_reduc;
    if(c9 & lsb3) c9 ^= c3_reduc;
    if(c10 & lsb3) c10 ^= c3_reduc;
    mask ^= lsb3;

    uint64_t lsb4 = uint64_t_lsb(c4 & mask); // x4
    if(unlikely(!lsb4)) {
        return -1; // singular
    }

    uint64_t c4_reduc = c4 ^ lsb4;
    if(const_col & lsb4) const_col ^= c4_reduc;
    if(c5 & lsb4) c5 ^= c4_reduc;
    if(c6 & lsb4) c6 ^= c4_reduc;
    if(c7 & lsb4) c7 ^= c4_reduc;
    if(c8 & lsb4) c8 ^= c4_reduc;
    if(c9 & lsb4) c9 ^= c4_reduc;
    if(c10 & lsb4) c10 ^= c4_reduc;
    mask ^= lsb4;

    uint64_t lsb5 = uint64_t_lsb(c5 & mask); // x5
    if(unlikely(!lsb5)) {
        return -1; // singular
    }

    uint64_t c5_reduc = c5 ^ lsb5;
    if(const_col & lsb5) const_col ^= c5_reduc;
    if(c6 & lsb5) c6 ^= c5_reduc;
    if(c7 & lsb5) c7 ^= c5_reduc;
    if(c8 & lsb5) c8 ^= c5_reduc;
    if(c9 & lsb5) c9 ^= c5_reduc;
    if(c10 & lsb5) c10 ^= c5_reduc;
    mask ^= lsb5;

    uint64_t lsb6 = uint64_t_lsb(c6 & mask); // x6
    if(unlikely(!lsb6)) {
        return -1; // singular
    }

    uint64_t c6_reduc = c6 ^ lsb6;
    if(const_col & lsb6) const_col ^= c6_reduc;
    if(c7 & lsb6) c7 ^= c6_reduc;
    if(c8 & lsb6) c8 ^= c6_reduc;
    if(c9 & lsb6) c9 ^= c6_reduc;
    if(c10 & lsb6) c10 ^= c6_reduc;
    mask ^= lsb6;

    uint64_t lsb7 = uint64_t_lsb(c7 & mask); // x7
    if(unlikely(!lsb7)) {
        return -1; // singular
    }

    uint64_t c7_reduc = c7 ^ lsb7;
    if(const_col & lsb7) const_col ^= c7_reduc;
    if(c8 & lsb7) c8 ^= c7_reduc;
    if(c9 & lsb7) c9 ^= c7_reduc;
    if(c10 & lsb7) c10 ^= c7_reduc;
    mask ^= lsb7;

    uint64_t lsb8 = uint64_t_lsb(c8 & mask); // x8
    if(unlikely(!lsb8)) {
        return -1; // singular
    }

    uint64_t c8_reduc = c8 ^ lsb8;
    if(const_col & lsb8) const_col ^= c8_reduc;
    if(c9 & lsb8) c9 ^= c8_reduc;
    if(c10 & lsb8) c10 ^= c8_reduc;
    mask ^= lsb8;

    uint64_t lsb9 = uint64_t_lsb(c9 & mask); // x9
    if(unlikely(!lsb9)) {
        return -1; // singular
    }

    uint64_t c9_reduc = c9 ^ lsb9;
    if(const_col & lsb9) const_col ^= c9_reduc;
    if(c10 & lsb9) c10 ^= c9_reduc;
    mask ^= lsb9;

    uint64_t lsb10 = uint64_t_lsb(c10 & mask); // x10
    if(unlikely(!lsb10)) {
        return -1; // singular
    }

    if(const_col & lsb10) const_col ^= c10 ^ lsb10;
    mask ^= lsb10;

    if(likely(mask & const_col)) { // check system consistency
        return mask & const_col; // not solvable
    }

    uint64_t s = 0x0ULL;
    if(const_col & lsb1) s = uint64_t_toggle_at(s, 0);
    if(const_col & lsb2) s = uint64_t_toggle_at(s, 1);
    if(const_col & lsb3) s = uint64_t_toggle_at(s, 2);
    if(const_col & lsb4) s = uint64_t_toggle_at(s, 3);
    if(const_col & lsb5) s = uint64_t_toggle_at(s, 4);
    if(const_col & lsb6) s = uint64_t_toggle_at(s, 5);
    if(const_col & lsb7) s = uint64_t_toggle_at(s, 6);
    if(const_col & lsb8) s = uint64_t_toggle_at(s, 7);
    if(const_col & lsb9) s = uint64_t_toggle_at(s, 8);
    if(const_col & lsb10) s = uint64_t_toggle_at(s, 9);
    *sol = s;
    return 0;
}

int64_t
uint64a_gj_v11_generic(const uint64_t m[12], uint64_t* const restrict sol) {
    uint64_t const_col = m[0];
    uint64_t c1 = m[1]; // x1
    uint64_t c2 = m[2]; // x2
    uint64_t c3 = m[3]; // x3
    uint64_t c4 = m[4]; // x4
    uint64_t c5 = m[5]; // x5
    uint64_t c6 = m[6]; // x6
    uint64_t c7 = m[7]; // x7
    uint64_t c8 = m[8]; // x8
    uint64_t c9 = m[9]; // x9
    uint64_t c10 = m[10]; // x10
    uint64_t c11 = m[11]; // x11
    uint64_t lsb1 = uint64_t_lsb(c1); // initially all eqs are candidates
    if(unlikely(!lsb1)) {
        return -1; // singular
    }

    uint64_t c1_reduc = c1 ^ lsb1; // for row reduction
    if(const_col & lsb1) const_col ^= c1_reduc;
    if(c2 & lsb1) c2 ^= c1_reduc;
    if(c3 & lsb1) c3 ^= c1_reduc;
    if(c4 & lsb1) c4 ^= c1_reduc;
    if(c5 & lsb1) c5 ^= c1_reduc;
    if(c6 & lsb1) c6 ^= c1_reduc;
    if(c7 & lsb1) c7 ^= c1_reduc;
    if(c8 & lsb1) c8 ^= c1_reduc;
    if(c9 & lsb1) c9 ^= c1_reduc;
    if(c10 & lsb1) c10 ^= c1_reduc;
    if(c11 & lsb1) c11 ^= c1_reduc;
    uint64_t mask = ~lsb1;

    uint64_t lsb2 = uint64_t_lsb(c2 & mask); // x2
    if(unlikely(!lsb2)) {
        return -1; // singular
    }

    uint64_t c2_reduc = c2 ^ lsb2;
    if(const_col & lsb2) const_col ^= c2_reduc;
    if(c3 & lsb2) c3 ^= c2_reduc;
    if(c4 & lsb2) c4 ^= c2_reduc;
    if(c5 & lsb2) c5 ^= c2_reduc;
    if(c6 & lsb2) c6 ^= c2_reduc;
    if(c7 & lsb2) c7 ^= c2_reduc;
    if(c8 & lsb2) c8 ^= c2_reduc;
    if(c9 & lsb2) c9 ^= c2_reduc;
    if(c10 & lsb2) c10 ^= c2_reduc;
    if(c11 & lsb2) c11 ^= c2_reduc;
    mask ^= lsb2;

    uint64_t lsb3 = uint64_t_lsb(c3 & mask); // x3
    if(unlikely(!lsb3)) {
        return -1; // singular
    }

    uint64_t c3_reduc = c3 ^ lsb3;
    if(const_col & lsb3) const_col ^= c3_reduc;
    if(c4 & lsb3) c4 ^= c3_reduc;
    if(c5 & lsb3) c5 ^= c3_reduc;
    if(c6 & lsb3) c6 ^= c3_reduc;
    if(c7 & lsb3) c7 ^= c3_reduc;
    if(c8 & lsb3) c8 ^= c3_reduc;
    if(c9 & lsb3) c9 ^= c3_reduc;
    if(c10 & lsb3) c10 ^= c3_reduc;
    if(c11 & lsb3) c11 ^= c3_reduc;
    mask ^= lsb3;

    uint64_t lsb4 = uint64_t_lsb(c4 & mask); // x4
    if(unlikely(!lsb4)) {
        return -1; // singular
    }

    uint64_t c4_reduc = c4 ^ lsb4;
    if(const_col & lsb4) const_col ^= c4_reduc;
    if(c5 & lsb4) c5 ^= c4_reduc;
    if(c6 & lsb4) c6 ^= c4_reduc;
    if(c7 & lsb4) c7 ^= c4_reduc;
    if(c8 & lsb4) c8 ^= c4_reduc;
    if(c9 & lsb4) c9 ^= c4_reduc;
    if(c10 & lsb4) c10 ^= c4_reduc;
    if(c11 & lsb4) c11 ^= c4_reduc;
    mask ^= lsb4;

    uint64_t lsb5 = uint64_t_lsb(c5 & mask); // x5
    if(unlikely(!lsb5)) {
        return -1; // singular
    }

    uint64_t c5_reduc = c5 ^ lsb5;
    if(const_col & lsb5) const_col ^= c5_reduc;
    if(c6 & lsb5) c6 ^= c5_reduc;
    if(c7 & lsb5) c7 ^= c5_reduc;
    if(c8 & lsb5) c8 ^= c5_reduc;
    if(c9 & lsb5) c9 ^= c5_reduc;
    if(c10 & lsb5) c10 ^= c5_reduc;
    if(c11 & lsb5) c11 ^= c5_reduc;
    mask ^= lsb5;

    uint64_t lsb6 = uint64_t_lsb(c6 & mask); // x6
    if(unlikely(!lsb6)) {
        return -1; // singular
    }

    uint64_t c6_reduc = c6 ^ lsb6;
    if(const_col & lsb6) const_col ^= c6_reduc;
    if(c7 & lsb6) c7 ^= c6_reduc;
    if(c8 & lsb6) c8 ^= c6_reduc;
    if(c9 & lsb6) c9 ^= c6_reduc;
    if(c10 & lsb6) c10 ^= c6_reduc;
    if(c11 & lsb6) c11 ^= c6_reduc;
    mask ^= lsb6;

    uint64_t lsb7 = uint64_t_lsb(c7 & mask); // x7
    if(unlikely(!lsb7)) {
        return -1; // singular
    }

    uint64_t c7_reduc = c7 ^ lsb7;
    if(const_col & lsb7) const_col ^= c7_reduc;
    if(c8 & lsb7) c8 ^= c7_reduc;
    if(c9 & lsb7) c9 ^= c7_reduc;
    if(c10 & lsb7) c10 ^= c7_reduc;
    if(c11 & lsb7) c11 ^= c7_reduc;
    mask ^= lsb7;

    uint64_t lsb8 = uint64_t_lsb(c8 & mask); // x8
    if(unlikely(!lsb8)) {
        return -1; // singular
    }

    uint64_t c8_reduc = c8 ^ lsb8;
    if(const_col & lsb8) const_col ^= c8_reduc;
    if(c9 & lsb8) c9 ^= c8_reduc;
    if(c10 & lsb8) c10 ^= c8_reduc;
    if(c11 & lsb8) c11 ^= c8_reduc;
    mask ^= lsb8;

    uint64_t lsb9 = uint64_t_lsb(c9 & mask); // x9
    if(unlikely(!lsb9)) {
        return -1; // singular
    }

    uint64_t c9_reduc = c9 ^ lsb9;
    if(const_col & lsb9) const_col ^= c9_reduc;
    if(c10 & lsb9) c10 ^= c9_reduc;
    if(c11 & lsb9) c11 ^= c9_reduc;
    mask ^= lsb9;

    uint64_t lsb10 = uint64_t_lsb(c10 & mask); // x10
    if(unlikely(!lsb10)) {
        return -1; // singular
    }

    uint64_t c10_reduc = c10 ^ lsb10;
    if(const_col & lsb10) const_col ^= c10_reduc;
    if(c11 & lsb10) c11 ^= c10_reduc;
    mask ^= lsb10;

    uint64_t lsb11 = uint64_t_lsb(c11 & mask); // x11
    if(unlikely(!lsb11)) {
        return -1; // singular
    }

    if(const_col & lsb11) const_col ^= c11 ^ lsb11;
    mask ^= lsb11;

    if(likely(mask & const_col)) { // check system consistency
        return mask & const_col; // not solvable
    }

    uint64_t s = 0x0ULL;
    if(const_col & lsb1) s = uint64_t_toggle_at(s, 0);
    if(const_col & lsb2) s = uint64_t_toggle_at(s, 1);
    if(const_col & lsb3) s = uint64_t_toggle_at(s, 2);
    if(const_col & lsb4) s = uint64_t_toggle_at(s, 3);
    if(const_col & lsb5) s = uint64_t_toggle_at(s, 4);
    if(const_col & lsb6) s = uint64_t_toggle_at(s, 5);
    if(const_col & lsb7) s = uint64_t_toggle_at(s, 6);
    if(const_col & lsb8) s = uint64_t_toggle_at(s, 7);
    if(const_col & lsb9) s = uint64_t_toggle_at(s, 8);
    if(const_col & lsb10) s = uint64_t_toggle_at(s, 9);
    if(const_col & lsb11) s = uint64_t_toggle_at(s, 10);
    *sol = s;
    return 0;
}

int64_t
uint64a_gj_v12_generic(const uint64_t m[13], uint64_t* const restrict sol) {
    uint64_t const_col = m[0];
    uint64_t c1 = m[1]; // x1
    uint64_t c2 = m[2]; // x2
    uint64_t c3 = m[3]; // x3
    uint64_t c4 = m[4]; // x4
    uint64_t c5 = m[5]; // x5
    uint64_t c6 = m[6]; // x6
    uint64_t c7 = m[7]; // x7
    uint64_t c8 = m[8]; // x8
    uint64_t c9 = m[9]; // x9
    uint64_t c10 = m[10]; // x10
    uint64_t c11 = m[11]; // x11
    uint64_t c12 = m[12]; // x12
    uint64_t lsb1 = uint64_t_lsb(c1); // initially all eqs are candidates
    if(unlikely(!lsb1)) {
        return -1; // singular
    }

    uint64_t c1_reduc = c1 ^ lsb1; // for row reduction
    if(const_col & lsb1) const_col ^= c1_reduc;
    if(c2 & lsb1) c2 ^= c1_reduc;
    if(c3 & lsb1) c3 ^= c1_reduc;
    if(c4 & lsb1) c4 ^= c1_reduc;
    if(c5 & lsb1) c5 ^= c1_reduc;
    if(c6 & lsb1) c6 ^= c1_reduc;
    if(c7 & lsb1) c7 ^= c1_reduc;
    if(c8 & lsb1) c8 ^= c1_reduc;
    if(c9 & lsb1) c9 ^= c1_reduc;
    if(c10 & lsb1) c10 ^= c1_reduc;
    if(c11 & lsb1) c11 ^= c1_reduc;
    if(c12 & lsb1) c12 ^= c1_reduc;
    uint64_t mask = ~lsb1;

    uint64_t lsb2 = uint64_t_lsb(c2 & mask); // x2
    if(unlikely(!lsb2)) {
        return -1; // singular
    }

    uint64_t c2_reduc = c2 ^ lsb2;
    if(const_col & lsb2) const_col ^= c2_reduc;
    if(c3 & lsb2) c3 ^= c2_reduc;
    if(c4 & lsb2) c4 ^= c2_reduc;
    if(c5 & lsb2) c5 ^= c2_reduc;
    if(c6 & lsb2) c6 ^= c2_reduc;
    if(c7 & lsb2) c7 ^= c2_reduc;
    if(c8 & lsb2) c8 ^= c2_reduc;
    if(c9 & lsb2) c9 ^= c2_reduc;
    if(c10 & lsb2) c10 ^= c2_reduc;
    if(c11 & lsb2) c11 ^= c2_reduc;
    if(c12 & lsb2) c12 ^= c2_reduc;
    mask ^= lsb2;

    uint64_t lsb3 = uint64_t_lsb(c3 & mask); // x3
    if(unlikely(!lsb3)) {
        return -1; // singular
    }

    uint64_t c3_reduc = c3 ^ lsb3;
    if(const_col & lsb3) const_col ^= c3_reduc;
    if(c4 & lsb3) c4 ^= c3_reduc;
    if(c5 & lsb3) c5 ^= c3_reduc;
    if(c6 & lsb3) c6 ^= c3_reduc;
    if(c7 & lsb3) c7 ^= c3_reduc;
    if(c8 & lsb3) c8 ^= c3_reduc;
    if(c9 & lsb3) c9 ^= c3_reduc;
    if(c10 & lsb3) c10 ^= c3_reduc;
    if(c11 & lsb3) c11 ^= c3_reduc;
    if(c12 & lsb3) c12 ^= c3_reduc;
    mask ^= lsb3;

    uint64_t lsb4 = uint64_t_lsb(c4 & mask); // x4
    if(unlikely(!lsb4)) {
        return -1; // singular
    }

    uint64_t c4_reduc = c4 ^ lsb4;
    if(const_col & lsb4) const_col ^= c4_reduc;
    if(c5 & lsb4) c5 ^= c4_reduc;
    if(c6 & lsb4) c6 ^= c4_reduc;
    if(c7 & lsb4) c7 ^= c4_reduc;
    if(c8 & lsb4) c8 ^= c4_reduc;
    if(c9 & lsb4) c9 ^= c4_reduc;
    if(c10 & lsb4) c10 ^= c4_reduc;
    if(c11 & lsb4) c11 ^= c4_reduc;
    if(c12 & lsb4) c12 ^= c4_reduc;
    mask ^= lsb4;

    uint64_t lsb5 = uint64_t_lsb(c5 & mask); // x5
    if(unlikely(!lsb5)) {
        return -1; // singular
    }

    uint64_t c5_reduc = c5 ^ lsb5;
    if(const_col & lsb5) const_col ^= c5_reduc;
    if(c6 & lsb5) c6 ^= c5_reduc;
    if(c7 & lsb5) c7 ^= c5_reduc;
    if(c8 & lsb5) c8 ^= c5_reduc;
    if(c9 & lsb5) c9 ^= c5_reduc;
    if(c10 & lsb5) c10 ^= c5_reduc;
    if(c11 & lsb5) c11 ^= c5_reduc;
    if(c12 & lsb5) c12 ^= c5_reduc;
    mask ^= lsb5;

    uint64_t lsb6 = uint64_t_lsb(c6 & mask); // x6
    if(unlikely(!lsb6)) {
        return -1; // singular
    }

    uint64_t c6_reduc = c6 ^ lsb6;
    if(const_col & lsb6) const_col ^= c6_reduc;
    if(c7 & lsb6) c7 ^= c6_reduc;
    if(c8 & lsb6) c8 ^= c6_reduc;
    if(c9 & lsb6) c9 ^= c6_reduc;
    if(c10 & lsb6) c10 ^= c6_reduc;
    if(c11 & lsb6) c11 ^= c6_reduc;
    if(c12 & lsb6) c12 ^= c6_reduc;
    mask ^= lsb6;

    uint64_t lsb7 = uint64_t_lsb(c7 & mask); // x7
    if(unlikely(!lsb7)) {
        return -1; // singular
    }

    uint64_t c7_reduc = c7 ^ lsb7;
    if(const_col & lsb7) const_col ^= c7_reduc;
    if(c8 & lsb7) c8 ^= c7_reduc;
    if(c9 & lsb7) c9 ^= c7_reduc;
    if(c10 & lsb7) c10 ^= c7_reduc;
    if(c11 & lsb7) c11 ^= c7_reduc;
    if(c12 & lsb7) c12 ^= c7_reduc;
    mask ^= lsb7;

    uint64_t lsb8 = uint64_t_lsb(c8 & mask); // x8
    if(unlikely(!lsb8)) {
        return -1; // singular
    }

    uint64_t c8_reduc = c8 ^ lsb8;
    if(const_col & lsb8) const_col ^= c8_reduc;
    if(c9 & lsb8) c9 ^= c8_reduc;
    if(c10 & lsb8) c10 ^= c8_reduc;
    if(c11 & lsb8) c11 ^= c8_reduc;
    if(c12 & lsb8) c12 ^= c8_reduc;
    mask ^= lsb8;

    uint64_t lsb9 = uint64_t_lsb(c9 & mask); // x9
    if(unlikely(!lsb9)) {
        return -1; // singular
    }

    uint64_t c9_reduc = c9 ^ lsb9;
    if(const_col & lsb9) const_col ^= c9_reduc;
    if(c10 & lsb9) c10 ^= c9_reduc;
    if(c11 & lsb9) c11 ^= c9_reduc;
    if(c12 & lsb9) c12 ^= c9_reduc;
    mask ^= lsb9;

    uint64_t lsb10 = uint64_t_lsb(c10 & mask); // x10
    if(unlikely(!lsb10)) {
        return -1; // singular
    }

    uint64_t c10_reduc = c10 ^ lsb10;
    if(const_col & lsb10) const_col ^= c10_reduc;
    if(c11 & lsb10) c11 ^= c10_reduc;
    if(c12 & lsb10) c12 ^= c10_reduc;
    mask ^= lsb10;

    uint64_t lsb11 = uint64_t_lsb(c11 & mask); // x11
    if(unlikely(!lsb11)) {
        return -1; // singular
    }

    uint64_t c11_reduc = c11 ^ lsb11;
    if(const_col & lsb11) const_col ^= c11_reduc;
    if(c12 & lsb11) c12 ^= c11_reduc;
    mask ^= lsb11;

    uint64_t lsb12 = uint64_t_lsb(c12 & mask); // x12
    if(unlikely(!lsb12)) {
        return -1; // singular
    }

    if(const_col & lsb12) const_col ^= c12 ^ lsb12;
    mask ^= lsb12;

    if(likely(mask & const_col)) { // check system consistency
        return mask & const_col; // not solvable
    }

    uint64_t s = 0x0ULL;
    if(const_col & lsb1) s = uint64_t_toggle_at(s, 0);
    if(const_col & lsb2) s = uint64_t_toggle_at(s, 1);
    if(const_col & lsb3) s = uint64_t_toggle_at(s, 2);
    if(const_col & lsb4) s = uint64_t_toggle_at(s, 3);
    if(const_col & lsb5) s = uint64_t_toggle_at(s, 4);
    if(const_col & lsb6) s = uint64_t_toggle_at(s, 5);
    if(const_col & lsb7) s = uint64_t_toggle_at(s, 6);
    if(const_col & lsb8) s = uint64_t_toggle_at(s, 7);
    if(const_col & lsb9) s = uint64_t_toggle_at(s, 8);
    if(const_col & lsb10) s = uint64_t_toggle_at(s, 9);
    if(const_col & lsb11) s = uint64_t_toggle_at(s, 10);
    if(const_col & lsb12) s = uint64_t_toggle_at(s, 11);
    *sol = s;
    return 0;
}

int64_t
uint64a_gj_v13_generic(const uint64_t m[14], uint64_t* const restrict sol) {
    uint64_t const_col = m[0];
    uint64_t c1 = m[1]; // x1
    uint64_t c2 = m[2]; // x2
    uint64_t c3 = m[3]; // x3
    uint64_t c4 = m[4]; // x4
    uint64_t c5 = m[5]; // x5
    uint64_t c6 = m[6]; // x6
    uint64_t c7 = m[7]; // x7
    uint64_t c8 = m[8]; // x8
    uint64_t c9 = m[9]; // x9
    uint64_t c10 = m[10]; // x10
    uint64_t c11 = m[11]; // x11
    uint64_t c12 = m[12]; // x12
    uint64_t c13 = m[13]; // x13
    uint64_t lsb1 = uint64_t_lsb(c1); // initially all eqs are candidates
    if(unlikely(!lsb1)) {
        return -1; // singular
    }

    uint64_t c1_reduc = c1 ^ lsb1; // for row reduction
    if(const_col & lsb1) const_col ^= c1_reduc;
    if(c2 & lsb1) c2 ^= c1_reduc;
    if(c3 & lsb1) c3 ^= c1_reduc;
    if(c4 & lsb1) c4 ^= c1_reduc;
    if(c5 & lsb1) c5 ^= c1_reduc;
    if(c6 & lsb1) c6 ^= c1_reduc;
    if(c7 & lsb1) c7 ^= c1_reduc;
    if(c8 & lsb1) c8 ^= c1_reduc;
    if(c9 & lsb1) c9 ^= c1_reduc;
    if(c10 & lsb1) c10 ^= c1_reduc;
    if(c11 & lsb1) c11 ^= c1_reduc;
    if(c12 & lsb1) c12 ^= c1_reduc;
    if(c13 & lsb1) c13 ^= c1_reduc;
    uint64_t mask = ~lsb1;

    uint64_t lsb2 = uint64_t_lsb(c2 & mask); // x2
    if(unlikely(!lsb2)) {
        return -1; // singular
    }

    uint64_t c2_reduc = c2 ^ lsb2;
    if(const_col & lsb2) const_col ^= c2_reduc;
    if(c3 & lsb2) c3 ^= c2_reduc;
    if(c4 & lsb2) c4 ^= c2_reduc;
    if(c5 & lsb2) c5 ^= c2_reduc;
    if(c6 & lsb2) c6 ^= c2_reduc;
    if(c7 & lsb2) c7 ^= c2_reduc;
    if(c8 & lsb2) c8 ^= c2_reduc;
    if(c9 & lsb2) c9 ^= c2_reduc;
    if(c10 & lsb2) c10 ^= c2_reduc;
    if(c11 & lsb2) c11 ^= c2_reduc;
    if(c12 & lsb2) c12 ^= c2_reduc;
    if(c13 & lsb2) c13 ^= c2_reduc;
    mask ^= lsb2;

    uint64_t lsb3 = uint64_t_lsb(c3 & mask); // x3
    if(unlikely(!lsb3)) {
        return -1; // singular
    }

    uint64_t c3_reduc = c3 ^ lsb3;
    if(const_col & lsb3) const_col ^= c3_reduc;
    if(c4 & lsb3) c4 ^= c3_reduc;
    if(c5 & lsb3) c5 ^= c3_reduc;
    if(c6 & lsb3) c6 ^= c3_reduc;
    if(c7 & lsb3) c7 ^= c3_reduc;
    if(c8 & lsb3) c8 ^= c3_reduc;
    if(c9 & lsb3) c9 ^= c3_reduc;
    if(c10 & lsb3) c10 ^= c3_reduc;
    if(c11 & lsb3) c11 ^= c3_reduc;
    if(c12 & lsb3) c12 ^= c3_reduc;
    if(c13 & lsb3) c13 ^= c3_reduc;
    mask ^= lsb3;

    uint64_t lsb4 = uint64_t_lsb(c4 & mask); // x4
    if(unlikely(!lsb4)) {
        return -1; // singular
    }

    uint64_t c4_reduc = c4 ^ lsb4;
    if(const_col & lsb4) const_col ^= c4_reduc;
    if(c5 & lsb4) c5 ^= c4_reduc;
    if(c6 & lsb4) c6 ^= c4_reduc;
    if(c7 & lsb4) c7 ^= c4_reduc;
    if(c8 & lsb4) c8 ^= c4_reduc;
    if(c9 & lsb4) c9 ^= c4_reduc;
    if(c10 & lsb4) c10 ^= c4_reduc;
    if(c11 & lsb4) c11 ^= c4_reduc;
    if(c12 & lsb4) c12 ^= c4_reduc;
    if(c13 & lsb4) c13 ^= c4_reduc;
    mask ^= lsb4;

    uint64_t lsb5 = uint64_t_lsb(c5 & mask); // x5
    if(unlikely(!lsb5)) {
        return -1; // singular
    }

    uint64_t c5_reduc = c5 ^ lsb5;
    if(const_col & lsb5) const_col ^= c5_reduc;
    if(c6 & lsb5) c6 ^= c5_reduc;
    if(c7 & lsb5) c7 ^= c5_reduc;
    if(c8 & lsb5) c8 ^= c5_reduc;
    if(c9 & lsb5) c9 ^= c5_reduc;
    if(c10 & lsb5) c10 ^= c5_reduc;
    if(c11 & lsb5) c11 ^= c5_reduc;
    if(c12 & lsb5) c12 ^= c5_reduc;
    if(c13 & lsb5) c13 ^= c5_reduc;
    mask ^= lsb5;

    uint64_t lsb6 = uint64_t_lsb(c6 & mask); // x6
    if(unlikely(!lsb6)) {
        return -1; // singular
    }

    uint64_t c6_reduc = c6 ^ lsb6;
    if(const_col & lsb6) const_col ^= c6_reduc;
    if(c7 & lsb6) c7 ^= c6_reduc;
    if(c8 & lsb6) c8 ^= c6_reduc;
    if(c9 & lsb6) c9 ^= c6_reduc;
    if(c10 & lsb6) c10 ^= c6_reduc;
    if(c11 & lsb6) c11 ^= c6_reduc;
    if(c12 & lsb6) c12 ^= c6_reduc;
    if(c13 & lsb6) c13 ^= c6_reduc;
    mask ^= lsb6;

    uint64_t lsb7 = uint64_t_lsb(c7 & mask); // x7
    if(unlikely(!lsb7)) {
        return -1; // singular
    }

    uint64_t c7_reduc = c7 ^ lsb7;
    if(const_col & lsb7) const_col ^= c7_reduc;
    if(c8 & lsb7) c8 ^= c7_reduc;
    if(c9 & lsb7) c9 ^= c7_reduc;
    if(c10 & lsb7) c10 ^= c7_reduc;
    if(c11 & lsb7) c11 ^= c7_reduc;
    if(c12 & lsb7) c12 ^= c7_reduc;
    if(c13 & lsb7) c13 ^= c7_reduc;
    mask ^= lsb7;

    uint64_t lsb8 = uint64_t_lsb(c8 & mask); // x8
    if(unlikely(!lsb8)) {
        return -1; // singular
    }

    uint64_t c8_reduc = c8 ^ lsb8;
    if(const_col & lsb8) const_col ^= c8_reduc;
    if(c9 & lsb8) c9 ^= c8_reduc;
    if(c10 & lsb8) c10 ^= c8_reduc;
    if(c11 & lsb8) c11 ^= c8_reduc;
    if(c12 & lsb8) c12 ^= c8_reduc;
    if(c13 & lsb8) c13 ^= c8_reduc;
    mask ^= lsb8;

    uint64_t lsb9 = uint64_t_lsb(c9 & mask); // x9
    if(unlikely(!lsb9)) {
        return -1; // singular
    }

    uint64_t c9_reduc = c9 ^ lsb9;
    if(const_col & lsb9) const_col ^= c9_reduc;
    if(c10 & lsb9) c10 ^= c9_reduc;
    if(c11 & lsb9) c11 ^= c9_reduc;
    if(c12 & lsb9) c12 ^= c9_reduc;
    if(c13 & lsb9) c13 ^= c9_reduc;
    mask ^= lsb9;

    uint64_t lsb10 = uint64_t_lsb(c10 & mask); // x10
    if(unlikely(!lsb10)) {
        return -1; // singular
    }

    uint64_t c10_reduc = c10 ^ lsb10;
    if(const_col & lsb10) const_col ^= c10_reduc;
    if(c11 & lsb10) c11 ^= c10_reduc;
    if(c12 & lsb10) c12 ^= c10_reduc;
    if(c13 & lsb10) c13 ^= c10_reduc;
    mask ^= lsb10;

    uint64_t lsb11 = uint64_t_lsb(c11 & mask); // x11
    if(unlikely(!lsb11)) {
        return -1; // singular
    }

    uint64_t c11_reduc = c11 ^ lsb11;
    if(const_col & lsb11) const_col ^= c11_reduc;
    if(c12 & lsb11) c12 ^= c11_reduc;
    if(c13 & lsb11) c13 ^= c11_reduc;
    mask ^= lsb11;

    uint64_t lsb12 = uint64_t_lsb(c12 & mask); // x12
    if(unlikely(!lsb12)) {
        return -1; // singular
    }

    uint64_t c12_reduc = c12 ^ lsb12;
    if(const_col & lsb12) const_col ^= c12_reduc;
    if(c13 & lsb12) c13 ^= c12_reduc;
    mask ^= lsb12;

    uint64_t lsb13 = uint64_t_lsb(c13 & mask); // x13
    if(unlikely(!lsb13)) {
        return -1; // singular
    }

    if(const_col & lsb13) const_col ^= c13 ^ lsb13;
    mask ^= lsb13;

    if(likely(mask & const_col)) { // check system consistency
        return mask & const_col; // not solvable
    }

    uint64_t s = 0x0ULL;
    if(const_col & lsb1) s = uint64_t_toggle_at(s, 0);
    if(const_col & lsb2) s = uint64_t_toggle_at(s, 1);
    if(const_col & lsb3) s = uint64_t_toggle_at(s, 2);
    if(const_col & lsb4) s = uint64_t_toggle_at(s, 3);
    if(const_col & lsb5) s = uint64_t_toggle_at(s, 4);
    if(const_col & lsb6) s = uint64_t_toggle_at(s, 5);
    if(const_col & lsb7) s = uint64_t_toggle_at(s, 6);
    if(const_col & lsb8) s = uint64_t_toggle_at(s, 7);
    if(const_col & lsb9) s = uint64_t_toggle_at(s, 8);
    if(const_col & lsb10) s = uint64_t_toggle_at(s, 9);
    if(const_col & lsb11) s = uint64_t_toggle_at(s, 10);
    if(const_col & lsb12) s = uint64_t_toggle_at(s, 11);
    if(const_col & lsb13) s = uint64_t_toggle_at(s, 12);
    *sol = s;
    return 0;
}

int64_t
uint64a_gj_v14_generic(const uint64_t m[15], uint64_t* const restrict sol) {
    uint64_t const_col = m[0];
    uint64_t c1 = m[1]; // x1
    uint64_t c2 = m[2]; // x2
    uint64_t c3 = m[3]; // x3
    uint64_t c4 = m[4]; // x4
    uint64_t c5 = m[5]; // x5
    uint64_t c6 = m[6]; // x6
    uint64_t c7 = m[7]; // x7
    uint64_t c8 = m[8]; // x8
    uint64_t c9 = m[9]; // x9
    uint64_t c10 = m[10]; // x10
    uint64_t c11 = m[11]; // x11
    uint64_t c12 = m[12]; // x12
    uint64_t c13 = m[13]; // x13
    uint64_t c14 = m[14]; // x14
    uint64_t lsb1 = uint64_t_lsb(c1); // initially all eqs are candidates
    if(unlikely(!lsb1)) {
        return -1; // singular
    }

    uint64_t c1_reduc = c1 ^ lsb1; // for row reduction
    if(const_col & lsb1) const_col ^= c1_reduc;
    if(c2 & lsb1) c2 ^= c1_reduc;
    if(c3 & lsb1) c3 ^= c1_reduc;
    if(c4 & lsb1) c4 ^= c1_reduc;
    if(c5 & lsb1) c5 ^= c1_reduc;
    if(c6 & lsb1) c6 ^= c1_reduc;
    if(c7 & lsb1) c7 ^= c1_reduc;
    if(c8 & lsb1) c8 ^= c1_reduc;
    if(c9 & lsb1) c9 ^= c1_reduc;
    if(c10 & lsb1) c10 ^= c1_reduc;
    if(c11 & lsb1) c11 ^= c1_reduc;
    if(c12 & lsb1) c12 ^= c1_reduc;
    if(c13 & lsb1) c13 ^= c1_reduc;
    if(c14 & lsb1) c14 ^= c1_reduc;
    uint64_t mask = ~lsb1;

    uint64_t lsb2 = uint64_t_lsb(c2 & mask); // x2
    if(unlikely(!lsb2)) {
        return -1; // singular
    }

    uint64_t c2_reduc = c2 ^ lsb2;
    if(const_col & lsb2) const_col ^= c2_reduc;
    if(c3 & lsb2) c3 ^= c2_reduc;
    if(c4 & lsb2) c4 ^= c2_reduc;
    if(c5 & lsb2) c5 ^= c2_reduc;
    if(c6 & lsb2) c6 ^= c2_reduc;
    if(c7 & lsb2) c7 ^= c2_reduc;
    if(c8 & lsb2) c8 ^= c2_reduc;
    if(c9 & lsb2) c9 ^= c2_reduc;
    if(c10 & lsb2) c10 ^= c2_reduc;
    if(c11 & lsb2) c11 ^= c2_reduc;
    if(c12 & lsb2) c12 ^= c2_reduc;
    if(c13 & lsb2) c13 ^= c2_reduc;
    if(c14 & lsb2) c14 ^= c2_reduc;
    mask ^= lsb2;

    uint64_t lsb3 = uint64_t_lsb(c3 & mask); // x3
    if(unlikely(!lsb3)) {
        return -1; // singular
    }

    uint64_t c3_reduc = c3 ^ lsb3;
    if(const_col & lsb3) const_col ^= c3_reduc;
    if(c4 & lsb3) c4 ^= c3_reduc;
    if(c5 & lsb3) c5 ^= c3_reduc;
    if(c6 & lsb3) c6 ^= c3_reduc;
    if(c7 & lsb3) c7 ^= c3_reduc;
    if(c8 & lsb3) c8 ^= c3_reduc;
    if(c9 & lsb3) c9 ^= c3_reduc;
    if(c10 & lsb3) c10 ^= c3_reduc;
    if(c11 & lsb3) c11 ^= c3_reduc;
    if(c12 & lsb3) c12 ^= c3_reduc;
    if(c13 & lsb3) c13 ^= c3_reduc;
    if(c14 & lsb3) c14 ^= c3_reduc;
    mask ^= lsb3;

    uint64_t lsb4 = uint64_t_lsb(c4 & mask); // x4
    if(unlikely(!lsb4)) {
        return -1; // singular
    }

    uint64_t c4_reduc = c4 ^ lsb4;
    if(const_col & lsb4) const_col ^= c4_reduc;
    if(c5 & lsb4) c5 ^= c4_reduc;
    if(c6 & lsb4) c6 ^= c4_reduc;
    if(c7 & lsb4) c7 ^= c4_reduc;
    if(c8 & lsb4) c8 ^= c4_reduc;
    if(c9 & lsb4) c9 ^= c4_reduc;
    if(c10 & lsb4) c10 ^= c4_reduc;
    if(c11 & lsb4) c11 ^= c4_reduc;
    if(c12 & lsb4) c12 ^= c4_reduc;
    if(c13 & lsb4) c13 ^= c4_reduc;
    if(c14 & lsb4) c14 ^= c4_reduc;
    mask ^= lsb4;

    uint64_t lsb5 = uint64_t_lsb(c5 & mask); // x5
    if(unlikely(!lsb5)) {
        return -1; // singular
    }

    uint64_t c5_reduc = c5 ^ lsb5;
    if(const_col & lsb5) const_col ^= c5_reduc;
    if(c6 & lsb5) c6 ^= c5_reduc;
    if(c7 & lsb5) c7 ^= c5_reduc;
    if(c8 & lsb5) c8 ^= c5_reduc;
    if(c9 & lsb5) c9 ^= c5_reduc;
    if(c10 & lsb5) c10 ^= c5_reduc;
    if(c11 & lsb5) c11 ^= c5_reduc;
    if(c12 & lsb5) c12 ^= c5_reduc;
    if(c13 & lsb5) c13 ^= c5_reduc;
    if(c14 & lsb5) c14 ^= c5_reduc;
    mask ^= lsb5;

    uint64_t lsb6 = uint64_t_lsb(c6 & mask); // x6
    if(unlikely(!lsb6)) {
        return -1; // singular
    }

    uint64_t c6_reduc = c6 ^ lsb6;
    if(const_col & lsb6) const_col ^= c6_reduc;
    if(c7 & lsb6) c7 ^= c6_reduc;
    if(c8 & lsb6) c8 ^= c6_reduc;
    if(c9 & lsb6) c9 ^= c6_reduc;
    if(c10 & lsb6) c10 ^= c6_reduc;
    if(c11 & lsb6) c11 ^= c6_reduc;
    if(c12 & lsb6) c12 ^= c6_reduc;
    if(c13 & lsb6) c13 ^= c6_reduc;
    if(c14 & lsb6) c14 ^= c6_reduc;
    mask ^= lsb6;

    uint64_t lsb7 = uint64_t_lsb(c7 & mask); // x7
    if(unlikely(!lsb7)) {
        return -1; // singular
    }

    uint64_t c7_reduc = c7 ^ lsb7;
    if(const_col & lsb7) const_col ^= c7_reduc;
    if(c8 & lsb7) c8 ^= c7_reduc;
    if(c9 & lsb7) c9 ^= c7_reduc;
    if(c10 & lsb7) c10 ^= c7_reduc;
    if(c11 & lsb7) c11 ^= c7_reduc;
    if(c12 & lsb7) c12 ^= c7_reduc;
    if(c13 & lsb7) c13 ^= c7_reduc;
    if(c14 & lsb7) c14 ^= c7_reduc;
    mask ^= lsb7;

    uint64_t lsb8 = uint64_t_lsb(c8 & mask); // x8
    if(unlikely(!lsb8)) {
        return -1; // singular
    }

    uint64_t c8_reduc = c8 ^ lsb8;
    if(const_col & lsb8) const_col ^= c8_reduc;
    if(c9 & lsb8) c9 ^= c8_reduc;
    if(c10 & lsb8) c10 ^= c8_reduc;
    if(c11 & lsb8) c11 ^= c8_reduc;
    if(c12 & lsb8) c12 ^= c8_reduc;
    if(c13 & lsb8) c13 ^= c8_reduc;
    if(c14 & lsb8) c14 ^= c8_reduc;
    mask ^= lsb8;

    uint64_t lsb9 = uint64_t_lsb(c9 & mask); // x9
    if(unlikely(!lsb9)) {
        return -1; // singular
    }

    uint64_t c9_reduc = c9 ^ lsb9;
    if(const_col & lsb9) const_col ^= c9_reduc;
    if(c10 & lsb9) c10 ^= c9_reduc;
    if(c11 & lsb9) c11 ^= c9_reduc;
    if(c12 & lsb9) c12 ^= c9_reduc;
    if(c13 & lsb9) c13 ^= c9_reduc;
    if(c14 & lsb9) c14 ^= c9_reduc;
    mask ^= lsb9;

    uint64_t lsb10 = uint64_t_lsb(c10 & mask); // x10
    if(unlikely(!lsb10)) {
        return -1; // singular
    }

    uint64_t c10_reduc = c10 ^ lsb10;
    if(const_col & lsb10) const_col ^= c10_reduc;
    if(c11 & lsb10) c11 ^= c10_reduc;
    if(c12 & lsb10) c12 ^= c10_reduc;
    if(c13 & lsb10) c13 ^= c10_reduc;
    if(c14 & lsb10) c14 ^= c10_reduc;
    mask ^= lsb10;

    uint64_t lsb11 = uint64_t_lsb(c11 & mask); // x11
    if(unlikely(!lsb11)) {
        return -1; // singular
    }

    uint64_t c11_reduc = c11 ^ lsb11;
    if(const_col & lsb11) const_col ^= c11_reduc;
    if(c12 & lsb11) c12 ^= c11_reduc;
    if(c13 & lsb11) c13 ^= c11_reduc;
    if(c14 & lsb11) c14 ^= c11_reduc;
    mask ^= lsb11;

    uint64_t lsb12 = uint64_t_lsb(c12 & mask); // x12
    if(unlikely(!lsb12)) {
        return -1; // singular
    }

    uint64_t c12_reduc = c12 ^ lsb12;
    if(const_col & lsb12) const_col ^= c12_reduc;
    if(c13 & lsb12) c13 ^= c12_reduc;
    if(c14 & lsb12) c14 ^= c12_reduc;
    mask ^= lsb12;

    uint64_t lsb13 = uint64_t_lsb(c13 & mask); // x13
    if(unlikely(!lsb13)) {
        return -1; // singular
    }

    uint64_t c13_reduc = c13 ^ lsb13;
    if(const_col & lsb13) const_col ^= c13_reduc;
    if(c14 & lsb13) c14 ^= c13_reduc;
    mask ^= lsb13;

    uint64_t lsb14 = uint64_t_lsb(c14 & mask); // x14
    if(unlikely(!lsb14)) {
        return -1; // singular
    }

    if(const_col & lsb14) const_col ^= c14 ^ lsb14;
    mask ^= lsb14;

    if(likely(mask & const_col)) { // check system consistency
        return mask & const_col; // not solvable
    }

    uint64_t s = 0x0ULL;
    if(const_col & lsb1) s = uint64_t_toggle_at(s, 0);
    if(const_col & lsb2) s = uint64_t_toggle_at(s, 1);
    if(const_col & lsb3) s = uint64_t_toggle_at(s, 2);
    if(const_col & lsb4) s = uint64_t_toggle_at(s, 3);
    if(const_col & lsb5) s = uint64_t_toggle_at(s, 4);
    if(const_col & lsb6) s = uint64_t_toggle_at(s, 5);
    if(const_col & lsb7) s = uint64_t_toggle_at(s, 6);
    if(const_col & lsb8) s = uint64_t_toggle_at(s, 7);
    if(const_col & lsb9) s = uint64_t_toggle_at(s, 8);
    if(const_col & lsb10) s = uint64_t_toggle_at(s, 9);
    if(const_col & lsb11) s = uint64_t_toggle_at(s, 10);
    if(const_col & lsb12) s = uint64_t_toggle_at(s, 11);
    if(const_col & lsb13) s = uint64_t_toggle_at(s, 12);
    if(const_col & lsb14) s = uint64_t_toggle_at(s, 13);
    *sol = s;
    return 0;
}

int64_t
uint64a_gj_v15_generic(const uint64_t m[16], uint64_t* const restrict sol) {
    uint64_t const_col = m[0];
    uint64_t c1 = m[1]; // x1
    uint64_t c2 = m[2]; // x2
    uint64_t c3 = m[3]; // x3
    uint64_t c4 = m[4]; // x4
    uint64_t c5 = m[5]; // x5
    uint64_t c6 = m[6]; // x6
    uint64_t c7 = m[7]; // x7
    uint64_t c8 = m[8]; // x8
    uint64_t c9 = m[9]; // x9
    uint64_t c10 = m[10]; // x10
    uint64_t c11 = m[11]; // x11
    uint64_t c12 = m[12]; // x12
    uint64_t c13 = m[13]; // x13
    uint64_t c14 = m[14]; // x14
    uint64_t c15 = m[15]; // x15
    uint64_t lsb1 = uint64_t_lsb(c1); // initially all eqs are candidates
    if(unlikely(!lsb1)) {
        return -1; // singular
    }

    uint64_t c1_reduc = c1 ^ lsb1; // for row reduction
    if(const_col & lsb1) const_col ^= c1_reduc;
    if(c2 & lsb1) c2 ^= c1_reduc;
    if(c3 & lsb1) c3 ^= c1_reduc;
    if(c4 & lsb1) c4 ^= c1_reduc;
    if(c5 & lsb1) c5 ^= c1_reduc;
    if(c6 & lsb1) c6 ^= c1_reduc;
    if(c7 & lsb1) c7 ^= c1_reduc;
    if(c8 & lsb1) c8 ^= c1_reduc;
    if(c9 & lsb1) c9 ^= c1_reduc;
    if(c10 & lsb1) c10 ^= c1_reduc;
    if(c11 & lsb1) c11 ^= c1_reduc;
    if(c12 & lsb1) c12 ^= c1_reduc;
    if(c13 & lsb1) c13 ^= c1_reduc;
    if(c14 & lsb1) c14 ^= c1_reduc;
    if(c15 & lsb1) c15 ^= c1_reduc;
    uint64_t mask = ~lsb1;

    uint64_t lsb2 = uint64_t_lsb(c2 & mask); // x2
    if(unlikely(!lsb2)) {
        return -1; // singular
    }

    uint64_t c2_reduc = c2 ^ lsb2;
    if(const_col & lsb2) const_col ^= c2_reduc;
    if(c3 & lsb2) c3 ^= c2_reduc;
    if(c4 & lsb2) c4 ^= c2_reduc;
    if(c5 & lsb2) c5 ^= c2_reduc;
    if(c6 & lsb2) c6 ^= c2_reduc;
    if(c7 & lsb2) c7 ^= c2_reduc;
    if(c8 & lsb2) c8 ^= c2_reduc;
    if(c9 & lsb2) c9 ^= c2_reduc;
    if(c10 & lsb2) c10 ^= c2_reduc;
    if(c11 & lsb2) c11 ^= c2_reduc;
    if(c12 & lsb2) c12 ^= c2_reduc;
    if(c13 & lsb2) c13 ^= c2_reduc;
    if(c14 & lsb2) c14 ^= c2_reduc;
    if(c15 & lsb2) c15 ^= c2_reduc;
    mask ^= lsb2;

    uint64_t lsb3 = uint64_t_lsb(c3 & mask); // x3
    if(unlikely(!lsb3)) {
        return -1; // singular
    }

    uint64_t c3_reduc = c3 ^ lsb3;
    if(const_col & lsb3) const_col ^= c3_reduc;
    if(c4 & lsb3) c4 ^= c3_reduc;
    if(c5 & lsb3) c5 ^= c3_reduc;
    if(c6 & lsb3) c6 ^= c3_reduc;
    if(c7 & lsb3) c7 ^= c3_reduc;
    if(c8 & lsb3) c8 ^= c3_reduc;
    if(c9 & lsb3) c9 ^= c3_reduc;
    if(c10 & lsb3) c10 ^= c3_reduc;
    if(c11 & lsb3) c11 ^= c3_reduc;
    if(c12 & lsb3) c12 ^= c3_reduc;
    if(c13 & lsb3) c13 ^= c3_reduc;
    if(c14 & lsb3) c14 ^= c3_reduc;
    if(c15 & lsb3) c15 ^= c3_reduc;
    mask ^= lsb3;

    uint64_t lsb4 = uint64_t_lsb(c4 & mask); // x4
    if(unlikely(!lsb4)) {
        return -1; // singular
    }

    uint64_t c4_reduc = c4 ^ lsb4;
    if(const_col & lsb4) const_col ^= c4_reduc;
    if(c5 & lsb4) c5 ^= c4_reduc;
    if(c6 & lsb4) c6 ^= c4_reduc;
    if(c7 & lsb4) c7 ^= c4_reduc;
    if(c8 & lsb4) c8 ^= c4_reduc;
    if(c9 & lsb4) c9 ^= c4_reduc;
    if(c10 & lsb4) c10 ^= c4_reduc;
    if(c11 & lsb4) c11 ^= c4_reduc;
    if(c12 & lsb4) c12 ^= c4_reduc;
    if(c13 & lsb4) c13 ^= c4_reduc;
    if(c14 & lsb4) c14 ^= c4_reduc;
    if(c15 & lsb4) c15 ^= c4_reduc;
    mask ^= lsb4;

    uint64_t lsb5 = uint64_t_lsb(c5 & mask); // x5
    if(unlikely(!lsb5)) {
        return -1; // singular
    }

    uint64_t c5_reduc = c5 ^ lsb5;
    if(const_col & lsb5) const_col ^= c5_reduc;
    if(c6 & lsb5) c6 ^= c5_reduc;
    if(c7 & lsb5) c7 ^= c5_reduc;
    if(c8 & lsb5) c8 ^= c5_reduc;
    if(c9 & lsb5) c9 ^= c5_reduc;
    if(c10 & lsb5) c10 ^= c5_reduc;
    if(c11 & lsb5) c11 ^= c5_reduc;
    if(c12 & lsb5) c12 ^= c5_reduc;
    if(c13 & lsb5) c13 ^= c5_reduc;
    if(c14 & lsb5) c14 ^= c5_reduc;
    if(c15 & lsb5) c15 ^= c5_reduc;
    mask ^= lsb5;

    uint64_t lsb6 = uint64_t_lsb(c6 & mask); // x6
    if(unlikely(!lsb6)) {
        return -1; // singular
    }

    uint64_t c6_reduc = c6 ^ lsb6;
    if(const_col & lsb6) const_col ^= c6_reduc;
    if(c7 & lsb6) c7 ^= c6_reduc;
    if(c8 & lsb6) c8 ^= c6_reduc;
    if(c9 & lsb6) c9 ^= c6_reduc;
    if(c10 & lsb6) c10 ^= c6_reduc;
    if(c11 & lsb6) c11 ^= c6_reduc;
    if(c12 & lsb6) c12 ^= c6_reduc;
    if(c13 & lsb6) c13 ^= c6_reduc;
    if(c14 & lsb6) c14 ^= c6_reduc;
    if(c15 & lsb6) c15 ^= c6_reduc;
    mask ^= lsb6;

    uint64_t lsb7 = uint64_t_lsb(c7 & mask); // x7
    if(unlikely(!lsb7)) {
        return -1; // singular
    }

    uint64_t c7_reduc = c7 ^ lsb7;
    if(const_col & lsb7) const_col ^= c7_reduc;
    if(c8 & lsb7) c8 ^= c7_reduc;
    if(c9 & lsb7) c9 ^= c7_reduc;
    if(c10 & lsb7) c10 ^= c7_reduc;
    if(c11 & lsb7) c11 ^= c7_reduc;
    if(c12 & lsb7) c12 ^= c7_reduc;
    if(c13 & lsb7) c13 ^= c7_reduc;
    if(c14 & lsb7) c14 ^= c7_reduc;
    if(c15 & lsb7) c15 ^= c7_reduc;
    mask ^= lsb7;

    uint64_t lsb8 = uint64_t_lsb(c8 & mask); // x8
    if(unlikely(!lsb8)) {
        return -1; // singular
    }

    uint64_t c8_reduc = c8 ^ lsb8;
    if(const_col & lsb8) const_col ^= c8_reduc;
    if(c9 & lsb8) c9 ^= c8_reduc;
    if(c10 & lsb8) c10 ^= c8_reduc;
    if(c11 & lsb8) c11 ^= c8_reduc;
    if(c12 & lsb8) c12 ^= c8_reduc;
    if(c13 & lsb8) c13 ^= c8_reduc;
    if(c14 & lsb8) c14 ^= c8_reduc;
    if(c15 & lsb8) c15 ^= c8_reduc;
    mask ^= lsb8;

    uint64_t lsb9 = uint64_t_lsb(c9 & mask); // x9
    if(unlikely(!lsb9)) {
        return -1; // singular
    }

    uint64_t c9_reduc = c9 ^ lsb9;
    if(const_col & lsb9) const_col ^= c9_reduc;
    if(c10 & lsb9) c10 ^= c9_reduc;
    if(c11 & lsb9) c11 ^= c9_reduc;
    if(c12 & lsb9) c12 ^= c9_reduc;
    if(c13 & lsb9) c13 ^= c9_reduc;
    if(c14 & lsb9) c14 ^= c9_reduc;
    if(c15 & lsb9) c15 ^= c9_reduc;
    mask ^= lsb9;

    uint64_t lsb10 = uint64_t_lsb(c10 & mask); // x10
    if(unlikely(!lsb10)) {
        return -1; // singular
    }

    uint64_t c10_reduc = c10 ^ lsb10;
    if(const_col & lsb10) const_col ^= c10_reduc;
    if(c11 & lsb10) c11 ^= c10_reduc;
    if(c12 & lsb10) c12 ^= c10_reduc;
    if(c13 & lsb10) c13 ^= c10_reduc;
    if(c14 & lsb10) c14 ^= c10_reduc;
    if(c15 & lsb10) c15 ^= c10_reduc;
    mask ^= lsb10;

    uint64_t lsb11 = uint64_t_lsb(c11 & mask); // x11
    if(unlikely(!lsb11)) {
        return -1; // singular
    }

    uint64_t c11_reduc = c11 ^ lsb11;
    if(const_col & lsb11) const_col ^= c11_reduc;
    if(c12 & lsb11) c12 ^= c11_reduc;
    if(c13 & lsb11) c13 ^= c11_reduc;
    if(c14 & lsb11) c14 ^= c11_reduc;
    if(c15 & lsb11) c15 ^= c11_reduc;
    mask ^= lsb11;

    uint64_t lsb12 = uint64_t_lsb(c12 & mask); // x12
    if(unlikely(!lsb12)) {
        return -1; // singular
    }

    uint64_t c12_reduc = c12 ^ lsb12;
    if(const_col & lsb12) const_col ^= c12_reduc;
    if(c13 & lsb12) c13 ^= c12_reduc;
    if(c14 & lsb12) c14 ^= c12_reduc;
    if(c15 & lsb12) c15 ^= c12_reduc;
    mask ^= lsb12;

    uint64_t lsb13 = uint64_t_lsb(c13 & mask); // x13
    if(unlikely(!lsb13)) {
        return -1; // singular
    }

    uint64_t c13_reduc = c13 ^ lsb13;
    if(const_col & lsb13) const_col ^= c13_reduc;
    if(c14 & lsb13) c14 ^= c13_reduc;
    if(c15 & lsb13) c15 ^= c13_reduc;
    mask ^= lsb13;

    uint64_t lsb14 = uint64_t_lsb(c14 & mask); // x14
    if(unlikely(!lsb14)) {
        return -1; // singular
    }

    uint64_t c14_reduc = c14 ^ lsb14;
    if(const_col & lsb14) const_col ^= c14_reduc;
    if(c15 & lsb14) c15 ^= c14_reduc;
    mask ^= lsb14;

    uint64_t lsb15 = uint64_t_lsb(c15 & mask); // x15
    if(unlikely(!lsb15)) {
        return -1; // singular
    }

    if(const_col & lsb15) const_col ^= c15 ^ lsb15;
    mask ^= lsb15;

    if(likely(mask & const_col)) { // check system consistency
        return mask & const_col; // not solvable
    }

    uint64_t s = 0x0ULL;
    if(const_col & lsb1) s = uint64_t_toggle_at(s, 0);
    if(const_col & lsb2) s = uint64_t_toggle_at(s, 1);
    if(const_col & lsb3) s = uint64_t_toggle_at(s, 2);
    if(const_col & lsb4) s = uint64_t_toggle_at(s, 3);
    if(const_col & lsb5) s = uint64_t_toggle_at(s, 4);
    if(const_col & lsb6) s = uint64_t_toggle_at(s, 5);
    if(const_col & lsb7) s = uint64_t_toggle_at(s, 6);
    if(const_col & lsb8) s = uint64_t_toggle_at(s, 7);
    if(const_col & lsb9) s = uint64_t_toggle_at(s, 8);
    if(const_col & lsb10) s = uint64_t_toggle_at(s, 9);
    if(const_col & lsb11) s = uint64_t_toggle_at(s, 10);
    if(const_col & lsb12) s = uint64_t_toggle_at(s, 11);
    if(const_col & lsb13) s = uint64_t_toggle_at(s, 12);
    if(const_col & lsb14) s = uint64_t_toggle_at(s, 13);
    if(const_col & lsb15) s = uint64_t_toggle_at(s, 14);
    *sol = s;
    return 0;
}

int64_t
uint64a_gj_v16_generic(const uint64_t m[17], uint64_t* const restrict sol) {
    uint64_t const_col = m[0];
    uint64_t c1 = m[1]; // x1
    uint64_t c2 = m[2]; // x2
    uint64_t c3 = m[3]; // x3
    uint64_t c4 = m[4]; // x4
    uint64_t c5 = m[5]; // x5
    uint64_t c6 = m[6]; // x6
    uint64_t c7 = m[7]; // x7
    uint64_t c8 = m[8]; // x8
    uint64_t c9 = m[9]; // x9
    uint64_t c10 = m[10]; // x10
    uint64_t c11 = m[11]; // x11
    uint64_t c12 = m[12]; // x12
    uint64_t c13 = m[13]; // x13
    uint64_t c14 = m[14]; // x14
    uint64_t c15 = m[15]; // x15
    uint64_t c16 = m[16]; // x16
    uint64_t lsb1 = uint64_t_lsb(c1); // initially all eqs are candidates
    if(unlikely(!lsb1)) {
        return -1; // singular
    }

    uint64_t c1_reduc = c1 ^ lsb1; // for row reduction
    if(const_col & lsb1) const_col ^= c1_reduc;
    if(c2 & lsb1) c2 ^= c1_reduc;
    if(c3 & lsb1) c3 ^= c1_reduc;
    if(c4 & lsb1) c4 ^= c1_reduc;
    if(c5 & lsb1) c5 ^= c1_reduc;
    if(c6 & lsb1) c6 ^= c1_reduc;
    if(c7 & lsb1) c7 ^= c1_reduc;
    if(c8 & lsb1) c8 ^= c1_reduc;
    if(c9 & lsb1) c9 ^= c1_reduc;
    if(c10 & lsb1) c10 ^= c1_reduc;
    if(c11 & lsb1) c11 ^= c1_reduc;
    if(c12 & lsb1) c12 ^= c1_reduc;
    if(c13 & lsb1) c13 ^= c1_reduc;
    if(c14 & lsb1) c14 ^= c1_reduc;
    if(c15 & lsb1) c15 ^= c1_reduc;
    if(c16 & lsb1) c16 ^= c1_reduc;
    uint64_t mask = ~lsb1;

    uint64_t lsb2 = uint64_t_lsb(c2 & mask); // x2
    if(unlikely(!lsb2)) {
        return -1; // singular
    }

    uint64_t c2_reduc = c2 ^ lsb2;
    if(const_col & lsb2) const_col ^= c2_reduc;
    if(c3 & lsb2) c3 ^= c2_reduc;
    if(c4 & lsb2) c4 ^= c2_reduc;
    if(c5 & lsb2) c5 ^= c2_reduc;
    if(c6 & lsb2) c6 ^= c2_reduc;
    if(c7 & lsb2) c7 ^= c2_reduc;
    if(c8 & lsb2) c8 ^= c2_reduc;
    if(c9 & lsb2) c9 ^= c2_reduc;
    if(c10 & lsb2) c10 ^= c2_reduc;
    if(c11 & lsb2) c11 ^= c2_reduc;
    if(c12 & lsb2) c12 ^= c2_reduc;
    if(c13 & lsb2) c13 ^= c2_reduc;
    if(c14 & lsb2) c14 ^= c2_reduc;
    if(c15 & lsb2) c15 ^= c2_reduc;
    if(c16 & lsb2) c16 ^= c2_reduc;
    mask ^= lsb2;

    uint64_t lsb3 = uint64_t_lsb(c3 & mask); // x3
    if(unlikely(!lsb3)) {
        return -1; // singular
    }

    uint64_t c3_reduc = c3 ^ lsb3;
    if(const_col & lsb3) const_col ^= c3_reduc;
    if(c4 & lsb3) c4 ^= c3_reduc;
    if(c5 & lsb3) c5 ^= c3_reduc;
    if(c6 & lsb3) c6 ^= c3_reduc;
    if(c7 & lsb3) c7 ^= c3_reduc;
    if(c8 & lsb3) c8 ^= c3_reduc;
    if(c9 & lsb3) c9 ^= c3_reduc;
    if(c10 & lsb3) c10 ^= c3_reduc;
    if(c11 & lsb3) c11 ^= c3_reduc;
    if(c12 & lsb3) c12 ^= c3_reduc;
    if(c13 & lsb3) c13 ^= c3_reduc;
    if(c14 & lsb3) c14 ^= c3_reduc;
    if(c15 & lsb3) c15 ^= c3_reduc;
    if(c16 & lsb3) c16 ^= c3_reduc;
    mask ^= lsb3;

    uint64_t lsb4 = uint64_t_lsb(c4 & mask); // x4
    if(unlikely(!lsb4)) {
        return -1; // singular
    }

    uint64_t c4_reduc = c4 ^ lsb4;
    if(const_col & lsb4) const_col ^= c4_reduc;
    if(c5 & lsb4) c5 ^= c4_reduc;
    if(c6 & lsb4) c6 ^= c4_reduc;
    if(c7 & lsb4) c7 ^= c4_reduc;
    if(c8 & lsb4) c8 ^= c4_reduc;
    if(c9 & lsb4) c9 ^= c4_reduc;
    if(c10 & lsb4) c10 ^= c4_reduc;
    if(c11 & lsb4) c11 ^= c4_reduc;
    if(c12 & lsb4) c12 ^= c4_reduc;
    if(c13 & lsb4) c13 ^= c4_reduc;
    if(c14 & lsb4) c14 ^= c4_reduc;
    if(c15 & lsb4) c15 ^= c4_reduc;
    if(c16 & lsb4) c16 ^= c4_reduc;
    mask ^= lsb4;

    uint64_t lsb5 = uint64_t_lsb(c5 & mask); // x5
    if(unlikely(!lsb5)) {
        return -1; // singular
    }

    uint64_t c5_reduc = c5 ^ lsb5;
    if(const_col & lsb5) const_col ^= c5_reduc;
    if(c6 & lsb5) c6 ^= c5_reduc;
    if(c7 & lsb5) c7 ^= c5_reduc;
    if(c8 & lsb5) c8 ^= c5_reduc;
    if(c9 & lsb5) c9 ^= c5_reduc;
    if(c10 & lsb5) c10 ^= c5_reduc;
    if(c11 & lsb5) c11 ^= c5_reduc;
    if(c12 & lsb5) c12 ^= c5_reduc;
    if(c13 & lsb5) c13 ^= c5_reduc;
    if(c14 & lsb5) c14 ^= c5_reduc;
    if(c15 & lsb5) c15 ^= c5_reduc;
    if(c16 & lsb5) c16 ^= c5_reduc;
    mask ^= lsb5;

    uint64_t lsb6 = uint64_t_lsb(c6 & mask); // x6
    if(unlikely(!lsb6)) {
        return -1; // singular
    }

    uint64_t c6_reduc = c6 ^ lsb6;
    if(const_col & lsb6) const_col ^= c6_reduc;
    if(c7 & lsb6) c7 ^= c6_reduc;
    if(c8 & lsb6) c8 ^= c6_reduc;
    if(c9 & lsb6) c9 ^= c6_reduc;
    if(c10 & lsb6) c10 ^= c6_reduc;
    if(c11 & lsb6) c11 ^= c6_reduc;
    if(c12 & lsb6) c12 ^= c6_reduc;
    if(c13 & lsb6) c13 ^= c6_reduc;
    if(c14 & lsb6) c14 ^= c6_reduc;
    if(c15 & lsb6) c15 ^= c6_reduc;
    if(c16 & lsb6) c16 ^= c6_reduc;
    mask ^= lsb6;

    uint64_t lsb7 = uint64_t_lsb(c7 & mask); // x7
    if(unlikely(!lsb7)) {
        return -1; // singular
    }

    uint64_t c7_reduc = c7 ^ lsb7;
    if(const_col & lsb7) const_col ^= c7_reduc;
    if(c8 & lsb7) c8 ^= c7_reduc;
    if(c9 & lsb7) c9 ^= c7_reduc;
    if(c10 & lsb7) c10 ^= c7_reduc;
    if(c11 & lsb7) c11 ^= c7_reduc;
    if(c12 & lsb7) c12 ^= c7_reduc;
    if(c13 & lsb7) c13 ^= c7_reduc;
    if(c14 & lsb7) c14 ^= c7_reduc;
    if(c15 & lsb7) c15 ^= c7_reduc;
    if(c16 & lsb7) c16 ^= c7_reduc;
    mask ^= lsb7;

    uint64_t lsb8 = uint64_t_lsb(c8 & mask); // x8
    if(unlikely(!lsb8)) {
        return -1; // singular
    }

    uint64_t c8_reduc = c8 ^ lsb8;
    if(const_col & lsb8) const_col ^= c8_reduc;
    if(c9 & lsb8) c9 ^= c8_reduc;
    if(c10 & lsb8) c10 ^= c8_reduc;
    if(c11 & lsb8) c11 ^= c8_reduc;
    if(c12 & lsb8) c12 ^= c8_reduc;
    if(c13 & lsb8) c13 ^= c8_reduc;
    if(c14 & lsb8) c14 ^= c8_reduc;
    if(c15 & lsb8) c15 ^= c8_reduc;
    if(c16 & lsb8) c16 ^= c8_reduc;
    mask ^= lsb8;

    uint64_t lsb9 = uint64_t_lsb(c9 & mask); // x9
    if(unlikely(!lsb9)) {
        return -1; // singular
    }

    uint64_t c9_reduc = c9 ^ lsb9;
    if(const_col & lsb9) const_col ^= c9_reduc;
    if(c10 & lsb9) c10 ^= c9_reduc;
    if(c11 & lsb9) c11 ^= c9_reduc;
    if(c12 & lsb9) c12 ^= c9_reduc;
    if(c13 & lsb9) c13 ^= c9_reduc;
    if(c14 & lsb9) c14 ^= c9_reduc;
    if(c15 & lsb9) c15 ^= c9_reduc;
    if(c16 & lsb9) c16 ^= c9_reduc;
    mask ^= lsb9;

    uint64_t lsb10 = uint64_t_lsb(c10 & mask); // x10
    if(unlikely(!lsb10)) {
        return -1; // singular
    }

    uint64_t c10_reduc = c10 ^ lsb10;
    if(const_col & lsb10) const_col ^= c10_reduc;
    if(c11 & lsb10) c11 ^= c10_reduc;
    if(c12 & lsb10) c12 ^= c10_reduc;
    if(c13 & lsb10) c13 ^= c10_reduc;
    if(c14 & lsb10) c14 ^= c10_reduc;
    if(c15 & lsb10) c15 ^= c10_reduc;
    if(c16 & lsb10) c16 ^= c10_reduc;
    mask ^= lsb10;

    uint64_t lsb11 = uint64_t_lsb(c11 & mask); // x11
    if(unlikely(!lsb11)) {
        return -1; // singular
    }

    uint64_t c11_reduc = c11 ^ lsb11;
    if(const_col & lsb11) const_col ^= c11_reduc;
    if(c12 & lsb11) c12 ^= c11_reduc;
    if(c13 & lsb11) c13 ^= c11_reduc;
    if(c14 & lsb11) c14 ^= c11_reduc;
    if(c15 & lsb11) c15 ^= c11_reduc;
    if(c16 & lsb11) c16 ^= c11_reduc;
    mask ^= lsb11;

    uint64_t lsb12 = uint64_t_lsb(c12 & mask); // x12
    if(unlikely(!lsb12)) {
        return -1; // singular
    }

    uint64_t c12_reduc = c12 ^ lsb12;
    if(const_col & lsb12) const_col ^= c12_reduc;
    if(c13 & lsb12) c13 ^= c12_reduc;
    if(c14 & lsb12) c14 ^= c12_reduc;
    if(c15 & lsb12) c15 ^= c12_reduc;
    if(c16 & lsb12) c16 ^= c12_reduc;
    mask ^= lsb12;

    uint64_t lsb13 = uint64_t_lsb(c13 & mask); // x13
    if(unlikely(!lsb13)) {
        return -1; // singular
    }

    uint64_t c13_reduc = c13 ^ lsb13;
    if(const_col & lsb13) const_col ^= c13_reduc;
    if(c14 & lsb13) c14 ^= c13_reduc;
    if(c15 & lsb13) c15 ^= c13_reduc;
    if(c16 & lsb13) c16 ^= c13_reduc;
    mask ^= lsb13;

    uint64_t lsb14 = uint64_t_lsb(c14 & mask); // x14
    if(unlikely(!lsb14)) {
        return -1; // singular
    }

    uint64_t c14_reduc = c14 ^ lsb14;
    if(const_col & lsb14) const_col ^= c14_reduc;
    if(c15 & lsb14) c15 ^= c14_reduc;
    if(c16 & lsb14) c16 ^= c14_reduc;
    mask ^= lsb14;

    uint64_t lsb15 = uint64_t_lsb(c15 & mask); // x15
    if(unlikely(!lsb15)) {
        return -1; // singular
    }

    uint64_t c15_reduc = c15 ^ lsb15;
    if(const_col & lsb15) const_col ^= c15_reduc;
    if(c16 & lsb15) c16 ^= c15_reduc;
    mask ^= lsb15;

    uint64_t lsb16 = uint64_t_lsb(c16 & mask); // x16
    if(unlikely(!lsb16)) {
        return -1; // singular
    }

    if(const_col & lsb16) const_col ^= c16 ^ lsb16;
    mask ^= lsb16;

    if(likely(mask & const_col)) { // check system consistency
        return mask & const_col; // not solvable
    }

    uint64_t s = 0x0ULL;
    if(const_col & lsb1) s = uint64_t_toggle_at(s, 0);
    if(const_col & lsb2) s = uint64_t_toggle_at(s, 1);
    if(const_col & lsb3) s = uint64_t_toggle_at(s, 2);
    if(const_col & lsb4) s = uint64_t_toggle_at(s, 3);
    if(const_col & lsb5) s = uint64_t_toggle_at(s, 4);
    if(const_col & lsb6) s = uint64_t_toggle_at(s, 5);
    if(const_col & lsb7) s = uint64_t_toggle_at(s, 6);
    if(const_col & lsb8) s = uint64_t_toggle_at(s, 7);
    if(const_col & lsb9) s = uint64_t_toggle_at(s, 8);
    if(const_col & lsb10) s = uint64_t_toggle_at(s, 9);
    if(const_col & lsb11) s = uint64_t_toggle_at(s, 10);
    if(const_col & lsb12) s = uint64_t_toggle_at(s, 11);
    if(const_col & lsb13) s = uint64_t_toggle_at(s, 12);
    if(const_col & lsb14) s = uint64_t_toggle_at(s, 13);
    if(const_col & lsb15) s = uint64_t_toggle_at(s, 14);
    if(const_col & lsb16) s = uint64_t_toggle_at(s, 15);
    *sol = s;
    return 0;
}

int64_t
uint64a_gj_v17_generic(const uint64_t m[18], uint64_t* const restrict sol) {
    uint64_t const_col = m[0];
    uint64_t c1 = m[1]; // x1
    uint64_t c2 = m[2]; // x2
    uint64_t c3 = m[3]; // x3
    uint64_t c4 = m[4]; // x4
    uint64_t c5 = m[5]; // x5
    uint64_t c6 = m[6]; // x6
    uint64_t c7 = m[7]; // x7
    uint64_t c8 = m[8]; // x8
    uint64_t c9 = m[9]; // x9
    uint64_t c10 = m[10]; // x10
    uint64_t c11 = m[11]; // x11
    uint64_t c12 = m[12]; // x12
    uint64_t c13 = m[13]; // x13
    uint64_t c14 = m[14]; // x14
    uint64_t c15 = m[15]; // x15
    uint64_t c16 = m[16]; // x16
    uint64_t c17 = m[17]; // x17
    uint64_t lsb1 = uint64_t_lsb(c1); // initially all eqs are candidates
    if(unlikely(!lsb1)) {
        return -1; // singular
    }

    uint64_t c1_reduc = c1 ^ lsb1; // for row reduction
    if(const_col & lsb1) const_col ^= c1_reduc;
    if(c2 & lsb1) c2 ^= c1_reduc;
    if(c3 & lsb1) c3 ^= c1_reduc;
    if(c4 & lsb1) c4 ^= c1_reduc;
    if(c5 & lsb1) c5 ^= c1_reduc;
    if(c6 & lsb1) c6 ^= c1_reduc;
    if(c7 & lsb1) c7 ^= c1_reduc;
    if(c8 & lsb1) c8 ^= c1_reduc;
    if(c9 & lsb1) c9 ^= c1_reduc;
    if(c10 & lsb1) c10 ^= c1_reduc;
    if(c11 & lsb1) c11 ^= c1_reduc;
    if(c12 & lsb1) c12 ^= c1_reduc;
    if(c13 & lsb1) c13 ^= c1_reduc;
    if(c14 & lsb1) c14 ^= c1_reduc;
    if(c15 & lsb1) c15 ^= c1_reduc;
    if(c16 & lsb1) c16 ^= c1_reduc;
    if(c17 & lsb1) c17 ^= c1_reduc;
    uint64_t mask = ~lsb1;

    uint64_t lsb2 = uint64_t_lsb(c2 & mask); // x2
    if(unlikely(!lsb2)) {
        return -1; // singular
    }

    uint64_t c2_reduc = c2 ^ lsb2;
    if(const_col & lsb2) const_col ^= c2_reduc;
    if(c3 & lsb2) c3 ^= c2_reduc;
    if(c4 & lsb2) c4 ^= c2_reduc;
    if(c5 & lsb2) c5 ^= c2_reduc;
    if(c6 & lsb2) c6 ^= c2_reduc;
    if(c7 & lsb2) c7 ^= c2_reduc;
    if(c8 & lsb2) c8 ^= c2_reduc;
    if(c9 & lsb2) c9 ^= c2_reduc;
    if(c10 & lsb2) c10 ^= c2_reduc;
    if(c11 & lsb2) c11 ^= c2_reduc;
    if(c12 & lsb2) c12 ^= c2_reduc;
    if(c13 & lsb2) c13 ^= c2_reduc;
    if(c14 & lsb2) c14 ^= c2_reduc;
    if(c15 & lsb2) c15 ^= c2_reduc;
    if(c16 & lsb2) c16 ^= c2_reduc;
    if(c17 & lsb2) c17 ^= c2_reduc;
    mask ^= lsb2;

    uint64_t lsb3 = uint64_t_lsb(c3 & mask); // x3
    if(unlikely(!lsb3)) {
        return -1; // singular
    }

    uint64_t c3_reduc = c3 ^ lsb3;
    if(const_col & lsb3) const_col ^= c3_reduc;
    if(c4 & lsb3) c4 ^= c3_reduc;
    if(c5 & lsb3) c5 ^= c3_reduc;
    if(c6 & lsb3) c6 ^= c3_reduc;
    if(c7 & lsb3) c7 ^= c3_reduc;
    if(c8 & lsb3) c8 ^= c3_reduc;
    if(c9 & lsb3) c9 ^= c3_reduc;
    if(c10 & lsb3) c10 ^= c3_reduc;
    if(c11 & lsb3) c11 ^= c3_reduc;
    if(c12 & lsb3) c12 ^= c3_reduc;
    if(c13 & lsb3) c13 ^= c3_reduc;
    if(c14 & lsb3) c14 ^= c3_reduc;
    if(c15 & lsb3) c15 ^= c3_reduc;
    if(c16 & lsb3) c16 ^= c3_reduc;
    if(c17 & lsb3) c17 ^= c3_reduc;
    mask ^= lsb3;

    uint64_t lsb4 = uint64_t_lsb(c4 & mask); // x4
    if(unlikely(!lsb4)) {
        return -1; // singular
    }

    uint64_t c4_reduc = c4 ^ lsb4;
    if(const_col & lsb4) const_col ^= c4_reduc;
    if(c5 & lsb4) c5 ^= c4_reduc;
    if(c6 & lsb4) c6 ^= c4_reduc;
    if(c7 & lsb4) c7 ^= c4_reduc;
    if(c8 & lsb4) c8 ^= c4_reduc;
    if(c9 & lsb4) c9 ^= c4_reduc;
    if(c10 & lsb4) c10 ^= c4_reduc;
    if(c11 & lsb4) c11 ^= c4_reduc;
    if(c12 & lsb4) c12 ^= c4_reduc;
    if(c13 & lsb4) c13 ^= c4_reduc;
    if(c14 & lsb4) c14 ^= c4_reduc;
    if(c15 & lsb4) c15 ^= c4_reduc;
    if(c16 & lsb4) c16 ^= c4_reduc;
    if(c17 & lsb4) c17 ^= c4_reduc;
    mask ^= lsb4;

    uint64_t lsb5 = uint64_t_lsb(c5 & mask); // x5
    if(unlikely(!lsb5)) {
        return -1; // singular
    }

    uint64_t c5_reduc = c5 ^ lsb5;
    if(const_col & lsb5) const_col ^= c5_reduc;
    if(c6 & lsb5) c6 ^= c5_reduc;
    if(c7 & lsb5) c7 ^= c5_reduc;
    if(c8 & lsb5) c8 ^= c5_reduc;
    if(c9 & lsb5) c9 ^= c5_reduc;
    if(c10 & lsb5) c10 ^= c5_reduc;
    if(c11 & lsb5) c11 ^= c5_reduc;
    if(c12 & lsb5) c12 ^= c5_reduc;
    if(c13 & lsb5) c13 ^= c5_reduc;
    if(c14 & lsb5) c14 ^= c5_reduc;
    if(c15 & lsb5) c15 ^= c5_reduc;
    if(c16 & lsb5) c16 ^= c5_reduc;
    if(c17 & lsb5) c17 ^= c5_reduc;
    mask ^= lsb5;

    uint64_t lsb6 = uint64_t_lsb(c6 & mask); // x6
    if(unlikely(!lsb6)) {
        return -1; // singular
    }

    uint64_t c6_reduc = c6 ^ lsb6;
    if(const_col & lsb6) const_col ^= c6_reduc;
    if(c7 & lsb6) c7 ^= c6_reduc;
    if(c8 & lsb6) c8 ^= c6_reduc;
    if(c9 & lsb6) c9 ^= c6_reduc;
    if(c10 & lsb6) c10 ^= c6_reduc;
    if(c11 & lsb6) c11 ^= c6_reduc;
    if(c12 & lsb6) c12 ^= c6_reduc;
    if(c13 & lsb6) c13 ^= c6_reduc;
    if(c14 & lsb6) c14 ^= c6_reduc;
    if(c15 & lsb6) c15 ^= c6_reduc;
    if(c16 & lsb6) c16 ^= c6_reduc;
    if(c17 & lsb6) c17 ^= c6_reduc;
    mask ^= lsb6;

    uint64_t lsb7 = uint64_t_lsb(c7 & mask); // x7
    if(unlikely(!lsb7)) {
        return -1; // singular
    }

    uint64_t c7_reduc = c7 ^ lsb7;
    if(const_col & lsb7) const_col ^= c7_reduc;
    if(c8 & lsb7) c8 ^= c7_reduc;
    if(c9 & lsb7) c9 ^= c7_reduc;
    if(c10 & lsb7) c10 ^= c7_reduc;
    if(c11 & lsb7) c11 ^= c7_reduc;
    if(c12 & lsb7) c12 ^= c7_reduc;
    if(c13 & lsb7) c13 ^= c7_reduc;
    if(c14 & lsb7) c14 ^= c7_reduc;
    if(c15 & lsb7) c15 ^= c7_reduc;
    if(c16 & lsb7) c16 ^= c7_reduc;
    if(c17 & lsb7) c17 ^= c7_reduc;
    mask ^= lsb7;

    uint64_t lsb8 = uint64_t_lsb(c8 & mask); // x8
    if(unlikely(!lsb8)) {
        return -1; // singular
    }

    uint64_t c8_reduc = c8 ^ lsb8;
    if(const_col & lsb8) const_col ^= c8_reduc;
    if(c9 & lsb8) c9 ^= c8_reduc;
    if(c10 & lsb8) c10 ^= c8_reduc;
    if(c11 & lsb8) c11 ^= c8_reduc;
    if(c12 & lsb8) c12 ^= c8_reduc;
    if(c13 & lsb8) c13 ^= c8_reduc;
    if(c14 & lsb8) c14 ^= c8_reduc;
    if(c15 & lsb8) c15 ^= c8_reduc;
    if(c16 & lsb8) c16 ^= c8_reduc;
    if(c17 & lsb8) c17 ^= c8_reduc;
    mask ^= lsb8;

    uint64_t lsb9 = uint64_t_lsb(c9 & mask); // x9
    if(unlikely(!lsb9)) {
        return -1; // singular
    }

    uint64_t c9_reduc = c9 ^ lsb9;
    if(const_col & lsb9) const_col ^= c9_reduc;
    if(c10 & lsb9) c10 ^= c9_reduc;
    if(c11 & lsb9) c11 ^= c9_reduc;
    if(c12 & lsb9) c12 ^= c9_reduc;
    if(c13 & lsb9) c13 ^= c9_reduc;
    if(c14 & lsb9) c14 ^= c9_reduc;
    if(c15 & lsb9) c15 ^= c9_reduc;
    if(c16 & lsb9) c16 ^= c9_reduc;
    if(c17 & lsb9) c17 ^= c9_reduc;
    mask ^= lsb9;

    uint64_t lsb10 = uint64_t_lsb(c10 & mask); // x10
    if(unlikely(!lsb10)) {
        return -1; // singular
    }

    uint64_t c10_reduc = c10 ^ lsb10;
    if(const_col & lsb10) const_col ^= c10_reduc;
    if(c11 & lsb10) c11 ^= c10_reduc;
    if(c12 & lsb10) c12 ^= c10_reduc;
    if(c13 & lsb10) c13 ^= c10_reduc;
    if(c14 & lsb10) c14 ^= c10_reduc;
    if(c15 & lsb10) c15 ^= c10_reduc;
    if(c16 & lsb10) c16 ^= c10_reduc;
    if(c17 & lsb10) c17 ^= c10_reduc;
    mask ^= lsb10;

    uint64_t lsb11 = uint64_t_lsb(c11 & mask); // x11
    if(unlikely(!lsb11)) {
        return -1; // singular
    }

    uint64_t c11_reduc = c11 ^ lsb11;
    if(const_col & lsb11) const_col ^= c11_reduc;
    if(c12 & lsb11) c12 ^= c11_reduc;
    if(c13 & lsb11) c13 ^= c11_reduc;
    if(c14 & lsb11) c14 ^= c11_reduc;
    if(c15 & lsb11) c15 ^= c11_reduc;
    if(c16 & lsb11) c16 ^= c11_reduc;
    if(c17 & lsb11) c17 ^= c11_reduc;
    mask ^= lsb11;

    uint64_t lsb12 = uint64_t_lsb(c12 & mask); // x12
    if(unlikely(!lsb12)) {
        return -1; // singular
    }

    uint64_t c12_reduc = c12 ^ lsb12;
    if(const_col & lsb12) const_col ^= c12_reduc;
    if(c13 & lsb12) c13 ^= c12_reduc;
    if(c14 & lsb12) c14 ^= c12_reduc;
    if(c15 & lsb12) c15 ^= c12_reduc;
    if(c16 & lsb12) c16 ^= c12_reduc;
    if(c17 & lsb12) c17 ^= c12_reduc;
    mask ^= lsb12;

    uint64_t lsb13 = uint64_t_lsb(c13 & mask); // x13
    if(unlikely(!lsb13)) {
        return -1; // singular
    }

    uint64_t c13_reduc = c13 ^ lsb13;
    if(const_col & lsb13) const_col ^= c13_reduc;
    if(c14 & lsb13) c14 ^= c13_reduc;
    if(c15 & lsb13) c15 ^= c13_reduc;
    if(c16 & lsb13) c16 ^= c13_reduc;
    if(c17 & lsb13) c17 ^= c13_reduc;
    mask ^= lsb13;

    uint64_t lsb14 = uint64_t_lsb(c14 & mask); // x14
    if(unlikely(!lsb14)) {
        return -1; // singular
    }

    uint64_t c14_reduc = c14 ^ lsb14;
    if(const_col & lsb14) const_col ^= c14_reduc;
    if(c15 & lsb14) c15 ^= c14_reduc;
    if(c16 & lsb14) c16 ^= c14_reduc;
    if(c17 & lsb14) c17 ^= c14_reduc;
    mask ^= lsb14;

    uint64_t lsb15 = uint64_t_lsb(c15 & mask); // x15
    if(unlikely(!lsb15)) {
        return -1; // singular
    }

    uint64_t c15_reduc = c15 ^ lsb15;
    if(const_col & lsb15) const_col ^= c15_reduc;
    if(c16 & lsb15) c16 ^= c15_reduc;
    if(c17 & lsb15) c17 ^= c15_reduc;
    mask ^= lsb15;

    uint64_t lsb16 = uint64_t_lsb(c16 & mask); // x16
    if(unlikely(!lsb16)) {
        return -1; // singular
    }

    uint64_t c16_reduc = c16 ^ lsb16;
    if(const_col & lsb16) const_col ^= c16_reduc;
    if(c17 & lsb16) c17 ^= c16_reduc;
    mask ^= lsb16;

    uint64_t lsb17 = uint64_t_lsb(c17 & mask); // x17
    if(unlikely(!lsb17)) {
        return -1; // singular
    }

    if(const_col & lsb17) const_col ^= c17 ^ lsb17;
    mask ^= lsb17;

    if(likely(mask & const_col)) { // check system consistency
        return mask & const_col; // not solvable
    }

    uint64_t s = 0x0ULL;
    if(const_col & lsb1) s = uint64_t_toggle_at(s, 0);
    if(const_col & lsb2) s = uint64_t_toggle_at(s, 1);
    if(const_col & lsb3) s = uint64_t_toggle_at(s, 2);
    if(const_col & lsb4) s = uint64_t_toggle_at(s, 3);
    if(const_col & lsb5) s = uint64_t_toggle_at(s, 4);
    if(const_col & lsb6) s = uint64_t_toggle_at(s, 5);
    if(const_col & lsb7) s = uint64_t_toggle_at(s, 6);
    if(const_col & lsb8) s = uint64_t_toggle_at(s, 7);
    if(const_col & lsb9) s = uint64_t_toggle_at(s, 8);
    if(const_col & lsb10) s = uint64_t_toggle_at(s, 9);
    if(const_col & lsb11) s = uint64_t_toggle_at(s, 10);
    if(const_col & lsb12) s = uint64_t_toggle_at(s, 11);
    if(const_col & lsb13) s = uint64_t_toggle_at(s, 12);
    if(const_col & lsb14) s = uint64_t_toggle_at(s, 13);
    if(const_col & lsb15) s = uint64_t_toggle_at(s, 14);
    if(const_col & lsb16) s = uint64_t_toggle_at(s, 15);
    if(const_col & lsb17) s = uint64_t_toggle_at(s, 16);
    *sol = s;
    return 0;
}

int64_t
uint64a_gj_v18_generic(const uint64_t m[19], uint64_t* const restrict sol) {
    uint64_t const_col = m[0];
    uint64_t c1 = m[1]; // x1
    uint64_t c2 = m[2]; // x2
    uint64_t c3 = m[3]; // x3
    uint64_t c4 = m[4]; // x4
    uint64_t c5 = m[5]; // x5
    uint64_t c6 = m[6]; // x6
    uint64_t c7 = m[7]; // x7
    uint64_t c8 = m[8]; // x8
    uint64_t c9 = m[9]; // x9
    uint64_t c10 = m[10]; // x10
    uint64_t c11 = m[11]; // x11
    uint64_t c12 = m[12]; // x12
    uint64_t c13 = m[13]; // x13
    uint64_t c14 = m[14]; // x14
    uint64_t c15 = m[15]; // x15
    uint64_t c16 = m[16]; // x16
    uint64_t c17 = m[17]; // x17
    uint64_t c18 = m[18]; // x18
    uint64_t lsb1 = uint64_t_lsb(c1); // initially all eqs are candidates
    if(unlikely(!lsb1)) {
        return -1; // singular
    }

    uint64_t c1_reduc = c1 ^ lsb1; // for row reduction
    if(const_col & lsb1) const_col ^= c1_reduc;
    if(c2 & lsb1) c2 ^= c1_reduc;
    if(c3 & lsb1) c3 ^= c1_reduc;
    if(c4 & lsb1) c4 ^= c1_reduc;
    if(c5 & lsb1) c5 ^= c1_reduc;
    if(c6 & lsb1) c6 ^= c1_reduc;
    if(c7 & lsb1) c7 ^= c1_reduc;
    if(c8 & lsb1) c8 ^= c1_reduc;
    if(c9 & lsb1) c9 ^= c1_reduc;
    if(c10 & lsb1) c10 ^= c1_reduc;
    if(c11 & lsb1) c11 ^= c1_reduc;
    if(c12 & lsb1) c12 ^= c1_reduc;
    if(c13 & lsb1) c13 ^= c1_reduc;
    if(c14 & lsb1) c14 ^= c1_reduc;
    if(c15 & lsb1) c15 ^= c1_reduc;
    if(c16 & lsb1) c16 ^= c1_reduc;
    if(c17 & lsb1) c17 ^= c1_reduc;
    if(c18 & lsb1) c18 ^= c1_reduc;
    uint64_t mask = ~lsb1;

    uint64_t lsb2 = uint64_t_lsb(c2 & mask); // x2
    if(unlikely(!lsb2)) {
        return -1; // singular
    }

    uint64_t c2_reduc = c2 ^ lsb2;
    if(const_col & lsb2) const_col ^= c2_reduc;
    if(c3 & lsb2) c3 ^= c2_reduc;
    if(c4 & lsb2) c4 ^= c2_reduc;
    if(c5 & lsb2) c5 ^= c2_reduc;
    if(c6 & lsb2) c6 ^= c2_reduc;
    if(c7 & lsb2) c7 ^= c2_reduc;
    if(c8 & lsb2) c8 ^= c2_reduc;
    if(c9 & lsb2) c9 ^= c2_reduc;
    if(c10 & lsb2) c10 ^= c2_reduc;
    if(c11 & lsb2) c11 ^= c2_reduc;
    if(c12 & lsb2) c12 ^= c2_reduc;
    if(c13 & lsb2) c13 ^= c2_reduc;
    if(c14 & lsb2) c14 ^= c2_reduc;
    if(c15 & lsb2) c15 ^= c2_reduc;
    if(c16 & lsb2) c16 ^= c2_reduc;
    if(c17 & lsb2) c17 ^= c2_reduc;
    if(c18 & lsb2) c18 ^= c2_reduc;
    mask ^= lsb2;

    uint64_t lsb3 = uint64_t_lsb(c3 & mask); // x3
    if(unlikely(!lsb3)) {
        return -1; // singular
    }

    uint64_t c3_reduc = c3 ^ lsb3;
    if(const_col & lsb3) const_col ^= c3_reduc;
    if(c4 & lsb3) c4 ^= c3_reduc;
    if(c5 & lsb3) c5 ^= c3_reduc;
    if(c6 & lsb3) c6 ^= c3_reduc;
    if(c7 & lsb3) c7 ^= c3_reduc;
    if(c8 & lsb3) c8 ^= c3_reduc;
    if(c9 & lsb3) c9 ^= c3_reduc;
    if(c10 & lsb3) c10 ^= c3_reduc;
    if(c11 & lsb3) c11 ^= c3_reduc;
    if(c12 & lsb3) c12 ^= c3_reduc;
    if(c13 & lsb3) c13 ^= c3_reduc;
    if(c14 & lsb3) c14 ^= c3_reduc;
    if(c15 & lsb3) c15 ^= c3_reduc;
    if(c16 & lsb3) c16 ^= c3_reduc;
    if(c17 & lsb3) c17 ^= c3_reduc;
    if(c18 & lsb3) c18 ^= c3_reduc;
    mask ^= lsb3;

    uint64_t lsb4 = uint64_t_lsb(c4 & mask); // x4
    if(unlikely(!lsb4)) {
        return -1; // singular
    }

    uint64_t c4_reduc = c4 ^ lsb4;
    if(const_col & lsb4) const_col ^= c4_reduc;
    if(c5 & lsb4) c5 ^= c4_reduc;
    if(c6 & lsb4) c6 ^= c4_reduc;
    if(c7 & lsb4) c7 ^= c4_reduc;
    if(c8 & lsb4) c8 ^= c4_reduc;
    if(c9 & lsb4) c9 ^= c4_reduc;
    if(c10 & lsb4) c10 ^= c4_reduc;
    if(c11 & lsb4) c11 ^= c4_reduc;
    if(c12 & lsb4) c12 ^= c4_reduc;
    if(c13 & lsb4) c13 ^= c4_reduc;
    if(c14 & lsb4) c14 ^= c4_reduc;
    if(c15 & lsb4) c15 ^= c4_reduc;
    if(c16 & lsb4) c16 ^= c4_reduc;
    if(c17 & lsb4) c17 ^= c4_reduc;
    if(c18 & lsb4) c18 ^= c4_reduc;
    mask ^= lsb4;

    uint64_t lsb5 = uint64_t_lsb(c5 & mask); // x5
    if(unlikely(!lsb5)) {
        return -1; // singular
    }

    uint64_t c5_reduc = c5 ^ lsb5;
    if(const_col & lsb5) const_col ^= c5_reduc;
    if(c6 & lsb5) c6 ^= c5_reduc;
    if(c7 & lsb5) c7 ^= c5_reduc;
    if(c8 & lsb5) c8 ^= c5_reduc;
    if(c9 & lsb5) c9 ^= c5_reduc;
    if(c10 & lsb5) c10 ^= c5_reduc;
    if(c11 & lsb5) c11 ^= c5_reduc;
    if(c12 & lsb5) c12 ^= c5_reduc;
    if(c13 & lsb5) c13 ^= c5_reduc;
    if(c14 & lsb5) c14 ^= c5_reduc;
    if(c15 & lsb5) c15 ^= c5_reduc;
    if(c16 & lsb5) c16 ^= c5_reduc;
    if(c17 & lsb5) c17 ^= c5_reduc;
    if(c18 & lsb5) c18 ^= c5_reduc;
    mask ^= lsb5;

    uint64_t lsb6 = uint64_t_lsb(c6 & mask); // x6
    if(unlikely(!lsb6)) {
        return -1; // singular
    }

    uint64_t c6_reduc = c6 ^ lsb6;
    if(const_col & lsb6) const_col ^= c6_reduc;
    if(c7 & lsb6) c7 ^= c6_reduc;
    if(c8 & lsb6) c8 ^= c6_reduc;
    if(c9 & lsb6) c9 ^= c6_reduc;
    if(c10 & lsb6) c10 ^= c6_reduc;
    if(c11 & lsb6) c11 ^= c6_reduc;
    if(c12 & lsb6) c12 ^= c6_reduc;
    if(c13 & lsb6) c13 ^= c6_reduc;
    if(c14 & lsb6) c14 ^= c6_reduc;
    if(c15 & lsb6) c15 ^= c6_reduc;
    if(c16 & lsb6) c16 ^= c6_reduc;
    if(c17 & lsb6) c17 ^= c6_reduc;
    if(c18 & lsb6) c18 ^= c6_reduc;
    mask ^= lsb6;

    uint64_t lsb7 = uint64_t_lsb(c7 & mask); // x7
    if(unlikely(!lsb7)) {
        return -1; // singular
    }

    uint64_t c7_reduc = c7 ^ lsb7;
    if(const_col & lsb7) const_col ^= c7_reduc;
    if(c8 & lsb7) c8 ^= c7_reduc;
    if(c9 & lsb7) c9 ^= c7_reduc;
    if(c10 & lsb7) c10 ^= c7_reduc;
    if(c11 & lsb7) c11 ^= c7_reduc;
    if(c12 & lsb7) c12 ^= c7_reduc;
    if(c13 & lsb7) c13 ^= c7_reduc;
    if(c14 & lsb7) c14 ^= c7_reduc;
    if(c15 & lsb7) c15 ^= c7_reduc;
    if(c16 & lsb7) c16 ^= c7_reduc;
    if(c17 & lsb7) c17 ^= c7_reduc;
    if(c18 & lsb7) c18 ^= c7_reduc;
    mask ^= lsb7;

    uint64_t lsb8 = uint64_t_lsb(c8 & mask); // x8
    if(unlikely(!lsb8)) {
        return -1; // singular
    }

    uint64_t c8_reduc = c8 ^ lsb8;
    if(const_col & lsb8) const_col ^= c8_reduc;
    if(c9 & lsb8) c9 ^= c8_reduc;
    if(c10 & lsb8) c10 ^= c8_reduc;
    if(c11 & lsb8) c11 ^= c8_reduc;
    if(c12 & lsb8) c12 ^= c8_reduc;
    if(c13 & lsb8) c13 ^= c8_reduc;
    if(c14 & lsb8) c14 ^= c8_reduc;
    if(c15 & lsb8) c15 ^= c8_reduc;
    if(c16 & lsb8) c16 ^= c8_reduc;
    if(c17 & lsb8) c17 ^= c8_reduc;
    if(c18 & lsb8) c18 ^= c8_reduc;
    mask ^= lsb8;

    uint64_t lsb9 = uint64_t_lsb(c9 & mask); // x9
    if(unlikely(!lsb9)) {
        return -1; // singular
    }

    uint64_t c9_reduc = c9 ^ lsb9;
    if(const_col & lsb9) const_col ^= c9_reduc;
    if(c10 & lsb9) c10 ^= c9_reduc;
    if(c11 & lsb9) c11 ^= c9_reduc;
    if(c12 & lsb9) c12 ^= c9_reduc;
    if(c13 & lsb9) c13 ^= c9_reduc;
    if(c14 & lsb9) c14 ^= c9_reduc;
    if(c15 & lsb9) c15 ^= c9_reduc;
    if(c16 & lsb9) c16 ^= c9_reduc;
    if(c17 & lsb9) c17 ^= c9_reduc;
    if(c18 & lsb9) c18 ^= c9_reduc;
    mask ^= lsb9;

    uint64_t lsb10 = uint64_t_lsb(c10 & mask); // x10
    if(unlikely(!lsb10)) {
        return -1; // singular
    }

    uint64_t c10_reduc = c10 ^ lsb10;
    if(const_col & lsb10) const_col ^= c10_reduc;
    if(c11 & lsb10) c11 ^= c10_reduc;
    if(c12 & lsb10) c12 ^= c10_reduc;
    if(c13 & lsb10) c13 ^= c10_reduc;
    if(c14 & lsb10) c14 ^= c10_reduc;
    if(c15 & lsb10) c15 ^= c10_reduc;
    if(c16 & lsb10) c16 ^= c10_reduc;
    if(c17 & lsb10) c17 ^= c10_reduc;
    if(c18 & lsb10) c18 ^= c10_reduc;
    mask ^= lsb10;

    uint64_t lsb11 = uint64_t_lsb(c11 & mask); // x11
    if(unlikely(!lsb11)) {
        return -1; // singular
    }

    uint64_t c11_reduc = c11 ^ lsb11;
    if(const_col & lsb11) const_col ^= c11_reduc;
    if(c12 & lsb11) c12 ^= c11_reduc;
    if(c13 & lsb11) c13 ^= c11_reduc;
    if(c14 & lsb11) c14 ^= c11_reduc;
    if(c15 & lsb11) c15 ^= c11_reduc;
    if(c16 & lsb11) c16 ^= c11_reduc;
    if(c17 & lsb11) c17 ^= c11_reduc;
    if(c18 & lsb11) c18 ^= c11_reduc;
    mask ^= lsb11;

    uint64_t lsb12 = uint64_t_lsb(c12 & mask); // x12
    if(unlikely(!lsb12)) {
        return -1; // singular
    }

    uint64_t c12_reduc = c12 ^ lsb12;
    if(const_col & lsb12) const_col ^= c12_reduc;
    if(c13 & lsb12) c13 ^= c12_reduc;
    if(c14 & lsb12) c14 ^= c12_reduc;
    if(c15 & lsb12) c15 ^= c12_reduc;
    if(c16 & lsb12) c16 ^= c12_reduc;
    if(c17 & lsb12) c17 ^= c12_reduc;
    if(c18 & lsb12) c18 ^= c12_reduc;
    mask ^= lsb12;

    uint64_t lsb13 = uint64_t_lsb(c13 & mask); // x13
    if(unlikely(!lsb13)) {
        return -1; // singular
    }

    uint64_t c13_reduc = c13 ^ lsb13;
    if(const_col & lsb13) const_col ^= c13_reduc;
    if(c14 & lsb13) c14 ^= c13_reduc;
    if(c15 & lsb13) c15 ^= c13_reduc;
    if(c16 & lsb13) c16 ^= c13_reduc;
    if(c17 & lsb13) c17 ^= c13_reduc;
    if(c18 & lsb13) c18 ^= c13_reduc;
    mask ^= lsb13;

    uint64_t lsb14 = uint64_t_lsb(c14 & mask); // x14
    if(unlikely(!lsb14)) {
        return -1; // singular
    }

    uint64_t c14_reduc = c14 ^ lsb14;
    if(const_col & lsb14) const_col ^= c14_reduc;
    if(c15 & lsb14) c15 ^= c14_reduc;
    if(c16 & lsb14) c16 ^= c14_reduc;
    if(c17 & lsb14) c17 ^= c14_reduc;
    if(c18 & lsb14) c18 ^= c14_reduc;
    mask ^= lsb14;

    uint64_t lsb15 = uint64_t_lsb(c15 & mask); // x15
    if(unlikely(!lsb15)) {
        return -1; // singular
    }

    uint64_t c15_reduc = c15 ^ lsb15;
    if(const_col & lsb15) const_col ^= c15_reduc;
    if(c16 & lsb15) c16 ^= c15_reduc;
    if(c17 & lsb15) c17 ^= c15_reduc;
    if(c18 & lsb15) c18 ^= c15_reduc;
    mask ^= lsb15;

    uint64_t lsb16 = uint64_t_lsb(c16 & mask); // x16
    if(unlikely(!lsb16)) {
        return -1; // singular
    }

    uint64_t c16_reduc = c16 ^ lsb16;
    if(const_col & lsb16) const_col ^= c16_reduc;
    if(c17 & lsb16) c17 ^= c16_reduc;
    if(c18 & lsb16) c18 ^= c16_reduc;
    mask ^= lsb16;

    uint64_t lsb17 = uint64_t_lsb(c17 & mask); // x17
    if(unlikely(!lsb17)) {
        return -1; // singular
    }

    uint64_t c17_reduc = c17 ^ lsb17;
    if(const_col & lsb17) const_col ^= c17_reduc;
    if(c18 & lsb17) c18 ^= c17_reduc;
    mask ^= lsb17;

    uint64_t lsb18 = uint64_t_lsb(c18 & mask); // x18
    if(unlikely(!lsb18)) {
        return -1; // singular
    }

    if(const_col & lsb18) const_col ^= c18 ^ lsb18;
    mask ^= lsb18;

    if(likely(mask & const_col)) { // check system consistency
        return mask & const_col; // not solvable
    }

    uint64_t s = 0x0ULL;
    if(const_col & lsb1) s = uint64_t_toggle_at(s, 0);
    if(const_col & lsb2) s = uint64_t_toggle_at(s, 1);
    if(const_col & lsb3) s = uint64_t_toggle_at(s, 2);
    if(const_col & lsb4) s = uint64_t_toggle_at(s, 3);
    if(const_col & lsb5) s = uint64_t_toggle_at(s, 4);
    if(const_col & lsb6) s = uint64_t_toggle_at(s, 5);
    if(const_col & lsb7) s = uint64_t_toggle_at(s, 6);
    if(const_col & lsb8) s = uint64_t_toggle_at(s, 7);
    if(const_col & lsb9) s = uint64_t_toggle_at(s, 8);
    if(const_col & lsb10) s = uint64_t_toggle_at(s, 9);
    if(const_col & lsb11) s = uint64_t_toggle_at(s, 10);
    if(const_col & lsb12) s = uint64_t_toggle_at(s, 11);
    if(const_col & lsb13) s = uint64_t_toggle_at(s, 12);
    if(const_col & lsb14) s = uint64_t_toggle_at(s, 13);
    if(const_col & lsb15) s = uint64_t_toggle_at(s, 14);
    if(const_col & lsb16) s = uint64_t_toggle_at(s, 15);
    if(const_col & lsb17) s = uint64_t_toggle_at(s, 16);
    if(const_col & lsb18) s = uint64_t_toggle_at(s, 17);
    *sol = s;
    return 0;
}

int64_t
uint64a_gj_v19_generic(const uint64_t m[20], uint64_t* const restrict sol) {
    uint64_t const_col = m[0];
    uint64_t c1 = m[1]; // x1
    uint64_t c2 = m[2]; // x2
    uint64_t c3 = m[3]; // x3
    uint64_t c4 = m[4]; // x4
    uint64_t c5 = m[5]; // x5
    uint64_t c6 = m[6]; // x6
    uint64_t c7 = m[7]; // x7
    uint64_t c8 = m[8]; // x8
    uint64_t c9 = m[9]; // x9
    uint64_t c10 = m[10]; // x10
    uint64_t c11 = m[11]; // x11
    uint64_t c12 = m[12]; // x12
    uint64_t c13 = m[13]; // x13
    uint64_t c14 = m[14]; // x14
    uint64_t c15 = m[15]; // x15
    uint64_t c16 = m[16]; // x16
    uint64_t c17 = m[17]; // x17
    uint64_t c18 = m[18]; // x18
    uint64_t c19 = m[19]; // x19
    uint64_t lsb1 = uint64_t_lsb(c1); // initially all eqs are candidates
    if(unlikely(!lsb1)) {
        return -1; // singular
    }

    uint64_t c1_reduc = c1 ^ lsb1; // for row reduction
    if(const_col & lsb1) const_col ^= c1_reduc;
    if(c2 & lsb1) c2 ^= c1_reduc;
    if(c3 & lsb1) c3 ^= c1_reduc;
    if(c4 & lsb1) c4 ^= c1_reduc;
    if(c5 & lsb1) c5 ^= c1_reduc;
    if(c6 & lsb1) c6 ^= c1_reduc;
    if(c7 & lsb1) c7 ^= c1_reduc;
    if(c8 & lsb1) c8 ^= c1_reduc;
    if(c9 & lsb1) c9 ^= c1_reduc;
    if(c10 & lsb1) c10 ^= c1_reduc;
    if(c11 & lsb1) c11 ^= c1_reduc;
    if(c12 & lsb1) c12 ^= c1_reduc;
    if(c13 & lsb1) c13 ^= c1_reduc;
    if(c14 & lsb1) c14 ^= c1_reduc;
    if(c15 & lsb1) c15 ^= c1_reduc;
    if(c16 & lsb1) c16 ^= c1_reduc;
    if(c17 & lsb1) c17 ^= c1_reduc;
    if(c18 & lsb1) c18 ^= c1_reduc;
    if(c19 & lsb1) c19 ^= c1_reduc;
    uint64_t mask = ~lsb1;

    uint64_t lsb2 = uint64_t_lsb(c2 & mask); // x2
    if(unlikely(!lsb2)) {
        return -1; // singular
    }

    uint64_t c2_reduc = c2 ^ lsb2;
    if(const_col & lsb2) const_col ^= c2_reduc;
    if(c3 & lsb2) c3 ^= c2_reduc;
    if(c4 & lsb2) c4 ^= c2_reduc;
    if(c5 & lsb2) c5 ^= c2_reduc;
    if(c6 & lsb2) c6 ^= c2_reduc;
    if(c7 & lsb2) c7 ^= c2_reduc;
    if(c8 & lsb2) c8 ^= c2_reduc;
    if(c9 & lsb2) c9 ^= c2_reduc;
    if(c10 & lsb2) c10 ^= c2_reduc;
    if(c11 & lsb2) c11 ^= c2_reduc;
    if(c12 & lsb2) c12 ^= c2_reduc;
    if(c13 & lsb2) c13 ^= c2_reduc;
    if(c14 & lsb2) c14 ^= c2_reduc;
    if(c15 & lsb2) c15 ^= c2_reduc;
    if(c16 & lsb2) c16 ^= c2_reduc;
    if(c17 & lsb2) c17 ^= c2_reduc;
    if(c18 & lsb2) c18 ^= c2_reduc;
    if(c19 & lsb2) c19 ^= c2_reduc;
    mask ^= lsb2;

    uint64_t lsb3 = uint64_t_lsb(c3 & mask); // x3
    if(unlikely(!lsb3)) {
        return -1; // singular
    }

    uint64_t c3_reduc = c3 ^ lsb3;
    if(const_col & lsb3) const_col ^= c3_reduc;
    if(c4 & lsb3) c4 ^= c3_reduc;
    if(c5 & lsb3) c5 ^= c3_reduc;
    if(c6 & lsb3) c6 ^= c3_reduc;
    if(c7 & lsb3) c7 ^= c3_reduc;
    if(c8 & lsb3) c8 ^= c3_reduc;
    if(c9 & lsb3) c9 ^= c3_reduc;
    if(c10 & lsb3) c10 ^= c3_reduc;
    if(c11 & lsb3) c11 ^= c3_reduc;
    if(c12 & lsb3) c12 ^= c3_reduc;
    if(c13 & lsb3) c13 ^= c3_reduc;
    if(c14 & lsb3) c14 ^= c3_reduc;
    if(c15 & lsb3) c15 ^= c3_reduc;
    if(c16 & lsb3) c16 ^= c3_reduc;
    if(c17 & lsb3) c17 ^= c3_reduc;
    if(c18 & lsb3) c18 ^= c3_reduc;
    if(c19 & lsb3) c19 ^= c3_reduc;
    mask ^= lsb3;

    uint64_t lsb4 = uint64_t_lsb(c4 & mask); // x4
    if(unlikely(!lsb4)) {
        return -1; // singular
    }

    uint64_t c4_reduc = c4 ^ lsb4;
    if(const_col & lsb4) const_col ^= c4_reduc;
    if(c5 & lsb4) c5 ^= c4_reduc;
    if(c6 & lsb4) c6 ^= c4_reduc;
    if(c7 & lsb4) c7 ^= c4_reduc;
    if(c8 & lsb4) c8 ^= c4_reduc;
    if(c9 & lsb4) c9 ^= c4_reduc;
    if(c10 & lsb4) c10 ^= c4_reduc;
    if(c11 & lsb4) c11 ^= c4_reduc;
    if(c12 & lsb4) c12 ^= c4_reduc;
    if(c13 & lsb4) c13 ^= c4_reduc;
    if(c14 & lsb4) c14 ^= c4_reduc;
    if(c15 & lsb4) c15 ^= c4_reduc;
    if(c16 & lsb4) c16 ^= c4_reduc;
    if(c17 & lsb4) c17 ^= c4_reduc;
    if(c18 & lsb4) c18 ^= c4_reduc;
    if(c19 & lsb4) c19 ^= c4_reduc;
    mask ^= lsb4;

    uint64_t lsb5 = uint64_t_lsb(c5 & mask); // x5
    if(unlikely(!lsb5)) {
        return -1; // singular
    }

    uint64_t c5_reduc = c5 ^ lsb5;
    if(const_col & lsb5) const_col ^= c5_reduc;
    if(c6 & lsb5) c6 ^= c5_reduc;
    if(c7 & lsb5) c7 ^= c5_reduc;
    if(c8 & lsb5) c8 ^= c5_reduc;
    if(c9 & lsb5) c9 ^= c5_reduc;
    if(c10 & lsb5) c10 ^= c5_reduc;
    if(c11 & lsb5) c11 ^= c5_reduc;
    if(c12 & lsb5) c12 ^= c5_reduc;
    if(c13 & lsb5) c13 ^= c5_reduc;
    if(c14 & lsb5) c14 ^= c5_reduc;
    if(c15 & lsb5) c15 ^= c5_reduc;
    if(c16 & lsb5) c16 ^= c5_reduc;
    if(c17 & lsb5) c17 ^= c5_reduc;
    if(c18 & lsb5) c18 ^= c5_reduc;
    if(c19 & lsb5) c19 ^= c5_reduc;
    mask ^= lsb5;

    uint64_t lsb6 = uint64_t_lsb(c6 & mask); // x6
    if(unlikely(!lsb6)) {
        return -1; // singular
    }

    uint64_t c6_reduc = c6 ^ lsb6;
    if(const_col & lsb6) const_col ^= c6_reduc;
    if(c7 & lsb6) c7 ^= c6_reduc;
    if(c8 & lsb6) c8 ^= c6_reduc;
    if(c9 & lsb6) c9 ^= c6_reduc;
    if(c10 & lsb6) c10 ^= c6_reduc;
    if(c11 & lsb6) c11 ^= c6_reduc;
    if(c12 & lsb6) c12 ^= c6_reduc;
    if(c13 & lsb6) c13 ^= c6_reduc;
    if(c14 & lsb6) c14 ^= c6_reduc;
    if(c15 & lsb6) c15 ^= c6_reduc;
    if(c16 & lsb6) c16 ^= c6_reduc;
    if(c17 & lsb6) c17 ^= c6_reduc;
    if(c18 & lsb6) c18 ^= c6_reduc;
    if(c19 & lsb6) c19 ^= c6_reduc;
    mask ^= lsb6;

    uint64_t lsb7 = uint64_t_lsb(c7 & mask); // x7
    if(unlikely(!lsb7)) {
        return -1; // singular
    }

    uint64_t c7_reduc = c7 ^ lsb7;
    if(const_col & lsb7) const_col ^= c7_reduc;
    if(c8 & lsb7) c8 ^= c7_reduc;
    if(c9 & lsb7) c9 ^= c7_reduc;
    if(c10 & lsb7) c10 ^= c7_reduc;
    if(c11 & lsb7) c11 ^= c7_reduc;
    if(c12 & lsb7) c12 ^= c7_reduc;
    if(c13 & lsb7) c13 ^= c7_reduc;
    if(c14 & lsb7) c14 ^= c7_reduc;
    if(c15 & lsb7) c15 ^= c7_reduc;
    if(c16 & lsb7) c16 ^= c7_reduc;
    if(c17 & lsb7) c17 ^= c7_reduc;
    if(c18 & lsb7) c18 ^= c7_reduc;
    if(c19 & lsb7) c19 ^= c7_reduc;
    mask ^= lsb7;

    uint64_t lsb8 = uint64_t_lsb(c8 & mask); // x8
    if(unlikely(!lsb8)) {
        return -1; // singular
    }

    uint64_t c8_reduc = c8 ^ lsb8;
    if(const_col & lsb8) const_col ^= c8_reduc;
    if(c9 & lsb8) c9 ^= c8_reduc;
    if(c10 & lsb8) c10 ^= c8_reduc;
    if(c11 & lsb8) c11 ^= c8_reduc;
    if(c12 & lsb8) c12 ^= c8_reduc;
    if(c13 & lsb8) c13 ^= c8_reduc;
    if(c14 & lsb8) c14 ^= c8_reduc;
    if(c15 & lsb8) c15 ^= c8_reduc;
    if(c16 & lsb8) c16 ^= c8_reduc;
    if(c17 & lsb8) c17 ^= c8_reduc;
    if(c18 & lsb8) c18 ^= c8_reduc;
    if(c19 & lsb8) c19 ^= c8_reduc;
    mask ^= lsb8;

    uint64_t lsb9 = uint64_t_lsb(c9 & mask); // x9
    if(unlikely(!lsb9)) {
        return -1; // singular
    }

    uint64_t c9_reduc = c9 ^ lsb9;
    if(const_col & lsb9) const_col ^= c9_reduc;
    if(c10 & lsb9) c10 ^= c9_reduc;
    if(c11 & lsb9) c11 ^= c9_reduc;
    if(c12 & lsb9) c12 ^= c9_reduc;
    if(c13 & lsb9) c13 ^= c9_reduc;
    if(c14 & lsb9) c14 ^= c9_reduc;
    if(c15 & lsb9) c15 ^= c9_reduc;
    if(c16 & lsb9) c16 ^= c9_reduc;
    if(c17 & lsb9) c17 ^= c9_reduc;
    if(c18 & lsb9) c18 ^= c9_reduc;
    if(c19 & lsb9) c19 ^= c9_reduc;
    mask ^= lsb9;

    uint64_t lsb10 = uint64_t_lsb(c10 & mask); // x10
    if(unlikely(!lsb10)) {
        return -1; // singular
    }

    uint64_t c10_reduc = c10 ^ lsb10;
    if(const_col & lsb10) const_col ^= c10_reduc;
    if(c11 & lsb10) c11 ^= c10_reduc;
    if(c12 & lsb10) c12 ^= c10_reduc;
    if(c13 & lsb10) c13 ^= c10_reduc;
    if(c14 & lsb10) c14 ^= c10_reduc;
    if(c15 & lsb10) c15 ^= c10_reduc;
    if(c16 & lsb10) c16 ^= c10_reduc;
    if(c17 & lsb10) c17 ^= c10_reduc;
    if(c18 & lsb10) c18 ^= c10_reduc;
    if(c19 & lsb10) c19 ^= c10_reduc;
    mask ^= lsb10;

    uint64_t lsb11 = uint64_t_lsb(c11 & mask); // x11
    if(unlikely(!lsb11)) {
        return -1; // singular
    }

    uint64_t c11_reduc = c11 ^ lsb11;
    if(const_col & lsb11) const_col ^= c11_reduc;
    if(c12 & lsb11) c12 ^= c11_reduc;
    if(c13 & lsb11) c13 ^= c11_reduc;
    if(c14 & lsb11) c14 ^= c11_reduc;
    if(c15 & lsb11) c15 ^= c11_reduc;
    if(c16 & lsb11) c16 ^= c11_reduc;
    if(c17 & lsb11) c17 ^= c11_reduc;
    if(c18 & lsb11) c18 ^= c11_reduc;
    if(c19 & lsb11) c19 ^= c11_reduc;
    mask ^= lsb11;

    uint64_t lsb12 = uint64_t_lsb(c12 & mask); // x12
    if(unlikely(!lsb12)) {
        return -1; // singular
    }

    uint64_t c12_reduc = c12 ^ lsb12;
    if(const_col & lsb12) const_col ^= c12_reduc;
    if(c13 & lsb12) c13 ^= c12_reduc;
    if(c14 & lsb12) c14 ^= c12_reduc;
    if(c15 & lsb12) c15 ^= c12_reduc;
    if(c16 & lsb12) c16 ^= c12_reduc;
    if(c17 & lsb12) c17 ^= c12_reduc;
    if(c18 & lsb12) c18 ^= c12_reduc;
    if(c19 & lsb12) c19 ^= c12_reduc;
    mask ^= lsb12;

    uint64_t lsb13 = uint64_t_lsb(c13 & mask); // x13
    if(unlikely(!lsb13)) {
        return -1; // singular
    }

    uint64_t c13_reduc = c13 ^ lsb13;
    if(const_col & lsb13) const_col ^= c13_reduc;
    if(c14 & lsb13) c14 ^= c13_reduc;
    if(c15 & lsb13) c15 ^= c13_reduc;
    if(c16 & lsb13) c16 ^= c13_reduc;
    if(c17 & lsb13) c17 ^= c13_reduc;
    if(c18 & lsb13) c18 ^= c13_reduc;
    if(c19 & lsb13) c19 ^= c13_reduc;
    mask ^= lsb13;

    uint64_t lsb14 = uint64_t_lsb(c14 & mask); // x14
    if(unlikely(!lsb14)) {
        return -1; // singular
    }

    uint64_t c14_reduc = c14 ^ lsb14;
    if(const_col & lsb14) const_col ^= c14_reduc;
    if(c15 & lsb14) c15 ^= c14_reduc;
    if(c16 & lsb14) c16 ^= c14_reduc;
    if(c17 & lsb14) c17 ^= c14_reduc;
    if(c18 & lsb14) c18 ^= c14_reduc;
    if(c19 & lsb14) c19 ^= c14_reduc;
    mask ^= lsb14;

    uint64_t lsb15 = uint64_t_lsb(c15 & mask); // x15
    if(unlikely(!lsb15)) {
        return -1; // singular
    }

    uint64_t c15_reduc = c15 ^ lsb15;
    if(const_col & lsb15) const_col ^= c15_reduc;
    if(c16 & lsb15) c16 ^= c15_reduc;
    if(c17 & lsb15) c17 ^= c15_reduc;
    if(c18 & lsb15) c18 ^= c15_reduc;
    if(c19 & lsb15) c19 ^= c15_reduc;
    mask ^= lsb15;

    uint64_t lsb16 = uint64_t_lsb(c16 & mask); // x16
    if(unlikely(!lsb16)) {
        return -1; // singular
    }

    uint64_t c16_reduc = c16 ^ lsb16;
    if(const_col & lsb16) const_col ^= c16_reduc;
    if(c17 & lsb16) c17 ^= c16_reduc;
    if(c18 & lsb16) c18 ^= c16_reduc;
    if(c19 & lsb16) c19 ^= c16_reduc;
    mask ^= lsb16;

    uint64_t lsb17 = uint64_t_lsb(c17 & mask); // x17
    if(unlikely(!lsb17)) {
        return -1; // singular
    }

    uint64_t c17_reduc = c17 ^ lsb17;
    if(const_col & lsb17) const_col ^= c17_reduc;
    if(c18 & lsb17) c18 ^= c17_reduc;
    if(c19 & lsb17) c19 ^= c17_reduc;
    mask ^= lsb17;

    uint64_t lsb18 = uint64_t_lsb(c18 & mask); // x18
    if(unlikely(!lsb18)) {
        return -1; // singular
    }

    uint64_t c18_reduc = c18 ^ lsb18;
    if(const_col & lsb18) const_col ^= c18_reduc;
    if(c19 & lsb18) c19 ^= c18_reduc;
    mask ^= lsb18;

    uint64_t lsb19 = uint64_t_lsb(c19 & mask); // x19
    if(unlikely(!lsb19)) {
        return -1; // singular
    }

    if(const_col & lsb19) const_col ^= c19 ^ lsb19;
    mask ^= lsb19;

    if(likely(mask & const_col)) { // check system consistency
        return mask & const_col; // not solvable
    }

    uint64_t s = 0x0ULL;
    if(const_col & lsb1) s = uint64_t_toggle_at(s, 0);
    if(const_col & lsb2) s = uint64_t_toggle_at(s, 1);
    if(const_col & lsb3) s = uint64_t_toggle_at(s, 2);
    if(const_col & lsb4) s = uint64_t_toggle_at(s, 3);
    if(const_col & lsb5) s = uint64_t_toggle_at(s, 4);
    if(const_col & lsb6) s = uint64_t_toggle_at(s, 5);
    if(const_col & lsb7) s = uint64_t_toggle_at(s, 6);
    if(const_col & lsb8) s = uint64_t_toggle_at(s, 7);
    if(const_col & lsb9) s = uint64_t_toggle_at(s, 8);
    if(const_col & lsb10) s = uint64_t_toggle_at(s, 9);
    if(const_col & lsb11) s = uint64_t_toggle_at(s, 10);
    if(const_col & lsb12) s = uint64_t_toggle_at(s, 11);
    if(const_col & lsb13) s = uint64_t_toggle_at(s, 12);
    if(const_col & lsb14) s = uint64_t_toggle_at(s, 13);
    if(const_col & lsb15) s = uint64_t_toggle_at(s, 14);
    if(const_col & lsb16) s = uint64_t_toggle_at(s, 15);
    if(const_col & lsb17) s = uint64_t_toggle_at(s, 16);
    if(const_col & lsb18) s = uint64_t_toggle_at(s, 17);
    if(const_col & lsb19) s = uint64_t_toggle_at(s, 18);
    *sol = s;
    return 0;
}

int64_t
uint64a_gj_v20_generic(const uint64_t m[21], uint64_t* const restrict sol) {
    uint64_t const_col = m[0];
    uint64_t c1 = m[1]; // x1
    uint64_t c2 = m[2]; // x2
    uint64_t c3 = m[3]; // x3
    uint64_t c4 = m[4]; // x4
    uint64_t c5 = m[5]; // x5
    uint64_t c6 = m[6]; // x6
    uint64_t c7 = m[7]; // x7
    uint64_t c8 = m[8]; // x8
    uint64_t c9 = m[9]; // x9
    uint64_t c10 = m[10]; // x10
    uint64_t c11 = m[11]; // x11
    uint64_t c12 = m[12]; // x12
    uint64_t c13 = m[13]; // x13
    uint64_t c14 = m[14]; // x14
    uint64_t c15 = m[15]; // x15
    uint64_t c16 = m[16]; // x16
    uint64_t c17 = m[17]; // x17
    uint64_t c18 = m[18]; // x18
    uint64_t c19 = m[19]; // x19
    uint64_t c20 = m[20]; // x20
    uint64_t lsb1 = uint64_t_lsb(c1); // initially all eqs are candidates
    if(unlikely(!lsb1)) {
        return -1; // singular
    }

    uint64_t c1_reduc = c1 ^ lsb1; // for row reduction
    if(const_col & lsb1) const_col ^= c1_reduc;
    if(c2 & lsb1) c2 ^= c1_reduc;
    if(c3 & lsb1) c3 ^= c1_reduc;
    if(c4 & lsb1) c4 ^= c1_reduc;
    if(c5 & lsb1) c5 ^= c1_reduc;
    if(c6 & lsb1) c6 ^= c1_reduc;
    if(c7 & lsb1) c7 ^= c1_reduc;
    if(c8 & lsb1) c8 ^= c1_reduc;
    if(c9 & lsb1) c9 ^= c1_reduc;
    if(c10 & lsb1) c10 ^= c1_reduc;
    if(c11 & lsb1) c11 ^= c1_reduc;
    if(c12 & lsb1) c12 ^= c1_reduc;
    if(c13 & lsb1) c13 ^= c1_reduc;
    if(c14 & lsb1) c14 ^= c1_reduc;
    if(c15 & lsb1) c15 ^= c1_reduc;
    if(c16 & lsb1) c16 ^= c1_reduc;
    if(c17 & lsb1) c17 ^= c1_reduc;
    if(c18 & lsb1) c18 ^= c1_reduc;
    if(c19 & lsb1) c19 ^= c1_reduc;
    if(c20 & lsb1) c20 ^= c1_reduc;
    uint64_t mask = ~lsb1;

    uint64_t lsb2 = uint64_t_lsb(c2 & mask); // x2
    if(unlikely(!lsb2)) {
        return -1; // singular
    }

    uint64_t c2_reduc = c2 ^ lsb2;
    if(const_col & lsb2) const_col ^= c2_reduc;
    if(c3 & lsb2) c3 ^= c2_reduc;
    if(c4 & lsb2) c4 ^= c2_reduc;
    if(c5 & lsb2) c5 ^= c2_reduc;
    if(c6 & lsb2) c6 ^= c2_reduc;
    if(c7 & lsb2) c7 ^= c2_reduc;
    if(c8 & lsb2) c8 ^= c2_reduc;
    if(c9 & lsb2) c9 ^= c2_reduc;
    if(c10 & lsb2) c10 ^= c2_reduc;
    if(c11 & lsb2) c11 ^= c2_reduc;
    if(c12 & lsb2) c12 ^= c2_reduc;
    if(c13 & lsb2) c13 ^= c2_reduc;
    if(c14 & lsb2) c14 ^= c2_reduc;
    if(c15 & lsb2) c15 ^= c2_reduc;
    if(c16 & lsb2) c16 ^= c2_reduc;
    if(c17 & lsb2) c17 ^= c2_reduc;
    if(c18 & lsb2) c18 ^= c2_reduc;
    if(c19 & lsb2) c19 ^= c2_reduc;
    if(c20 & lsb2) c20 ^= c2_reduc;
    mask ^= lsb2;

    uint64_t lsb3 = uint64_t_lsb(c3 & mask); // x3
    if(unlikely(!lsb3)) {
        return -1; // singular
    }

    uint64_t c3_reduc = c3 ^ lsb3;
    if(const_col & lsb3) const_col ^= c3_reduc;
    if(c4 & lsb3) c4 ^= c3_reduc;
    if(c5 & lsb3) c5 ^= c3_reduc;
    if(c6 & lsb3) c6 ^= c3_reduc;
    if(c7 & lsb3) c7 ^= c3_reduc;
    if(c8 & lsb3) c8 ^= c3_reduc;
    if(c9 & lsb3) c9 ^= c3_reduc;
    if(c10 & lsb3) c10 ^= c3_reduc;
    if(c11 & lsb3) c11 ^= c3_reduc;
    if(c12 & lsb3) c12 ^= c3_reduc;
    if(c13 & lsb3) c13 ^= c3_reduc;
    if(c14 & lsb3) c14 ^= c3_reduc;
    if(c15 & lsb3) c15 ^= c3_reduc;
    if(c16 & lsb3) c16 ^= c3_reduc;
    if(c17 & lsb3) c17 ^= c3_reduc;
    if(c18 & lsb3) c18 ^= c3_reduc;
    if(c19 & lsb3) c19 ^= c3_reduc;
    if(c20 & lsb3) c20 ^= c3_reduc;
    mask ^= lsb3;

    uint64_t lsb4 = uint64_t_lsb(c4 & mask); // x4
    if(unlikely(!lsb4)) {
        return -1; // singular
    }

    uint64_t c4_reduc = c4 ^ lsb4;
    if(const_col & lsb4) const_col ^= c4_reduc;
    if(c5 & lsb4) c5 ^= c4_reduc;
    if(c6 & lsb4) c6 ^= c4_reduc;
    if(c7 & lsb4) c7 ^= c4_reduc;
    if(c8 & lsb4) c8 ^= c4_reduc;
    if(c9 & lsb4) c9 ^= c4_reduc;
    if(c10 & lsb4) c10 ^= c4_reduc;
    if(c11 & lsb4) c11 ^= c4_reduc;
    if(c12 & lsb4) c12 ^= c4_reduc;
    if(c13 & lsb4) c13 ^= c4_reduc;
    if(c14 & lsb4) c14 ^= c4_reduc;
    if(c15 & lsb4) c15 ^= c4_reduc;
    if(c16 & lsb4) c16 ^= c4_reduc;
    if(c17 & lsb4) c17 ^= c4_reduc;
    if(c18 & lsb4) c18 ^= c4_reduc;
    if(c19 & lsb4) c19 ^= c4_reduc;
    if(c20 & lsb4) c20 ^= c4_reduc;
    mask ^= lsb4;

    uint64_t lsb5 = uint64_t_lsb(c5 & mask); // x5
    if(unlikely(!lsb5)) {
        return -1; // singular
    }

    uint64_t c5_reduc = c5 ^ lsb5;
    if(const_col & lsb5) const_col ^= c5_reduc;
    if(c6 & lsb5) c6 ^= c5_reduc;
    if(c7 & lsb5) c7 ^= c5_reduc;
    if(c8 & lsb5) c8 ^= c5_reduc;
    if(c9 & lsb5) c9 ^= c5_reduc;
    if(c10 & lsb5) c10 ^= c5_reduc;
    if(c11 & lsb5) c11 ^= c5_reduc;
    if(c12 & lsb5) c12 ^= c5_reduc;
    if(c13 & lsb5) c13 ^= c5_reduc;
    if(c14 & lsb5) c14 ^= c5_reduc;
    if(c15 & lsb5) c15 ^= c5_reduc;
    if(c16 & lsb5) c16 ^= c5_reduc;
    if(c17 & lsb5) c17 ^= c5_reduc;
    if(c18 & lsb5) c18 ^= c5_reduc;
    if(c19 & lsb5) c19 ^= c5_reduc;
    if(c20 & lsb5) c20 ^= c5_reduc;
    mask ^= lsb5;

    uint64_t lsb6 = uint64_t_lsb(c6 & mask); // x6
    if(unlikely(!lsb6)) {
        return -1; // singular
    }

    uint64_t c6_reduc = c6 ^ lsb6;
    if(const_col & lsb6) const_col ^= c6_reduc;
    if(c7 & lsb6) c7 ^= c6_reduc;
    if(c8 & lsb6) c8 ^= c6_reduc;
    if(c9 & lsb6) c9 ^= c6_reduc;
    if(c10 & lsb6) c10 ^= c6_reduc;
    if(c11 & lsb6) c11 ^= c6_reduc;
    if(c12 & lsb6) c12 ^= c6_reduc;
    if(c13 & lsb6) c13 ^= c6_reduc;
    if(c14 & lsb6) c14 ^= c6_reduc;
    if(c15 & lsb6) c15 ^= c6_reduc;
    if(c16 & lsb6) c16 ^= c6_reduc;
    if(c17 & lsb6) c17 ^= c6_reduc;
    if(c18 & lsb6) c18 ^= c6_reduc;
    if(c19 & lsb6) c19 ^= c6_reduc;
    if(c20 & lsb6) c20 ^= c6_reduc;
    mask ^= lsb6;

    uint64_t lsb7 = uint64_t_lsb(c7 & mask); // x7
    if(unlikely(!lsb7)) {
        return -1; // singular
    }

    uint64_t c7_reduc = c7 ^ lsb7;
    if(const_col & lsb7) const_col ^= c7_reduc;
    if(c8 & lsb7) c8 ^= c7_reduc;
    if(c9 & lsb7) c9 ^= c7_reduc;
    if(c10 & lsb7) c10 ^= c7_reduc;
    if(c11 & lsb7) c11 ^= c7_reduc;
    if(c12 & lsb7) c12 ^= c7_reduc;
    if(c13 & lsb7) c13 ^= c7_reduc;
    if(c14 & lsb7) c14 ^= c7_reduc;
    if(c15 & lsb7) c15 ^= c7_reduc;
    if(c16 & lsb7) c16 ^= c7_reduc;
    if(c17 & lsb7) c17 ^= c7_reduc;
    if(c18 & lsb7) c18 ^= c7_reduc;
    if(c19 & lsb7) c19 ^= c7_reduc;
    if(c20 & lsb7) c20 ^= c7_reduc;
    mask ^= lsb7;

    uint64_t lsb8 = uint64_t_lsb(c8 & mask); // x8
    if(unlikely(!lsb8)) {
        return -1; // singular
    }

    uint64_t c8_reduc = c8 ^ lsb8;
    if(const_col & lsb8) const_col ^= c8_reduc;
    if(c9 & lsb8) c9 ^= c8_reduc;
    if(c10 & lsb8) c10 ^= c8_reduc;
    if(c11 & lsb8) c11 ^= c8_reduc;
    if(c12 & lsb8) c12 ^= c8_reduc;
    if(c13 & lsb8) c13 ^= c8_reduc;
    if(c14 & lsb8) c14 ^= c8_reduc;
    if(c15 & lsb8) c15 ^= c8_reduc;
    if(c16 & lsb8) c16 ^= c8_reduc;
    if(c17 & lsb8) c17 ^= c8_reduc;
    if(c18 & lsb8) c18 ^= c8_reduc;
    if(c19 & lsb8) c19 ^= c8_reduc;
    if(c20 & lsb8) c20 ^= c8_reduc;
    mask ^= lsb8;

    uint64_t lsb9 = uint64_t_lsb(c9 & mask); // x9
    if(unlikely(!lsb9)) {
        return -1; // singular
    }

    uint64_t c9_reduc = c9 ^ lsb9;
    if(const_col & lsb9) const_col ^= c9_reduc;
    if(c10 & lsb9) c10 ^= c9_reduc;
    if(c11 & lsb9) c11 ^= c9_reduc;
    if(c12 & lsb9) c12 ^= c9_reduc;
    if(c13 & lsb9) c13 ^= c9_reduc;
    if(c14 & lsb9) c14 ^= c9_reduc;
    if(c15 & lsb9) c15 ^= c9_reduc;
    if(c16 & lsb9) c16 ^= c9_reduc;
    if(c17 & lsb9) c17 ^= c9_reduc;
    if(c18 & lsb9) c18 ^= c9_reduc;
    if(c19 & lsb9) c19 ^= c9_reduc;
    if(c20 & lsb9) c20 ^= c9_reduc;
    mask ^= lsb9;

    uint64_t lsb10 = uint64_t_lsb(c10 & mask); // x10
    if(unlikely(!lsb10)) {
        return -1; // singular
    }

    uint64_t c10_reduc = c10 ^ lsb10;
    if(const_col & lsb10) const_col ^= c10_reduc;
    if(c11 & lsb10) c11 ^= c10_reduc;
    if(c12 & lsb10) c12 ^= c10_reduc;
    if(c13 & lsb10) c13 ^= c10_reduc;
    if(c14 & lsb10) c14 ^= c10_reduc;
    if(c15 & lsb10) c15 ^= c10_reduc;
    if(c16 & lsb10) c16 ^= c10_reduc;
    if(c17 & lsb10) c17 ^= c10_reduc;
    if(c18 & lsb10) c18 ^= c10_reduc;
    if(c19 & lsb10) c19 ^= c10_reduc;
    if(c20 & lsb10) c20 ^= c10_reduc;
    mask ^= lsb10;

    uint64_t lsb11 = uint64_t_lsb(c11 & mask); // x11
    if(unlikely(!lsb11)) {
        return -1; // singular
    }

    uint64_t c11_reduc = c11 ^ lsb11;
    if(const_col & lsb11) const_col ^= c11_reduc;
    if(c12 & lsb11) c12 ^= c11_reduc;
    if(c13 & lsb11) c13 ^= c11_reduc;
    if(c14 & lsb11) c14 ^= c11_reduc;
    if(c15 & lsb11) c15 ^= c11_reduc;
    if(c16 & lsb11) c16 ^= c11_reduc;
    if(c17 & lsb11) c17 ^= c11_reduc;
    if(c18 & lsb11) c18 ^= c11_reduc;
    if(c19 & lsb11) c19 ^= c11_reduc;
    if(c20 & lsb11) c20 ^= c11_reduc;
    mask ^= lsb11;

    uint64_t lsb12 = uint64_t_lsb(c12 & mask); // x12
    if(unlikely(!lsb12)) {
        return -1; // singular
    }

    uint64_t c12_reduc = c12 ^ lsb12;
    if(const_col & lsb12) const_col ^= c12_reduc;
    if(c13 & lsb12) c13 ^= c12_reduc;
    if(c14 & lsb12) c14 ^= c12_reduc;
    if(c15 & lsb12) c15 ^= c12_reduc;
    if(c16 & lsb12) c16 ^= c12_reduc;
    if(c17 & lsb12) c17 ^= c12_reduc;
    if(c18 & lsb12) c18 ^= c12_reduc;
    if(c19 & lsb12) c19 ^= c12_reduc;
    if(c20 & lsb12) c20 ^= c12_reduc;
    mask ^= lsb12;

    uint64_t lsb13 = uint64_t_lsb(c13 & mask); // x13
    if(unlikely(!lsb13)) {
        return -1; // singular
    }

    uint64_t c13_reduc = c13 ^ lsb13;
    if(const_col & lsb13) const_col ^= c13_reduc;
    if(c14 & lsb13) c14 ^= c13_reduc;
    if(c15 & lsb13) c15 ^= c13_reduc;
    if(c16 & lsb13) c16 ^= c13_reduc;
    if(c17 & lsb13) c17 ^= c13_reduc;
    if(c18 & lsb13) c18 ^= c13_reduc;
    if(c19 & lsb13) c19 ^= c13_reduc;
    if(c20 & lsb13) c20 ^= c13_reduc;
    mask ^= lsb13;

    uint64_t lsb14 = uint64_t_lsb(c14 & mask); // x14
    if(unlikely(!lsb14)) {
        return -1; // singular
    }

    uint64_t c14_reduc = c14 ^ lsb14;
    if(const_col & lsb14) const_col ^= c14_reduc;
    if(c15 & lsb14) c15 ^= c14_reduc;
    if(c16 & lsb14) c16 ^= c14_reduc;
    if(c17 & lsb14) c17 ^= c14_reduc;
    if(c18 & lsb14) c18 ^= c14_reduc;
    if(c19 & lsb14) c19 ^= c14_reduc;
    if(c20 & lsb14) c20 ^= c14_reduc;
    mask ^= lsb14;

    uint64_t lsb15 = uint64_t_lsb(c15 & mask); // x15
    if(unlikely(!lsb15)) {
        return -1; // singular
    }

    uint64_t c15_reduc = c15 ^ lsb15;
    if(const_col & lsb15) const_col ^= c15_reduc;
    if(c16 & lsb15) c16 ^= c15_reduc;
    if(c17 & lsb15) c17 ^= c15_reduc;
    if(c18 & lsb15) c18 ^= c15_reduc;
    if(c19 & lsb15) c19 ^= c15_reduc;
    if(c20 & lsb15) c20 ^= c15_reduc;
    mask ^= lsb15;

    uint64_t lsb16 = uint64_t_lsb(c16 & mask); // x16
    if(unlikely(!lsb16)) {
        return -1; // singular
    }

    uint64_t c16_reduc = c16 ^ lsb16;
    if(const_col & lsb16) const_col ^= c16_reduc;
    if(c17 & lsb16) c17 ^= c16_reduc;
    if(c18 & lsb16) c18 ^= c16_reduc;
    if(c19 & lsb16) c19 ^= c16_reduc;
    if(c20 & lsb16) c20 ^= c16_reduc;
    mask ^= lsb16;

    uint64_t lsb17 = uint64_t_lsb(c17 & mask); // x17
    if(unlikely(!lsb17)) {
        return -1; // singular
    }

    uint64_t c17_reduc = c17 ^ lsb17;
    if(const_col & lsb17) const_col ^= c17_reduc;
    if(c18 & lsb17) c18 ^= c17_reduc;
    if(c19 & lsb17) c19 ^= c17_reduc;
    if(c20 & lsb17) c20 ^= c17_reduc;
    mask ^= lsb17;

    uint64_t lsb18 = uint64_t_lsb(c18 & mask); // x18
    if(unlikely(!lsb18)) {
        return -1; // singular
    }

    uint64_t c18_reduc = c18 ^ lsb18;
    if(const_col & lsb18) const_col ^= c18_reduc;
    if(c19 & lsb18) c19 ^= c18_reduc;
    if(c20 & lsb18) c20 ^= c18_reduc;
    mask ^= lsb18;

    uint64_t lsb19 = uint64_t_lsb(c19 & mask); // x19
    if(unlikely(!lsb19)) {
        return -1; // singular
    }

    uint64_t c19_reduc = c19 ^ lsb19;
    if(const_col & lsb19) const_col ^= c19_reduc;
    if(c20 & lsb19) c20 ^= c19_reduc;
    mask ^= lsb19;

    uint64_t lsb20 = uint64_t_lsb(c20 & mask); // x20
    if(unlikely(!lsb20)) {
        return -1; // singular
    }

    if(const_col & lsb20) const_col ^= c20 ^ lsb20;
    mask ^= lsb20;

    if(likely(mask & const_col)) { // check system consistency
        return mask & const_col; // not solvable
    }

    uint64_t s = 0x0ULL;
    if(const_col & lsb1) s = uint64_t_toggle_at(s, 0);
    if(const_col & lsb2) s = uint64_t_toggle_at(s, 1);
    if(const_col & lsb3) s = uint64_t_toggle_at(s, 2);
    if(const_col & lsb4) s = uint64_t_toggle_at(s, 3);
    if(const_col & lsb5) s = uint64_t_toggle_at(s, 4);
    if(const_col & lsb6) s = uint64_t_toggle_at(s, 5);
    if(const_col & lsb7) s = uint64_t_toggle_at(s, 6);
    if(const_col & lsb8) s = uint64_t_toggle_at(s, 7);
    if(const_col & lsb9) s = uint64_t_toggle_at(s, 8);
    if(const_col & lsb10) s = uint64_t_toggle_at(s, 9);
    if(const_col & lsb11) s = uint64_t_toggle_at(s, 10);
    if(const_col & lsb12) s = uint64_t_toggle_at(s, 11);
    if(const_col & lsb13) s = uint64_t_toggle_at(s, 12);
    if(const_col & lsb14) s = uint64_t_toggle_at(s, 13);
    if(const_col & lsb15) s = uint64_t_toggle_at(s, 14);
    if(const_col & lsb16) s = uint64_t_toggle_at(s, 15);
    if(const_col & lsb17) s = uint64_t_toggle_at(s, 16);
    if(const_col & lsb18) s = uint64_t_toggle_at(s, 17);
    if(const_col & lsb19) s = uint64_t_toggle_at(s, 18);
    if(const_col & lsb20) s = uint64_t_toggle_at(s, 19);
    *sol = s;
    return 0;
}

int64_t
uint64a_gj_v21_generic(const uint64_t m[22], uint64_t* const restrict sol) {
    uint64_t const_col = m[0];
    uint64_t c1 = m[1]; // x1
    uint64_t c2 = m[2]; // x2
    uint64_t c3 = m[3]; // x3
    uint64_t c4 = m[4]; // x4
    uint64_t c5 = m[5]; // x5
    uint64_t c6 = m[6]; // x6
    uint64_t c7 = m[7]; // x7
    uint64_t c8 = m[8]; // x8
    uint64_t c9 = m[9]; // x9
    uint64_t c10 = m[10]; // x10
    uint64_t c11 = m[11]; // x11
    uint64_t c12 = m[12]; // x12
    uint64_t c13 = m[13]; // x13
    uint64_t c14 = m[14]; // x14
    uint64_t c15 = m[15]; // x15
    uint64_t c16 = m[16]; // x16
    uint64_t c17 = m[17]; // x17
    uint64_t c18 = m[18]; // x18
    uint64_t c19 = m[19]; // x19
    uint64_t c20 = m[20]; // x20
    uint64_t c21 = m[21]; // x21
    uint64_t lsb1 = uint64_t_lsb(c1); // initially all eqs are candidates
    if(unlikely(!lsb1)) {
        return -1; // singular
    }

    uint64_t c1_reduc = c1 ^ lsb1; // for row reduction
    if(const_col & lsb1) const_col ^= c1_reduc;
    if(c2 & lsb1) c2 ^= c1_reduc;
    if(c3 & lsb1) c3 ^= c1_reduc;
    if(c4 & lsb1) c4 ^= c1_reduc;
    if(c5 & lsb1) c5 ^= c1_reduc;
    if(c6 & lsb1) c6 ^= c1_reduc;
    if(c7 & lsb1) c7 ^= c1_reduc;
    if(c8 & lsb1) c8 ^= c1_reduc;
    if(c9 & lsb1) c9 ^= c1_reduc;
    if(c10 & lsb1) c10 ^= c1_reduc;
    if(c11 & lsb1) c11 ^= c1_reduc;
    if(c12 & lsb1) c12 ^= c1_reduc;
    if(c13 & lsb1) c13 ^= c1_reduc;
    if(c14 & lsb1) c14 ^= c1_reduc;
    if(c15 & lsb1) c15 ^= c1_reduc;
    if(c16 & lsb1) c16 ^= c1_reduc;
    if(c17 & lsb1) c17 ^= c1_reduc;
    if(c18 & lsb1) c18 ^= c1_reduc;
    if(c19 & lsb1) c19 ^= c1_reduc;
    if(c20 & lsb1) c20 ^= c1_reduc;
    if(c21 & lsb1) c21 ^= c1_reduc;
    uint64_t mask = ~lsb1;

    uint64_t lsb2 = uint64_t_lsb(c2 & mask); // x2
    if(unlikely(!lsb2)) {
        return -1; // singular
    }

    uint64_t c2_reduc = c2 ^ lsb2;
    if(const_col & lsb2) const_col ^= c2_reduc;
    if(c3 & lsb2) c3 ^= c2_reduc;
    if(c4 & lsb2) c4 ^= c2_reduc;
    if(c5 & lsb2) c5 ^= c2_reduc;
    if(c6 & lsb2) c6 ^= c2_reduc;
    if(c7 & lsb2) c7 ^= c2_reduc;
    if(c8 & lsb2) c8 ^= c2_reduc;
    if(c9 & lsb2) c9 ^= c2_reduc;
    if(c10 & lsb2) c10 ^= c2_reduc;
    if(c11 & lsb2) c11 ^= c2_reduc;
    if(c12 & lsb2) c12 ^= c2_reduc;
    if(c13 & lsb2) c13 ^= c2_reduc;
    if(c14 & lsb2) c14 ^= c2_reduc;
    if(c15 & lsb2) c15 ^= c2_reduc;
    if(c16 & lsb2) c16 ^= c2_reduc;
    if(c17 & lsb2) c17 ^= c2_reduc;
    if(c18 & lsb2) c18 ^= c2_reduc;
    if(c19 & lsb2) c19 ^= c2_reduc;
    if(c20 & lsb2) c20 ^= c2_reduc;
    if(c21 & lsb2) c21 ^= c2_reduc;
    mask ^= lsb2;

    uint64_t lsb3 = uint64_t_lsb(c3 & mask); // x3
    if(unlikely(!lsb3)) {
        return -1; // singular
    }

    uint64_t c3_reduc = c3 ^ lsb3;
    if(const_col & lsb3) const_col ^= c3_reduc;
    if(c4 & lsb3) c4 ^= c3_reduc;
    if(c5 & lsb3) c5 ^= c3_reduc;
    if(c6 & lsb3) c6 ^= c3_reduc;
    if(c7 & lsb3) c7 ^= c3_reduc;
    if(c8 & lsb3) c8 ^= c3_reduc;
    if(c9 & lsb3) c9 ^= c3_reduc;
    if(c10 & lsb3) c10 ^= c3_reduc;
    if(c11 & lsb3) c11 ^= c3_reduc;
    if(c12 & lsb3) c12 ^= c3_reduc;
    if(c13 & lsb3) c13 ^= c3_reduc;
    if(c14 & lsb3) c14 ^= c3_reduc;
    if(c15 & lsb3) c15 ^= c3_reduc;
    if(c16 & lsb3) c16 ^= c3_reduc;
    if(c17 & lsb3) c17 ^= c3_reduc;
    if(c18 & lsb3) c18 ^= c3_reduc;
    if(c19 & lsb3) c19 ^= c3_reduc;
    if(c20 & lsb3) c20 ^= c3_reduc;
    if(c21 & lsb3) c21 ^= c3_reduc;
    mask ^= lsb3;

    uint64_t lsb4 = uint64_t_lsb(c4 & mask); // x4
    if(unlikely(!lsb4)) {
        return -1; // singular
    }

    uint64_t c4_reduc = c4 ^ lsb4;
    if(const_col & lsb4) const_col ^= c4_reduc;
    if(c5 & lsb4) c5 ^= c4_reduc;
    if(c6 & lsb4) c6 ^= c4_reduc;
    if(c7 & lsb4) c7 ^= c4_reduc;
    if(c8 & lsb4) c8 ^= c4_reduc;
    if(c9 & lsb4) c9 ^= c4_reduc;
    if(c10 & lsb4) c10 ^= c4_reduc;
    if(c11 & lsb4) c11 ^= c4_reduc;
    if(c12 & lsb4) c12 ^= c4_reduc;
    if(c13 & lsb4) c13 ^= c4_reduc;
    if(c14 & lsb4) c14 ^= c4_reduc;
    if(c15 & lsb4) c15 ^= c4_reduc;
    if(c16 & lsb4) c16 ^= c4_reduc;
    if(c17 & lsb4) c17 ^= c4_reduc;
    if(c18 & lsb4) c18 ^= c4_reduc;
    if(c19 & lsb4) c19 ^= c4_reduc;
    if(c20 & lsb4) c20 ^= c4_reduc;
    if(c21 & lsb4) c21 ^= c4_reduc;
    mask ^= lsb4;

    uint64_t lsb5 = uint64_t_lsb(c5 & mask); // x5
    if(unlikely(!lsb5)) {
        return -1; // singular
    }

    uint64_t c5_reduc = c5 ^ lsb5;
    if(const_col & lsb5) const_col ^= c5_reduc;
    if(c6 & lsb5) c6 ^= c5_reduc;
    if(c7 & lsb5) c7 ^= c5_reduc;
    if(c8 & lsb5) c8 ^= c5_reduc;
    if(c9 & lsb5) c9 ^= c5_reduc;
    if(c10 & lsb5) c10 ^= c5_reduc;
    if(c11 & lsb5) c11 ^= c5_reduc;
    if(c12 & lsb5) c12 ^= c5_reduc;
    if(c13 & lsb5) c13 ^= c5_reduc;
    if(c14 & lsb5) c14 ^= c5_reduc;
    if(c15 & lsb5) c15 ^= c5_reduc;
    if(c16 & lsb5) c16 ^= c5_reduc;
    if(c17 & lsb5) c17 ^= c5_reduc;
    if(c18 & lsb5) c18 ^= c5_reduc;
    if(c19 & lsb5) c19 ^= c5_reduc;
    if(c20 & lsb5) c20 ^= c5_reduc;
    if(c21 & lsb5) c21 ^= c5_reduc;
    mask ^= lsb5;

    uint64_t lsb6 = uint64_t_lsb(c6 & mask); // x6
    if(unlikely(!lsb6)) {
        return -1; // singular
    }

    uint64_t c6_reduc = c6 ^ lsb6;
    if(const_col & lsb6) const_col ^= c6_reduc;
    if(c7 & lsb6) c7 ^= c6_reduc;
    if(c8 & lsb6) c8 ^= c6_reduc;
    if(c9 & lsb6) c9 ^= c6_reduc;
    if(c10 & lsb6) c10 ^= c6_reduc;
    if(c11 & lsb6) c11 ^= c6_reduc;
    if(c12 & lsb6) c12 ^= c6_reduc;
    if(c13 & lsb6) c13 ^= c6_reduc;
    if(c14 & lsb6) c14 ^= c6_reduc;
    if(c15 & lsb6) c15 ^= c6_reduc;
    if(c16 & lsb6) c16 ^= c6_reduc;
    if(c17 & lsb6) c17 ^= c6_reduc;
    if(c18 & lsb6) c18 ^= c6_reduc;
    if(c19 & lsb6) c19 ^= c6_reduc;
    if(c20 & lsb6) c20 ^= c6_reduc;
    if(c21 & lsb6) c21 ^= c6_reduc;
    mask ^= lsb6;

    uint64_t lsb7 = uint64_t_lsb(c7 & mask); // x7
    if(unlikely(!lsb7)) {
        return -1; // singular
    }

    uint64_t c7_reduc = c7 ^ lsb7;
    if(const_col & lsb7) const_col ^= c7_reduc;
    if(c8 & lsb7) c8 ^= c7_reduc;
    if(c9 & lsb7) c9 ^= c7_reduc;
    if(c10 & lsb7) c10 ^= c7_reduc;
    if(c11 & lsb7) c11 ^= c7_reduc;
    if(c12 & lsb7) c12 ^= c7_reduc;
    if(c13 & lsb7) c13 ^= c7_reduc;
    if(c14 & lsb7) c14 ^= c7_reduc;
    if(c15 & lsb7) c15 ^= c7_reduc;
    if(c16 & lsb7) c16 ^= c7_reduc;
    if(c17 & lsb7) c17 ^= c7_reduc;
    if(c18 & lsb7) c18 ^= c7_reduc;
    if(c19 & lsb7) c19 ^= c7_reduc;
    if(c20 & lsb7) c20 ^= c7_reduc;
    if(c21 & lsb7) c21 ^= c7_reduc;
    mask ^= lsb7;

    uint64_t lsb8 = uint64_t_lsb(c8 & mask); // x8
    if(unlikely(!lsb8)) {
        return -1; // singular
    }

    uint64_t c8_reduc = c8 ^ lsb8;
    if(const_col & lsb8) const_col ^= c8_reduc;
    if(c9 & lsb8) c9 ^= c8_reduc;
    if(c10 & lsb8) c10 ^= c8_reduc;
    if(c11 & lsb8) c11 ^= c8_reduc;
    if(c12 & lsb8) c12 ^= c8_reduc;
    if(c13 & lsb8) c13 ^= c8_reduc;
    if(c14 & lsb8) c14 ^= c8_reduc;
    if(c15 & lsb8) c15 ^= c8_reduc;
    if(c16 & lsb8) c16 ^= c8_reduc;
    if(c17 & lsb8) c17 ^= c8_reduc;
    if(c18 & lsb8) c18 ^= c8_reduc;
    if(c19 & lsb8) c19 ^= c8_reduc;
    if(c20 & lsb8) c20 ^= c8_reduc;
    if(c21 & lsb8) c21 ^= c8_reduc;
    mask ^= lsb8;

    uint64_t lsb9 = uint64_t_lsb(c9 & mask); // x9
    if(unlikely(!lsb9)) {
        return -1; // singular
    }

    uint64_t c9_reduc = c9 ^ lsb9;
    if(const_col & lsb9) const_col ^= c9_reduc;
    if(c10 & lsb9) c10 ^= c9_reduc;
    if(c11 & lsb9) c11 ^= c9_reduc;
    if(c12 & lsb9) c12 ^= c9_reduc;
    if(c13 & lsb9) c13 ^= c9_reduc;
    if(c14 & lsb9) c14 ^= c9_reduc;
    if(c15 & lsb9) c15 ^= c9_reduc;
    if(c16 & lsb9) c16 ^= c9_reduc;
    if(c17 & lsb9) c17 ^= c9_reduc;
    if(c18 & lsb9) c18 ^= c9_reduc;
    if(c19 & lsb9) c19 ^= c9_reduc;
    if(c20 & lsb9) c20 ^= c9_reduc;
    if(c21 & lsb9) c21 ^= c9_reduc;
    mask ^= lsb9;

    uint64_t lsb10 = uint64_t_lsb(c10 & mask); // x10
    if(unlikely(!lsb10)) {
        return -1; // singular
    }

    uint64_t c10_reduc = c10 ^ lsb10;
    if(const_col & lsb10) const_col ^= c10_reduc;
    if(c11 & lsb10) c11 ^= c10_reduc;
    if(c12 & lsb10) c12 ^= c10_reduc;
    if(c13 & lsb10) c13 ^= c10_reduc;
    if(c14 & lsb10) c14 ^= c10_reduc;
    if(c15 & lsb10) c15 ^= c10_reduc;
    if(c16 & lsb10) c16 ^= c10_reduc;
    if(c17 & lsb10) c17 ^= c10_reduc;
    if(c18 & lsb10) c18 ^= c10_reduc;
    if(c19 & lsb10) c19 ^= c10_reduc;
    if(c20 & lsb10) c20 ^= c10_reduc;
    if(c21 & lsb10) c21 ^= c10_reduc;
    mask ^= lsb10;

    uint64_t lsb11 = uint64_t_lsb(c11 & mask); // x11
    if(unlikely(!lsb11)) {
        return -1; // singular
    }

    uint64_t c11_reduc = c11 ^ lsb11;
    if(const_col & lsb11) const_col ^= c11_reduc;
    if(c12 & lsb11) c12 ^= c11_reduc;
    if(c13 & lsb11) c13 ^= c11_reduc;
    if(c14 & lsb11) c14 ^= c11_reduc;
    if(c15 & lsb11) c15 ^= c11_reduc;
    if(c16 & lsb11) c16 ^= c11_reduc;
    if(c17 & lsb11) c17 ^= c11_reduc;
    if(c18 & lsb11) c18 ^= c11_reduc;
    if(c19 & lsb11) c19 ^= c11_reduc;
    if(c20 & lsb11) c20 ^= c11_reduc;
    if(c21 & lsb11) c21 ^= c11_reduc;
    mask ^= lsb11;

    uint64_t lsb12 = uint64_t_lsb(c12 & mask); // x12
    if(unlikely(!lsb12)) {
        return -1; // singular
    }

    uint64_t c12_reduc = c12 ^ lsb12;
    if(const_col & lsb12) const_col ^= c12_reduc;
    if(c13 & lsb12) c13 ^= c12_reduc;
    if(c14 & lsb12) c14 ^= c12_reduc;
    if(c15 & lsb12) c15 ^= c12_reduc;
    if(c16 & lsb12) c16 ^= c12_reduc;
    if(c17 & lsb12) c17 ^= c12_reduc;
    if(c18 & lsb12) c18 ^= c12_reduc;
    if(c19 & lsb12) c19 ^= c12_reduc;
    if(c20 & lsb12) c20 ^= c12_reduc;
    if(c21 & lsb12) c21 ^= c12_reduc;
    mask ^= lsb12;

    uint64_t lsb13 = uint64_t_lsb(c13 & mask); // x13
    if(unlikely(!lsb13)) {
        return -1; // singular
    }

    uint64_t c13_reduc = c13 ^ lsb13;
    if(const_col & lsb13) const_col ^= c13_reduc;
    if(c14 & lsb13) c14 ^= c13_reduc;
    if(c15 & lsb13) c15 ^= c13_reduc;
    if(c16 & lsb13) c16 ^= c13_reduc;
    if(c17 & lsb13) c17 ^= c13_reduc;
    if(c18 & lsb13) c18 ^= c13_reduc;
    if(c19 & lsb13) c19 ^= c13_reduc;
    if(c20 & lsb13) c20 ^= c13_reduc;
    if(c21 & lsb13) c21 ^= c13_reduc;
    mask ^= lsb13;

    uint64_t lsb14 = uint64_t_lsb(c14 & mask); // x14
    if(unlikely(!lsb14)) {
        return -1; // singular
    }

    uint64_t c14_reduc = c14 ^ lsb14;
    if(const_col & lsb14) const_col ^= c14_reduc;
    if(c15 & lsb14) c15 ^= c14_reduc;
    if(c16 & lsb14) c16 ^= c14_reduc;
    if(c17 & lsb14) c17 ^= c14_reduc;
    if(c18 & lsb14) c18 ^= c14_reduc;
    if(c19 & lsb14) c19 ^= c14_reduc;
    if(c20 & lsb14) c20 ^= c14_reduc;
    if(c21 & lsb14) c21 ^= c14_reduc;
    mask ^= lsb14;

    uint64_t lsb15 = uint64_t_lsb(c15 & mask); // x15
    if(unlikely(!lsb15)) {
        return -1; // singular
    }

    uint64_t c15_reduc = c15 ^ lsb15;
    if(const_col & lsb15) const_col ^= c15_reduc;
    if(c16 & lsb15) c16 ^= c15_reduc;
    if(c17 & lsb15) c17 ^= c15_reduc;
    if(c18 & lsb15) c18 ^= c15_reduc;
    if(c19 & lsb15) c19 ^= c15_reduc;
    if(c20 & lsb15) c20 ^= c15_reduc;
    if(c21 & lsb15) c21 ^= c15_reduc;
    mask ^= lsb15;

    uint64_t lsb16 = uint64_t_lsb(c16 & mask); // x16
    if(unlikely(!lsb16)) {
        return -1; // singular
    }

    uint64_t c16_reduc = c16 ^ lsb16;
    if(const_col & lsb16) const_col ^= c16_reduc;
    if(c17 & lsb16) c17 ^= c16_reduc;
    if(c18 & lsb16) c18 ^= c16_reduc;
    if(c19 & lsb16) c19 ^= c16_reduc;
    if(c20 & lsb16) c20 ^= c16_reduc;
    if(c21 & lsb16) c21 ^= c16_reduc;
    mask ^= lsb16;

    uint64_t lsb17 = uint64_t_lsb(c17 & mask); // x17
    if(unlikely(!lsb17)) {
        return -1; // singular
    }

    uint64_t c17_reduc = c17 ^ lsb17;
    if(const_col & lsb17) const_col ^= c17_reduc;
    if(c18 & lsb17) c18 ^= c17_reduc;
    if(c19 & lsb17) c19 ^= c17_reduc;
    if(c20 & lsb17) c20 ^= c17_reduc;
    if(c21 & lsb17) c21 ^= c17_reduc;
    mask ^= lsb17;

    uint64_t lsb18 = uint64_t_lsb(c18 & mask); // x18
    if(unlikely(!lsb18)) {
        return -1; // singular
    }

    uint64_t c18_reduc = c18 ^ lsb18;
    if(const_col & lsb18) const_col ^= c18_reduc;
    if(c19 & lsb18) c19 ^= c18_reduc;
    if(c20 & lsb18) c20 ^= c18_reduc;
    if(c21 & lsb18) c21 ^= c18_reduc;
    mask ^= lsb18;

    uint64_t lsb19 = uint64_t_lsb(c19 & mask); // x19
    if(unlikely(!lsb19)) {
        return -1; // singular
    }

    uint64_t c19_reduc = c19 ^ lsb19;
    if(const_col & lsb19) const_col ^= c19_reduc;
    if(c20 & lsb19) c20 ^= c19_reduc;
    if(c21 & lsb19) c21 ^= c19_reduc;
    mask ^= lsb19;

    uint64_t lsb20 = uint64_t_lsb(c20 & mask); // x20
    if(unlikely(!lsb20)) {
        return -1; // singular
    }

    uint64_t c20_reduc = c20 ^ lsb20;
    if(const_col & lsb20) const_col ^= c20_reduc;
    if(c21 & lsb20) c21 ^= c20_reduc;
    mask ^= lsb20;

    uint64_t lsb21 = uint64_t_lsb(c21 & mask); // x21
    if(unlikely(!lsb21)) {
        return -1; // singular
    }

    if(const_col & lsb21) const_col ^= c21 ^ lsb21;
    mask ^= lsb21;

    if(likely(mask & const_col)) { // check system consistency
        return mask & const_col; // not solvable
    }

    uint64_t s = 0x0ULL;
    if(const_col & lsb1) s = uint64_t_toggle_at(s, 0);
    if(const_col & lsb2) s = uint64_t_toggle_at(s, 1);
    if(const_col & lsb3) s = uint64_t_toggle_at(s, 2);
    if(const_col & lsb4) s = uint64_t_toggle_at(s, 3);
    if(const_col & lsb5) s = uint64_t_toggle_at(s, 4);
    if(const_col & lsb6) s = uint64_t_toggle_at(s, 5);
    if(const_col & lsb7) s = uint64_t_toggle_at(s, 6);
    if(const_col & lsb8) s = uint64_t_toggle_at(s, 7);
    if(const_col & lsb9) s = uint64_t_toggle_at(s, 8);
    if(const_col & lsb10) s = uint64_t_toggle_at(s, 9);
    if(const_col & lsb11) s = uint64_t_toggle_at(s, 10);
    if(const_col & lsb12) s = uint64_t_toggle_at(s, 11);
    if(const_col & lsb13) s = uint64_t_toggle_at(s, 12);
    if(const_col & lsb14) s = uint64_t_toggle_at(s, 13);
    if(const_col & lsb15) s = uint64_t_toggle_at(s, 14);
    if(const_col & lsb16) s = uint64_t_toggle_at(s, 15);
    if(const_col & lsb17) s = uint64_t_toggle_at(s, 16);
    if(const_col & lsb18) s = uint64_t_toggle_at(s, 17);
    if(const_col & lsb19) s = uint64_t_toggle_at(s, 18);
    if(const_col & lsb20) s = uint64_t_toggle_at(s, 19);
    if(const_col & lsb21) s = uint64_t_toggle_at(s, 20);
    *sol = s;
    return 0;
}

int64_t
uint64a_gj_v22_generic(const uint64_t m[23], uint64_t* const restrict sol) {
    uint64_t const_col = m[0];
    uint64_t c1 = m[1]; // x1
    uint64_t c2 = m[2]; // x2
    uint64_t c3 = m[3]; // x3
    uint64_t c4 = m[4]; // x4
    uint64_t c5 = m[5]; // x5
    uint64_t c6 = m[6]; // x6
    uint64_t c7 = m[7]; // x7
    uint64_t c8 = m[8]; // x8
    uint64_t c9 = m[9]; // x9
    uint64_t c10 = m[10]; // x10
    uint64_t c11 = m[11]; // x11
    uint64_t c12 = m[12]; // x12
    uint64_t c13 = m[13]; // x13
    uint64_t c14 = m[14]; // x14
    uint64_t c15 = m[15]; // x15
    uint64_t c16 = m[16]; // x16
    uint64_t c17 = m[17]; // x17
    uint64_t c18 = m[18]; // x18
    uint64_t c19 = m[19]; // x19
    uint64_t c20 = m[20]; // x20
    uint64_t c21 = m[21]; // x21
    uint64_t c22 = m[22]; // x22
    uint64_t lsb1 = uint64_t_lsb(c1); // initially all eqs are candidates
    if(unlikely(!lsb1)) {
        return -1; // singular
    }

    uint64_t c1_reduc = c1 ^ lsb1; // for row reduction
    if(const_col & lsb1) const_col ^= c1_reduc;
    if(c2 & lsb1) c2 ^= c1_reduc;
    if(c3 & lsb1) c3 ^= c1_reduc;
    if(c4 & lsb1) c4 ^= c1_reduc;
    if(c5 & lsb1) c5 ^= c1_reduc;
    if(c6 & lsb1) c6 ^= c1_reduc;
    if(c7 & lsb1) c7 ^= c1_reduc;
    if(c8 & lsb1) c8 ^= c1_reduc;
    if(c9 & lsb1) c9 ^= c1_reduc;
    if(c10 & lsb1) c10 ^= c1_reduc;
    if(c11 & lsb1) c11 ^= c1_reduc;
    if(c12 & lsb1) c12 ^= c1_reduc;
    if(c13 & lsb1) c13 ^= c1_reduc;
    if(c14 & lsb1) c14 ^= c1_reduc;
    if(c15 & lsb1) c15 ^= c1_reduc;
    if(c16 & lsb1) c16 ^= c1_reduc;
    if(c17 & lsb1) c17 ^= c1_reduc;
    if(c18 & lsb1) c18 ^= c1_reduc;
    if(c19 & lsb1) c19 ^= c1_reduc;
    if(c20 & lsb1) c20 ^= c1_reduc;
    if(c21 & lsb1) c21 ^= c1_reduc;
    if(c22 & lsb1) c22 ^= c1_reduc;
    uint64_t mask = ~lsb1;

    uint64_t lsb2 = uint64_t_lsb(c2 & mask); // x2
    if(unlikely(!lsb2)) {
        return -1; // singular
    }

    uint64_t c2_reduc = c2 ^ lsb2;
    if(const_col & lsb2) const_col ^= c2_reduc;
    if(c3 & lsb2) c3 ^= c2_reduc;
    if(c4 & lsb2) c4 ^= c2_reduc;
    if(c5 & lsb2) c5 ^= c2_reduc;
    if(c6 & lsb2) c6 ^= c2_reduc;
    if(c7 & lsb2) c7 ^= c2_reduc;
    if(c8 & lsb2) c8 ^= c2_reduc;
    if(c9 & lsb2) c9 ^= c2_reduc;
    if(c10 & lsb2) c10 ^= c2_reduc;
    if(c11 & lsb2) c11 ^= c2_reduc;
    if(c12 & lsb2) c12 ^= c2_reduc;
    if(c13 & lsb2) c13 ^= c2_reduc;
    if(c14 & lsb2) c14 ^= c2_reduc;
    if(c15 & lsb2) c15 ^= c2_reduc;
    if(c16 & lsb2) c16 ^= c2_reduc;
    if(c17 & lsb2) c17 ^= c2_reduc;
    if(c18 & lsb2) c18 ^= c2_reduc;
    if(c19 & lsb2) c19 ^= c2_reduc;
    if(c20 & lsb2) c20 ^= c2_reduc;
    if(c21 & lsb2) c21 ^= c2_reduc;
    if(c22 & lsb2) c22 ^= c2_reduc;
    mask ^= lsb2;

    uint64_t lsb3 = uint64_t_lsb(c3 & mask); // x3
    if(unlikely(!lsb3)) {
        return -1; // singular
    }

    uint64_t c3_reduc = c3 ^ lsb3;
    if(const_col & lsb3) const_col ^= c3_reduc;
    if(c4 & lsb3) c4 ^= c3_reduc;
    if(c5 & lsb3) c5 ^= c3_reduc;
    if(c6 & lsb3) c6 ^= c3_reduc;
    if(c7 & lsb3) c7 ^= c3_reduc;
    if(c8 & lsb3) c8 ^= c3_reduc;
    if(c9 & lsb3) c9 ^= c3_reduc;
    if(c10 & lsb3) c10 ^= c3_reduc;
    if(c11 & lsb3) c11 ^= c3_reduc;
    if(c12 & lsb3) c12 ^= c3_reduc;
    if(c13 & lsb3) c13 ^= c3_reduc;
    if(c14 & lsb3) c14 ^= c3_reduc;
    if(c15 & lsb3) c15 ^= c3_reduc;
    if(c16 & lsb3) c16 ^= c3_reduc;
    if(c17 & lsb3) c17 ^= c3_reduc;
    if(c18 & lsb3) c18 ^= c3_reduc;
    if(c19 & lsb3) c19 ^= c3_reduc;
    if(c20 & lsb3) c20 ^= c3_reduc;
    if(c21 & lsb3) c21 ^= c3_reduc;
    if(c22 & lsb3) c22 ^= c3_reduc;
    mask ^= lsb3;

    uint64_t lsb4 = uint64_t_lsb(c4 & mask); // x4
    if(unlikely(!lsb4)) {
        return -1; // singular
    }

    uint64_t c4_reduc = c4 ^ lsb4;
    if(const_col & lsb4) const_col ^= c4_reduc;
    if(c5 & lsb4) c5 ^= c4_reduc;
    if(c6 & lsb4) c6 ^= c4_reduc;
    if(c7 & lsb4) c7 ^= c4_reduc;
    if(c8 & lsb4) c8 ^= c4_reduc;
    if(c9 & lsb4) c9 ^= c4_reduc;
    if(c10 & lsb4) c10 ^= c4_reduc;
    if(c11 & lsb4) c11 ^= c4_reduc;
    if(c12 & lsb4) c12 ^= c4_reduc;
    if(c13 & lsb4) c13 ^= c4_reduc;
    if(c14 & lsb4) c14 ^= c4_reduc;
    if(c15 & lsb4) c15 ^= c4_reduc;
    if(c16 & lsb4) c16 ^= c4_reduc;
    if(c17 & lsb4) c17 ^= c4_reduc;
    if(c18 & lsb4) c18 ^= c4_reduc;
    if(c19 & lsb4) c19 ^= c4_reduc;
    if(c20 & lsb4) c20 ^= c4_reduc;
    if(c21 & lsb4) c21 ^= c4_reduc;
    if(c22 & lsb4) c22 ^= c4_reduc;
    mask ^= lsb4;

    uint64_t lsb5 = uint64_t_lsb(c5 & mask); // x5
    if(unlikely(!lsb5)) {
        return -1; // singular
    }

    uint64_t c5_reduc = c5 ^ lsb5;
    if(const_col & lsb5) const_col ^= c5_reduc;
    if(c6 & lsb5) c6 ^= c5_reduc;
    if(c7 & lsb5) c7 ^= c5_reduc;
    if(c8 & lsb5) c8 ^= c5_reduc;
    if(c9 & lsb5) c9 ^= c5_reduc;
    if(c10 & lsb5) c10 ^= c5_reduc;
    if(c11 & lsb5) c11 ^= c5_reduc;
    if(c12 & lsb5) c12 ^= c5_reduc;
    if(c13 & lsb5) c13 ^= c5_reduc;
    if(c14 & lsb5) c14 ^= c5_reduc;
    if(c15 & lsb5) c15 ^= c5_reduc;
    if(c16 & lsb5) c16 ^= c5_reduc;
    if(c17 & lsb5) c17 ^= c5_reduc;
    if(c18 & lsb5) c18 ^= c5_reduc;
    if(c19 & lsb5) c19 ^= c5_reduc;
    if(c20 & lsb5) c20 ^= c5_reduc;
    if(c21 & lsb5) c21 ^= c5_reduc;
    if(c22 & lsb5) c22 ^= c5_reduc;
    mask ^= lsb5;

    uint64_t lsb6 = uint64_t_lsb(c6 & mask); // x6
    if(unlikely(!lsb6)) {
        return -1; // singular
    }

    uint64_t c6_reduc = c6 ^ lsb6;
    if(const_col & lsb6) const_col ^= c6_reduc;
    if(c7 & lsb6) c7 ^= c6_reduc;
    if(c8 & lsb6) c8 ^= c6_reduc;
    if(c9 & lsb6) c9 ^= c6_reduc;
    if(c10 & lsb6) c10 ^= c6_reduc;
    if(c11 & lsb6) c11 ^= c6_reduc;
    if(c12 & lsb6) c12 ^= c6_reduc;
    if(c13 & lsb6) c13 ^= c6_reduc;
    if(c14 & lsb6) c14 ^= c6_reduc;
    if(c15 & lsb6) c15 ^= c6_reduc;
    if(c16 & lsb6) c16 ^= c6_reduc;
    if(c17 & lsb6) c17 ^= c6_reduc;
    if(c18 & lsb6) c18 ^= c6_reduc;
    if(c19 & lsb6) c19 ^= c6_reduc;
    if(c20 & lsb6) c20 ^= c6_reduc;
    if(c21 & lsb6) c21 ^= c6_reduc;
    if(c22 & lsb6) c22 ^= c6_reduc;
    mask ^= lsb6;

    uint64_t lsb7 = uint64_t_lsb(c7 & mask); // x7
    if(unlikely(!lsb7)) {
        return -1; // singular
    }

    uint64_t c7_reduc = c7 ^ lsb7;
    if(const_col & lsb7) const_col ^= c7_reduc;
    if(c8 & lsb7) c8 ^= c7_reduc;
    if(c9 & lsb7) c9 ^= c7_reduc;
    if(c10 & lsb7) c10 ^= c7_reduc;
    if(c11 & lsb7) c11 ^= c7_reduc;
    if(c12 & lsb7) c12 ^= c7_reduc;
    if(c13 & lsb7) c13 ^= c7_reduc;
    if(c14 & lsb7) c14 ^= c7_reduc;
    if(c15 & lsb7) c15 ^= c7_reduc;
    if(c16 & lsb7) c16 ^= c7_reduc;
    if(c17 & lsb7) c17 ^= c7_reduc;
    if(c18 & lsb7) c18 ^= c7_reduc;
    if(c19 & lsb7) c19 ^= c7_reduc;
    if(c20 & lsb7) c20 ^= c7_reduc;
    if(c21 & lsb7) c21 ^= c7_reduc;
    if(c22 & lsb7) c22 ^= c7_reduc;
    mask ^= lsb7;

    uint64_t lsb8 = uint64_t_lsb(c8 & mask); // x8
    if(unlikely(!lsb8)) {
        return -1; // singular
    }

    uint64_t c8_reduc = c8 ^ lsb8;
    if(const_col & lsb8) const_col ^= c8_reduc;
    if(c9 & lsb8) c9 ^= c8_reduc;
    if(c10 & lsb8) c10 ^= c8_reduc;
    if(c11 & lsb8) c11 ^= c8_reduc;
    if(c12 & lsb8) c12 ^= c8_reduc;
    if(c13 & lsb8) c13 ^= c8_reduc;
    if(c14 & lsb8) c14 ^= c8_reduc;
    if(c15 & lsb8) c15 ^= c8_reduc;
    if(c16 & lsb8) c16 ^= c8_reduc;
    if(c17 & lsb8) c17 ^= c8_reduc;
    if(c18 & lsb8) c18 ^= c8_reduc;
    if(c19 & lsb8) c19 ^= c8_reduc;
    if(c20 & lsb8) c20 ^= c8_reduc;
    if(c21 & lsb8) c21 ^= c8_reduc;
    if(c22 & lsb8) c22 ^= c8_reduc;
    mask ^= lsb8;

    uint64_t lsb9 = uint64_t_lsb(c9 & mask); // x9
    if(unlikely(!lsb9)) {
        return -1; // singular
    }

    uint64_t c9_reduc = c9 ^ lsb9;
    if(const_col & lsb9) const_col ^= c9_reduc;
    if(c10 & lsb9) c10 ^= c9_reduc;
    if(c11 & lsb9) c11 ^= c9_reduc;
    if(c12 & lsb9) c12 ^= c9_reduc;
    if(c13 & lsb9) c13 ^= c9_reduc;
    if(c14 & lsb9) c14 ^= c9_reduc;
    if(c15 & lsb9) c15 ^= c9_reduc;
    if(c16 & lsb9) c16 ^= c9_reduc;
    if(c17 & lsb9) c17 ^= c9_reduc;
    if(c18 & lsb9) c18 ^= c9_reduc;
    if(c19 & lsb9) c19 ^= c9_reduc;
    if(c20 & lsb9) c20 ^= c9_reduc;
    if(c21 & lsb9) c21 ^= c9_reduc;
    if(c22 & lsb9) c22 ^= c9_reduc;
    mask ^= lsb9;

    uint64_t lsb10 = uint64_t_lsb(c10 & mask); // x10
    if(unlikely(!lsb10)) {
        return -1; // singular
    }

    uint64_t c10_reduc = c10 ^ lsb10;
    if(const_col & lsb10) const_col ^= c10_reduc;
    if(c11 & lsb10) c11 ^= c10_reduc;
    if(c12 & lsb10) c12 ^= c10_reduc;
    if(c13 & lsb10) c13 ^= c10_reduc;
    if(c14 & lsb10) c14 ^= c10_reduc;
    if(c15 & lsb10) c15 ^= c10_reduc;
    if(c16 & lsb10) c16 ^= c10_reduc;
    if(c17 & lsb10) c17 ^= c10_reduc;
    if(c18 & lsb10) c18 ^= c10_reduc;
    if(c19 & lsb10) c19 ^= c10_reduc;
    if(c20 & lsb10) c20 ^= c10_reduc;
    if(c21 & lsb10) c21 ^= c10_reduc;
    if(c22 & lsb10) c22 ^= c10_reduc;
    mask ^= lsb10;

    uint64_t lsb11 = uint64_t_lsb(c11 & mask); // x11
    if(unlikely(!lsb11)) {
        return -1; // singular
    }

    uint64_t c11_reduc = c11 ^ lsb11;
    if(const_col & lsb11) const_col ^= c11_reduc;
    if(c12 & lsb11) c12 ^= c11_reduc;
    if(c13 & lsb11) c13 ^= c11_reduc;
    if(c14 & lsb11) c14 ^= c11_reduc;
    if(c15 & lsb11) c15 ^= c11_reduc;
    if(c16 & lsb11) c16 ^= c11_reduc;
    if(c17 & lsb11) c17 ^= c11_reduc;
    if(c18 & lsb11) c18 ^= c11_reduc;
    if(c19 & lsb11) c19 ^= c11_reduc;
    if(c20 & lsb11) c20 ^= c11_reduc;
    if(c21 & lsb11) c21 ^= c11_reduc;
    if(c22 & lsb11) c22 ^= c11_reduc;
    mask ^= lsb11;

    uint64_t lsb12 = uint64_t_lsb(c12 & mask); // x12
    if(unlikely(!lsb12)) {
        return -1; // singular
    }

    uint64_t c12_reduc = c12 ^ lsb12;
    if(const_col & lsb12) const_col ^= c12_reduc;
    if(c13 & lsb12) c13 ^= c12_reduc;
    if(c14 & lsb12) c14 ^= c12_reduc;
    if(c15 & lsb12) c15 ^= c12_reduc;
    if(c16 & lsb12) c16 ^= c12_reduc;
    if(c17 & lsb12) c17 ^= c12_reduc;
    if(c18 & lsb12) c18 ^= c12_reduc;
    if(c19 & lsb12) c19 ^= c12_reduc;
    if(c20 & lsb12) c20 ^= c12_reduc;
    if(c21 & lsb12) c21 ^= c12_reduc;
    if(c22 & lsb12) c22 ^= c12_reduc;
    mask ^= lsb12;

    uint64_t lsb13 = uint64_t_lsb(c13 & mask); // x13
    if(unlikely(!lsb13)) {
        return -1; // singular
    }

    uint64_t c13_reduc = c13 ^ lsb13;
    if(const_col & lsb13) const_col ^= c13_reduc;
    if(c14 & lsb13) c14 ^= c13_reduc;
    if(c15 & lsb13) c15 ^= c13_reduc;
    if(c16 & lsb13) c16 ^= c13_reduc;
    if(c17 & lsb13) c17 ^= c13_reduc;
    if(c18 & lsb13) c18 ^= c13_reduc;
    if(c19 & lsb13) c19 ^= c13_reduc;
    if(c20 & lsb13) c20 ^= c13_reduc;
    if(c21 & lsb13) c21 ^= c13_reduc;
    if(c22 & lsb13) c22 ^= c13_reduc;
    mask ^= lsb13;

    uint64_t lsb14 = uint64_t_lsb(c14 & mask); // x14
    if(unlikely(!lsb14)) {
        return -1; // singular
    }

    uint64_t c14_reduc = c14 ^ lsb14;
    if(const_col & lsb14) const_col ^= c14_reduc;
    if(c15 & lsb14) c15 ^= c14_reduc;
    if(c16 & lsb14) c16 ^= c14_reduc;
    if(c17 & lsb14) c17 ^= c14_reduc;
    if(c18 & lsb14) c18 ^= c14_reduc;
    if(c19 & lsb14) c19 ^= c14_reduc;
    if(c20 & lsb14) c20 ^= c14_reduc;
    if(c21 & lsb14) c21 ^= c14_reduc;
    if(c22 & lsb14) c22 ^= c14_reduc;
    mask ^= lsb14;

    uint64_t lsb15 = uint64_t_lsb(c15 & mask); // x15
    if(unlikely(!lsb15)) {
        return -1; // singular
    }

    uint64_t c15_reduc = c15 ^ lsb15;
    if(const_col & lsb15) const_col ^= c15_reduc;
    if(c16 & lsb15) c16 ^= c15_reduc;
    if(c17 & lsb15) c17 ^= c15_reduc;
    if(c18 & lsb15) c18 ^= c15_reduc;
    if(c19 & lsb15) c19 ^= c15_reduc;
    if(c20 & lsb15) c20 ^= c15_reduc;
    if(c21 & lsb15) c21 ^= c15_reduc;
    if(c22 & lsb15) c22 ^= c15_reduc;
    mask ^= lsb15;

    uint64_t lsb16 = uint64_t_lsb(c16 & mask); // x16
    if(unlikely(!lsb16)) {
        return -1; // singular
    }

    uint64_t c16_reduc = c16 ^ lsb16;
    if(const_col & lsb16) const_col ^= c16_reduc;
    if(c17 & lsb16) c17 ^= c16_reduc;
    if(c18 & lsb16) c18 ^= c16_reduc;
    if(c19 & lsb16) c19 ^= c16_reduc;
    if(c20 & lsb16) c20 ^= c16_reduc;
    if(c21 & lsb16) c21 ^= c16_reduc;
    if(c22 & lsb16) c22 ^= c16_reduc;
    mask ^= lsb16;

    uint64_t lsb17 = uint64_t_lsb(c17 & mask); // x17
    if(unlikely(!lsb17)) {
        return -1; // singular
    }

    uint64_t c17_reduc = c17 ^ lsb17;
    if(const_col & lsb17) const_col ^= c17_reduc;
    if(c18 & lsb17) c18 ^= c17_reduc;
    if(c19 & lsb17) c19 ^= c17_reduc;
    if(c20 & lsb17) c20 ^= c17_reduc;
    if(c21 & lsb17) c21 ^= c17_reduc;
    if(c22 & lsb17) c22 ^= c17_reduc;
    mask ^= lsb17;

    uint64_t lsb18 = uint64_t_lsb(c18 & mask); // x18
    if(unlikely(!lsb18)) {
        return -1; // singular
    }

    uint64_t c18_reduc = c18 ^ lsb18;
    if(const_col & lsb18) const_col ^= c18_reduc;
    if(c19 & lsb18) c19 ^= c18_reduc;
    if(c20 & lsb18) c20 ^= c18_reduc;
    if(c21 & lsb18) c21 ^= c18_reduc;
    if(c22 & lsb18) c22 ^= c18_reduc;
    mask ^= lsb18;

    uint64_t lsb19 = uint64_t_lsb(c19 & mask); // x19
    if(unlikely(!lsb19)) {
        return -1; // singular
    }

    uint64_t c19_reduc = c19 ^ lsb19;
    if(const_col & lsb19) const_col ^= c19_reduc;
    if(c20 & lsb19) c20 ^= c19_reduc;
    if(c21 & lsb19) c21 ^= c19_reduc;
    if(c22 & lsb19) c22 ^= c19_reduc;
    mask ^= lsb19;

    uint64_t lsb20 = uint64_t_lsb(c20 & mask); // x20
    if(unlikely(!lsb20)) {
        return -1; // singular
    }

    uint64_t c20_reduc = c20 ^ lsb20;
    if(const_col & lsb20) const_col ^= c20_reduc;
    if(c21 & lsb20) c21 ^= c20_reduc;
    if(c22 & lsb20) c22 ^= c20_reduc;
    mask ^= lsb20;

    uint64_t lsb21 = uint64_t_lsb(c21 & mask); // x21
    if(unlikely(!lsb21)) {
        return -1; // singular
    }

    uint64_t c21_reduc = c21 ^ lsb21;
    if(const_col & lsb21) const_col ^= c21_reduc;
    if(c22 & lsb21) c22 ^= c21_reduc;
    mask ^= lsb21;

    uint64_t lsb22 = uint64_t_lsb(c22 & mask); // x22
    if(unlikely(!lsb22)) {
        return -1; // singular
    }

    if(const_col & lsb22) const_col ^= c22 ^ lsb22;
    mask ^= lsb22;

    if(likely(mask & const_col)) { // check system consistency
        return mask & const_col; // not solvable
    }

    uint64_t s = 0x0ULL;
    if(const_col & lsb1) s = uint64_t_toggle_at(s, 0);
    if(const_col & lsb2) s = uint64_t_toggle_at(s, 1);
    if(const_col & lsb3) s = uint64_t_toggle_at(s, 2);
    if(const_col & lsb4) s = uint64_t_toggle_at(s, 3);
    if(const_col & lsb5) s = uint64_t_toggle_at(s, 4);
    if(const_col & lsb6) s = uint64_t_toggle_at(s, 5);
    if(const_col & lsb7) s = uint64_t_toggle_at(s, 6);
    if(const_col & lsb8) s = uint64_t_toggle_at(s, 7);
    if(const_col & lsb9) s = uint64_t_toggle_at(s, 8);
    if(const_col & lsb10) s = uint64_t_toggle_at(s, 9);
    if(const_col & lsb11) s = uint64_t_toggle_at(s, 10);
    if(const_col & lsb12) s = uint64_t_toggle_at(s, 11);
    if(const_col & lsb13) s = uint64_t_toggle_at(s, 12);
    if(const_col & lsb14) s = uint64_t_toggle_at(s, 13);
    if(const_col & lsb15) s = uint64_t_toggle_at(s, 14);
    if(const_col & lsb16) s = uint64_t_toggle_at(s, 15);
    if(const_col & lsb17) s = uint64_t_toggle_at(s, 16);
    if(const_col & lsb18) s = uint64_t_toggle_at(s, 17);
    if(const_col & lsb19) s = uint64_t_toggle_at(s, 18);
    if(const_col & lsb20) s = uint64_t_toggle_at(s, 19);
    if(const_col & lsb21) s = uint64_t_toggle_at(s, 20);
    if(const_col & lsb22) s = uint64_t_toggle_at(s, 21);
    *sol = s;
    return 0;
}

int64_t
uint64a_gj_v23_generic(const uint64_t m[24], uint64_t* const restrict sol) {
    uint64_t const_col = m[0];
    uint64_t c1 = m[1]; // x1
    uint64_t c2 = m[2]; // x2
    uint64_t c3 = m[3]; // x3
    uint64_t c4 = m[4]; // x4
    uint64_t c5 = m[5]; // x5
    uint64_t c6 = m[6]; // x6
    uint64_t c7 = m[7]; // x7
    uint64_t c8 = m[8]; // x8
    uint64_t c9 = m[9]; // x9
    uint64_t c10 = m[10]; // x10
    uint64_t c11 = m[11]; // x11
    uint64_t c12 = m[12]; // x12
    uint64_t c13 = m[13]; // x13
    uint64_t c14 = m[14]; // x14
    uint64_t c15 = m[15]; // x15
    uint64_t c16 = m[16]; // x16
    uint64_t c17 = m[17]; // x17
    uint64_t c18 = m[18]; // x18
    uint64_t c19 = m[19]; // x19
    uint64_t c20 = m[20]; // x20
    uint64_t c21 = m[21]; // x21
    uint64_t c22 = m[22]; // x22
    uint64_t c23 = m[23]; // x23
    uint64_t lsb1 = uint64_t_lsb(c1); // initially all eqs are candidates
    if(unlikely(!lsb1)) {
        return -1; // singular
    }

    uint64_t c1_reduc = c1 ^ lsb1; // for row reduction
    if(const_col & lsb1) const_col ^= c1_reduc;
    if(c2 & lsb1) c2 ^= c1_reduc;
    if(c3 & lsb1) c3 ^= c1_reduc;
    if(c4 & lsb1) c4 ^= c1_reduc;
    if(c5 & lsb1) c5 ^= c1_reduc;
    if(c6 & lsb1) c6 ^= c1_reduc;
    if(c7 & lsb1) c7 ^= c1_reduc;
    if(c8 & lsb1) c8 ^= c1_reduc;
    if(c9 & lsb1) c9 ^= c1_reduc;
    if(c10 & lsb1) c10 ^= c1_reduc;
    if(c11 & lsb1) c11 ^= c1_reduc;
    if(c12 & lsb1) c12 ^= c1_reduc;
    if(c13 & lsb1) c13 ^= c1_reduc;
    if(c14 & lsb1) c14 ^= c1_reduc;
    if(c15 & lsb1) c15 ^= c1_reduc;
    if(c16 & lsb1) c16 ^= c1_reduc;
    if(c17 & lsb1) c17 ^= c1_reduc;
    if(c18 & lsb1) c18 ^= c1_reduc;
    if(c19 & lsb1) c19 ^= c1_reduc;
    if(c20 & lsb1) c20 ^= c1_reduc;
    if(c21 & lsb1) c21 ^= c1_reduc;
    if(c22 & lsb1) c22 ^= c1_reduc;
    if(c23 & lsb1) c23 ^= c1_reduc;
    uint64_t mask = ~lsb1;

    uint64_t lsb2 = uint64_t_lsb(c2 & mask); // x2
    if(unlikely(!lsb2)) {
        return -1; // singular
    }

    uint64_t c2_reduc = c2 ^ lsb2;
    if(const_col & lsb2) const_col ^= c2_reduc;
    if(c3 & lsb2) c3 ^= c2_reduc;
    if(c4 & lsb2) c4 ^= c2_reduc;
    if(c5 & lsb2) c5 ^= c2_reduc;
    if(c6 & lsb2) c6 ^= c2_reduc;
    if(c7 & lsb2) c7 ^= c2_reduc;
    if(c8 & lsb2) c8 ^= c2_reduc;
    if(c9 & lsb2) c9 ^= c2_reduc;
    if(c10 & lsb2) c10 ^= c2_reduc;
    if(c11 & lsb2) c11 ^= c2_reduc;
    if(c12 & lsb2) c12 ^= c2_reduc;
    if(c13 & lsb2) c13 ^= c2_reduc;
    if(c14 & lsb2) c14 ^= c2_reduc;
    if(c15 & lsb2) c15 ^= c2_reduc;
    if(c16 & lsb2) c16 ^= c2_reduc;
    if(c17 & lsb2) c17 ^= c2_reduc;
    if(c18 & lsb2) c18 ^= c2_reduc;
    if(c19 & lsb2) c19 ^= c2_reduc;
    if(c20 & lsb2) c20 ^= c2_reduc;
    if(c21 & lsb2) c21 ^= c2_reduc;
    if(c22 & lsb2) c22 ^= c2_reduc;
    if(c23 & lsb2) c23 ^= c2_reduc;
    mask ^= lsb2;

    uint64_t lsb3 = uint64_t_lsb(c3 & mask); // x3
    if(unlikely(!lsb3)) {
        return -1; // singular
    }

    uint64_t c3_reduc = c3 ^ lsb3;
    if(const_col & lsb3) const_col ^= c3_reduc;
    if(c4 & lsb3) c4 ^= c3_reduc;
    if(c5 & lsb3) c5 ^= c3_reduc;
    if(c6 & lsb3) c6 ^= c3_reduc;
    if(c7 & lsb3) c7 ^= c3_reduc;
    if(c8 & lsb3) c8 ^= c3_reduc;
    if(c9 & lsb3) c9 ^= c3_reduc;
    if(c10 & lsb3) c10 ^= c3_reduc;
    if(c11 & lsb3) c11 ^= c3_reduc;
    if(c12 & lsb3) c12 ^= c3_reduc;
    if(c13 & lsb3) c13 ^= c3_reduc;
    if(c14 & lsb3) c14 ^= c3_reduc;
    if(c15 & lsb3) c15 ^= c3_reduc;
    if(c16 & lsb3) c16 ^= c3_reduc;
    if(c17 & lsb3) c17 ^= c3_reduc;
    if(c18 & lsb3) c18 ^= c3_reduc;
    if(c19 & lsb3) c19 ^= c3_reduc;
    if(c20 & lsb3) c20 ^= c3_reduc;
    if(c21 & lsb3) c21 ^= c3_reduc;
    if(c22 & lsb3) c22 ^= c3_reduc;
    if(c23 & lsb3) c23 ^= c3_reduc;
    mask ^= lsb3;

    uint64_t lsb4 = uint64_t_lsb(c4 & mask); // x4
    if(unlikely(!lsb4)) {
        return -1; // singular
    }

    uint64_t c4_reduc = c4 ^ lsb4;
    if(const_col & lsb4) const_col ^= c4_reduc;
    if(c5 & lsb4) c5 ^= c4_reduc;
    if(c6 & lsb4) c6 ^= c4_reduc;
    if(c7 & lsb4) c7 ^= c4_reduc;
    if(c8 & lsb4) c8 ^= c4_reduc;
    if(c9 & lsb4) c9 ^= c4_reduc;
    if(c10 & lsb4) c10 ^= c4_reduc;
    if(c11 & lsb4) c11 ^= c4_reduc;
    if(c12 & lsb4) c12 ^= c4_reduc;
    if(c13 & lsb4) c13 ^= c4_reduc;
    if(c14 & lsb4) c14 ^= c4_reduc;
    if(c15 & lsb4) c15 ^= c4_reduc;
    if(c16 & lsb4) c16 ^= c4_reduc;
    if(c17 & lsb4) c17 ^= c4_reduc;
    if(c18 & lsb4) c18 ^= c4_reduc;
    if(c19 & lsb4) c19 ^= c4_reduc;
    if(c20 & lsb4) c20 ^= c4_reduc;
    if(c21 & lsb4) c21 ^= c4_reduc;
    if(c22 & lsb4) c22 ^= c4_reduc;
    if(c23 & lsb4) c23 ^= c4_reduc;
    mask ^= lsb4;

    uint64_t lsb5 = uint64_t_lsb(c5 & mask); // x5
    if(unlikely(!lsb5)) {
        return -1; // singular
    }

    uint64_t c5_reduc = c5 ^ lsb5;
    if(const_col & lsb5) const_col ^= c5_reduc;
    if(c6 & lsb5) c6 ^= c5_reduc;
    if(c7 & lsb5) c7 ^= c5_reduc;
    if(c8 & lsb5) c8 ^= c5_reduc;
    if(c9 & lsb5) c9 ^= c5_reduc;
    if(c10 & lsb5) c10 ^= c5_reduc;
    if(c11 & lsb5) c11 ^= c5_reduc;
    if(c12 & lsb5) c12 ^= c5_reduc;
    if(c13 & lsb5) c13 ^= c5_reduc;
    if(c14 & lsb5) c14 ^= c5_reduc;
    if(c15 & lsb5) c15 ^= c5_reduc;
    if(c16 & lsb5) c16 ^= c5_reduc;
    if(c17 & lsb5) c17 ^= c5_reduc;
    if(c18 & lsb5) c18 ^= c5_reduc;
    if(c19 & lsb5) c19 ^= c5_reduc;
    if(c20 & lsb5) c20 ^= c5_reduc;
    if(c21 & lsb5) c21 ^= c5_reduc;
    if(c22 & lsb5) c22 ^= c5_reduc;
    if(c23 & lsb5) c23 ^= c5_reduc;
    mask ^= lsb5;

    uint64_t lsb6 = uint64_t_lsb(c6 & mask); // x6
    if(unlikely(!lsb6)) {
        return -1; // singular
    }

    uint64_t c6_reduc = c6 ^ lsb6;
    if(const_col & lsb6) const_col ^= c6_reduc;
    if(c7 & lsb6) c7 ^= c6_reduc;
    if(c8 & lsb6) c8 ^= c6_reduc;
    if(c9 & lsb6) c9 ^= c6_reduc;
    if(c10 & lsb6) c10 ^= c6_reduc;
    if(c11 & lsb6) c11 ^= c6_reduc;
    if(c12 & lsb6) c12 ^= c6_reduc;
    if(c13 & lsb6) c13 ^= c6_reduc;
    if(c14 & lsb6) c14 ^= c6_reduc;
    if(c15 & lsb6) c15 ^= c6_reduc;
    if(c16 & lsb6) c16 ^= c6_reduc;
    if(c17 & lsb6) c17 ^= c6_reduc;
    if(c18 & lsb6) c18 ^= c6_reduc;
    if(c19 & lsb6) c19 ^= c6_reduc;
    if(c20 & lsb6) c20 ^= c6_reduc;
    if(c21 & lsb6) c21 ^= c6_reduc;
    if(c22 & lsb6) c22 ^= c6_reduc;
    if(c23 & lsb6) c23 ^= c6_reduc;
    mask ^= lsb6;

    uint64_t lsb7 = uint64_t_lsb(c7 & mask); // x7
    if(unlikely(!lsb7)) {
        return -1; // singular
    }

    uint64_t c7_reduc = c7 ^ lsb7;
    if(const_col & lsb7) const_col ^= c7_reduc;
    if(c8 & lsb7) c8 ^= c7_reduc;
    if(c9 & lsb7) c9 ^= c7_reduc;
    if(c10 & lsb7) c10 ^= c7_reduc;
    if(c11 & lsb7) c11 ^= c7_reduc;
    if(c12 & lsb7) c12 ^= c7_reduc;
    if(c13 & lsb7) c13 ^= c7_reduc;
    if(c14 & lsb7) c14 ^= c7_reduc;
    if(c15 & lsb7) c15 ^= c7_reduc;
    if(c16 & lsb7) c16 ^= c7_reduc;
    if(c17 & lsb7) c17 ^= c7_reduc;
    if(c18 & lsb7) c18 ^= c7_reduc;
    if(c19 & lsb7) c19 ^= c7_reduc;
    if(c20 & lsb7) c20 ^= c7_reduc;
    if(c21 & lsb7) c21 ^= c7_reduc;
    if(c22 & lsb7) c22 ^= c7_reduc;
    if(c23 & lsb7) c23 ^= c7_reduc;
    mask ^= lsb7;

    uint64_t lsb8 = uint64_t_lsb(c8 & mask); // x8
    if(unlikely(!lsb8)) {
        return -1; // singular
    }

    uint64_t c8_reduc = c8 ^ lsb8;
    if(const_col & lsb8) const_col ^= c8_reduc;
    if(c9 & lsb8) c9 ^= c8_reduc;
    if(c10 & lsb8) c10 ^= c8_reduc;
    if(c11 & lsb8) c11 ^= c8_reduc;
    if(c12 & lsb8) c12 ^= c8_reduc;
    if(c13 & lsb8) c13 ^= c8_reduc;
    if(c14 & lsb8) c14 ^= c8_reduc;
    if(c15 & lsb8) c15 ^= c8_reduc;
    if(c16 & lsb8) c16 ^= c8_reduc;
    if(c17 & lsb8) c17 ^= c8_reduc;
    if(c18 & lsb8) c18 ^= c8_reduc;
    if(c19 & lsb8) c19 ^= c8_reduc;
    if(c20 & lsb8) c20 ^= c8_reduc;
    if(c21 & lsb8) c21 ^= c8_reduc;
    if(c22 & lsb8) c22 ^= c8_reduc;
    if(c23 & lsb8) c23 ^= c8_reduc;
    mask ^= lsb8;

    uint64_t lsb9 = uint64_t_lsb(c9 & mask); // x9
    if(unlikely(!lsb9)) {
        return -1; // singular
    }

    uint64_t c9_reduc = c9 ^ lsb9;
    if(const_col & lsb9) const_col ^= c9_reduc;
    if(c10 & lsb9) c10 ^= c9_reduc;
    if(c11 & lsb9) c11 ^= c9_reduc;
    if(c12 & lsb9) c12 ^= c9_reduc;
    if(c13 & lsb9) c13 ^= c9_reduc;
    if(c14 & lsb9) c14 ^= c9_reduc;
    if(c15 & lsb9) c15 ^= c9_reduc;
    if(c16 & lsb9) c16 ^= c9_reduc;
    if(c17 & lsb9) c17 ^= c9_reduc;
    if(c18 & lsb9) c18 ^= c9_reduc;
    if(c19 & lsb9) c19 ^= c9_reduc;
    if(c20 & lsb9) c20 ^= c9_reduc;
    if(c21 & lsb9) c21 ^= c9_reduc;
    if(c22 & lsb9) c22 ^= c9_reduc;
    if(c23 & lsb9) c23 ^= c9_reduc;
    mask ^= lsb9;

    uint64_t lsb10 = uint64_t_lsb(c10 & mask); // x10
    if(unlikely(!lsb10)) {
        return -1; // singular
    }

    uint64_t c10_reduc = c10 ^ lsb10;
    if(const_col & lsb10) const_col ^= c10_reduc;
    if(c11 & lsb10) c11 ^= c10_reduc;
    if(c12 & lsb10) c12 ^= c10_reduc;
    if(c13 & lsb10) c13 ^= c10_reduc;
    if(c14 & lsb10) c14 ^= c10_reduc;
    if(c15 & lsb10) c15 ^= c10_reduc;
    if(c16 & lsb10) c16 ^= c10_reduc;
    if(c17 & lsb10) c17 ^= c10_reduc;
    if(c18 & lsb10) c18 ^= c10_reduc;
    if(c19 & lsb10) c19 ^= c10_reduc;
    if(c20 & lsb10) c20 ^= c10_reduc;
    if(c21 & lsb10) c21 ^= c10_reduc;
    if(c22 & lsb10) c22 ^= c10_reduc;
    if(c23 & lsb10) c23 ^= c10_reduc;
    mask ^= lsb10;

    uint64_t lsb11 = uint64_t_lsb(c11 & mask); // x11
    if(unlikely(!lsb11)) {
        return -1; // singular
    }

    uint64_t c11_reduc = c11 ^ lsb11;
    if(const_col & lsb11) const_col ^= c11_reduc;
    if(c12 & lsb11) c12 ^= c11_reduc;
    if(c13 & lsb11) c13 ^= c11_reduc;
    if(c14 & lsb11) c14 ^= c11_reduc;
    if(c15 & lsb11) c15 ^= c11_reduc;
    if(c16 & lsb11) c16 ^= c11_reduc;
    if(c17 & lsb11) c17 ^= c11_reduc;
    if(c18 & lsb11) c18 ^= c11_reduc;
    if(c19 & lsb11) c19 ^= c11_reduc;
    if(c20 & lsb11) c20 ^= c11_reduc;
    if(c21 & lsb11) c21 ^= c11_reduc;
    if(c22 & lsb11) c22 ^= c11_reduc;
    if(c23 & lsb11) c23 ^= c11_reduc;
    mask ^= lsb11;

    uint64_t lsb12 = uint64_t_lsb(c12 & mask); // x12
    if(unlikely(!lsb12)) {
        return -1; // singular
    }

    uint64_t c12_reduc = c12 ^ lsb12;
    if(const_col & lsb12) const_col ^= c12_reduc;
    if(c13 & lsb12) c13 ^= c12_reduc;
    if(c14 & lsb12) c14 ^= c12_reduc;
    if(c15 & lsb12) c15 ^= c12_reduc;
    if(c16 & lsb12) c16 ^= c12_reduc;
    if(c17 & lsb12) c17 ^= c12_reduc;
    if(c18 & lsb12) c18 ^= c12_reduc;
    if(c19 & lsb12) c19 ^= c12_reduc;
    if(c20 & lsb12) c20 ^= c12_reduc;
    if(c21 & lsb12) c21 ^= c12_reduc;
    if(c22 & lsb12) c22 ^= c12_reduc;
    if(c23 & lsb12) c23 ^= c12_reduc;
    mask ^= lsb12;

    uint64_t lsb13 = uint64_t_lsb(c13 & mask); // x13
    if(unlikely(!lsb13)) {
        return -1; // singular
    }

    uint64_t c13_reduc = c13 ^ lsb13;
    if(const_col & lsb13) const_col ^= c13_reduc;
    if(c14 & lsb13) c14 ^= c13_reduc;
    if(c15 & lsb13) c15 ^= c13_reduc;
    if(c16 & lsb13) c16 ^= c13_reduc;
    if(c17 & lsb13) c17 ^= c13_reduc;
    if(c18 & lsb13) c18 ^= c13_reduc;
    if(c19 & lsb13) c19 ^= c13_reduc;
    if(c20 & lsb13) c20 ^= c13_reduc;
    if(c21 & lsb13) c21 ^= c13_reduc;
    if(c22 & lsb13) c22 ^= c13_reduc;
    if(c23 & lsb13) c23 ^= c13_reduc;
    mask ^= lsb13;

    uint64_t lsb14 = uint64_t_lsb(c14 & mask); // x14
    if(unlikely(!lsb14)) {
        return -1; // singular
    }

    uint64_t c14_reduc = c14 ^ lsb14;
    if(const_col & lsb14) const_col ^= c14_reduc;
    if(c15 & lsb14) c15 ^= c14_reduc;
    if(c16 & lsb14) c16 ^= c14_reduc;
    if(c17 & lsb14) c17 ^= c14_reduc;
    if(c18 & lsb14) c18 ^= c14_reduc;
    if(c19 & lsb14) c19 ^= c14_reduc;
    if(c20 & lsb14) c20 ^= c14_reduc;
    if(c21 & lsb14) c21 ^= c14_reduc;
    if(c22 & lsb14) c22 ^= c14_reduc;
    if(c23 & lsb14) c23 ^= c14_reduc;
    mask ^= lsb14;

    uint64_t lsb15 = uint64_t_lsb(c15 & mask); // x15
    if(unlikely(!lsb15)) {
        return -1; // singular
    }

    uint64_t c15_reduc = c15 ^ lsb15;
    if(const_col & lsb15) const_col ^= c15_reduc;
    if(c16 & lsb15) c16 ^= c15_reduc;
    if(c17 & lsb15) c17 ^= c15_reduc;
    if(c18 & lsb15) c18 ^= c15_reduc;
    if(c19 & lsb15) c19 ^= c15_reduc;
    if(c20 & lsb15) c20 ^= c15_reduc;
    if(c21 & lsb15) c21 ^= c15_reduc;
    if(c22 & lsb15) c22 ^= c15_reduc;
    if(c23 & lsb15) c23 ^= c15_reduc;
    mask ^= lsb15;

    uint64_t lsb16 = uint64_t_lsb(c16 & mask); // x16
    if(unlikely(!lsb16)) {
        return -1; // singular
    }

    uint64_t c16_reduc = c16 ^ lsb16;
    if(const_col & lsb16) const_col ^= c16_reduc;
    if(c17 & lsb16) c17 ^= c16_reduc;
    if(c18 & lsb16) c18 ^= c16_reduc;
    if(c19 & lsb16) c19 ^= c16_reduc;
    if(c20 & lsb16) c20 ^= c16_reduc;
    if(c21 & lsb16) c21 ^= c16_reduc;
    if(c22 & lsb16) c22 ^= c16_reduc;
    if(c23 & lsb16) c23 ^= c16_reduc;
    mask ^= lsb16;

    uint64_t lsb17 = uint64_t_lsb(c17 & mask); // x17
    if(unlikely(!lsb17)) {
        return -1; // singular
    }

    uint64_t c17_reduc = c17 ^ lsb17;
    if(const_col & lsb17) const_col ^= c17_reduc;
    if(c18 & lsb17) c18 ^= c17_reduc;
    if(c19 & lsb17) c19 ^= c17_reduc;
    if(c20 & lsb17) c20 ^= c17_reduc;
    if(c21 & lsb17) c21 ^= c17_reduc;
    if(c22 & lsb17) c22 ^= c17_reduc;
    if(c23 & lsb17) c23 ^= c17_reduc;
    mask ^= lsb17;

    uint64_t lsb18 = uint64_t_lsb(c18 & mask); // x18
    if(unlikely(!lsb18)) {
        return -1; // singular
    }

    uint64_t c18_reduc = c18 ^ lsb18;
    if(const_col & lsb18) const_col ^= c18_reduc;
    if(c19 & lsb18) c19 ^= c18_reduc;
    if(c20 & lsb18) c20 ^= c18_reduc;
    if(c21 & lsb18) c21 ^= c18_reduc;
    if(c22 & lsb18) c22 ^= c18_reduc;
    if(c23 & lsb18) c23 ^= c18_reduc;
    mask ^= lsb18;

    uint64_t lsb19 = uint64_t_lsb(c19 & mask); // x19
    if(unlikely(!lsb19)) {
        return -1; // singular
    }

    uint64_t c19_reduc = c19 ^ lsb19;
    if(const_col & lsb19) const_col ^= c19_reduc;
    if(c20 & lsb19) c20 ^= c19_reduc;
    if(c21 & lsb19) c21 ^= c19_reduc;
    if(c22 & lsb19) c22 ^= c19_reduc;
    if(c23 & lsb19) c23 ^= c19_reduc;
    mask ^= lsb19;

    uint64_t lsb20 = uint64_t_lsb(c20 & mask); // x20
    if(unlikely(!lsb20)) {
        return -1; // singular
    }

    uint64_t c20_reduc = c20 ^ lsb20;
    if(const_col & lsb20) const_col ^= c20_reduc;
    if(c21 & lsb20) c21 ^= c20_reduc;
    if(c22 & lsb20) c22 ^= c20_reduc;
    if(c23 & lsb20) c23 ^= c20_reduc;
    mask ^= lsb20;

    uint64_t lsb21 = uint64_t_lsb(c21 & mask); // x21
    if(unlikely(!lsb21)) {
        return -1; // singular
    }

    uint64_t c21_reduc = c21 ^ lsb21;
    if(const_col & lsb21) const_col ^= c21_reduc;
    if(c22 & lsb21) c22 ^= c21_reduc;
    if(c23 & lsb21) c23 ^= c21_reduc;
    mask ^= lsb21;

    uint64_t lsb22 = uint64_t_lsb(c22 & mask); // x22
    if(unlikely(!lsb22)) {
        return -1; // singular
    }

    uint64_t c22_reduc = c22 ^ lsb22;
    if(const_col & lsb22) const_col ^= c22_reduc;
    if(c23 & lsb22) c23 ^= c22_reduc;
    mask ^= lsb22;

    uint64_t lsb23 = uint64_t_lsb(c23 & mask); // x23
    if(unlikely(!lsb23)) {
        return -1; // singular
    }

    if(const_col & lsb23) const_col ^= c23 ^ lsb23;
    mask ^= lsb23;

    if(likely(mask & const_col)) { // check system consistency
        return mask & const_col; // not solvable
    }

    uint64_t s = 0x0ULL;
    if(const_col & lsb1) s = uint64_t_toggle_at(s, 0);
    if(const_col & lsb2) s = uint64_t_toggle_at(s, 1);
    if(const_col & lsb3) s = uint64_t_toggle_at(s, 2);
    if(const_col & lsb4) s = uint64_t_toggle_at(s, 3);
    if(const_col & lsb5) s = uint64_t_toggle_at(s, 4);
    if(const_col & lsb6) s = uint64_t_toggle_at(s, 5);
    if(const_col & lsb7) s = uint64_t_toggle_at(s, 6);
    if(const_col & lsb8) s = uint64_t_toggle_at(s, 7);
    if(const_col & lsb9) s = uint64_t_toggle_at(s, 8);
    if(const_col & lsb10) s = uint64_t_toggle_at(s, 9);
    if(const_col & lsb11) s = uint64_t_toggle_at(s, 10);
    if(const_col & lsb12) s = uint64_t_toggle_at(s, 11);
    if(const_col & lsb13) s = uint64_t_toggle_at(s, 12);
    if(const_col & lsb14) s = uint64_t_toggle_at(s, 13);
    if(const_col & lsb15) s = uint64_t_toggle_at(s, 14);
    if(const_col & lsb16) s = uint64_t_toggle_at(s, 15);
    if(const_col & lsb17) s = uint64_t_toggle_at(s, 16);
    if(const_col & lsb18) s = uint64_t_toggle_at(s, 17);
    if(const_col & lsb19) s = uint64_t_toggle_at(s, 18);
    if(const_col & lsb20) s = uint64_t_toggle_at(s, 19);
    if(const_col & lsb21) s = uint64_t_toggle_at(s, 20);
    if(const_col & lsb22) s = uint64_t_toggle_at(s, 21);
    if(const_col & lsb23) s = uint64_t_toggle_at(s, 22);
    *sol = s;
    return 0;
}

int64_t
uint64a_gj_v24_generic(const uint64_t m[25], uint64_t* const restrict sol) {
    uint64_t const_col = m[0];
    uint64_t c1 = m[1]; // x1
    uint64_t c2 = m[2]; // x2
    uint64_t c3 = m[3]; // x3
    uint64_t c4 = m[4]; // x4
    uint64_t c5 = m[5]; // x5
    uint64_t c6 = m[6]; // x6
    uint64_t c7 = m[7]; // x7
    uint64_t c8 = m[8]; // x8
    uint64_t c9 = m[9]; // x9
    uint64_t c10 = m[10]; // x10
    uint64_t c11 = m[11]; // x11
    uint64_t c12 = m[12]; // x12
    uint64_t c13 = m[13]; // x13
    uint64_t c14 = m[14]; // x14
    uint64_t c15 = m[15]; // x15
    uint64_t c16 = m[16]; // x16
    uint64_t c17 = m[17]; // x17
    uint64_t c18 = m[18]; // x18
    uint64_t c19 = m[19]; // x19
    uint64_t c20 = m[20]; // x20
    uint64_t c21 = m[21]; // x21
    uint64_t c22 = m[22]; // x22
    uint64_t c23 = m[23]; // x23
    uint64_t c24 = m[24]; // x24
    uint64_t lsb1 = uint64_t_lsb(c1); // initially all eqs are candidates
    if(unlikely(!lsb1)) {
        return -1; // singular
    }

    uint64_t c1_reduc = c1 ^ lsb1; // for row reduction
    if(const_col & lsb1) const_col ^= c1_reduc;
    if(c2 & lsb1) c2 ^= c1_reduc;
    if(c3 & lsb1) c3 ^= c1_reduc;
    if(c4 & lsb1) c4 ^= c1_reduc;
    if(c5 & lsb1) c5 ^= c1_reduc;
    if(c6 & lsb1) c6 ^= c1_reduc;
    if(c7 & lsb1) c7 ^= c1_reduc;
    if(c8 & lsb1) c8 ^= c1_reduc;
    if(c9 & lsb1) c9 ^= c1_reduc;
    if(c10 & lsb1) c10 ^= c1_reduc;
    if(c11 & lsb1) c11 ^= c1_reduc;
    if(c12 & lsb1) c12 ^= c1_reduc;
    if(c13 & lsb1) c13 ^= c1_reduc;
    if(c14 & lsb1) c14 ^= c1_reduc;
    if(c15 & lsb1) c15 ^= c1_reduc;
    if(c16 & lsb1) c16 ^= c1_reduc;
    if(c17 & lsb1) c17 ^= c1_reduc;
    if(c18 & lsb1) c18 ^= c1_reduc;
    if(c19 & lsb1) c19 ^= c1_reduc;
    if(c20 & lsb1) c20 ^= c1_reduc;
    if(c21 & lsb1) c21 ^= c1_reduc;
    if(c22 & lsb1) c22 ^= c1_reduc;
    if(c23 & lsb1) c23 ^= c1_reduc;
    if(c24 & lsb1) c24 ^= c1_reduc;
    uint64_t mask = ~lsb1;

    uint64_t lsb2 = uint64_t_lsb(c2 & mask); // x2
    if(unlikely(!lsb2)) {
        return -1; // singular
    }

    uint64_t c2_reduc = c2 ^ lsb2;
    if(const_col & lsb2) const_col ^= c2_reduc;
    if(c3 & lsb2) c3 ^= c2_reduc;
    if(c4 & lsb2) c4 ^= c2_reduc;
    if(c5 & lsb2) c5 ^= c2_reduc;
    if(c6 & lsb2) c6 ^= c2_reduc;
    if(c7 & lsb2) c7 ^= c2_reduc;
    if(c8 & lsb2) c8 ^= c2_reduc;
    if(c9 & lsb2) c9 ^= c2_reduc;
    if(c10 & lsb2) c10 ^= c2_reduc;
    if(c11 & lsb2) c11 ^= c2_reduc;
    if(c12 & lsb2) c12 ^= c2_reduc;
    if(c13 & lsb2) c13 ^= c2_reduc;
    if(c14 & lsb2) c14 ^= c2_reduc;
    if(c15 & lsb2) c15 ^= c2_reduc;
    if(c16 & lsb2) c16 ^= c2_reduc;
    if(c17 & lsb2) c17 ^= c2_reduc;
    if(c18 & lsb2) c18 ^= c2_reduc;
    if(c19 & lsb2) c19 ^= c2_reduc;
    if(c20 & lsb2) c20 ^= c2_reduc;
    if(c21 & lsb2) c21 ^= c2_reduc;
    if(c22 & lsb2) c22 ^= c2_reduc;
    if(c23 & lsb2) c23 ^= c2_reduc;
    if(c24 & lsb2) c24 ^= c2_reduc;
    mask ^= lsb2;

    uint64_t lsb3 = uint64_t_lsb(c3 & mask); // x3
    if(unlikely(!lsb3)) {
        return -1; // singular
    }

    uint64_t c3_reduc = c3 ^ lsb3;
    if(const_col & lsb3) const_col ^= c3_reduc;
    if(c4 & lsb3) c4 ^= c3_reduc;
    if(c5 & lsb3) c5 ^= c3_reduc;
    if(c6 & lsb3) c6 ^= c3_reduc;
    if(c7 & lsb3) c7 ^= c3_reduc;
    if(c8 & lsb3) c8 ^= c3_reduc;
    if(c9 & lsb3) c9 ^= c3_reduc;
    if(c10 & lsb3) c10 ^= c3_reduc;
    if(c11 & lsb3) c11 ^= c3_reduc;
    if(c12 & lsb3) c12 ^= c3_reduc;
    if(c13 & lsb3) c13 ^= c3_reduc;
    if(c14 & lsb3) c14 ^= c3_reduc;
    if(c15 & lsb3) c15 ^= c3_reduc;
    if(c16 & lsb3) c16 ^= c3_reduc;
    if(c17 & lsb3) c17 ^= c3_reduc;
    if(c18 & lsb3) c18 ^= c3_reduc;
    if(c19 & lsb3) c19 ^= c3_reduc;
    if(c20 & lsb3) c20 ^= c3_reduc;
    if(c21 & lsb3) c21 ^= c3_reduc;
    if(c22 & lsb3) c22 ^= c3_reduc;
    if(c23 & lsb3) c23 ^= c3_reduc;
    if(c24 & lsb3) c24 ^= c3_reduc;
    mask ^= lsb3;

    uint64_t lsb4 = uint64_t_lsb(c4 & mask); // x4
    if(unlikely(!lsb4)) {
        return -1; // singular
    }

    uint64_t c4_reduc = c4 ^ lsb4;
    if(const_col & lsb4) const_col ^= c4_reduc;
    if(c5 & lsb4) c5 ^= c4_reduc;
    if(c6 & lsb4) c6 ^= c4_reduc;
    if(c7 & lsb4) c7 ^= c4_reduc;
    if(c8 & lsb4) c8 ^= c4_reduc;
    if(c9 & lsb4) c9 ^= c4_reduc;
    if(c10 & lsb4) c10 ^= c4_reduc;
    if(c11 & lsb4) c11 ^= c4_reduc;
    if(c12 & lsb4) c12 ^= c4_reduc;
    if(c13 & lsb4) c13 ^= c4_reduc;
    if(c14 & lsb4) c14 ^= c4_reduc;
    if(c15 & lsb4) c15 ^= c4_reduc;
    if(c16 & lsb4) c16 ^= c4_reduc;
    if(c17 & lsb4) c17 ^= c4_reduc;
    if(c18 & lsb4) c18 ^= c4_reduc;
    if(c19 & lsb4) c19 ^= c4_reduc;
    if(c20 & lsb4) c20 ^= c4_reduc;
    if(c21 & lsb4) c21 ^= c4_reduc;
    if(c22 & lsb4) c22 ^= c4_reduc;
    if(c23 & lsb4) c23 ^= c4_reduc;
    if(c24 & lsb4) c24 ^= c4_reduc;
    mask ^= lsb4;

    uint64_t lsb5 = uint64_t_lsb(c5 & mask); // x5
    if(unlikely(!lsb5)) {
        return -1; // singular
    }

    uint64_t c5_reduc = c5 ^ lsb5;
    if(const_col & lsb5) const_col ^= c5_reduc;
    if(c6 & lsb5) c6 ^= c5_reduc;
    if(c7 & lsb5) c7 ^= c5_reduc;
    if(c8 & lsb5) c8 ^= c5_reduc;
    if(c9 & lsb5) c9 ^= c5_reduc;
    if(c10 & lsb5) c10 ^= c5_reduc;
    if(c11 & lsb5) c11 ^= c5_reduc;
    if(c12 & lsb5) c12 ^= c5_reduc;
    if(c13 & lsb5) c13 ^= c5_reduc;
    if(c14 & lsb5) c14 ^= c5_reduc;
    if(c15 & lsb5) c15 ^= c5_reduc;
    if(c16 & lsb5) c16 ^= c5_reduc;
    if(c17 & lsb5) c17 ^= c5_reduc;
    if(c18 & lsb5) c18 ^= c5_reduc;
    if(c19 & lsb5) c19 ^= c5_reduc;
    if(c20 & lsb5) c20 ^= c5_reduc;
    if(c21 & lsb5) c21 ^= c5_reduc;
    if(c22 & lsb5) c22 ^= c5_reduc;
    if(c23 & lsb5) c23 ^= c5_reduc;
    if(c24 & lsb5) c24 ^= c5_reduc;
    mask ^= lsb5;

    uint64_t lsb6 = uint64_t_lsb(c6 & mask); // x6
    if(unlikely(!lsb6)) {
        return -1; // singular
    }

    uint64_t c6_reduc = c6 ^ lsb6;
    if(const_col & lsb6) const_col ^= c6_reduc;
    if(c7 & lsb6) c7 ^= c6_reduc;
    if(c8 & lsb6) c8 ^= c6_reduc;
    if(c9 & lsb6) c9 ^= c6_reduc;
    if(c10 & lsb6) c10 ^= c6_reduc;
    if(c11 & lsb6) c11 ^= c6_reduc;
    if(c12 & lsb6) c12 ^= c6_reduc;
    if(c13 & lsb6) c13 ^= c6_reduc;
    if(c14 & lsb6) c14 ^= c6_reduc;
    if(c15 & lsb6) c15 ^= c6_reduc;
    if(c16 & lsb6) c16 ^= c6_reduc;
    if(c17 & lsb6) c17 ^= c6_reduc;
    if(c18 & lsb6) c18 ^= c6_reduc;
    if(c19 & lsb6) c19 ^= c6_reduc;
    if(c20 & lsb6) c20 ^= c6_reduc;
    if(c21 & lsb6) c21 ^= c6_reduc;
    if(c22 & lsb6) c22 ^= c6_reduc;
    if(c23 & lsb6) c23 ^= c6_reduc;
    if(c24 & lsb6) c24 ^= c6_reduc;
    mask ^= lsb6;

    uint64_t lsb7 = uint64_t_lsb(c7 & mask); // x7
    if(unlikely(!lsb7)) {
        return -1; // singular
    }

    uint64_t c7_reduc = c7 ^ lsb7;
    if(const_col & lsb7) const_col ^= c7_reduc;
    if(c8 & lsb7) c8 ^= c7_reduc;
    if(c9 & lsb7) c9 ^= c7_reduc;
    if(c10 & lsb7) c10 ^= c7_reduc;
    if(c11 & lsb7) c11 ^= c7_reduc;
    if(c12 & lsb7) c12 ^= c7_reduc;
    if(c13 & lsb7) c13 ^= c7_reduc;
    if(c14 & lsb7) c14 ^= c7_reduc;
    if(c15 & lsb7) c15 ^= c7_reduc;
    if(c16 & lsb7) c16 ^= c7_reduc;
    if(c17 & lsb7) c17 ^= c7_reduc;
    if(c18 & lsb7) c18 ^= c7_reduc;
    if(c19 & lsb7) c19 ^= c7_reduc;
    if(c20 & lsb7) c20 ^= c7_reduc;
    if(c21 & lsb7) c21 ^= c7_reduc;
    if(c22 & lsb7) c22 ^= c7_reduc;
    if(c23 & lsb7) c23 ^= c7_reduc;
    if(c24 & lsb7) c24 ^= c7_reduc;
    mask ^= lsb7;

    uint64_t lsb8 = uint64_t_lsb(c8 & mask); // x8
    if(unlikely(!lsb8)) {
        return -1; // singular
    }

    uint64_t c8_reduc = c8 ^ lsb8;
    if(const_col & lsb8) const_col ^= c8_reduc;
    if(c9 & lsb8) c9 ^= c8_reduc;
    if(c10 & lsb8) c10 ^= c8_reduc;
    if(c11 & lsb8) c11 ^= c8_reduc;
    if(c12 & lsb8) c12 ^= c8_reduc;
    if(c13 & lsb8) c13 ^= c8_reduc;
    if(c14 & lsb8) c14 ^= c8_reduc;
    if(c15 & lsb8) c15 ^= c8_reduc;
    if(c16 & lsb8) c16 ^= c8_reduc;
    if(c17 & lsb8) c17 ^= c8_reduc;
    if(c18 & lsb8) c18 ^= c8_reduc;
    if(c19 & lsb8) c19 ^= c8_reduc;
    if(c20 & lsb8) c20 ^= c8_reduc;
    if(c21 & lsb8) c21 ^= c8_reduc;
    if(c22 & lsb8) c22 ^= c8_reduc;
    if(c23 & lsb8) c23 ^= c8_reduc;
    if(c24 & lsb8) c24 ^= c8_reduc;
    mask ^= lsb8;

    uint64_t lsb9 = uint64_t_lsb(c9 & mask); // x9
    if(unlikely(!lsb9)) {
        return -1; // singular
    }

    uint64_t c9_reduc = c9 ^ lsb9;
    if(const_col & lsb9) const_col ^= c9_reduc;
    if(c10 & lsb9) c10 ^= c9_reduc;
    if(c11 & lsb9) c11 ^= c9_reduc;
    if(c12 & lsb9) c12 ^= c9_reduc;
    if(c13 & lsb9) c13 ^= c9_reduc;
    if(c14 & lsb9) c14 ^= c9_reduc;
    if(c15 & lsb9) c15 ^= c9_reduc;
    if(c16 & lsb9) c16 ^= c9_reduc;
    if(c17 & lsb9) c17 ^= c9_reduc;
    if(c18 & lsb9) c18 ^= c9_reduc;
    if(c19 & lsb9) c19 ^= c9_reduc;
    if(c20 & lsb9) c20 ^= c9_reduc;
    if(c21 & lsb9) c21 ^= c9_reduc;
    if(c22 & lsb9) c22 ^= c9_reduc;
    if(c23 & lsb9) c23 ^= c9_reduc;
    if(c24 & lsb9) c24 ^= c9_reduc;
    mask ^= lsb9;

    uint64_t lsb10 = uint64_t_lsb(c10 & mask); // x10
    if(unlikely(!lsb10)) {
        return -1; // singular
    }

    uint64_t c10_reduc = c10 ^ lsb10;
    if(const_col & lsb10) const_col ^= c10_reduc;
    if(c11 & lsb10) c11 ^= c10_reduc;
    if(c12 & lsb10) c12 ^= c10_reduc;
    if(c13 & lsb10) c13 ^= c10_reduc;
    if(c14 & lsb10) c14 ^= c10_reduc;
    if(c15 & lsb10) c15 ^= c10_reduc;
    if(c16 & lsb10) c16 ^= c10_reduc;
    if(c17 & lsb10) c17 ^= c10_reduc;
    if(c18 & lsb10) c18 ^= c10_reduc;
    if(c19 & lsb10) c19 ^= c10_reduc;
    if(c20 & lsb10) c20 ^= c10_reduc;
    if(c21 & lsb10) c21 ^= c10_reduc;
    if(c22 & lsb10) c22 ^= c10_reduc;
    if(c23 & lsb10) c23 ^= c10_reduc;
    if(c24 & lsb10) c24 ^= c10_reduc;
    mask ^= lsb10;

    uint64_t lsb11 = uint64_t_lsb(c11 & mask); // x11
    if(unlikely(!lsb11)) {
        return -1; // singular
    }

    uint64_t c11_reduc = c11 ^ lsb11;
    if(const_col & lsb11) const_col ^= c11_reduc;
    if(c12 & lsb11) c12 ^= c11_reduc;
    if(c13 & lsb11) c13 ^= c11_reduc;
    if(c14 & lsb11) c14 ^= c11_reduc;
    if(c15 & lsb11) c15 ^= c11_reduc;
    if(c16 & lsb11) c16 ^= c11_reduc;
    if(c17 & lsb11) c17 ^= c11_reduc;
    if(c18 & lsb11) c18 ^= c11_reduc;
    if(c19 & lsb11) c19 ^= c11_reduc;
    if(c20 & lsb11) c20 ^= c11_reduc;
    if(c21 & lsb11) c21 ^= c11_reduc;
    if(c22 & lsb11) c22 ^= c11_reduc;
    if(c23 & lsb11) c23 ^= c11_reduc;
    if(c24 & lsb11) c24 ^= c11_reduc;
    mask ^= lsb11;

    uint64_t lsb12 = uint64_t_lsb(c12 & mask); // x12
    if(unlikely(!lsb12)) {
        return -1; // singular
    }

    uint64_t c12_reduc = c12 ^ lsb12;
    if(const_col & lsb12) const_col ^= c12_reduc;
    if(c13 & lsb12) c13 ^= c12_reduc;
    if(c14 & lsb12) c14 ^= c12_reduc;
    if(c15 & lsb12) c15 ^= c12_reduc;
    if(c16 & lsb12) c16 ^= c12_reduc;
    if(c17 & lsb12) c17 ^= c12_reduc;
    if(c18 & lsb12) c18 ^= c12_reduc;
    if(c19 & lsb12) c19 ^= c12_reduc;
    if(c20 & lsb12) c20 ^= c12_reduc;
    if(c21 & lsb12) c21 ^= c12_reduc;
    if(c22 & lsb12) c22 ^= c12_reduc;
    if(c23 & lsb12) c23 ^= c12_reduc;
    if(c24 & lsb12) c24 ^= c12_reduc;
    mask ^= lsb12;

    uint64_t lsb13 = uint64_t_lsb(c13 & mask); // x13
    if(unlikely(!lsb13)) {
        return -1; // singular
    }

    uint64_t c13_reduc = c13 ^ lsb13;
    if(const_col & lsb13) const_col ^= c13_reduc;
    if(c14 & lsb13) c14 ^= c13_reduc;
    if(c15 & lsb13) c15 ^= c13_reduc;
    if(c16 & lsb13) c16 ^= c13_reduc;
    if(c17 & lsb13) c17 ^= c13_reduc;
    if(c18 & lsb13) c18 ^= c13_reduc;
    if(c19 & lsb13) c19 ^= c13_reduc;
    if(c20 & lsb13) c20 ^= c13_reduc;
    if(c21 & lsb13) c21 ^= c13_reduc;
    if(c22 & lsb13) c22 ^= c13_reduc;
    if(c23 & lsb13) c23 ^= c13_reduc;
    if(c24 & lsb13) c24 ^= c13_reduc;
    mask ^= lsb13;

    uint64_t lsb14 = uint64_t_lsb(c14 & mask); // x14
    if(unlikely(!lsb14)) {
        return -1; // singular
    }

    uint64_t c14_reduc = c14 ^ lsb14;
    if(const_col & lsb14) const_col ^= c14_reduc;
    if(c15 & lsb14) c15 ^= c14_reduc;
    if(c16 & lsb14) c16 ^= c14_reduc;
    if(c17 & lsb14) c17 ^= c14_reduc;
    if(c18 & lsb14) c18 ^= c14_reduc;
    if(c19 & lsb14) c19 ^= c14_reduc;
    if(c20 & lsb14) c20 ^= c14_reduc;
    if(c21 & lsb14) c21 ^= c14_reduc;
    if(c22 & lsb14) c22 ^= c14_reduc;
    if(c23 & lsb14) c23 ^= c14_reduc;
    if(c24 & lsb14) c24 ^= c14_reduc;
    mask ^= lsb14;

    uint64_t lsb15 = uint64_t_lsb(c15 & mask); // x15
    if(unlikely(!lsb15)) {
        return -1; // singular
    }

    uint64_t c15_reduc = c15 ^ lsb15;
    if(const_col & lsb15) const_col ^= c15_reduc;
    if(c16 & lsb15) c16 ^= c15_reduc;
    if(c17 & lsb15) c17 ^= c15_reduc;
    if(c18 & lsb15) c18 ^= c15_reduc;
    if(c19 & lsb15) c19 ^= c15_reduc;
    if(c20 & lsb15) c20 ^= c15_reduc;
    if(c21 & lsb15) c21 ^= c15_reduc;
    if(c22 & lsb15) c22 ^= c15_reduc;
    if(c23 & lsb15) c23 ^= c15_reduc;
    if(c24 & lsb15) c24 ^= c15_reduc;
    mask ^= lsb15;

    uint64_t lsb16 = uint64_t_lsb(c16 & mask); // x16
    if(unlikely(!lsb16)) {
        return -1; // singular
    }

    uint64_t c16_reduc = c16 ^ lsb16;
    if(const_col & lsb16) const_col ^= c16_reduc;
    if(c17 & lsb16) c17 ^= c16_reduc;
    if(c18 & lsb16) c18 ^= c16_reduc;
    if(c19 & lsb16) c19 ^= c16_reduc;
    if(c20 & lsb16) c20 ^= c16_reduc;
    if(c21 & lsb16) c21 ^= c16_reduc;
    if(c22 & lsb16) c22 ^= c16_reduc;
    if(c23 & lsb16) c23 ^= c16_reduc;
    if(c24 & lsb16) c24 ^= c16_reduc;
    mask ^= lsb16;

    uint64_t lsb17 = uint64_t_lsb(c17 & mask); // x17
    if(unlikely(!lsb17)) {
        return -1; // singular
    }

    uint64_t c17_reduc = c17 ^ lsb17;
    if(const_col & lsb17) const_col ^= c17_reduc;
    if(c18 & lsb17) c18 ^= c17_reduc;
    if(c19 & lsb17) c19 ^= c17_reduc;
    if(c20 & lsb17) c20 ^= c17_reduc;
    if(c21 & lsb17) c21 ^= c17_reduc;
    if(c22 & lsb17) c22 ^= c17_reduc;
    if(c23 & lsb17) c23 ^= c17_reduc;
    if(c24 & lsb17) c24 ^= c17_reduc;
    mask ^= lsb17;

    uint64_t lsb18 = uint64_t_lsb(c18 & mask); // x18
    if(unlikely(!lsb18)) {
        return -1; // singular
    }

    uint64_t c18_reduc = c18 ^ lsb18;
    if(const_col & lsb18) const_col ^= c18_reduc;
    if(c19 & lsb18) c19 ^= c18_reduc;
    if(c20 & lsb18) c20 ^= c18_reduc;
    if(c21 & lsb18) c21 ^= c18_reduc;
    if(c22 & lsb18) c22 ^= c18_reduc;
    if(c23 & lsb18) c23 ^= c18_reduc;
    if(c24 & lsb18) c24 ^= c18_reduc;
    mask ^= lsb18;

    uint64_t lsb19 = uint64_t_lsb(c19 & mask); // x19
    if(unlikely(!lsb19)) {
        return -1; // singular
    }

    uint64_t c19_reduc = c19 ^ lsb19;
    if(const_col & lsb19) const_col ^= c19_reduc;
    if(c20 & lsb19) c20 ^= c19_reduc;
    if(c21 & lsb19) c21 ^= c19_reduc;
    if(c22 & lsb19) c22 ^= c19_reduc;
    if(c23 & lsb19) c23 ^= c19_reduc;
    if(c24 & lsb19) c24 ^= c19_reduc;
    mask ^= lsb19;

    uint64_t lsb20 = uint64_t_lsb(c20 & mask); // x20
    if(unlikely(!lsb20)) {
        return -1; // singular
    }

    uint64_t c20_reduc = c20 ^ lsb20;
    if(const_col & lsb20) const_col ^= c20_reduc;
    if(c21 & lsb20) c21 ^= c20_reduc;
    if(c22 & lsb20) c22 ^= c20_reduc;
    if(c23 & lsb20) c23 ^= c20_reduc;
    if(c24 & lsb20) c24 ^= c20_reduc;
    mask ^= lsb20;

    uint64_t lsb21 = uint64_t_lsb(c21 & mask); // x21
    if(unlikely(!lsb21)) {
        return -1; // singular
    }

    uint64_t c21_reduc = c21 ^ lsb21;
    if(const_col & lsb21) const_col ^= c21_reduc;
    if(c22 & lsb21) c22 ^= c21_reduc;
    if(c23 & lsb21) c23 ^= c21_reduc;
    if(c24 & lsb21) c24 ^= c21_reduc;
    mask ^= lsb21;

    uint64_t lsb22 = uint64_t_lsb(c22 & mask); // x22
    if(unlikely(!lsb22)) {
        return -1; // singular
    }

    uint64_t c22_reduc = c22 ^ lsb22;
    if(const_col & lsb22) const_col ^= c22_reduc;
    if(c23 & lsb22) c23 ^= c22_reduc;
    if(c24 & lsb22) c24 ^= c22_reduc;
    mask ^= lsb22;

    uint64_t lsb23 = uint64_t_lsb(c23 & mask); // x23
    if(unlikely(!lsb23)) {
        return -1; // singular
    }

    uint64_t c23_reduc = c23 ^ lsb23;
    if(const_col & lsb23) const_col ^= c23_reduc;
    if(c24 & lsb23) c24 ^= c23_reduc;
    mask ^= lsb23;

    uint64_t lsb24 = uint64_t_lsb(c24 & mask); // x24
    if(unlikely(!lsb24)) {
        return -1; // singular
    }

    if(const_col & lsb24) const_col ^= c24 ^ lsb24;
    mask ^= lsb24;

    if(likely(mask & const_col)) { // check system consistency
        return mask & const_col; // not solvable
    }

    uint64_t s = 0x0ULL;
    if(const_col & lsb1) s = uint64_t_toggle_at(s, 0);
    if(const_col & lsb2) s = uint64_t_toggle_at(s, 1);
    if(const_col & lsb3) s = uint64_t_toggle_at(s, 2);
    if(const_col & lsb4) s = uint64_t_toggle_at(s, 3);
    if(const_col & lsb5) s = uint64_t_toggle_at(s, 4);
    if(const_col & lsb6) s = uint64_t_toggle_at(s, 5);
    if(const_col & lsb7) s = uint64_t_toggle_at(s, 6);
    if(const_col & lsb8) s = uint64_t_toggle_at(s, 7);
    if(const_col & lsb9) s = uint64_t_toggle_at(s, 8);
    if(const_col & lsb10) s = uint64_t_toggle_at(s, 9);
    if(const_col & lsb11) s = uint64_t_toggle_at(s, 10);
    if(const_col & lsb12) s = uint64_t_toggle_at(s, 11);
    if(const_col & lsb13) s = uint64_t_toggle_at(s, 12);
    if(const_col & lsb14) s = uint64_t_toggle_at(s, 13);
    if(const_col & lsb15) s = uint64_t_toggle_at(s, 14);
    if(const_col & lsb16) s = uint64_t_toggle_at(s, 15);
    if(const_col & lsb17) s = uint64_t_toggle_at(s, 16);
    if(const_col & lsb18) s = uint64_t_toggle_at(s, 17);
    if(const_col & lsb19) s = uint64_t_toggle_at(s, 18);
    if(const_col & lsb20) s = uint64_t_toggle_at(s, 19);
    if(const_col & lsb21) s = uint64_t_toggle_at(s, 20);
    if(const_col & lsb22) s = uint64_t_toggle_at(s, 21);
    if(const_col & lsb23) s = uint64_t_toggle_at(s, 22);
    if(const_col & lsb24) s = uint64_t_toggle_at(s, 23);
    *sol = s;
    return 0;
}

int64_t
uint64a_gj_v25_generic(const uint64_t m[26], uint64_t* const restrict sol) {
    uint64_t const_col = m[0];
    uint64_t c1 = m[1]; // x1
    uint64_t c2 = m[2]; // x2
    uint64_t c3 = m[3]; // x3
    uint64_t c4 = m[4]; // x4
    uint64_t c5 = m[5]; // x5
    uint64_t c6 = m[6]; // x6
    uint64_t c7 = m[7]; // x7
    uint64_t c8 = m[8]; // x8
    uint64_t c9 = m[9]; // x9
    uint64_t c10 = m[10]; // x10
    uint64_t c11 = m[11]; // x11
    uint64_t c12 = m[12]; // x12
    uint64_t c13 = m[13]; // x13
    uint64_t c14 = m[14]; // x14
    uint64_t c15 = m[15]; // x15
    uint64_t c16 = m[16]; // x16
    uint64_t c17 = m[17]; // x17
    uint64_t c18 = m[18]; // x18
    uint64_t c19 = m[19]; // x19
    uint64_t c20 = m[20]; // x20
    uint64_t c21 = m[21]; // x21
    uint64_t c22 = m[22]; // x22
    uint64_t c23 = m[23]; // x23
    uint64_t c24 = m[24]; // x24
    uint64_t c25 = m[25]; // x25
    uint64_t lsb1 = uint64_t_lsb(c1); // initially all eqs are candidates
    if(unlikely(!lsb1)) {
        return -1; // singular
    }

    uint64_t c1_reduc = c1 ^ lsb1; // for row reduction
    if(const_col & lsb1) const_col ^= c1_reduc;
    if(c2 & lsb1) c2 ^= c1_reduc;
    if(c3 & lsb1) c3 ^= c1_reduc;
    if(c4 & lsb1) c4 ^= c1_reduc;
    if(c5 & lsb1) c5 ^= c1_reduc;
    if(c6 & lsb1) c6 ^= c1_reduc;
    if(c7 & lsb1) c7 ^= c1_reduc;
    if(c8 & lsb1) c8 ^= c1_reduc;
    if(c9 & lsb1) c9 ^= c1_reduc;
    if(c10 & lsb1) c10 ^= c1_reduc;
    if(c11 & lsb1) c11 ^= c1_reduc;
    if(c12 & lsb1) c12 ^= c1_reduc;
    if(c13 & lsb1) c13 ^= c1_reduc;
    if(c14 & lsb1) c14 ^= c1_reduc;
    if(c15 & lsb1) c15 ^= c1_reduc;
    if(c16 & lsb1) c16 ^= c1_reduc;
    if(c17 & lsb1) c17 ^= c1_reduc;
    if(c18 & lsb1) c18 ^= c1_reduc;
    if(c19 & lsb1) c19 ^= c1_reduc;
    if(c20 & lsb1) c20 ^= c1_reduc;
    if(c21 & lsb1) c21 ^= c1_reduc;
    if(c22 & lsb1) c22 ^= c1_reduc;
    if(c23 & lsb1) c23 ^= c1_reduc;
    if(c24 & lsb1) c24 ^= c1_reduc;
    if(c25 & lsb1) c25 ^= c1_reduc;
    uint64_t mask = ~lsb1;

    uint64_t lsb2 = uint64_t_lsb(c2 & mask); // x2
    if(unlikely(!lsb2)) {
        return -1; // singular
    }

    uint64_t c2_reduc = c2 ^ lsb2;
    if(const_col & lsb2) const_col ^= c2_reduc;
    if(c3 & lsb2) c3 ^= c2_reduc;
    if(c4 & lsb2) c4 ^= c2_reduc;
    if(c5 & lsb2) c5 ^= c2_reduc;
    if(c6 & lsb2) c6 ^= c2_reduc;
    if(c7 & lsb2) c7 ^= c2_reduc;
    if(c8 & lsb2) c8 ^= c2_reduc;
    if(c9 & lsb2) c9 ^= c2_reduc;
    if(c10 & lsb2) c10 ^= c2_reduc;
    if(c11 & lsb2) c11 ^= c2_reduc;
    if(c12 & lsb2) c12 ^= c2_reduc;
    if(c13 & lsb2) c13 ^= c2_reduc;
    if(c14 & lsb2) c14 ^= c2_reduc;
    if(c15 & lsb2) c15 ^= c2_reduc;
    if(c16 & lsb2) c16 ^= c2_reduc;
    if(c17 & lsb2) c17 ^= c2_reduc;
    if(c18 & lsb2) c18 ^= c2_reduc;
    if(c19 & lsb2) c19 ^= c2_reduc;
    if(c20 & lsb2) c20 ^= c2_reduc;
    if(c21 & lsb2) c21 ^= c2_reduc;
    if(c22 & lsb2) c22 ^= c2_reduc;
    if(c23 & lsb2) c23 ^= c2_reduc;
    if(c24 & lsb2) c24 ^= c2_reduc;
    if(c25 & lsb2) c25 ^= c2_reduc;
    mask ^= lsb2;

    uint64_t lsb3 = uint64_t_lsb(c3 & mask); // x3
    if(unlikely(!lsb3)) {
        return -1; // singular
    }

    uint64_t c3_reduc = c3 ^ lsb3;
    if(const_col & lsb3) const_col ^= c3_reduc;
    if(c4 & lsb3) c4 ^= c3_reduc;
    if(c5 & lsb3) c5 ^= c3_reduc;
    if(c6 & lsb3) c6 ^= c3_reduc;
    if(c7 & lsb3) c7 ^= c3_reduc;
    if(c8 & lsb3) c8 ^= c3_reduc;
    if(c9 & lsb3) c9 ^= c3_reduc;
    if(c10 & lsb3) c10 ^= c3_reduc;
    if(c11 & lsb3) c11 ^= c3_reduc;
    if(c12 & lsb3) c12 ^= c3_reduc;
    if(c13 & lsb3) c13 ^= c3_reduc;
    if(c14 & lsb3) c14 ^= c3_reduc;
    if(c15 & lsb3) c15 ^= c3_reduc;
    if(c16 & lsb3) c16 ^= c3_reduc;
    if(c17 & lsb3) c17 ^= c3_reduc;
    if(c18 & lsb3) c18 ^= c3_reduc;
    if(c19 & lsb3) c19 ^= c3_reduc;
    if(c20 & lsb3) c20 ^= c3_reduc;
    if(c21 & lsb3) c21 ^= c3_reduc;
    if(c22 & lsb3) c22 ^= c3_reduc;
    if(c23 & lsb3) c23 ^= c3_reduc;
    if(c24 & lsb3) c24 ^= c3_reduc;
    if(c25 & lsb3) c25 ^= c3_reduc;
    mask ^= lsb3;

    uint64_t lsb4 = uint64_t_lsb(c4 & mask); // x4
    if(unlikely(!lsb4)) {
        return -1; // singular
    }

    uint64_t c4_reduc = c4 ^ lsb4;
    if(const_col & lsb4) const_col ^= c4_reduc;
    if(c5 & lsb4) c5 ^= c4_reduc;
    if(c6 & lsb4) c6 ^= c4_reduc;
    if(c7 & lsb4) c7 ^= c4_reduc;
    if(c8 & lsb4) c8 ^= c4_reduc;
    if(c9 & lsb4) c9 ^= c4_reduc;
    if(c10 & lsb4) c10 ^= c4_reduc;
    if(c11 & lsb4) c11 ^= c4_reduc;
    if(c12 & lsb4) c12 ^= c4_reduc;
    if(c13 & lsb4) c13 ^= c4_reduc;
    if(c14 & lsb4) c14 ^= c4_reduc;
    if(c15 & lsb4) c15 ^= c4_reduc;
    if(c16 & lsb4) c16 ^= c4_reduc;
    if(c17 & lsb4) c17 ^= c4_reduc;
    if(c18 & lsb4) c18 ^= c4_reduc;
    if(c19 & lsb4) c19 ^= c4_reduc;
    if(c20 & lsb4) c20 ^= c4_reduc;
    if(c21 & lsb4) c21 ^= c4_reduc;
    if(c22 & lsb4) c22 ^= c4_reduc;
    if(c23 & lsb4) c23 ^= c4_reduc;
    if(c24 & lsb4) c24 ^= c4_reduc;
    if(c25 & lsb4) c25 ^= c4_reduc;
    mask ^= lsb4;

    uint64_t lsb5 = uint64_t_lsb(c5 & mask); // x5
    if(unlikely(!lsb5)) {
        return -1; // singular
    }

    uint64_t c5_reduc = c5 ^ lsb5;
    if(const_col & lsb5) const_col ^= c5_reduc;
    if(c6 & lsb5) c6 ^= c5_reduc;
    if(c7 & lsb5) c7 ^= c5_reduc;
    if(c8 & lsb5) c8 ^= c5_reduc;
    if(c9 & lsb5) c9 ^= c5_reduc;
    if(c10 & lsb5) c10 ^= c5_reduc;
    if(c11 & lsb5) c11 ^= c5_reduc;
    if(c12 & lsb5) c12 ^= c5_reduc;
    if(c13 & lsb5) c13 ^= c5_reduc;
    if(c14 & lsb5) c14 ^= c5_reduc;
    if(c15 & lsb5) c15 ^= c5_reduc;
    if(c16 & lsb5) c16 ^= c5_reduc;
    if(c17 & lsb5) c17 ^= c5_reduc;
    if(c18 & lsb5) c18 ^= c5_reduc;
    if(c19 & lsb5) c19 ^= c5_reduc;
    if(c20 & lsb5) c20 ^= c5_reduc;
    if(c21 & lsb5) c21 ^= c5_reduc;
    if(c22 & lsb5) c22 ^= c5_reduc;
    if(c23 & lsb5) c23 ^= c5_reduc;
    if(c24 & lsb5) c24 ^= c5_reduc;
    if(c25 & lsb5) c25 ^= c5_reduc;
    mask ^= lsb5;

    uint64_t lsb6 = uint64_t_lsb(c6 & mask); // x6
    if(unlikely(!lsb6)) {
        return -1; // singular
    }

    uint64_t c6_reduc = c6 ^ lsb6;
    if(const_col & lsb6) const_col ^= c6_reduc;
    if(c7 & lsb6) c7 ^= c6_reduc;
    if(c8 & lsb6) c8 ^= c6_reduc;
    if(c9 & lsb6) c9 ^= c6_reduc;
    if(c10 & lsb6) c10 ^= c6_reduc;
    if(c11 & lsb6) c11 ^= c6_reduc;
    if(c12 & lsb6) c12 ^= c6_reduc;
    if(c13 & lsb6) c13 ^= c6_reduc;
    if(c14 & lsb6) c14 ^= c6_reduc;
    if(c15 & lsb6) c15 ^= c6_reduc;
    if(c16 & lsb6) c16 ^= c6_reduc;
    if(c17 & lsb6) c17 ^= c6_reduc;
    if(c18 & lsb6) c18 ^= c6_reduc;
    if(c19 & lsb6) c19 ^= c6_reduc;
    if(c20 & lsb6) c20 ^= c6_reduc;
    if(c21 & lsb6) c21 ^= c6_reduc;
    if(c22 & lsb6) c22 ^= c6_reduc;
    if(c23 & lsb6) c23 ^= c6_reduc;
    if(c24 & lsb6) c24 ^= c6_reduc;
    if(c25 & lsb6) c25 ^= c6_reduc;
    mask ^= lsb6;

    uint64_t lsb7 = uint64_t_lsb(c7 & mask); // x7
    if(unlikely(!lsb7)) {
        return -1; // singular
    }

    uint64_t c7_reduc = c7 ^ lsb7;
    if(const_col & lsb7) const_col ^= c7_reduc;
    if(c8 & lsb7) c8 ^= c7_reduc;
    if(c9 & lsb7) c9 ^= c7_reduc;
    if(c10 & lsb7) c10 ^= c7_reduc;
    if(c11 & lsb7) c11 ^= c7_reduc;
    if(c12 & lsb7) c12 ^= c7_reduc;
    if(c13 & lsb7) c13 ^= c7_reduc;
    if(c14 & lsb7) c14 ^= c7_reduc;
    if(c15 & lsb7) c15 ^= c7_reduc;
    if(c16 & lsb7) c16 ^= c7_reduc;
    if(c17 & lsb7) c17 ^= c7_reduc;
    if(c18 & lsb7) c18 ^= c7_reduc;
    if(c19 & lsb7) c19 ^= c7_reduc;
    if(c20 & lsb7) c20 ^= c7_reduc;
    if(c21 & lsb7) c21 ^= c7_reduc;
    if(c22 & lsb7) c22 ^= c7_reduc;
    if(c23 & lsb7) c23 ^= c7_reduc;
    if(c24 & lsb7) c24 ^= c7_reduc;
    if(c25 & lsb7) c25 ^= c7_reduc;
    mask ^= lsb7;

    uint64_t lsb8 = uint64_t_lsb(c8 & mask); // x8
    if(unlikely(!lsb8)) {
        return -1; // singular
    }

    uint64_t c8_reduc = c8 ^ lsb8;
    if(const_col & lsb8) const_col ^= c8_reduc;
    if(c9 & lsb8) c9 ^= c8_reduc;
    if(c10 & lsb8) c10 ^= c8_reduc;
    if(c11 & lsb8) c11 ^= c8_reduc;
    if(c12 & lsb8) c12 ^= c8_reduc;
    if(c13 & lsb8) c13 ^= c8_reduc;
    if(c14 & lsb8) c14 ^= c8_reduc;
    if(c15 & lsb8) c15 ^= c8_reduc;
    if(c16 & lsb8) c16 ^= c8_reduc;
    if(c17 & lsb8) c17 ^= c8_reduc;
    if(c18 & lsb8) c18 ^= c8_reduc;
    if(c19 & lsb8) c19 ^= c8_reduc;
    if(c20 & lsb8) c20 ^= c8_reduc;
    if(c21 & lsb8) c21 ^= c8_reduc;
    if(c22 & lsb8) c22 ^= c8_reduc;
    if(c23 & lsb8) c23 ^= c8_reduc;
    if(c24 & lsb8) c24 ^= c8_reduc;
    if(c25 & lsb8) c25 ^= c8_reduc;
    mask ^= lsb8;

    uint64_t lsb9 = uint64_t_lsb(c9 & mask); // x9
    if(unlikely(!lsb9)) {
        return -1; // singular
    }

    uint64_t c9_reduc = c9 ^ lsb9;
    if(const_col & lsb9) const_col ^= c9_reduc;
    if(c10 & lsb9) c10 ^= c9_reduc;
    if(c11 & lsb9) c11 ^= c9_reduc;
    if(c12 & lsb9) c12 ^= c9_reduc;
    if(c13 & lsb9) c13 ^= c9_reduc;
    if(c14 & lsb9) c14 ^= c9_reduc;
    if(c15 & lsb9) c15 ^= c9_reduc;
    if(c16 & lsb9) c16 ^= c9_reduc;
    if(c17 & lsb9) c17 ^= c9_reduc;
    if(c18 & lsb9) c18 ^= c9_reduc;
    if(c19 & lsb9) c19 ^= c9_reduc;
    if(c20 & lsb9) c20 ^= c9_reduc;
    if(c21 & lsb9) c21 ^= c9_reduc;
    if(c22 & lsb9) c22 ^= c9_reduc;
    if(c23 & lsb9) c23 ^= c9_reduc;
    if(c24 & lsb9) c24 ^= c9_reduc;
    if(c25 & lsb9) c25 ^= c9_reduc;
    mask ^= lsb9;

    uint64_t lsb10 = uint64_t_lsb(c10 & mask); // x10
    if(unlikely(!lsb10)) {
        return -1; // singular
    }

    uint64_t c10_reduc = c10 ^ lsb10;
    if(const_col & lsb10) const_col ^= c10_reduc;
    if(c11 & lsb10) c11 ^= c10_reduc;
    if(c12 & lsb10) c12 ^= c10_reduc;
    if(c13 & lsb10) c13 ^= c10_reduc;
    if(c14 & lsb10) c14 ^= c10_reduc;
    if(c15 & lsb10) c15 ^= c10_reduc;
    if(c16 & lsb10) c16 ^= c10_reduc;
    if(c17 & lsb10) c17 ^= c10_reduc;
    if(c18 & lsb10) c18 ^= c10_reduc;
    if(c19 & lsb10) c19 ^= c10_reduc;
    if(c20 & lsb10) c20 ^= c10_reduc;
    if(c21 & lsb10) c21 ^= c10_reduc;
    if(c22 & lsb10) c22 ^= c10_reduc;
    if(c23 & lsb10) c23 ^= c10_reduc;
    if(c24 & lsb10) c24 ^= c10_reduc;
    if(c25 & lsb10) c25 ^= c10_reduc;
    mask ^= lsb10;

    uint64_t lsb11 = uint64_t_lsb(c11 & mask); // x11
    if(unlikely(!lsb11)) {
        return -1; // singular
    }

    uint64_t c11_reduc = c11 ^ lsb11;
    if(const_col & lsb11) const_col ^= c11_reduc;
    if(c12 & lsb11) c12 ^= c11_reduc;
    if(c13 & lsb11) c13 ^= c11_reduc;
    if(c14 & lsb11) c14 ^= c11_reduc;
    if(c15 & lsb11) c15 ^= c11_reduc;
    if(c16 & lsb11) c16 ^= c11_reduc;
    if(c17 & lsb11) c17 ^= c11_reduc;
    if(c18 & lsb11) c18 ^= c11_reduc;
    if(c19 & lsb11) c19 ^= c11_reduc;
    if(c20 & lsb11) c20 ^= c11_reduc;
    if(c21 & lsb11) c21 ^= c11_reduc;
    if(c22 & lsb11) c22 ^= c11_reduc;
    if(c23 & lsb11) c23 ^= c11_reduc;
    if(c24 & lsb11) c24 ^= c11_reduc;
    if(c25 & lsb11) c25 ^= c11_reduc;
    mask ^= lsb11;

    uint64_t lsb12 = uint64_t_lsb(c12 & mask); // x12
    if(unlikely(!lsb12)) {
        return -1; // singular
    }

    uint64_t c12_reduc = c12 ^ lsb12;
    if(const_col & lsb12) const_col ^= c12_reduc;
    if(c13 & lsb12) c13 ^= c12_reduc;
    if(c14 & lsb12) c14 ^= c12_reduc;
    if(c15 & lsb12) c15 ^= c12_reduc;
    if(c16 & lsb12) c16 ^= c12_reduc;
    if(c17 & lsb12) c17 ^= c12_reduc;
    if(c18 & lsb12) c18 ^= c12_reduc;
    if(c19 & lsb12) c19 ^= c12_reduc;
    if(c20 & lsb12) c20 ^= c12_reduc;
    if(c21 & lsb12) c21 ^= c12_reduc;
    if(c22 & lsb12) c22 ^= c12_reduc;
    if(c23 & lsb12) c23 ^= c12_reduc;
    if(c24 & lsb12) c24 ^= c12_reduc;
    if(c25 & lsb12) c25 ^= c12_reduc;
    mask ^= lsb12;

    uint64_t lsb13 = uint64_t_lsb(c13 & mask); // x13
    if(unlikely(!lsb13)) {
        return -1; // singular
    }

    uint64_t c13_reduc = c13 ^ lsb13;
    if(const_col & lsb13) const_col ^= c13_reduc;
    if(c14 & lsb13) c14 ^= c13_reduc;
    if(c15 & lsb13) c15 ^= c13_reduc;
    if(c16 & lsb13) c16 ^= c13_reduc;
    if(c17 & lsb13) c17 ^= c13_reduc;
    if(c18 & lsb13) c18 ^= c13_reduc;
    if(c19 & lsb13) c19 ^= c13_reduc;
    if(c20 & lsb13) c20 ^= c13_reduc;
    if(c21 & lsb13) c21 ^= c13_reduc;
    if(c22 & lsb13) c22 ^= c13_reduc;
    if(c23 & lsb13) c23 ^= c13_reduc;
    if(c24 & lsb13) c24 ^= c13_reduc;
    if(c25 & lsb13) c25 ^= c13_reduc;
    mask ^= lsb13;

    uint64_t lsb14 = uint64_t_lsb(c14 & mask); // x14
    if(unlikely(!lsb14)) {
        return -1; // singular
    }

    uint64_t c14_reduc = c14 ^ lsb14;
    if(const_col & lsb14) const_col ^= c14_reduc;
    if(c15 & lsb14) c15 ^= c14_reduc;
    if(c16 & lsb14) c16 ^= c14_reduc;
    if(c17 & lsb14) c17 ^= c14_reduc;
    if(c18 & lsb14) c18 ^= c14_reduc;
    if(c19 & lsb14) c19 ^= c14_reduc;
    if(c20 & lsb14) c20 ^= c14_reduc;
    if(c21 & lsb14) c21 ^= c14_reduc;
    if(c22 & lsb14) c22 ^= c14_reduc;
    if(c23 & lsb14) c23 ^= c14_reduc;
    if(c24 & lsb14) c24 ^= c14_reduc;
    if(c25 & lsb14) c25 ^= c14_reduc;
    mask ^= lsb14;

    uint64_t lsb15 = uint64_t_lsb(c15 & mask); // x15
    if(unlikely(!lsb15)) {
        return -1; // singular
    }

    uint64_t c15_reduc = c15 ^ lsb15;
    if(const_col & lsb15) const_col ^= c15_reduc;
    if(c16 & lsb15) c16 ^= c15_reduc;
    if(c17 & lsb15) c17 ^= c15_reduc;
    if(c18 & lsb15) c18 ^= c15_reduc;
    if(c19 & lsb15) c19 ^= c15_reduc;
    if(c20 & lsb15) c20 ^= c15_reduc;
    if(c21 & lsb15) c21 ^= c15_reduc;
    if(c22 & lsb15) c22 ^= c15_reduc;
    if(c23 & lsb15) c23 ^= c15_reduc;
    if(c24 & lsb15) c24 ^= c15_reduc;
    if(c25 & lsb15) c25 ^= c15_reduc;
    mask ^= lsb15;

    uint64_t lsb16 = uint64_t_lsb(c16 & mask); // x16
    if(unlikely(!lsb16)) {
        return -1; // singular
    }

    uint64_t c16_reduc = c16 ^ lsb16;
    if(const_col & lsb16) const_col ^= c16_reduc;
    if(c17 & lsb16) c17 ^= c16_reduc;
    if(c18 & lsb16) c18 ^= c16_reduc;
    if(c19 & lsb16) c19 ^= c16_reduc;
    if(c20 & lsb16) c20 ^= c16_reduc;
    if(c21 & lsb16) c21 ^= c16_reduc;
    if(c22 & lsb16) c22 ^= c16_reduc;
    if(c23 & lsb16) c23 ^= c16_reduc;
    if(c24 & lsb16) c24 ^= c16_reduc;
    if(c25 & lsb16) c25 ^= c16_reduc;
    mask ^= lsb16;

    uint64_t lsb17 = uint64_t_lsb(c17 & mask); // x17
    if(unlikely(!lsb17)) {
        return -1; // singular
    }

    uint64_t c17_reduc = c17 ^ lsb17;
    if(const_col & lsb17) const_col ^= c17_reduc;
    if(c18 & lsb17) c18 ^= c17_reduc;
    if(c19 & lsb17) c19 ^= c17_reduc;
    if(c20 & lsb17) c20 ^= c17_reduc;
    if(c21 & lsb17) c21 ^= c17_reduc;
    if(c22 & lsb17) c22 ^= c17_reduc;
    if(c23 & lsb17) c23 ^= c17_reduc;
    if(c24 & lsb17) c24 ^= c17_reduc;
    if(c25 & lsb17) c25 ^= c17_reduc;
    mask ^= lsb17;

    uint64_t lsb18 = uint64_t_lsb(c18 & mask); // x18
    if(unlikely(!lsb18)) {
        return -1; // singular
    }

    uint64_t c18_reduc = c18 ^ lsb18;
    if(const_col & lsb18) const_col ^= c18_reduc;
    if(c19 & lsb18) c19 ^= c18_reduc;
    if(c20 & lsb18) c20 ^= c18_reduc;
    if(c21 & lsb18) c21 ^= c18_reduc;
    if(c22 & lsb18) c22 ^= c18_reduc;
    if(c23 & lsb18) c23 ^= c18_reduc;
    if(c24 & lsb18) c24 ^= c18_reduc;
    if(c25 & lsb18) c25 ^= c18_reduc;
    mask ^= lsb18;

    uint64_t lsb19 = uint64_t_lsb(c19 & mask); // x19
    if(unlikely(!lsb19)) {
        return -1; // singular
    }

    uint64_t c19_reduc = c19 ^ lsb19;
    if(const_col & lsb19) const_col ^= c19_reduc;
    if(c20 & lsb19) c20 ^= c19_reduc;
    if(c21 & lsb19) c21 ^= c19_reduc;
    if(c22 & lsb19) c22 ^= c19_reduc;
    if(c23 & lsb19) c23 ^= c19_reduc;
    if(c24 & lsb19) c24 ^= c19_reduc;
    if(c25 & lsb19) c25 ^= c19_reduc;
    mask ^= lsb19;

    uint64_t lsb20 = uint64_t_lsb(c20 & mask); // x20
    if(unlikely(!lsb20)) {
        return -1; // singular
    }

    uint64_t c20_reduc = c20 ^ lsb20;
    if(const_col & lsb20) const_col ^= c20_reduc;
    if(c21 & lsb20) c21 ^= c20_reduc;
    if(c22 & lsb20) c22 ^= c20_reduc;
    if(c23 & lsb20) c23 ^= c20_reduc;
    if(c24 & lsb20) c24 ^= c20_reduc;
    if(c25 & lsb20) c25 ^= c20_reduc;
    mask ^= lsb20;

    uint64_t lsb21 = uint64_t_lsb(c21 & mask); // x21
    if(unlikely(!lsb21)) {
        return -1; // singular
    }

    uint64_t c21_reduc = c21 ^ lsb21;
    if(const_col & lsb21) const_col ^= c21_reduc;
    if(c22 & lsb21) c22 ^= c21_reduc;
    if(c23 & lsb21) c23 ^= c21_reduc;
    if(c24 & lsb21) c24 ^= c21_reduc;
    if(c25 & lsb21) c25 ^= c21_reduc;
    mask ^= lsb21;

    uint64_t lsb22 = uint64_t_lsb(c22 & mask); // x22
    if(unlikely(!lsb22)) {
        return -1; // singular
    }

    uint64_t c22_reduc = c22 ^ lsb22;
    if(const_col & lsb22) const_col ^= c22_reduc;
    if(c23 & lsb22) c23 ^= c22_reduc;
    if(c24 & lsb22) c24 ^= c22_reduc;
    if(c25 & lsb22) c25 ^= c22_reduc;
    mask ^= lsb22;

    uint64_t lsb23 = uint64_t_lsb(c23 & mask); // x23
    if(unlikely(!lsb23)) {
        return -1; // singular
    }

    uint64_t c23_reduc = c23 ^ lsb23;
    if(const_col & lsb23) const_col ^= c23_reduc;
    if(c24 & lsb23) c24 ^= c23_reduc;
    if(c25 & lsb23) c25 ^= c23_reduc;
    mask ^= lsb23;

    uint64_t lsb24 = uint64_t_lsb(c24 & mask); // x24
    if(unlikely(!lsb24)) {
        return -1; // singular
    }

    uint64_t c24_reduc = c24 ^ lsb24;
    if(const_col & lsb24) const_col ^= c24_reduc;
    if(c25 & lsb24) c25 ^= c24_reduc;
    mask ^= lsb24;

    uint64_t lsb25 = uint64_t_lsb(c25 & mask); // x25
    if(unlikely(!lsb25)) {
        return -1; // singular
    }

    if(const_col & lsb25) const_col ^= c25 ^ lsb25;
    mask ^= lsb25;

    if(likely(mask & const_col)) { // check system consistency
        return mask & const_col; // not solvable
    }

    uint64_t s = 0x0ULL;
    if(const_col & lsb1) s = uint64_t_toggle_at(s, 0);
    if(const_col & lsb2) s = uint64_t_toggle_at(s, 1);
    if(const_col & lsb3) s = uint64_t_toggle_at(s, 2);
    if(const_col & lsb4) s = uint64_t_toggle_at(s, 3);
    if(const_col & lsb5) s = uint64_t_toggle_at(s, 4);
    if(const_col & lsb6) s = uint64_t_toggle_at(s, 5);
    if(const_col & lsb7) s = uint64_t_toggle_at(s, 6);
    if(const_col & lsb8) s = uint64_t_toggle_at(s, 7);
    if(const_col & lsb9) s = uint64_t_toggle_at(s, 8);
    if(const_col & lsb10) s = uint64_t_toggle_at(s, 9);
    if(const_col & lsb11) s = uint64_t_toggle_at(s, 10);
    if(const_col & lsb12) s = uint64_t_toggle_at(s, 11);
    if(const_col & lsb13) s = uint64_t_toggle_at(s, 12);
    if(const_col & lsb14) s = uint64_t_toggle_at(s, 13);
    if(const_col & lsb15) s = uint64_t_toggle_at(s, 14);
    if(const_col & lsb16) s = uint64_t_toggle_at(s, 15);
    if(const_col & lsb17) s = uint64_t_toggle_at(s, 16);
    if(const_col & lsb18) s = uint64_t_toggle_at(s, 17);
    if(const_col & lsb19) s = uint64_t_toggle_at(s, 18);
    if(const_col & lsb20) s = uint64_t_toggle_at(s, 19);
    if(const_col & lsb21) s = uint64_t_toggle_at(s, 20);
    if(const_col & lsb22) s = uint64_t_toggle_at(s, 21);
    if(const_col & lsb23) s = uint64_t_toggle_at(s, 22);
    if(const_col & lsb24) s = uint64_t_toggle_at(s, 23);
    if(const_col & lsb25) s = uint64_t_toggle_at(s, 24);
    *sol = s;
    return 0;
}

int64_t
uint64a_gj_v26_generic(const uint64_t m[27], uint64_t* const restrict sol) {
    uint64_t const_col = m[0];
    uint64_t c1 = m[1]; // x1
    uint64_t c2 = m[2]; // x2
    uint64_t c3 = m[3]; // x3
    uint64_t c4 = m[4]; // x4
    uint64_t c5 = m[5]; // x5
    uint64_t c6 = m[6]; // x6
    uint64_t c7 = m[7]; // x7
    uint64_t c8 = m[8]; // x8
    uint64_t c9 = m[9]; // x9
    uint64_t c10 = m[10]; // x10
    uint64_t c11 = m[11]; // x11
    uint64_t c12 = m[12]; // x12
    uint64_t c13 = m[13]; // x13
    uint64_t c14 = m[14]; // x14
    uint64_t c15 = m[15]; // x15
    uint64_t c16 = m[16]; // x16
    uint64_t c17 = m[17]; // x17
    uint64_t c18 = m[18]; // x18
    uint64_t c19 = m[19]; // x19
    uint64_t c20 = m[20]; // x20
    uint64_t c21 = m[21]; // x21
    uint64_t c22 = m[22]; // x22
    uint64_t c23 = m[23]; // x23
    uint64_t c24 = m[24]; // x24
    uint64_t c25 = m[25]; // x25
    uint64_t c26 = m[26]; // x26
    uint64_t lsb1 = uint64_t_lsb(c1); // initially all eqs are candidates
    if(unlikely(!lsb1)) {
        return -1; // singular
    }

    uint64_t c1_reduc = c1 ^ lsb1; // for row reduction
    if(const_col & lsb1) const_col ^= c1_reduc;
    if(c2 & lsb1) c2 ^= c1_reduc;
    if(c3 & lsb1) c3 ^= c1_reduc;
    if(c4 & lsb1) c4 ^= c1_reduc;
    if(c5 & lsb1) c5 ^= c1_reduc;
    if(c6 & lsb1) c6 ^= c1_reduc;
    if(c7 & lsb1) c7 ^= c1_reduc;
    if(c8 & lsb1) c8 ^= c1_reduc;
    if(c9 & lsb1) c9 ^= c1_reduc;
    if(c10 & lsb1) c10 ^= c1_reduc;
    if(c11 & lsb1) c11 ^= c1_reduc;
    if(c12 & lsb1) c12 ^= c1_reduc;
    if(c13 & lsb1) c13 ^= c1_reduc;
    if(c14 & lsb1) c14 ^= c1_reduc;
    if(c15 & lsb1) c15 ^= c1_reduc;
    if(c16 & lsb1) c16 ^= c1_reduc;
    if(c17 & lsb1) c17 ^= c1_reduc;
    if(c18 & lsb1) c18 ^= c1_reduc;
    if(c19 & lsb1) c19 ^= c1_reduc;
    if(c20 & lsb1) c20 ^= c1_reduc;
    if(c21 & lsb1) c21 ^= c1_reduc;
    if(c22 & lsb1) c22 ^= c1_reduc;
    if(c23 & lsb1) c23 ^= c1_reduc;
    if(c24 & lsb1) c24 ^= c1_reduc;
    if(c25 & lsb1) c25 ^= c1_reduc;
    if(c26 & lsb1) c26 ^= c1_reduc;
    uint64_t mask = ~lsb1;

    uint64_t lsb2 = uint64_t_lsb(c2 & mask); // x2
    if(unlikely(!lsb2)) {
        return -1; // singular
    }

    uint64_t c2_reduc = c2 ^ lsb2;
    if(const_col & lsb2) const_col ^= c2_reduc;
    if(c3 & lsb2) c3 ^= c2_reduc;
    if(c4 & lsb2) c4 ^= c2_reduc;
    if(c5 & lsb2) c5 ^= c2_reduc;
    if(c6 & lsb2) c6 ^= c2_reduc;
    if(c7 & lsb2) c7 ^= c2_reduc;
    if(c8 & lsb2) c8 ^= c2_reduc;
    if(c9 & lsb2) c9 ^= c2_reduc;
    if(c10 & lsb2) c10 ^= c2_reduc;
    if(c11 & lsb2) c11 ^= c2_reduc;
    if(c12 & lsb2) c12 ^= c2_reduc;
    if(c13 & lsb2) c13 ^= c2_reduc;
    if(c14 & lsb2) c14 ^= c2_reduc;
    if(c15 & lsb2) c15 ^= c2_reduc;
    if(c16 & lsb2) c16 ^= c2_reduc;
    if(c17 & lsb2) c17 ^= c2_reduc;
    if(c18 & lsb2) c18 ^= c2_reduc;
    if(c19 & lsb2) c19 ^= c2_reduc;
    if(c20 & lsb2) c20 ^= c2_reduc;
    if(c21 & lsb2) c21 ^= c2_reduc;
    if(c22 & lsb2) c22 ^= c2_reduc;
    if(c23 & lsb2) c23 ^= c2_reduc;
    if(c24 & lsb2) c24 ^= c2_reduc;
    if(c25 & lsb2) c25 ^= c2_reduc;
    if(c26 & lsb2) c26 ^= c2_reduc;
    mask ^= lsb2;

    uint64_t lsb3 = uint64_t_lsb(c3 & mask); // x3
    if(unlikely(!lsb3)) {
        return -1; // singular
    }

    uint64_t c3_reduc = c3 ^ lsb3;
    if(const_col & lsb3) const_col ^= c3_reduc;
    if(c4 & lsb3) c4 ^= c3_reduc;
    if(c5 & lsb3) c5 ^= c3_reduc;
    if(c6 & lsb3) c6 ^= c3_reduc;
    if(c7 & lsb3) c7 ^= c3_reduc;
    if(c8 & lsb3) c8 ^= c3_reduc;
    if(c9 & lsb3) c9 ^= c3_reduc;
    if(c10 & lsb3) c10 ^= c3_reduc;
    if(c11 & lsb3) c11 ^= c3_reduc;
    if(c12 & lsb3) c12 ^= c3_reduc;
    if(c13 & lsb3) c13 ^= c3_reduc;
    if(c14 & lsb3) c14 ^= c3_reduc;
    if(c15 & lsb3) c15 ^= c3_reduc;
    if(c16 & lsb3) c16 ^= c3_reduc;
    if(c17 & lsb3) c17 ^= c3_reduc;
    if(c18 & lsb3) c18 ^= c3_reduc;
    if(c19 & lsb3) c19 ^= c3_reduc;
    if(c20 & lsb3) c20 ^= c3_reduc;
    if(c21 & lsb3) c21 ^= c3_reduc;
    if(c22 & lsb3) c22 ^= c3_reduc;
    if(c23 & lsb3) c23 ^= c3_reduc;
    if(c24 & lsb3) c24 ^= c3_reduc;
    if(c25 & lsb3) c25 ^= c3_reduc;
    if(c26 & lsb3) c26 ^= c3_reduc;
    mask ^= lsb3;

    uint64_t lsb4 = uint64_t_lsb(c4 & mask); // x4
    if(unlikely(!lsb4)) {
        return -1; // singular
    }

    uint64_t c4_reduc = c4 ^ lsb4;
    if(const_col & lsb4) const_col ^= c4_reduc;
    if(c5 & lsb4) c5 ^= c4_reduc;
    if(c6 & lsb4) c6 ^= c4_reduc;
    if(c7 & lsb4) c7 ^= c4_reduc;
    if(c8 & lsb4) c8 ^= c4_reduc;
    if(c9 & lsb4) c9 ^= c4_reduc;
    if(c10 & lsb4) c10 ^= c4_reduc;
    if(c11 & lsb4) c11 ^= c4_reduc;
    if(c12 & lsb4) c12 ^= c4_reduc;
    if(c13 & lsb4) c13 ^= c4_reduc;
    if(c14 & lsb4) c14 ^= c4_reduc;
    if(c15 & lsb4) c15 ^= c4_reduc;
    if(c16 & lsb4) c16 ^= c4_reduc;
    if(c17 & lsb4) c17 ^= c4_reduc;
    if(c18 & lsb4) c18 ^= c4_reduc;
    if(c19 & lsb4) c19 ^= c4_reduc;
    if(c20 & lsb4) c20 ^= c4_reduc;
    if(c21 & lsb4) c21 ^= c4_reduc;
    if(c22 & lsb4) c22 ^= c4_reduc;
    if(c23 & lsb4) c23 ^= c4_reduc;
    if(c24 & lsb4) c24 ^= c4_reduc;
    if(c25 & lsb4) c25 ^= c4_reduc;
    if(c26 & lsb4) c26 ^= c4_reduc;
    mask ^= lsb4;

    uint64_t lsb5 = uint64_t_lsb(c5 & mask); // x5
    if(unlikely(!lsb5)) {
        return -1; // singular
    }

    uint64_t c5_reduc = c5 ^ lsb5;
    if(const_col & lsb5) const_col ^= c5_reduc;
    if(c6 & lsb5) c6 ^= c5_reduc;
    if(c7 & lsb5) c7 ^= c5_reduc;
    if(c8 & lsb5) c8 ^= c5_reduc;
    if(c9 & lsb5) c9 ^= c5_reduc;
    if(c10 & lsb5) c10 ^= c5_reduc;
    if(c11 & lsb5) c11 ^= c5_reduc;
    if(c12 & lsb5) c12 ^= c5_reduc;
    if(c13 & lsb5) c13 ^= c5_reduc;
    if(c14 & lsb5) c14 ^= c5_reduc;
    if(c15 & lsb5) c15 ^= c5_reduc;
    if(c16 & lsb5) c16 ^= c5_reduc;
    if(c17 & lsb5) c17 ^= c5_reduc;
    if(c18 & lsb5) c18 ^= c5_reduc;
    if(c19 & lsb5) c19 ^= c5_reduc;
    if(c20 & lsb5) c20 ^= c5_reduc;
    if(c21 & lsb5) c21 ^= c5_reduc;
    if(c22 & lsb5) c22 ^= c5_reduc;
    if(c23 & lsb5) c23 ^= c5_reduc;
    if(c24 & lsb5) c24 ^= c5_reduc;
    if(c25 & lsb5) c25 ^= c5_reduc;
    if(c26 & lsb5) c26 ^= c5_reduc;
    mask ^= lsb5;

    uint64_t lsb6 = uint64_t_lsb(c6 & mask); // x6
    if(unlikely(!lsb6)) {
        return -1; // singular
    }

    uint64_t c6_reduc = c6 ^ lsb6;
    if(const_col & lsb6) const_col ^= c6_reduc;
    if(c7 & lsb6) c7 ^= c6_reduc;
    if(c8 & lsb6) c8 ^= c6_reduc;
    if(c9 & lsb6) c9 ^= c6_reduc;
    if(c10 & lsb6) c10 ^= c6_reduc;
    if(c11 & lsb6) c11 ^= c6_reduc;
    if(c12 & lsb6) c12 ^= c6_reduc;
    if(c13 & lsb6) c13 ^= c6_reduc;
    if(c14 & lsb6) c14 ^= c6_reduc;
    if(c15 & lsb6) c15 ^= c6_reduc;
    if(c16 & lsb6) c16 ^= c6_reduc;
    if(c17 & lsb6) c17 ^= c6_reduc;
    if(c18 & lsb6) c18 ^= c6_reduc;
    if(c19 & lsb6) c19 ^= c6_reduc;
    if(c20 & lsb6) c20 ^= c6_reduc;
    if(c21 & lsb6) c21 ^= c6_reduc;
    if(c22 & lsb6) c22 ^= c6_reduc;
    if(c23 & lsb6) c23 ^= c6_reduc;
    if(c24 & lsb6) c24 ^= c6_reduc;
    if(c25 & lsb6) c25 ^= c6_reduc;
    if(c26 & lsb6) c26 ^= c6_reduc;
    mask ^= lsb6;

    uint64_t lsb7 = uint64_t_lsb(c7 & mask); // x7
    if(unlikely(!lsb7)) {
        return -1; // singular
    }

    uint64_t c7_reduc = c7 ^ lsb7;
    if(const_col & lsb7) const_col ^= c7_reduc;
    if(c8 & lsb7) c8 ^= c7_reduc;
    if(c9 & lsb7) c9 ^= c7_reduc;
    if(c10 & lsb7) c10 ^= c7_reduc;
    if(c11 & lsb7) c11 ^= c7_reduc;
    if(c12 & lsb7) c12 ^= c7_reduc;
    if(c13 & lsb7) c13 ^= c7_reduc;
    if(c14 & lsb7) c14 ^= c7_reduc;
    if(c15 & lsb7) c15 ^= c7_reduc;
    if(c16 & lsb7) c16 ^= c7_reduc;
    if(c17 & lsb7) c17 ^= c7_reduc;
    if(c18 & lsb7) c18 ^= c7_reduc;
    if(c19 & lsb7) c19 ^= c7_reduc;
    if(c20 & lsb7) c20 ^= c7_reduc;
    if(c21 & lsb7) c21 ^= c7_reduc;
    if(c22 & lsb7) c22 ^= c7_reduc;
    if(c23 & lsb7) c23 ^= c7_reduc;
    if(c24 & lsb7) c24 ^= c7_reduc;
    if(c25 & lsb7) c25 ^= c7_reduc;
    if(c26 & lsb7) c26 ^= c7_reduc;
    mask ^= lsb7;

    uint64_t lsb8 = uint64_t_lsb(c8 & mask); // x8
    if(unlikely(!lsb8)) {
        return -1; // singular
    }

    uint64_t c8_reduc = c8 ^ lsb8;
    if(const_col & lsb8) const_col ^= c8_reduc;
    if(c9 & lsb8) c9 ^= c8_reduc;
    if(c10 & lsb8) c10 ^= c8_reduc;
    if(c11 & lsb8) c11 ^= c8_reduc;
    if(c12 & lsb8) c12 ^= c8_reduc;
    if(c13 & lsb8) c13 ^= c8_reduc;
    if(c14 & lsb8) c14 ^= c8_reduc;
    if(c15 & lsb8) c15 ^= c8_reduc;
    if(c16 & lsb8) c16 ^= c8_reduc;
    if(c17 & lsb8) c17 ^= c8_reduc;
    if(c18 & lsb8) c18 ^= c8_reduc;
    if(c19 & lsb8) c19 ^= c8_reduc;
    if(c20 & lsb8) c20 ^= c8_reduc;
    if(c21 & lsb8) c21 ^= c8_reduc;
    if(c22 & lsb8) c22 ^= c8_reduc;
    if(c23 & lsb8) c23 ^= c8_reduc;
    if(c24 & lsb8) c24 ^= c8_reduc;
    if(c25 & lsb8) c25 ^= c8_reduc;
    if(c26 & lsb8) c26 ^= c8_reduc;
    mask ^= lsb8;

    uint64_t lsb9 = uint64_t_lsb(c9 & mask); // x9
    if(unlikely(!lsb9)) {
        return -1; // singular
    }

    uint64_t c9_reduc = c9 ^ lsb9;
    if(const_col & lsb9) const_col ^= c9_reduc;
    if(c10 & lsb9) c10 ^= c9_reduc;
    if(c11 & lsb9) c11 ^= c9_reduc;
    if(c12 & lsb9) c12 ^= c9_reduc;
    if(c13 & lsb9) c13 ^= c9_reduc;
    if(c14 & lsb9) c14 ^= c9_reduc;
    if(c15 & lsb9) c15 ^= c9_reduc;
    if(c16 & lsb9) c16 ^= c9_reduc;
    if(c17 & lsb9) c17 ^= c9_reduc;
    if(c18 & lsb9) c18 ^= c9_reduc;
    if(c19 & lsb9) c19 ^= c9_reduc;
    if(c20 & lsb9) c20 ^= c9_reduc;
    if(c21 & lsb9) c21 ^= c9_reduc;
    if(c22 & lsb9) c22 ^= c9_reduc;
    if(c23 & lsb9) c23 ^= c9_reduc;
    if(c24 & lsb9) c24 ^= c9_reduc;
    if(c25 & lsb9) c25 ^= c9_reduc;
    if(c26 & lsb9) c26 ^= c9_reduc;
    mask ^= lsb9;

    uint64_t lsb10 = uint64_t_lsb(c10 & mask); // x10
    if(unlikely(!lsb10)) {
        return -1; // singular
    }

    uint64_t c10_reduc = c10 ^ lsb10;
    if(const_col & lsb10) const_col ^= c10_reduc;
    if(c11 & lsb10) c11 ^= c10_reduc;
    if(c12 & lsb10) c12 ^= c10_reduc;
    if(c13 & lsb10) c13 ^= c10_reduc;
    if(c14 & lsb10) c14 ^= c10_reduc;
    if(c15 & lsb10) c15 ^= c10_reduc;
    if(c16 & lsb10) c16 ^= c10_reduc;
    if(c17 & lsb10) c17 ^= c10_reduc;
    if(c18 & lsb10) c18 ^= c10_reduc;
    if(c19 & lsb10) c19 ^= c10_reduc;
    if(c20 & lsb10) c20 ^= c10_reduc;
    if(c21 & lsb10) c21 ^= c10_reduc;
    if(c22 & lsb10) c22 ^= c10_reduc;
    if(c23 & lsb10) c23 ^= c10_reduc;
    if(c24 & lsb10) c24 ^= c10_reduc;
    if(c25 & lsb10) c25 ^= c10_reduc;
    if(c26 & lsb10) c26 ^= c10_reduc;
    mask ^= lsb10;

    uint64_t lsb11 = uint64_t_lsb(c11 & mask); // x11
    if(unlikely(!lsb11)) {
        return -1; // singular
    }

    uint64_t c11_reduc = c11 ^ lsb11;
    if(const_col & lsb11) const_col ^= c11_reduc;
    if(c12 & lsb11) c12 ^= c11_reduc;
    if(c13 & lsb11) c13 ^= c11_reduc;
    if(c14 & lsb11) c14 ^= c11_reduc;
    if(c15 & lsb11) c15 ^= c11_reduc;
    if(c16 & lsb11) c16 ^= c11_reduc;
    if(c17 & lsb11) c17 ^= c11_reduc;
    if(c18 & lsb11) c18 ^= c11_reduc;
    if(c19 & lsb11) c19 ^= c11_reduc;
    if(c20 & lsb11) c20 ^= c11_reduc;
    if(c21 & lsb11) c21 ^= c11_reduc;
    if(c22 & lsb11) c22 ^= c11_reduc;
    if(c23 & lsb11) c23 ^= c11_reduc;
    if(c24 & lsb11) c24 ^= c11_reduc;
    if(c25 & lsb11) c25 ^= c11_reduc;
    if(c26 & lsb11) c26 ^= c11_reduc;
    mask ^= lsb11;

    uint64_t lsb12 = uint64_t_lsb(c12 & mask); // x12
    if(unlikely(!lsb12)) {
        return -1; // singular
    }

    uint64_t c12_reduc = c12 ^ lsb12;
    if(const_col & lsb12) const_col ^= c12_reduc;
    if(c13 & lsb12) c13 ^= c12_reduc;
    if(c14 & lsb12) c14 ^= c12_reduc;
    if(c15 & lsb12) c15 ^= c12_reduc;
    if(c16 & lsb12) c16 ^= c12_reduc;
    if(c17 & lsb12) c17 ^= c12_reduc;
    if(c18 & lsb12) c18 ^= c12_reduc;
    if(c19 & lsb12) c19 ^= c12_reduc;
    if(c20 & lsb12) c20 ^= c12_reduc;
    if(c21 & lsb12) c21 ^= c12_reduc;
    if(c22 & lsb12) c22 ^= c12_reduc;
    if(c23 & lsb12) c23 ^= c12_reduc;
    if(c24 & lsb12) c24 ^= c12_reduc;
    if(c25 & lsb12) c25 ^= c12_reduc;
    if(c26 & lsb12) c26 ^= c12_reduc;
    mask ^= lsb12;

    uint64_t lsb13 = uint64_t_lsb(c13 & mask); // x13
    if(unlikely(!lsb13)) {
        return -1; // singular
    }

    uint64_t c13_reduc = c13 ^ lsb13;
    if(const_col & lsb13) const_col ^= c13_reduc;
    if(c14 & lsb13) c14 ^= c13_reduc;
    if(c15 & lsb13) c15 ^= c13_reduc;
    if(c16 & lsb13) c16 ^= c13_reduc;
    if(c17 & lsb13) c17 ^= c13_reduc;
    if(c18 & lsb13) c18 ^= c13_reduc;
    if(c19 & lsb13) c19 ^= c13_reduc;
    if(c20 & lsb13) c20 ^= c13_reduc;
    if(c21 & lsb13) c21 ^= c13_reduc;
    if(c22 & lsb13) c22 ^= c13_reduc;
    if(c23 & lsb13) c23 ^= c13_reduc;
    if(c24 & lsb13) c24 ^= c13_reduc;
    if(c25 & lsb13) c25 ^= c13_reduc;
    if(c26 & lsb13) c26 ^= c13_reduc;
    mask ^= lsb13;

    uint64_t lsb14 = uint64_t_lsb(c14 & mask); // x14
    if(unlikely(!lsb14)) {
        return -1; // singular
    }

    uint64_t c14_reduc = c14 ^ lsb14;
    if(const_col & lsb14) const_col ^= c14_reduc;
    if(c15 & lsb14) c15 ^= c14_reduc;
    if(c16 & lsb14) c16 ^= c14_reduc;
    if(c17 & lsb14) c17 ^= c14_reduc;
    if(c18 & lsb14) c18 ^= c14_reduc;
    if(c19 & lsb14) c19 ^= c14_reduc;
    if(c20 & lsb14) c20 ^= c14_reduc;
    if(c21 & lsb14) c21 ^= c14_reduc;
    if(c22 & lsb14) c22 ^= c14_reduc;
    if(c23 & lsb14) c23 ^= c14_reduc;
    if(c24 & lsb14) c24 ^= c14_reduc;
    if(c25 & lsb14) c25 ^= c14_reduc;
    if(c26 & lsb14) c26 ^= c14_reduc;
    mask ^= lsb14;

    uint64_t lsb15 = uint64_t_lsb(c15 & mask); // x15
    if(unlikely(!lsb15)) {
        return -1; // singular
    }

    uint64_t c15_reduc = c15 ^ lsb15;
    if(const_col & lsb15) const_col ^= c15_reduc;
    if(c16 & lsb15) c16 ^= c15_reduc;
    if(c17 & lsb15) c17 ^= c15_reduc;
    if(c18 & lsb15) c18 ^= c15_reduc;
    if(c19 & lsb15) c19 ^= c15_reduc;
    if(c20 & lsb15) c20 ^= c15_reduc;
    if(c21 & lsb15) c21 ^= c15_reduc;
    if(c22 & lsb15) c22 ^= c15_reduc;
    if(c23 & lsb15) c23 ^= c15_reduc;
    if(c24 & lsb15) c24 ^= c15_reduc;
    if(c25 & lsb15) c25 ^= c15_reduc;
    if(c26 & lsb15) c26 ^= c15_reduc;
    mask ^= lsb15;

    uint64_t lsb16 = uint64_t_lsb(c16 & mask); // x16
    if(unlikely(!lsb16)) {
        return -1; // singular
    }

    uint64_t c16_reduc = c16 ^ lsb16;
    if(const_col & lsb16) const_col ^= c16_reduc;
    if(c17 & lsb16) c17 ^= c16_reduc;
    if(c18 & lsb16) c18 ^= c16_reduc;
    if(c19 & lsb16) c19 ^= c16_reduc;
    if(c20 & lsb16) c20 ^= c16_reduc;
    if(c21 & lsb16) c21 ^= c16_reduc;
    if(c22 & lsb16) c22 ^= c16_reduc;
    if(c23 & lsb16) c23 ^= c16_reduc;
    if(c24 & lsb16) c24 ^= c16_reduc;
    if(c25 & lsb16) c25 ^= c16_reduc;
    if(c26 & lsb16) c26 ^= c16_reduc;
    mask ^= lsb16;

    uint64_t lsb17 = uint64_t_lsb(c17 & mask); // x17
    if(unlikely(!lsb17)) {
        return -1; // singular
    }

    uint64_t c17_reduc = c17 ^ lsb17;
    if(const_col & lsb17) const_col ^= c17_reduc;
    if(c18 & lsb17) c18 ^= c17_reduc;
    if(c19 & lsb17) c19 ^= c17_reduc;
    if(c20 & lsb17) c20 ^= c17_reduc;
    if(c21 & lsb17) c21 ^= c17_reduc;
    if(c22 & lsb17) c22 ^= c17_reduc;
    if(c23 & lsb17) c23 ^= c17_reduc;
    if(c24 & lsb17) c24 ^= c17_reduc;
    if(c25 & lsb17) c25 ^= c17_reduc;
    if(c26 & lsb17) c26 ^= c17_reduc;
    mask ^= lsb17;

    uint64_t lsb18 = uint64_t_lsb(c18 & mask); // x18
    if(unlikely(!lsb18)) {
        return -1; // singular
    }

    uint64_t c18_reduc = c18 ^ lsb18;
    if(const_col & lsb18) const_col ^= c18_reduc;
    if(c19 & lsb18) c19 ^= c18_reduc;
    if(c20 & lsb18) c20 ^= c18_reduc;
    if(c21 & lsb18) c21 ^= c18_reduc;
    if(c22 & lsb18) c22 ^= c18_reduc;
    if(c23 & lsb18) c23 ^= c18_reduc;
    if(c24 & lsb18) c24 ^= c18_reduc;
    if(c25 & lsb18) c25 ^= c18_reduc;
    if(c26 & lsb18) c26 ^= c18_reduc;
    mask ^= lsb18;

    uint64_t lsb19 = uint64_t_lsb(c19 & mask); // x19
    if(unlikely(!lsb19)) {
        return -1; // singular
    }

    uint64_t c19_reduc = c19 ^ lsb19;
    if(const_col & lsb19) const_col ^= c19_reduc;
    if(c20 & lsb19) c20 ^= c19_reduc;
    if(c21 & lsb19) c21 ^= c19_reduc;
    if(c22 & lsb19) c22 ^= c19_reduc;
    if(c23 & lsb19) c23 ^= c19_reduc;
    if(c24 & lsb19) c24 ^= c19_reduc;
    if(c25 & lsb19) c25 ^= c19_reduc;
    if(c26 & lsb19) c26 ^= c19_reduc;
    mask ^= lsb19;

    uint64_t lsb20 = uint64_t_lsb(c20 & mask); // x20
    if(unlikely(!lsb20)) {
        return -1; // singular
    }

    uint64_t c20_reduc = c20 ^ lsb20;
    if(const_col & lsb20) const_col ^= c20_reduc;
    if(c21 & lsb20) c21 ^= c20_reduc;
    if(c22 & lsb20) c22 ^= c20_reduc;
    if(c23 & lsb20) c23 ^= c20_reduc;
    if(c24 & lsb20) c24 ^= c20_reduc;
    if(c25 & lsb20) c25 ^= c20_reduc;
    if(c26 & lsb20) c26 ^= c20_reduc;
    mask ^= lsb20;

    uint64_t lsb21 = uint64_t_lsb(c21 & mask); // x21
    if(unlikely(!lsb21)) {
        return -1; // singular
    }

    uint64_t c21_reduc = c21 ^ lsb21;
    if(const_col & lsb21) const_col ^= c21_reduc;
    if(c22 & lsb21) c22 ^= c21_reduc;
    if(c23 & lsb21) c23 ^= c21_reduc;
    if(c24 & lsb21) c24 ^= c21_reduc;
    if(c25 & lsb21) c25 ^= c21_reduc;
    if(c26 & lsb21) c26 ^= c21_reduc;
    mask ^= lsb21;

    uint64_t lsb22 = uint64_t_lsb(c22 & mask); // x22
    if(unlikely(!lsb22)) {
        return -1; // singular
    }

    uint64_t c22_reduc = c22 ^ lsb22;
    if(const_col & lsb22) const_col ^= c22_reduc;
    if(c23 & lsb22) c23 ^= c22_reduc;
    if(c24 & lsb22) c24 ^= c22_reduc;
    if(c25 & lsb22) c25 ^= c22_reduc;
    if(c26 & lsb22) c26 ^= c22_reduc;
    mask ^= lsb22;

    uint64_t lsb23 = uint64_t_lsb(c23 & mask); // x23
    if(unlikely(!lsb23)) {
        return -1; // singular
    }

    uint64_t c23_reduc = c23 ^ lsb23;
    if(const_col & lsb23) const_col ^= c23_reduc;
    if(c24 & lsb23) c24 ^= c23_reduc;
    if(c25 & lsb23) c25 ^= c23_reduc;
    if(c26 & lsb23) c26 ^= c23_reduc;
    mask ^= lsb23;

    uint64_t lsb24 = uint64_t_lsb(c24 & mask); // x24
    if(unlikely(!lsb24)) {
        return -1; // singular
    }

    uint64_t c24_reduc = c24 ^ lsb24;
    if(const_col & lsb24) const_col ^= c24_reduc;
    if(c25 & lsb24) c25 ^= c24_reduc;
    if(c26 & lsb24) c26 ^= c24_reduc;
    mask ^= lsb24;

    uint64_t lsb25 = uint64_t_lsb(c25 & mask); // x25
    if(unlikely(!lsb25)) {
        return -1; // singular
    }

    uint64_t c25_reduc = c25 ^ lsb25;
    if(const_col & lsb25) const_col ^= c25_reduc;
    if(c26 & lsb25) c26 ^= c25_reduc;
    mask ^= lsb25;

    uint64_t lsb26 = uint64_t_lsb(c26 & mask); // x26
    if(unlikely(!lsb26)) {
        return -1; // singular
    }

    if(const_col & lsb26) const_col ^= c26 ^ lsb26;
    mask ^= lsb26;

    if(likely(mask & const_col)) { // check system consistency
        return mask & const_col; // not solvable
    }

    uint64_t s = 0x0ULL;
    if(const_col & lsb1) s = uint64_t_toggle_at(s, 0);
    if(const_col & lsb2) s = uint64_t_toggle_at(s, 1);
    if(const_col & lsb3) s = uint64_t_toggle_at(s, 2);
    if(const_col & lsb4) s = uint64_t_toggle_at(s, 3);
    if(const_col & lsb5) s = uint64_t_toggle_at(s, 4);
    if(const_col & lsb6) s = uint64_t_toggle_at(s, 5);
    if(const_col & lsb7) s = uint64_t_toggle_at(s, 6);
    if(const_col & lsb8) s = uint64_t_toggle_at(s, 7);
    if(const_col & lsb9) s = uint64_t_toggle_at(s, 8);
    if(const_col & lsb10) s = uint64_t_toggle_at(s, 9);
    if(const_col & lsb11) s = uint64_t_toggle_at(s, 10);
    if(const_col & lsb12) s = uint64_t_toggle_at(s, 11);
    if(const_col & lsb13) s = uint64_t_toggle_at(s, 12);
    if(const_col & lsb14) s = uint64_t_toggle_at(s, 13);
    if(const_col & lsb15) s = uint64_t_toggle_at(s, 14);
    if(const_col & lsb16) s = uint64_t_toggle_at(s, 15);
    if(const_col & lsb17) s = uint64_t_toggle_at(s, 16);
    if(const_col & lsb18) s = uint64_t_toggle_at(s, 17);
    if(const_col & lsb19) s = uint64_t_toggle_at(s, 18);
    if(const_col & lsb20) s = uint64_t_toggle_at(s, 19);
    if(const_col & lsb21) s = uint64_t_toggle_at(s, 20);
    if(const_col & lsb22) s = uint64_t_toggle_at(s, 21);
    if(const_col & lsb23) s = uint64_t_toggle_at(s, 22);
    if(const_col & lsb24) s = uint64_t_toggle_at(s, 23);
    if(const_col & lsb25) s = uint64_t_toggle_at(s, 24);
    if(const_col & lsb26) s = uint64_t_toggle_at(s, 25);
    *sol = s;
    return 0;
}

int64_t
uint64a_gj_v27_generic(const uint64_t m[28], uint64_t* const restrict sol) {
    uint64_t const_col = m[0];
    uint64_t c1 = m[1]; // x1
    uint64_t c2 = m[2]; // x2
    uint64_t c3 = m[3]; // x3
    uint64_t c4 = m[4]; // x4
    uint64_t c5 = m[5]; // x5
    uint64_t c6 = m[6]; // x6
    uint64_t c7 = m[7]; // x7
    uint64_t c8 = m[8]; // x8
    uint64_t c9 = m[9]; // x9
    uint64_t c10 = m[10]; // x10
    uint64_t c11 = m[11]; // x11
    uint64_t c12 = m[12]; // x12
    uint64_t c13 = m[13]; // x13
    uint64_t c14 = m[14]; // x14
    uint64_t c15 = m[15]; // x15
    uint64_t c16 = m[16]; // x16
    uint64_t c17 = m[17]; // x17
    uint64_t c18 = m[18]; // x18
    uint64_t c19 = m[19]; // x19
    uint64_t c20 = m[20]; // x20
    uint64_t c21 = m[21]; // x21
    uint64_t c22 = m[22]; // x22
    uint64_t c23 = m[23]; // x23
    uint64_t c24 = m[24]; // x24
    uint64_t c25 = m[25]; // x25
    uint64_t c26 = m[26]; // x26
    uint64_t c27 = m[27]; // x27
    uint64_t lsb1 = uint64_t_lsb(c1); // initially all eqs are candidates
    if(unlikely(!lsb1)) {
        return -1; // singular
    }

    uint64_t c1_reduc = c1 ^ lsb1; // for row reduction
    if(const_col & lsb1) const_col ^= c1_reduc;
    if(c2 & lsb1) c2 ^= c1_reduc;
    if(c3 & lsb1) c3 ^= c1_reduc;
    if(c4 & lsb1) c4 ^= c1_reduc;
    if(c5 & lsb1) c5 ^= c1_reduc;
    if(c6 & lsb1) c6 ^= c1_reduc;
    if(c7 & lsb1) c7 ^= c1_reduc;
    if(c8 & lsb1) c8 ^= c1_reduc;
    if(c9 & lsb1) c9 ^= c1_reduc;
    if(c10 & lsb1) c10 ^= c1_reduc;
    if(c11 & lsb1) c11 ^= c1_reduc;
    if(c12 & lsb1) c12 ^= c1_reduc;
    if(c13 & lsb1) c13 ^= c1_reduc;
    if(c14 & lsb1) c14 ^= c1_reduc;
    if(c15 & lsb1) c15 ^= c1_reduc;
    if(c16 & lsb1) c16 ^= c1_reduc;
    if(c17 & lsb1) c17 ^= c1_reduc;
    if(c18 & lsb1) c18 ^= c1_reduc;
    if(c19 & lsb1) c19 ^= c1_reduc;
    if(c20 & lsb1) c20 ^= c1_reduc;
    if(c21 & lsb1) c21 ^= c1_reduc;
    if(c22 & lsb1) c22 ^= c1_reduc;
    if(c23 & lsb1) c23 ^= c1_reduc;
    if(c24 & lsb1) c24 ^= c1_reduc;
    if(c25 & lsb1) c25 ^= c1_reduc;
    if(c26 & lsb1) c26 ^= c1_reduc;
    if(c27 & lsb1) c27 ^= c1_reduc;
    uint64_t mask = ~lsb1;

    uint64_t lsb2 = uint64_t_lsb(c2 & mask); // x2
    if(unlikely(!lsb2)) {
        return -1; // singular
    }

    uint64_t c2_reduc = c2 ^ lsb2;
    if(const_col & lsb2) const_col ^= c2_reduc;
    if(c3 & lsb2) c3 ^= c2_reduc;
    if(c4 & lsb2) c4 ^= c2_reduc;
    if(c5 & lsb2) c5 ^= c2_reduc;
    if(c6 & lsb2) c6 ^= c2_reduc;
    if(c7 & lsb2) c7 ^= c2_reduc;
    if(c8 & lsb2) c8 ^= c2_reduc;
    if(c9 & lsb2) c9 ^= c2_reduc;
    if(c10 & lsb2) c10 ^= c2_reduc;
    if(c11 & lsb2) c11 ^= c2_reduc;
    if(c12 & lsb2) c12 ^= c2_reduc;
    if(c13 & lsb2) c13 ^= c2_reduc;
    if(c14 & lsb2) c14 ^= c2_reduc;
    if(c15 & lsb2) c15 ^= c2_reduc;
    if(c16 & lsb2) c16 ^= c2_reduc;
    if(c17 & lsb2) c17 ^= c2_reduc;
    if(c18 & lsb2) c18 ^= c2_reduc;
    if(c19 & lsb2) c19 ^= c2_reduc;
    if(c20 & lsb2) c20 ^= c2_reduc;
    if(c21 & lsb2) c21 ^= c2_reduc;
    if(c22 & lsb2) c22 ^= c2_reduc;
    if(c23 & lsb2) c23 ^= c2_reduc;
    if(c24 & lsb2) c24 ^= c2_reduc;
    if(c25 & lsb2) c25 ^= c2_reduc;
    if(c26 & lsb2) c26 ^= c2_reduc;
    if(c27 & lsb2) c27 ^= c2_reduc;
    mask ^= lsb2;

    uint64_t lsb3 = uint64_t_lsb(c3 & mask); // x3
    if(unlikely(!lsb3)) {
        return -1; // singular
    }

    uint64_t c3_reduc = c3 ^ lsb3;
    if(const_col & lsb3) const_col ^= c3_reduc;
    if(c4 & lsb3) c4 ^= c3_reduc;
    if(c5 & lsb3) c5 ^= c3_reduc;
    if(c6 & lsb3) c6 ^= c3_reduc;
    if(c7 & lsb3) c7 ^= c3_reduc;
    if(c8 & lsb3) c8 ^= c3_reduc;
    if(c9 & lsb3) c9 ^= c3_reduc;
    if(c10 & lsb3) c10 ^= c3_reduc;
    if(c11 & lsb3) c11 ^= c3_reduc;
    if(c12 & lsb3) c12 ^= c3_reduc;
    if(c13 & lsb3) c13 ^= c3_reduc;
    if(c14 & lsb3) c14 ^= c3_reduc;
    if(c15 & lsb3) c15 ^= c3_reduc;
    if(c16 & lsb3) c16 ^= c3_reduc;
    if(c17 & lsb3) c17 ^= c3_reduc;
    if(c18 & lsb3) c18 ^= c3_reduc;
    if(c19 & lsb3) c19 ^= c3_reduc;
    if(c20 & lsb3) c20 ^= c3_reduc;
    if(c21 & lsb3) c21 ^= c3_reduc;
    if(c22 & lsb3) c22 ^= c3_reduc;
    if(c23 & lsb3) c23 ^= c3_reduc;
    if(c24 & lsb3) c24 ^= c3_reduc;
    if(c25 & lsb3) c25 ^= c3_reduc;
    if(c26 & lsb3) c26 ^= c3_reduc;
    if(c27 & lsb3) c27 ^= c3_reduc;
    mask ^= lsb3;

    uint64_t lsb4 = uint64_t_lsb(c4 & mask); // x4
    if(unlikely(!lsb4)) {
        return -1; // singular
    }

    uint64_t c4_reduc = c4 ^ lsb4;
    if(const_col & lsb4) const_col ^= c4_reduc;
    if(c5 & lsb4) c5 ^= c4_reduc;
    if(c6 & lsb4) c6 ^= c4_reduc;
    if(c7 & lsb4) c7 ^= c4_reduc;
    if(c8 & lsb4) c8 ^= c4_reduc;
    if(c9 & lsb4) c9 ^= c4_reduc;
    if(c10 & lsb4) c10 ^= c4_reduc;
    if(c11 & lsb4) c11 ^= c4_reduc;
    if(c12 & lsb4) c12 ^= c4_reduc;
    if(c13 & lsb4) c13 ^= c4_reduc;
    if(c14 & lsb4) c14 ^= c4_reduc;
    if(c15 & lsb4) c15 ^= c4_reduc;
    if(c16 & lsb4) c16 ^= c4_reduc;
    if(c17 & lsb4) c17 ^= c4_reduc;
    if(c18 & lsb4) c18 ^= c4_reduc;
    if(c19 & lsb4) c19 ^= c4_reduc;
    if(c20 & lsb4) c20 ^= c4_reduc;
    if(c21 & lsb4) c21 ^= c4_reduc;
    if(c22 & lsb4) c22 ^= c4_reduc;
    if(c23 & lsb4) c23 ^= c4_reduc;
    if(c24 & lsb4) c24 ^= c4_reduc;
    if(c25 & lsb4) c25 ^= c4_reduc;
    if(c26 & lsb4) c26 ^= c4_reduc;
    if(c27 & lsb4) c27 ^= c4_reduc;
    mask ^= lsb4;

    uint64_t lsb5 = uint64_t_lsb(c5 & mask); // x5
    if(unlikely(!lsb5)) {
        return -1; // singular
    }

    uint64_t c5_reduc = c5 ^ lsb5;
    if(const_col & lsb5) const_col ^= c5_reduc;
    if(c6 & lsb5) c6 ^= c5_reduc;
    if(c7 & lsb5) c7 ^= c5_reduc;
    if(c8 & lsb5) c8 ^= c5_reduc;
    if(c9 & lsb5) c9 ^= c5_reduc;
    if(c10 & lsb5) c10 ^= c5_reduc;
    if(c11 & lsb5) c11 ^= c5_reduc;
    if(c12 & lsb5) c12 ^= c5_reduc;
    if(c13 & lsb5) c13 ^= c5_reduc;
    if(c14 & lsb5) c14 ^= c5_reduc;
    if(c15 & lsb5) c15 ^= c5_reduc;
    if(c16 & lsb5) c16 ^= c5_reduc;
    if(c17 & lsb5) c17 ^= c5_reduc;
    if(c18 & lsb5) c18 ^= c5_reduc;
    if(c19 & lsb5) c19 ^= c5_reduc;
    if(c20 & lsb5) c20 ^= c5_reduc;
    if(c21 & lsb5) c21 ^= c5_reduc;
    if(c22 & lsb5) c22 ^= c5_reduc;
    if(c23 & lsb5) c23 ^= c5_reduc;
    if(c24 & lsb5) c24 ^= c5_reduc;
    if(c25 & lsb5) c25 ^= c5_reduc;
    if(c26 & lsb5) c26 ^= c5_reduc;
    if(c27 & lsb5) c27 ^= c5_reduc;
    mask ^= lsb5;

    uint64_t lsb6 = uint64_t_lsb(c6 & mask); // x6
    if(unlikely(!lsb6)) {
        return -1; // singular
    }

    uint64_t c6_reduc = c6 ^ lsb6;
    if(const_col & lsb6) const_col ^= c6_reduc;
    if(c7 & lsb6) c7 ^= c6_reduc;
    if(c8 & lsb6) c8 ^= c6_reduc;
    if(c9 & lsb6) c9 ^= c6_reduc;
    if(c10 & lsb6) c10 ^= c6_reduc;
    if(c11 & lsb6) c11 ^= c6_reduc;
    if(c12 & lsb6) c12 ^= c6_reduc;
    if(c13 & lsb6) c13 ^= c6_reduc;
    if(c14 & lsb6) c14 ^= c6_reduc;
    if(c15 & lsb6) c15 ^= c6_reduc;
    if(c16 & lsb6) c16 ^= c6_reduc;
    if(c17 & lsb6) c17 ^= c6_reduc;
    if(c18 & lsb6) c18 ^= c6_reduc;
    if(c19 & lsb6) c19 ^= c6_reduc;
    if(c20 & lsb6) c20 ^= c6_reduc;
    if(c21 & lsb6) c21 ^= c6_reduc;
    if(c22 & lsb6) c22 ^= c6_reduc;
    if(c23 & lsb6) c23 ^= c6_reduc;
    if(c24 & lsb6) c24 ^= c6_reduc;
    if(c25 & lsb6) c25 ^= c6_reduc;
    if(c26 & lsb6) c26 ^= c6_reduc;
    if(c27 & lsb6) c27 ^= c6_reduc;
    mask ^= lsb6;

    uint64_t lsb7 = uint64_t_lsb(c7 & mask); // x7
    if(unlikely(!lsb7)) {
        return -1; // singular
    }

    uint64_t c7_reduc = c7 ^ lsb7;
    if(const_col & lsb7) const_col ^= c7_reduc;
    if(c8 & lsb7) c8 ^= c7_reduc;
    if(c9 & lsb7) c9 ^= c7_reduc;
    if(c10 & lsb7) c10 ^= c7_reduc;
    if(c11 & lsb7) c11 ^= c7_reduc;
    if(c12 & lsb7) c12 ^= c7_reduc;
    if(c13 & lsb7) c13 ^= c7_reduc;
    if(c14 & lsb7) c14 ^= c7_reduc;
    if(c15 & lsb7) c15 ^= c7_reduc;
    if(c16 & lsb7) c16 ^= c7_reduc;
    if(c17 & lsb7) c17 ^= c7_reduc;
    if(c18 & lsb7) c18 ^= c7_reduc;
    if(c19 & lsb7) c19 ^= c7_reduc;
    if(c20 & lsb7) c20 ^= c7_reduc;
    if(c21 & lsb7) c21 ^= c7_reduc;
    if(c22 & lsb7) c22 ^= c7_reduc;
    if(c23 & lsb7) c23 ^= c7_reduc;
    if(c24 & lsb7) c24 ^= c7_reduc;
    if(c25 & lsb7) c25 ^= c7_reduc;
    if(c26 & lsb7) c26 ^= c7_reduc;
    if(c27 & lsb7) c27 ^= c7_reduc;
    mask ^= lsb7;

    uint64_t lsb8 = uint64_t_lsb(c8 & mask); // x8
    if(unlikely(!lsb8)) {
        return -1; // singular
    }

    uint64_t c8_reduc = c8 ^ lsb8;
    if(const_col & lsb8) const_col ^= c8_reduc;
    if(c9 & lsb8) c9 ^= c8_reduc;
    if(c10 & lsb8) c10 ^= c8_reduc;
    if(c11 & lsb8) c11 ^= c8_reduc;
    if(c12 & lsb8) c12 ^= c8_reduc;
    if(c13 & lsb8) c13 ^= c8_reduc;
    if(c14 & lsb8) c14 ^= c8_reduc;
    if(c15 & lsb8) c15 ^= c8_reduc;
    if(c16 & lsb8) c16 ^= c8_reduc;
    if(c17 & lsb8) c17 ^= c8_reduc;
    if(c18 & lsb8) c18 ^= c8_reduc;
    if(c19 & lsb8) c19 ^= c8_reduc;
    if(c20 & lsb8) c20 ^= c8_reduc;
    if(c21 & lsb8) c21 ^= c8_reduc;
    if(c22 & lsb8) c22 ^= c8_reduc;
    if(c23 & lsb8) c23 ^= c8_reduc;
    if(c24 & lsb8) c24 ^= c8_reduc;
    if(c25 & lsb8) c25 ^= c8_reduc;
    if(c26 & lsb8) c26 ^= c8_reduc;
    if(c27 & lsb8) c27 ^= c8_reduc;
    mask ^= lsb8;

    uint64_t lsb9 = uint64_t_lsb(c9 & mask); // x9
    if(unlikely(!lsb9)) {
        return -1; // singular
    }

    uint64_t c9_reduc = c9 ^ lsb9;
    if(const_col & lsb9) const_col ^= c9_reduc;
    if(c10 & lsb9) c10 ^= c9_reduc;
    if(c11 & lsb9) c11 ^= c9_reduc;
    if(c12 & lsb9) c12 ^= c9_reduc;
    if(c13 & lsb9) c13 ^= c9_reduc;
    if(c14 & lsb9) c14 ^= c9_reduc;
    if(c15 & lsb9) c15 ^= c9_reduc;
    if(c16 & lsb9) c16 ^= c9_reduc;
    if(c17 & lsb9) c17 ^= c9_reduc;
    if(c18 & lsb9) c18 ^= c9_reduc;
    if(c19 & lsb9) c19 ^= c9_reduc;
    if(c20 & lsb9) c20 ^= c9_reduc;
    if(c21 & lsb9) c21 ^= c9_reduc;
    if(c22 & lsb9) c22 ^= c9_reduc;
    if(c23 & lsb9) c23 ^= c9_reduc;
    if(c24 & lsb9) c24 ^= c9_reduc;
    if(c25 & lsb9) c25 ^= c9_reduc;
    if(c26 & lsb9) c26 ^= c9_reduc;
    if(c27 & lsb9) c27 ^= c9_reduc;
    mask ^= lsb9;

    uint64_t lsb10 = uint64_t_lsb(c10 & mask); // x10
    if(unlikely(!lsb10)) {
        return -1; // singular
    }

    uint64_t c10_reduc = c10 ^ lsb10;
    if(const_col & lsb10) const_col ^= c10_reduc;
    if(c11 & lsb10) c11 ^= c10_reduc;
    if(c12 & lsb10) c12 ^= c10_reduc;
    if(c13 & lsb10) c13 ^= c10_reduc;
    if(c14 & lsb10) c14 ^= c10_reduc;
    if(c15 & lsb10) c15 ^= c10_reduc;
    if(c16 & lsb10) c16 ^= c10_reduc;
    if(c17 & lsb10) c17 ^= c10_reduc;
    if(c18 & lsb10) c18 ^= c10_reduc;
    if(c19 & lsb10) c19 ^= c10_reduc;
    if(c20 & lsb10) c20 ^= c10_reduc;
    if(c21 & lsb10) c21 ^= c10_reduc;
    if(c22 & lsb10) c22 ^= c10_reduc;
    if(c23 & lsb10) c23 ^= c10_reduc;
    if(c24 & lsb10) c24 ^= c10_reduc;
    if(c25 & lsb10) c25 ^= c10_reduc;
    if(c26 & lsb10) c26 ^= c10_reduc;
    if(c27 & lsb10) c27 ^= c10_reduc;
    mask ^= lsb10;

    uint64_t lsb11 = uint64_t_lsb(c11 & mask); // x11
    if(unlikely(!lsb11)) {
        return -1; // singular
    }

    uint64_t c11_reduc = c11 ^ lsb11;
    if(const_col & lsb11) const_col ^= c11_reduc;
    if(c12 & lsb11) c12 ^= c11_reduc;
    if(c13 & lsb11) c13 ^= c11_reduc;
    if(c14 & lsb11) c14 ^= c11_reduc;
    if(c15 & lsb11) c15 ^= c11_reduc;
    if(c16 & lsb11) c16 ^= c11_reduc;
    if(c17 & lsb11) c17 ^= c11_reduc;
    if(c18 & lsb11) c18 ^= c11_reduc;
    if(c19 & lsb11) c19 ^= c11_reduc;
    if(c20 & lsb11) c20 ^= c11_reduc;
    if(c21 & lsb11) c21 ^= c11_reduc;
    if(c22 & lsb11) c22 ^= c11_reduc;
    if(c23 & lsb11) c23 ^= c11_reduc;
    if(c24 & lsb11) c24 ^= c11_reduc;
    if(c25 & lsb11) c25 ^= c11_reduc;
    if(c26 & lsb11) c26 ^= c11_reduc;
    if(c27 & lsb11) c27 ^= c11_reduc;
    mask ^= lsb11;

    uint64_t lsb12 = uint64_t_lsb(c12 & mask); // x12
    if(unlikely(!lsb12)) {
        return -1; // singular
    }

    uint64_t c12_reduc = c12 ^ lsb12;
    if(const_col & lsb12) const_col ^= c12_reduc;
    if(c13 & lsb12) c13 ^= c12_reduc;
    if(c14 & lsb12) c14 ^= c12_reduc;
    if(c15 & lsb12) c15 ^= c12_reduc;
    if(c16 & lsb12) c16 ^= c12_reduc;
    if(c17 & lsb12) c17 ^= c12_reduc;
    if(c18 & lsb12) c18 ^= c12_reduc;
    if(c19 & lsb12) c19 ^= c12_reduc;
    if(c20 & lsb12) c20 ^= c12_reduc;
    if(c21 & lsb12) c21 ^= c12_reduc;
    if(c22 & lsb12) c22 ^= c12_reduc;
    if(c23 & lsb12) c23 ^= c12_reduc;
    if(c24 & lsb12) c24 ^= c12_reduc;
    if(c25 & lsb12) c25 ^= c12_reduc;
    if(c26 & lsb12) c26 ^= c12_reduc;
    if(c27 & lsb12) c27 ^= c12_reduc;
    mask ^= lsb12;

    uint64_t lsb13 = uint64_t_lsb(c13 & mask); // x13
    if(unlikely(!lsb13)) {
        return -1; // singular
    }

    uint64_t c13_reduc = c13 ^ lsb13;
    if(const_col & lsb13) const_col ^= c13_reduc;
    if(c14 & lsb13) c14 ^= c13_reduc;
    if(c15 & lsb13) c15 ^= c13_reduc;
    if(c16 & lsb13) c16 ^= c13_reduc;
    if(c17 & lsb13) c17 ^= c13_reduc;
    if(c18 & lsb13) c18 ^= c13_reduc;
    if(c19 & lsb13) c19 ^= c13_reduc;
    if(c20 & lsb13) c20 ^= c13_reduc;
    if(c21 & lsb13) c21 ^= c13_reduc;
    if(c22 & lsb13) c22 ^= c13_reduc;
    if(c23 & lsb13) c23 ^= c13_reduc;
    if(c24 & lsb13) c24 ^= c13_reduc;
    if(c25 & lsb13) c25 ^= c13_reduc;
    if(c26 & lsb13) c26 ^= c13_reduc;
    if(c27 & lsb13) c27 ^= c13_reduc;
    mask ^= lsb13;

    uint64_t lsb14 = uint64_t_lsb(c14 & mask); // x14
    if(unlikely(!lsb14)) {
        return -1; // singular
    }

    uint64_t c14_reduc = c14 ^ lsb14;
    if(const_col & lsb14) const_col ^= c14_reduc;
    if(c15 & lsb14) c15 ^= c14_reduc;
    if(c16 & lsb14) c16 ^= c14_reduc;
    if(c17 & lsb14) c17 ^= c14_reduc;
    if(c18 & lsb14) c18 ^= c14_reduc;
    if(c19 & lsb14) c19 ^= c14_reduc;
    if(c20 & lsb14) c20 ^= c14_reduc;
    if(c21 & lsb14) c21 ^= c14_reduc;
    if(c22 & lsb14) c22 ^= c14_reduc;
    if(c23 & lsb14) c23 ^= c14_reduc;
    if(c24 & lsb14) c24 ^= c14_reduc;
    if(c25 & lsb14) c25 ^= c14_reduc;
    if(c26 & lsb14) c26 ^= c14_reduc;
    if(c27 & lsb14) c27 ^= c14_reduc;
    mask ^= lsb14;

    uint64_t lsb15 = uint64_t_lsb(c15 & mask); // x15
    if(unlikely(!lsb15)) {
        return -1; // singular
    }

    uint64_t c15_reduc = c15 ^ lsb15;
    if(const_col & lsb15) const_col ^= c15_reduc;
    if(c16 & lsb15) c16 ^= c15_reduc;
    if(c17 & lsb15) c17 ^= c15_reduc;
    if(c18 & lsb15) c18 ^= c15_reduc;
    if(c19 & lsb15) c19 ^= c15_reduc;
    if(c20 & lsb15) c20 ^= c15_reduc;
    if(c21 & lsb15) c21 ^= c15_reduc;
    if(c22 & lsb15) c22 ^= c15_reduc;
    if(c23 & lsb15) c23 ^= c15_reduc;
    if(c24 & lsb15) c24 ^= c15_reduc;
    if(c25 & lsb15) c25 ^= c15_reduc;
    if(c26 & lsb15) c26 ^= c15_reduc;
    if(c27 & lsb15) c27 ^= c15_reduc;
    mask ^= lsb15;

    uint64_t lsb16 = uint64_t_lsb(c16 & mask); // x16
    if(unlikely(!lsb16)) {
        return -1; // singular
    }

    uint64_t c16_reduc = c16 ^ lsb16;
    if(const_col & lsb16) const_col ^= c16_reduc;
    if(c17 & lsb16) c17 ^= c16_reduc;
    if(c18 & lsb16) c18 ^= c16_reduc;
    if(c19 & lsb16) c19 ^= c16_reduc;
    if(c20 & lsb16) c20 ^= c16_reduc;
    if(c21 & lsb16) c21 ^= c16_reduc;
    if(c22 & lsb16) c22 ^= c16_reduc;
    if(c23 & lsb16) c23 ^= c16_reduc;
    if(c24 & lsb16) c24 ^= c16_reduc;
    if(c25 & lsb16) c25 ^= c16_reduc;
    if(c26 & lsb16) c26 ^= c16_reduc;
    if(c27 & lsb16) c27 ^= c16_reduc;
    mask ^= lsb16;

    uint64_t lsb17 = uint64_t_lsb(c17 & mask); // x17
    if(unlikely(!lsb17)) {
        return -1; // singular
    }

    uint64_t c17_reduc = c17 ^ lsb17;
    if(const_col & lsb17) const_col ^= c17_reduc;
    if(c18 & lsb17) c18 ^= c17_reduc;
    if(c19 & lsb17) c19 ^= c17_reduc;
    if(c20 & lsb17) c20 ^= c17_reduc;
    if(c21 & lsb17) c21 ^= c17_reduc;
    if(c22 & lsb17) c22 ^= c17_reduc;
    if(c23 & lsb17) c23 ^= c17_reduc;
    if(c24 & lsb17) c24 ^= c17_reduc;
    if(c25 & lsb17) c25 ^= c17_reduc;
    if(c26 & lsb17) c26 ^= c17_reduc;
    if(c27 & lsb17) c27 ^= c17_reduc;
    mask ^= lsb17;

    uint64_t lsb18 = uint64_t_lsb(c18 & mask); // x18
    if(unlikely(!lsb18)) {
        return -1; // singular
    }

    uint64_t c18_reduc = c18 ^ lsb18;
    if(const_col & lsb18) const_col ^= c18_reduc;
    if(c19 & lsb18) c19 ^= c18_reduc;
    if(c20 & lsb18) c20 ^= c18_reduc;
    if(c21 & lsb18) c21 ^= c18_reduc;
    if(c22 & lsb18) c22 ^= c18_reduc;
    if(c23 & lsb18) c23 ^= c18_reduc;
    if(c24 & lsb18) c24 ^= c18_reduc;
    if(c25 & lsb18) c25 ^= c18_reduc;
    if(c26 & lsb18) c26 ^= c18_reduc;
    if(c27 & lsb18) c27 ^= c18_reduc;
    mask ^= lsb18;

    uint64_t lsb19 = uint64_t_lsb(c19 & mask); // x19
    if(unlikely(!lsb19)) {
        return -1; // singular
    }

    uint64_t c19_reduc = c19 ^ lsb19;
    if(const_col & lsb19) const_col ^= c19_reduc;
    if(c20 & lsb19) c20 ^= c19_reduc;
    if(c21 & lsb19) c21 ^= c19_reduc;
    if(c22 & lsb19) c22 ^= c19_reduc;
    if(c23 & lsb19) c23 ^= c19_reduc;
    if(c24 & lsb19) c24 ^= c19_reduc;
    if(c25 & lsb19) c25 ^= c19_reduc;
    if(c26 & lsb19) c26 ^= c19_reduc;
    if(c27 & lsb19) c27 ^= c19_reduc;
    mask ^= lsb19;

    uint64_t lsb20 = uint64_t_lsb(c20 & mask); // x20
    if(unlikely(!lsb20)) {
        return -1; // singular
    }

    uint64_t c20_reduc = c20 ^ lsb20;
    if(const_col & lsb20) const_col ^= c20_reduc;
    if(c21 & lsb20) c21 ^= c20_reduc;
    if(c22 & lsb20) c22 ^= c20_reduc;
    if(c23 & lsb20) c23 ^= c20_reduc;
    if(c24 & lsb20) c24 ^= c20_reduc;
    if(c25 & lsb20) c25 ^= c20_reduc;
    if(c26 & lsb20) c26 ^= c20_reduc;
    if(c27 & lsb20) c27 ^= c20_reduc;
    mask ^= lsb20;

    uint64_t lsb21 = uint64_t_lsb(c21 & mask); // x21
    if(unlikely(!lsb21)) {
        return -1; // singular
    }

    uint64_t c21_reduc = c21 ^ lsb21;
    if(const_col & lsb21) const_col ^= c21_reduc;
    if(c22 & lsb21) c22 ^= c21_reduc;
    if(c23 & lsb21) c23 ^= c21_reduc;
    if(c24 & lsb21) c24 ^= c21_reduc;
    if(c25 & lsb21) c25 ^= c21_reduc;
    if(c26 & lsb21) c26 ^= c21_reduc;
    if(c27 & lsb21) c27 ^= c21_reduc;
    mask ^= lsb21;

    uint64_t lsb22 = uint64_t_lsb(c22 & mask); // x22
    if(unlikely(!lsb22)) {
        return -1; // singular
    }

    uint64_t c22_reduc = c22 ^ lsb22;
    if(const_col & lsb22) const_col ^= c22_reduc;
    if(c23 & lsb22) c23 ^= c22_reduc;
    if(c24 & lsb22) c24 ^= c22_reduc;
    if(c25 & lsb22) c25 ^= c22_reduc;
    if(c26 & lsb22) c26 ^= c22_reduc;
    if(c27 & lsb22) c27 ^= c22_reduc;
    mask ^= lsb22;

    uint64_t lsb23 = uint64_t_lsb(c23 & mask); // x23
    if(unlikely(!lsb23)) {
        return -1; // singular
    }

    uint64_t c23_reduc = c23 ^ lsb23;
    if(const_col & lsb23) const_col ^= c23_reduc;
    if(c24 & lsb23) c24 ^= c23_reduc;
    if(c25 & lsb23) c25 ^= c23_reduc;
    if(c26 & lsb23) c26 ^= c23_reduc;
    if(c27 & lsb23) c27 ^= c23_reduc;
    mask ^= lsb23;

    uint64_t lsb24 = uint64_t_lsb(c24 & mask); // x24
    if(unlikely(!lsb24)) {
        return -1; // singular
    }

    uint64_t c24_reduc = c24 ^ lsb24;
    if(const_col & lsb24) const_col ^= c24_reduc;
    if(c25 & lsb24) c25 ^= c24_reduc;
    if(c26 & lsb24) c26 ^= c24_reduc;
    if(c27 & lsb24) c27 ^= c24_reduc;
    mask ^= lsb24;

    uint64_t lsb25 = uint64_t_lsb(c25 & mask); // x25
    if(unlikely(!lsb25)) {
        return -1; // singular
    }

    uint64_t c25_reduc = c25 ^ lsb25;
    if(const_col & lsb25) const_col ^= c25_reduc;
    if(c26 & lsb25) c26 ^= c25_reduc;
    if(c27 & lsb25) c27 ^= c25_reduc;
    mask ^= lsb25;

    uint64_t lsb26 = uint64_t_lsb(c26 & mask); // x26
    if(unlikely(!lsb26)) {
        return -1; // singular
    }

    uint64_t c26_reduc = c26 ^ lsb26;
    if(const_col & lsb26) const_col ^= c26_reduc;
    if(c27 & lsb26) c27 ^= c26_reduc;
    mask ^= lsb26;

    uint64_t lsb27 = uint64_t_lsb(c27 & mask); // x27
    if(unlikely(!lsb27)) {
        return -1; // singular
    }

    if(const_col & lsb27) const_col ^= c27 ^ lsb27;
    mask ^= lsb27;

    if(likely(mask & const_col)) { // check system consistency
        return mask & const_col; // not solvable
    }

    uint64_t s = 0x0ULL;
    if(const_col & lsb1) s = uint64_t_toggle_at(s, 0);
    if(const_col & lsb2) s = uint64_t_toggle_at(s, 1);
    if(const_col & lsb3) s = uint64_t_toggle_at(s, 2);
    if(const_col & lsb4) s = uint64_t_toggle_at(s, 3);
    if(const_col & lsb5) s = uint64_t_toggle_at(s, 4);
    if(const_col & lsb6) s = uint64_t_toggle_at(s, 5);
    if(const_col & lsb7) s = uint64_t_toggle_at(s, 6);
    if(const_col & lsb8) s = uint64_t_toggle_at(s, 7);
    if(const_col & lsb9) s = uint64_t_toggle_at(s, 8);
    if(const_col & lsb10) s = uint64_t_toggle_at(s, 9);
    if(const_col & lsb11) s = uint64_t_toggle_at(s, 10);
    if(const_col & lsb12) s = uint64_t_toggle_at(s, 11);
    if(const_col & lsb13) s = uint64_t_toggle_at(s, 12);
    if(const_col & lsb14) s = uint64_t_toggle_at(s, 13);
    if(const_col & lsb15) s = uint64_t_toggle_at(s, 14);
    if(const_col & lsb16) s = uint64_t_toggle_at(s, 15);
    if(const_col & lsb17) s = uint64_t_toggle_at(s, 16);
    if(const_col & lsb18) s = uint64_t_toggle_at(s, 17);
    if(const_col & lsb19) s = uint64_t_toggle_at(s, 18);
    if(const_col & lsb20) s = uint64_t_toggle_at(s, 19);
    if(const_col & lsb21) s = uint64_t_toggle_at(s, 20);
    if(const_col & lsb22) s = uint64_t_toggle_at(s, 21);
    if(const_col & lsb23) s = uint64_t_toggle_at(s, 22);
    if(const_col & lsb24) s = uint64_t_toggle_at(s, 23);
    if(const_col & lsb25) s = uint64_t_toggle_at(s, 24);
    if(const_col & lsb26) s = uint64_t_toggle_at(s, 25);
    if(const_col & lsb27) s = uint64_t_toggle_at(s, 26);
    *sol = s;
    return 0;
}

int64_t
uint64a_gj_v28_generic(const uint64_t m[29], uint64_t* const restrict sol) {
    uint64_t const_col = m[0];
    uint64_t c1 = m[1]; // x1
    uint64_t c2 = m[2]; // x2
    uint64_t c3 = m[3]; // x3
    uint64_t c4 = m[4]; // x4
    uint64_t c5 = m[5]; // x5
    uint64_t c6 = m[6]; // x6
    uint64_t c7 = m[7]; // x7
    uint64_t c8 = m[8]; // x8
    uint64_t c9 = m[9]; // x9
    uint64_t c10 = m[10]; // x10
    uint64_t c11 = m[11]; // x11
    uint64_t c12 = m[12]; // x12
    uint64_t c13 = m[13]; // x13
    uint64_t c14 = m[14]; // x14
    uint64_t c15 = m[15]; // x15
    uint64_t c16 = m[16]; // x16
    uint64_t c17 = m[17]; // x17
    uint64_t c18 = m[18]; // x18
    uint64_t c19 = m[19]; // x19
    uint64_t c20 = m[20]; // x20
    uint64_t c21 = m[21]; // x21
    uint64_t c22 = m[22]; // x22
    uint64_t c23 = m[23]; // x23
    uint64_t c24 = m[24]; // x24
    uint64_t c25 = m[25]; // x25
    uint64_t c26 = m[26]; // x26
    uint64_t c27 = m[27]; // x27
    uint64_t c28 = m[28]; // x28
    uint64_t lsb1 = uint64_t_lsb(c1); // initially all eqs are candidates
    if(unlikely(!lsb1)) {
        return -1; // singular
    }

    uint64_t c1_reduc = c1 ^ lsb1; // for row reduction
    if(const_col & lsb1) const_col ^= c1_reduc;
    if(c2 & lsb1) c2 ^= c1_reduc;
    if(c3 & lsb1) c3 ^= c1_reduc;
    if(c4 & lsb1) c4 ^= c1_reduc;
    if(c5 & lsb1) c5 ^= c1_reduc;
    if(c6 & lsb1) c6 ^= c1_reduc;
    if(c7 & lsb1) c7 ^= c1_reduc;
    if(c8 & lsb1) c8 ^= c1_reduc;
    if(c9 & lsb1) c9 ^= c1_reduc;
    if(c10 & lsb1) c10 ^= c1_reduc;
    if(c11 & lsb1) c11 ^= c1_reduc;
    if(c12 & lsb1) c12 ^= c1_reduc;
    if(c13 & lsb1) c13 ^= c1_reduc;
    if(c14 & lsb1) c14 ^= c1_reduc;
    if(c15 & lsb1) c15 ^= c1_reduc;
    if(c16 & lsb1) c16 ^= c1_reduc;
    if(c17 & lsb1) c17 ^= c1_reduc;
    if(c18 & lsb1) c18 ^= c1_reduc;
    if(c19 & lsb1) c19 ^= c1_reduc;
    if(c20 & lsb1) c20 ^= c1_reduc;
    if(c21 & lsb1) c21 ^= c1_reduc;
    if(c22 & lsb1) c22 ^= c1_reduc;
    if(c23 & lsb1) c23 ^= c1_reduc;
    if(c24 & lsb1) c24 ^= c1_reduc;
    if(c25 & lsb1) c25 ^= c1_reduc;
    if(c26 & lsb1) c26 ^= c1_reduc;
    if(c27 & lsb1) c27 ^= c1_reduc;
    if(c28 & lsb1) c28 ^= c1_reduc;
    uint64_t mask = ~lsb1;

    uint64_t lsb2 = uint64_t_lsb(c2 & mask); // x2
    if(unlikely(!lsb2)) {
        return -1; // singular
    }

    uint64_t c2_reduc = c2 ^ lsb2;
    if(const_col & lsb2) const_col ^= c2_reduc;
    if(c3 & lsb2) c3 ^= c2_reduc;
    if(c4 & lsb2) c4 ^= c2_reduc;
    if(c5 & lsb2) c5 ^= c2_reduc;
    if(c6 & lsb2) c6 ^= c2_reduc;
    if(c7 & lsb2) c7 ^= c2_reduc;
    if(c8 & lsb2) c8 ^= c2_reduc;
    if(c9 & lsb2) c9 ^= c2_reduc;
    if(c10 & lsb2) c10 ^= c2_reduc;
    if(c11 & lsb2) c11 ^= c2_reduc;
    if(c12 & lsb2) c12 ^= c2_reduc;
    if(c13 & lsb2) c13 ^= c2_reduc;
    if(c14 & lsb2) c14 ^= c2_reduc;
    if(c15 & lsb2) c15 ^= c2_reduc;
    if(c16 & lsb2) c16 ^= c2_reduc;
    if(c17 & lsb2) c17 ^= c2_reduc;
    if(c18 & lsb2) c18 ^= c2_reduc;
    if(c19 & lsb2) c19 ^= c2_reduc;
    if(c20 & lsb2) c20 ^= c2_reduc;
    if(c21 & lsb2) c21 ^= c2_reduc;
    if(c22 & lsb2) c22 ^= c2_reduc;
    if(c23 & lsb2) c23 ^= c2_reduc;
    if(c24 & lsb2) c24 ^= c2_reduc;
    if(c25 & lsb2) c25 ^= c2_reduc;
    if(c26 & lsb2) c26 ^= c2_reduc;
    if(c27 & lsb2) c27 ^= c2_reduc;
    if(c28 & lsb2) c28 ^= c2_reduc;
    mask ^= lsb2;

    uint64_t lsb3 = uint64_t_lsb(c3 & mask); // x3
    if(unlikely(!lsb3)) {
        return -1; // singular
    }

    uint64_t c3_reduc = c3 ^ lsb3;
    if(const_col & lsb3) const_col ^= c3_reduc;
    if(c4 & lsb3) c4 ^= c3_reduc;
    if(c5 & lsb3) c5 ^= c3_reduc;
    if(c6 & lsb3) c6 ^= c3_reduc;
    if(c7 & lsb3) c7 ^= c3_reduc;
    if(c8 & lsb3) c8 ^= c3_reduc;
    if(c9 & lsb3) c9 ^= c3_reduc;
    if(c10 & lsb3) c10 ^= c3_reduc;
    if(c11 & lsb3) c11 ^= c3_reduc;
    if(c12 & lsb3) c12 ^= c3_reduc;
    if(c13 & lsb3) c13 ^= c3_reduc;
    if(c14 & lsb3) c14 ^= c3_reduc;
    if(c15 & lsb3) c15 ^= c3_reduc;
    if(c16 & lsb3) c16 ^= c3_reduc;
    if(c17 & lsb3) c17 ^= c3_reduc;
    if(c18 & lsb3) c18 ^= c3_reduc;
    if(c19 & lsb3) c19 ^= c3_reduc;
    if(c20 & lsb3) c20 ^= c3_reduc;
    if(c21 & lsb3) c21 ^= c3_reduc;
    if(c22 & lsb3) c22 ^= c3_reduc;
    if(c23 & lsb3) c23 ^= c3_reduc;
    if(c24 & lsb3) c24 ^= c3_reduc;
    if(c25 & lsb3) c25 ^= c3_reduc;
    if(c26 & lsb3) c26 ^= c3_reduc;
    if(c27 & lsb3) c27 ^= c3_reduc;
    if(c28 & lsb3) c28 ^= c3_reduc;
    mask ^= lsb3;

    uint64_t lsb4 = uint64_t_lsb(c4 & mask); // x4
    if(unlikely(!lsb4)) {
        return -1; // singular
    }

    uint64_t c4_reduc = c4 ^ lsb4;
    if(const_col & lsb4) const_col ^= c4_reduc;
    if(c5 & lsb4) c5 ^= c4_reduc;
    if(c6 & lsb4) c6 ^= c4_reduc;
    if(c7 & lsb4) c7 ^= c4_reduc;
    if(c8 & lsb4) c8 ^= c4_reduc;
    if(c9 & lsb4) c9 ^= c4_reduc;
    if(c10 & lsb4) c10 ^= c4_reduc;
    if(c11 & lsb4) c11 ^= c4_reduc;
    if(c12 & lsb4) c12 ^= c4_reduc;
    if(c13 & lsb4) c13 ^= c4_reduc;
    if(c14 & lsb4) c14 ^= c4_reduc;
    if(c15 & lsb4) c15 ^= c4_reduc;
    if(c16 & lsb4) c16 ^= c4_reduc;
    if(c17 & lsb4) c17 ^= c4_reduc;
    if(c18 & lsb4) c18 ^= c4_reduc;
    if(c19 & lsb4) c19 ^= c4_reduc;
    if(c20 & lsb4) c20 ^= c4_reduc;
    if(c21 & lsb4) c21 ^= c4_reduc;
    if(c22 & lsb4) c22 ^= c4_reduc;
    if(c23 & lsb4) c23 ^= c4_reduc;
    if(c24 & lsb4) c24 ^= c4_reduc;
    if(c25 & lsb4) c25 ^= c4_reduc;
    if(c26 & lsb4) c26 ^= c4_reduc;
    if(c27 & lsb4) c27 ^= c4_reduc;
    if(c28 & lsb4) c28 ^= c4_reduc;
    mask ^= lsb4;

    uint64_t lsb5 = uint64_t_lsb(c5 & mask); // x5
    if(unlikely(!lsb5)) {
        return -1; // singular
    }

    uint64_t c5_reduc = c5 ^ lsb5;
    if(const_col & lsb5) const_col ^= c5_reduc;
    if(c6 & lsb5) c6 ^= c5_reduc;
    if(c7 & lsb5) c7 ^= c5_reduc;
    if(c8 & lsb5) c8 ^= c5_reduc;
    if(c9 & lsb5) c9 ^= c5_reduc;
    if(c10 & lsb5) c10 ^= c5_reduc;
    if(c11 & lsb5) c11 ^= c5_reduc;
    if(c12 & lsb5) c12 ^= c5_reduc;
    if(c13 & lsb5) c13 ^= c5_reduc;
    if(c14 & lsb5) c14 ^= c5_reduc;
    if(c15 & lsb5) c15 ^= c5_reduc;
    if(c16 & lsb5) c16 ^= c5_reduc;
    if(c17 & lsb5) c17 ^= c5_reduc;
    if(c18 & lsb5) c18 ^= c5_reduc;
    if(c19 & lsb5) c19 ^= c5_reduc;
    if(c20 & lsb5) c20 ^= c5_reduc;
    if(c21 & lsb5) c21 ^= c5_reduc;
    if(c22 & lsb5) c22 ^= c5_reduc;
    if(c23 & lsb5) c23 ^= c5_reduc;
    if(c24 & lsb5) c24 ^= c5_reduc;
    if(c25 & lsb5) c25 ^= c5_reduc;
    if(c26 & lsb5) c26 ^= c5_reduc;
    if(c27 & lsb5) c27 ^= c5_reduc;
    if(c28 & lsb5) c28 ^= c5_reduc;
    mask ^= lsb5;

    uint64_t lsb6 = uint64_t_lsb(c6 & mask); // x6
    if(unlikely(!lsb6)) {
        return -1; // singular
    }

    uint64_t c6_reduc = c6 ^ lsb6;
    if(const_col & lsb6) const_col ^= c6_reduc;
    if(c7 & lsb6) c7 ^= c6_reduc;
    if(c8 & lsb6) c8 ^= c6_reduc;
    if(c9 & lsb6) c9 ^= c6_reduc;
    if(c10 & lsb6) c10 ^= c6_reduc;
    if(c11 & lsb6) c11 ^= c6_reduc;
    if(c12 & lsb6) c12 ^= c6_reduc;
    if(c13 & lsb6) c13 ^= c6_reduc;
    if(c14 & lsb6) c14 ^= c6_reduc;
    if(c15 & lsb6) c15 ^= c6_reduc;
    if(c16 & lsb6) c16 ^= c6_reduc;
    if(c17 & lsb6) c17 ^= c6_reduc;
    if(c18 & lsb6) c18 ^= c6_reduc;
    if(c19 & lsb6) c19 ^= c6_reduc;
    if(c20 & lsb6) c20 ^= c6_reduc;
    if(c21 & lsb6) c21 ^= c6_reduc;
    if(c22 & lsb6) c22 ^= c6_reduc;
    if(c23 & lsb6) c23 ^= c6_reduc;
    if(c24 & lsb6) c24 ^= c6_reduc;
    if(c25 & lsb6) c25 ^= c6_reduc;
    if(c26 & lsb6) c26 ^= c6_reduc;
    if(c27 & lsb6) c27 ^= c6_reduc;
    if(c28 & lsb6) c28 ^= c6_reduc;
    mask ^= lsb6;

    uint64_t lsb7 = uint64_t_lsb(c7 & mask); // x7
    if(unlikely(!lsb7)) {
        return -1; // singular
    }

    uint64_t c7_reduc = c7 ^ lsb7;
    if(const_col & lsb7) const_col ^= c7_reduc;
    if(c8 & lsb7) c8 ^= c7_reduc;
    if(c9 & lsb7) c9 ^= c7_reduc;
    if(c10 & lsb7) c10 ^= c7_reduc;
    if(c11 & lsb7) c11 ^= c7_reduc;
    if(c12 & lsb7) c12 ^= c7_reduc;
    if(c13 & lsb7) c13 ^= c7_reduc;
    if(c14 & lsb7) c14 ^= c7_reduc;
    if(c15 & lsb7) c15 ^= c7_reduc;
    if(c16 & lsb7) c16 ^= c7_reduc;
    if(c17 & lsb7) c17 ^= c7_reduc;
    if(c18 & lsb7) c18 ^= c7_reduc;
    if(c19 & lsb7) c19 ^= c7_reduc;
    if(c20 & lsb7) c20 ^= c7_reduc;
    if(c21 & lsb7) c21 ^= c7_reduc;
    if(c22 & lsb7) c22 ^= c7_reduc;
    if(c23 & lsb7) c23 ^= c7_reduc;
    if(c24 & lsb7) c24 ^= c7_reduc;
    if(c25 & lsb7) c25 ^= c7_reduc;
    if(c26 & lsb7) c26 ^= c7_reduc;
    if(c27 & lsb7) c27 ^= c7_reduc;
    if(c28 & lsb7) c28 ^= c7_reduc;
    mask ^= lsb7;

    uint64_t lsb8 = uint64_t_lsb(c8 & mask); // x8
    if(unlikely(!lsb8)) {
        return -1; // singular
    }

    uint64_t c8_reduc = c8 ^ lsb8;
    if(const_col & lsb8) const_col ^= c8_reduc;
    if(c9 & lsb8) c9 ^= c8_reduc;
    if(c10 & lsb8) c10 ^= c8_reduc;
    if(c11 & lsb8) c11 ^= c8_reduc;
    if(c12 & lsb8) c12 ^= c8_reduc;
    if(c13 & lsb8) c13 ^= c8_reduc;
    if(c14 & lsb8) c14 ^= c8_reduc;
    if(c15 & lsb8) c15 ^= c8_reduc;
    if(c16 & lsb8) c16 ^= c8_reduc;
    if(c17 & lsb8) c17 ^= c8_reduc;
    if(c18 & lsb8) c18 ^= c8_reduc;
    if(c19 & lsb8) c19 ^= c8_reduc;
    if(c20 & lsb8) c20 ^= c8_reduc;
    if(c21 & lsb8) c21 ^= c8_reduc;
    if(c22 & lsb8) c22 ^= c8_reduc;
    if(c23 & lsb8) c23 ^= c8_reduc;
    if(c24 & lsb8) c24 ^= c8_reduc;
    if(c25 & lsb8) c25 ^= c8_reduc;
    if(c26 & lsb8) c26 ^= c8_reduc;
    if(c27 & lsb8) c27 ^= c8_reduc;
    if(c28 & lsb8) c28 ^= c8_reduc;
    mask ^= lsb8;

    uint64_t lsb9 = uint64_t_lsb(c9 & mask); // x9
    if(unlikely(!lsb9)) {
        return -1; // singular
    }

    uint64_t c9_reduc = c9 ^ lsb9;
    if(const_col & lsb9) const_col ^= c9_reduc;
    if(c10 & lsb9) c10 ^= c9_reduc;
    if(c11 & lsb9) c11 ^= c9_reduc;
    if(c12 & lsb9) c12 ^= c9_reduc;
    if(c13 & lsb9) c13 ^= c9_reduc;
    if(c14 & lsb9) c14 ^= c9_reduc;
    if(c15 & lsb9) c15 ^= c9_reduc;
    if(c16 & lsb9) c16 ^= c9_reduc;
    if(c17 & lsb9) c17 ^= c9_reduc;
    if(c18 & lsb9) c18 ^= c9_reduc;
    if(c19 & lsb9) c19 ^= c9_reduc;
    if(c20 & lsb9) c20 ^= c9_reduc;
    if(c21 & lsb9) c21 ^= c9_reduc;
    if(c22 & lsb9) c22 ^= c9_reduc;
    if(c23 & lsb9) c23 ^= c9_reduc;
    if(c24 & lsb9) c24 ^= c9_reduc;
    if(c25 & lsb9) c25 ^= c9_reduc;
    if(c26 & lsb9) c26 ^= c9_reduc;
    if(c27 & lsb9) c27 ^= c9_reduc;
    if(c28 & lsb9) c28 ^= c9_reduc;
    mask ^= lsb9;

    uint64_t lsb10 = uint64_t_lsb(c10 & mask); // x10
    if(unlikely(!lsb10)) {
        return -1; // singular
    }

    uint64_t c10_reduc = c10 ^ lsb10;
    if(const_col & lsb10) const_col ^= c10_reduc;
    if(c11 & lsb10) c11 ^= c10_reduc;
    if(c12 & lsb10) c12 ^= c10_reduc;
    if(c13 & lsb10) c13 ^= c10_reduc;
    if(c14 & lsb10) c14 ^= c10_reduc;
    if(c15 & lsb10) c15 ^= c10_reduc;
    if(c16 & lsb10) c16 ^= c10_reduc;
    if(c17 & lsb10) c17 ^= c10_reduc;
    if(c18 & lsb10) c18 ^= c10_reduc;
    if(c19 & lsb10) c19 ^= c10_reduc;
    if(c20 & lsb10) c20 ^= c10_reduc;
    if(c21 & lsb10) c21 ^= c10_reduc;
    if(c22 & lsb10) c22 ^= c10_reduc;
    if(c23 & lsb10) c23 ^= c10_reduc;
    if(c24 & lsb10) c24 ^= c10_reduc;
    if(c25 & lsb10) c25 ^= c10_reduc;
    if(c26 & lsb10) c26 ^= c10_reduc;
    if(c27 & lsb10) c27 ^= c10_reduc;
    if(c28 & lsb10) c28 ^= c10_reduc;
    mask ^= lsb10;

    uint64_t lsb11 = uint64_t_lsb(c11 & mask); // x11
    if(unlikely(!lsb11)) {
        return -1; // singular
    }

    uint64_t c11_reduc = c11 ^ lsb11;
    if(const_col & lsb11) const_col ^= c11_reduc;
    if(c12 & lsb11) c12 ^= c11_reduc;
    if(c13 & lsb11) c13 ^= c11_reduc;
    if(c14 & lsb11) c14 ^= c11_reduc;
    if(c15 & lsb11) c15 ^= c11_reduc;
    if(c16 & lsb11) c16 ^= c11_reduc;
    if(c17 & lsb11) c17 ^= c11_reduc;
    if(c18 & lsb11) c18 ^= c11_reduc;
    if(c19 & lsb11) c19 ^= c11_reduc;
    if(c20 & lsb11) c20 ^= c11_reduc;
    if(c21 & lsb11) c21 ^= c11_reduc;
    if(c22 & lsb11) c22 ^= c11_reduc;
    if(c23 & lsb11) c23 ^= c11_reduc;
    if(c24 & lsb11) c24 ^= c11_reduc;
    if(c25 & lsb11) c25 ^= c11_reduc;
    if(c26 & lsb11) c26 ^= c11_reduc;
    if(c27 & lsb11) c27 ^= c11_reduc;
    if(c28 & lsb11) c28 ^= c11_reduc;
    mask ^= lsb11;

    uint64_t lsb12 = uint64_t_lsb(c12 & mask); // x12
    if(unlikely(!lsb12)) {
        return -1; // singular
    }

    uint64_t c12_reduc = c12 ^ lsb12;
    if(const_col & lsb12) const_col ^= c12_reduc;
    if(c13 & lsb12) c13 ^= c12_reduc;
    if(c14 & lsb12) c14 ^= c12_reduc;
    if(c15 & lsb12) c15 ^= c12_reduc;
    if(c16 & lsb12) c16 ^= c12_reduc;
    if(c17 & lsb12) c17 ^= c12_reduc;
    if(c18 & lsb12) c18 ^= c12_reduc;
    if(c19 & lsb12) c19 ^= c12_reduc;
    if(c20 & lsb12) c20 ^= c12_reduc;
    if(c21 & lsb12) c21 ^= c12_reduc;
    if(c22 & lsb12) c22 ^= c12_reduc;
    if(c23 & lsb12) c23 ^= c12_reduc;
    if(c24 & lsb12) c24 ^= c12_reduc;
    if(c25 & lsb12) c25 ^= c12_reduc;
    if(c26 & lsb12) c26 ^= c12_reduc;
    if(c27 & lsb12) c27 ^= c12_reduc;
    if(c28 & lsb12) c28 ^= c12_reduc;
    mask ^= lsb12;

    uint64_t lsb13 = uint64_t_lsb(c13 & mask); // x13
    if(unlikely(!lsb13)) {
        return -1; // singular
    }

    uint64_t c13_reduc = c13 ^ lsb13;
    if(const_col & lsb13) const_col ^= c13_reduc;
    if(c14 & lsb13) c14 ^= c13_reduc;
    if(c15 & lsb13) c15 ^= c13_reduc;
    if(c16 & lsb13) c16 ^= c13_reduc;
    if(c17 & lsb13) c17 ^= c13_reduc;
    if(c18 & lsb13) c18 ^= c13_reduc;
    if(c19 & lsb13) c19 ^= c13_reduc;
    if(c20 & lsb13) c20 ^= c13_reduc;
    if(c21 & lsb13) c21 ^= c13_reduc;
    if(c22 & lsb13) c22 ^= c13_reduc;
    if(c23 & lsb13) c23 ^= c13_reduc;
    if(c24 & lsb13) c24 ^= c13_reduc;
    if(c25 & lsb13) c25 ^= c13_reduc;
    if(c26 & lsb13) c26 ^= c13_reduc;
    if(c27 & lsb13) c27 ^= c13_reduc;
    if(c28 & lsb13) c28 ^= c13_reduc;
    mask ^= lsb13;

    uint64_t lsb14 = uint64_t_lsb(c14 & mask); // x14
    if(unlikely(!lsb14)) {
        return -1; // singular
    }

    uint64_t c14_reduc = c14 ^ lsb14;
    if(const_col & lsb14) const_col ^= c14_reduc;
    if(c15 & lsb14) c15 ^= c14_reduc;
    if(c16 & lsb14) c16 ^= c14_reduc;
    if(c17 & lsb14) c17 ^= c14_reduc;
    if(c18 & lsb14) c18 ^= c14_reduc;
    if(c19 & lsb14) c19 ^= c14_reduc;
    if(c20 & lsb14) c20 ^= c14_reduc;
    if(c21 & lsb14) c21 ^= c14_reduc;
    if(c22 & lsb14) c22 ^= c14_reduc;
    if(c23 & lsb14) c23 ^= c14_reduc;
    if(c24 & lsb14) c24 ^= c14_reduc;
    if(c25 & lsb14) c25 ^= c14_reduc;
    if(c26 & lsb14) c26 ^= c14_reduc;
    if(c27 & lsb14) c27 ^= c14_reduc;
    if(c28 & lsb14) c28 ^= c14_reduc;
    mask ^= lsb14;

    uint64_t lsb15 = uint64_t_lsb(c15 & mask); // x15
    if(unlikely(!lsb15)) {
        return -1; // singular
    }

    uint64_t c15_reduc = c15 ^ lsb15;
    if(const_col & lsb15) const_col ^= c15_reduc;
    if(c16 & lsb15) c16 ^= c15_reduc;
    if(c17 & lsb15) c17 ^= c15_reduc;
    if(c18 & lsb15) c18 ^= c15_reduc;
    if(c19 & lsb15) c19 ^= c15_reduc;
    if(c20 & lsb15) c20 ^= c15_reduc;
    if(c21 & lsb15) c21 ^= c15_reduc;
    if(c22 & lsb15) c22 ^= c15_reduc;
    if(c23 & lsb15) c23 ^= c15_reduc;
    if(c24 & lsb15) c24 ^= c15_reduc;
    if(c25 & lsb15) c25 ^= c15_reduc;
    if(c26 & lsb15) c26 ^= c15_reduc;
    if(c27 & lsb15) c27 ^= c15_reduc;
    if(c28 & lsb15) c28 ^= c15_reduc;
    mask ^= lsb15;

    uint64_t lsb16 = uint64_t_lsb(c16 & mask); // x16
    if(unlikely(!lsb16)) {
        return -1; // singular
    }

    uint64_t c16_reduc = c16 ^ lsb16;
    if(const_col & lsb16) const_col ^= c16_reduc;
    if(c17 & lsb16) c17 ^= c16_reduc;
    if(c18 & lsb16) c18 ^= c16_reduc;
    if(c19 & lsb16) c19 ^= c16_reduc;
    if(c20 & lsb16) c20 ^= c16_reduc;
    if(c21 & lsb16) c21 ^= c16_reduc;
    if(c22 & lsb16) c22 ^= c16_reduc;
    if(c23 & lsb16) c23 ^= c16_reduc;
    if(c24 & lsb16) c24 ^= c16_reduc;
    if(c25 & lsb16) c25 ^= c16_reduc;
    if(c26 & lsb16) c26 ^= c16_reduc;
    if(c27 & lsb16) c27 ^= c16_reduc;
    if(c28 & lsb16) c28 ^= c16_reduc;
    mask ^= lsb16;

    uint64_t lsb17 = uint64_t_lsb(c17 & mask); // x17
    if(unlikely(!lsb17)) {
        return -1; // singular
    }

    uint64_t c17_reduc = c17 ^ lsb17;
    if(const_col & lsb17) const_col ^= c17_reduc;
    if(c18 & lsb17) c18 ^= c17_reduc;
    if(c19 & lsb17) c19 ^= c17_reduc;
    if(c20 & lsb17) c20 ^= c17_reduc;
    if(c21 & lsb17) c21 ^= c17_reduc;
    if(c22 & lsb17) c22 ^= c17_reduc;
    if(c23 & lsb17) c23 ^= c17_reduc;
    if(c24 & lsb17) c24 ^= c17_reduc;
    if(c25 & lsb17) c25 ^= c17_reduc;
    if(c26 & lsb17) c26 ^= c17_reduc;
    if(c27 & lsb17) c27 ^= c17_reduc;
    if(c28 & lsb17) c28 ^= c17_reduc;
    mask ^= lsb17;

    uint64_t lsb18 = uint64_t_lsb(c18 & mask); // x18
    if(unlikely(!lsb18)) {
        return -1; // singular
    }

    uint64_t c18_reduc = c18 ^ lsb18;
    if(const_col & lsb18) const_col ^= c18_reduc;
    if(c19 & lsb18) c19 ^= c18_reduc;
    if(c20 & lsb18) c20 ^= c18_reduc;
    if(c21 & lsb18) c21 ^= c18_reduc;
    if(c22 & lsb18) c22 ^= c18_reduc;
    if(c23 & lsb18) c23 ^= c18_reduc;
    if(c24 & lsb18) c24 ^= c18_reduc;
    if(c25 & lsb18) c25 ^= c18_reduc;
    if(c26 & lsb18) c26 ^= c18_reduc;
    if(c27 & lsb18) c27 ^= c18_reduc;
    if(c28 & lsb18) c28 ^= c18_reduc;
    mask ^= lsb18;

    uint64_t lsb19 = uint64_t_lsb(c19 & mask); // x19
    if(unlikely(!lsb19)) {
        return -1; // singular
    }

    uint64_t c19_reduc = c19 ^ lsb19;
    if(const_col & lsb19) const_col ^= c19_reduc;
    if(c20 & lsb19) c20 ^= c19_reduc;
    if(c21 & lsb19) c21 ^= c19_reduc;
    if(c22 & lsb19) c22 ^= c19_reduc;
    if(c23 & lsb19) c23 ^= c19_reduc;
    if(c24 & lsb19) c24 ^= c19_reduc;
    if(c25 & lsb19) c25 ^= c19_reduc;
    if(c26 & lsb19) c26 ^= c19_reduc;
    if(c27 & lsb19) c27 ^= c19_reduc;
    if(c28 & lsb19) c28 ^= c19_reduc;
    mask ^= lsb19;

    uint64_t lsb20 = uint64_t_lsb(c20 & mask); // x20
    if(unlikely(!lsb20)) {
        return -1; // singular
    }

    uint64_t c20_reduc = c20 ^ lsb20;
    if(const_col & lsb20) const_col ^= c20_reduc;
    if(c21 & lsb20) c21 ^= c20_reduc;
    if(c22 & lsb20) c22 ^= c20_reduc;
    if(c23 & lsb20) c23 ^= c20_reduc;
    if(c24 & lsb20) c24 ^= c20_reduc;
    if(c25 & lsb20) c25 ^= c20_reduc;
    if(c26 & lsb20) c26 ^= c20_reduc;
    if(c27 & lsb20) c27 ^= c20_reduc;
    if(c28 & lsb20) c28 ^= c20_reduc;
    mask ^= lsb20;

    uint64_t lsb21 = uint64_t_lsb(c21 & mask); // x21
    if(unlikely(!lsb21)) {
        return -1; // singular
    }

    uint64_t c21_reduc = c21 ^ lsb21;
    if(const_col & lsb21) const_col ^= c21_reduc;
    if(c22 & lsb21) c22 ^= c21_reduc;
    if(c23 & lsb21) c23 ^= c21_reduc;
    if(c24 & lsb21) c24 ^= c21_reduc;
    if(c25 & lsb21) c25 ^= c21_reduc;
    if(c26 & lsb21) c26 ^= c21_reduc;
    if(c27 & lsb21) c27 ^= c21_reduc;
    if(c28 & lsb21) c28 ^= c21_reduc;
    mask ^= lsb21;

    uint64_t lsb22 = uint64_t_lsb(c22 & mask); // x22
    if(unlikely(!lsb22)) {
        return -1; // singular
    }

    uint64_t c22_reduc = c22 ^ lsb22;
    if(const_col & lsb22) const_col ^= c22_reduc;
    if(c23 & lsb22) c23 ^= c22_reduc;
    if(c24 & lsb22) c24 ^= c22_reduc;
    if(c25 & lsb22) c25 ^= c22_reduc;
    if(c26 & lsb22) c26 ^= c22_reduc;
    if(c27 & lsb22) c27 ^= c22_reduc;
    if(c28 & lsb22) c28 ^= c22_reduc;
    mask ^= lsb22;

    uint64_t lsb23 = uint64_t_lsb(c23 & mask); // x23
    if(unlikely(!lsb23)) {
        return -1; // singular
    }

    uint64_t c23_reduc = c23 ^ lsb23;
    if(const_col & lsb23) const_col ^= c23_reduc;
    if(c24 & lsb23) c24 ^= c23_reduc;
    if(c25 & lsb23) c25 ^= c23_reduc;
    if(c26 & lsb23) c26 ^= c23_reduc;
    if(c27 & lsb23) c27 ^= c23_reduc;
    if(c28 & lsb23) c28 ^= c23_reduc;
    mask ^= lsb23;

    uint64_t lsb24 = uint64_t_lsb(c24 & mask); // x24
    if(unlikely(!lsb24)) {
        return -1; // singular
    }

    uint64_t c24_reduc = c24 ^ lsb24;
    if(const_col & lsb24) const_col ^= c24_reduc;
    if(c25 & lsb24) c25 ^= c24_reduc;
    if(c26 & lsb24) c26 ^= c24_reduc;
    if(c27 & lsb24) c27 ^= c24_reduc;
    if(c28 & lsb24) c28 ^= c24_reduc;
    mask ^= lsb24;

    uint64_t lsb25 = uint64_t_lsb(c25 & mask); // x25
    if(unlikely(!lsb25)) {
        return -1; // singular
    }

    uint64_t c25_reduc = c25 ^ lsb25;
    if(const_col & lsb25) const_col ^= c25_reduc;
    if(c26 & lsb25) c26 ^= c25_reduc;
    if(c27 & lsb25) c27 ^= c25_reduc;
    if(c28 & lsb25) c28 ^= c25_reduc;
    mask ^= lsb25;

    uint64_t lsb26 = uint64_t_lsb(c26 & mask); // x26
    if(unlikely(!lsb26)) {
        return -1; // singular
    }

    uint64_t c26_reduc = c26 ^ lsb26;
    if(const_col & lsb26) const_col ^= c26_reduc;
    if(c27 & lsb26) c27 ^= c26_reduc;
    if(c28 & lsb26) c28 ^= c26_reduc;
    mask ^= lsb26;

    uint64_t lsb27 = uint64_t_lsb(c27 & mask); // x27
    if(unlikely(!lsb27)) {
        return -1; // singular
    }

    uint64_t c27_reduc = c27 ^ lsb27;
    if(const_col & lsb27) const_col ^= c27_reduc;
    if(c28 & lsb27) c28 ^= c27_reduc;
    mask ^= lsb27;

    uint64_t lsb28 = uint64_t_lsb(c28 & mask); // x28
    if(unlikely(!lsb28)) {
        return -1; // singular
    }

    if(const_col & lsb28) const_col ^= c28 ^ lsb28;
    mask ^= lsb28;

    if(likely(mask & const_col)) { // check system consistency
        return mask & const_col; // not solvable
    }

    uint64_t s = 0x0ULL;
    if(const_col & lsb1) s = uint64_t_toggle_at(s, 0);
    if(const_col & lsb2) s = uint64_t_toggle_at(s, 1);
    if(const_col & lsb3) s = uint64_t_toggle_at(s, 2);
    if(const_col & lsb4) s = uint64_t_toggle_at(s, 3);
    if(const_col & lsb5) s = uint64_t_toggle_at(s, 4);
    if(const_col & lsb6) s = uint64_t_toggle_at(s, 5);
    if(const_col & lsb7) s = uint64_t_toggle_at(s, 6);
    if(const_col & lsb8) s = uint64_t_toggle_at(s, 7);
    if(const_col & lsb9) s = uint64_t_toggle_at(s, 8);
    if(const_col & lsb10) s = uint64_t_toggle_at(s, 9);
    if(const_col & lsb11) s = uint64_t_toggle_at(s, 10);
    if(const_col & lsb12) s = uint64_t_toggle_at(s, 11);
    if(const_col & lsb13) s = uint64_t_toggle_at(s, 12);
    if(const_col & lsb14) s = uint64_t_toggle_at(s, 13);
    if(const_col & lsb15) s = uint64_t_toggle_at(s, 14);
    if(const_col & lsb16) s = uint64_t_toggle_at(s, 15);
    if(const_col & lsb17) s = uint64_t_toggle_at(s, 16);
    if(const_col & lsb18) s = uint64_t_toggle_at(s, 17);
    if(const_col & lsb19) s = uint64_t_toggle_at(s, 18);
    if(const_col & lsb20) s = uint64_t_toggle_at(s, 19);
    if(const_col & lsb21) s = uint64_t_toggle_at(s, 20);
    if(const_col & lsb22) s = uint64_t_toggle_at(s, 21);
    if(const_col & lsb23) s = uint64_t_toggle_at(s, 22);
    if(const_col & lsb24) s = uint64_t_toggle_at(s, 23);
    if(const_col & lsb25) s = uint64_t_toggle_at(s, 24);
    if(const_col & lsb26) s = uint64_t_toggle_at(s, 25);
    if(const_col & lsb27) s = uint64_t_toggle_at(s, 26);
    if(const_col & lsb28) s = uint64_t_toggle_at(s, 27);
    *sol = s;
    return 0;
}

int64_t
uint64a_gj_v29_generic(const uint64_t m[30], uint64_t* const restrict sol) {
    uint64_t const_col = m[0];
    uint64_t c1 = m[1]; // x1
    uint64_t c2 = m[2]; // x2
    uint64_t c3 = m[3]; // x3
    uint64_t c4 = m[4]; // x4
    uint64_t c5 = m[5]; // x5
    uint64_t c6 = m[6]; // x6
    uint64_t c7 = m[7]; // x7
    uint64_t c8 = m[8]; // x8
    uint64_t c9 = m[9]; // x9
    uint64_t c10 = m[10]; // x10
    uint64_t c11 = m[11]; // x11
    uint64_t c12 = m[12]; // x12
    uint64_t c13 = m[13]; // x13
    uint64_t c14 = m[14]; // x14
    uint64_t c15 = m[15]; // x15
    uint64_t c16 = m[16]; // x16
    uint64_t c17 = m[17]; // x17
    uint64_t c18 = m[18]; // x18
    uint64_t c19 = m[19]; // x19
    uint64_t c20 = m[20]; // x20
    uint64_t c21 = m[21]; // x21
    uint64_t c22 = m[22]; // x22
    uint64_t c23 = m[23]; // x23
    uint64_t c24 = m[24]; // x24
    uint64_t c25 = m[25]; // x25
    uint64_t c26 = m[26]; // x26
    uint64_t c27 = m[27]; // x27
    uint64_t c28 = m[28]; // x28
    uint64_t c29 = m[29]; // x29
    uint64_t lsb1 = uint64_t_lsb(c1); // initially all eqs are candidates
    if(unlikely(!lsb1)) {
        return -1; // singular
    }

    uint64_t c1_reduc = c1 ^ lsb1; // for row reduction
    if(const_col & lsb1) const_col ^= c1_reduc;
    if(c2 & lsb1) c2 ^= c1_reduc;
    if(c3 & lsb1) c3 ^= c1_reduc;
    if(c4 & lsb1) c4 ^= c1_reduc;
    if(c5 & lsb1) c5 ^= c1_reduc;
    if(c6 & lsb1) c6 ^= c1_reduc;
    if(c7 & lsb1) c7 ^= c1_reduc;
    if(c8 & lsb1) c8 ^= c1_reduc;
    if(c9 & lsb1) c9 ^= c1_reduc;
    if(c10 & lsb1) c10 ^= c1_reduc;
    if(c11 & lsb1) c11 ^= c1_reduc;
    if(c12 & lsb1) c12 ^= c1_reduc;
    if(c13 & lsb1) c13 ^= c1_reduc;
    if(c14 & lsb1) c14 ^= c1_reduc;
    if(c15 & lsb1) c15 ^= c1_reduc;
    if(c16 & lsb1) c16 ^= c1_reduc;
    if(c17 & lsb1) c17 ^= c1_reduc;
    if(c18 & lsb1) c18 ^= c1_reduc;
    if(c19 & lsb1) c19 ^= c1_reduc;
    if(c20 & lsb1) c20 ^= c1_reduc;
    if(c21 & lsb1) c21 ^= c1_reduc;
    if(c22 & lsb1) c22 ^= c1_reduc;
    if(c23 & lsb1) c23 ^= c1_reduc;
    if(c24 & lsb1) c24 ^= c1_reduc;
    if(c25 & lsb1) c25 ^= c1_reduc;
    if(c26 & lsb1) c26 ^= c1_reduc;
    if(c27 & lsb1) c27 ^= c1_reduc;
    if(c28 & lsb1) c28 ^= c1_reduc;
    if(c29 & lsb1) c29 ^= c1_reduc;
    uint64_t mask = ~lsb1;

    uint64_t lsb2 = uint64_t_lsb(c2 & mask); // x2
    if(unlikely(!lsb2)) {
        return -1; // singular
    }

    uint64_t c2_reduc = c2 ^ lsb2;
    if(const_col & lsb2) const_col ^= c2_reduc;
    if(c3 & lsb2) c3 ^= c2_reduc;
    if(c4 & lsb2) c4 ^= c2_reduc;
    if(c5 & lsb2) c5 ^= c2_reduc;
    if(c6 & lsb2) c6 ^= c2_reduc;
    if(c7 & lsb2) c7 ^= c2_reduc;
    if(c8 & lsb2) c8 ^= c2_reduc;
    if(c9 & lsb2) c9 ^= c2_reduc;
    if(c10 & lsb2) c10 ^= c2_reduc;
    if(c11 & lsb2) c11 ^= c2_reduc;
    if(c12 & lsb2) c12 ^= c2_reduc;
    if(c13 & lsb2) c13 ^= c2_reduc;
    if(c14 & lsb2) c14 ^= c2_reduc;
    if(c15 & lsb2) c15 ^= c2_reduc;
    if(c16 & lsb2) c16 ^= c2_reduc;
    if(c17 & lsb2) c17 ^= c2_reduc;
    if(c18 & lsb2) c18 ^= c2_reduc;
    if(c19 & lsb2) c19 ^= c2_reduc;
    if(c20 & lsb2) c20 ^= c2_reduc;
    if(c21 & lsb2) c21 ^= c2_reduc;
    if(c22 & lsb2) c22 ^= c2_reduc;
    if(c23 & lsb2) c23 ^= c2_reduc;
    if(c24 & lsb2) c24 ^= c2_reduc;
    if(c25 & lsb2) c25 ^= c2_reduc;
    if(c26 & lsb2) c26 ^= c2_reduc;
    if(c27 & lsb2) c27 ^= c2_reduc;
    if(c28 & lsb2) c28 ^= c2_reduc;
    if(c29 & lsb2) c29 ^= c2_reduc;
    mask ^= lsb2;

    uint64_t lsb3 = uint64_t_lsb(c3 & mask); // x3
    if(unlikely(!lsb3)) {
        return -1; // singular
    }

    uint64_t c3_reduc = c3 ^ lsb3;
    if(const_col & lsb3) const_col ^= c3_reduc;
    if(c4 & lsb3) c4 ^= c3_reduc;
    if(c5 & lsb3) c5 ^= c3_reduc;
    if(c6 & lsb3) c6 ^= c3_reduc;
    if(c7 & lsb3) c7 ^= c3_reduc;
    if(c8 & lsb3) c8 ^= c3_reduc;
    if(c9 & lsb3) c9 ^= c3_reduc;
    if(c10 & lsb3) c10 ^= c3_reduc;
    if(c11 & lsb3) c11 ^= c3_reduc;
    if(c12 & lsb3) c12 ^= c3_reduc;
    if(c13 & lsb3) c13 ^= c3_reduc;
    if(c14 & lsb3) c14 ^= c3_reduc;
    if(c15 & lsb3) c15 ^= c3_reduc;
    if(c16 & lsb3) c16 ^= c3_reduc;
    if(c17 & lsb3) c17 ^= c3_reduc;
    if(c18 & lsb3) c18 ^= c3_reduc;
    if(c19 & lsb3) c19 ^= c3_reduc;
    if(c20 & lsb3) c20 ^= c3_reduc;
    if(c21 & lsb3) c21 ^= c3_reduc;
    if(c22 & lsb3) c22 ^= c3_reduc;
    if(c23 & lsb3) c23 ^= c3_reduc;
    if(c24 & lsb3) c24 ^= c3_reduc;
    if(c25 & lsb3) c25 ^= c3_reduc;
    if(c26 & lsb3) c26 ^= c3_reduc;
    if(c27 & lsb3) c27 ^= c3_reduc;
    if(c28 & lsb3) c28 ^= c3_reduc;
    if(c29 & lsb3) c29 ^= c3_reduc;
    mask ^= lsb3;

    uint64_t lsb4 = uint64_t_lsb(c4 & mask); // x4
    if(unlikely(!lsb4)) {
        return -1; // singular
    }

    uint64_t c4_reduc = c4 ^ lsb4;
    if(const_col & lsb4) const_col ^= c4_reduc;
    if(c5 & lsb4) c5 ^= c4_reduc;
    if(c6 & lsb4) c6 ^= c4_reduc;
    if(c7 & lsb4) c7 ^= c4_reduc;
    if(c8 & lsb4) c8 ^= c4_reduc;
    if(c9 & lsb4) c9 ^= c4_reduc;
    if(c10 & lsb4) c10 ^= c4_reduc;
    if(c11 & lsb4) c11 ^= c4_reduc;
    if(c12 & lsb4) c12 ^= c4_reduc;
    if(c13 & lsb4) c13 ^= c4_reduc;
    if(c14 & lsb4) c14 ^= c4_reduc;
    if(c15 & lsb4) c15 ^= c4_reduc;
    if(c16 & lsb4) c16 ^= c4_reduc;
    if(c17 & lsb4) c17 ^= c4_reduc;
    if(c18 & lsb4) c18 ^= c4_reduc;
    if(c19 & lsb4) c19 ^= c4_reduc;
    if(c20 & lsb4) c20 ^= c4_reduc;
    if(c21 & lsb4) c21 ^= c4_reduc;
    if(c22 & lsb4) c22 ^= c4_reduc;
    if(c23 & lsb4) c23 ^= c4_reduc;
    if(c24 & lsb4) c24 ^= c4_reduc;
    if(c25 & lsb4) c25 ^= c4_reduc;
    if(c26 & lsb4) c26 ^= c4_reduc;
    if(c27 & lsb4) c27 ^= c4_reduc;
    if(c28 & lsb4) c28 ^= c4_reduc;
    if(c29 & lsb4) c29 ^= c4_reduc;
    mask ^= lsb4;

    uint64_t lsb5 = uint64_t_lsb(c5 & mask); // x5
    if(unlikely(!lsb5)) {
        return -1; // singular
    }

    uint64_t c5_reduc = c5 ^ lsb5;
    if(const_col & lsb5) const_col ^= c5_reduc;
    if(c6 & lsb5) c6 ^= c5_reduc;
    if(c7 & lsb5) c7 ^= c5_reduc;
    if(c8 & lsb5) c8 ^= c5_reduc;
    if(c9 & lsb5) c9 ^= c5_reduc;
    if(c10 & lsb5) c10 ^= c5_reduc;
    if(c11 & lsb5) c11 ^= c5_reduc;
    if(c12 & lsb5) c12 ^= c5_reduc;
    if(c13 & lsb5) c13 ^= c5_reduc;
    if(c14 & lsb5) c14 ^= c5_reduc;
    if(c15 & lsb5) c15 ^= c5_reduc;
    if(c16 & lsb5) c16 ^= c5_reduc;
    if(c17 & lsb5) c17 ^= c5_reduc;
    if(c18 & lsb5) c18 ^= c5_reduc;
    if(c19 & lsb5) c19 ^= c5_reduc;
    if(c20 & lsb5) c20 ^= c5_reduc;
    if(c21 & lsb5) c21 ^= c5_reduc;
    if(c22 & lsb5) c22 ^= c5_reduc;
    if(c23 & lsb5) c23 ^= c5_reduc;
    if(c24 & lsb5) c24 ^= c5_reduc;
    if(c25 & lsb5) c25 ^= c5_reduc;
    if(c26 & lsb5) c26 ^= c5_reduc;
    if(c27 & lsb5) c27 ^= c5_reduc;
    if(c28 & lsb5) c28 ^= c5_reduc;
    if(c29 & lsb5) c29 ^= c5_reduc;
    mask ^= lsb5;

    uint64_t lsb6 = uint64_t_lsb(c6 & mask); // x6
    if(unlikely(!lsb6)) {
        return -1; // singular
    }

    uint64_t c6_reduc = c6 ^ lsb6;
    if(const_col & lsb6) const_col ^= c6_reduc;
    if(c7 & lsb6) c7 ^= c6_reduc;
    if(c8 & lsb6) c8 ^= c6_reduc;
    if(c9 & lsb6) c9 ^= c6_reduc;
    if(c10 & lsb6) c10 ^= c6_reduc;
    if(c11 & lsb6) c11 ^= c6_reduc;
    if(c12 & lsb6) c12 ^= c6_reduc;
    if(c13 & lsb6) c13 ^= c6_reduc;
    if(c14 & lsb6) c14 ^= c6_reduc;
    if(c15 & lsb6) c15 ^= c6_reduc;
    if(c16 & lsb6) c16 ^= c6_reduc;
    if(c17 & lsb6) c17 ^= c6_reduc;
    if(c18 & lsb6) c18 ^= c6_reduc;
    if(c19 & lsb6) c19 ^= c6_reduc;
    if(c20 & lsb6) c20 ^= c6_reduc;
    if(c21 & lsb6) c21 ^= c6_reduc;
    if(c22 & lsb6) c22 ^= c6_reduc;
    if(c23 & lsb6) c23 ^= c6_reduc;
    if(c24 & lsb6) c24 ^= c6_reduc;
    if(c25 & lsb6) c25 ^= c6_reduc;
    if(c26 & lsb6) c26 ^= c6_reduc;
    if(c27 & lsb6) c27 ^= c6_reduc;
    if(c28 & lsb6) c28 ^= c6_reduc;
    if(c29 & lsb6) c29 ^= c6_reduc;
    mask ^= lsb6;

    uint64_t lsb7 = uint64_t_lsb(c7 & mask); // x7
    if(unlikely(!lsb7)) {
        return -1; // singular
    }

    uint64_t c7_reduc = c7 ^ lsb7;
    if(const_col & lsb7) const_col ^= c7_reduc;
    if(c8 & lsb7) c8 ^= c7_reduc;
    if(c9 & lsb7) c9 ^= c7_reduc;
    if(c10 & lsb7) c10 ^= c7_reduc;
    if(c11 & lsb7) c11 ^= c7_reduc;
    if(c12 & lsb7) c12 ^= c7_reduc;
    if(c13 & lsb7) c13 ^= c7_reduc;
    if(c14 & lsb7) c14 ^= c7_reduc;
    if(c15 & lsb7) c15 ^= c7_reduc;
    if(c16 & lsb7) c16 ^= c7_reduc;
    if(c17 & lsb7) c17 ^= c7_reduc;
    if(c18 & lsb7) c18 ^= c7_reduc;
    if(c19 & lsb7) c19 ^= c7_reduc;
    if(c20 & lsb7) c20 ^= c7_reduc;
    if(c21 & lsb7) c21 ^= c7_reduc;
    if(c22 & lsb7) c22 ^= c7_reduc;
    if(c23 & lsb7) c23 ^= c7_reduc;
    if(c24 & lsb7) c24 ^= c7_reduc;
    if(c25 & lsb7) c25 ^= c7_reduc;
    if(c26 & lsb7) c26 ^= c7_reduc;
    if(c27 & lsb7) c27 ^= c7_reduc;
    if(c28 & lsb7) c28 ^= c7_reduc;
    if(c29 & lsb7) c29 ^= c7_reduc;
    mask ^= lsb7;

    uint64_t lsb8 = uint64_t_lsb(c8 & mask); // x8
    if(unlikely(!lsb8)) {
        return -1; // singular
    }

    uint64_t c8_reduc = c8 ^ lsb8;
    if(const_col & lsb8) const_col ^= c8_reduc;
    if(c9 & lsb8) c9 ^= c8_reduc;
    if(c10 & lsb8) c10 ^= c8_reduc;
    if(c11 & lsb8) c11 ^= c8_reduc;
    if(c12 & lsb8) c12 ^= c8_reduc;
    if(c13 & lsb8) c13 ^= c8_reduc;
    if(c14 & lsb8) c14 ^= c8_reduc;
    if(c15 & lsb8) c15 ^= c8_reduc;
    if(c16 & lsb8) c16 ^= c8_reduc;
    if(c17 & lsb8) c17 ^= c8_reduc;
    if(c18 & lsb8) c18 ^= c8_reduc;
    if(c19 & lsb8) c19 ^= c8_reduc;
    if(c20 & lsb8) c20 ^= c8_reduc;
    if(c21 & lsb8) c21 ^= c8_reduc;
    if(c22 & lsb8) c22 ^= c8_reduc;
    if(c23 & lsb8) c23 ^= c8_reduc;
    if(c24 & lsb8) c24 ^= c8_reduc;
    if(c25 & lsb8) c25 ^= c8_reduc;
    if(c26 & lsb8) c26 ^= c8_reduc;
    if(c27 & lsb8) c27 ^= c8_reduc;
    if(c28 & lsb8) c28 ^= c8_reduc;
    if(c29 & lsb8) c29 ^= c8_reduc;
    mask ^= lsb8;

    uint64_t lsb9 = uint64_t_lsb(c9 & mask); // x9
    if(unlikely(!lsb9)) {
        return -1; // singular
    }

    uint64_t c9_reduc = c9 ^ lsb9;
    if(const_col & lsb9) const_col ^= c9_reduc;
    if(c10 & lsb9) c10 ^= c9_reduc;
    if(c11 & lsb9) c11 ^= c9_reduc;
    if(c12 & lsb9) c12 ^= c9_reduc;
    if(c13 & lsb9) c13 ^= c9_reduc;
    if(c14 & lsb9) c14 ^= c9_reduc;
    if(c15 & lsb9) c15 ^= c9_reduc;
    if(c16 & lsb9) c16 ^= c9_reduc;
    if(c17 & lsb9) c17 ^= c9_reduc;
    if(c18 & lsb9) c18 ^= c9_reduc;
    if(c19 & lsb9) c19 ^= c9_reduc;
    if(c20 & lsb9) c20 ^= c9_reduc;
    if(c21 & lsb9) c21 ^= c9_reduc;
    if(c22 & lsb9) c22 ^= c9_reduc;
    if(c23 & lsb9) c23 ^= c9_reduc;
    if(c24 & lsb9) c24 ^= c9_reduc;
    if(c25 & lsb9) c25 ^= c9_reduc;
    if(c26 & lsb9) c26 ^= c9_reduc;
    if(c27 & lsb9) c27 ^= c9_reduc;
    if(c28 & lsb9) c28 ^= c9_reduc;
    if(c29 & lsb9) c29 ^= c9_reduc;
    mask ^= lsb9;

    uint64_t lsb10 = uint64_t_lsb(c10 & mask); // x10
    if(unlikely(!lsb10)) {
        return -1; // singular
    }

    uint64_t c10_reduc = c10 ^ lsb10;
    if(const_col & lsb10) const_col ^= c10_reduc;
    if(c11 & lsb10) c11 ^= c10_reduc;
    if(c12 & lsb10) c12 ^= c10_reduc;
    if(c13 & lsb10) c13 ^= c10_reduc;
    if(c14 & lsb10) c14 ^= c10_reduc;
    if(c15 & lsb10) c15 ^= c10_reduc;
    if(c16 & lsb10) c16 ^= c10_reduc;
    if(c17 & lsb10) c17 ^= c10_reduc;
    if(c18 & lsb10) c18 ^= c10_reduc;
    if(c19 & lsb10) c19 ^= c10_reduc;
    if(c20 & lsb10) c20 ^= c10_reduc;
    if(c21 & lsb10) c21 ^= c10_reduc;
    if(c22 & lsb10) c22 ^= c10_reduc;
    if(c23 & lsb10) c23 ^= c10_reduc;
    if(c24 & lsb10) c24 ^= c10_reduc;
    if(c25 & lsb10) c25 ^= c10_reduc;
    if(c26 & lsb10) c26 ^= c10_reduc;
    if(c27 & lsb10) c27 ^= c10_reduc;
    if(c28 & lsb10) c28 ^= c10_reduc;
    if(c29 & lsb10) c29 ^= c10_reduc;
    mask ^= lsb10;

    uint64_t lsb11 = uint64_t_lsb(c11 & mask); // x11
    if(unlikely(!lsb11)) {
        return -1; // singular
    }

    uint64_t c11_reduc = c11 ^ lsb11;
    if(const_col & lsb11) const_col ^= c11_reduc;
    if(c12 & lsb11) c12 ^= c11_reduc;
    if(c13 & lsb11) c13 ^= c11_reduc;
    if(c14 & lsb11) c14 ^= c11_reduc;
    if(c15 & lsb11) c15 ^= c11_reduc;
    if(c16 & lsb11) c16 ^= c11_reduc;
    if(c17 & lsb11) c17 ^= c11_reduc;
    if(c18 & lsb11) c18 ^= c11_reduc;
    if(c19 & lsb11) c19 ^= c11_reduc;
    if(c20 & lsb11) c20 ^= c11_reduc;
    if(c21 & lsb11) c21 ^= c11_reduc;
    if(c22 & lsb11) c22 ^= c11_reduc;
    if(c23 & lsb11) c23 ^= c11_reduc;
    if(c24 & lsb11) c24 ^= c11_reduc;
    if(c25 & lsb11) c25 ^= c11_reduc;
    if(c26 & lsb11) c26 ^= c11_reduc;
    if(c27 & lsb11) c27 ^= c11_reduc;
    if(c28 & lsb11) c28 ^= c11_reduc;
    if(c29 & lsb11) c29 ^= c11_reduc;
    mask ^= lsb11;

    uint64_t lsb12 = uint64_t_lsb(c12 & mask); // x12
    if(unlikely(!lsb12)) {
        return -1; // singular
    }

    uint64_t c12_reduc = c12 ^ lsb12;
    if(const_col & lsb12) const_col ^= c12_reduc;
    if(c13 & lsb12) c13 ^= c12_reduc;
    if(c14 & lsb12) c14 ^= c12_reduc;
    if(c15 & lsb12) c15 ^= c12_reduc;
    if(c16 & lsb12) c16 ^= c12_reduc;
    if(c17 & lsb12) c17 ^= c12_reduc;
    if(c18 & lsb12) c18 ^= c12_reduc;
    if(c19 & lsb12) c19 ^= c12_reduc;
    if(c20 & lsb12) c20 ^= c12_reduc;
    if(c21 & lsb12) c21 ^= c12_reduc;
    if(c22 & lsb12) c22 ^= c12_reduc;
    if(c23 & lsb12) c23 ^= c12_reduc;
    if(c24 & lsb12) c24 ^= c12_reduc;
    if(c25 & lsb12) c25 ^= c12_reduc;
    if(c26 & lsb12) c26 ^= c12_reduc;
    if(c27 & lsb12) c27 ^= c12_reduc;
    if(c28 & lsb12) c28 ^= c12_reduc;
    if(c29 & lsb12) c29 ^= c12_reduc;
    mask ^= lsb12;

    uint64_t lsb13 = uint64_t_lsb(c13 & mask); // x13
    if(unlikely(!lsb13)) {
        return -1; // singular
    }

    uint64_t c13_reduc = c13 ^ lsb13;
    if(const_col & lsb13) const_col ^= c13_reduc;
    if(c14 & lsb13) c14 ^= c13_reduc;
    if(c15 & lsb13) c15 ^= c13_reduc;
    if(c16 & lsb13) c16 ^= c13_reduc;
    if(c17 & lsb13) c17 ^= c13_reduc;
    if(c18 & lsb13) c18 ^= c13_reduc;
    if(c19 & lsb13) c19 ^= c13_reduc;
    if(c20 & lsb13) c20 ^= c13_reduc;
    if(c21 & lsb13) c21 ^= c13_reduc;
    if(c22 & lsb13) c22 ^= c13_reduc;
    if(c23 & lsb13) c23 ^= c13_reduc;
    if(c24 & lsb13) c24 ^= c13_reduc;
    if(c25 & lsb13) c25 ^= c13_reduc;
    if(c26 & lsb13) c26 ^= c13_reduc;
    if(c27 & lsb13) c27 ^= c13_reduc;
    if(c28 & lsb13) c28 ^= c13_reduc;
    if(c29 & lsb13) c29 ^= c13_reduc;
    mask ^= lsb13;

    uint64_t lsb14 = uint64_t_lsb(c14 & mask); // x14
    if(unlikely(!lsb14)) {
        return -1; // singular
    }

    uint64_t c14_reduc = c14 ^ lsb14;
    if(const_col & lsb14) const_col ^= c14_reduc;
    if(c15 & lsb14) c15 ^= c14_reduc;
    if(c16 & lsb14) c16 ^= c14_reduc;
    if(c17 & lsb14) c17 ^= c14_reduc;
    if(c18 & lsb14) c18 ^= c14_reduc;
    if(c19 & lsb14) c19 ^= c14_reduc;
    if(c20 & lsb14) c20 ^= c14_reduc;
    if(c21 & lsb14) c21 ^= c14_reduc;
    if(c22 & lsb14) c22 ^= c14_reduc;
    if(c23 & lsb14) c23 ^= c14_reduc;
    if(c24 & lsb14) c24 ^= c14_reduc;
    if(c25 & lsb14) c25 ^= c14_reduc;
    if(c26 & lsb14) c26 ^= c14_reduc;
    if(c27 & lsb14) c27 ^= c14_reduc;
    if(c28 & lsb14) c28 ^= c14_reduc;
    if(c29 & lsb14) c29 ^= c14_reduc;
    mask ^= lsb14;

    uint64_t lsb15 = uint64_t_lsb(c15 & mask); // x15
    if(unlikely(!lsb15)) {
        return -1; // singular
    }

    uint64_t c15_reduc = c15 ^ lsb15;
    if(const_col & lsb15) const_col ^= c15_reduc;
    if(c16 & lsb15) c16 ^= c15_reduc;
    if(c17 & lsb15) c17 ^= c15_reduc;
    if(c18 & lsb15) c18 ^= c15_reduc;
    if(c19 & lsb15) c19 ^= c15_reduc;
    if(c20 & lsb15) c20 ^= c15_reduc;
    if(c21 & lsb15) c21 ^= c15_reduc;
    if(c22 & lsb15) c22 ^= c15_reduc;
    if(c23 & lsb15) c23 ^= c15_reduc;
    if(c24 & lsb15) c24 ^= c15_reduc;
    if(c25 & lsb15) c25 ^= c15_reduc;
    if(c26 & lsb15) c26 ^= c15_reduc;
    if(c27 & lsb15) c27 ^= c15_reduc;
    if(c28 & lsb15) c28 ^= c15_reduc;
    if(c29 & lsb15) c29 ^= c15_reduc;
    mask ^= lsb15;

    uint64_t lsb16 = uint64_t_lsb(c16 & mask); // x16
    if(unlikely(!lsb16)) {
        return -1; // singular
    }

    uint64_t c16_reduc = c16 ^ lsb16;
    if(const_col & lsb16) const_col ^= c16_reduc;
    if(c17 & lsb16) c17 ^= c16_reduc;
    if(c18 & lsb16) c18 ^= c16_reduc;
    if(c19 & lsb16) c19 ^= c16_reduc;
    if(c20 & lsb16) c20 ^= c16_reduc;
    if(c21 & lsb16) c21 ^= c16_reduc;
    if(c22 & lsb16) c22 ^= c16_reduc;
    if(c23 & lsb16) c23 ^= c16_reduc;
    if(c24 & lsb16) c24 ^= c16_reduc;
    if(c25 & lsb16) c25 ^= c16_reduc;
    if(c26 & lsb16) c26 ^= c16_reduc;
    if(c27 & lsb16) c27 ^= c16_reduc;
    if(c28 & lsb16) c28 ^= c16_reduc;
    if(c29 & lsb16) c29 ^= c16_reduc;
    mask ^= lsb16;

    uint64_t lsb17 = uint64_t_lsb(c17 & mask); // x17
    if(unlikely(!lsb17)) {
        return -1; // singular
    }

    uint64_t c17_reduc = c17 ^ lsb17;
    if(const_col & lsb17) const_col ^= c17_reduc;
    if(c18 & lsb17) c18 ^= c17_reduc;
    if(c19 & lsb17) c19 ^= c17_reduc;
    if(c20 & lsb17) c20 ^= c17_reduc;
    if(c21 & lsb17) c21 ^= c17_reduc;
    if(c22 & lsb17) c22 ^= c17_reduc;
    if(c23 & lsb17) c23 ^= c17_reduc;
    if(c24 & lsb17) c24 ^= c17_reduc;
    if(c25 & lsb17) c25 ^= c17_reduc;
    if(c26 & lsb17) c26 ^= c17_reduc;
    if(c27 & lsb17) c27 ^= c17_reduc;
    if(c28 & lsb17) c28 ^= c17_reduc;
    if(c29 & lsb17) c29 ^= c17_reduc;
    mask ^= lsb17;

    uint64_t lsb18 = uint64_t_lsb(c18 & mask); // x18
    if(unlikely(!lsb18)) {
        return -1; // singular
    }

    uint64_t c18_reduc = c18 ^ lsb18;
    if(const_col & lsb18) const_col ^= c18_reduc;
    if(c19 & lsb18) c19 ^= c18_reduc;
    if(c20 & lsb18) c20 ^= c18_reduc;
    if(c21 & lsb18) c21 ^= c18_reduc;
    if(c22 & lsb18) c22 ^= c18_reduc;
    if(c23 & lsb18) c23 ^= c18_reduc;
    if(c24 & lsb18) c24 ^= c18_reduc;
    if(c25 & lsb18) c25 ^= c18_reduc;
    if(c26 & lsb18) c26 ^= c18_reduc;
    if(c27 & lsb18) c27 ^= c18_reduc;
    if(c28 & lsb18) c28 ^= c18_reduc;
    if(c29 & lsb18) c29 ^= c18_reduc;
    mask ^= lsb18;

    uint64_t lsb19 = uint64_t_lsb(c19 & mask); // x19
    if(unlikely(!lsb19)) {
        return -1; // singular
    }

    uint64_t c19_reduc = c19 ^ lsb19;
    if(const_col & lsb19) const_col ^= c19_reduc;
    if(c20 & lsb19) c20 ^= c19_reduc;
    if(c21 & lsb19) c21 ^= c19_reduc;
    if(c22 & lsb19) c22 ^= c19_reduc;
    if(c23 & lsb19) c23 ^= c19_reduc;
    if(c24 & lsb19) c24 ^= c19_reduc;
    if(c25 & lsb19) c25 ^= c19_reduc;
    if(c26 & lsb19) c26 ^= c19_reduc;
    if(c27 & lsb19) c27 ^= c19_reduc;
    if(c28 & lsb19) c28 ^= c19_reduc;
    if(c29 & lsb19) c29 ^= c19_reduc;
    mask ^= lsb19;

    uint64_t lsb20 = uint64_t_lsb(c20 & mask); // x20
    if(unlikely(!lsb20)) {
        return -1; // singular
    }

    uint64_t c20_reduc = c20 ^ lsb20;
    if(const_col & lsb20) const_col ^= c20_reduc;
    if(c21 & lsb20) c21 ^= c20_reduc;
    if(c22 & lsb20) c22 ^= c20_reduc;
    if(c23 & lsb20) c23 ^= c20_reduc;
    if(c24 & lsb20) c24 ^= c20_reduc;
    if(c25 & lsb20) c25 ^= c20_reduc;
    if(c26 & lsb20) c26 ^= c20_reduc;
    if(c27 & lsb20) c27 ^= c20_reduc;
    if(c28 & lsb20) c28 ^= c20_reduc;
    if(c29 & lsb20) c29 ^= c20_reduc;
    mask ^= lsb20;

    uint64_t lsb21 = uint64_t_lsb(c21 & mask); // x21
    if(unlikely(!lsb21)) {
        return -1; // singular
    }

    uint64_t c21_reduc = c21 ^ lsb21;
    if(const_col & lsb21) const_col ^= c21_reduc;
    if(c22 & lsb21) c22 ^= c21_reduc;
    if(c23 & lsb21) c23 ^= c21_reduc;
    if(c24 & lsb21) c24 ^= c21_reduc;
    if(c25 & lsb21) c25 ^= c21_reduc;
    if(c26 & lsb21) c26 ^= c21_reduc;
    if(c27 & lsb21) c27 ^= c21_reduc;
    if(c28 & lsb21) c28 ^= c21_reduc;
    if(c29 & lsb21) c29 ^= c21_reduc;
    mask ^= lsb21;

    uint64_t lsb22 = uint64_t_lsb(c22 & mask); // x22
    if(unlikely(!lsb22)) {
        return -1; // singular
    }

    uint64_t c22_reduc = c22 ^ lsb22;
    if(const_col & lsb22) const_col ^= c22_reduc;
    if(c23 & lsb22) c23 ^= c22_reduc;
    if(c24 & lsb22) c24 ^= c22_reduc;
    if(c25 & lsb22) c25 ^= c22_reduc;
    if(c26 & lsb22) c26 ^= c22_reduc;
    if(c27 & lsb22) c27 ^= c22_reduc;
    if(c28 & lsb22) c28 ^= c22_reduc;
    if(c29 & lsb22) c29 ^= c22_reduc;
    mask ^= lsb22;

    uint64_t lsb23 = uint64_t_lsb(c23 & mask); // x23
    if(unlikely(!lsb23)) {
        return -1; // singular
    }

    uint64_t c23_reduc = c23 ^ lsb23;
    if(const_col & lsb23) const_col ^= c23_reduc;
    if(c24 & lsb23) c24 ^= c23_reduc;
    if(c25 & lsb23) c25 ^= c23_reduc;
    if(c26 & lsb23) c26 ^= c23_reduc;
    if(c27 & lsb23) c27 ^= c23_reduc;
    if(c28 & lsb23) c28 ^= c23_reduc;
    if(c29 & lsb23) c29 ^= c23_reduc;
    mask ^= lsb23;

    uint64_t lsb24 = uint64_t_lsb(c24 & mask); // x24
    if(unlikely(!lsb24)) {
        return -1; // singular
    }

    uint64_t c24_reduc = c24 ^ lsb24;
    if(const_col & lsb24) const_col ^= c24_reduc;
    if(c25 & lsb24) c25 ^= c24_reduc;
    if(c26 & lsb24) c26 ^= c24_reduc;
    if(c27 & lsb24) c27 ^= c24_reduc;
    if(c28 & lsb24) c28 ^= c24_reduc;
    if(c29 & lsb24) c29 ^= c24_reduc;
    mask ^= lsb24;

    uint64_t lsb25 = uint64_t_lsb(c25 & mask); // x25
    if(unlikely(!lsb25)) {
        return -1; // singular
    }

    uint64_t c25_reduc = c25 ^ lsb25;
    if(const_col & lsb25) const_col ^= c25_reduc;
    if(c26 & lsb25) c26 ^= c25_reduc;
    if(c27 & lsb25) c27 ^= c25_reduc;
    if(c28 & lsb25) c28 ^= c25_reduc;
    if(c29 & lsb25) c29 ^= c25_reduc;
    mask ^= lsb25;

    uint64_t lsb26 = uint64_t_lsb(c26 & mask); // x26
    if(unlikely(!lsb26)) {
        return -1; // singular
    }

    uint64_t c26_reduc = c26 ^ lsb26;
    if(const_col & lsb26) const_col ^= c26_reduc;
    if(c27 & lsb26) c27 ^= c26_reduc;
    if(c28 & lsb26) c28 ^= c26_reduc;
    if(c29 & lsb26) c29 ^= c26_reduc;
    mask ^= lsb26;

    uint64_t lsb27 = uint64_t_lsb(c27 & mask); // x27
    if(unlikely(!lsb27)) {
        return -1; // singular
    }

    uint64_t c27_reduc = c27 ^ lsb27;
    if(const_col & lsb27) const_col ^= c27_reduc;
    if(c28 & lsb27) c28 ^= c27_reduc;
    if(c29 & lsb27) c29 ^= c27_reduc;
    mask ^= lsb27;

    uint64_t lsb28 = uint64_t_lsb(c28 & mask); // x28
    if(unlikely(!lsb28)) {
        return -1; // singular
    }

    uint64_t c28_reduc = c28 ^ lsb28;
    if(const_col & lsb28) const_col ^= c28_reduc;
    if(c29 & lsb28) c29 ^= c28_reduc;
    mask ^= lsb28;

    uint64_t lsb29 = uint64_t_lsb(c29 & mask); // x29
    if(unlikely(!lsb29)) {
        return -1; // singular
    }

    if(const_col & lsb29) const_col ^= c29 ^ lsb29;
    mask ^= lsb29;

    if(likely(mask & const_col)) { // check system consistency
        return mask & const_col; // not solvable
    }

    uint64_t s = 0x0ULL;
    if(const_col & lsb1) s = uint64_t_toggle_at(s, 0);
    if(const_col & lsb2) s = uint64_t_toggle_at(s, 1);
    if(const_col & lsb3) s = uint64_t_toggle_at(s, 2);
    if(const_col & lsb4) s = uint64_t_toggle_at(s, 3);
    if(const_col & lsb5) s = uint64_t_toggle_at(s, 4);
    if(const_col & lsb6) s = uint64_t_toggle_at(s, 5);
    if(const_col & lsb7) s = uint64_t_toggle_at(s, 6);
    if(const_col & lsb8) s = uint64_t_toggle_at(s, 7);
    if(const_col & lsb9) s = uint64_t_toggle_at(s, 8);
    if(const_col & lsb10) s = uint64_t_toggle_at(s, 9);
    if(const_col & lsb11) s = uint64_t_toggle_at(s, 10);
    if(const_col & lsb12) s = uint64_t_toggle_at(s, 11);
    if(const_col & lsb13) s = uint64_t_toggle_at(s, 12);
    if(const_col & lsb14) s = uint64_t_toggle_at(s, 13);
    if(const_col & lsb15) s = uint64_t_toggle_at(s, 14);
    if(const_col & lsb16) s = uint64_t_toggle_at(s, 15);
    if(const_col & lsb17) s = uint64_t_toggle_at(s, 16);
    if(const_col & lsb18) s = uint64_t_toggle_at(s, 17);
    if(const_col & lsb19) s = uint64_t_toggle_at(s, 18);
    if(const_col & lsb20) s = uint64_t_toggle_at(s, 19);
    if(const_col & lsb21) s = uint64_t_toggle_at(s, 20);
    if(const_col & lsb22) s = uint64_t_toggle_at(s, 21);
    if(const_col & lsb23) s = uint64_t_toggle_at(s, 22);
    if(const_col & lsb24) s = uint64_t_toggle_at(s, 23);
    if(const_col & lsb25) s = uint64_t_toggle_at(s, 24);
    if(const_col & lsb26) s = uint64_t_toggle_at(s, 25);
    if(const_col & lsb27) s = uint64_t_toggle_at(s, 26);
    if(const_col & lsb28) s = uint64_t_toggle_at(s, 27);
    if(const_col & lsb29) s = uint64_t_toggle_at(s, 28);
    *sol = s;
    return 0;
}

int64_t
uint64a_gj_v30_generic(const uint64_t m[31], uint64_t* const restrict sol) {
    uint64_t const_col = m[0];
    uint64_t c1 = m[1]; // x1
    uint64_t c2 = m[2]; // x2
    uint64_t c3 = m[3]; // x3
    uint64_t c4 = m[4]; // x4
    uint64_t c5 = m[5]; // x5
    uint64_t c6 = m[6]; // x6
    uint64_t c7 = m[7]; // x7
    uint64_t c8 = m[8]; // x8
    uint64_t c9 = m[9]; // x9
    uint64_t c10 = m[10]; // x10
    uint64_t c11 = m[11]; // x11
    uint64_t c12 = m[12]; // x12
    uint64_t c13 = m[13]; // x13
    uint64_t c14 = m[14]; // x14
    uint64_t c15 = m[15]; // x15
    uint64_t c16 = m[16]; // x16
    uint64_t c17 = m[17]; // x17
    uint64_t c18 = m[18]; // x18
    uint64_t c19 = m[19]; // x19
    uint64_t c20 = m[20]; // x20
    uint64_t c21 = m[21]; // x21
    uint64_t c22 = m[22]; // x22
    uint64_t c23 = m[23]; // x23
    uint64_t c24 = m[24]; // x24
    uint64_t c25 = m[25]; // x25
    uint64_t c26 = m[26]; // x26
    uint64_t c27 = m[27]; // x27
    uint64_t c28 = m[28]; // x28
    uint64_t c29 = m[29]; // x29
    uint64_t c30 = m[30]; // x30
    uint64_t lsb1 = uint64_t_lsb(c1); // initially all eqs are candidates
    if(unlikely(!lsb1)) {
        return -1; // singular
    }

    uint64_t c1_reduc = c1 ^ lsb1; // for row reduction
    if(const_col & lsb1) const_col ^= c1_reduc;
    if(c2 & lsb1) c2 ^= c1_reduc;
    if(c3 & lsb1) c3 ^= c1_reduc;
    if(c4 & lsb1) c4 ^= c1_reduc;
    if(c5 & lsb1) c5 ^= c1_reduc;
    if(c6 & lsb1) c6 ^= c1_reduc;
    if(c7 & lsb1) c7 ^= c1_reduc;
    if(c8 & lsb1) c8 ^= c1_reduc;
    if(c9 & lsb1) c9 ^= c1_reduc;
    if(c10 & lsb1) c10 ^= c1_reduc;
    if(c11 & lsb1) c11 ^= c1_reduc;
    if(c12 & lsb1) c12 ^= c1_reduc;
    if(c13 & lsb1) c13 ^= c1_reduc;
    if(c14 & lsb1) c14 ^= c1_reduc;
    if(c15 & lsb1) c15 ^= c1_reduc;
    if(c16 & lsb1) c16 ^= c1_reduc;
    if(c17 & lsb1) c17 ^= c1_reduc;
    if(c18 & lsb1) c18 ^= c1_reduc;
    if(c19 & lsb1) c19 ^= c1_reduc;
    if(c20 & lsb1) c20 ^= c1_reduc;
    if(c21 & lsb1) c21 ^= c1_reduc;
    if(c22 & lsb1) c22 ^= c1_reduc;
    if(c23 & lsb1) c23 ^= c1_reduc;
    if(c24 & lsb1) c24 ^= c1_reduc;
    if(c25 & lsb1) c25 ^= c1_reduc;
    if(c26 & lsb1) c26 ^= c1_reduc;
    if(c27 & lsb1) c27 ^= c1_reduc;
    if(c28 & lsb1) c28 ^= c1_reduc;
    if(c29 & lsb1) c29 ^= c1_reduc;
    if(c30 & lsb1) c30 ^= c1_reduc;
    uint64_t mask = ~lsb1;

    uint64_t lsb2 = uint64_t_lsb(c2 & mask); // x2
    if(unlikely(!lsb2)) {
        return -1; // singular
    }

    uint64_t c2_reduc = c2 ^ lsb2;
    if(const_col & lsb2) const_col ^= c2_reduc;
    if(c3 & lsb2) c3 ^= c2_reduc;
    if(c4 & lsb2) c4 ^= c2_reduc;
    if(c5 & lsb2) c5 ^= c2_reduc;
    if(c6 & lsb2) c6 ^= c2_reduc;
    if(c7 & lsb2) c7 ^= c2_reduc;
    if(c8 & lsb2) c8 ^= c2_reduc;
    if(c9 & lsb2) c9 ^= c2_reduc;
    if(c10 & lsb2) c10 ^= c2_reduc;
    if(c11 & lsb2) c11 ^= c2_reduc;
    if(c12 & lsb2) c12 ^= c2_reduc;
    if(c13 & lsb2) c13 ^= c2_reduc;
    if(c14 & lsb2) c14 ^= c2_reduc;
    if(c15 & lsb2) c15 ^= c2_reduc;
    if(c16 & lsb2) c16 ^= c2_reduc;
    if(c17 & lsb2) c17 ^= c2_reduc;
    if(c18 & lsb2) c18 ^= c2_reduc;
    if(c19 & lsb2) c19 ^= c2_reduc;
    if(c20 & lsb2) c20 ^= c2_reduc;
    if(c21 & lsb2) c21 ^= c2_reduc;
    if(c22 & lsb2) c22 ^= c2_reduc;
    if(c23 & lsb2) c23 ^= c2_reduc;
    if(c24 & lsb2) c24 ^= c2_reduc;
    if(c25 & lsb2) c25 ^= c2_reduc;
    if(c26 & lsb2) c26 ^= c2_reduc;
    if(c27 & lsb2) c27 ^= c2_reduc;
    if(c28 & lsb2) c28 ^= c2_reduc;
    if(c29 & lsb2) c29 ^= c2_reduc;
    if(c30 & lsb2) c30 ^= c2_reduc;
    mask ^= lsb2;

    uint64_t lsb3 = uint64_t_lsb(c3 & mask); // x3
    if(unlikely(!lsb3)) {
        return -1; // singular
    }

    uint64_t c3_reduc = c3 ^ lsb3;
    if(const_col & lsb3) const_col ^= c3_reduc;
    if(c4 & lsb3) c4 ^= c3_reduc;
    if(c5 & lsb3) c5 ^= c3_reduc;
    if(c6 & lsb3) c6 ^= c3_reduc;
    if(c7 & lsb3) c7 ^= c3_reduc;
    if(c8 & lsb3) c8 ^= c3_reduc;
    if(c9 & lsb3) c9 ^= c3_reduc;
    if(c10 & lsb3) c10 ^= c3_reduc;
    if(c11 & lsb3) c11 ^= c3_reduc;
    if(c12 & lsb3) c12 ^= c3_reduc;
    if(c13 & lsb3) c13 ^= c3_reduc;
    if(c14 & lsb3) c14 ^= c3_reduc;
    if(c15 & lsb3) c15 ^= c3_reduc;
    if(c16 & lsb3) c16 ^= c3_reduc;
    if(c17 & lsb3) c17 ^= c3_reduc;
    if(c18 & lsb3) c18 ^= c3_reduc;
    if(c19 & lsb3) c19 ^= c3_reduc;
    if(c20 & lsb3) c20 ^= c3_reduc;
    if(c21 & lsb3) c21 ^= c3_reduc;
    if(c22 & lsb3) c22 ^= c3_reduc;
    if(c23 & lsb3) c23 ^= c3_reduc;
    if(c24 & lsb3) c24 ^= c3_reduc;
    if(c25 & lsb3) c25 ^= c3_reduc;
    if(c26 & lsb3) c26 ^= c3_reduc;
    if(c27 & lsb3) c27 ^= c3_reduc;
    if(c28 & lsb3) c28 ^= c3_reduc;
    if(c29 & lsb3) c29 ^= c3_reduc;
    if(c30 & lsb3) c30 ^= c3_reduc;
    mask ^= lsb3;

    uint64_t lsb4 = uint64_t_lsb(c4 & mask); // x4
    if(unlikely(!lsb4)) {
        return -1; // singular
    }

    uint64_t c4_reduc = c4 ^ lsb4;
    if(const_col & lsb4) const_col ^= c4_reduc;
    if(c5 & lsb4) c5 ^= c4_reduc;
    if(c6 & lsb4) c6 ^= c4_reduc;
    if(c7 & lsb4) c7 ^= c4_reduc;
    if(c8 & lsb4) c8 ^= c4_reduc;
    if(c9 & lsb4) c9 ^= c4_reduc;
    if(c10 & lsb4) c10 ^= c4_reduc;
    if(c11 & lsb4) c11 ^= c4_reduc;
    if(c12 & lsb4) c12 ^= c4_reduc;
    if(c13 & lsb4) c13 ^= c4_reduc;
    if(c14 & lsb4) c14 ^= c4_reduc;
    if(c15 & lsb4) c15 ^= c4_reduc;
    if(c16 & lsb4) c16 ^= c4_reduc;
    if(c17 & lsb4) c17 ^= c4_reduc;
    if(c18 & lsb4) c18 ^= c4_reduc;
    if(c19 & lsb4) c19 ^= c4_reduc;
    if(c20 & lsb4) c20 ^= c4_reduc;
    if(c21 & lsb4) c21 ^= c4_reduc;
    if(c22 & lsb4) c22 ^= c4_reduc;
    if(c23 & lsb4) c23 ^= c4_reduc;
    if(c24 & lsb4) c24 ^= c4_reduc;
    if(c25 & lsb4) c25 ^= c4_reduc;
    if(c26 & lsb4) c26 ^= c4_reduc;
    if(c27 & lsb4) c27 ^= c4_reduc;
    if(c28 & lsb4) c28 ^= c4_reduc;
    if(c29 & lsb4) c29 ^= c4_reduc;
    if(c30 & lsb4) c30 ^= c4_reduc;
    mask ^= lsb4;

    uint64_t lsb5 = uint64_t_lsb(c5 & mask); // x5
    if(unlikely(!lsb5)) {
        return -1; // singular
    }

    uint64_t c5_reduc = c5 ^ lsb5;
    if(const_col & lsb5) const_col ^= c5_reduc;
    if(c6 & lsb5) c6 ^= c5_reduc;
    if(c7 & lsb5) c7 ^= c5_reduc;
    if(c8 & lsb5) c8 ^= c5_reduc;
    if(c9 & lsb5) c9 ^= c5_reduc;
    if(c10 & lsb5) c10 ^= c5_reduc;
    if(c11 & lsb5) c11 ^= c5_reduc;
    if(c12 & lsb5) c12 ^= c5_reduc;
    if(c13 & lsb5) c13 ^= c5_reduc;
    if(c14 & lsb5) c14 ^= c5_reduc;
    if(c15 & lsb5) c15 ^= c5_reduc;
    if(c16 & lsb5) c16 ^= c5_reduc;
    if(c17 & lsb5) c17 ^= c5_reduc;
    if(c18 & lsb5) c18 ^= c5_reduc;
    if(c19 & lsb5) c19 ^= c5_reduc;
    if(c20 & lsb5) c20 ^= c5_reduc;
    if(c21 & lsb5) c21 ^= c5_reduc;
    if(c22 & lsb5) c22 ^= c5_reduc;
    if(c23 & lsb5) c23 ^= c5_reduc;
    if(c24 & lsb5) c24 ^= c5_reduc;
    if(c25 & lsb5) c25 ^= c5_reduc;
    if(c26 & lsb5) c26 ^= c5_reduc;
    if(c27 & lsb5) c27 ^= c5_reduc;
    if(c28 & lsb5) c28 ^= c5_reduc;
    if(c29 & lsb5) c29 ^= c5_reduc;
    if(c30 & lsb5) c30 ^= c5_reduc;
    mask ^= lsb5;

    uint64_t lsb6 = uint64_t_lsb(c6 & mask); // x6
    if(unlikely(!lsb6)) {
        return -1; // singular
    }

    uint64_t c6_reduc = c6 ^ lsb6;
    if(const_col & lsb6) const_col ^= c6_reduc;
    if(c7 & lsb6) c7 ^= c6_reduc;
    if(c8 & lsb6) c8 ^= c6_reduc;
    if(c9 & lsb6) c9 ^= c6_reduc;
    if(c10 & lsb6) c10 ^= c6_reduc;
    if(c11 & lsb6) c11 ^= c6_reduc;
    if(c12 & lsb6) c12 ^= c6_reduc;
    if(c13 & lsb6) c13 ^= c6_reduc;
    if(c14 & lsb6) c14 ^= c6_reduc;
    if(c15 & lsb6) c15 ^= c6_reduc;
    if(c16 & lsb6) c16 ^= c6_reduc;
    if(c17 & lsb6) c17 ^= c6_reduc;
    if(c18 & lsb6) c18 ^= c6_reduc;
    if(c19 & lsb6) c19 ^= c6_reduc;
    if(c20 & lsb6) c20 ^= c6_reduc;
    if(c21 & lsb6) c21 ^= c6_reduc;
    if(c22 & lsb6) c22 ^= c6_reduc;
    if(c23 & lsb6) c23 ^= c6_reduc;
    if(c24 & lsb6) c24 ^= c6_reduc;
    if(c25 & lsb6) c25 ^= c6_reduc;
    if(c26 & lsb6) c26 ^= c6_reduc;
    if(c27 & lsb6) c27 ^= c6_reduc;
    if(c28 & lsb6) c28 ^= c6_reduc;
    if(c29 & lsb6) c29 ^= c6_reduc;
    if(c30 & lsb6) c30 ^= c6_reduc;
    mask ^= lsb6;

    uint64_t lsb7 = uint64_t_lsb(c7 & mask); // x7
    if(unlikely(!lsb7)) {
        return -1; // singular
    }

    uint64_t c7_reduc = c7 ^ lsb7;
    if(const_col & lsb7) const_col ^= c7_reduc;
    if(c8 & lsb7) c8 ^= c7_reduc;
    if(c9 & lsb7) c9 ^= c7_reduc;
    if(c10 & lsb7) c10 ^= c7_reduc;
    if(c11 & lsb7) c11 ^= c7_reduc;
    if(c12 & lsb7) c12 ^= c7_reduc;
    if(c13 & lsb7) c13 ^= c7_reduc;
    if(c14 & lsb7) c14 ^= c7_reduc;
    if(c15 & lsb7) c15 ^= c7_reduc;
    if(c16 & lsb7) c16 ^= c7_reduc;
    if(c17 & lsb7) c17 ^= c7_reduc;
    if(c18 & lsb7) c18 ^= c7_reduc;
    if(c19 & lsb7) c19 ^= c7_reduc;
    if(c20 & lsb7) c20 ^= c7_reduc;
    if(c21 & lsb7) c21 ^= c7_reduc;
    if(c22 & lsb7) c22 ^= c7_reduc;
    if(c23 & lsb7) c23 ^= c7_reduc;
    if(c24 & lsb7) c24 ^= c7_reduc;
    if(c25 & lsb7) c25 ^= c7_reduc;
    if(c26 & lsb7) c26 ^= c7_reduc;
    if(c27 & lsb7) c27 ^= c7_reduc;
    if(c28 & lsb7) c28 ^= c7_reduc;
    if(c29 & lsb7) c29 ^= c7_reduc;
    if(c30 & lsb7) c30 ^= c7_reduc;
    mask ^= lsb7;

    uint64_t lsb8 = uint64_t_lsb(c8 & mask); // x8
    if(unlikely(!lsb8)) {
        return -1; // singular
    }

    uint64_t c8_reduc = c8 ^ lsb8;
    if(const_col & lsb8) const_col ^= c8_reduc;
    if(c9 & lsb8) c9 ^= c8_reduc;
    if(c10 & lsb8) c10 ^= c8_reduc;
    if(c11 & lsb8) c11 ^= c8_reduc;
    if(c12 & lsb8) c12 ^= c8_reduc;
    if(c13 & lsb8) c13 ^= c8_reduc;
    if(c14 & lsb8) c14 ^= c8_reduc;
    if(c15 & lsb8) c15 ^= c8_reduc;
    if(c16 & lsb8) c16 ^= c8_reduc;
    if(c17 & lsb8) c17 ^= c8_reduc;
    if(c18 & lsb8) c18 ^= c8_reduc;
    if(c19 & lsb8) c19 ^= c8_reduc;
    if(c20 & lsb8) c20 ^= c8_reduc;
    if(c21 & lsb8) c21 ^= c8_reduc;
    if(c22 & lsb8) c22 ^= c8_reduc;
    if(c23 & lsb8) c23 ^= c8_reduc;
    if(c24 & lsb8) c24 ^= c8_reduc;
    if(c25 & lsb8) c25 ^= c8_reduc;
    if(c26 & lsb8) c26 ^= c8_reduc;
    if(c27 & lsb8) c27 ^= c8_reduc;
    if(c28 & lsb8) c28 ^= c8_reduc;
    if(c29 & lsb8) c29 ^= c8_reduc;
    if(c30 & lsb8) c30 ^= c8_reduc;
    mask ^= lsb8;

    uint64_t lsb9 = uint64_t_lsb(c9 & mask); // x9
    if(unlikely(!lsb9)) {
        return -1; // singular
    }

    uint64_t c9_reduc = c9 ^ lsb9;
    if(const_col & lsb9) const_col ^= c9_reduc;
    if(c10 & lsb9) c10 ^= c9_reduc;
    if(c11 & lsb9) c11 ^= c9_reduc;
    if(c12 & lsb9) c12 ^= c9_reduc;
    if(c13 & lsb9) c13 ^= c9_reduc;
    if(c14 & lsb9) c14 ^= c9_reduc;
    if(c15 & lsb9) c15 ^= c9_reduc;
    if(c16 & lsb9) c16 ^= c9_reduc;
    if(c17 & lsb9) c17 ^= c9_reduc;
    if(c18 & lsb9) c18 ^= c9_reduc;
    if(c19 & lsb9) c19 ^= c9_reduc;
    if(c20 & lsb9) c20 ^= c9_reduc;
    if(c21 & lsb9) c21 ^= c9_reduc;
    if(c22 & lsb9) c22 ^= c9_reduc;
    if(c23 & lsb9) c23 ^= c9_reduc;
    if(c24 & lsb9) c24 ^= c9_reduc;
    if(c25 & lsb9) c25 ^= c9_reduc;
    if(c26 & lsb9) c26 ^= c9_reduc;
    if(c27 & lsb9) c27 ^= c9_reduc;
    if(c28 & lsb9) c28 ^= c9_reduc;
    if(c29 & lsb9) c29 ^= c9_reduc;
    if(c30 & lsb9) c30 ^= c9_reduc;
    mask ^= lsb9;

    uint64_t lsb10 = uint64_t_lsb(c10 & mask); // x10
    if(unlikely(!lsb10)) {
        return -1; // singular
    }

    uint64_t c10_reduc = c10 ^ lsb10;
    if(const_col & lsb10) const_col ^= c10_reduc;
    if(c11 & lsb10) c11 ^= c10_reduc;
    if(c12 & lsb10) c12 ^= c10_reduc;
    if(c13 & lsb10) c13 ^= c10_reduc;
    if(c14 & lsb10) c14 ^= c10_reduc;
    if(c15 & lsb10) c15 ^= c10_reduc;
    if(c16 & lsb10) c16 ^= c10_reduc;
    if(c17 & lsb10) c17 ^= c10_reduc;
    if(c18 & lsb10) c18 ^= c10_reduc;
    if(c19 & lsb10) c19 ^= c10_reduc;
    if(c20 & lsb10) c20 ^= c10_reduc;
    if(c21 & lsb10) c21 ^= c10_reduc;
    if(c22 & lsb10) c22 ^= c10_reduc;
    if(c23 & lsb10) c23 ^= c10_reduc;
    if(c24 & lsb10) c24 ^= c10_reduc;
    if(c25 & lsb10) c25 ^= c10_reduc;
    if(c26 & lsb10) c26 ^= c10_reduc;
    if(c27 & lsb10) c27 ^= c10_reduc;
    if(c28 & lsb10) c28 ^= c10_reduc;
    if(c29 & lsb10) c29 ^= c10_reduc;
    if(c30 & lsb10) c30 ^= c10_reduc;
    mask ^= lsb10;

    uint64_t lsb11 = uint64_t_lsb(c11 & mask); // x11
    if(unlikely(!lsb11)) {
        return -1; // singular
    }

    uint64_t c11_reduc = c11 ^ lsb11;
    if(const_col & lsb11) const_col ^= c11_reduc;
    if(c12 & lsb11) c12 ^= c11_reduc;
    if(c13 & lsb11) c13 ^= c11_reduc;
    if(c14 & lsb11) c14 ^= c11_reduc;
    if(c15 & lsb11) c15 ^= c11_reduc;
    if(c16 & lsb11) c16 ^= c11_reduc;
    if(c17 & lsb11) c17 ^= c11_reduc;
    if(c18 & lsb11) c18 ^= c11_reduc;
    if(c19 & lsb11) c19 ^= c11_reduc;
    if(c20 & lsb11) c20 ^= c11_reduc;
    if(c21 & lsb11) c21 ^= c11_reduc;
    if(c22 & lsb11) c22 ^= c11_reduc;
    if(c23 & lsb11) c23 ^= c11_reduc;
    if(c24 & lsb11) c24 ^= c11_reduc;
    if(c25 & lsb11) c25 ^= c11_reduc;
    if(c26 & lsb11) c26 ^= c11_reduc;
    if(c27 & lsb11) c27 ^= c11_reduc;
    if(c28 & lsb11) c28 ^= c11_reduc;
    if(c29 & lsb11) c29 ^= c11_reduc;
    if(c30 & lsb11) c30 ^= c11_reduc;
    mask ^= lsb11;

    uint64_t lsb12 = uint64_t_lsb(c12 & mask); // x12
    if(unlikely(!lsb12)) {
        return -1; // singular
    }

    uint64_t c12_reduc = c12 ^ lsb12;
    if(const_col & lsb12) const_col ^= c12_reduc;
    if(c13 & lsb12) c13 ^= c12_reduc;
    if(c14 & lsb12) c14 ^= c12_reduc;
    if(c15 & lsb12) c15 ^= c12_reduc;
    if(c16 & lsb12) c16 ^= c12_reduc;
    if(c17 & lsb12) c17 ^= c12_reduc;
    if(c18 & lsb12) c18 ^= c12_reduc;
    if(c19 & lsb12) c19 ^= c12_reduc;
    if(c20 & lsb12) c20 ^= c12_reduc;
    if(c21 & lsb12) c21 ^= c12_reduc;
    if(c22 & lsb12) c22 ^= c12_reduc;
    if(c23 & lsb12) c23 ^= c12_reduc;
    if(c24 & lsb12) c24 ^= c12_reduc;
    if(c25 & lsb12) c25 ^= c12_reduc;
    if(c26 & lsb12) c26 ^= c12_reduc;
    if(c27 & lsb12) c27 ^= c12_reduc;
    if(c28 & lsb12) c28 ^= c12_reduc;
    if(c29 & lsb12) c29 ^= c12_reduc;
    if(c30 & lsb12) c30 ^= c12_reduc;
    mask ^= lsb12;

    uint64_t lsb13 = uint64_t_lsb(c13 & mask); // x13
    if(unlikely(!lsb13)) {
        return -1; // singular
    }

    uint64_t c13_reduc = c13 ^ lsb13;
    if(const_col & lsb13) const_col ^= c13_reduc;
    if(c14 & lsb13) c14 ^= c13_reduc;
    if(c15 & lsb13) c15 ^= c13_reduc;
    if(c16 & lsb13) c16 ^= c13_reduc;
    if(c17 & lsb13) c17 ^= c13_reduc;
    if(c18 & lsb13) c18 ^= c13_reduc;
    if(c19 & lsb13) c19 ^= c13_reduc;
    if(c20 & lsb13) c20 ^= c13_reduc;
    if(c21 & lsb13) c21 ^= c13_reduc;
    if(c22 & lsb13) c22 ^= c13_reduc;
    if(c23 & lsb13) c23 ^= c13_reduc;
    if(c24 & lsb13) c24 ^= c13_reduc;
    if(c25 & lsb13) c25 ^= c13_reduc;
    if(c26 & lsb13) c26 ^= c13_reduc;
    if(c27 & lsb13) c27 ^= c13_reduc;
    if(c28 & lsb13) c28 ^= c13_reduc;
    if(c29 & lsb13) c29 ^= c13_reduc;
    if(c30 & lsb13) c30 ^= c13_reduc;
    mask ^= lsb13;

    uint64_t lsb14 = uint64_t_lsb(c14 & mask); // x14
    if(unlikely(!lsb14)) {
        return -1; // singular
    }

    uint64_t c14_reduc = c14 ^ lsb14;
    if(const_col & lsb14) const_col ^= c14_reduc;
    if(c15 & lsb14) c15 ^= c14_reduc;
    if(c16 & lsb14) c16 ^= c14_reduc;
    if(c17 & lsb14) c17 ^= c14_reduc;
    if(c18 & lsb14) c18 ^= c14_reduc;
    if(c19 & lsb14) c19 ^= c14_reduc;
    if(c20 & lsb14) c20 ^= c14_reduc;
    if(c21 & lsb14) c21 ^= c14_reduc;
    if(c22 & lsb14) c22 ^= c14_reduc;
    if(c23 & lsb14) c23 ^= c14_reduc;
    if(c24 & lsb14) c24 ^= c14_reduc;
    if(c25 & lsb14) c25 ^= c14_reduc;
    if(c26 & lsb14) c26 ^= c14_reduc;
    if(c27 & lsb14) c27 ^= c14_reduc;
    if(c28 & lsb14) c28 ^= c14_reduc;
    if(c29 & lsb14) c29 ^= c14_reduc;
    if(c30 & lsb14) c30 ^= c14_reduc;
    mask ^= lsb14;

    uint64_t lsb15 = uint64_t_lsb(c15 & mask); // x15
    if(unlikely(!lsb15)) {
        return -1; // singular
    }

    uint64_t c15_reduc = c15 ^ lsb15;
    if(const_col & lsb15) const_col ^= c15_reduc;
    if(c16 & lsb15) c16 ^= c15_reduc;
    if(c17 & lsb15) c17 ^= c15_reduc;
    if(c18 & lsb15) c18 ^= c15_reduc;
    if(c19 & lsb15) c19 ^= c15_reduc;
    if(c20 & lsb15) c20 ^= c15_reduc;
    if(c21 & lsb15) c21 ^= c15_reduc;
    if(c22 & lsb15) c22 ^= c15_reduc;
    if(c23 & lsb15) c23 ^= c15_reduc;
    if(c24 & lsb15) c24 ^= c15_reduc;
    if(c25 & lsb15) c25 ^= c15_reduc;
    if(c26 & lsb15) c26 ^= c15_reduc;
    if(c27 & lsb15) c27 ^= c15_reduc;
    if(c28 & lsb15) c28 ^= c15_reduc;
    if(c29 & lsb15) c29 ^= c15_reduc;
    if(c30 & lsb15) c30 ^= c15_reduc;
    mask ^= lsb15;

    uint64_t lsb16 = uint64_t_lsb(c16 & mask); // x16
    if(unlikely(!lsb16)) {
        return -1; // singular
    }

    uint64_t c16_reduc = c16 ^ lsb16;
    if(const_col & lsb16) const_col ^= c16_reduc;
    if(c17 & lsb16) c17 ^= c16_reduc;
    if(c18 & lsb16) c18 ^= c16_reduc;
    if(c19 & lsb16) c19 ^= c16_reduc;
    if(c20 & lsb16) c20 ^= c16_reduc;
    if(c21 & lsb16) c21 ^= c16_reduc;
    if(c22 & lsb16) c22 ^= c16_reduc;
    if(c23 & lsb16) c23 ^= c16_reduc;
    if(c24 & lsb16) c24 ^= c16_reduc;
    if(c25 & lsb16) c25 ^= c16_reduc;
    if(c26 & lsb16) c26 ^= c16_reduc;
    if(c27 & lsb16) c27 ^= c16_reduc;
    if(c28 & lsb16) c28 ^= c16_reduc;
    if(c29 & lsb16) c29 ^= c16_reduc;
    if(c30 & lsb16) c30 ^= c16_reduc;
    mask ^= lsb16;

    uint64_t lsb17 = uint64_t_lsb(c17 & mask); // x17
    if(unlikely(!lsb17)) {
        return -1; // singular
    }

    uint64_t c17_reduc = c17 ^ lsb17;
    if(const_col & lsb17) const_col ^= c17_reduc;
    if(c18 & lsb17) c18 ^= c17_reduc;
    if(c19 & lsb17) c19 ^= c17_reduc;
    if(c20 & lsb17) c20 ^= c17_reduc;
    if(c21 & lsb17) c21 ^= c17_reduc;
    if(c22 & lsb17) c22 ^= c17_reduc;
    if(c23 & lsb17) c23 ^= c17_reduc;
    if(c24 & lsb17) c24 ^= c17_reduc;
    if(c25 & lsb17) c25 ^= c17_reduc;
    if(c26 & lsb17) c26 ^= c17_reduc;
    if(c27 & lsb17) c27 ^= c17_reduc;
    if(c28 & lsb17) c28 ^= c17_reduc;
    if(c29 & lsb17) c29 ^= c17_reduc;
    if(c30 & lsb17) c30 ^= c17_reduc;
    mask ^= lsb17;

    uint64_t lsb18 = uint64_t_lsb(c18 & mask); // x18
    if(unlikely(!lsb18)) {
        return -1; // singular
    }

    uint64_t c18_reduc = c18 ^ lsb18;
    if(const_col & lsb18) const_col ^= c18_reduc;
    if(c19 & lsb18) c19 ^= c18_reduc;
    if(c20 & lsb18) c20 ^= c18_reduc;
    if(c21 & lsb18) c21 ^= c18_reduc;
    if(c22 & lsb18) c22 ^= c18_reduc;
    if(c23 & lsb18) c23 ^= c18_reduc;
    if(c24 & lsb18) c24 ^= c18_reduc;
    if(c25 & lsb18) c25 ^= c18_reduc;
    if(c26 & lsb18) c26 ^= c18_reduc;
    if(c27 & lsb18) c27 ^= c18_reduc;
    if(c28 & lsb18) c28 ^= c18_reduc;
    if(c29 & lsb18) c29 ^= c18_reduc;
    if(c30 & lsb18) c30 ^= c18_reduc;
    mask ^= lsb18;

    uint64_t lsb19 = uint64_t_lsb(c19 & mask); // x19
    if(unlikely(!lsb19)) {
        return -1; // singular
    }

    uint64_t c19_reduc = c19 ^ lsb19;
    if(const_col & lsb19) const_col ^= c19_reduc;
    if(c20 & lsb19) c20 ^= c19_reduc;
    if(c21 & lsb19) c21 ^= c19_reduc;
    if(c22 & lsb19) c22 ^= c19_reduc;
    if(c23 & lsb19) c23 ^= c19_reduc;
    if(c24 & lsb19) c24 ^= c19_reduc;
    if(c25 & lsb19) c25 ^= c19_reduc;
    if(c26 & lsb19) c26 ^= c19_reduc;
    if(c27 & lsb19) c27 ^= c19_reduc;
    if(c28 & lsb19) c28 ^= c19_reduc;
    if(c29 & lsb19) c29 ^= c19_reduc;
    if(c30 & lsb19) c30 ^= c19_reduc;
    mask ^= lsb19;

    uint64_t lsb20 = uint64_t_lsb(c20 & mask); // x20
    if(unlikely(!lsb20)) {
        return -1; // singular
    }

    uint64_t c20_reduc = c20 ^ lsb20;
    if(const_col & lsb20) const_col ^= c20_reduc;
    if(c21 & lsb20) c21 ^= c20_reduc;
    if(c22 & lsb20) c22 ^= c20_reduc;
    if(c23 & lsb20) c23 ^= c20_reduc;
    if(c24 & lsb20) c24 ^= c20_reduc;
    if(c25 & lsb20) c25 ^= c20_reduc;
    if(c26 & lsb20) c26 ^= c20_reduc;
    if(c27 & lsb20) c27 ^= c20_reduc;
    if(c28 & lsb20) c28 ^= c20_reduc;
    if(c29 & lsb20) c29 ^= c20_reduc;
    if(c30 & lsb20) c30 ^= c20_reduc;
    mask ^= lsb20;

    uint64_t lsb21 = uint64_t_lsb(c21 & mask); // x21
    if(unlikely(!lsb21)) {
        return -1; // singular
    }

    uint64_t c21_reduc = c21 ^ lsb21;
    if(const_col & lsb21) const_col ^= c21_reduc;
    if(c22 & lsb21) c22 ^= c21_reduc;
    if(c23 & lsb21) c23 ^= c21_reduc;
    if(c24 & lsb21) c24 ^= c21_reduc;
    if(c25 & lsb21) c25 ^= c21_reduc;
    if(c26 & lsb21) c26 ^= c21_reduc;
    if(c27 & lsb21) c27 ^= c21_reduc;
    if(c28 & lsb21) c28 ^= c21_reduc;
    if(c29 & lsb21) c29 ^= c21_reduc;
    if(c30 & lsb21) c30 ^= c21_reduc;
    mask ^= lsb21;

    uint64_t lsb22 = uint64_t_lsb(c22 & mask); // x22
    if(unlikely(!lsb22)) {
        return -1; // singular
    }

    uint64_t c22_reduc = c22 ^ lsb22;
    if(const_col & lsb22) const_col ^= c22_reduc;
    if(c23 & lsb22) c23 ^= c22_reduc;
    if(c24 & lsb22) c24 ^= c22_reduc;
    if(c25 & lsb22) c25 ^= c22_reduc;
    if(c26 & lsb22) c26 ^= c22_reduc;
    if(c27 & lsb22) c27 ^= c22_reduc;
    if(c28 & lsb22) c28 ^= c22_reduc;
    if(c29 & lsb22) c29 ^= c22_reduc;
    if(c30 & lsb22) c30 ^= c22_reduc;
    mask ^= lsb22;

    uint64_t lsb23 = uint64_t_lsb(c23 & mask); // x23
    if(unlikely(!lsb23)) {
        return -1; // singular
    }

    uint64_t c23_reduc = c23 ^ lsb23;
    if(const_col & lsb23) const_col ^= c23_reduc;
    if(c24 & lsb23) c24 ^= c23_reduc;
    if(c25 & lsb23) c25 ^= c23_reduc;
    if(c26 & lsb23) c26 ^= c23_reduc;
    if(c27 & lsb23) c27 ^= c23_reduc;
    if(c28 & lsb23) c28 ^= c23_reduc;
    if(c29 & lsb23) c29 ^= c23_reduc;
    if(c30 & lsb23) c30 ^= c23_reduc;
    mask ^= lsb23;

    uint64_t lsb24 = uint64_t_lsb(c24 & mask); // x24
    if(unlikely(!lsb24)) {
        return -1; // singular
    }

    uint64_t c24_reduc = c24 ^ lsb24;
    if(const_col & lsb24) const_col ^= c24_reduc;
    if(c25 & lsb24) c25 ^= c24_reduc;
    if(c26 & lsb24) c26 ^= c24_reduc;
    if(c27 & lsb24) c27 ^= c24_reduc;
    if(c28 & lsb24) c28 ^= c24_reduc;
    if(c29 & lsb24) c29 ^= c24_reduc;
    if(c30 & lsb24) c30 ^= c24_reduc;
    mask ^= lsb24;

    uint64_t lsb25 = uint64_t_lsb(c25 & mask); // x25
    if(unlikely(!lsb25)) {
        return -1; // singular
    }

    uint64_t c25_reduc = c25 ^ lsb25;
    if(const_col & lsb25) const_col ^= c25_reduc;
    if(c26 & lsb25) c26 ^= c25_reduc;
    if(c27 & lsb25) c27 ^= c25_reduc;
    if(c28 & lsb25) c28 ^= c25_reduc;
    if(c29 & lsb25) c29 ^= c25_reduc;
    if(c30 & lsb25) c30 ^= c25_reduc;
    mask ^= lsb25;

    uint64_t lsb26 = uint64_t_lsb(c26 & mask); // x26
    if(unlikely(!lsb26)) {
        return -1; // singular
    }

    uint64_t c26_reduc = c26 ^ lsb26;
    if(const_col & lsb26) const_col ^= c26_reduc;
    if(c27 & lsb26) c27 ^= c26_reduc;
    if(c28 & lsb26) c28 ^= c26_reduc;
    if(c29 & lsb26) c29 ^= c26_reduc;
    if(c30 & lsb26) c30 ^= c26_reduc;
    mask ^= lsb26;

    uint64_t lsb27 = uint64_t_lsb(c27 & mask); // x27
    if(unlikely(!lsb27)) {
        return -1; // singular
    }

    uint64_t c27_reduc = c27 ^ lsb27;
    if(const_col & lsb27) const_col ^= c27_reduc;
    if(c28 & lsb27) c28 ^= c27_reduc;
    if(c29 & lsb27) c29 ^= c27_reduc;
    if(c30 & lsb27) c30 ^= c27_reduc;
    mask ^= lsb27;

    uint64_t lsb28 = uint64_t_lsb(c28 & mask); // x28
    if(unlikely(!lsb28)) {
        return -1; // singular
    }

    uint64_t c28_reduc = c28 ^ lsb28;
    if(const_col & lsb28) const_col ^= c28_reduc;
    if(c29 & lsb28) c29 ^= c28_reduc;
    if(c30 & lsb28) c30 ^= c28_reduc;
    mask ^= lsb28;

    uint64_t lsb29 = uint64_t_lsb(c29 & mask); // x29
    if(unlikely(!lsb29)) {
        return -1; // singular
    }

    uint64_t c29_reduc = c29 ^ lsb29;
    if(const_col & lsb29) const_col ^= c29_reduc;
    if(c30 & lsb29) c30 ^= c29_reduc;
    mask ^= lsb29;

    uint64_t lsb30 = uint64_t_lsb(c30 & mask); // x30
    if(unlikely(!lsb30)) {
        return -1; // singular
    }

    if(const_col & lsb30) const_col ^= c30 ^ lsb30;
    mask ^= lsb30;

    if(likely(mask & const_col)) { // check system consistency
        return mask & const_col; // not solvable
    }

    uint64_t s = 0x0ULL;
    if(const_col & lsb1) s = uint64_t_toggle_at(s, 0);
    if(const_col & lsb2) s = uint64_t_toggle_at(s, 1);
    if(const_col & lsb3) s = uint64_t_toggle_at(s, 2);
    if(const_col & lsb4) s = uint64_t_toggle_at(s, 3);
    if(const_col & lsb5) s = uint64_t_toggle_at(s, 4);
    if(const_col & lsb6) s = uint64_t_toggle_at(s, 5);
    if(const_col & lsb7) s = uint64_t_toggle_at(s, 6);
    if(const_col & lsb8) s = uint64_t_toggle_at(s, 7);
    if(const_col & lsb9) s = uint64_t_toggle_at(s, 8);
    if(const_col & lsb10) s = uint64_t_toggle_at(s, 9);
    if(const_col & lsb11) s = uint64_t_toggle_at(s, 10);
    if(const_col & lsb12) s = uint64_t_toggle_at(s, 11);
    if(const_col & lsb13) s = uint64_t_toggle_at(s, 12);
    if(const_col & lsb14) s = uint64_t_toggle_at(s, 13);
    if(const_col & lsb15) s = uint64_t_toggle_at(s, 14);
    if(const_col & lsb16) s = uint64_t_toggle_at(s, 15);
    if(const_col & lsb17) s = uint64_t_toggle_at(s, 16);
    if(const_col & lsb18) s = uint64_t_toggle_at(s, 17);
    if(const_col & lsb19) s = uint64_t_toggle_at(s, 18);
    if(const_col & lsb20) s = uint64_t_toggle_at(s, 19);
    if(const_col & lsb21) s = uint64_t_toggle_at(s, 20);
    if(const_col & lsb22) s = uint64_t_toggle_at(s, 21);
    if(const_col & lsb23) s = uint64_t_toggle_at(s, 22);
    if(const_col & lsb24) s = uint64_t_toggle_at(s, 23);
    if(const_col & lsb25) s = uint64_t_toggle_at(s, 24);
    if(const_col & lsb26) s = uint64_t_toggle_at(s, 25);
    if(const_col & lsb27) s = uint64_t_toggle_at(s, 26);
    if(const_col & lsb28) s = uint64_t_toggle_at(s, 27);
    if(const_col & lsb29) s = uint64_t_toggle_at(s, 28);
    if(const_col & lsb30) s = uint64_t_toggle_at(s, 29);
    *sol = s;
    return 0;
}

int64_t
uint64a_gj_v31_generic(const uint64_t m[32], uint64_t* const restrict sol) {
    uint64_t const_col = m[0];
    uint64_t c1 = m[1]; // x1
    uint64_t c2 = m[2]; // x2
    uint64_t c3 = m[3]; // x3
    uint64_t c4 = m[4]; // x4
    uint64_t c5 = m[5]; // x5
    uint64_t c6 = m[6]; // x6
    uint64_t c7 = m[7]; // x7
    uint64_t c8 = m[8]; // x8
    uint64_t c9 = m[9]; // x9
    uint64_t c10 = m[10]; // x10
    uint64_t c11 = m[11]; // x11
    uint64_t c12 = m[12]; // x12
    uint64_t c13 = m[13]; // x13
    uint64_t c14 = m[14]; // x14
    uint64_t c15 = m[15]; // x15
    uint64_t c16 = m[16]; // x16
    uint64_t c17 = m[17]; // x17
    uint64_t c18 = m[18]; // x18
    uint64_t c19 = m[19]; // x19
    uint64_t c20 = m[20]; // x20
    uint64_t c21 = m[21]; // x21
    uint64_t c22 = m[22]; // x22
    uint64_t c23 = m[23]; // x23
    uint64_t c24 = m[24]; // x24
    uint64_t c25 = m[25]; // x25
    uint64_t c26 = m[26]; // x26
    uint64_t c27 = m[27]; // x27
    uint64_t c28 = m[28]; // x28
    uint64_t c29 = m[29]; // x29
    uint64_t c30 = m[30]; // x30
    uint64_t c31 = m[31]; // x31
    uint64_t lsb1 = uint64_t_lsb(c1); // initially all eqs are candidates
    if(unlikely(!lsb1)) {
        return -1; // singular
    }

    uint64_t c1_reduc = c1 ^ lsb1; // for row reduction
    if(const_col & lsb1) const_col ^= c1_reduc;
    if(c2 & lsb1) c2 ^= c1_reduc;
    if(c3 & lsb1) c3 ^= c1_reduc;
    if(c4 & lsb1) c4 ^= c1_reduc;
    if(c5 & lsb1) c5 ^= c1_reduc;
    if(c6 & lsb1) c6 ^= c1_reduc;
    if(c7 & lsb1) c7 ^= c1_reduc;
    if(c8 & lsb1) c8 ^= c1_reduc;
    if(c9 & lsb1) c9 ^= c1_reduc;
    if(c10 & lsb1) c10 ^= c1_reduc;
    if(c11 & lsb1) c11 ^= c1_reduc;
    if(c12 & lsb1) c12 ^= c1_reduc;
    if(c13 & lsb1) c13 ^= c1_reduc;
    if(c14 & lsb1) c14 ^= c1_reduc;
    if(c15 & lsb1) c15 ^= c1_reduc;
    if(c16 & lsb1) c16 ^= c1_reduc;
    if(c17 & lsb1) c17 ^= c1_reduc;
    if(c18 & lsb1) c18 ^= c1_reduc;
    if(c19 & lsb1) c19 ^= c1_reduc;
    if(c20 & lsb1) c20 ^= c1_reduc;
    if(c21 & lsb1) c21 ^= c1_reduc;
    if(c22 & lsb1) c22 ^= c1_reduc;
    if(c23 & lsb1) c23 ^= c1_reduc;
    if(c24 & lsb1) c24 ^= c1_reduc;
    if(c25 & lsb1) c25 ^= c1_reduc;
    if(c26 & lsb1) c26 ^= c1_reduc;
    if(c27 & lsb1) c27 ^= c1_reduc;
    if(c28 & lsb1) c28 ^= c1_reduc;
    if(c29 & lsb1) c29 ^= c1_reduc;
    if(c30 & lsb1) c30 ^= c1_reduc;
    if(c31 & lsb1) c31 ^= c1_reduc;
    uint64_t mask = ~lsb1;

    uint64_t lsb2 = uint64_t_lsb(c2 & mask); // x2
    if(unlikely(!lsb2)) {
        return -1; // singular
    }

    uint64_t c2_reduc = c2 ^ lsb2;
    if(const_col & lsb2) const_col ^= c2_reduc;
    if(c3 & lsb2) c3 ^= c2_reduc;
    if(c4 & lsb2) c4 ^= c2_reduc;
    if(c5 & lsb2) c5 ^= c2_reduc;
    if(c6 & lsb2) c6 ^= c2_reduc;
    if(c7 & lsb2) c7 ^= c2_reduc;
    if(c8 & lsb2) c8 ^= c2_reduc;
    if(c9 & lsb2) c9 ^= c2_reduc;
    if(c10 & lsb2) c10 ^= c2_reduc;
    if(c11 & lsb2) c11 ^= c2_reduc;
    if(c12 & lsb2) c12 ^= c2_reduc;
    if(c13 & lsb2) c13 ^= c2_reduc;
    if(c14 & lsb2) c14 ^= c2_reduc;
    if(c15 & lsb2) c15 ^= c2_reduc;
    if(c16 & lsb2) c16 ^= c2_reduc;
    if(c17 & lsb2) c17 ^= c2_reduc;
    if(c18 & lsb2) c18 ^= c2_reduc;
    if(c19 & lsb2) c19 ^= c2_reduc;
    if(c20 & lsb2) c20 ^= c2_reduc;
    if(c21 & lsb2) c21 ^= c2_reduc;
    if(c22 & lsb2) c22 ^= c2_reduc;
    if(c23 & lsb2) c23 ^= c2_reduc;
    if(c24 & lsb2) c24 ^= c2_reduc;
    if(c25 & lsb2) c25 ^= c2_reduc;
    if(c26 & lsb2) c26 ^= c2_reduc;
    if(c27 & lsb2) c27 ^= c2_reduc;
    if(c28 & lsb2) c28 ^= c2_reduc;
    if(c29 & lsb2) c29 ^= c2_reduc;
    if(c30 & lsb2) c30 ^= c2_reduc;
    if(c31 & lsb2) c31 ^= c2_reduc;
    mask ^= lsb2;

    uint64_t lsb3 = uint64_t_lsb(c3 & mask); // x3
    if(unlikely(!lsb3)) {
        return -1; // singular
    }

    uint64_t c3_reduc = c3 ^ lsb3;
    if(const_col & lsb3) const_col ^= c3_reduc;
    if(c4 & lsb3) c4 ^= c3_reduc;
    if(c5 & lsb3) c5 ^= c3_reduc;
    if(c6 & lsb3) c6 ^= c3_reduc;
    if(c7 & lsb3) c7 ^= c3_reduc;
    if(c8 & lsb3) c8 ^= c3_reduc;
    if(c9 & lsb3) c9 ^= c3_reduc;
    if(c10 & lsb3) c10 ^= c3_reduc;
    if(c11 & lsb3) c11 ^= c3_reduc;
    if(c12 & lsb3) c12 ^= c3_reduc;
    if(c13 & lsb3) c13 ^= c3_reduc;
    if(c14 & lsb3) c14 ^= c3_reduc;
    if(c15 & lsb3) c15 ^= c3_reduc;
    if(c16 & lsb3) c16 ^= c3_reduc;
    if(c17 & lsb3) c17 ^= c3_reduc;
    if(c18 & lsb3) c18 ^= c3_reduc;
    if(c19 & lsb3) c19 ^= c3_reduc;
    if(c20 & lsb3) c20 ^= c3_reduc;
    if(c21 & lsb3) c21 ^= c3_reduc;
    if(c22 & lsb3) c22 ^= c3_reduc;
    if(c23 & lsb3) c23 ^= c3_reduc;
    if(c24 & lsb3) c24 ^= c3_reduc;
    if(c25 & lsb3) c25 ^= c3_reduc;
    if(c26 & lsb3) c26 ^= c3_reduc;
    if(c27 & lsb3) c27 ^= c3_reduc;
    if(c28 & lsb3) c28 ^= c3_reduc;
    if(c29 & lsb3) c29 ^= c3_reduc;
    if(c30 & lsb3) c30 ^= c3_reduc;
    if(c31 & lsb3) c31 ^= c3_reduc;
    mask ^= lsb3;

    uint64_t lsb4 = uint64_t_lsb(c4 & mask); // x4
    if(unlikely(!lsb4)) {
        return -1; // singular
    }

    uint64_t c4_reduc = c4 ^ lsb4;
    if(const_col & lsb4) const_col ^= c4_reduc;
    if(c5 & lsb4) c5 ^= c4_reduc;
    if(c6 & lsb4) c6 ^= c4_reduc;
    if(c7 & lsb4) c7 ^= c4_reduc;
    if(c8 & lsb4) c8 ^= c4_reduc;
    if(c9 & lsb4) c9 ^= c4_reduc;
    if(c10 & lsb4) c10 ^= c4_reduc;
    if(c11 & lsb4) c11 ^= c4_reduc;
    if(c12 & lsb4) c12 ^= c4_reduc;
    if(c13 & lsb4) c13 ^= c4_reduc;
    if(c14 & lsb4) c14 ^= c4_reduc;
    if(c15 & lsb4) c15 ^= c4_reduc;
    if(c16 & lsb4) c16 ^= c4_reduc;
    if(c17 & lsb4) c17 ^= c4_reduc;
    if(c18 & lsb4) c18 ^= c4_reduc;
    if(c19 & lsb4) c19 ^= c4_reduc;
    if(c20 & lsb4) c20 ^= c4_reduc;
    if(c21 & lsb4) c21 ^= c4_reduc;
    if(c22 & lsb4) c22 ^= c4_reduc;
    if(c23 & lsb4) c23 ^= c4_reduc;
    if(c24 & lsb4) c24 ^= c4_reduc;
    if(c25 & lsb4) c25 ^= c4_reduc;
    if(c26 & lsb4) c26 ^= c4_reduc;
    if(c27 & lsb4) c27 ^= c4_reduc;
    if(c28 & lsb4) c28 ^= c4_reduc;
    if(c29 & lsb4) c29 ^= c4_reduc;
    if(c30 & lsb4) c30 ^= c4_reduc;
    if(c31 & lsb4) c31 ^= c4_reduc;
    mask ^= lsb4;

    uint64_t lsb5 = uint64_t_lsb(c5 & mask); // x5
    if(unlikely(!lsb5)) {
        return -1; // singular
    }

    uint64_t c5_reduc = c5 ^ lsb5;
    if(const_col & lsb5) const_col ^= c5_reduc;
    if(c6 & lsb5) c6 ^= c5_reduc;
    if(c7 & lsb5) c7 ^= c5_reduc;
    if(c8 & lsb5) c8 ^= c5_reduc;
    if(c9 & lsb5) c9 ^= c5_reduc;
    if(c10 & lsb5) c10 ^= c5_reduc;
    if(c11 & lsb5) c11 ^= c5_reduc;
    if(c12 & lsb5) c12 ^= c5_reduc;
    if(c13 & lsb5) c13 ^= c5_reduc;
    if(c14 & lsb5) c14 ^= c5_reduc;
    if(c15 & lsb5) c15 ^= c5_reduc;
    if(c16 & lsb5) c16 ^= c5_reduc;
    if(c17 & lsb5) c17 ^= c5_reduc;
    if(c18 & lsb5) c18 ^= c5_reduc;
    if(c19 & lsb5) c19 ^= c5_reduc;
    if(c20 & lsb5) c20 ^= c5_reduc;
    if(c21 & lsb5) c21 ^= c5_reduc;
    if(c22 & lsb5) c22 ^= c5_reduc;
    if(c23 & lsb5) c23 ^= c5_reduc;
    if(c24 & lsb5) c24 ^= c5_reduc;
    if(c25 & lsb5) c25 ^= c5_reduc;
    if(c26 & lsb5) c26 ^= c5_reduc;
    if(c27 & lsb5) c27 ^= c5_reduc;
    if(c28 & lsb5) c28 ^= c5_reduc;
    if(c29 & lsb5) c29 ^= c5_reduc;
    if(c30 & lsb5) c30 ^= c5_reduc;
    if(c31 & lsb5) c31 ^= c5_reduc;
    mask ^= lsb5;

    uint64_t lsb6 = uint64_t_lsb(c6 & mask); // x6
    if(unlikely(!lsb6)) {
        return -1; // singular
    }

    uint64_t c6_reduc = c6 ^ lsb6;
    if(const_col & lsb6) const_col ^= c6_reduc;
    if(c7 & lsb6) c7 ^= c6_reduc;
    if(c8 & lsb6) c8 ^= c6_reduc;
    if(c9 & lsb6) c9 ^= c6_reduc;
    if(c10 & lsb6) c10 ^= c6_reduc;
    if(c11 & lsb6) c11 ^= c6_reduc;
    if(c12 & lsb6) c12 ^= c6_reduc;
    if(c13 & lsb6) c13 ^= c6_reduc;
    if(c14 & lsb6) c14 ^= c6_reduc;
    if(c15 & lsb6) c15 ^= c6_reduc;
    if(c16 & lsb6) c16 ^= c6_reduc;
    if(c17 & lsb6) c17 ^= c6_reduc;
    if(c18 & lsb6) c18 ^= c6_reduc;
    if(c19 & lsb6) c19 ^= c6_reduc;
    if(c20 & lsb6) c20 ^= c6_reduc;
    if(c21 & lsb6) c21 ^= c6_reduc;
    if(c22 & lsb6) c22 ^= c6_reduc;
    if(c23 & lsb6) c23 ^= c6_reduc;
    if(c24 & lsb6) c24 ^= c6_reduc;
    if(c25 & lsb6) c25 ^= c6_reduc;
    if(c26 & lsb6) c26 ^= c6_reduc;
    if(c27 & lsb6) c27 ^= c6_reduc;
    if(c28 & lsb6) c28 ^= c6_reduc;
    if(c29 & lsb6) c29 ^= c6_reduc;
    if(c30 & lsb6) c30 ^= c6_reduc;
    if(c31 & lsb6) c31 ^= c6_reduc;
    mask ^= lsb6;

    uint64_t lsb7 = uint64_t_lsb(c7 & mask); // x7
    if(unlikely(!lsb7)) {
        return -1; // singular
    }

    uint64_t c7_reduc = c7 ^ lsb7;
    if(const_col & lsb7) const_col ^= c7_reduc;
    if(c8 & lsb7) c8 ^= c7_reduc;
    if(c9 & lsb7) c9 ^= c7_reduc;
    if(c10 & lsb7) c10 ^= c7_reduc;
    if(c11 & lsb7) c11 ^= c7_reduc;
    if(c12 & lsb7) c12 ^= c7_reduc;
    if(c13 & lsb7) c13 ^= c7_reduc;
    if(c14 & lsb7) c14 ^= c7_reduc;
    if(c15 & lsb7) c15 ^= c7_reduc;
    if(c16 & lsb7) c16 ^= c7_reduc;
    if(c17 & lsb7) c17 ^= c7_reduc;
    if(c18 & lsb7) c18 ^= c7_reduc;
    if(c19 & lsb7) c19 ^= c7_reduc;
    if(c20 & lsb7) c20 ^= c7_reduc;
    if(c21 & lsb7) c21 ^= c7_reduc;
    if(c22 & lsb7) c22 ^= c7_reduc;
    if(c23 & lsb7) c23 ^= c7_reduc;
    if(c24 & lsb7) c24 ^= c7_reduc;
    if(c25 & lsb7) c25 ^= c7_reduc;
    if(c26 & lsb7) c26 ^= c7_reduc;
    if(c27 & lsb7) c27 ^= c7_reduc;
    if(c28 & lsb7) c28 ^= c7_reduc;
    if(c29 & lsb7) c29 ^= c7_reduc;
    if(c30 & lsb7) c30 ^= c7_reduc;
    if(c31 & lsb7) c31 ^= c7_reduc;
    mask ^= lsb7;

    uint64_t lsb8 = uint64_t_lsb(c8 & mask); // x8
    if(unlikely(!lsb8)) {
        return -1; // singular
    }

    uint64_t c8_reduc = c8 ^ lsb8;
    if(const_col & lsb8) const_col ^= c8_reduc;
    if(c9 & lsb8) c9 ^= c8_reduc;
    if(c10 & lsb8) c10 ^= c8_reduc;
    if(c11 & lsb8) c11 ^= c8_reduc;
    if(c12 & lsb8) c12 ^= c8_reduc;
    if(c13 & lsb8) c13 ^= c8_reduc;
    if(c14 & lsb8) c14 ^= c8_reduc;
    if(c15 & lsb8) c15 ^= c8_reduc;
    if(c16 & lsb8) c16 ^= c8_reduc;
    if(c17 & lsb8) c17 ^= c8_reduc;
    if(c18 & lsb8) c18 ^= c8_reduc;
    if(c19 & lsb8) c19 ^= c8_reduc;
    if(c20 & lsb8) c20 ^= c8_reduc;
    if(c21 & lsb8) c21 ^= c8_reduc;
    if(c22 & lsb8) c22 ^= c8_reduc;
    if(c23 & lsb8) c23 ^= c8_reduc;
    if(c24 & lsb8) c24 ^= c8_reduc;
    if(c25 & lsb8) c25 ^= c8_reduc;
    if(c26 & lsb8) c26 ^= c8_reduc;
    if(c27 & lsb8) c27 ^= c8_reduc;
    if(c28 & lsb8) c28 ^= c8_reduc;
    if(c29 & lsb8) c29 ^= c8_reduc;
    if(c30 & lsb8) c30 ^= c8_reduc;
    if(c31 & lsb8) c31 ^= c8_reduc;
    mask ^= lsb8;

    uint64_t lsb9 = uint64_t_lsb(c9 & mask); // x9
    if(unlikely(!lsb9)) {
        return -1; // singular
    }

    uint64_t c9_reduc = c9 ^ lsb9;
    if(const_col & lsb9) const_col ^= c9_reduc;
    if(c10 & lsb9) c10 ^= c9_reduc;
    if(c11 & lsb9) c11 ^= c9_reduc;
    if(c12 & lsb9) c12 ^= c9_reduc;
    if(c13 & lsb9) c13 ^= c9_reduc;
    if(c14 & lsb9) c14 ^= c9_reduc;
    if(c15 & lsb9) c15 ^= c9_reduc;
    if(c16 & lsb9) c16 ^= c9_reduc;
    if(c17 & lsb9) c17 ^= c9_reduc;
    if(c18 & lsb9) c18 ^= c9_reduc;
    if(c19 & lsb9) c19 ^= c9_reduc;
    if(c20 & lsb9) c20 ^= c9_reduc;
    if(c21 & lsb9) c21 ^= c9_reduc;
    if(c22 & lsb9) c22 ^= c9_reduc;
    if(c23 & lsb9) c23 ^= c9_reduc;
    if(c24 & lsb9) c24 ^= c9_reduc;
    if(c25 & lsb9) c25 ^= c9_reduc;
    if(c26 & lsb9) c26 ^= c9_reduc;
    if(c27 & lsb9) c27 ^= c9_reduc;
    if(c28 & lsb9) c28 ^= c9_reduc;
    if(c29 & lsb9) c29 ^= c9_reduc;
    if(c30 & lsb9) c30 ^= c9_reduc;
    if(c31 & lsb9) c31 ^= c9_reduc;
    mask ^= lsb9;

    uint64_t lsb10 = uint64_t_lsb(c10 & mask); // x10
    if(unlikely(!lsb10)) {
        return -1; // singular
    }

    uint64_t c10_reduc = c10 ^ lsb10;
    if(const_col & lsb10) const_col ^= c10_reduc;
    if(c11 & lsb10) c11 ^= c10_reduc;
    if(c12 & lsb10) c12 ^= c10_reduc;
    if(c13 & lsb10) c13 ^= c10_reduc;
    if(c14 & lsb10) c14 ^= c10_reduc;
    if(c15 & lsb10) c15 ^= c10_reduc;
    if(c16 & lsb10) c16 ^= c10_reduc;
    if(c17 & lsb10) c17 ^= c10_reduc;
    if(c18 & lsb10) c18 ^= c10_reduc;
    if(c19 & lsb10) c19 ^= c10_reduc;
    if(c20 & lsb10) c20 ^= c10_reduc;
    if(c21 & lsb10) c21 ^= c10_reduc;
    if(c22 & lsb10) c22 ^= c10_reduc;
    if(c23 & lsb10) c23 ^= c10_reduc;
    if(c24 & lsb10) c24 ^= c10_reduc;
    if(c25 & lsb10) c25 ^= c10_reduc;
    if(c26 & lsb10) c26 ^= c10_reduc;
    if(c27 & lsb10) c27 ^= c10_reduc;
    if(c28 & lsb10) c28 ^= c10_reduc;
    if(c29 & lsb10) c29 ^= c10_reduc;
    if(c30 & lsb10) c30 ^= c10_reduc;
    if(c31 & lsb10) c31 ^= c10_reduc;
    mask ^= lsb10;

    uint64_t lsb11 = uint64_t_lsb(c11 & mask); // x11
    if(unlikely(!lsb11)) {
        return -1; // singular
    }

    uint64_t c11_reduc = c11 ^ lsb11;
    if(const_col & lsb11) const_col ^= c11_reduc;
    if(c12 & lsb11) c12 ^= c11_reduc;
    if(c13 & lsb11) c13 ^= c11_reduc;
    if(c14 & lsb11) c14 ^= c11_reduc;
    if(c15 & lsb11) c15 ^= c11_reduc;
    if(c16 & lsb11) c16 ^= c11_reduc;
    if(c17 & lsb11) c17 ^= c11_reduc;
    if(c18 & lsb11) c18 ^= c11_reduc;
    if(c19 & lsb11) c19 ^= c11_reduc;
    if(c20 & lsb11) c20 ^= c11_reduc;
    if(c21 & lsb11) c21 ^= c11_reduc;
    if(c22 & lsb11) c22 ^= c11_reduc;
    if(c23 & lsb11) c23 ^= c11_reduc;
    if(c24 & lsb11) c24 ^= c11_reduc;
    if(c25 & lsb11) c25 ^= c11_reduc;
    if(c26 & lsb11) c26 ^= c11_reduc;
    if(c27 & lsb11) c27 ^= c11_reduc;
    if(c28 & lsb11) c28 ^= c11_reduc;
    if(c29 & lsb11) c29 ^= c11_reduc;
    if(c30 & lsb11) c30 ^= c11_reduc;
    if(c31 & lsb11) c31 ^= c11_reduc;
    mask ^= lsb11;

    uint64_t lsb12 = uint64_t_lsb(c12 & mask); // x12
    if(unlikely(!lsb12)) {
        return -1; // singular
    }

    uint64_t c12_reduc = c12 ^ lsb12;
    if(const_col & lsb12) const_col ^= c12_reduc;
    if(c13 & lsb12) c13 ^= c12_reduc;
    if(c14 & lsb12) c14 ^= c12_reduc;
    if(c15 & lsb12) c15 ^= c12_reduc;
    if(c16 & lsb12) c16 ^= c12_reduc;
    if(c17 & lsb12) c17 ^= c12_reduc;
    if(c18 & lsb12) c18 ^= c12_reduc;
    if(c19 & lsb12) c19 ^= c12_reduc;
    if(c20 & lsb12) c20 ^= c12_reduc;
    if(c21 & lsb12) c21 ^= c12_reduc;
    if(c22 & lsb12) c22 ^= c12_reduc;
    if(c23 & lsb12) c23 ^= c12_reduc;
    if(c24 & lsb12) c24 ^= c12_reduc;
    if(c25 & lsb12) c25 ^= c12_reduc;
    if(c26 & lsb12) c26 ^= c12_reduc;
    if(c27 & lsb12) c27 ^= c12_reduc;
    if(c28 & lsb12) c28 ^= c12_reduc;
    if(c29 & lsb12) c29 ^= c12_reduc;
    if(c30 & lsb12) c30 ^= c12_reduc;
    if(c31 & lsb12) c31 ^= c12_reduc;
    mask ^= lsb12;

    uint64_t lsb13 = uint64_t_lsb(c13 & mask); // x13
    if(unlikely(!lsb13)) {
        return -1; // singular
    }

    uint64_t c13_reduc = c13 ^ lsb13;
    if(const_col & lsb13) const_col ^= c13_reduc;
    if(c14 & lsb13) c14 ^= c13_reduc;
    if(c15 & lsb13) c15 ^= c13_reduc;
    if(c16 & lsb13) c16 ^= c13_reduc;
    if(c17 & lsb13) c17 ^= c13_reduc;
    if(c18 & lsb13) c18 ^= c13_reduc;
    if(c19 & lsb13) c19 ^= c13_reduc;
    if(c20 & lsb13) c20 ^= c13_reduc;
    if(c21 & lsb13) c21 ^= c13_reduc;
    if(c22 & lsb13) c22 ^= c13_reduc;
    if(c23 & lsb13) c23 ^= c13_reduc;
    if(c24 & lsb13) c24 ^= c13_reduc;
    if(c25 & lsb13) c25 ^= c13_reduc;
    if(c26 & lsb13) c26 ^= c13_reduc;
    if(c27 & lsb13) c27 ^= c13_reduc;
    if(c28 & lsb13) c28 ^= c13_reduc;
    if(c29 & lsb13) c29 ^= c13_reduc;
    if(c30 & lsb13) c30 ^= c13_reduc;
    if(c31 & lsb13) c31 ^= c13_reduc;
    mask ^= lsb13;

    uint64_t lsb14 = uint64_t_lsb(c14 & mask); // x14
    if(unlikely(!lsb14)) {
        return -1; // singular
    }

    uint64_t c14_reduc = c14 ^ lsb14;
    if(const_col & lsb14) const_col ^= c14_reduc;
    if(c15 & lsb14) c15 ^= c14_reduc;
    if(c16 & lsb14) c16 ^= c14_reduc;
    if(c17 & lsb14) c17 ^= c14_reduc;
    if(c18 & lsb14) c18 ^= c14_reduc;
    if(c19 & lsb14) c19 ^= c14_reduc;
    if(c20 & lsb14) c20 ^= c14_reduc;
    if(c21 & lsb14) c21 ^= c14_reduc;
    if(c22 & lsb14) c22 ^= c14_reduc;
    if(c23 & lsb14) c23 ^= c14_reduc;
    if(c24 & lsb14) c24 ^= c14_reduc;
    if(c25 & lsb14) c25 ^= c14_reduc;
    if(c26 & lsb14) c26 ^= c14_reduc;
    if(c27 & lsb14) c27 ^= c14_reduc;
    if(c28 & lsb14) c28 ^= c14_reduc;
    if(c29 & lsb14) c29 ^= c14_reduc;
    if(c30 & lsb14) c30 ^= c14_reduc;
    if(c31 & lsb14) c31 ^= c14_reduc;
    mask ^= lsb14;

    uint64_t lsb15 = uint64_t_lsb(c15 & mask); // x15
    if(unlikely(!lsb15)) {
        return -1; // singular
    }

    uint64_t c15_reduc = c15 ^ lsb15;
    if(const_col & lsb15) const_col ^= c15_reduc;
    if(c16 & lsb15) c16 ^= c15_reduc;
    if(c17 & lsb15) c17 ^= c15_reduc;
    if(c18 & lsb15) c18 ^= c15_reduc;
    if(c19 & lsb15) c19 ^= c15_reduc;
    if(c20 & lsb15) c20 ^= c15_reduc;
    if(c21 & lsb15) c21 ^= c15_reduc;
    if(c22 & lsb15) c22 ^= c15_reduc;
    if(c23 & lsb15) c23 ^= c15_reduc;
    if(c24 & lsb15) c24 ^= c15_reduc;
    if(c25 & lsb15) c25 ^= c15_reduc;
    if(c26 & lsb15) c26 ^= c15_reduc;
    if(c27 & lsb15) c27 ^= c15_reduc;
    if(c28 & lsb15) c28 ^= c15_reduc;
    if(c29 & lsb15) c29 ^= c15_reduc;
    if(c30 & lsb15) c30 ^= c15_reduc;
    if(c31 & lsb15) c31 ^= c15_reduc;
    mask ^= lsb15;

    uint64_t lsb16 = uint64_t_lsb(c16 & mask); // x16
    if(unlikely(!lsb16)) {
        return -1; // singular
    }

    uint64_t c16_reduc = c16 ^ lsb16;
    if(const_col & lsb16) const_col ^= c16_reduc;
    if(c17 & lsb16) c17 ^= c16_reduc;
    if(c18 & lsb16) c18 ^= c16_reduc;
    if(c19 & lsb16) c19 ^= c16_reduc;
    if(c20 & lsb16) c20 ^= c16_reduc;
    if(c21 & lsb16) c21 ^= c16_reduc;
    if(c22 & lsb16) c22 ^= c16_reduc;
    if(c23 & lsb16) c23 ^= c16_reduc;
    if(c24 & lsb16) c24 ^= c16_reduc;
    if(c25 & lsb16) c25 ^= c16_reduc;
    if(c26 & lsb16) c26 ^= c16_reduc;
    if(c27 & lsb16) c27 ^= c16_reduc;
    if(c28 & lsb16) c28 ^= c16_reduc;
    if(c29 & lsb16) c29 ^= c16_reduc;
    if(c30 & lsb16) c30 ^= c16_reduc;
    if(c31 & lsb16) c31 ^= c16_reduc;
    mask ^= lsb16;

    uint64_t lsb17 = uint64_t_lsb(c17 & mask); // x17
    if(unlikely(!lsb17)) {
        return -1; // singular
    }

    uint64_t c17_reduc = c17 ^ lsb17;
    if(const_col & lsb17) const_col ^= c17_reduc;
    if(c18 & lsb17) c18 ^= c17_reduc;
    if(c19 & lsb17) c19 ^= c17_reduc;
    if(c20 & lsb17) c20 ^= c17_reduc;
    if(c21 & lsb17) c21 ^= c17_reduc;
    if(c22 & lsb17) c22 ^= c17_reduc;
    if(c23 & lsb17) c23 ^= c17_reduc;
    if(c24 & lsb17) c24 ^= c17_reduc;
    if(c25 & lsb17) c25 ^= c17_reduc;
    if(c26 & lsb17) c26 ^= c17_reduc;
    if(c27 & lsb17) c27 ^= c17_reduc;
    if(c28 & lsb17) c28 ^= c17_reduc;
    if(c29 & lsb17) c29 ^= c17_reduc;
    if(c30 & lsb17) c30 ^= c17_reduc;
    if(c31 & lsb17) c31 ^= c17_reduc;
    mask ^= lsb17;

    uint64_t lsb18 = uint64_t_lsb(c18 & mask); // x18
    if(unlikely(!lsb18)) {
        return -1; // singular
    }

    uint64_t c18_reduc = c18 ^ lsb18;
    if(const_col & lsb18) const_col ^= c18_reduc;
    if(c19 & lsb18) c19 ^= c18_reduc;
    if(c20 & lsb18) c20 ^= c18_reduc;
    if(c21 & lsb18) c21 ^= c18_reduc;
    if(c22 & lsb18) c22 ^= c18_reduc;
    if(c23 & lsb18) c23 ^= c18_reduc;
    if(c24 & lsb18) c24 ^= c18_reduc;
    if(c25 & lsb18) c25 ^= c18_reduc;
    if(c26 & lsb18) c26 ^= c18_reduc;
    if(c27 & lsb18) c27 ^= c18_reduc;
    if(c28 & lsb18) c28 ^= c18_reduc;
    if(c29 & lsb18) c29 ^= c18_reduc;
    if(c30 & lsb18) c30 ^= c18_reduc;
    if(c31 & lsb18) c31 ^= c18_reduc;
    mask ^= lsb18;

    uint64_t lsb19 = uint64_t_lsb(c19 & mask); // x19
    if(unlikely(!lsb19)) {
        return -1; // singular
    }

    uint64_t c19_reduc = c19 ^ lsb19;
    if(const_col & lsb19) const_col ^= c19_reduc;
    if(c20 & lsb19) c20 ^= c19_reduc;
    if(c21 & lsb19) c21 ^= c19_reduc;
    if(c22 & lsb19) c22 ^= c19_reduc;
    if(c23 & lsb19) c23 ^= c19_reduc;
    if(c24 & lsb19) c24 ^= c19_reduc;
    if(c25 & lsb19) c25 ^= c19_reduc;
    if(c26 & lsb19) c26 ^= c19_reduc;
    if(c27 & lsb19) c27 ^= c19_reduc;
    if(c28 & lsb19) c28 ^= c19_reduc;
    if(c29 & lsb19) c29 ^= c19_reduc;
    if(c30 & lsb19) c30 ^= c19_reduc;
    if(c31 & lsb19) c31 ^= c19_reduc;
    mask ^= lsb19;

    uint64_t lsb20 = uint64_t_lsb(c20 & mask); // x20
    if(unlikely(!lsb20)) {
        return -1; // singular
    }

    uint64_t c20_reduc = c20 ^ lsb20;
    if(const_col & lsb20) const_col ^= c20_reduc;
    if(c21 & lsb20) c21 ^= c20_reduc;
    if(c22 & lsb20) c22 ^= c20_reduc;
    if(c23 & lsb20) c23 ^= c20_reduc;
    if(c24 & lsb20) c24 ^= c20_reduc;
    if(c25 & lsb20) c25 ^= c20_reduc;
    if(c26 & lsb20) c26 ^= c20_reduc;
    if(c27 & lsb20) c27 ^= c20_reduc;
    if(c28 & lsb20) c28 ^= c20_reduc;
    if(c29 & lsb20) c29 ^= c20_reduc;
    if(c30 & lsb20) c30 ^= c20_reduc;
    if(c31 & lsb20) c31 ^= c20_reduc;
    mask ^= lsb20;

    uint64_t lsb21 = uint64_t_lsb(c21 & mask); // x21
    if(unlikely(!lsb21)) {
        return -1; // singular
    }

    uint64_t c21_reduc = c21 ^ lsb21;
    if(const_col & lsb21) const_col ^= c21_reduc;
    if(c22 & lsb21) c22 ^= c21_reduc;
    if(c23 & lsb21) c23 ^= c21_reduc;
    if(c24 & lsb21) c24 ^= c21_reduc;
    if(c25 & lsb21) c25 ^= c21_reduc;
    if(c26 & lsb21) c26 ^= c21_reduc;
    if(c27 & lsb21) c27 ^= c21_reduc;
    if(c28 & lsb21) c28 ^= c21_reduc;
    if(c29 & lsb21) c29 ^= c21_reduc;
    if(c30 & lsb21) c30 ^= c21_reduc;
    if(c31 & lsb21) c31 ^= c21_reduc;
    mask ^= lsb21;

    uint64_t lsb22 = uint64_t_lsb(c22 & mask); // x22
    if(unlikely(!lsb22)) {
        return -1; // singular
    }

    uint64_t c22_reduc = c22 ^ lsb22;
    if(const_col & lsb22) const_col ^= c22_reduc;
    if(c23 & lsb22) c23 ^= c22_reduc;
    if(c24 & lsb22) c24 ^= c22_reduc;
    if(c25 & lsb22) c25 ^= c22_reduc;
    if(c26 & lsb22) c26 ^= c22_reduc;
    if(c27 & lsb22) c27 ^= c22_reduc;
    if(c28 & lsb22) c28 ^= c22_reduc;
    if(c29 & lsb22) c29 ^= c22_reduc;
    if(c30 & lsb22) c30 ^= c22_reduc;
    if(c31 & lsb22) c31 ^= c22_reduc;
    mask ^= lsb22;

    uint64_t lsb23 = uint64_t_lsb(c23 & mask); // x23
    if(unlikely(!lsb23)) {
        return -1; // singular
    }

    uint64_t c23_reduc = c23 ^ lsb23;
    if(const_col & lsb23) const_col ^= c23_reduc;
    if(c24 & lsb23) c24 ^= c23_reduc;
    if(c25 & lsb23) c25 ^= c23_reduc;
    if(c26 & lsb23) c26 ^= c23_reduc;
    if(c27 & lsb23) c27 ^= c23_reduc;
    if(c28 & lsb23) c28 ^= c23_reduc;
    if(c29 & lsb23) c29 ^= c23_reduc;
    if(c30 & lsb23) c30 ^= c23_reduc;
    if(c31 & lsb23) c31 ^= c23_reduc;
    mask ^= lsb23;

    uint64_t lsb24 = uint64_t_lsb(c24 & mask); // x24
    if(unlikely(!lsb24)) {
        return -1; // singular
    }

    uint64_t c24_reduc = c24 ^ lsb24;
    if(const_col & lsb24) const_col ^= c24_reduc;
    if(c25 & lsb24) c25 ^= c24_reduc;
    if(c26 & lsb24) c26 ^= c24_reduc;
    if(c27 & lsb24) c27 ^= c24_reduc;
    if(c28 & lsb24) c28 ^= c24_reduc;
    if(c29 & lsb24) c29 ^= c24_reduc;
    if(c30 & lsb24) c30 ^= c24_reduc;
    if(c31 & lsb24) c31 ^= c24_reduc;
    mask ^= lsb24;

    uint64_t lsb25 = uint64_t_lsb(c25 & mask); // x25
    if(unlikely(!lsb25)) {
        return -1; // singular
    }

    uint64_t c25_reduc = c25 ^ lsb25;
    if(const_col & lsb25) const_col ^= c25_reduc;
    if(c26 & lsb25) c26 ^= c25_reduc;
    if(c27 & lsb25) c27 ^= c25_reduc;
    if(c28 & lsb25) c28 ^= c25_reduc;
    if(c29 & lsb25) c29 ^= c25_reduc;
    if(c30 & lsb25) c30 ^= c25_reduc;
    if(c31 & lsb25) c31 ^= c25_reduc;
    mask ^= lsb25;

    uint64_t lsb26 = uint64_t_lsb(c26 & mask); // x26
    if(unlikely(!lsb26)) {
        return -1; // singular
    }

    uint64_t c26_reduc = c26 ^ lsb26;
    if(const_col & lsb26) const_col ^= c26_reduc;
    if(c27 & lsb26) c27 ^= c26_reduc;
    if(c28 & lsb26) c28 ^= c26_reduc;
    if(c29 & lsb26) c29 ^= c26_reduc;
    if(c30 & lsb26) c30 ^= c26_reduc;
    if(c31 & lsb26) c31 ^= c26_reduc;
    mask ^= lsb26;

    uint64_t lsb27 = uint64_t_lsb(c27 & mask); // x27
    if(unlikely(!lsb27)) {
        return -1; // singular
    }

    uint64_t c27_reduc = c27 ^ lsb27;
    if(const_col & lsb27) const_col ^= c27_reduc;
    if(c28 & lsb27) c28 ^= c27_reduc;
    if(c29 & lsb27) c29 ^= c27_reduc;
    if(c30 & lsb27) c30 ^= c27_reduc;
    if(c31 & lsb27) c31 ^= c27_reduc;
    mask ^= lsb27;

    uint64_t lsb28 = uint64_t_lsb(c28 & mask); // x28
    if(unlikely(!lsb28)) {
        return -1; // singular
    }

    uint64_t c28_reduc = c28 ^ lsb28;
    if(const_col & lsb28) const_col ^= c28_reduc;
    if(c29 & lsb28) c29 ^= c28_reduc;
    if(c30 & lsb28) c30 ^= c28_reduc;
    if(c31 & lsb28) c31 ^= c28_reduc;
    mask ^= lsb28;

    uint64_t lsb29 = uint64_t_lsb(c29 & mask); // x29
    if(unlikely(!lsb29)) {
        return -1; // singular
    }

    uint64_t c29_reduc = c29 ^ lsb29;
    if(const_col & lsb29) const_col ^= c29_reduc;
    if(c30 & lsb29) c30 ^= c29_reduc;
    if(c31 & lsb29) c31 ^= c29_reduc;
    mask ^= lsb29;

    uint64_t lsb30 = uint64_t_lsb(c30 & mask); // x30
    if(unlikely(!lsb30)) {
        return -1; // singular
    }

    uint64_t c30_reduc = c30 ^ lsb30;
    if(const_col & lsb30) const_col ^= c30_reduc;
    if(c31 & lsb30) c31 ^= c30_reduc;
    mask ^= lsb30;

    uint64_t lsb31 = uint64_t_lsb(c31 & mask); // x31
    if(unlikely(!lsb31)) {
        return -1; // singular
    }

    if(const_col & lsb31) const_col ^= c31 ^ lsb31;
    mask ^= lsb31;

    if(likely(mask & const_col)) { // check system consistency
        return mask & const_col; // not solvable
    }

    uint64_t s = 0x0ULL;
    if(const_col & lsb1) s = uint64_t_toggle_at(s, 0);
    if(const_col & lsb2) s = uint64_t_toggle_at(s, 1);
    if(const_col & lsb3) s = uint64_t_toggle_at(s, 2);
    if(const_col & lsb4) s = uint64_t_toggle_at(s, 3);
    if(const_col & lsb5) s = uint64_t_toggle_at(s, 4);
    if(const_col & lsb6) s = uint64_t_toggle_at(s, 5);
    if(const_col & lsb7) s = uint64_t_toggle_at(s, 6);
    if(const_col & lsb8) s = uint64_t_toggle_at(s, 7);
    if(const_col & lsb9) s = uint64_t_toggle_at(s, 8);
    if(const_col & lsb10) s = uint64_t_toggle_at(s, 9);
    if(const_col & lsb11) s = uint64_t_toggle_at(s, 10);
    if(const_col & lsb12) s = uint64_t_toggle_at(s, 11);
    if(const_col & lsb13) s = uint64_t_toggle_at(s, 12);
    if(const_col & lsb14) s = uint64_t_toggle_at(s, 13);
    if(const_col & lsb15) s = uint64_t_toggle_at(s, 14);
    if(const_col & lsb16) s = uint64_t_toggle_at(s, 15);
    if(const_col & lsb17) s = uint64_t_toggle_at(s, 16);
    if(const_col & lsb18) s = uint64_t_toggle_at(s, 17);
    if(const_col & lsb19) s = uint64_t_toggle_at(s, 18);
    if(const_col & lsb20) s = uint64_t_toggle_at(s, 19);
    if(const_col & lsb21) s = uint64_t_toggle_at(s, 20);
    if(const_col & lsb22) s = uint64_t_toggle_at(s, 21);
    if(const_col & lsb23) s = uint64_t_toggle_at(s, 22);
    if(const_col & lsb24) s = uint64_t_toggle_at(s, 23);
    if(const_col & lsb25) s = uint64_t_toggle_at(s, 24);
    if(const_col & lsb26) s = uint64_t_toggle_at(s, 25);
    if(const_col & lsb27) s = uint64_t_toggle_at(s, 26);
    if(const_col & lsb28) s = uint64_t_toggle_at(s, 27);
    if(const_col & lsb29) s = uint64_t_toggle_at(s, 28);
    if(const_col & lsb30) s = uint64_t_toggle_at(s, 29);
    if(const_col & lsb31) s = uint64_t_toggle_at(s, 30);
    *sol = s;
    return 0;
}

int64_t
uint64a_gj_v32_generic(const uint64_t m[33], uint64_t* const restrict sol) {
    uint64_t const_col = m[0];
    uint64_t c1 = m[1]; // x1
    uint64_t c2 = m[2]; // x2
    uint64_t c3 = m[3]; // x3
    uint64_t c4 = m[4]; // x4
    uint64_t c5 = m[5]; // x5
    uint64_t c6 = m[6]; // x6
    uint64_t c7 = m[7]; // x7
    uint64_t c8 = m[8]; // x8
    uint64_t c9 = m[9]; // x9
    uint64_t c10 = m[10]; // x10
    uint64_t c11 = m[11]; // x11
    uint64_t c12 = m[12]; // x12
    uint64_t c13 = m[13]; // x13
    uint64_t c14 = m[14]; // x14
    uint64_t c15 = m[15]; // x15
    uint64_t c16 = m[16]; // x16
    uint64_t c17 = m[17]; // x17
    uint64_t c18 = m[18]; // x18
    uint64_t c19 = m[19]; // x19
    uint64_t c20 = m[20]; // x20
    uint64_t c21 = m[21]; // x21
    uint64_t c22 = m[22]; // x22
    uint64_t c23 = m[23]; // x23
    uint64_t c24 = m[24]; // x24
    uint64_t c25 = m[25]; // x25
    uint64_t c26 = m[26]; // x26
    uint64_t c27 = m[27]; // x27
    uint64_t c28 = m[28]; // x28
    uint64_t c29 = m[29]; // x29
    uint64_t c30 = m[30]; // x30
    uint64_t c31 = m[31]; // x31
    uint64_t c32 = m[32]; // x32
    uint64_t lsb1 = uint64_t_lsb(c1); // initially all eqs are candidates
    if(unlikely(!lsb1)) {
        return -1; // singular
    }

    uint64_t c1_reduc = c1 ^ lsb1; // for row reduction
    if(const_col & lsb1) const_col ^= c1_reduc;
    if(c2 & lsb1) c2 ^= c1_reduc;
    if(c3 & lsb1) c3 ^= c1_reduc;
    if(c4 & lsb1) c4 ^= c1_reduc;
    if(c5 & lsb1) c5 ^= c1_reduc;
    if(c6 & lsb1) c6 ^= c1_reduc;
    if(c7 & lsb1) c7 ^= c1_reduc;
    if(c8 & lsb1) c8 ^= c1_reduc;
    if(c9 & lsb1) c9 ^= c1_reduc;
    if(c10 & lsb1) c10 ^= c1_reduc;
    if(c11 & lsb1) c11 ^= c1_reduc;
    if(c12 & lsb1) c12 ^= c1_reduc;
    if(c13 & lsb1) c13 ^= c1_reduc;
    if(c14 & lsb1) c14 ^= c1_reduc;
    if(c15 & lsb1) c15 ^= c1_reduc;
    if(c16 & lsb1) c16 ^= c1_reduc;
    if(c17 & lsb1) c17 ^= c1_reduc;
    if(c18 & lsb1) c18 ^= c1_reduc;
    if(c19 & lsb1) c19 ^= c1_reduc;
    if(c20 & lsb1) c20 ^= c1_reduc;
    if(c21 & lsb1) c21 ^= c1_reduc;
    if(c22 & lsb1) c22 ^= c1_reduc;
    if(c23 & lsb1) c23 ^= c1_reduc;
    if(c24 & lsb1) c24 ^= c1_reduc;
    if(c25 & lsb1) c25 ^= c1_reduc;
    if(c26 & lsb1) c26 ^= c1_reduc;
    if(c27 & lsb1) c27 ^= c1_reduc;
    if(c28 & lsb1) c28 ^= c1_reduc;
    if(c29 & lsb1) c29 ^= c1_reduc;
    if(c30 & lsb1) c30 ^= c1_reduc;
    if(c31 & lsb1) c31 ^= c1_reduc;
    if(c32 & lsb1) c32 ^= c1_reduc;
    uint64_t mask = ~lsb1;

    uint64_t lsb2 = uint64_t_lsb(c2 & mask); // x2
    if(unlikely(!lsb2)) {
        return -1; // singular
    }

    uint64_t c2_reduc = c2 ^ lsb2;
    if(const_col & lsb2) const_col ^= c2_reduc;
    if(c3 & lsb2) c3 ^= c2_reduc;
    if(c4 & lsb2) c4 ^= c2_reduc;
    if(c5 & lsb2) c5 ^= c2_reduc;
    if(c6 & lsb2) c6 ^= c2_reduc;
    if(c7 & lsb2) c7 ^= c2_reduc;
    if(c8 & lsb2) c8 ^= c2_reduc;
    if(c9 & lsb2) c9 ^= c2_reduc;
    if(c10 & lsb2) c10 ^= c2_reduc;
    if(c11 & lsb2) c11 ^= c2_reduc;
    if(c12 & lsb2) c12 ^= c2_reduc;
    if(c13 & lsb2) c13 ^= c2_reduc;
    if(c14 & lsb2) c14 ^= c2_reduc;
    if(c15 & lsb2) c15 ^= c2_reduc;
    if(c16 & lsb2) c16 ^= c2_reduc;
    if(c17 & lsb2) c17 ^= c2_reduc;
    if(c18 & lsb2) c18 ^= c2_reduc;
    if(c19 & lsb2) c19 ^= c2_reduc;
    if(c20 & lsb2) c20 ^= c2_reduc;
    if(c21 & lsb2) c21 ^= c2_reduc;
    if(c22 & lsb2) c22 ^= c2_reduc;
    if(c23 & lsb2) c23 ^= c2_reduc;
    if(c24 & lsb2) c24 ^= c2_reduc;
    if(c25 & lsb2) c25 ^= c2_reduc;
    if(c26 & lsb2) c26 ^= c2_reduc;
    if(c27 & lsb2) c27 ^= c2_reduc;
    if(c28 & lsb2) c28 ^= c2_reduc;
    if(c29 & lsb2) c29 ^= c2_reduc;
    if(c30 & lsb2) c30 ^= c2_reduc;
    if(c31 & lsb2) c31 ^= c2_reduc;
    if(c32 & lsb2) c32 ^= c2_reduc;
    mask ^= lsb2;

    uint64_t lsb3 = uint64_t_lsb(c3 & mask); // x3
    if(unlikely(!lsb3)) {
        return -1; // singular
    }

    uint64_t c3_reduc = c3 ^ lsb3;
    if(const_col & lsb3) const_col ^= c3_reduc;
    if(c4 & lsb3) c4 ^= c3_reduc;
    if(c5 & lsb3) c5 ^= c3_reduc;
    if(c6 & lsb3) c6 ^= c3_reduc;
    if(c7 & lsb3) c7 ^= c3_reduc;
    if(c8 & lsb3) c8 ^= c3_reduc;
    if(c9 & lsb3) c9 ^= c3_reduc;
    if(c10 & lsb3) c10 ^= c3_reduc;
    if(c11 & lsb3) c11 ^= c3_reduc;
    if(c12 & lsb3) c12 ^= c3_reduc;
    if(c13 & lsb3) c13 ^= c3_reduc;
    if(c14 & lsb3) c14 ^= c3_reduc;
    if(c15 & lsb3) c15 ^= c3_reduc;
    if(c16 & lsb3) c16 ^= c3_reduc;
    if(c17 & lsb3) c17 ^= c3_reduc;
    if(c18 & lsb3) c18 ^= c3_reduc;
    if(c19 & lsb3) c19 ^= c3_reduc;
    if(c20 & lsb3) c20 ^= c3_reduc;
    if(c21 & lsb3) c21 ^= c3_reduc;
    if(c22 & lsb3) c22 ^= c3_reduc;
    if(c23 & lsb3) c23 ^= c3_reduc;
    if(c24 & lsb3) c24 ^= c3_reduc;
    if(c25 & lsb3) c25 ^= c3_reduc;
    if(c26 & lsb3) c26 ^= c3_reduc;
    if(c27 & lsb3) c27 ^= c3_reduc;
    if(c28 & lsb3) c28 ^= c3_reduc;
    if(c29 & lsb3) c29 ^= c3_reduc;
    if(c30 & lsb3) c30 ^= c3_reduc;
    if(c31 & lsb3) c31 ^= c3_reduc;
    if(c32 & lsb3) c32 ^= c3_reduc;
    mask ^= lsb3;

    uint64_t lsb4 = uint64_t_lsb(c4 & mask); // x4
    if(unlikely(!lsb4)) {
        return -1; // singular
    }

    uint64_t c4_reduc = c4 ^ lsb4;
    if(const_col & lsb4) const_col ^= c4_reduc;
    if(c5 & lsb4) c5 ^= c4_reduc;
    if(c6 & lsb4) c6 ^= c4_reduc;
    if(c7 & lsb4) c7 ^= c4_reduc;
    if(c8 & lsb4) c8 ^= c4_reduc;
    if(c9 & lsb4) c9 ^= c4_reduc;
    if(c10 & lsb4) c10 ^= c4_reduc;
    if(c11 & lsb4) c11 ^= c4_reduc;
    if(c12 & lsb4) c12 ^= c4_reduc;
    if(c13 & lsb4) c13 ^= c4_reduc;
    if(c14 & lsb4) c14 ^= c4_reduc;
    if(c15 & lsb4) c15 ^= c4_reduc;
    if(c16 & lsb4) c16 ^= c4_reduc;
    if(c17 & lsb4) c17 ^= c4_reduc;
    if(c18 & lsb4) c18 ^= c4_reduc;
    if(c19 & lsb4) c19 ^= c4_reduc;
    if(c20 & lsb4) c20 ^= c4_reduc;
    if(c21 & lsb4) c21 ^= c4_reduc;
    if(c22 & lsb4) c22 ^= c4_reduc;
    if(c23 & lsb4) c23 ^= c4_reduc;
    if(c24 & lsb4) c24 ^= c4_reduc;
    if(c25 & lsb4) c25 ^= c4_reduc;
    if(c26 & lsb4) c26 ^= c4_reduc;
    if(c27 & lsb4) c27 ^= c4_reduc;
    if(c28 & lsb4) c28 ^= c4_reduc;
    if(c29 & lsb4) c29 ^= c4_reduc;
    if(c30 & lsb4) c30 ^= c4_reduc;
    if(c31 & lsb4) c31 ^= c4_reduc;
    if(c32 & lsb4) c32 ^= c4_reduc;
    mask ^= lsb4;

    uint64_t lsb5 = uint64_t_lsb(c5 & mask); // x5
    if(unlikely(!lsb5)) {
        return -1; // singular
    }

    uint64_t c5_reduc = c5 ^ lsb5;
    if(const_col & lsb5) const_col ^= c5_reduc;
    if(c6 & lsb5) c6 ^= c5_reduc;
    if(c7 & lsb5) c7 ^= c5_reduc;
    if(c8 & lsb5) c8 ^= c5_reduc;
    if(c9 & lsb5) c9 ^= c5_reduc;
    if(c10 & lsb5) c10 ^= c5_reduc;
    if(c11 & lsb5) c11 ^= c5_reduc;
    if(c12 & lsb5) c12 ^= c5_reduc;
    if(c13 & lsb5) c13 ^= c5_reduc;
    if(c14 & lsb5) c14 ^= c5_reduc;
    if(c15 & lsb5) c15 ^= c5_reduc;
    if(c16 & lsb5) c16 ^= c5_reduc;
    if(c17 & lsb5) c17 ^= c5_reduc;
    if(c18 & lsb5) c18 ^= c5_reduc;
    if(c19 & lsb5) c19 ^= c5_reduc;
    if(c20 & lsb5) c20 ^= c5_reduc;
    if(c21 & lsb5) c21 ^= c5_reduc;
    if(c22 & lsb5) c22 ^= c5_reduc;
    if(c23 & lsb5) c23 ^= c5_reduc;
    if(c24 & lsb5) c24 ^= c5_reduc;
    if(c25 & lsb5) c25 ^= c5_reduc;
    if(c26 & lsb5) c26 ^= c5_reduc;
    if(c27 & lsb5) c27 ^= c5_reduc;
    if(c28 & lsb5) c28 ^= c5_reduc;
    if(c29 & lsb5) c29 ^= c5_reduc;
    if(c30 & lsb5) c30 ^= c5_reduc;
    if(c31 & lsb5) c31 ^= c5_reduc;
    if(c32 & lsb5) c32 ^= c5_reduc;
    mask ^= lsb5;

    uint64_t lsb6 = uint64_t_lsb(c6 & mask); // x6
    if(unlikely(!lsb6)) {
        return -1; // singular
    }

    uint64_t c6_reduc = c6 ^ lsb6;
    if(const_col & lsb6) const_col ^= c6_reduc;
    if(c7 & lsb6) c7 ^= c6_reduc;
    if(c8 & lsb6) c8 ^= c6_reduc;
    if(c9 & lsb6) c9 ^= c6_reduc;
    if(c10 & lsb6) c10 ^= c6_reduc;
    if(c11 & lsb6) c11 ^= c6_reduc;
    if(c12 & lsb6) c12 ^= c6_reduc;
    if(c13 & lsb6) c13 ^= c6_reduc;
    if(c14 & lsb6) c14 ^= c6_reduc;
    if(c15 & lsb6) c15 ^= c6_reduc;
    if(c16 & lsb6) c16 ^= c6_reduc;
    if(c17 & lsb6) c17 ^= c6_reduc;
    if(c18 & lsb6) c18 ^= c6_reduc;
    if(c19 & lsb6) c19 ^= c6_reduc;
    if(c20 & lsb6) c20 ^= c6_reduc;
    if(c21 & lsb6) c21 ^= c6_reduc;
    if(c22 & lsb6) c22 ^= c6_reduc;
    if(c23 & lsb6) c23 ^= c6_reduc;
    if(c24 & lsb6) c24 ^= c6_reduc;
    if(c25 & lsb6) c25 ^= c6_reduc;
    if(c26 & lsb6) c26 ^= c6_reduc;
    if(c27 & lsb6) c27 ^= c6_reduc;
    if(c28 & lsb6) c28 ^= c6_reduc;
    if(c29 & lsb6) c29 ^= c6_reduc;
    if(c30 & lsb6) c30 ^= c6_reduc;
    if(c31 & lsb6) c31 ^= c6_reduc;
    if(c32 & lsb6) c32 ^= c6_reduc;
    mask ^= lsb6;

    uint64_t lsb7 = uint64_t_lsb(c7 & mask); // x7
    if(unlikely(!lsb7)) {
        return -1; // singular
    }

    uint64_t c7_reduc = c7 ^ lsb7;
    if(const_col & lsb7) const_col ^= c7_reduc;
    if(c8 & lsb7) c8 ^= c7_reduc;
    if(c9 & lsb7) c9 ^= c7_reduc;
    if(c10 & lsb7) c10 ^= c7_reduc;
    if(c11 & lsb7) c11 ^= c7_reduc;
    if(c12 & lsb7) c12 ^= c7_reduc;
    if(c13 & lsb7) c13 ^= c7_reduc;
    if(c14 & lsb7) c14 ^= c7_reduc;
    if(c15 & lsb7) c15 ^= c7_reduc;
    if(c16 & lsb7) c16 ^= c7_reduc;
    if(c17 & lsb7) c17 ^= c7_reduc;
    if(c18 & lsb7) c18 ^= c7_reduc;
    if(c19 & lsb7) c19 ^= c7_reduc;
    if(c20 & lsb7) c20 ^= c7_reduc;
    if(c21 & lsb7) c21 ^= c7_reduc;
    if(c22 & lsb7) c22 ^= c7_reduc;
    if(c23 & lsb7) c23 ^= c7_reduc;
    if(c24 & lsb7) c24 ^= c7_reduc;
    if(c25 & lsb7) c25 ^= c7_reduc;
    if(c26 & lsb7) c26 ^= c7_reduc;
    if(c27 & lsb7) c27 ^= c7_reduc;
    if(c28 & lsb7) c28 ^= c7_reduc;
    if(c29 & lsb7) c29 ^= c7_reduc;
    if(c30 & lsb7) c30 ^= c7_reduc;
    if(c31 & lsb7) c31 ^= c7_reduc;
    if(c32 & lsb7) c32 ^= c7_reduc;
    mask ^= lsb7;

    uint64_t lsb8 = uint64_t_lsb(c8 & mask); // x8
    if(unlikely(!lsb8)) {
        return -1; // singular
    }

    uint64_t c8_reduc = c8 ^ lsb8;
    if(const_col & lsb8) const_col ^= c8_reduc;
    if(c9 & lsb8) c9 ^= c8_reduc;
    if(c10 & lsb8) c10 ^= c8_reduc;
    if(c11 & lsb8) c11 ^= c8_reduc;
    if(c12 & lsb8) c12 ^= c8_reduc;
    if(c13 & lsb8) c13 ^= c8_reduc;
    if(c14 & lsb8) c14 ^= c8_reduc;
    if(c15 & lsb8) c15 ^= c8_reduc;
    if(c16 & lsb8) c16 ^= c8_reduc;
    if(c17 & lsb8) c17 ^= c8_reduc;
    if(c18 & lsb8) c18 ^= c8_reduc;
    if(c19 & lsb8) c19 ^= c8_reduc;
    if(c20 & lsb8) c20 ^= c8_reduc;
    if(c21 & lsb8) c21 ^= c8_reduc;
    if(c22 & lsb8) c22 ^= c8_reduc;
    if(c23 & lsb8) c23 ^= c8_reduc;
    if(c24 & lsb8) c24 ^= c8_reduc;
    if(c25 & lsb8) c25 ^= c8_reduc;
    if(c26 & lsb8) c26 ^= c8_reduc;
    if(c27 & lsb8) c27 ^= c8_reduc;
    if(c28 & lsb8) c28 ^= c8_reduc;
    if(c29 & lsb8) c29 ^= c8_reduc;
    if(c30 & lsb8) c30 ^= c8_reduc;
    if(c31 & lsb8) c31 ^= c8_reduc;
    if(c32 & lsb8) c32 ^= c8_reduc;
    mask ^= lsb8;

    uint64_t lsb9 = uint64_t_lsb(c9 & mask); // x9
    if(unlikely(!lsb9)) {
        return -1; // singular
    }

    uint64_t c9_reduc = c9 ^ lsb9;
    if(const_col & lsb9) const_col ^= c9_reduc;
    if(c10 & lsb9) c10 ^= c9_reduc;
    if(c11 & lsb9) c11 ^= c9_reduc;
    if(c12 & lsb9) c12 ^= c9_reduc;
    if(c13 & lsb9) c13 ^= c9_reduc;
    if(c14 & lsb9) c14 ^= c9_reduc;
    if(c15 & lsb9) c15 ^= c9_reduc;
    if(c16 & lsb9) c16 ^= c9_reduc;
    if(c17 & lsb9) c17 ^= c9_reduc;
    if(c18 & lsb9) c18 ^= c9_reduc;
    if(c19 & lsb9) c19 ^= c9_reduc;
    if(c20 & lsb9) c20 ^= c9_reduc;
    if(c21 & lsb9) c21 ^= c9_reduc;
    if(c22 & lsb9) c22 ^= c9_reduc;
    if(c23 & lsb9) c23 ^= c9_reduc;
    if(c24 & lsb9) c24 ^= c9_reduc;
    if(c25 & lsb9) c25 ^= c9_reduc;
    if(c26 & lsb9) c26 ^= c9_reduc;
    if(c27 & lsb9) c27 ^= c9_reduc;
    if(c28 & lsb9) c28 ^= c9_reduc;
    if(c29 & lsb9) c29 ^= c9_reduc;
    if(c30 & lsb9) c30 ^= c9_reduc;
    if(c31 & lsb9) c31 ^= c9_reduc;
    if(c32 & lsb9) c32 ^= c9_reduc;
    mask ^= lsb9;

    uint64_t lsb10 = uint64_t_lsb(c10 & mask); // x10
    if(unlikely(!lsb10)) {
        return -1; // singular
    }

    uint64_t c10_reduc = c10 ^ lsb10;
    if(const_col & lsb10) const_col ^= c10_reduc;
    if(c11 & lsb10) c11 ^= c10_reduc;
    if(c12 & lsb10) c12 ^= c10_reduc;
    if(c13 & lsb10) c13 ^= c10_reduc;
    if(c14 & lsb10) c14 ^= c10_reduc;
    if(c15 & lsb10) c15 ^= c10_reduc;
    if(c16 & lsb10) c16 ^= c10_reduc;
    if(c17 & lsb10) c17 ^= c10_reduc;
    if(c18 & lsb10) c18 ^= c10_reduc;
    if(c19 & lsb10) c19 ^= c10_reduc;
    if(c20 & lsb10) c20 ^= c10_reduc;
    if(c21 & lsb10) c21 ^= c10_reduc;
    if(c22 & lsb10) c22 ^= c10_reduc;
    if(c23 & lsb10) c23 ^= c10_reduc;
    if(c24 & lsb10) c24 ^= c10_reduc;
    if(c25 & lsb10) c25 ^= c10_reduc;
    if(c26 & lsb10) c26 ^= c10_reduc;
    if(c27 & lsb10) c27 ^= c10_reduc;
    if(c28 & lsb10) c28 ^= c10_reduc;
    if(c29 & lsb10) c29 ^= c10_reduc;
    if(c30 & lsb10) c30 ^= c10_reduc;
    if(c31 & lsb10) c31 ^= c10_reduc;
    if(c32 & lsb10) c32 ^= c10_reduc;
    mask ^= lsb10;

    uint64_t lsb11 = uint64_t_lsb(c11 & mask); // x11
    if(unlikely(!lsb11)) {
        return -1; // singular
    }

    uint64_t c11_reduc = c11 ^ lsb11;
    if(const_col & lsb11) const_col ^= c11_reduc;
    if(c12 & lsb11) c12 ^= c11_reduc;
    if(c13 & lsb11) c13 ^= c11_reduc;
    if(c14 & lsb11) c14 ^= c11_reduc;
    if(c15 & lsb11) c15 ^= c11_reduc;
    if(c16 & lsb11) c16 ^= c11_reduc;
    if(c17 & lsb11) c17 ^= c11_reduc;
    if(c18 & lsb11) c18 ^= c11_reduc;
    if(c19 & lsb11) c19 ^= c11_reduc;
    if(c20 & lsb11) c20 ^= c11_reduc;
    if(c21 & lsb11) c21 ^= c11_reduc;
    if(c22 & lsb11) c22 ^= c11_reduc;
    if(c23 & lsb11) c23 ^= c11_reduc;
    if(c24 & lsb11) c24 ^= c11_reduc;
    if(c25 & lsb11) c25 ^= c11_reduc;
    if(c26 & lsb11) c26 ^= c11_reduc;
    if(c27 & lsb11) c27 ^= c11_reduc;
    if(c28 & lsb11) c28 ^= c11_reduc;
    if(c29 & lsb11) c29 ^= c11_reduc;
    if(c30 & lsb11) c30 ^= c11_reduc;
    if(c31 & lsb11) c31 ^= c11_reduc;
    if(c32 & lsb11) c32 ^= c11_reduc;
    mask ^= lsb11;

    uint64_t lsb12 = uint64_t_lsb(c12 & mask); // x12
    if(unlikely(!lsb12)) {
        return -1; // singular
    }

    uint64_t c12_reduc = c12 ^ lsb12;
    if(const_col & lsb12) const_col ^= c12_reduc;
    if(c13 & lsb12) c13 ^= c12_reduc;
    if(c14 & lsb12) c14 ^= c12_reduc;
    if(c15 & lsb12) c15 ^= c12_reduc;
    if(c16 & lsb12) c16 ^= c12_reduc;
    if(c17 & lsb12) c17 ^= c12_reduc;
    if(c18 & lsb12) c18 ^= c12_reduc;
    if(c19 & lsb12) c19 ^= c12_reduc;
    if(c20 & lsb12) c20 ^= c12_reduc;
    if(c21 & lsb12) c21 ^= c12_reduc;
    if(c22 & lsb12) c22 ^= c12_reduc;
    if(c23 & lsb12) c23 ^= c12_reduc;
    if(c24 & lsb12) c24 ^= c12_reduc;
    if(c25 & lsb12) c25 ^= c12_reduc;
    if(c26 & lsb12) c26 ^= c12_reduc;
    if(c27 & lsb12) c27 ^= c12_reduc;
    if(c28 & lsb12) c28 ^= c12_reduc;
    if(c29 & lsb12) c29 ^= c12_reduc;
    if(c30 & lsb12) c30 ^= c12_reduc;
    if(c31 & lsb12) c31 ^= c12_reduc;
    if(c32 & lsb12) c32 ^= c12_reduc;
    mask ^= lsb12;

    uint64_t lsb13 = uint64_t_lsb(c13 & mask); // x13
    if(unlikely(!lsb13)) {
        return -1; // singular
    }

    uint64_t c13_reduc = c13 ^ lsb13;
    if(const_col & lsb13) const_col ^= c13_reduc;
    if(c14 & lsb13) c14 ^= c13_reduc;
    if(c15 & lsb13) c15 ^= c13_reduc;
    if(c16 & lsb13) c16 ^= c13_reduc;
    if(c17 & lsb13) c17 ^= c13_reduc;
    if(c18 & lsb13) c18 ^= c13_reduc;
    if(c19 & lsb13) c19 ^= c13_reduc;
    if(c20 & lsb13) c20 ^= c13_reduc;
    if(c21 & lsb13) c21 ^= c13_reduc;
    if(c22 & lsb13) c22 ^= c13_reduc;
    if(c23 & lsb13) c23 ^= c13_reduc;
    if(c24 & lsb13) c24 ^= c13_reduc;
    if(c25 & lsb13) c25 ^= c13_reduc;
    if(c26 & lsb13) c26 ^= c13_reduc;
    if(c27 & lsb13) c27 ^= c13_reduc;
    if(c28 & lsb13) c28 ^= c13_reduc;
    if(c29 & lsb13) c29 ^= c13_reduc;
    if(c30 & lsb13) c30 ^= c13_reduc;
    if(c31 & lsb13) c31 ^= c13_reduc;
    if(c32 & lsb13) c32 ^= c13_reduc;
    mask ^= lsb13;

    uint64_t lsb14 = uint64_t_lsb(c14 & mask); // x14
    if(unlikely(!lsb14)) {
        return -1; // singular
    }

    uint64_t c14_reduc = c14 ^ lsb14;
    if(const_col & lsb14) const_col ^= c14_reduc;
    if(c15 & lsb14) c15 ^= c14_reduc;
    if(c16 & lsb14) c16 ^= c14_reduc;
    if(c17 & lsb14) c17 ^= c14_reduc;
    if(c18 & lsb14) c18 ^= c14_reduc;
    if(c19 & lsb14) c19 ^= c14_reduc;
    if(c20 & lsb14) c20 ^= c14_reduc;
    if(c21 & lsb14) c21 ^= c14_reduc;
    if(c22 & lsb14) c22 ^= c14_reduc;
    if(c23 & lsb14) c23 ^= c14_reduc;
    if(c24 & lsb14) c24 ^= c14_reduc;
    if(c25 & lsb14) c25 ^= c14_reduc;
    if(c26 & lsb14) c26 ^= c14_reduc;
    if(c27 & lsb14) c27 ^= c14_reduc;
    if(c28 & lsb14) c28 ^= c14_reduc;
    if(c29 & lsb14) c29 ^= c14_reduc;
    if(c30 & lsb14) c30 ^= c14_reduc;
    if(c31 & lsb14) c31 ^= c14_reduc;
    if(c32 & lsb14) c32 ^= c14_reduc;
    mask ^= lsb14;

    uint64_t lsb15 = uint64_t_lsb(c15 & mask); // x15
    if(unlikely(!lsb15)) {
        return -1; // singular
    }

    uint64_t c15_reduc = c15 ^ lsb15;
    if(const_col & lsb15) const_col ^= c15_reduc;
    if(c16 & lsb15) c16 ^= c15_reduc;
    if(c17 & lsb15) c17 ^= c15_reduc;
    if(c18 & lsb15) c18 ^= c15_reduc;
    if(c19 & lsb15) c19 ^= c15_reduc;
    if(c20 & lsb15) c20 ^= c15_reduc;
    if(c21 & lsb15) c21 ^= c15_reduc;
    if(c22 & lsb15) c22 ^= c15_reduc;
    if(c23 & lsb15) c23 ^= c15_reduc;
    if(c24 & lsb15) c24 ^= c15_reduc;
    if(c25 & lsb15) c25 ^= c15_reduc;
    if(c26 & lsb15) c26 ^= c15_reduc;
    if(c27 & lsb15) c27 ^= c15_reduc;
    if(c28 & lsb15) c28 ^= c15_reduc;
    if(c29 & lsb15) c29 ^= c15_reduc;
    if(c30 & lsb15) c30 ^= c15_reduc;
    if(c31 & lsb15) c31 ^= c15_reduc;
    if(c32 & lsb15) c32 ^= c15_reduc;
    mask ^= lsb15;

    uint64_t lsb16 = uint64_t_lsb(c16 & mask); // x16
    if(unlikely(!lsb16)) {
        return -1; // singular
    }

    uint64_t c16_reduc = c16 ^ lsb16;
    if(const_col & lsb16) const_col ^= c16_reduc;
    if(c17 & lsb16) c17 ^= c16_reduc;
    if(c18 & lsb16) c18 ^= c16_reduc;
    if(c19 & lsb16) c19 ^= c16_reduc;
    if(c20 & lsb16) c20 ^= c16_reduc;
    if(c21 & lsb16) c21 ^= c16_reduc;
    if(c22 & lsb16) c22 ^= c16_reduc;
    if(c23 & lsb16) c23 ^= c16_reduc;
    if(c24 & lsb16) c24 ^= c16_reduc;
    if(c25 & lsb16) c25 ^= c16_reduc;
    if(c26 & lsb16) c26 ^= c16_reduc;
    if(c27 & lsb16) c27 ^= c16_reduc;
    if(c28 & lsb16) c28 ^= c16_reduc;
    if(c29 & lsb16) c29 ^= c16_reduc;
    if(c30 & lsb16) c30 ^= c16_reduc;
    if(c31 & lsb16) c31 ^= c16_reduc;
    if(c32 & lsb16) c32 ^= c16_reduc;
    mask ^= lsb16;

    uint64_t lsb17 = uint64_t_lsb(c17 & mask); // x17
    if(unlikely(!lsb17)) {
        return -1; // singular
    }

    uint64_t c17_reduc = c17 ^ lsb17;
    if(const_col & lsb17) const_col ^= c17_reduc;
    if(c18 & lsb17) c18 ^= c17_reduc;
    if(c19 & lsb17) c19 ^= c17_reduc;
    if(c20 & lsb17) c20 ^= c17_reduc;
    if(c21 & lsb17) c21 ^= c17_reduc;
    if(c22 & lsb17) c22 ^= c17_reduc;
    if(c23 & lsb17) c23 ^= c17_reduc;
    if(c24 & lsb17) c24 ^= c17_reduc;
    if(c25 & lsb17) c25 ^= c17_reduc;
    if(c26 & lsb17) c26 ^= c17_reduc;
    if(c27 & lsb17) c27 ^= c17_reduc;
    if(c28 & lsb17) c28 ^= c17_reduc;
    if(c29 & lsb17) c29 ^= c17_reduc;
    if(c30 & lsb17) c30 ^= c17_reduc;
    if(c31 & lsb17) c31 ^= c17_reduc;
    if(c32 & lsb17) c32 ^= c17_reduc;
    mask ^= lsb17;

    uint64_t lsb18 = uint64_t_lsb(c18 & mask); // x18
    if(unlikely(!lsb18)) {
        return -1; // singular
    }

    uint64_t c18_reduc = c18 ^ lsb18;
    if(const_col & lsb18) const_col ^= c18_reduc;
    if(c19 & lsb18) c19 ^= c18_reduc;
    if(c20 & lsb18) c20 ^= c18_reduc;
    if(c21 & lsb18) c21 ^= c18_reduc;
    if(c22 & lsb18) c22 ^= c18_reduc;
    if(c23 & lsb18) c23 ^= c18_reduc;
    if(c24 & lsb18) c24 ^= c18_reduc;
    if(c25 & lsb18) c25 ^= c18_reduc;
    if(c26 & lsb18) c26 ^= c18_reduc;
    if(c27 & lsb18) c27 ^= c18_reduc;
    if(c28 & lsb18) c28 ^= c18_reduc;
    if(c29 & lsb18) c29 ^= c18_reduc;
    if(c30 & lsb18) c30 ^= c18_reduc;
    if(c31 & lsb18) c31 ^= c18_reduc;
    if(c32 & lsb18) c32 ^= c18_reduc;
    mask ^= lsb18;

    uint64_t lsb19 = uint64_t_lsb(c19 & mask); // x19
    if(unlikely(!lsb19)) {
        return -1; // singular
    }

    uint64_t c19_reduc = c19 ^ lsb19;
    if(const_col & lsb19) const_col ^= c19_reduc;
    if(c20 & lsb19) c20 ^= c19_reduc;
    if(c21 & lsb19) c21 ^= c19_reduc;
    if(c22 & lsb19) c22 ^= c19_reduc;
    if(c23 & lsb19) c23 ^= c19_reduc;
    if(c24 & lsb19) c24 ^= c19_reduc;
    if(c25 & lsb19) c25 ^= c19_reduc;
    if(c26 & lsb19) c26 ^= c19_reduc;
    if(c27 & lsb19) c27 ^= c19_reduc;
    if(c28 & lsb19) c28 ^= c19_reduc;
    if(c29 & lsb19) c29 ^= c19_reduc;
    if(c30 & lsb19) c30 ^= c19_reduc;
    if(c31 & lsb19) c31 ^= c19_reduc;
    if(c32 & lsb19) c32 ^= c19_reduc;
    mask ^= lsb19;

    uint64_t lsb20 = uint64_t_lsb(c20 & mask); // x20
    if(unlikely(!lsb20)) {
        return -1; // singular
    }

    uint64_t c20_reduc = c20 ^ lsb20;
    if(const_col & lsb20) const_col ^= c20_reduc;
    if(c21 & lsb20) c21 ^= c20_reduc;
    if(c22 & lsb20) c22 ^= c20_reduc;
    if(c23 & lsb20) c23 ^= c20_reduc;
    if(c24 & lsb20) c24 ^= c20_reduc;
    if(c25 & lsb20) c25 ^= c20_reduc;
    if(c26 & lsb20) c26 ^= c20_reduc;
    if(c27 & lsb20) c27 ^= c20_reduc;
    if(c28 & lsb20) c28 ^= c20_reduc;
    if(c29 & lsb20) c29 ^= c20_reduc;
    if(c30 & lsb20) c30 ^= c20_reduc;
    if(c31 & lsb20) c31 ^= c20_reduc;
    if(c32 & lsb20) c32 ^= c20_reduc;
    mask ^= lsb20;

    uint64_t lsb21 = uint64_t_lsb(c21 & mask); // x21
    if(unlikely(!lsb21)) {
        return -1; // singular
    }

    uint64_t c21_reduc = c21 ^ lsb21;
    if(const_col & lsb21) const_col ^= c21_reduc;
    if(c22 & lsb21) c22 ^= c21_reduc;
    if(c23 & lsb21) c23 ^= c21_reduc;
    if(c24 & lsb21) c24 ^= c21_reduc;
    if(c25 & lsb21) c25 ^= c21_reduc;
    if(c26 & lsb21) c26 ^= c21_reduc;
    if(c27 & lsb21) c27 ^= c21_reduc;
    if(c28 & lsb21) c28 ^= c21_reduc;
    if(c29 & lsb21) c29 ^= c21_reduc;
    if(c30 & lsb21) c30 ^= c21_reduc;
    if(c31 & lsb21) c31 ^= c21_reduc;
    if(c32 & lsb21) c32 ^= c21_reduc;
    mask ^= lsb21;

    uint64_t lsb22 = uint64_t_lsb(c22 & mask); // x22
    if(unlikely(!lsb22)) {
        return -1; // singular
    }

    uint64_t c22_reduc = c22 ^ lsb22;
    if(const_col & lsb22) const_col ^= c22_reduc;
    if(c23 & lsb22) c23 ^= c22_reduc;
    if(c24 & lsb22) c24 ^= c22_reduc;
    if(c25 & lsb22) c25 ^= c22_reduc;
    if(c26 & lsb22) c26 ^= c22_reduc;
    if(c27 & lsb22) c27 ^= c22_reduc;
    if(c28 & lsb22) c28 ^= c22_reduc;
    if(c29 & lsb22) c29 ^= c22_reduc;
    if(c30 & lsb22) c30 ^= c22_reduc;
    if(c31 & lsb22) c31 ^= c22_reduc;
    if(c32 & lsb22) c32 ^= c22_reduc;
    mask ^= lsb22;

    uint64_t lsb23 = uint64_t_lsb(c23 & mask); // x23
    if(unlikely(!lsb23)) {
        return -1; // singular
    }

    uint64_t c23_reduc = c23 ^ lsb23;
    if(const_col & lsb23) const_col ^= c23_reduc;
    if(c24 & lsb23) c24 ^= c23_reduc;
    if(c25 & lsb23) c25 ^= c23_reduc;
    if(c26 & lsb23) c26 ^= c23_reduc;
    if(c27 & lsb23) c27 ^= c23_reduc;
    if(c28 & lsb23) c28 ^= c23_reduc;
    if(c29 & lsb23) c29 ^= c23_reduc;
    if(c30 & lsb23) c30 ^= c23_reduc;
    if(c31 & lsb23) c31 ^= c23_reduc;
    if(c32 & lsb23) c32 ^= c23_reduc;
    mask ^= lsb23;

    uint64_t lsb24 = uint64_t_lsb(c24 & mask); // x24
    if(unlikely(!lsb24)) {
        return -1; // singular
    }

    uint64_t c24_reduc = c24 ^ lsb24;
    if(const_col & lsb24) const_col ^= c24_reduc;
    if(c25 & lsb24) c25 ^= c24_reduc;
    if(c26 & lsb24) c26 ^= c24_reduc;
    if(c27 & lsb24) c27 ^= c24_reduc;
    if(c28 & lsb24) c28 ^= c24_reduc;
    if(c29 & lsb24) c29 ^= c24_reduc;
    if(c30 & lsb24) c30 ^= c24_reduc;
    if(c31 & lsb24) c31 ^= c24_reduc;
    if(c32 & lsb24) c32 ^= c24_reduc;
    mask ^= lsb24;

    uint64_t lsb25 = uint64_t_lsb(c25 & mask); // x25
    if(unlikely(!lsb25)) {
        return -1; // singular
    }

    uint64_t c25_reduc = c25 ^ lsb25;
    if(const_col & lsb25) const_col ^= c25_reduc;
    if(c26 & lsb25) c26 ^= c25_reduc;
    if(c27 & lsb25) c27 ^= c25_reduc;
    if(c28 & lsb25) c28 ^= c25_reduc;
    if(c29 & lsb25) c29 ^= c25_reduc;
    if(c30 & lsb25) c30 ^= c25_reduc;
    if(c31 & lsb25) c31 ^= c25_reduc;
    if(c32 & lsb25) c32 ^= c25_reduc;
    mask ^= lsb25;

    uint64_t lsb26 = uint64_t_lsb(c26 & mask); // x26
    if(unlikely(!lsb26)) {
        return -1; // singular
    }

    uint64_t c26_reduc = c26 ^ lsb26;
    if(const_col & lsb26) const_col ^= c26_reduc;
    if(c27 & lsb26) c27 ^= c26_reduc;
    if(c28 & lsb26) c28 ^= c26_reduc;
    if(c29 & lsb26) c29 ^= c26_reduc;
    if(c30 & lsb26) c30 ^= c26_reduc;
    if(c31 & lsb26) c31 ^= c26_reduc;
    if(c32 & lsb26) c32 ^= c26_reduc;
    mask ^= lsb26;

    uint64_t lsb27 = uint64_t_lsb(c27 & mask); // x27
    if(unlikely(!lsb27)) {
        return -1; // singular
    }

    uint64_t c27_reduc = c27 ^ lsb27;
    if(const_col & lsb27) const_col ^= c27_reduc;
    if(c28 & lsb27) c28 ^= c27_reduc;
    if(c29 & lsb27) c29 ^= c27_reduc;
    if(c30 & lsb27) c30 ^= c27_reduc;
    if(c31 & lsb27) c31 ^= c27_reduc;
    if(c32 & lsb27) c32 ^= c27_reduc;
    mask ^= lsb27;

    uint64_t lsb28 = uint64_t_lsb(c28 & mask); // x28
    if(unlikely(!lsb28)) {
        return -1; // singular
    }

    uint64_t c28_reduc = c28 ^ lsb28;
    if(const_col & lsb28) const_col ^= c28_reduc;
    if(c29 & lsb28) c29 ^= c28_reduc;
    if(c30 & lsb28) c30 ^= c28_reduc;
    if(c31 & lsb28) c31 ^= c28_reduc;
    if(c32 & lsb28) c32 ^= c28_reduc;
    mask ^= lsb28;

    uint64_t lsb29 = uint64_t_lsb(c29 & mask); // x29
    if(unlikely(!lsb29)) {
        return -1; // singular
    }

    uint64_t c29_reduc = c29 ^ lsb29;
    if(const_col & lsb29) const_col ^= c29_reduc;
    if(c30 & lsb29) c30 ^= c29_reduc;
    if(c31 & lsb29) c31 ^= c29_reduc;
    if(c32 & lsb29) c32 ^= c29_reduc;
    mask ^= lsb29;

    uint64_t lsb30 = uint64_t_lsb(c30 & mask); // x30
    if(unlikely(!lsb30)) {
        return -1; // singular
    }

    uint64_t c30_reduc = c30 ^ lsb30;
    if(const_col & lsb30) const_col ^= c30_reduc;
    if(c31 & lsb30) c31 ^= c30_reduc;
    if(c32 & lsb30) c32 ^= c30_reduc;
    mask ^= lsb30;

    uint64_t lsb31 = uint64_t_lsb(c31 & mask); // x31
    if(unlikely(!lsb31)) {
        return -1; // singular
    }

    uint64_t c31_reduc = c31 ^ lsb31;
    if(const_col & lsb31) const_col ^= c31_reduc;
    if(c32 & lsb31) c32 ^= c31_reduc;
    mask ^= lsb31;

    uint64_t lsb32 = uint64_t_lsb(c32 & mask); // x32
    if(unlikely(!lsb32)) {
        return -1; // singular
    }

    if(const_col & lsb32) const_col ^= c32 ^ lsb32;
    mask ^= lsb32;

    if(likely(mask & const_col)) { // check system consistency
        return mask & const_col; // not solvable
    }

    uint64_t s = 0x0ULL;
    if(const_col & lsb1) s = uint64_t_toggle_at(s, 0);
    if(const_col & lsb2) s = uint64_t_toggle_at(s, 1);
    if(const_col & lsb3) s = uint64_t_toggle_at(s, 2);
    if(const_col & lsb4) s = uint64_t_toggle_at(s, 3);
    if(const_col & lsb5) s = uint64_t_toggle_at(s, 4);
    if(const_col & lsb6) s = uint64_t_toggle_at(s, 5);
    if(const_col & lsb7) s = uint64_t_toggle_at(s, 6);
    if(const_col & lsb8) s = uint64_t_toggle_at(s, 7);
    if(const_col & lsb9) s = uint64_t_toggle_at(s, 8);
    if(const_col & lsb10) s = uint64_t_toggle_at(s, 9);
    if(const_col & lsb11) s = uint64_t_toggle_at(s, 10);
    if(const_col & lsb12) s = uint64_t_toggle_at(s, 11);
    if(const_col & lsb13) s = uint64_t_toggle_at(s, 12);
    if(const_col & lsb14) s = uint64_t_toggle_at(s, 13);
    if(const_col & lsb15) s = uint64_t_toggle_at(s, 14);
    if(const_col & lsb16) s = uint64_t_toggle_at(s, 15);
    if(const_col & lsb17) s = uint64_t_toggle_at(s, 16);
    if(const_col & lsb18) s = uint64_t_toggle_at(s, 17);
    if(const_col & lsb19) s = uint64_t_toggle_at(s, 18);
    if(const_col & lsb20) s = uint64_t_toggle_at(s, 19);
    if(const_col & lsb21) s = uint64_t_toggle_at(s, 20);
    if(const_col & lsb22) s = uint64_t_toggle_at(s, 21);
    if(const_col & lsb23) s = uint64_t_toggle_at(s, 22);
    if(const_col & lsb24) s = uint64_t_toggle_at(s, 23);
    if(const_col & lsb25) s = uint64_t_toggle_at(s, 24);
    if(const_col & lsb26) s = uint64_t_toggle_at(s, 25);
    if(const_col & lsb27) s = uint64_t_toggle_at(s, 26);
    if(const_col & lsb28) s = uint64_t_toggle_at(s, 27);
    if(const_col & lsb29) s = uint64_t_toggle_at(s, 28);
    if(const_col & lsb30) s = uint64_t_toggle_at(s, 29);
    if(const_col & lsb31) s = uint64_t_toggle_at(s, 30);
    if(const_col & lsb32) s = uint64_t_toggle_at(s, 31);
    *sol = s;
    return 0;
}
