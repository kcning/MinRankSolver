#include "gfm.h"
#include "bytearray.h"

#include <stdlib.h>

/* ========================================================================
 * struct GFM definition
 * ======================================================================== */

struct GFM {
    uint64_t nrow;          // number of rows
    uint64_t ncol;          // number of columns
    ByteArray* rows;        // row-majored
};

/* ========================================================================
 * function implementations
 * ======================================================================== */

/* usage: Given the size of the matrix and optional its elements, create a GFM
 * params:
 *      1) nrow: number of rows
 *      2) ncol: number of columns
 *      3) vals: optional. Elements of the matrix stored in row-majored format
 * return: ptr to GFM, on error return NULL */
GFM*
gfm_create(uint64_t nrow, uint64_t ncol, const gf_t* vals) {
    if(nrow == 0 || ncol == 0)
        return NULL;

    GFM* m = malloc(sizeof(GFM));
    if(!m)
        return NULL;

    // TODO: check if nrow * ncol overflows
    const uint64_t needed_bytes = nrow * ncol;
    ByteArray* b = bytearray_create(needed_bytes);
    if(!b) {
        free(m);
        return NULL;
    }

    m->rows = b;

    if(vals) { // initialize the matrix with the values
        bytearray_zero(b);
        // both memory are row-majored, and gf_t is one byte, so we can
        // directly copy the whole block
        static_assert(sizeof(gf_t) == sizeof(int8_t),
                      "size of gf_t must be 1 byte");
        memcpy(bytearray_memblk(m->rows), vals, needed_bytes);
    }

    m->nrow = nrow;
    m->ncol = ncol;

    return m;
}

/* usage: Release a struct GFM
 * params:
 *      1) m: ptr to struct GFM
 * return: void */
void
gfm_free(GFM* m) {
    bytearray_free(m->rows, true);
    free(m);
}

/* usage: Release an array of struct GFM
 * params:
 *      1) ms: ptr to an array of struct GFM
 *      2) n: size of the array
 * return: void */
void
gfm_arr_free(GFM* ms, uint64_t n) {
    for(uint64_t i = 0; i < n; ++i) {
        bytearray_free(ms[i].rows, true);
    }
    free(ms);
}

/* usage: Create an array of struct GFM
 * params:
 *      1) nrow: number of rows
 *      2) ncol: number of columns
 *      3) n: size of the array
 *      4) vals: optional. Elements of the matrices stored in row-majored format
 * return: ptr to GFM, on error return NULL */
GFM*
gfm_arr_create(uint32_t nrow, uint32_t ncol, uint32_t n, const gf_t* vals) {
    if(!n)
        return NULL;

    GFM* ms = malloc(sizeof(GFM) * n);
    if(!ms)
        return NULL;

    const uint64_t elements_per_matrix = nrow * ncol;
    for(uint32_t i = 0; i < n; ++i) {
        ms[i].nrow = nrow;
        ms[i].ncol = ncol;
        ByteArray* b = bytearray_create(elements_per_matrix);

        if(!b) { // clean up
            gfm_arr_free(ms, i);
            return NULL;
        }
        ms[i].rows = b;
    }

    if(vals) {
        for(uint32_t i = 0; i < n; ++i)
            gfm_set_from_arr(ms + i, vals + elements_per_matrix * i);
    }
    return ms;
}

/* usage: Given an array of struct GFM, return its i-th entry
 * params:
 *      1) ms: ptr to an array of struct GFM
 *      2) i: index of target entry
 * return: ptr to target entry */
GFM*
gfm_arr_at(GFM* ms, uint64_t i) {
    return ms + i;
}

/* usage: Given a struct GFM, set all its coefficients to zero
 * params:
 *      1) m: ptr to struct GFM
 * return: void */
void
gfm_zero(GFM* m) {
    bytearray_zero(m->rows);
}

/* usage: Given a struct GFM, randomize its coefficients
 * params:
 *      1) m: ptr to struct GFM
 * return: void */
void
gfm_rand(GFM* m) {
    static_assert(sizeof(int) / sizeof(uint8_t) == 4, "int is not 4 bytes");
    // TODO: optimize by copying memory directly into ByteArray if necessary
    uint8_t buf[4];
    for(uint64_t i = 0; i < gfm_nrow(m); ++i) {
        const uint64_t tail = gfm_ncol(m) & ~0x3ULL;

        for(uint64_t j = 0; j < tail; j += 4) {
            // TODO: check if this is correct
            *((int*) buf) = rand();
            gfm_set_at(m, i, j + 0, gf_t_reduc(buf[0]));
            gfm_set_at(m, i, j + 1, gf_t_reduc(buf[1]));
            gfm_set_at(m, i, j + 2, gf_t_reduc(buf[2]));
            gfm_set_at(m, i, j + 3, gf_t_reduc(buf[3]));
        }

        *((int*) buf) = rand();
        for(uint64_t j = tail; j < gfm_ncol(m); ++j) {
            gfm_set_at(m, i, j, gf_t_reduc(buf[j-tail]));
        }
    }
}

/* usage: Generate an array of randomize GFMs
 * params:
 *      1) nrow: number of rows in each matrix
 *      2) ncol: number of columns in each matrix
 *      3) num: number of matrices
 * return: ptr to an array of GFMs */
GFM*
gfm_rand_matrices(uint64_t nrow, uint64_t ncol, uint64_t num) {
    GFM* ms = gfm_arr_create(nrow, ncol, num, NULL);
    if(!ms)
        return NULL;

    for(uint64_t i = 0; i < num; ++i)
        gfm_rand(ms + i);

    return ms;
}

/* usage: Given a struct GFM, return its number of rows
 * params:
 *      1) m: ptr to struct GFM
 * return: number of rows */
uint64_t
gfm_nrow(const GFM* m) {
    return m->nrow;
}

/* usage: Given a struct GFM, return its number of columns
 * params:
 *      1) m: ptr to struct GFM
 * return: number of columns */
uint64_t
gfm_ncol(const GFM* m) {
    return m->ncol;
}

/* usage: Given a struct GFM, return addr to its i-th row as an array of gf_t
 * params:
 *      1) m: ptr to struct GFM
 *      2) ri: index of the row
 * return: ptr to the target row */
const gf_t*
gfm_row_addr(const GFM* m, uint64_t ri) {
    uint64_t offset = ri * gfm_ncol(m);
    return bytearray_addr_at(m->rows, offset);
}

/* usage: Given a struct GFM and an array of gf_t, copy the array into the i-th
 *      row of the GFM
 * params:
 *      1) m: ptr to struct GFM
 *      2) ri: index of the row
 *      3) row: an array of gf_t whose length >= gfm_ncol(m)
 * return: void */
void
gfm_row_copy_from(GFM* restrict m, uint64_t ri, const gf_t* restrict row) {
    gf_t* dst = (gf_t*) gfm_row_addr(m, ri);
    memcpy(dst, row, sizeof(gf_t) * gfm_ncol(m));
}

/* usage: Given a struct GFM and n arrays of gf_t, copy the arrays into the
 *      i-th ~ (i+n)-th rows of the GFM
 * params:
 *      1) m: ptr to struct GFM
 *      2) ri: index of the row
 *      3) n: number of rows to copy
 *      4) rows: an array of size n * gfm_ncol(m) which stores elements of the n arrays
 *              consecutively
 * return: void */
void
gfm_rows_copy_from(GFM* restrict m, uint64_t ri, uint64_t n, const gf_t* restrict rows) {
    gf_t* dst = (gf_t*) gfm_row_addr(m, ri);
    memcpy(dst, rows, sizeof(gf_t) * gfm_ncol(m) * n);
}

/* usage: Given a struct GFM and both row and column indices, return the coefficient
 *      of the given entry
 * params:
 *      1) m: ptr to struct GFM
 *      2) ri: index of the row
 *      3) ci: index of the column
 * return: coefficient of the target entry */
gf_t
gfm_at(const GFM* m, uint64_t ri, uint64_t ci) {
    uint64_t idx = ri * gfm_ncol(m) + ci;
    return bytearray_at(m->rows, idx);
}

/* usage: Given a struct GFM and both row and column indices, set the coefficient
 *      of the given entry
 * params:
 *      1) m: ptr to struct GFM
 *      2) ri: index of the row
 *      3) ci: index of the column
 *      4) c: coefficient
 * return: void */
void
gfm_set_at(GFM* m, uint64_t ri, uint64_t ci, gf_t c) {
    uint64_t idx = ri * gfm_ncol(m) + ci;
    bytearray_set_at(m->rows, idx, c);
}

/* usage: Given a struct GFM and an array of gf_t, retreat the array as coefficients
 *      stored in row-major format, and copy them into the struct GFM
 * params:
 *      1) m: ptr to struct GFM
 *      2) cs: ptr to gf_t, must be have >= gfm_nrow(m) * gfm_ncol(m) elements
 * return: void */
void
gfm_set_from_arr(GFM* restrict m, const gf_t* restrict cs) {
    uint64_t num_coeffs = gfm_nrow(m) * gfm_ncol(m);
    memcpy(bytearray_memblk(m->rows), cs, sizeof(gf_t) * num_coeffs);
}

/* usage: Print a struct GFM
 * params:
 *      1) m: ptr to struct GFM
 * return: void */
void
gfm_print(const GFM* m) {
    for(uint64_t i = 0; i < gfm_nrow(m); ++i) {
        for(uint64_t j = 0; j < gfm_ncol(m); ++j) {
            printf("%02d ", gfm_at(m, i, j));
        }
        printf("\n");
    }
}

/* usage: Given a struct GFM, count the number of zero coefficients in it
 * params:
 *      1) m: ptr to struct GFM
 * return: number of zero coefficients */
uint32_t
gfm_cz(const GFM* m) {
    return bytearray_cz(m->rows);
}

/* usage: Given a struct GFM, count the number of nonzero coefficients in it
 * params:
 *      1) m: ptr to struct GFM
 * return: number of zero coefficients */
uint32_t
gfm_cnz(const GFM* m) {
    return gfm_nrow(m) * gfm_ncol(m) - bytearray_cz(m->rows);
}

/* usage: find the max number of non-zero monomials in any row
 * params:
 *      1) m: ptr to struct GFM
 * return: the max number of non-zero entries in any row */
uint64_t
gfm_find_max_tnum_per_eq(const GFM* m) {
    uint64_t max = 0;
    for(uint64_t i = 0; i < gfm_nrow(m); ++i) {
        const gf_t* eq = gfm_row_addr(m, i);
        uint64_t count = 0;
        for(uint64_t j = 0; j < gfm_ncol(m); ++j) {
            if(eq[j] != 0)
                ++count;
        }
        if(count > max)
            max = count;
    }

    return max;
}
