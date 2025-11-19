#include "loader.h"
#include "gf.h"
#include "util.h"

#include <stdio.h>
#include <string.h>


/* ========================================================================
 * struct Loader definition
 * ======================================================================== */

//struct Loader {
//};

/* ========================================================================
 * function implementations
 * ======================================================================== */

static inline int32_t
loader_gfm_store_row(gf_t* coeffs, uint32_t ncol, char buf[]) {
    char* token = strtok(buf, " ");
    uint32_t i = 0;
    while(token) {
        gf_t v = atoi(token);

        if(v > GF_MAX)
            return 1;

        coeffs[i++] = v;
        token = strtok(NULL, " ");
    }

    if(i != ncol)
        return 1;

    return 0;
}

static inline enum LoaderGFMfromFileCode
loader_gfm_load_1_matrix(gf_t* restrict coeffs, uint32_t idx,
                         uint32_t nrow, uint32_t ncol,
                         FILE* restrict f, char buf[], uint32_t buf_size) {
    if(!fgets(buf, buf_size, f)) // skip the id line
        return FILE_EOF;

    char id[128];
    // TODO: check for exceptions
    snprintf(id, 128, "M%d", idx);
    if(strncmp(buf, id, 2))
        return FORMAT_ERR;

    if(!fgets(buf, buf_size, f)) // load the first row
        return FORMAT_ERR;

    if(ncol != count_int_in_str(buf) ) // file changed
        return FORMAT_ERR;

    uint32_t r = 1;
    if(loader_gfm_store_row(coeffs, ncol, buf))
        return FORMAT_ERR;

    coeffs += ncol;

    while(fgets(buf, buf_size, f) && strnlen(buf, buf_size-1) > 1) { // \n counts
        if(ncol != count_int_in_str(buf))
            return FORMAT_ERR;

        if(loader_gfm_store_row(coeffs, ncol, buf))
            return FORMAT_ERR;

        coeffs += ncol;
        ++r;
    }

    if(r != nrow)
        return FORMAT_ERR;

    return SUCCESS;
}

enum LoaderGFMfromFileCode
loader_gfm_from_file(LoaderGFMfromFileRet* restrict rt, const char fname[]) {
    FILE* file = fopen(fname, "r");
    gf_t* coeffs = NULL;

    if(!file)
        return FOPEN_FAIL;

    char buf[4096];
    enum LoaderGFMfromFileCode code = SUCCESS;
    if(4 != fscanf(file, "n = %u\nm = %u\nk = %u\nr = %u\n",
                   &rt->nrow, &rt->ncol, &rt->k, &rt->r)) {
        code = FORMAT_ERR;
        goto loader_gfm_from_file_cleanup;
    }

    const uint32_t ele_per_mat = rt->nrow * rt->ncol;
    coeffs = malloc(sizeof(gf_t) * ele_per_mat * (rt->k+1));
    if(!coeffs) {
        code = MEM_ERR;
        goto loader_gfm_from_file_cleanup;
    }

    uint32_t mat_num = 0;
    for(; mat_num <= rt->k; ++mat_num) {
        code = loader_gfm_load_1_matrix(coeffs + ele_per_mat * mat_num,
                                        mat_num, rt->nrow, rt->ncol,
                                        file, buf, 4096);
        if(code != SUCCESS)
            goto loader_gfm_from_file_cleanup;
    }

    if( (rt->k+1) != mat_num) {
        code = FORMAT_ERR;
        goto loader_gfm_from_file_cleanup;
    }

    rt->m0 = gfm_create(rt->nrow, rt->ncol, coeffs);
    rt->ms = gfm_arr_create(rt->nrow, rt->ncol, rt->k, coeffs + ele_per_mat);

loader_gfm_from_file_cleanup:
    free(coeffs);
    fclose(file);
    return SUCCESS;
}
