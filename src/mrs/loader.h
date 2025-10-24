#ifndef __LOADER_H__
#define __LOADER_H__

#include <stdint.h>

#include "gfm.h"

//typedef struct Loader Loader;

/* ========================================================================
 * function prototypes
 * ======================================================================== */

struct LoaderGFMfromFileRet {
    uint32_t nrow;
    uint32_t ncol;
    uint32_t k;
    uint32_t r;
    GFM* restrict m0;
    GFM* restrict ms;
};

typedef struct LoaderGFMfromFileRet LoaderGFMfromFileRet;

enum LoaderGFMfromFileCode {
    SUCCESS = 0,
    FOPEN_FAIL = 1,
    FORMAT_ERR = 2,
    FILE_EOF = 3,
    MEM_ERR = 4,
};

enum LoaderGFMfromFileCode
loader_gfm_from_file(LoaderGFMfromFileRet* restrict rt, const char fname[]);

#endif // __LOADER_H__
