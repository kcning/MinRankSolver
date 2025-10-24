#ifndef __R64M_GF16_PARALLEL_H__
#define __R64M_GF16_PARALLEL_H__

#include "r64m_gf16.h"
#include "rc64m_gf16.h"
#include <stdint.h>

typedef struct {
    R64MGF16* restrict a;
    const R64MGF16* restrict b;
    RC64MGF16* restrict c;
    RC64MGF16* restrict buf;
    const uint64_t* restrict d;
    uint64_t sidx;
    uint64_t eidx;
    void* restrict ptr; // a generic ptr
} R64MGF16PArg;

/* ========================================================================
 * function prototypes
 * ======================================================================== */

#endif // __R64M_GF16_PARALLEL_H__
