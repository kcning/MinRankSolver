#ifndef __GF16_H__
#define __GF16_H__

#include <stdint.h>
#include <stdlib.h>
#include <assert.h>

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

#define GF16_MIN (0)
#define GF16_MAX (15)

typedef uint8_t gf16_t;

/* ========================================================================
 * function prototypes
 * ======================================================================== */

gf16_t
gf16_t_reduc_7b(uint8_t v);

gf16_t
gf16_t_reduc_32b(uint32_t v);

static inline gf16_t
gf16_t_reduc(uint32_t v) {
    return gf16_t_reduc_7b(v & 0x7FU);
}

static inline gf16_t
gf16_t_rand(void) {
    return rand() & 0xFU;
}

static inline void
gf16_t_arr_rand(gf16_t* buf, uint64_t size) {
    // TODO: optimize if needed
    for(uint64_t i = 0; i < size; ++i) {
        buf[i] = gf16_t_rand();
    }
}

gf16_t
gf16_t_add(gf16_t a, gf16_t b);

gf16_t
gf16_t_sub(gf16_t a, gf16_t b);

gf16_t
gf16_t_mul(gf16_t a, gf16_t b);

gf16_t
gf16_t_square(gf16_t a);

gf16_t
gf16_t_inv_by_table(gf16_t a);

gf16_t
gf16_t_inv_by_squaring(gf16_t a);

gf16_t
gf16_t_inv(gf16_t a);

void
gf16_t_arr_addi(gf16_t* restrict a, gf16_t* restrict b, uint32_t sz);

void
gf16_t_arr_addi_64(gf16_t* restrict a, gf16_t* restrict b);

void
gf16_t_arr_andi_64(gf16_t* restrict a, gf16_t* restrict b);

#if defined(__AVX2__)

void
gf16_t_arr_andi_64_reg_avx2(gf16_t* a, __m256i high, __m256i low);

#endif

void
gf16_t_arr_muli_scalar64(gf16_t* arr, gf16_t x);

void
gf16_t_arr_mul_scalar(gf16_t* restrict res, gf16_t* restrict arr,
                      uint32_t sz, gf16_t x);

void
gf16_t_arr_muli_scalar(gf16_t* arr, uint32_t sz, gf16_t x);

void
gf16_t_arr_fmaddi_scalar(gf16_t* restrict a, const gf16_t* restrict b,
                         uint32_t sz, gf16_t c);

void
gf16_t_arr_fmaddi_scalar64(gf16_t* restrict a, const gf16_t* restrict b, gf16_t c);

void
gf16_t_arr_fmaddi_scalar64_x2(gf16_t* restrict a,
                              const gf16_t* restrict b0,
                              const gf16_t* restrict b1, gf16_t c0, gf16_t c1);

void
gf16_t_arr_mask_from_64b(gf16_t* a, uint64_t mask);

#if defined(__AVX2__)

void
gf16_t_arr_mask_from_64b_reg_avx2(__m256i* restrict high, __m256i* restrict low, uint64_t mask);

#endif

void
gf16_t_arr_zero_64b(gf16_t* a, uint64_t mask);

void
gf16_t_arr_fmaddi_scalar_mask64(gf16_t* restrict a, const gf16_t* restrict b,
                                gf16_t c, uint64_t d);

void
gf16_t_arr_fmaddi_scalar_mask64_ref(gf16_t* restrict a, const gf16_t* restrict b,
                                    gf16_t c, uint64_t d);

void
gf16_t_arr_fmsubi_scalar(gf16_t* restrict a, const gf16_t* restrict b,
                         uint32_t sz, gf16_t c);

void
gf16_t_arr_fmsubi_scalar64(gf16_t* restrict a, const gf16_t* restrict b, gf16_t c);

void
gf16_t_arr_fmsubi_scalar_mask64(gf16_t* restrict a, const gf16_t* restrict b,
                                gf16_t c, uint64_t d);

void
gf16_t_arr_fmsubi_scalar_mask64_ref(gf16_t* restrict a, const gf16_t* restrict b,
                                    gf16_t c, uint64_t d);

uint64_t
gf16_t_arr_nzc(const gf16_t* a, uint64_t sz);

uint32_t
gf16_t_arr_zc(const gf16_t* a, uint32_t sz);

void
gf16_t_arr_reduc_64(gf16_t* arr);

#endif // __GF16_H__
