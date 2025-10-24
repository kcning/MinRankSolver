#ifndef __GF_H__
#define __GF_H__

#include <gf16.h>
#include <gf31.h>

// TODO: define a compiler switch for the GF size
#ifndef GF_SIZE
#define GF_SIZE 16
#endif

#if GF_SIZE == 16
typedef gf16_t gf_t;
#define GF_MIN GF16_MIN
#define GF_MAX GF16_MAX

#define gf_t_rand                       gf16_t_rand
#define gf_t_arr_rand                   gf16_t_arr_rand
#define gf_t_reduc                      gf16_t_reduc
#define gf_t_add                        gf16_t_add
#define gf_t_mul                        gf16_t_mul
#define gf_t_inv                        gf16_t_inv
#define gf_t_arr_mul_scalar             gf16_t_arr_mul_scalar
#define gf_t_arr_muli_scalar            gf16_t_arr_muli_scalar
#define gf_t_arr_muli_scalar64          gf16_t_arr_muli_scalar64
#define gf_t_arr_fmaddi_scalar          gf16_t_arr_fmaddi_scalar
#define gf_t_arr_fmaddi_scalar64        gf16_t_arr_fmaddi_scalar64
#define gf_t_arr_fmaddi_scalar_mask64   gf16_t_arr_fmaddi_scalar_mask64
#define gf_t_arr_fmsubi_scalar          gf16_t_arr_fmsubi_scalar
#define gf_t_arr_fmsubi_scalar64        gf16_t_arr_fmsubi_scalar64
#define gf_t_arr_fmsubi_scalar_mask64   gf16_t_arr_fmsubi_scalar_mask64
#define gf_t_arr_nzc                    gf16_t_arr_nzc
#define gf_t_arr_zc                     gf16_t_arr_zc
#define gf_t_arr_zero_64b               gf16_t_arr_zero_64b
#define gf_t_arr_mask_from_64b          gf16_t_arr_mask_from_64b
#define gf_t_arr_mask_from_64b_reg_avx2 gf16_t_arr_mask_from_64b_reg_avx2
#define gf_t_arr_andi_64                gf16_t_arr_andi_64
#define gf_t_arr_andi_64_reg_avx2       gf16_t_arr_andi_64_reg_avx2
#define gf_t_arr_reduc_64               gf16_t_arr_reduc_64
#define gf_t_arr_fmaddi_scalar64_x2     gf16_t_arr_fmaddi_scalar64_x2
#define gf_t_arr_addi                   gf16_t_arr_addi
#define gf_t_arr_addi_64                gf16_t_arr_addi_64

#elif GF_SIZE == 31

typedef gf31_t gf_t;
#define GF_MIN GF31_MIN
#define GF_MAX GF31_MAX

#define gf_t_rand                       gf31_t_rand
#define gf_t_arr_rand                   gf31_t_arr_rand
#define gf_t_reduc                      gf31_t_reduc
#define gf_t_add                        gf31_t_add
#define gf_t_mul                        gf31_t_mul
#define gf_t_inv                        gf31_t_inv
#define gf_t_arr_mul_scalar             gf31_t_arr_mul_scalar
#define gf_t_arr_muli_scalar            gf31_t_arr_muli_scalar
#define gf_t_arr_muli_scalar64          gf31_t_arr_muli_scalar64
#define gf_t_arr_fmaddi_scalar          gf31_t_arr_fmaddi_scalar
#define gf_t_arr_fmaddi_scalar64        gf31_t_arr_fmaddi_scalar64
#define gf_t_arr_fmaddi_scalar_mask64   gf31_t_arr_fmaddi_scalar_mask64
#define gf_t_arr_fmsubi_scalar          gf31_t_arr_fmsubi_scalar
#define gf_t_arr_fmsubi_scalar64        gf31_t_arr_fmsubi_scalar64
#define gf_t_arr_fmsubi_scalar_mask64   gf31_t_arr_fmsubi_scalar_mask64
#define gf_t_arr_nzc                    gf31_t_arr_nzc
#define gf_t_arr_zc                     gf31_t_arr_zc
#define gf_t_arr_zero_64b               gf31_t_arr_zero_64b
#define gf_t_arr_mask_from_64b          gf31_t_arr_mask_from_64b
#define gf_t_arr_mask_from_64b_reg_avx2 gf31_t_arr_mask_from_64b_reg_avx2
#define gf_t_arr_andi_64                gf31_t_arr_andi_64
#define gf_t_arr_andi_64_reg_avx2       gf31_t_arr_andi_64_reg_avx2
#define gf_t_arr_reduc_64               gf31_t_arr_reduc_64
#define gf_t_arr_addi                   gf31_t_arr_addi
#define gf_t_arr_addi_64                gf31_t_arr_addi_64

#endif

#endif // __GF_H__
