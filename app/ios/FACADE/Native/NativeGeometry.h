#ifndef FACADE_NATIVE_GEOMETRY_H
#define FACADE_NATIVE_GEOMETRY_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
int facade_cv_resize_rgb(const uint8_t*,int,int,uint8_t*,int,int);
int facade_cv_warp_rgb(const uint8_t*,int,int,const float*,uint8_t*,int);
int facade_cv_lift(const float*,int,int,float*,int,int);
int facade_similarity(const float *source10,const float *destination10,float *matrix6);
int facade_inverse(const float *matrix6,float *inverse6);
void facade_float_crop(const float*,int,int,const float*,float*,int);
void facade_crop_backward(const float*,int,const float*,float*,int,int);
void facade_resize_float(const float*,int,int,float*,int,int,int);
void facade_resize_backward(const float*,int,int,float*,int,int,int);
void facade_dct_low(const float*,float*,int,int,int,int);
#ifdef __cplusplus
}
#endif
#endif
