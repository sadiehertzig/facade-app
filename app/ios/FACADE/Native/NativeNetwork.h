#ifndef FACADE_NATIVE_NETWORK_H
#define FACADE_NATIVE_NETWORK_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
void *facade_network_load(const char *graph,const char *weights,char *error,size_t capacity);
void facade_network_free(void *model);
int facade_network_forward(void *model,const float *rgb,const float *reference,float *embedding,float *score,char *error,size_t capacity);
int facade_network_predict(void *model,const float *rgb,const float *reference,float *embedding,float *score,float *gradient,char *error,size_t capacity);
int facade_columns_fixture(const float *input,size_t inputCount,const int *geometry,size_t geometryCount,int optimized,float *output,size_t outputCount,int *specialized,char *error,size_t capacity);
int facade_columns_benchmark(const float *input,size_t inputCount,const int *geometry,size_t geometryCount,int optimized,int repeats,float *output,size_t outputCount,int *specialized,double *seconds,char *error,size_t capacity);
const char* facade_equal_add_report(const char* manifest,int mode,char* error,size_t capacity);
const char* facade_add_mul_seam_report(const char* path,char* error,size_t capacity);
const char* facade_prelu_report(const char* manifest,int mode,char* error,size_t capacity);
const char* facade_bn_report(const char* path,int mode,char* error,size_t capacity);
int facade_scatter_fixture(const float *col,size_t columnCount,const float *initial,size_t initialCount,const int *geometry,size_t geometryCount,int constant,int optimized,float *output,size_t outputCapacity,size_t *actualCount,int *specialized,char *error,size_t capacity);
int facade_scatter_benchmark(const float *col,size_t columnCount,const float *initial,size_t initialCount,const int *geometry,size_t geometryCount,int optimized,int repeats,float *output,size_t outputCapacity,int *specialized,double *seconds,char *error,size_t capacity);
int facade_mul_fixture(const float *a,size_t ac,const float *b,size_t bc,const float *g,size_t n,const float *ag,size_t agc,const float *bg,size_t bgc,const int *geometry,size_t geometryCount,int constantA,int constantB,int optimized,float *out,size_t capacity,size_t *counts,int *flags,char *error,size_t errorCapacity);
int facade_mul_benchmark(const float *a,size_t ac,const float *b,size_t bc,const float *g,size_t n,const float *ag,size_t agc,const float *bg,size_t bgc,const int *geometry,size_t geometryCount,int optimized,int direction,int repeats,float *out,size_t capacity,size_t *counts,int *dispatched,int *gradientCleared,double *seconds,char *error,size_t errorCapacity);
#ifdef __cplusplus
}
#endif
#endif
