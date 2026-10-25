// NativeNetwork.h
//
// C interface to the on-device face-recognition network in NativeNetwork.cpp.
// Swift sees these functions through the CNative module
// (include/NativeBridge.h); App/NativeTensor.swift wraps the first four.
//
// Conventions shared by every function:
//  * A loaded network is an opaque `void *` handle: create it with
//    facade_network_load and release it with facade_network_free.
//  * For the network functions, images are 3 x 112 x 112 floats in CHW order
//    (all red, then green, then blue), values 0..1. Embeddings and reference
//    vectors are 512 floats.
//  * On failure a function writes a short message into `error` (at most
//    `capacity` bytes, always NUL-terminated) and returns -1 or NULL.
//    C++ exceptions never cross into Swift.
//
// Only the first four functions are used while protecting a photo. The rest
// are self-checks that run one kernel with the fast path on or off (and time
// it) so an outside test can confirm both give identical bits; the app never
// calls them.
#ifndef FACADE_NATIVE_NETWORK_H
#define FACADE_NATIVE_NETWORK_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
// Loads graph.json (network structure) and weights.f32 (all weights as one
// flat array of 32-bit floats). Returns NULL with a message on any problem.
void *facade_network_load(const char *graph,const char *weights,char *error,size_t capacity);
// Frees a handle from facade_network_load. Passing NULL is safe.
void facade_network_free(void *model);
// Forward pass only: writes the 512-float embedding and the score
// (embedding . reference). No gradient.
int facade_network_forward(void *model,const float *rgb,const float *reference,float *embedding,float *score,char *error,size_t capacity);
// Forward and backward: embedding, score, and `gradient`, the derivative of
// the score with respect to each of the 3*112*112 input values.
int facade_network_predict(void *model,const float *rgb,const float *reference,float *embedding,float *score,float *gradient,char *error,size_t capacity);
// Self-check: im2col (unrolling a convolution input into a matrix) once, with
// the fast 3x3 kernel allowed (optimized = 1) or forced off (0). geometry is
// 11 integers: channels, height, width, kernel h/w, stride h/w, padding h/w,
// output h/w. *specialized = 1 if the fast kernel was used.
int facade_columns_fixture(const float *input,size_t inputCount,const int *geometry,size_t geometryCount,int optimized,float *output,size_t outputCount,int *specialized,char *error,size_t capacity);
// Same as facade_columns_fixture, run `repeats` times; *seconds = total time.
int facade_columns_benchmark(const float *input,size_t inputCount,const int *geometry,size_t geometryCount,int optimized,int repeats,float *output,size_t outputCount,int *specialized,double *seconds,char *error,size_t capacity);
// Self-check reports. Each reads a JSON manifest of test cases from the given
// path and returns a JSON report (valid until the next call on the same
// thread). mode 0 = compare the fast kernel with the plain loop; mode 1 and 2
// = time the plain loop and the fast kernel. Equal-shape add:
const char* facade_equal_add_report(const char* manifest,int mode,char* error,size_t capacity);
// Graphs mixing add and mul nodes (only the comparison, no timing):
const char* facade_add_mul_seam_report(const char* path,char* error,size_t capacity);
// PReLU:
const char* facade_prelu_report(const char* manifest,int mode,char* error,size_t capacity);
// Batch norm:
const char* facade_bn_report(const char* path,int mode,char* error,size_t capacity);
// Self-check: col2im, the convolution backward step that adds the column
// matrix `col` back onto the input gradient. `initial` optionally pre-fills
// that gradient (pass initialCount = 0 for none); constant = 1 means the
// destination is a constant and must stay untouched.
int facade_scatter_fixture(const float *col,size_t columnCount,const float *initial,size_t initialCount,const int *geometry,size_t geometryCount,int constant,int optimized,float *output,size_t outputCapacity,size_t *actualCount,int *specialized,char *error,size_t capacity);
// Same as facade_scatter_fixture, timed over `repeats` runs (a multiple of 4).
int facade_scatter_benchmark(const float *col,size_t columnCount,const float *initial,size_t initialCount,const int *geometry,size_t geometryCount,int optimized,int repeats,float *output,size_t outputCapacity,int *specialized,double *seconds,char *error,size_t capacity);
// Self-check: one "mul" node, forward then backward. geometry is 16 integers:
// rank of a, b and out, an alias flag (1 = both inputs are the same node), then
// a, b and out shapes, each padded to 4 integers with zeros.
int facade_mul_fixture(const float *a,size_t ac,const float *b,size_t bc,const float *g,size_t n,const float *ag,size_t agc,const float *bg,size_t bgc,const int *geometry,size_t geometryCount,int constantA,int constantB,int optimized,float *out,size_t capacity,size_t *counts,int *flags,char *error,size_t errorCapacity);
// Same mul node, timed: direction 0 = forward, 1 = backward; repeats must be 128.
int facade_mul_benchmark(const float *a,size_t ac,const float *b,size_t bc,const float *g,size_t n,const float *ag,size_t agc,const float *bg,size_t bgc,const int *geometry,size_t geometryCount,int optimized,int direction,int repeats,float *out,size_t capacity,size_t *counts,int *dispatched,int *gradientCleared,double *seconds,char *error,size_t errorCapacity);
#ifdef __cplusplus
}
#endif
#endif
