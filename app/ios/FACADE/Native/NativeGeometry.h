// NativeGeometry.h
//
// C declarations for the image-geometry helpers in NativeGeometry.cpp: resizing,
// face alignment (fitting and applying a similarity transform), and the
// low-frequency DCT filter used by the optimizer. Swift calls them through the
// CNative module (include/NativeBridge.h); the Swift wrappers are in
// App/NativeTensor.swift, App/NativeVision.swift and App/NativePipeline.swift.
//
// Two pixel layouts appear here, so watch the type:
//  * uint8_t buffers are packed RGB, 3 bytes per pixel, row by row ("HWC").
//  * float buffers are CHW (all red values, then all green, then all blue),
//    except facade_cv_lift, which uses interleaved 3-float pixels (HWC).
// A 2x3 affine matrix is passed as 6 floats in row-major order:
//     [m0 m1 m2]     x' = m0*x + m1*y + m2
//     [m3 m4 m5]     y' = m3*x + m4*y + m5
// Functions returning int give 0 on success and a negative number on failure.
#ifndef FACADE_NATIVE_GEOMETRY_H
#define FACADE_NATIVE_GEOMETRY_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// Bilinear resize of a packed RGB8 image (w x h) into out (ow x oh), using
// OpenCV. Used to shrink the photo for the face detector.
int facade_cv_resize_rgb(const uint8_t*,int,int,uint8_t*,int,int);
// Cuts a size x size face crop out of a packed RGB8 image. `matrix` maps photo
// coordinates to crop coordinates (the output of facade_similarity).
int facade_cv_warp_rgb(const uint8_t*,int,int,const float*,uint8_t*,int);
// Resizes an interleaved 3-channel float image with Lanczos interpolation.
int facade_cv_lift(const float*,int,int,float*,int,int);
// Fits the rotation + uniform scale + translation that best maps five source
// points onto five destination points (each list is x0,y0,...,x4,y4).
int facade_similarity(const float *source10,const float *destination10,float *matrix6);
// Inverts a 2x3 affine matrix (returns -1 if it is singular).
int facade_inverse(const float *matrix6,float *inverse6);
// Float, differentiable version of the face crop. Takes the INVERSE matrix
// (crop coordinates to photo coordinates). Input w x h CHW, output size x size.
void facade_float_crop(const float*,int,int,const float*,float*,int);
// Gradient of facade_float_crop: maps a crop gradient back onto the photo.
void facade_crop_backward(const float*,int,const float*,float*,int,int);
// Separable bilinear resize of a 3-channel CHW float image; the last argument
// switches on antialiasing (used when shrinking).
void facade_resize_float(const float*,int,int,float*,int,int,int);
// Gradient of facade_resize_float, from the resized size back to the original.
void facade_resize_backward(const float*,int,int,float*,int,int,int);
// Keeps only the low-frequency part of every 8x8 block (block DCT filter).
void facade_dct_low(const float*,float*,int,int,int,int);
#ifdef __cplusplus
}
#endif
#endif
