// NativePixels.h
//
// C interface to NativePixels.c: two 8-bit image operations that simulate
// what happens to a photo after it is shared. App/NativeTensor.swift
// (RGBImage.jpeg and RGBImage.resize) calls them to build the "uploaded" copies
// of the photo that the protection is tested against: JPEG quality 70, and
// JPEG quality 75 followed by shrinking to half size and enlarging back (with
// Lanczos). The resize is also used to make the 512x512 input of the subject
// parser in App/NativeVision.swift.
#ifndef FACADE_NATIVE_PIXELS_H
#define FACADE_NATIVE_PIXELS_H
#include <stdint.h>
#include <stddef.h>
// All pixels are packed RGB8: 3 bytes (R, G, B) per pixel, row by row. The
// caller allocates the outputs, and an output must not be the same buffer as
// the input.
// Both functions return zero on success, negative on failure. Width and height
// must each be 1 to 8192.
// Compresses the image as a JPEG at `quality` (1 to 100) in memory, then
// decodes it again into `decoded` (same size). On a libjpeg error the message
// is copied into `error` (at most error_size bytes).
int facade_jpeg_roundtrip(const uint8_t *rgb, int width, int height, int quality,
                          uint8_t *decoded, char *error, size_t error_size);
// Resizes the image to out_width x out_height. filter: 1 = Pillow's bilinear,
// 3 = Pillow's Lanczos. Like Pillow, the result is rounded to 8 bits after the
// horizontal pass and again after the vertical pass.
int facade_resize_rgb(const uint8_t *rgb, int width, int height,
                      uint8_t *output, int out_width, int out_height, int filter);
#endif
