#ifndef FACADE_NATIVE_PIXELS_H
#define FACADE_NATIVE_PIXELS_H
#include <stdint.h>
#include <stddef.h>
// All pixels are packed RGB8. Caller owns outputs; inputs/outputs must not alias.
// Returns zero on success. Dimensions must be in 1...8192.
int facade_jpeg_roundtrip(const uint8_t *rgb, int width, int height, int quality,
                          uint8_t *decoded, char *error, size_t error_size);
// filter: 1 = Pillow bilinear; 3 = Pillow Lanczos, with per-pass RGB8 rounding.
int facade_resize_rgb(const uint8_t *rgb, int width, int height,
                      uint8_t *output, int out_width, int out_height, int filter);
#endif
