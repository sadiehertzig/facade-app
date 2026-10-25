// Mask clean-up, shared by the upload page and its tests.
//
// MASK POLARITY (the convention the worker expects):
//   white (255) = the region an editor might replace, e.g. clothing. PhotoGuard
//                 leaves these pixels exactly as they are.
//   black (0)   = the surrounding context. This is where PhotoGuard adds its
//                 small changes, because an inpainting editor reads the
//                 context to decide what to paint into the white region.
// Getting this backwards would protect the wrong part of the photo. The worker
// also checks, on its side, that the mask it receives is pure black and white.
//
// The brush on the page draws soft (antialiased) white strokes, so edge pixels
// come out grey. The worker accepts only pure black and pure white, so every
// pixel is rounded to one or the other at the halfway point (128).
/**
 * Converts the antialiased white-on-black brush raster to the encoder's binary
 * edit mask, in place.
 *
 * `pixels` is canvas RGBA data: four bytes (red, green, blue, alpha) per pixel.
 * Only the red channel is read (the brush is grey, so all three colour
 * channels are equal); all three are then set to 0 or 255 and alpha to fully
 * opaque.
 *
 * Returns true only when the mask has at least one white and one black pixel:
 * an all-black mask marks nothing an editor might replace, and an all-white
 * mask leaves no context for PhotoGuard to change. Either way there would be nothing useful to do.
 */
export function normalizeEditMask(pixels: Uint8ClampedArray): boolean {
  let edit = false, context = false;
  for (let i = 0; i < pixels.length; i += 4) {
    const value = pixels[i] >= 128 ? 255 : 0;
    pixels[i] = pixels[i + 1] = pixels[i + 2] = value;
    pixels[i + 3] = 255;
    // Remember whether we have seen at least one pixel of each kind.
    edit ||= value === 255; context ||= value === 0;
  }
  return edit && context;
}
