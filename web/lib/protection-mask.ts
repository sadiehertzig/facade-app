// Convert the antialiased white-on-black brush raster to the encoder's binary edit mask.
// White = region an editor may replace. Black = context PhotoGuard can perturb.
export function normalizeEditMask(pixels: Uint8ClampedArray): boolean {
  let edit = false, context = false;
  for (let i = 0; i < pixels.length; i += 4) {
    const value = pixels[i] >= 128 ? 255 : 0;
    pixels[i] = pixels[i + 1] = pixels[i + 2] = value;
    pixels[i + 3] = 255;
    edit ||= value === 255; context ||= value === 0;
  }
  return edit && context;
}
