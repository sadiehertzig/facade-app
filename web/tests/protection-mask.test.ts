import { describe, expect, it } from "vitest";
import { normalizeEditMask } from "@/lib/protection-mask";
import { PHOTO_GUARD_RECIPE } from "@/lib/protection-recipe";

describe("normalizeEditMask", () => {
  it("turns antialiased brush pixels into pure black or white", () => {
    // Four RGBA pixels: black, just below the 128 cut-off, at it, and white.
    const pixels = new Uint8ClampedArray([0, 0, 0, 255, 127, 127, 127, 255, 128, 128, 128, 255, 255, 255, 255, 255]);
    expect(normalizeEditMask(pixels)).toBe(true);
    expect([...pixels]).toEqual([0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255]);
  });

  it("rejects an all-black (nothing marked) or all-white (everything marked) mask", () => {
    for (const value of [0, 255]) {
      expect(normalizeEditMask(new Uint8ClampedArray([value, value, value, 255]))).toBe(false);
    }
  });
});

describe("recipe", () => {
  it("is the encoder recipe the worker serves", () => {
    expect(PHOTO_GUARD_RECIPE).toBe("photoguard-encoder-linf-v1");
  });
});
