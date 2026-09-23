// Face-swap test results shown on the "Our approach" tab.
//
// 48 fictional development images x 4 fixed recipients = 192 matched pairs per
// upload condition. A pair "passes" when FaceNet (VGGFace2) similarity is at
// least 0.4 for the clean swap, below 0.4 for the protected swap, and drops by
// at least 0.1. The 95% intervals resample the 48 source images, keeping each
// image's four recipients together.

export type ResultRow = {
  key: string;
  label: string;
  passes: number;
  validPairs: number;
  rate: string;
  interval: string;
};

export type SwapEngine = { name: string; caption: string; rows: ResultRow[] };

const ALL_CLEAN_QUALIFIED = "All 192 clean swaps qualified in each condition; no missing results.";

export const FACE_SWAP_RESULTS: SwapEngine[] = [
  {
    name: "SimSwap 224",
    caption: ALL_CLEAN_QUALIFIED,
    rows: [
      { key: "png_direct", label: "Original PNG", passes: 176, validPairs: 192, rate: "91.7%", interval: "84.4–97.4%" },
      { key: "jpeg_q75", label: "JPEG quality 75", passes: 170, validPairs: 192, rate: "88.5%", interval: "80.7–94.8%" },
      { key: "reupload_q75_half", label: "JPEG 75 + half-size reupload", passes: 171, validPairs: 192, rate: "89.1%", interval: "81.2–95.3%" },
    ],
  },
  {
    name: "InSwapper 128",
    caption: ALL_CLEAN_QUALIFIED,
    rows: [
      { key: "png_direct", label: "Original PNG", passes: 113, validPairs: 192, rate: "58.9%", interval: "47.4–69.8%" },
      { key: "jpeg_q75", label: "JPEG quality 75", passes: 71, validPairs: 192, rate: "37.0%", interval: "25.5–49.0%" },
      { key: "reupload_q75_half", label: "JPEG 75 + half-size reupload", passes: 69, validPairs: 192, rate: "35.9%", interval: "25.0–47.4%" },
    ],
  },
];
