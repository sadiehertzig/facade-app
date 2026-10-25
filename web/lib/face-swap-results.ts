// Face-swap test results shown on the "Our approach" tab
// (components/research-metrics.tsx draws them as two tables).
//
// THE TEST
//   48 AI-generated fictional source images (the set used while building and
//   tuning FACADE) x 4 fixed recipients = 192 matched pairs per upload
//   condition. For each pair a face-swap tool puts the source face onto the
//   recipient twice: once from the original photo ("clean swap") and once from
//   the FACADE-protected photo ("protected swap").
//
// WHAT "PASSES" MEANS
//   A pair passes when FaceNet (VGGFace2) similarity between the swap and the
//   source person is at least 0.4 for the clean swap (the attack worked on the
//   original), below 0.4 for the protected swap (it no longer looks like the
//   person), and drops by at least 0.1.
//
// THE INTERVALS
//   The 95% intervals come from bootstrap resampling: the 48 source images are
//   resampled with replacement many times, keeping each image's four
//   recipients together, because those four pairs are not independent.
//
// THE CONDITIONS
//   png_direct         the protected PNG is used as it is
//   jpeg_q75           it is first saved as a JPEG at quality 75
//   reupload_q75_half  JPEG 75 and shrunk to half size, as a social network might
//
// These numbers are copied from the measured results; they are not computed
// here. Change them only together with the measurement they come from.

/** One table row: one upload condition for one face-swap tool. */
export type ResultRow = {
  key: string;
  label: string;
  passes: number;
  validPairs: number;
  rate: string;
  interval: string;
};

/** One face-swap tool and its rows; `caption` is shown under its table. */
export type SwapEngine = { name: string; caption: string; rows: ResultRow[] };

const ALL_CLEAN_QUALIFIED = "All 192 clean swaps qualified in each condition; no missing results.";

// SimSwap works on 224 x 224 face crops, InSwapper on 128 x 128 ones.
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
