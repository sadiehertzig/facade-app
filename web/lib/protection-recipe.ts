// Constants shared by the upload workspace (components/protection-workspace.tsx),
// the home page (app/page.tsx) and the server-side gateway (lib/processor.ts).
// Keeping them in one file means the page and the server can never ask for
// different things.
//
// The name of the one protection recipe the site uses: PhotoGuard's "encoder"
// attack, limited by an L-infinity budget (no pixel value changes by more than
// a small fixed amount), version 1. It is shared by the upload UI and the
// server gate; a worker that reports a different recipe stays unavailable.
export const PHOTO_GUARD_RECIPE = "photoguard-encoder-linf-v1";

// The worker and the PhotoGuard demo run Stable Diffusion 1.5. Its license
// (CreativeML OpenRAIL-M) requires users of a hosted service to follow its use
// restrictions, so the page links to them wherever a photo is sent.
export const OPENRAIL_LICENSE_URL = "https://huggingface.co/spaces/CompVis/stable-diffusion-license";
