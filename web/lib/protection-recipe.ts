// Shared by the upload UI and server gate. A different worker recipe stays unavailable.
export const PHOTO_GUARD_RECIPE = "photoguard-encoder-linf-v1";

// The worker and the PhotoGuard demo run Stable Diffusion 1.5. Its license
// (CreativeML OpenRAIL-M) requires users of a hosted service to follow its use
// restrictions, so the page links to them wherever a photo is sent.
export const OPENRAIL_LICENSE_URL = "https://huggingface.co/spaces/CompVis/stable-diffusion-license";
