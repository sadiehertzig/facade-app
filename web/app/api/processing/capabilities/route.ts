// GET /api/processing/capabilities: is the PhotoGuard worker ready right now?
//
// The "Prepare a photo" page calls this when it loads, and again when the
// visitor presses "Check availability". The answer decides whether the upload
// button is enabled. All the checking happens in capabilities() in
// lib/processor.ts; this file only exposes it at a URL.
//
// The reply is { ready: false } or { ready: true, recipe, image_side, ... }. It
// never contains the worker's address or token.
import { capabilities, json } from "@/lib/processor";

// Route settings for Next.js:
//   runtime "nodejs"         run in the standard Node.js server runtime, not
//                            the restricted "edge" runtime.
//   dynamic "force-dynamic"  never pre-render or cache this route at build
//                            time; every request asks the worker afresh.
export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/** Answers whether the worker is ready, uncached. */
export async function GET() {
  return json(await capabilities());
}
