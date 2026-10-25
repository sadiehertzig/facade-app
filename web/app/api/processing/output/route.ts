// GET /api/processing/output: download this browser's finished PNG.
//
// The browser's job cookie says which job to fetch; the gateway asks the
// worker for it with the bearer token and the job token, and streams the
// image straight back without storing it. Only a reply that really is a PNG is
// passed on.
//
// After the first download the worker deletes the job about two minutes later
// (and in any case 30 minutes after upload), so the visitor should save it.
import { job, json, processor } from "@/lib/processor";

// Route settings for Next.js:
//   runtime "nodejs"         run in the standard Node.js server runtime, not
//                            the restricted "edge" runtime.
//   dynamic "force-dynamic"  never pre-render or cache this route at build
//                            time; every request asks the worker afresh.
export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/** Streams the protected PNG as a file download, or a JSON error. */
export async function GET(request: Request) {
  const current = job(request);
  if (!current) return json({ error: "No active job." }, 404);
  try {
    const result = await processor(`/v1/jobs/${current.id}/output`, {}, current.token);
    if (!result.ok || result.headers.get("content-type")?.split(";")[0] !== "image/png") {
      return json({ error: "The result is not available." }, 409);
    }
    // result.body is passed through as a stream, so the image is never held
    // in full on this server or written to disk.
    //   Content-Disposition: attachment  makes the browser save it as a file
    //                                    named facade-photoguard.png
    //   Cache-Control: no-store          no copy is kept in any cache
    return new Response(result.body, {
      headers: {
        "Content-Type": "image/png",
        "Content-Disposition": "attachment; filename=facade-photoguard.png",
        "Cache-Control": "no-store",
        "X-Content-Type-Options": "nosniff",
      },
    });
  } catch {
    return json({ error: "Cannot download the result right now." }, 503);
  }
}
