// GET /api/processing/output: download this browser's finished PNG.
import { job, json, processor } from "@/lib/processor";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  const current = job(request);
  if (!current) return json({ error: "No active job." }, 404);
  try {
    const result = await processor(`/v1/jobs/${current.id}/output`, {}, current.token);
    if (!result.ok || result.headers.get("content-type")?.split(";")[0] !== "image/png") {
      return json({ error: "The result is not available." }, 409);
    }
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
