// /api/processing/jobs: start (POST), check (GET) and clear (DELETE) this
// browser's one PhotoGuard job. The job's id and token live in an HttpOnly
// cookie, so page scripts never see them.
import {
  capabilities,
  job,
  jobCookie,
  json,
  processor,
  readUpload,
  sameOrigin,
  status,
  validToken,
} from "@/lib/processor";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const NOT_FROM_FACADE = "This request must come from FACADE.";

export async function POST(request: Request) {
  if (!sameOrigin(request)) return json({ error: NOT_FROM_FACADE }, 403);
  if (job(request)) return json({ error: "Clear your previous job before starting another." }, 409);
  if (!(await capabilities()).ready) {
    return json({ error: "Online processing is unavailable. Your photo was not sent to a processing server." }, 503);
  }

  let upload;
  try {
    upload = await readUpload(request);
  } catch {
    return json({ error: "Choose a valid 512-pixel PNG and region mask." }, 400);
  }

  try {
    const result = await processor("/v1/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(upload),
    });
    if (!result.ok) {
      // Never pass the worker's own error text through to the page.
      const busy = result.status === 409 || result.status === 429;
      return json(
        { error: busy ? "The processor is busy. Try again later." : "The processor could not accept this job." },
        busy ? 429 : 502,
      );
    }
    const body = (await result.json()) as Record<string, unknown>;
    const safe = status(body);
    if (!validToken(body.access_token)) throw new Error("invalid token");
    return json(safe, 202, { "Set-Cookie": jobCookie(request, `${safe.id}.${body.access_token}`) });
  } catch {
    return json(
      {
        error:
          "The connection ended before we could confirm the job. The server deletes any photo it received within 30 minutes; a run already in progress may take a little longer to stop.",
      },
      502,
    );
  }
}

export async function GET(request: Request) {
  const current = job(request);
  if (!current) return json({ error: "No active job." }, 404);
  try {
    const result = await processor(`/v1/jobs/${current.id}`, {}, current.token);
    if (result.status === 404) {
      return json({ error: "This job has expired or been cleared." }, 404, { "Set-Cookie": jobCookie(request, "", 0) });
    }
    if (!result.ok) throw new Error("unavailable");
    return json(status(await result.json()));
  } catch {
    return json({ error: "Cannot check the job right now. Try again; the server still deletes it within 30 minutes." }, 503);
  }
}

export async function DELETE(request: Request) {
  if (!sameOrigin(request)) return json({ error: NOT_FROM_FACADE }, 403);
  const current = job(request);
  if (!current) return json({ status: "cancelled" });
  try {
    const result = await processor(`/v1/jobs/${current.id}`, { method: "DELETE" }, current.token);
    if (result.status === 404) return json({ status: "cancelled" }, 200, { "Set-Cookie": jobCookie(request, "", 0) });
    if (!result.ok) throw new Error("unavailable");
    const safe = status(await result.json());
    // Keep the cookie until the worker confirms the job is gone.
    return json(safe, 200, safe.status === "cancelled" ? { "Set-Cookie": jobCookie(request, "", 0) } : {});
  } catch {
    return json({ error: "Deletion was not confirmed. Retry here; the server still deletes the job within 30 minutes." }, 503);
  }
}
