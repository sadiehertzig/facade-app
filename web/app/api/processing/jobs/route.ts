// /api/processing/jobs: start (POST), check (GET) and clear (DELETE) this
// browser's one PhotoGuard job. The job's id and token live in an HttpOnly
// cookie, so page scripts never see them.
//
// THE LIFE OF A JOB
//   1. POST   the page sends the 512 x 512 crop and its mask. The gateway
//             checks the request, forwards it to the worker with the bearer
//             token, and stores the job id and job token in a cookie.
//   2. GET    the page asks every few seconds how far the job has got
//             (0 to 200 steps). The browser sends the cookie automatically.
//   3. GET /api/processing/output  (a separate route) downloads the result.
//   4. DELETE the visitor clears the job; the worker deletes it at once.
//   Even without step 4, the worker deletes every job within 30 minutes.
//
// KEY IDEAS
//   * Uploads and deletes must come from FACADE's own pages (sameOrigin, the
//     Origin check), so another website cannot trigger them.
//   * One job per browser: a second upload is refused until the first is
//     cleared. The worker itself also holds only one job at a time.
//   * Fair use: ten runs per visitor per day and 300 per day for the whole
//     site (lib/quota.ts). The site has been public since 2026-10-25, and
//     every run costs GPU time; a refused or failed run is not charged.
//   * The page only ever sees plain, fixed error messages written here, never
//     the worker's own error text, which could reveal internal details.
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
import { RUNS_PER_VISITOR_PER_DAY, releaseRun, reserveRun, visitorKey } from "@/lib/quota";

// Route settings for Next.js:
//   runtime "nodejs"         run in the standard Node.js server runtime, not
//                            the restricted "edge" runtime.
//   dynamic "force-dynamic"  never pre-render or cache this route at build
//                            time; every request asks the worker afresh.
export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// Reply for a request that failed the Origin check.
const NOT_FROM_FACADE = "This request must come from FACADE.";
// Replies for a visitor, or the whole site, that has used today's runs.
const VISITOR_LIMIT = `You have used today's ${RUNS_PER_VISITOR_PER_DAY} runs on our server. Try again tomorrow, or protect photos on your iPhone with the FACADE app.`;
const SITE_LIMIT = "Our server has reached today's limit for everyone. Try again tomorrow.";

/**
 * Starts a job. The checks run cheapest first, and the photo is forwarded only
 * after all of them pass: same origin, no job already in this browser, worker
 * ready with the exact recipe, a well-formed upload, and a run left in today's
 * fair-use allowance for this visitor and for the site.
 */
export async function POST(request: Request) {
  if (!sameOrigin(request)) return json({ error: NOT_FROM_FACADE }, 403);
  if (job(request)) return json({ error: "Clear your previous job before starting another." }, 409);
  // Ask the worker again right before uploading: the page's earlier check may
  // be minutes old, and a photo must never go to a worker that is not ready.
  if (!(await capabilities()).ready) {
    return json({ error: "Online processing is unavailable. Your photo was not sent to a processing server." }, 503);
  }

  // readUpload throws on anything wrong (size, fields, PNG shape); the details
  // stay on the server and the page gets one plain message.
  let upload;
  try {
    upload = await readUpload(request);
  } catch {
    return json({ error: "Choose a valid 512-pixel PNG and region mask." }, 400);
  }

  // Count the run only now, after the upload has passed every check, so a bad
  // request costs nothing. The count is given back if the worker does not
  // take the job.
  const visitor = visitorKey(request);
  const allowance = reserveRun(visitor);
  if (!allowance.ok) return json({ error: allowance.reason === "visitor" ? VISITOR_LIMIT : SITE_LIMIT }, 429);

  try {
    const result = await processor("/v1/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(upload),
    });
    if (!result.ok) {
      releaseRun(visitor);
      // 409 and 429 mean the worker already holds a job (it runs one at a
      // time), so the visitor is told to try again later.
      // Never pass the worker's own error text through to the page.
      const busy = result.status === 409 || result.status === 429;
      return json(
        { error: busy ? "The processor is busy. Try again later." : "The processor could not accept this job." },
        busy ? 429 : 502,
      );
    }
    // Keep only the four safe status fields for the page, and put the job
    // token where only the browser's cookie store can hold it.
    const body = (await result.json()) as Record<string, unknown>;
    const safe = status(body);
    if (!validToken(body.access_token)) throw new Error("invalid token");
    return json(safe, 202, { "Set-Cookie": jobCookie(request, `${safe.id}.${body.access_token}`) });
  } catch {
    releaseRun(visitor);
    // If the upload may have reached the worker but its reply did not come back,
    // we cannot know whether a job exists. Say so honestly, and say when the
    // photo will be deleted anyway.
    return json(
      {
        error:
          "The connection ended before we could confirm the job. The server deletes any photo it received within 30 minutes; a run already in progress may take a little longer to stop.",
      },
      502,
    );
  }
}

/** Reports the progress of this browser's job, using the job token from its cookie. */
export async function GET(request: Request) {
  const current = job(request);
  if (!current) return json({ error: "No active job." }, 404);
  try {
    const result = await processor(`/v1/jobs/${current.id}`, {}, current.token);
    // The worker no longer has the job (it expired or was cleared), so delete
    // the cookie too; the browser is then free to start a new one.
    if (result.status === 404) {
      return json({ error: "This job has expired or been cleared." }, 404, { "Set-Cookie": jobCookie(request, "", 0) });
    }
    if (!result.ok) throw new Error("unavailable");
    return json(status(await result.json()));
  } catch {
    return json({ error: "Cannot check the job right now. Try again; the server still deletes it within 30 minutes." }, 503);
  }
}

/**
 * Clears this browser's job. Answers "cancelled" when there is nothing to
 * clear, so pressing the button twice is harmless.
 */
export async function DELETE(request: Request) {
  if (!sameOrigin(request)) return json({ error: NOT_FROM_FACADE }, 403);
  const current = job(request);
  if (!current) return json({ status: "cancelled" });
  try {
    const result = await processor(`/v1/jobs/${current.id}`, { method: "DELETE" }, current.token);
    // 404: the worker had already deleted the job, which is what we wanted.
    if (result.status === 404) return json({ status: "cancelled" }, 200, { "Set-Cookie": jobCookie(request, "", 0) });
    if (!result.ok) throw new Error("unavailable");
    const safe = status(await result.json());
    // Keep the cookie until the worker confirms the job is gone. A job that is
    // still running reports "cancelling" first; the page asks again later.
    return json(safe, 200, safe.status === "cancelled" ? { "Set-Cookie": jobCookie(request, "", 0) } : {});
  } catch {
    return json({ error: "Deletion was not confirmed. Retry here; the server still deletes the job within 30 minutes." }, 503);
  }
}
