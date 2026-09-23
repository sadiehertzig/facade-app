// Route handlers tested end to end against a fake worker (see helpers.ts).
import { describe, expect, it, vi } from "vitest";
import { GET as getCapabilities } from "@/app/api/processing/capabilities/route";
import { DELETE as deleteJob, GET as getJob, POST as postJob } from "@/app/api/processing/jobs/route";
import { GET as getOutput } from "@/app/api/processing/output/route";
import {
  BEARER,
  JOB_COOKIE,
  JOB_ID,
  JOB_TOKEN,
  RECIPE,
  VALID_CAPABILITIES,
  WORKER,
  fakeWorker,
  noNetwork,
  siteRequest,
  uploadBody,
  setProductionEnv,
} from "./helpers";

const QUEUED = { id: JOB_ID, status: "queued", iteration: 0, total_iterations: 200 };

function postUpload(init: { cookie?: string; origin?: string | null; headers?: Record<string, string> } = {}) {
  return siteRequest("/api/processing/jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json", ...init.headers },
    body: uploadBody(),
    cookie: init.cookie,
    origin: init.origin,
  });
}

/** Worker that is ready and answers POST /v1/jobs with the given status code. */
function workerAccepting(code: number, body: unknown) {
  return fakeWorker((url) => {
    if (url.endsWith("/v1/capabilities")) return Response.json(VALID_CAPABILITIES);
    if (url.endsWith("/v1/jobs")) return Response.json(body, { status: code });
    throw new Error(`unexpected ${url}`);
  });
}

describe("GET /api/processing/capabilities", () => {
  it("reports unavailable, uncached, when no worker is configured", async () => {
    vi.spyOn(console, "warn").mockImplementation(() => {});
    setProductionEnv({ FACADE_PROCESSOR_URL: "", FACADE_PROCESSOR_TOKEN: "" });
    noNetwork();
    const response = await getCapabilities();
    expect(await response.json()).toEqual({ ready: false });
    expect(response.headers.get("Cache-Control")).toBe("no-store");
  });

  it("reports ready for the validated encoder worker", async () => {
    setProductionEnv();
    fakeWorker(() => Response.json(VALID_CAPABILITIES));
    expect((await (await getCapabilities()).json()).ready).toBe(true);
  });
});

describe("POST /api/processing/jobs", () => {
  it("refuses a request from another site before contacting the worker", async () => {
    setProductionEnv();
    const calls = noNetwork();
    expect((await postJob(postUpload({ origin: "https://evil.example" }))).status).toBe(403);
    expect((await postJob(postUpload({ origin: null }))).status).toBe(403);
    expect((await postJob(postUpload({ headers: { "Sec-Fetch-Site": "cross-site" } }))).status).toBe(403);
    expect(calls).toHaveLength(0);
  });

  it("refuses a second job while one is active", async () => {
    setProductionEnv();
    noNetwork();
    expect((await postJob(postUpload({ cookie: JOB_COOKIE }))).status).toBe(409);
  });

  it("refuses to accept a job when the worker is not ready, without uploading", async () => {
    vi.spyOn(console, "warn").mockImplementation(() => {});
    setProductionEnv();
    const calls = fakeWorker(() => Response.json({ ...VALID_CAPABILITIES, ready: false }));
    const response = await postJob(postUpload());
    expect(response.status).toBe(503);
    expect(calls.map((c) => c.url)).toEqual([`${WORKER}/v1/capabilities`]);
  });

  it("refuses an invalid upload", async () => {
    setProductionEnv();
    workerAccepting(202, {});
    const bad = siteRequest("/api/processing/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: uploadBody("photoguard-diffusion-l2-v1"),
    });
    expect((await postJob(bad)).status).toBe(400);
  });

  it.each([
    [409, 429],
    [429, 429],
    [500, 502],
    [202, 202],
  ])("worker %i becomes gateway %i without leaking worker details", async (workerCode, gatewayCode) => {
    setProductionEnv();
    const workerBody = workerCode === 202 ? { ...QUEUED, access_token: JOB_TOKEN } : { error: "PRIVATE_DETAIL" };
    const calls = workerAccepting(workerCode, workerBody);

    const response = await postJob(postUpload());
    expect(response.status).toBe(gatewayCode);

    const submit = calls.find((c) => c.url === `${WORKER}/v1/jobs`)!;
    expect(JSON.parse(String(submit.init.body)).recipe).toBe(RECIPE);
    expect(new Headers(submit.init.headers).get("Authorization")).toBe(`Bearer ${BEARER}`);
    expect(new Headers(submit.init.headers).get("Content-Type")).toBe("application/json");

    const body = await response.json();
    expect(JSON.stringify(body)).not.toContain("PRIVATE_DETAIL");
    expect(body).not.toHaveProperty("access_token");
    expect(response.headers.get("Cache-Control")).toBe("no-store");

    if (workerCode === 202) {
      expect(body).toEqual(QUEUED);
      const cookie = response.headers.get("Set-Cookie")!;
      expect(cookie).toContain(`facade_job=${JOB_ID}.${JOB_TOKEN}`);
      for (const flag of ["HttpOnly", "SameSite=Strict", "Secure", "Path=/api/processing"]) expect(cookie).toContain(flag);
    } else {
      expect(response.headers.get("Set-Cookie")).toBeNull();
    }
  });

  it("does not set a cookie when the worker returns a malformed job token", async () => {
    setProductionEnv();
    workerAccepting(202, { ...QUEUED, access_token: "short" });
    const response = await postJob(postUpload());
    expect(response.status).toBe(502);
    expect(response.headers.get("Set-Cookie")).toBeNull();
  });
});

describe("GET /api/processing/jobs", () => {
  it("returns 404 when this browser has no job", async () => {
    setProductionEnv();
    noNetwork();
    expect((await getJob(siteRequest("/api/processing/jobs"))).status).toBe(404);
  });

  it("returns the sanitized status and sends the job token to the worker", async () => {
    setProductionEnv();
    const calls = fakeWorker(() => Response.json({ ...QUEUED, status: "running", iteration: 40, cleanup_complete: false }));
    const response = await getJob(siteRequest("/api/processing/jobs", { cookie: JOB_COOKIE }));
    expect(await response.json()).toEqual({ ...QUEUED, status: "running", iteration: 40 });
    expect(calls[0].url).toBe(`${WORKER}/v1/jobs/${JOB_ID}`);
    expect(new Headers(calls[0].init.headers).get("X-Job-Token")).toBe(JOB_TOKEN);
  });

  it("clears the cookie when the worker no longer has the job", async () => {
    setProductionEnv();
    fakeWorker(() => Response.json({ error: "not_found" }, { status: 404 }));
    const response = await getJob(siteRequest("/api/processing/jobs", { cookie: JOB_COOKIE }));
    expect(response.status).toBe(404);
    expect(response.headers.get("Set-Cookie")).toContain("Max-Age=0");
  });

  it("returns 503 when the worker cannot be reached", async () => {
    setProductionEnv();
    noNetwork();
    expect((await getJob(siteRequest("/api/processing/jobs", { cookie: JOB_COOKIE }))).status).toBe(503);
  });
});

describe("DELETE /api/processing/jobs", () => {
  const del = (init: { cookie?: string; origin?: string | null } = {}) =>
    siteRequest("/api/processing/jobs", { method: "DELETE", ...init });

  it("refuses a request from another site", async () => {
    setProductionEnv();
    noNetwork();
    expect((await deleteJob(del({ cookie: JOB_COOKIE, origin: "https://evil.example" }))).status).toBe(403);
  });

  it("clears the cookie once the worker confirms cancellation", async () => {
    setProductionEnv();
    const calls = fakeWorker(() => Response.json({ ...QUEUED, status: "cancelled" }));
    const response = await deleteJob(del({ cookie: JOB_COOKIE }));
    expect(response.status).toBe(200);
    expect(response.headers.get("Set-Cookie")).toContain("Max-Age=0");
    expect(calls[0].init.method).toBe("DELETE");
  });

  it("keeps the cookie while the worker is still cancelling", async () => {
    setProductionEnv();
    fakeWorker(() => Response.json({ ...QUEUED, status: "cancelling", iteration: 90 }));
    const response = await deleteJob(del({ cookie: JOB_COOKIE }));
    expect((await response.json()).status).toBe("cancelling");
    expect(response.headers.get("Set-Cookie")).toBeNull();
  });

  it("treats an already-expired job as cancelled", async () => {
    setProductionEnv();
    fakeWorker(() => Response.json({ error: "not_found" }, { status: 404 }));
    const response = await deleteJob(del({ cookie: JOB_COOKIE }));
    expect(await response.json()).toEqual({ status: "cancelled" });
    expect(response.headers.get("Set-Cookie")).toContain("Max-Age=0");
  });
});

describe("GET /api/processing/output", () => {
  it("returns 404 when this browser has no job", async () => {
    setProductionEnv();
    noNetwork();
    expect((await getOutput(siteRequest("/api/processing/output"))).status).toBe(404);
  });

  it("streams the worker's PNG as a download", async () => {
    setProductionEnv();
    const png = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
    const calls = fakeWorker(() => new Response(png, { headers: { "Content-Type": "image/png" } }));
    const response = await getOutput(siteRequest("/api/processing/output", { cookie: JOB_COOKIE }));
    expect(response.status).toBe(200);
    expect(response.headers.get("Content-Type")).toBe("image/png");
    expect(response.headers.get("Content-Disposition")).toContain("attachment");
    expect(new Uint8Array(await response.arrayBuffer())).toEqual(png);
    expect(calls[0].url).toBe(`${WORKER}/v1/jobs/${JOB_ID}/output`);
  });

  it("refuses anything that is not a PNG", async () => {
    setProductionEnv();
    fakeWorker(() => Response.json({ error: "not_ready" }, { status: 409 }));
    expect((await getOutput(siteRequest("/api/processing/output", { cookie: JOB_COOKIE }))).status).toBe(409);
  });
});
