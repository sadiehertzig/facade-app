// Tests for the server-side gateway, lib/processor.ts, function by function.
// Each safety rule described there has a test here: the strict capability
// check, the worker URL and token rules, the upload checks, the Origin check,
// the job cookie's flags, and the reduction of worker replies to four fields.
// routes.test.ts then tests the same rules through the API routes.
import { describe, expect, it, vi } from "vitest";
import { capabilities, job, jobCookie, readUpload, sameOrigin, status } from "@/lib/processor";
import {
  BEARER,
  JOB_COOKIE,
  JOB_ID,
  JOB_TOKEN,
  PNG_BASE64,
  RECIPE,
  SITE,
  VALID_CAPABILITIES,
  WORKER,
  fakeWorker,
  noNetwork,
  siteRequest,
  uploadBody,
  setProductionEnv,
} from "./helpers";

// The worker's capability report is accepted only when every field matches.
describe("capabilities", () => {
  const cases: [string, Record<string, unknown>, boolean][] = [
    ["validated encoder accepted", {}, true],
    ["diffusion worker refused", { recipe: "photoguard-diffusion-l2-v1" }, false],
    ["phone-size worker refused", { image_side: 256 }, false],
    ["unvalidated GPU refused", { gpu_smoke_validated: false }, false],
    ["reduced iterations refused", { total_iterations: 20 }, false],
    ["wrong expiry refused", { retention_seconds: 3600 }, false],
    ["non-experimental refused", { experimental: false }, false],
    ["offline refused", { ready: false }, false],
  ];

  it.each(cases)("%s", async (_label, change, expected) => {
    setProductionEnv();
    const calls = fakeWorker(() => Response.json({ ...VALID_CAPABILITIES, ...change }));
    expect((await capabilities()).ready).toBe(expected);
    expect(calls).toHaveLength(1);
    expect(calls[0].url).toBe(`${WORKER}/v1/capabilities`);
    expect(new Headers(calls[0].init.headers).get("Authorization")).toBe(`Bearer ${BEARER}`);
    expect(calls[0].init.redirect).toBe("manual");
    expect(calls[0].init.signal).toBeInstanceOf(AbortSignal);
  });

  it("is unavailable without calling anything when the worker is not configured", async () => {
    vi.spyOn(console, "warn").mockImplementation(() => {});
    setProductionEnv({ FACADE_PROCESSOR_URL: "" });
    const calls = noNetwork();
    expect(await capabilities()).toEqual({ ready: false });
    expect(calls).toHaveLength(0);
  });

  it("is unavailable when the worker errors or cannot be reached", async () => {
    vi.spyOn(console, "warn").mockImplementation(() => {});
    setProductionEnv();
    fakeWorker(() => Response.json({ error: "internal_error" }, { status: 500 }));
    expect((await capabilities()).ready).toBe(false);
    fakeWorker(() => {
      throw new TypeError(`fetch failed for ${WORKER} with ${"f".repeat(64)}`);
    });
    expect((await capabilities()).ready).toBe(false);
  });

  it("never logs the worker address or the bearer", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    setProductionEnv();
    fakeWorker(() => {
      throw new TypeError(`connect failed ${WORKER}/v1 ${"f".repeat(64)}`);
    });
    await capabilities();
    const logged = JSON.stringify(warn.mock.calls);
    expect(logged).not.toContain("worker.example");
    expect(logged).not.toContain("f".repeat(64));
    expect(logged).not.toContain(BEARER);
  });
});

// Only https://, or http:// on this computer, with nothing hidden in the URL;
// a refused setting means no request is made at all.
describe("worker URL and token rules", () => {
  const accepted = ["https://worker.example", "https://worker.example/base/", "http://127.0.0.1:8000", "http://localhost:8000"];
  const refused = [
    "http://worker.example",
    "http://10.0.0.5:8000",
    "http://127.0.0.1.nip.io:8000",
    "https://user:pass@worker.example",
    "https://worker.example/?x=1",
    "https://worker.example/#frag",
    "ftp://worker.example",
    "not a url",
  ];

  it.each(accepted)("accepts %s", async (url) => {
    setProductionEnv({ FACADE_PROCESSOR_URL: url });
    const calls = fakeWorker(() => Response.json(VALID_CAPABILITIES));
    expect((await capabilities()).ready).toBe(true);
    expect(calls[0].url).toBe(url.replace(/\/$/, "") + "/v1/capabilities");
  });

  it.each(refused)("refuses %s", async (url) => {
    vi.spyOn(console, "warn").mockImplementation(() => {});
    setProductionEnv({ FACADE_PROCESSOR_URL: url });
    const calls = noNetwork();
    expect((await capabilities()).ready).toBe(false);
    expect(calls).toHaveLength(0);
  });

  it.each([
    ["too short", "s".repeat(31)],
    ["contains a space", "s".repeat(20) + " " + "s".repeat(20)],
    ["contains a newline", "s".repeat(40) + "\nX-Other: 1"],
  ])("refuses a token that is %s", async (_label, token) => {
    vi.spyOn(console, "warn").mockImplementation(() => {});
    setProductionEnv({ FACADE_PROCESSOR_TOKEN: token });
    const calls = noNetwork();
    expect((await capabilities()).ready).toBe(false);
    expect(calls).toHaveLength(0);
  });
});

// Uploads must be JSON with exactly {recipe, image_png, mask_png}, PNGs only,
// at most 4 MB.
describe("readUpload", () => {
  const post = (body: string, contentType = "application/json") =>
    new Request(`${SITE}/api/processing/jobs`, { method: "POST", headers: { "Content-Type": contentType }, body });

  it("accepts the encoder request", async () => {
    expect((await readUpload(post(uploadBody()))).recipe).toBe(RECIPE);
  });

  it("refuses a stale diffusion request", async () => {
    await expect(readUpload(post(uploadBody("photoguard-diffusion-l2-v1")))).rejects.toThrow();
  });

  it("refuses extra or missing fields", async () => {
    await expect(readUpload(post(JSON.stringify({ recipe: RECIPE, image_png: PNG_BASE64 })))).rejects.toThrow();
    await expect(
      readUpload(post(JSON.stringify({ recipe: RECIPE, image_png: PNG_BASE64, mask_png: PNG_BASE64, filename: "me.jpg" }))),
    ).rejects.toThrow();
  });

  it("refuses data that is not a base64 PNG", async () => {
    const jpeg = "/9j/4AAQSkZJRgABAQ"; // base64 of a JPEG header
    await expect(readUpload(post(JSON.stringify({ recipe: RECIPE, image_png: jpeg, mask_png: PNG_BASE64 })))).rejects.toThrow();
  });

  it("refuses the wrong content type", async () => {
    await expect(readUpload(post(uploadBody(), "text/plain"))).rejects.toThrow();
  });

  it("refuses bodies over 4 MB", async () => {
    const huge = JSON.stringify({ recipe: RECIPE, image_png: PNG_BASE64 + "A".repeat(4 * 1024 * 1024), mask_png: PNG_BASE64 });
    await expect(readUpload(post(huge))).rejects.toThrow("too large");
  });
});

// The Origin check: uploads and deletes only from FACADE's own pages.
describe("sameOrigin", () => {
  it("accepts the public origin even though request.url is the local listen address", () => {
    setProductionEnv();
    expect(sameOrigin(siteRequest("/api/processing/jobs", { method: "POST" }))).toBe(true);
  });

  it("refuses other origins, a missing Origin header, and cross-site fetches", () => {
    setProductionEnv();
    expect(sameOrigin(siteRequest("/api/processing/jobs", { origin: "https://evil.example" }))).toBe(false);
    expect(sameOrigin(siteRequest("/api/processing/jobs", { origin: "http://127.0.0.1:3000" }))).toBe(false);
    expect(sameOrigin(siteRequest("/api/processing/jobs", { origin: null }))).toBe(false);
    expect(sameOrigin(siteRequest("/api/processing/jobs", { headers: { "Sec-Fetch-Site": "cross-site" } }))).toBe(false);
  });

  it("fails closed in production when FACADE_PUBLIC_ORIGIN is missing or not a bare origin", () => {
    for (const value of ["", "https://facade.example/path", "http://facade.example", "nonsense"]) {
      setProductionEnv({ FACADE_PUBLIC_ORIGIN: value });
      expect(sameOrigin(siteRequest("/api/processing/jobs", { origin: "http://127.0.0.1:3000" }))).toBe(false);
      expect(sameOrigin(siteRequest("/api/processing/jobs"))).toBe(false);
    }
  });

  it("uses the request's own origin in development when FACADE_PUBLIC_ORIGIN is unset", () => {
    setProductionEnv({ NODE_ENV: "development", FACADE_PUBLIC_ORIGIN: "" });
    expect(sameOrigin(siteRequest("/api/processing/jobs", { origin: "http://127.0.0.1:3000" }))).toBe(true);
    expect(sameOrigin(siteRequest("/api/processing/jobs"))).toBe(false);
  });
});

// The job cookie must be unreadable by page scripts and sent only to
// /api/processing, and a malformed cookie must be ignored.
describe("job cookie", () => {
  it("is HttpOnly, SameSite=Strict, scoped to /api/processing and Secure on an https site", () => {
    setProductionEnv();
    const cookie = jobCookie(siteRequest("/api/processing/jobs"), `${JOB_ID}.${JOB_TOKEN}`);
    for (const part of ["HttpOnly", "SameSite=Strict", "Path=/api/processing", "Max-Age=1800", "Secure"]) {
      expect(cookie).toContain(part);
    }
  });

  it("drops Secure only for a local http:// development origin", () => {
    setProductionEnv({ NODE_ENV: "development", FACADE_PUBLIC_ORIGIN: "http://localhost:3000" });
    expect(jobCookie(siteRequest("/api/processing/jobs"), "x")).not.toContain("Secure");
  });

  it("is read back only when the id and token have the right shape", () => {
    expect(job(siteRequest("/", { cookie: `other=1; ${JOB_COOKIE}` }))).toEqual({ id: JOB_ID, token: JOB_TOKEN });
    expect(job(siteRequest("/", { cookie: `facade_job=${JOB_ID}.short` }))).toBeNull();
    expect(job(siteRequest("/", { cookie: `facade_job=../../etc.${JOB_TOKEN}` }))).toBeNull();
    expect(job(siteRequest("/"))).toBeNull();
  });
});

// Worker replies are reduced to four checked fields before the page sees them.
describe("status", () => {
  it("keeps only the four fields the page needs", () => {
    const raw = { id: JOB_ID, status: "running", iteration: 12, total_iterations: 200, cleanup_complete: false, access_token: JOB_TOKEN };
    expect(status(raw)).toEqual({ id: JOB_ID, status: "running", iteration: 12, total_iterations: 200 });
  });

  it.each([
    ["an unknown state", { status: "exploded" }],
    ["too many steps", { iteration: 201 }],
    ["a fractional step", { iteration: 1.5 }],
    ["a different recipe length", { total_iterations: 20 }],
    ["a bad id", { id: "not-a-uuid" }],
  ])("rejects %s", (_label, change) => {
    expect(() => status({ id: JOB_ID, status: "running", iteration: 0, total_iterations: 200, ...change })).toThrow();
  });
});
