#!/usr/bin/env node
// A fake PhotoGuard worker for local development. It speaks the same /v1 API
// as server/worker/facade_photoguard_service.py but needs no GPU: each job
// "runs" 200 steps in about 10 seconds and returns your crop UNCHANGED.
// Nothing here protects a photo. Use it only to work on the website.
//
//   npm run fake-worker    (reads FACADE_PROCESSOR_TOKEN from .env.local)
//
// It listens on http://127.0.0.1:8000 (set FAKE_WORKER_PORT to change it).
//
// Like the real worker it holds one job at a time: a new upload gets 409 "busy"
// until the current job is cleared (DELETE), two minutes after its first
// download, or 30 minutes after upload, whichever comes first. It checks that
// both PNGs are 512 x 512 but, unlike the real worker, does not decode them or
// check that the mask is black and white.
//
// HOW IT IMITATES THE REAL WORKER
//   * Same routes: GET /v1/capabilities, POST /v1/jobs, GET and DELETE
//     /v1/jobs/<id>, GET /v1/jobs/<id>/output.
//   * Same security: every request needs "Authorization: Bearer <token>", and
//     each job has its own random job token that must be sent back in the
//     X-Job-Token header to see, clear or download that job.
//   * Same capability report (recipe, 512 px, 200 steps, 30 minutes), so the
//     website's strict check in lib/processor.ts accepts it.
//   * Same job states, progressing over time: queued -> running -> completed,
//     or cancelled when cleared.
//   * Same rules for keeping data: one job at a time, held in memory only,
//     dropped 30 minutes after upload or 2 minutes after the first download.
//
// WHAT IS DIFFERENT
//   * No GPU and no PhotoGuard: progress is just the time since upload
//     (20 steps a second), and the "result" is the uploaded crop itself.
//   * Lighter input checks (see above).
//   * Cancelling is instant, so there is no "cancelling" state.

import { randomBytes, randomUUID, timingSafeEqual } from "node:crypto";
import { createServer } from "node:http";

// The same values the real worker reports. lib/processor.ts refuses a worker
// whose report differs, so these must match it exactly.
const RECIPE = "photoguard-encoder-linf-v1";
const STEPS = 200;
// 200 steps at 20 a second: a fake job takes about 10 seconds.
const STEPS_PER_SECOND = 20;
const RETENTION_SECONDS = 1800;
const RELEASE_AFTER_OUTPUT_SECONDS = 120;
// The real worker's limit: at most 4 MiB of JSON per request.
const MAX_BODY = 4 * 1024 * 1024;
const PORT = Number(process.env.FAKE_WORKER_PORT ?? 8000);
const TOKEN = process.env.FACADE_PROCESSOR_TOKEN ?? "";

// Refuse to start without a proper token, the same rule the website applies
// (32-256 printable characters, no spaces). The website and the fake worker
// must be given the same value, usually through .env.local.
if (!/^[\x21-\x7e]{32,256}$/.test(TOKEN)) {
  console.error("Set FACADE_PROCESSOR_TOKEN to at least 32 printable characters (for example: openssl rand -hex 32).");
  process.exit(1);
}

let current = null; // the one job this worker holds: { id, token, png, startedAt, expiresAt, cancelled }

// The eight bytes every PNG file starts with.
const PNG_SIGNATURE = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);

// A 512 x 512 PNG, judged from its signature and IHDR header only.
// A PNG's first chunk is always IHDR ("image header"), at a fixed place: bytes
// 12-15 spell "IHDR", bytes 16-19 hold the width and 20-23 the height, each a
// big-endian 32-bit number. Returns the decoded bytes, or throws.
function png512(base64) {
  const png = Buffer.from(typeof base64 === "string" ? base64 : "", "base64");
  if (png.length < 24 || !png.subarray(0, 8).equals(PNG_SIGNATURE) || png.toString("latin1", 12, 16) !== "IHDR") {
    throw new Error("png");
  }
  if (png.readUInt32BE(16) !== 512 || png.readUInt32BE(20) !== 512) throw new Error("size");
  return png;
}

/** True once a job has passed its deletion time. */
function expired(job) {
  return Date.now() >= job.expiresAt;
}

/**
 * The job's status as the real worker would report it. Progress is worked out
 * from the time since upload, so nothing needs to run in the background. A
 * cancelled job stays at the step it had reached when it was cleared.
 */
function snapshot(job) {
  const elapsed = (Date.now() - job.startedAt) / 1000;
  const iteration = job.cancelled ? job.cancelledAt : Math.min(STEPS, Math.floor(elapsed * STEPS_PER_SECOND));
  const status = job.cancelled ? "cancelled" : iteration >= STEPS ? "completed" : iteration === 0 ? "queued" : "running";
  return { id: job.id, status, iteration, total_iterations: STEPS, cleanup_complete: job.cancelled };
}

/** Sends a reply with the same no-cache, no-sniff headers as the real worker. */
function send(res, code, body, type = "application/json") {
  const data = type === "application/json" ? JSON.stringify(body) : body;
  res.writeHead(code, { "Content-Type": type, "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff" });
  res.end(data);
}

/**
 * Checks the bearer token. timingSafeEqual takes the same time whether the
 * first or the last character is wrong, so an attacker cannot guess the token
 * one character at a time by measuring how long a refusal takes. (It needs
 * inputs of equal length, hence the length check first.)
 */
function authorized(req) {
  const given = Buffer.from(req.headers.authorization ?? "");
  const expected = Buffer.from(`Bearer ${TOKEN}`);
  return given.length === expected.length && timingSafeEqual(given, expected);
}

/** Reads the request body as text, refusing (413) anything over MAX_BODY. */
async function readBody(req) {
  const chunks = [];
  let size = 0;
  for await (const chunk of req) {
    size += chunk.length;
    if (size > MAX_BODY) throw Object.assign(new Error("too large"), { status: 413 });
    chunks.push(chunk);
  }
  return Buffer.concat(chunks).toString("utf8");
}

/**
 * The current job, if the id matches and the caller sent the right job token.
 * An expired job is dropped first, so it behaves as if it never existed.
 */
function findJob(id, req) {
  if (current && expired(current)) current = null;
  if (!current || current.id !== id || req.headers["x-job-token"] !== current.token) return null;
  return current;
}

// The request handler. Each block below answers one route of the /v1 API.
const server = createServer(async (req, res) => {
  // Log the method and path, with any job id hidden, like the real worker's
  // privacy-conscious logs.
  console.log(`${req.method} ${req.url?.replace(/[0-9a-f-]{36}/, "<job>")}`);
  if (!authorized(req)) return send(res, 401, { error: "unauthorized" });
  const url = new URL(req.url ?? "/", "http://localhost");

  // GET /v1/capabilities: what this worker can do (see lib/processor.ts).
  if (req.method === "GET" && url.pathname === "/v1/capabilities") {
    return send(res, 200, {
      ready: true,
      recipe: RECIPE,
      image_side: 512,
      total_iterations: STEPS,
      retention_seconds: RETENTION_SECONDS,
      experimental: true,
      gpu_smoke_validated: true, // a lie, so the website enables its button
      engine_smoke_validated: true,
      execution_device: "fake",
    });
  }

  // POST /v1/jobs: accept a new job. Body: { recipe, image_png, mask_png }.
  if (req.method === "POST" && url.pathname === "/v1/jobs") {
    // One job at a time, like the real worker: only a cleared or expired job
    // frees the slot, so a completed job still blocks a new upload.
    if (current && expired(current)) current = null;
    if (current && !current.cancelled) {
      return send(res, 409, { error: "busy" });
    }
    try {
      const body = JSON.parse(await readBody(req));
      const keys = Object.keys(body ?? {}).sort().join(",");
      if (keys !== "image_png,mask_png,recipe" || body.recipe !== RECIPE) throw new Error("fields");
      const png = png512(body.image_png);
      png512(body.mask_png);
      // A random id and a random 64-hex-character job token, as the real
      // worker makes them; the token is returned only in this one reply.
      const startedAt = Date.now();
      current = {
        id: randomUUID(),
        token: randomBytes(32).toString("hex"),
        png,
        startedAt,
        expiresAt: startedAt + RETENTION_SECONDS * 1000,
        cancelled: false,
      };
      return send(res, 202, { ...snapshot(current), access_token: current.token });
    } catch (error) {
      return send(res, error.status ?? 400, { error: error.status ? "request_too_large" : "invalid_request" });
    }
  }

  // /v1/jobs/<id> and /v1/jobs/<id>/output. match[1] is the job id and
  // match[2] is "/output" when present.
  const match = url.pathname.match(/^\/v1\/jobs\/([0-9a-f-]{36})(\/output)?$/);
  if (match) {
    const job = findJob(match[1], req);
    if (!job) return send(res, 404, { error: "not_found" });
    // DELETE /v1/jobs/<id>: clear the job.
    if (req.method === "DELETE" && !match[2]) {
      job.cancelledAt = snapshot(job).iteration;
      job.cancelled = true;
      // The cleared job stays visible as "cancelled" until the next upload.
      return send(res, 200, snapshot(job));
    }
    // GET /v1/jobs/<id>/output: the "protected" PNG (here, the unchanged
    // upload). The first download shortens the job's life to two minutes.
    if (req.method === "GET" && match[2]) {
      if (snapshot(job).status !== "completed") return send(res, 409, { error: "not_completed" });
      job.expiresAt = Math.min(job.expiresAt, Date.now() + RELEASE_AFTER_OUTPUT_SECONDS * 1000);
      return send(res, 200, job.png, "image/png");
    }
    // GET /v1/jobs/<id>: the job's progress.
    if (req.method === "GET") return send(res, 200, snapshot(job));
  }

  // Anything else is unknown.
  return send(res, 404, { error: "not_found" });
});

// Listen on this computer only (127.0.0.1), never on the network.
server.listen(PORT, "127.0.0.1", () => {
  console.log(`Fake PhotoGuard worker on http://127.0.0.1:${PORT} (returns crops unchanged; not real protection)`);
});
