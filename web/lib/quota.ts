// Fair-use limits on the public PhotoGuard flow: how many runs one visitor,
// and the whole site, may start per day on the GPU server.
//
// WHERE IT FITS
//   browser  --->  app/api/processing/jobs/route.ts (POST)  --->  this file
//   The jobs route asks reserveRun() right before it forwards an upload to the
//   worker, and calls releaseRun() if the worker did not take the job, so a
//   refused or failed run is not charged.
//
// KEY IDEAS
//   * The site is public (no password since 2026-10-25), and every run costs
//     GPU time. Ten runs per visitor per day is plenty for trying it out; the
//     site-wide cap bounds the day's bill whatever happens.
//   * Nothing is written to disk and no address is ever stored or logged. A
//     visitor is identified only by a salted SHA-256 hash of their IP address,
//     with a salt drawn at random when the server starts, so the hashes cannot
//     be matched to addresses, or to each other across restarts. A restart
//     simply starts everyone's count again.
//   * The address comes from X-Forwarded-For. The app listens on 127.0.0.1
//     only (deploy/systemd/facade-web.service), so every request arrives
//     through Caddy, which replaces any X-Forwarded-For a client sent with the
//     client's real address (Caddy trusts no upstream proxies unless told to).
//     The last entry is used, the one written by the closest proxy. With no
//     header at all (local development) every request shares one bucket.
//
// Only app/api/processing/jobs/route.ts and its tests import this file.

import { createHash, randomBytes } from "node:crypto";

/** Runs one visitor may start in a rolling 24 hours. */
export const RUNS_PER_VISITOR_PER_DAY = 10;
/** Runs the whole site may start in a rolling 24 hours. */
export const RUNS_PER_DAY = 300;

const DAY_MS = 24 * 60 * 60 * 1000;
// How many visitor buckets to keep before expired ones are swept out.
const SWEEP_ABOVE = 5000;

type Bucket = { start: number; count: number };
type Limits = { perVisitor: number; perSite: number };

let limits: Limits = { perVisitor: RUNS_PER_VISITOR_PER_DAY, perSite: RUNS_PER_DAY };
let salt = randomBytes(16);
let visitors = new Map<string, Bucket>();
let site: Bucket = { start: 0, count: 0 };

/** A visitor's key: the salted hash of their address, never the address itself. */
export function visitorKey(request: Request): string {
  const forwarded = request.headers.get("x-forwarded-for");
  const address = forwarded ? (forwarded.split(",").pop() ?? "").trim() : "";
  return createHash("sha256")
    .update(salt)
    .update(address || "direct")
    .digest("hex");
}

export type QuotaDecision = { ok: true } | { ok: false; reason: "visitor" | "site" };

// A bucket whose day has passed starts over.
function current(bucket: Bucket | undefined, now: number): Bucket {
  return bucket && now - bucket.start < DAY_MS ? bucket : { start: now, count: 0 };
}

/**
 * Counts one run for this visitor and for the site, or refuses. Refusals
 * charge nothing. Call releaseRun() if the run then does not happen.
 */
export function reserveRun(key: string, now = Date.now()): QuotaDecision {
  site = current(site, now);
  if (site.count >= limits.perSite) return { ok: false, reason: "site" };
  const mine = current(visitors.get(key), now);
  if (mine.count >= limits.perVisitor) return { ok: false, reason: "visitor" };
  mine.count += 1;
  site.count += 1;
  visitors.set(key, mine);
  if (visitors.size > SWEEP_ABOVE) {
    for (const [other, bucket] of visitors) if (now - bucket.start >= DAY_MS) visitors.delete(other);
  }
  return { ok: true };
}

/** Gives back a run that reserveRun() counted but the worker did not take. */
export function releaseRun(key: string) {
  const mine = visitors.get(key);
  if (mine && mine.count > 0) mine.count -= 1;
  if (site.count > 0) site.count -= 1;
}

/** Tests only: forget every count, draw a new salt, and optionally change the limits. */
export function resetQuotas(override: Partial<Limits> = {}) {
  limits = { perVisitor: RUNS_PER_VISITOR_PER_DAY, perSite: RUNS_PER_DAY, ...override };
  salt = randomBytes(16);
  visitors = new Map();
  site = { start: 0, count: 0 };
}
