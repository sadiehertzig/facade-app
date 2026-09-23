// Server-side gateway to the PhotoGuard worker (server/worker/ in this repo).
//
// Only the API routes under app/api/processing import this file. The browser
// never talks to the worker directly and never sees the worker's bearer token.
//
// Settings come from environment variables (see .env.example):
//   FACADE_PROCESSOR_URL    worker base URL; https://, or http:// on 127.0.0.1/localhost
//   FACADE_PROCESSOR_TOKEN  worker bearer token (32-256 printable characters)
//   FACADE_PUBLIC_ORIGIN    the site's public origin, e.g. https://facade.example.org
//
// Behind Caddy, request.url is the local listen address (http://127.0.0.1:3000),
// so the same-origin check and the cookie's Secure flag use FACADE_PUBLIC_ORIGIN.

import { PHOTO_GUARD_RECIPE } from "@/lib/protection-recipe";

export const RECIPE = PHOTO_GUARD_RECIPE;

// The worker only accepts this exact recipe. If it reports anything else, the
// site says "unavailable" instead of sending photos to it.
const EXPECTED = {
  recipe: RECIPE,
  image_side: 512,
  total_iterations: 200,
  retention_seconds: 1800,
  experimental: true,
} as const;

const MAX_UPLOAD_BYTES = 4 * 1024 * 1024; // whole JSON request body
const MAX_PNG_BASE64_CHARS = 2 * 1024 * 1024; // each of image_png and mask_png
const WORKER_TIMEOUT_MS = 15_000;
const JOB_COOKIE = "facade_job";
const JOB_COOKIE_SECONDS = 1800;

const UUID = /^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$/;
const JOB_TOKEN = /^[a-f0-9]{64}$/;
const BEARER = /^[\x21-\x7e]{32,256}$/; // printable ASCII, no spaces
const PNG_BASE64 = /^iVBORw0KGgo[A-Za-z0-9+/]*={0,2}$/; // base64 that starts with the PNG signature
const JOB_STATES = new Set(["queued", "running", "completed", "cancelling", "cancelled", "failed"]);

export type JobStatus = { id: string; status: string; iteration: number; total_iterations: number };

/** JSON response that is never cached. */
export function json(data: unknown, status = 200, headers: Record<string, string> = {}) {
  return Response.json(data, {
    status,
    headers: { "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff", ...headers },
  });
}

function isLoopback(hostname: string) {
  return hostname === "127.0.0.1" || hostname === "localhost";
}

/** Worker URL and token, or null when they are missing or unsafe. */
function workerSettings(): { url: string; token: string } | null {
  const address = process.env.FACADE_PROCESSOR_URL;
  const token = process.env.FACADE_PROCESSOR_TOKEN;
  if (!address || !token || !BEARER.test(token)) return null;
  try {
    const url = new URL(address);
    const schemeOk = url.protocol === "https:" || (url.protocol === "http:" && isLoopback(url.hostname));
    if (!schemeOk || url.username || url.password || url.search || url.hash) return null;
    return { url: url.href.replace(/\/$/, ""), token };
  } catch {
    return null;
  }
}

/**
 * The origin the browser uses for this site. In development (`npm run dev`)
 * it falls back to the request's own origin. In production it must be set;
 * without it, uploads and deletes are refused (fail closed).
 */
export function publicOrigin(request: Request): URL | null {
  const configured = process.env.FACADE_PUBLIC_ORIGIN;
  if (!configured) {
    return process.env.NODE_ENV === "production" ? null : new URL(new URL(request.url).origin);
  }
  try {
    const url = new URL(configured);
    const schemeOk = url.protocol === "https:" || (url.protocol === "http:" && isLoopback(url.hostname));
    const bareOrigin = url.pathname === "/" && !url.search && !url.hash && !url.username && !url.password;
    return schemeOk && bareOrigin ? url : null;
  } catch {
    return null;
  }
}

/** True only for requests sent by this site's own pages. Used for POST and DELETE. */
export function sameOrigin(request: Request) {
  const origin = publicOrigin(request);
  return (
    origin !== null &&
    request.headers.get("origin") === origin.origin &&
    request.headers.get("sec-fetch-site") !== "cross-site"
  );
}

/** Calls the worker with the bearer token (and the job's own token, if given). */
export async function processor(path: string, init: RequestInit = {}, jobToken?: string) {
  const settings = workerSettings();
  if (!settings) throw new Error("unavailable");
  const headers = new Headers(init.headers);
  headers.set("Authorization", `Bearer ${settings.token}`);
  headers.set("User-Agent", "FACADE-Web/1.0");
  if (jobToken) headers.set("X-Job-Token", jobToken);
  return fetch(settings.url + path, {
    ...init,
    headers,
    redirect: "manual",
    signal: AbortSignal.timeout(WORKER_TIMEOUT_MS),
  });
}

/** Removes URLs and long hex strings (tokens) from an error message before logging it. */
function redact(message: string) {
  return message
    .replace(/https?:\/\/[^\s]+/g, "[endpoint]")
    .replace(/[a-f0-9]{32,}/gi, "[redacted]")
    .slice(0, 180);
}

/**
 * Asks the worker what it can do. `ready` is true only when the worker reports
 * the exact expected recipe and a passed GPU smoke test.
 */
export async function capabilities() {
  if (!workerSettings()) {
    console.warn("FACADE capabilities unavailable: processor configuration missing");
    return { ready: false };
  }
  try {
    const response = await processor("/v1/capabilities");
    if (!response.ok) {
      console.warn("FACADE capabilities upstream rejected", { status: response.status });
      return { ready: false };
    }
    const reported = (await response.json()) as Record<string, unknown>;
    const ready =
      reported.ready === true &&
      reported.gpu_smoke_validated === true &&
      reported.recipe === EXPECTED.recipe &&
      reported.image_side === EXPECTED.image_side &&
      reported.total_iterations === EXPECTED.total_iterations &&
      reported.retention_seconds === EXPECTED.retention_seconds &&
      reported.experimental === EXPECTED.experimental;
    if (!ready) console.warn("FACADE capabilities recipe mismatch or worker not ready");
    return { ready, ...EXPECTED };
  } catch (error) {
    console.warn("FACADE capabilities request failed", {
      kind: error instanceof Error ? error.name : "unknown",
      detail: error instanceof Error ? redact(error.message) : "unknown",
    });
    return { ready: false };
  }
}

/** Reads this browser's job id and job token from the facade_job cookie. */
export function job(request: Request): { id: string; token: string } | null {
  const value = request.headers
    .get("cookie")
    ?.split(";")
    .map((part) => part.trim())
    .find((part) => part.startsWith(`${JOB_COOKIE}=`))
    ?.slice(JOB_COOKIE.length + 1);
  if (!value) return null;
  const [id, token] = value.split(".");
  return UUID.test(id ?? "") && JOB_TOKEN.test(token ?? "") ? { id, token } : null;
}

/**
 * Set-Cookie value for the job cookie. HttpOnly keeps it away from page
 * scripts; SameSite=Strict and the narrow Path keep it off other requests.
 * Use age 0 to delete it.
 */
export function jobCookie(request: Request, value: string, age = JOB_COOKIE_SECONDS) {
  const origin = publicOrigin(request);
  const secure = origin === null || origin.protocol === "https:";
  return `${JOB_COOKIE}=${value}; HttpOnly; SameSite=Strict; Path=/api/processing; Max-Age=${age}${secure ? "; Secure" : ""}`;
}

/** Keeps only the four job fields the page needs, and checks each one. */
export function status(value: unknown): JobStatus {
  const v = value as Record<string, unknown> | null;
  if (
    !v ||
    !UUID.test(String(v.id)) ||
    !JOB_STATES.has(String(v.status)) ||
    !Number.isInteger(v.iteration) ||
    Number(v.iteration) < 0 ||
    Number(v.iteration) > EXPECTED.total_iterations ||
    v.total_iterations !== EXPECTED.total_iterations
  ) {
    throw new Error("invalid status");
  }
  return {
    id: String(v.id),
    status: String(v.status),
    iteration: Number(v.iteration),
    total_iterations: EXPECTED.total_iterations,
  };
}

export function validToken(value: unknown): value is string {
  return typeof value === "string" && JOB_TOKEN.test(value);
}

/**
 * Reads and checks the upload: JSON with exactly {recipe, image_png, mask_png},
 * both PNGs base64-encoded, the whole body at most 4 MB. Throws otherwise.
 */
export async function readUpload(request: Request) {
  if (request.headers.get("content-type")?.split(";")[0] !== "application/json") throw new Error("invalid upload");
  const reader = request.body?.getReader();
  if (!reader) throw new Error("invalid upload");

  // Read the body in chunks so an oversized upload is cut off early.
  const chunks: Uint8Array[] = [];
  let size = 0;
  while (true) {
    const part = await reader.read();
    if (part.done) break;
    size += part.value.length;
    if (size > MAX_UPLOAD_BYTES) {
      await reader.cancel();
      throw new Error("too large");
    }
    chunks.push(part.value);
  }
  const bytes = new Uint8Array(size);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.length;
  }

  const value = JSON.parse(new TextDecoder().decode(bytes));
  if (!value || typeof value !== "object" || Object.keys(value).sort().join(",") !== "image_png,mask_png,recipe" || value.recipe !== RECIPE) {
    throw new Error("invalid upload");
  }
  for (const key of ["image_png", "mask_png"]) {
    const png = value[key];
    if (typeof png !== "string" || png.length > MAX_PNG_BASE64_CHARS || !PNG_BASE64.test(png)) throw new Error("invalid PNG");
  }
  return value as { recipe: string; image_png: string; mask_png: string };
}
