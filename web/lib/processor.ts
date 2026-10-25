// Server-side gateway to the PhotoGuard worker (server/worker/ in this repo).
//
// WHERE IT FITS
//   browser  --->  app/api/processing/*/route.ts  --->  this file  --->  worker /v1/*
//   The three API routes are thin: every safety rule lives here, in one place,
//   so it can be tested on its own (tests/processor.test.ts).
//
// KEY IDEAS
//   * The browser never talks to the worker directly and never sees the
//     worker's bearer token. The token is read from a server-only environment
//     variable and added to the request here, on the server. Anything the
//     browser could read (page code, cookies it can see, responses) never
//     contains it.
//   * Fail closed. If a setting is missing or looks wrong, or the worker
//     reports anything other than the exact expected recipe, the site says
//     "unavailable" rather than sending a photo somewhere it should not go.
//   * Check everything in both directions. Uploads are size-limited and
//     shape-checked before they are forwarded; worker replies are reduced to
//     the few fields the page needs before they are passed back.
//   * Each browser gets its own job id and job token, kept in an HttpOnly
//     cookie (see jobCookie below), so one visitor cannot see another's job.
//
// Only the API routes under app/api/processing import this file.
//
// Settings come from environment variables (see .env.example):
//   FACADE_PROCESSOR_URL    worker base URL; https://, or http:// on 127.0.0.1/localhost
//   FACADE_PROCESSOR_TOKEN  worker bearer token (32-256 printable characters)
//   FACADE_PUBLIC_ORIGIN    the site's public origin, e.g. https://facade.example.org
//
// Behind Caddy, request.url is the local listen address (http://127.0.0.1:3000),
// so the same-origin check and the cookie's Secure flag use FACADE_PUBLIC_ORIGIN.
// (Caddy is the web server in front of this app; see deploy/Caddyfile.)

import { PHOTO_GUARD_RECIPE } from "@/lib/protection-recipe";

// The recipe name is shared with the upload page (lib/protection-recipe.ts), so
// the page and the gateway can never disagree about what they are asking for.
export const RECIPE = PHOTO_GUARD_RECIPE;

// The worker only accepts this exact recipe. If it reports anything else, the
// site says "unavailable" instead of sending photos to it.
//   image_side         the worker processes exactly 512 x 512 pixels
//   total_iterations   200 optimisation steps, the full recipe (not a shortened run)
//   retention_seconds  the worker deletes every job at most 30 minutes after upload
//   experimental       the worker labels its output as experimental
const EXPECTED = {
  recipe: RECIPE,
  image_side: 512,
  total_iterations: 200,
  retention_seconds: 1800,
  experimental: true,
} as const;

// Size limits. A 512 x 512 PNG is at most about 1 MB raw, and base64 makes text
// about a third larger, so 2 MB per image and 4 MB for the whole JSON body leave
// room for any honest upload while refusing anything far bigger. Caddy also
// caps request bodies (12 MB), and the worker caps its own input at 4 MiB.
const MAX_UPLOAD_BYTES = 4 * 1024 * 1024; // whole JSON request body
const MAX_PNG_BASE64_CHARS = 2 * 1024 * 1024; // each of image_png and mask_png
// Give up on the worker after 15 seconds, so a stuck worker cannot keep the
// page waiting. Every call is short (start, check or fetch a job); the slow
// part, the 200-step optimisation, runs in the background on the worker.
const WORKER_TIMEOUT_MS = 15_000;
// The cookie lasts no longer than the worker can keep the job (30 minutes).
const JOB_COOKIE = "facade_job";
const JOB_COOKIE_SECONDS = 1800;

// Exact shapes for every value that crosses the gateway. Anything that does not
// match is treated as invalid, never "cleaned up" and used anyway.
//   UUID       a job id, as the worker generates it
//   JOB_TOKEN  the worker's per-job secret: 64 lowercase hex characters
//   BEARER     the worker's bearer token from the environment
//   PNG_BASE64 base64 text whose first bytes decode to the PNG file signature
//   JOB_STATES every status the worker is allowed to report
const UUID = /^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$/;
const JOB_TOKEN = /^[a-f0-9]{64}$/;
const BEARER = /^[\x21-\x7e]{32,256}$/; // printable ASCII, no spaces
const PNG_BASE64 = /^iVBORw0KGgo[A-Za-z0-9+/]*={0,2}$/; // base64 that starts with the PNG signature
const JOB_STATES = new Set(["queued", "running", "completed", "cancelling", "cancelled", "failed"]);

// The only job fields the page ever receives. The worker's reply also contains
// the job token and clean-up details, which the page must not see.
export type JobStatus = { id: string; status: string; iteration: number; total_iterations: number };

/**
 * JSON response that is never cached. "no-store" stops browsers and proxies
 * from keeping a copy of a job's status; "nosniff" stops a browser from
 * guessing a different content type for the reply.
 */
export function json(data: unknown, status = 200, headers: Record<string, string> = {}) {
  return Response.json(data, {
    status,
    headers: { "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff", ...headers },
  });
}

/** True for this computer's own address, where plain http:// is safe enough. */
function isLoopback(hostname: string) {
  return hostname === "127.0.0.1" || hostname === "localhost";
}

/**
 * Worker URL and token, or null when they are missing or unsafe.
 *
 * Refused: a token of the wrong shape; a URL that is not https:// (except on
 * this computer); and a URL with a user name, password, query string or
 * #fragment, any of which could change where the request really goes.
 */
function workerSettings(): { url: string; token: string } | null {
  const address = process.env.FACADE_PROCESSOR_URL;
  const token = process.env.FACADE_PROCESSOR_TOKEN;
  if (!address || !token || !BEARER.test(token)) return null;
  try {
    const url = new URL(address);
    const schemeOk = url.protocol === "https:" || (url.protocol === "http:" && isLoopback(url.hostname));
    if (!schemeOk || url.username || url.password || url.search || url.hash) return null;
    // Drop a trailing slash so "/v1/..." can be appended safely.
    return { url: url.href.replace(/\/$/, ""), token };
  } catch {
    return null;
  }
}

/**
 * The origin the browser uses for this site. In development (`npm run dev`)
 * it falls back to the request's own origin. In production it must be set;
 * without it, uploads and deletes are refused (fail closed).
 *
 * The configured value must be a bare origin: scheme, host and optional port,
 * with no path, query, fragment or credentials, e.g. https://facadephotoshield.com.
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

/**
 * True only for requests sent by this site's own pages. Used for POST and DELETE.
 *
 * The Origin check. Browsers add an Origin header to every POST and DELETE,
 * naming the site whose page sent it, and page scripts cannot forge it. If a
 * different website tried to make a visitor's browser upload a photo or delete
 * a job here (cross-site request forgery), its Origin would not match and the
 * request is refused. Sec-Fetch-Site is a second, newer browser signal that
 * says the same thing; "cross-site" is refused too. A missing Origin is refused.
 */
export function sameOrigin(request: Request) {
  const origin = publicOrigin(request);
  return (
    origin !== null &&
    request.headers.get("origin") === origin.origin &&
    request.headers.get("sec-fetch-site") !== "cross-site"
  );
}

/**
 * Calls the worker with the bearer token (and the job's own token, if given).
 *
 * This is the only place the bearer token is used. It goes into the
 * Authorization header of a server-to-server request and is never copied into
 * anything sent back to the browser. The job token (X-Job-Token) proves that
 * this browser owns the job it asks about.
 *
 * redirect: "manual" means a redirect from the worker is not followed, so the
 * token can never be forwarded to a different address.
 */
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

/**
 * Removes URLs and long hex strings (tokens) from an error message before
 * logging it, and keeps at most 180 characters. Server logs should help find a
 * fault without ever holding the worker address or a secret.
 */
function redact(message: string) {
  return message
    .replace(/https?:\/\/[^\s]+/g, "[endpoint]")
    .replace(/[a-f0-9]{32,}/gi, "[redacted]")
    .slice(0, 180);
}

/**
 * Asks the worker what it can do. `ready` is true only when the worker reports
 * the exact expected recipe and a passed GPU smoke test.
 *
 * The "smoke test" is a full 200-step run the worker does on itself when it
 * starts; it proves the GPU really works before any visitor's photo arrives.
 * The page calls this on load (to enable or disable its button), and the jobs
 * route calls it again right before every upload.
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
    // Every field must match exactly (===), so a worker running a different or
    // weaker recipe can never be mistaken for the expected one.
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
    // Report our own expected values, not the worker's, so nothing unexpected
    // from the worker reaches the page.
    return { ready, ...EXPECTED };
  } catch (error) {
    console.warn("FACADE capabilities request failed", {
      kind: error instanceof Error ? error.name : "unknown",
      detail: error instanceof Error ? redact(error.message) : "unknown",
    });
    return { ready: false };
  }
}

/**
 * Reads this browser's job id and job token from the facade_job cookie.
 *
 * The cookie value is "<job id>.<job token>". Both halves must have exactly
 * the expected shape, so a tampered cookie can never smuggle a path such as
 * "../" into the worker URL built from the id. Returns null when there is no
 * valid job.
 */
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
 *
 * Why a cookie: the job token is a secret that lets whoever holds it see or
 * download the result. HttpOnly means the browser sends it automatically but
 * no JavaScript on the page can read it, so even a bug or injected script on
 * the page could not steal it. SameSite=Strict means the browser sends it only
 * on requests that start from this site. Path=/api/processing means it is sent
 * only to the three API routes, not with every page and image.
 *
 * Secure (send only over HTTPS) is added for the real site. It is left off only
 * for a local http:// development address, where the browser would otherwise
 * refuse to store the cookie at all.
 */
export function jobCookie(request: Request, value: string, age = JOB_COOKIE_SECONDS) {
  const origin = publicOrigin(request);
  const secure = origin === null || origin.protocol === "https:";
  return `${JOB_COOKIE}=${value}; HttpOnly; SameSite=Strict; Path=/api/processing; Max-Age=${age}${secure ? "; Secure" : ""}`;
}

/**
 * Keeps only the four job fields the page needs, and checks each one.
 * Throws if the worker's reply has an unknown status, a step count outside
 * 0-200, or a different total, so a broken worker reply never reaches the page.
 * Everything else in the reply (notably the job token) is dropped here.
 */
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

/** True when a value is a well-formed job token (64 lowercase hex characters). */
export function validToken(value: unknown): value is string {
  return typeof value === "string" && JOB_TOKEN.test(value);
}

/**
 * Reads and checks the upload: JSON with exactly {recipe, image_png, mask_png},
 * both PNGs base64-encoded, the whole body at most 4 MB. Throws otherwise.
 *
 * The checks, in order: the Content-Type must be JSON; the body may not grow
 * past MAX_UPLOAD_BYTES; it must parse as JSON with exactly the three expected
 * keys (no extra fields, so no filename or other data can ride along); the
 * recipe must be ours; and each image must be base64 text that starts with the
 * PNG signature and is at most MAX_PNG_BASE64_CHARS long. The worker decodes
 * and checks the images fully (size, mask colours); this gateway only makes
 * sure nothing obviously wrong or oversized is forwarded.
 */
export async function readUpload(request: Request) {
  if (request.headers.get("content-type")?.split(";")[0] !== "application/json") throw new Error("invalid upload");
  const reader = request.body?.getReader();
  if (!reader) throw new Error("invalid upload");

  // Read the body in chunks so an oversized upload is cut off early. Reading it
  // all at once (request.json()) would first load the whole body into memory,
  // however large it is.
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
  // Join the chunks into one byte array, then decode it as UTF-8 text.
  const bytes = new Uint8Array(size);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.length;
  }

  // JSON.parse throws on malformed input; the calling route turns any error
  // into one plain "invalid upload" message.
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
