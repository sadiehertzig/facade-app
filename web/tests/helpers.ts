// Shared helpers for the gateway tests. No test ever reaches the network:
// global fetch is replaced by a fake worker, and anything unexpected throws.
//
// The constants are made-up values with the right shapes (a 32-character
// bearer, a UUID job id, a 64-hex job token). SITE and WORKER use the reserved
// .example domain, which can never belong to a real server.

import { vi } from "vitest";

export const SITE = "https://facade.example";
export const WORKER = "https://worker.example";
export const BEARER = "s".repeat(32);
export const RECIPE = "photoguard-encoder-linf-v1";
export const JOB_ID = "11111111-2222-4333-8444-555555555555";
export const JOB_TOKEN = "a".repeat(64);
export const JOB_COOKIE = `facade_job=${JOB_ID}.${JOB_TOKEN}`;

/** What a healthy worker reports on GET /v1/capabilities. */
export const VALID_CAPABILITIES = {
  ready: true,
  gpu_smoke_validated: true,
  recipe: RECIPE,
  image_side: 512,
  total_iterations: 200,
  retention_seconds: 1800,
  experimental: true,
};

/** A tiny body that passes the gateway's PNG checks (it starts with the PNG signature in base64). */
export const PNG_BASE64 = "iVBORw0KGgoAAAAA";

/**
 * Sets the environment variables a correctly configured production server
 * would have. A test passes `overrides` to break one setting on purpose.
 * vitest.config.ts undoes these after every test.
 */
export function setProductionEnv(overrides: Record<string, string | undefined> = {}) {
  vi.stubEnv("NODE_ENV", "production");
  vi.stubEnv("FACADE_PROCESSOR_URL", WORKER);
  vi.stubEnv("FACADE_PROCESSOR_TOKEN", BEARER);
  vi.stubEnv("FACADE_PUBLIC_ORIGIN", SITE);
  for (const [key, value] of Object.entries(overrides)) vi.stubEnv(key, value);
}

type Handler = (url: string, init: RequestInit) => Response | Promise<Response>;

/** Replaces global fetch. Every call is recorded and must use the worker bearer. */
export function fakeWorker(handler: Handler) {
  const calls: { url: string; init: RequestInit }[] = [];
  const fetchMock = vi.fn(async (input: string | URL | Request, init: RequestInit = {}) => {
    const url = String(input);
    calls.push({ url, init });
    return handler(url, init);
  });
  vi.stubGlobal("fetch", fetchMock);
  return calls;
}

/** fetch that fails the test if called. */
export function noNetwork() {
  return fakeWorker(() => {
    throw new Error("Network forbidden in this test");
  });
}

/**
 * A request as a browser on the public site would send it: with the site's
 * Origin header (unless `origin` is given, or null for none) and an optional
 * Cookie header.
 */
export function siteRequest(path: string, init: RequestInit & { cookie?: string; origin?: string | null } = {}) {
  const headers = new Headers(init.headers);
  if (init.origin !== null) headers.set("Origin", init.origin ?? SITE);
  if (init.cookie) headers.set("Cookie", init.cookie);
  // Behind Caddy the app sees its local listen address, not the public origin.
  return new Request(`http://127.0.0.1:3000${path}`, { ...init, headers });
}

/** A well-formed upload body; pass another recipe to make it invalid. */
export function uploadBody(recipe = RECIPE) {
  return JSON.stringify({ recipe, image_png: PNG_BASE64, mask_png: PNG_BASE64 });
}
