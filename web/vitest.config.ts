// Test runner settings (Vitest, run with `npm test`).
//
// The tests in tests/ exercise the server-side gateway and API routes in plain
// Node.js. No browser, worker or network is involved: tests/helpers.ts
// replaces fetch with a fake worker.
import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";

export default defineConfig({
  resolve: {
    // Same "@/..." import alias as tsconfig.json.
    alias: { "@": fileURLToPath(new URL(".", import.meta.url)) },
  },
  test: {
    // Run tests in plain Node.js, the same runtime the API routes use.
    environment: "node",
    include: ["tests/**/*.test.ts"],
    // Undo vi.stubEnv / vi.stubGlobal (fake fetch) after every test.
    unstubEnvs: true,
    unstubGlobals: true,
  },
});
