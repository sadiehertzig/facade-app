// ESLint settings (run with `npm run lint`).
//
// Uses Next.js's recommended rules: "core-web-vitals" catches React and
// Next.js mistakes that hurt page speed or correctness, and "typescript" adds
// TypeScript-aware rules. Build output and generated files are skipped.
import { defineConfig, globalIgnores } from "eslint/config";
import nextVitals from "eslint-config-next/core-web-vitals";
import nextTs from "eslint-config-next/typescript";

export default defineConfig([
  ...nextVitals,
  ...nextTs,
  globalIgnores([".next/**", "out/**", "coverage/**", "next-env.d.ts"]),
]);
