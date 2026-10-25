// Next.js settings for the FACADE website.
//
// Two things are set here: the "X-Powered-By: Next.js" header is switched off,
// and a list of security headers is added to every page and file the app
// serves. Caddy (deploy/Caddyfile) adds HTTPS in front.
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Don't advertise the framework in every response.
  poweredByHeader: false,
  // Headers added to every response ("/:path*" matches every path):
  //   X-Content-Type-Options: nosniff   the browser must trust the declared
  //                                     file type and never guess another one
  //   Referrer-Policy: same-origin      when a visitor follows a link to
  //                                     another site, it is not told which
  //                                     FACADE page they came from
  //   X-Frame-Options: DENY             no other site may show FACADE inside a
  //                                     frame (prevents "clickjacking", where a
  //                                     hidden page tricks a visitor into clicks)
  //   Permissions-Policy                the site never asks for the camera,
  //                                     microphone or location
  async headers() {
    return [
      {
        source: "/:path*",
        headers: [
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "Referrer-Policy", value: "same-origin" },
          { key: "X-Frame-Options", value: "DENY" },
          { key: "Permissions-Policy", value: "camera=(), microphone=(), geolocation=()" },
        ],
      },
    ];
  },
};

export default nextConfig;
