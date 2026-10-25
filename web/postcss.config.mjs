// PostCSS settings. Next.js runs PostCSS on app/globals.css at build time; the
// one plugin is Tailwind CSS v4, which turns the utility classes used in the
// components (e.g. "rounded-md") into real CSS.
const config = {
  plugins: {
    "@tailwindcss/postcss": {},
  },
};

export default config;
