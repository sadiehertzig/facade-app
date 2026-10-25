// The page shell shared by every page of the site: the <html> and <body>
// elements, the global stylesheet, and the page metadata (title, description,
// icon) that browsers and link previews show.
import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "FACADE — Photo protection research",
  description: "Explore experimental photo protection and find support when an image is misused.",
  icons: {
    icon: "/favicon.svg",
    shortcut: "/favicon.svg",
  },
  // Ask search engines not to list the site: it is a password-protected
  // research preview, not a public product page.
  robots: { index: false, follow: false },
};

/** Wraps every page. `children` is the page itself (app/page.tsx). */
export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
