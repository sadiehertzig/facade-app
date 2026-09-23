import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "FACADE — Photo protection research",
  description: "Explore experimental photo protection and find support when an image is misused.",
  icons: {
    icon: "/favicon.svg",
    shortcut: "/favicon.svg",
  },
  robots: { index: false, follow: false },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
