"use client";

import Image from "next/image";
import Link from "next/link";
import { useState } from "react";
import { ArrowUpRight, Cloud, Smartphone, WandSparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { IntroVideo } from "@/components/intro-video";
import { ProtectionWorkspace } from "@/components/protection-workspace";
import { ResearchMetrics } from "@/components/research-metrics";
import { OPENRAIL_LICENSE_URL } from "@/lib/protection-recipe";

// The full PhotoGuard demo (server/photoguard_demo) is a separate Gradio app.
// On the server, Caddy sends /photoguard/ to it; it is not part of this Next.js app.
// ?__theme=light keeps Gradio in light mode to match this site.
const PHOTOGUARD_DEMO_URL = "/photoguard/?__theme=light";

// Public TestFlight link for the iPhone beta (FACADE Photo Shield on App Store Connect).
const TESTFLIGHT_URL = "https://testflight.apple.com/join/dxMRcMkd";

export default function Home() {
  const [section, setSection] = useState("prepare");

  return (
    <Tabs value={section} onValueChange={setSection} className="site-shell">
      <header className="topbar">
        <Link href="/" className="brand" aria-label="FACADE home">
          <span className="brand-mark">F</span>FACADE
        </Link>
        <TabsList variant="line" aria-label="Main navigation" className="site-nav">
          <TabsTrigger value="prepare">Prepare a photo</TabsTrigger>
          <TabsTrigger value="help">Get help</TabsTrigger>
          <TabsTrigger value="research">Our approach</TabsTrigger>
        </TabsList>
        <span className="research-tag">Research preview</span>
      </header>

      <main className="main-content">
        <TabsContent value="prepare">
          <div className="page-heading">
            <div>
              <p className="eyebrow">BEFORE YOU SHARE</p>
              <h1>
                A little more control
                <br />
                over your photos.
              </h1>
            </div>
            <p className="heading-note">
              Explore experimental protection against AI edits. Prepare a crop in your browser, then choose online processing only when
              the service is available.
            </p>
          </div>

          <IntroVideo />

          <ProtectionWorkspace />

          <PhotoGuardDemoCard />

          <GetTheAppCard />

          <div className="support-strip">
            <div>
              <strong>Has someone already misused an intimate image?</strong>
              <p>Find removal and reporting support. You do not need to share the image with FACADE.</p>
            </div>
            <Button variant="outline" onClick={() => setSection("help")}>
              Find support <ArrowUpRight size={16} />
            </Button>
          </div>
        </TabsContent>

        <TabsContent value="help">
          <div className="text-heading">
            <p className="eyebrow">YOU HAVE OPTIONS</p>
            <h1>
              Help when an image
              <br />
              is misused.
            </h1>
            <p>These independent services may help with removal, reporting, or threats. FACADE does not collect abuse imagery.</p>
          </div>
          <div className="resource-grid">
            <Resource
              title="Take It Down"
              audience="Under 18 when the intimate image was taken"
              text="NCMEC’s service creates a digital fingerprint on your device. The image stays there. You can use it even if you are an adult now."
              url="https://takeitdown.ncmec.org/"
            />
            <Resource
              title="Find your rep"
              audience="Your voice in Congress"
              text="Find your U.S. House representative and official contact information through the House of Representatives lookup."
              url="https://www.house.gov/representatives/find-your-representative"
            />
            <Resource
              title="StopNCII.org"
              audience="Adults depicted in intimate images"
              text="For eligible images, including deepfakes. A digital fingerprint helps participating platforms identify matching material without uploading the image to this service."
              url="https://stopncii.org/"
            />
            <Resource
              title="NCMEC CyberTipline"
              audience="Threats or suspected child sexual exploitation"
              text="Report suspected online child sexual exploitation and seek help even if you do not have the image."
              url="https://report.cybertip.org/"
            />
          </div>
          <p className="resource-footnote">
            Do not download, forward, or request an intimate image just to use these services. They work with participating platforms
            and cannot guarantee removal everywhere. External websites have their own privacy policies.
          </p>
        </TabsContent>

        <TabsContent value="research">
          <div className="text-heading">
            <p className="eyebrow">PROTECTION, WITH CONTEXT</p>
            <h1>
              Two ways to explore
              <br />
              the same problem.
            </h1>
            <p>
              FACADE studies changes to an image that may make AI manipulation harder, while keeping the original useful for sharing.
            </p>
          </div>
          <div className="research-layout">
            <section className="explanation-card">
              <Smartphone size={28} />
              <h2>FACADE on your phone</h2>
              <p>
                The FACADE iOS app is our main method for faces. It processes photos on the device, so they never leave the phone. The
                app is in testing.
              </p>
            </section>
            <section className="explanation-card">
              <Cloud size={28} />
              <h2>PhotoGuard encoder online</h2>
              <p>
                This website sends a 512&nbsp;×&nbsp;512 crop to the team’s private GPU server. The server runs the encoder method from
                PhotoGuard’s official demo for 200 optimization steps, then returns a PNG. This is distinct from PhotoGuard’s heavier
                diffusion attack.
              </p>
            </section>
          </div>
          <ResearchMetrics />
          <div className="research-disclosure">
            <h2>What we can honestly say</h2>
            <p>
              Numerical and runtime checks show whether an implementation runs correctly. They do not show that a photo is protected
              against a particular editor. The face-swap measurements are reported above. Clothing-edit protection and automatic
              recommendations still need testing.
            </p>
            <p>
              PhotoGuard is research by Hadi Salman, Alaa Khaddaj, Guillaume Leclerc, Andrew Ilyas, and Aleksander Madry. FACADE is not
              affiliated with its authors.
            </p>
            <a href="https://github.com/MadryLab/photoguard" target="_blank" rel="noreferrer">
              Read the PhotoGuard research <ArrowUpRight size={16} />
            </a>
            <a href={PHOTOGUARD_DEMO_URL}>
              Try the full PhotoGuard demo <ArrowUpRight size={16} />
            </a>
          </div>
        </TabsContent>
      </main>

      <footer>
        <span className="footer-brand">FACADE</span>
        <span>Your photos. Informed choices.</span>
        <span>Experimental research · No guarantee of protection</span>
      </footer>
    </Tabs>
  );
}

function PhotoGuardDemoCard() {
  return (
    <section className="demo-card" aria-labelledby="demo-heading">
      <div className="demo-icon">
        <WandSparkles size={26} />
      </div>
      <div className="demo-text">
        <p className="eyebrow">SEE IT AGAINST A REAL EDIT</p>
        <h2 id="demo-heading">Full PhotoGuard demo</h2>
        <p>
          Upload an image, paint over the part to keep (such as a face), and describe an edit. The demo immunizes the image, then runs
          the same AI edit on the original and on the protected copy, so you can compare the two results side by side. A run takes
          about a minute.
        </p>
        <p className="demo-note">
          In the demo you paint what to keep; on this page you mark what could be replaced. Only upload photos of yourself, or of
          people who have agreed to it. The demo runs Stable Diffusion 1.5; by using it you agree to follow the model&apos;s{" "}
          <a href={OPENRAIL_LICENSE_URL} target="_blank" rel="noreferrer">
            use restrictions (CreativeML OpenRAIL-M)
          </a>
          .
        </p>
      </div>
      <a className="demo-link" href={PHOTOGUARD_DEMO_URL}>
        Open the PhotoGuard demo <ArrowUpRight size={16} />
      </a>
    </section>
  );
}

function GetTheAppCard() {
  return (
    <section className="demo-card" aria-labelledby="app-heading">
      <div className="demo-icon">
        <Smartphone size={26} />
      </div>
      <div className="demo-text">
        <p className="eyebrow">ON YOUR IPHONE</p>
        <h2 id="app-heading">Get the FACADE app (beta)</h2>
        <p>
          The FACADE iPhone app protects photos entirely on your phone; nothing is uploaded. It is a beta built for the 2026
          Congressional App Challenge. First install Apple&apos;s free TestFlight app from the App Store, then open the link below on
          your iPhone or scan the code with its camera.
        </p>
        <p className="demo-note">
          Protection is experimental. See the{" "}
          <a href="/privacy" target="_blank" rel="noreferrer">
            privacy policy
          </a>
          .
        </p>
        <a className="demo-link app-link" href={TESTFLIGHT_URL} target="_blank" rel="noreferrer">
          Join the TestFlight beta <ArrowUpRight size={16} />
        </a>
      </div>
      <Image className="app-qr" src="/testflight-qr.svg" alt="QR code for the FACADE TestFlight beta link" width={132} height={132} unoptimized />
    </section>
  );
}

function Resource({ title, audience, text, url }: { title: string; audience: string; text: string; url: string }) {
  return (
    <article className="resource-card">
      <p className="eyebrow">{audience}</p>
      <h2>{title}</h2>
      <p>{text}</p>
      <a href={url} target="_blank" rel="noreferrer">
        Open {title}
        <ArrowUpRight size={16} />
      </a>
    </article>
  );
}
