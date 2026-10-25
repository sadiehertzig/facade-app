"use client";
// The FACADE website's one page. It has three tabs (Home, Get help, Our
// approach) that switch in place, without loading a new page.
//
// WHAT IS ON IT
//   Home         the iPhone app and how to get it, the demo video, a worked
//                face-swap example, how the app works, the PhotoGuard demo
//                on the team's server, and who made FACADE
//   Get help     independent services for people whose images were misused
//   Our approach the two methods, the measured face-swap results
//                (components/research-metrics.tsx) and their limits
//
// "use client" (the line above) makes this a client component: it runs in the
// visitor's browser, which it must, because the selected tab is state that
// changes when the visitor clicks. No photo is uploaded from this page. Photos
// are protected in the iPhone app, or in the full PhotoGuard demo at
// /photoguard/, a separate app on the team's server. The in-site upload
// workspace (components/protection-workspace.tsx) is not shown on this page.
//
// The text on this page is careful by design: protection is described as
// experimental, and every number shown comes from a measurement.

import Image from "next/image";
import Link from "next/link";
import { useState } from "react";
import {
  ArrowUpRight,
  CloudOff,
  FlaskConical,
  ImagePlus,
  Images,
  Play,
  ScanFace,
  ShieldHalf,
  Smartphone,
  WandSparkles,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { IntroVideo } from "@/components/intro-video";
import { ResearchMetrics } from "@/components/research-metrics";
import { OPENRAIL_LICENSE_URL } from "@/lib/protection-recipe";

// The full PhotoGuard demo (server/photoguard_demo) is a separate Gradio app.
// On the server, Caddy sends /photoguard/ to it; it is not part of this Next.js app.
// ?__theme=light keeps Gradio in light mode to match this site.
const PHOTOGUARD_DEMO_URL = "/photoguard/?__theme=light";

// Public TestFlight link for the iPhone beta (FACADE Photo Shield on App Store Connect).
const TESTFLIGHT_URL = "https://testflight.apple.com/join/dxMRcMkd";
const TESTFLIGHT_APP_URL = "https://apps.apple.com/us/app/testflight/id899247664";

/**
 * The whole page. `section` is the selected tab; buttons elsewhere on the page
 * (e.g. "Find support") switch tabs by calling setSection.
 */
export default function Home() {
  const [section, setSection] = useState("home");

  return (
    <Tabs value={section} onValueChange={setSection} className="site-shell">
      {/* Top bar: the FACADE name and the three tab buttons. */}
      <header className="topbar">
        <Link href="/" className="brand" aria-label="FACADE home">
          <span className="brand-mark">F</span>FACADE
        </Link>
        <TabsList variant="line" aria-label="Main navigation" className="site-nav">
          <TabsTrigger value="home">Home</TabsTrigger>
          <TabsTrigger value="help">Get help</TabsTrigger>
          <TabsTrigger value="research">Our approach</TabsTrigger>
        </TabsList>
        <span className="research-tag">Research preview</span>
      </header>

      <main className="main-content">
        {/* Tab 1, Home: the app, the video, the examples and the team. */}
        <TabsContent value="home">
          <Hero />

          <IntroVideo />

          <FaceSwapExample />

          <HowItWorks />

          <PhotoGuardSection />

          <TeamSection />

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

        {/* Tab 2, Get help: independent support services (FACADE collects nothing here). */}
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

        {/* Tab 3, Our approach: the two methods, the measured results and their limits. */}
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
              <WandSparkles size={28} />
              <h2>Full PhotoGuard on our server</h2>
              <p>
                For AI edits beyond faces, the team’s GPU server runs PhotoGuard’s official demo. It protects a 512&nbsp;×&nbsp;512 crop
                with the encoder method (200 optimization steps), then runs the same Stable Diffusion 1.5 inpainting edit on the original
                and on the protected copy. This is distinct from PhotoGuard’s heavier diffusion attack.
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
        <span>Your photo. Your choice.</span>
        <span>
          <a href={SOURCE_URL} target="_blank" rel="noreferrer">
            Source code on GitHub
          </a>
        </span>
        <span>Experimental research · No guarantee of protection</span>
      </footer>
    </Tabs>
  );
}

// The catch phrase and the app, first. The phrase matches the app's own header.
// The button and the QR code both lead to the public TestFlight beta; the QR
// code is a static image of TESTFLIGHT_URL (public/testflight-qr.svg).
function Hero() {
  return (
    <section className="hero" aria-labelledby="hero-heading">
      <div className="hero-text">
        <p className="hero-eyebrow">FACADE FOR IPHONE · FREE BETA</p>
        <h1 id="hero-heading">
          Your photo.
          <br />
          Your choice.
        </h1>
        <p className="hero-lede">
          FACADE adds a layer of protection to your photos before you share them, to make AI face swaps harder. It runs entirely on
          your iPhone, so your photos are never uploaded.
        </p>
        <ul className="hero-chips" aria-label="Key facts">
          <li>
            <CloudOff size={16} aria-hidden /> No upload
          </li>
          <li>
            <Images size={16} aria-hidden /> Original kept
          </li>
          <li>
            <FlaskConical size={16} aria-hidden /> Experimental
          </li>
        </ul>
        <div className="hero-actions">
          <a className="hero-button" href={TESTFLIGHT_URL} target="_blank" rel="noreferrer">
            Get the iPhone app <ArrowUpRight size={16} />
          </a>
          <a className="hero-button hero-button-quiet" href="#video">
            <Play size={16} aria-hidden /> Watch the video
          </a>
        </div>
      </div>
      <div className="hero-qr">
        <Image src="/testflight-qr.svg" alt="QR code for the FACADE TestFlight beta link" width={148} height={148} unoptimized />
        <p>
          Scan with your iPhone camera. Install Apple&apos;s free{" "}
          <a href={TESTFLIGHT_APP_URL} target="_blank" rel="noreferrer">
            TestFlight
          </a>{" "}
          app first.
        </p>
      </div>
    </section>
  );
}

// The face-swap example from the contest video. The swaps are 128 × 128 because
// that is the size the swap model makes. Keep the "hand-picked" label next to it.
function FaceSwapExample() {
  return (
    <section className="example" aria-labelledby="example-heading">
      <div className="section-heading">
        <p className="eyebrow">WHAT FACADE DOES</p>
        <h2 id="example-heading">Same photo to you. A different face to a face-swap tool.</h2>
        <p>
          FACADE changes each color value by at most 5 out of 255, pushing the face&apos;s identity code away from you. Then we ask a
          face-swap tool to put the face on someone else.
        </p>
      </div>
      <div className="example-grid">
        <figure className="example-group">
          <figcaption>1. FACADE protects the photo</figcaption>
          <div className="example-tiles">
            <ExampleTile src="/example/your-photo.jpg" label="Your photo" />
            <ExampleTile src="/example/protected.jpg" label="Protected by FACADE" note="Looks the same" />
          </div>
        </figure>
        <figure className="example-group">
          <figcaption>2. A face-swap tool puts that face on someone else</figcaption>
          <div className="example-tiles">
            <ExampleTile src="/example/swap-target.jpg" label="Someone else" />
            <ExampleTile src="/example/fake-from-original.png" label="Fake made from your photo" note="Looks like her" pixel />
            <ExampleTile src="/example/fake-from-protected.png" label="Fake made from the protected photo" note="Doesn’t look like her" pixel />
          </div>
        </figure>
      </div>
      <p className="example-note">
        One hand-picked example, with AI-generated fictional people. In 192 test swaps, 92% of SimSwap fakes stopped matching the
        person, but only 59% for InSwapper. Protection is experimental and never guaranteed.
      </p>
    </section>
  );
}

/**
 * One captioned image in an example row. `note` is a short badge such as
 * "Looks the same". `pixel` shows a small image enlarged with sharp pixels
 * instead of blurring it, for the 128 x 128 face swaps, so visitors see what the
 * swap model really produced. The alt text repeats the label and the note for
 * screen readers.
 */
function ExampleTile({ src, label, note, pixel = false }: { src: string; label: string; note?: string; pixel?: boolean }) {
  return (
    <div className="example-tile">
      <Image src={src} alt={`${label}${note ? `: ${note.toLowerCase()}` : ""}`} width={320} height={320} unoptimized className={pixel ? "example-small" : undefined} />
      <span className="example-label">{label}</span>
      {note ? <span className="example-badge">{note}</span> : null}
    </div>
  );
}

/** "How the app works": three numbered steps, each with an icon. */
function HowItWorks() {
  const steps = [
    { icon: ImagePlus, title: "Pick a photo", text: "Choose one from your library or take a selfie. Use photos you have permission to use." },
    { icon: ShieldHalf, title: "Protect it", text: "FACADE changes the pixels slightly, on your phone. Our test photo took about a minute on an iPhone 17 Pro." },
    { icon: ScanFace, title: "Compare and test", text: "Check before and after, run the built-in face-swap test, and save the protected copy to share." },
  ];
  return (
    <section className="steps" aria-labelledby="steps-heading">
      <h2 id="steps-heading" className="steps-heading">
        How the app works
      </h2>
      <ol className="steps-list">
        {steps.map(({ icon: Icon, title, text }) => (
          <li key={title}>
            <span className="steps-icon">
              <Icon size={22} aria-hidden />
            </span>
            <h3>{title}</h3>
            <p>{text}</p>
          </li>
        ))}
      </ol>
    </section>
  );
}

// The one PhotoGuard entry point on the site: the full upstream demo on our GPU
// server (server/photoguard_demo), which protects a photo and then runs the same
// AI edit on the original and on the protected copy.
//
// The example is one run of that demo's code (PhotoGuardDemo.compare, pinned SD 1.5
// inpainting snapshot, seed 1234, 100 steps, guidance 7.5, 200 PGD steps) on the
// app's sample photo, with the face and hair kept. It ran on a Mac GPU, not the server.
function PhotoGuardSection() {
  return (
    <section className="example" aria-labelledby="demo-heading">
      <div className="section-heading">
        <p className="eyebrow">AI EDITS BEYOND FACES</p>
        <h2 id="demo-heading">Full PhotoGuard, on our server</h2>
        <p>
          PhotoGuard is MIT research that protects a photo against AI editors like Stable Diffusion, which can repaint part of a
          picture, such as clothing or the background. Our GPU server runs it: you upload a photo, paint the part to keep and describe
          an edit. It protects the photo, then runs the same edit on the original and on the protected copy so you can compare them.
          A run takes about a minute.
        </p>
      </div>
      <div className="example-tiles pg-tiles">
        <ExampleTile src="/example/pg-original.jpg" label="The photo" />
        <ExampleTile src="/example/pg-protected.jpg" label="Protected by PhotoGuard" note="Slight grain" />
        <ExampleTile src="/example/pg-edit-original.jpg" label="AI edit of the photo" note="Believable" />
        <ExampleTile src="/example/pg-edit-protected.jpg" label="Same AI edit of the protected copy" note="Falls apart" />
      </div>
      <p className="example-note">
        One run with the app&apos;s AI-generated sample photo and the demo&apos;s example prompt, &ldquo;a photo of a person at a
        wedding.&rdquo; We kept her face and hair and let the AI repaint the rest. Same code, model and settings as our server, run on a
        Mac. Results vary with the photo and the prompt.
      </p>
      <div className="pg-actions">
        <a className="demo-link" href={PHOTOGUARD_DEMO_URL}>
          Open full PhotoGuard <ArrowUpRight size={16} />
        </a>
        <p className="demo-note">
          The live demo needs a password, because it runs real AI edits on uploaded photos. Contest judges can ask us for access. Only
          upload photos of yourself, or of people who have agreed to it. The demo runs Stable Diffusion 1.5; by using it you agree to
          follow the model&apos;s{" "}
          <a href={OPENRAIL_LICENSE_URL} target="_blank" rel="noreferrer">
            use restrictions (CreativeML OpenRAIL-M)
          </a>
          .
        </p>
      </div>
    </section>
  );
}

// Where the code lives. The repository is public (MIT licence); the model
// files it needs are kept apart, in a private repository, because of their
// research-only licences (see app/ios/Models/README.md there).
const SOURCE_URL = "https://github.com/sadiehertzig/facade-app";

// Name, school and hometown, exactly as on the contest video's team tiles: no
// photos or contact details.
const TEAM = [
  { name: "Sadie Hertzig", school: "The Chapin School", hometown: "Hoboken, NJ" },
  { name: "Aria Sharma", school: "The Lawrenceville School", hometown: "Monroe Township, NJ" },
  { name: "Evelyn Yan", school: "The Chapin School", hometown: "Hoboken, NJ" },
  { name: "Catalina Zapata", school: "Staples High School", hometown: "Westport, CT" },
];

/** "Who we are": the team and where FACADE started. */
function TeamSection() {
  return (
    <section className="team" aria-labelledby="team-heading">
      <div className="section-heading">
        <p className="eyebrow">WHO WE ARE</p>
        <h2 id="team-heading">Made by four students</h2>
        <p>
          FACADE was born at Cornell Tech&apos;s Summer Innovation Intensives on Roosevelt Island. It is our entry in the 2026
          Congressional App Challenge.
        </p>
      </div>
      <ul className="team-list">
        {TEAM.map(({ name, school, hometown }) => (
          <li key={name}>
            <strong>{name}</strong>
            <span>
              {school} · {hometown}
            </span>
          </li>
        ))}
      </ul>
      <p className="team-source">
        Source code:{" "}
        <a href={SOURCE_URL} target="_blank" rel="noreferrer">
          github.com/sadiehertzig/facade-app
        </a>{" "}
        (MIT licence; the model files are not included).
      </p>
    </section>
  );
}

/**
 * One support service card on the "Get help" tab. Links open in a new tab;
 * rel="noreferrer" stops the other site from learning which page sent the
 * visitor (and from controlling this tab).
 */
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
