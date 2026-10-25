import Image from "next/image";

// The contest demo video, near the top of the Home tab. The file is served
// from web/public/video by this app (Next.js answers byte-range requests, which
// Safari needs to play and seek). The same video is on Vimeo for the
// Congressional App Challenge form, which requires a public Vimeo or YouTube link.
//
// If VIDEO_SRC is set to null, the cover image shows instead, with a "coming
// soon" note. To publish a new cut: put facade-demo.mp4 (and the captions as
// WebVTT) in public/video and set the two paths below.
const VIDEO_SRC: string | null = "/video/facade-demo.mp4";
const CAPTIONS_SRC: string | null = "/video/facade-demo.en.vtt";
const POSTER_SRC = "/video/facade-demo-poster.jpg";
const POSTER_ALT =
  "FACADE demo video cover: “Add a layer of protection before you share,” beside an AI-generated fictional adult whose face is half pixelated.";

/**
 * The video player, or the cover image when there is no video yet.
 *
 * preload="none" downloads nothing until the visitor presses play, so the page
 * stays fast. The poster is the still image shown before playback. playsInline
 * keeps it in the page on iPhones instead of jumping to full screen. The
 * captions track gives deaf and hard-of-hearing visitors the spoken words.
 * controlsList="nodownload" and the blocked right-click menu only hide the
 * browser's download shortcuts; they are a courtesy, not real copy protection.
 */
export function IntroVideo() {
  return (
    <section className="intro-video" id="video" aria-labelledby="intro-video-heading">
      <div className="intro-video-frame">
        {VIDEO_SRC ? (
          <video
            controls
            playsInline
            preload="none"
            poster={POSTER_SRC}
            controlsList="nodownload"
            onContextMenu={(event) => event.preventDefault()}
            aria-label="FACADE demo video"
          >
            <source src={VIDEO_SRC} type="video/mp4" />
            {CAPTIONS_SRC ? <track kind="captions" src={CAPTIONS_SRC} srcLang="en" label="English" /> : null}
          </video>
        ) : (
          <>
            <Image src={POSTER_SRC} alt={POSTER_ALT} width={1280} height={720} priority unoptimized />
            <span className="intro-video-soon">Video coming soon</span>
          </>
        )}
      </div>
      <div className="intro-video-text">
        <p className="eyebrow">CONGRESSIONAL APP CHALLENGE 2026</p>
        <h2 id="intro-video-heading">See FACADE in three minutes</h2>
        <p>
          What FACADE does to a photo, how we tested it against face-swap tools, and what it cannot do yet. The demo uses an
          AI-generated fictional person, not a real one.
        </p>
      </div>
    </section>
  );
}
