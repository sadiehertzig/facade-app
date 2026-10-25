"use client";

// The "Prepare a photo" workspace: choose a photo, crop it to 512 x 512 in the
// browser, mark the area an editor might replace, then (with consent) send the
// crop to the PhotoGuard worker through /api/processing.
//
// THE STEPS, AS THE VISITOR SEES THEM
//   1. Choose a JPEG or PNG. It is read and cropped here, in the browser;
//      nothing is uploaded yet.
//   2. Brush over the area an editor might replace (e.g. clothing).
//   3. Tick the consent box. Without it the upload button stays disabled.
//   4. Press "Apply PhotoGuard". Only the 512 x 512 crop and the mask are sent,
//      as freshly encoded PNGs, so no filename or photo metadata goes along.
//   5. Watch the progress (0-200 steps), download the result, clear the job.
//
// HOW IT WORKS
//   Two canvases sit on top of each other: the bottom one shows the cropped
//   photo, the top one is a transparent layer the visitor paints on in white.
//   On upload, the paint layer is turned into a pure black-and-white mask
//   (lib/protection-mask.ts explains which colour means what).
//   The page talks only to this site's own /api/processing routes. It never
//   knows the worker's address or token, and the job's id and secret token
//   live in an HttpOnly cookie that this code cannot read.
//
// Every button is disabled while a request is under way or a job exists
// ("locked"), so the visitor cannot change the photo halfway through.
//
// NOTE: the current home page (app/page.tsx) does not include this component;
// the site now points visitors to the iPhone app and the full PhotoGuard demo
// instead. The component and the /api/processing routes it uses are kept, and
// tested, so the workspace can be shown again by adding
// <ProtectionWorkspace /> to a tab.

import { useEffect, useRef, useState, type PointerEvent } from "react";
import { Cloud, Download, ImageIcon, Info, LockKeyhole, OctagonAlert, Smartphone, TriangleAlert, Upload, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Progress } from "@/components/ui/progress";
import { normalizeEditMask } from "@/lib/protection-mask";
import { OPENRAIL_LICENSE_URL, PHOTO_GUARD_RECIPE } from "@/lib/protection-recipe";

const SIDE = 512; // the worker's fixed input size
// Brush diameter, in pixels of the 512 x 512 crop.
const BRUSH = 36;
// Limits on the chosen file, checked before it is even decoded: 16 MB is more
// than a typical phone photo, and 40 million pixels (e.g. 8000 x 5000) keeps the browser
// from running out of memory while decoding it. Only the 512 x 512 crop is
// ever uploaded, so these limits never affect what the server receives.
const MAX_FILE_BYTES = 16 * 1024 * 1024;
const MAX_PIXELS = 40_000_000;
// How often to ask for a running job's progress: every 4 seconds.
const POLL_MS = 4000;

// What the API routes send back: a job's status, or an error message, or (for
// the capabilities route) whether the worker is ready.
type Job = { id: string; status: string; iteration: number; total_iterations: number };
type ApiReply = Partial<Job> & { error?: string; ready?: boolean };

/**
 * Parses an API reply as a JSON object, or throws. Every route answers in JSON
 * (except the PNG download), including errors, whose plain-language `error`
 * text is shown to the visitor as it is.
 */
async function readReply(response: Response): Promise<ApiReply> {
  const value: unknown = await response.json();
  if (!value || typeof value !== "object") throw new Error("The service returned an invalid response.");
  return value as ApiReply;
}

/** Asks the server whether the PhotoGuard worker is ready. Never throws. */
async function fetchReady(): Promise<boolean> {
  try {
    const response = await fetch("/api/processing/capabilities", { cache: "no-store" });
    const reply = await readReply(response);
    return response.ok && reply.ready === true;
  } catch {
    return false;
  }
}

/** The heading shown above the progress bar for each job status. */
function jobHeading(status: string) {
  switch (status) {
    case "completed":
      return "Experimental result ready";
    case "failed":
      return "Processing did not finish";
    case "cancelling":
      return "Cancelling and clearing data…";
    case "cancelled":
      return "Job cancelled";
    case "queued":
      return "Waiting for the processor";
    default:
      return "Processing your crop";
  }
}

/** The workspace: the photo panel on the left, the options panel on the right. */
export function ProtectionWorkspace() {
  // What the visitor has done so far, and what the server has told us.
  //   file      the chosen photo (still only on this computer)
  //   loaded    the crop has been drawn onto the photo canvas
  //   hasMask   something has been painted on the mask layer
  //   consent   the consent box is ticked
  //   ready     the worker is ready (from /api/processing/capabilities)
  //   checking  that check is still under way
  //   busy      an upload, delete or download request is under way
  //   job       the current online job's status, or null
  //   error     a message for the visitor, or "" for none
  const [file, setFile] = useState<File | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [hasMask, setHasMask] = useState(false);
  const [consent, setConsent] = useState(false);
  const [ready, setReady] = useState(false);
  const [checking, setChecking] = useState(true);
  const [busy, setBusy] = useState(false);
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState("");

  // References to page elements and drawing state that must survive between
  // pointer events without redrawing the component:
  //   input    the hidden file chooser
  //   photo    the bottom canvas with the 512 x 512 crop
  //   mask     the top canvas the visitor paints on
  //   drawing  true while the pointer is held down
  //   last     the previous brush point, to join points into a smooth stroke
  const input = useRef<HTMLInputElement>(null);
  const photo = useRef<HTMLCanvasElement>(null);
  const mask = useRef<HTMLCanvasElement>(null);
  const drawing = useRef(false);
  const last = useRef<{ x: number; y: number } | null>(null);

  // active: the job is still working, so keep polling.
  // locked: a job exists or a request is under way, so the photo, the mask and
  // the file chooser cannot be changed.
  const active = job !== null && ["queued", "running", "cancelling"].includes(job.status);
  const locked = job !== null || busy;

  /** Asks the server again whether the worker is ready ("Check availability"). */
  function checkAvailability() {
    setChecking(true);
    fetchReady()
      .then(setReady)
      .finally(() => setChecking(false));
  }

  // On load: check the worker, and pick up a job this browser already started.
  // After a page reload the job cookie is still there, so GET /api/processing/jobs
  // returns the job's status and the page shows its progress again.
  useEffect(() => {
    fetchReady()
      .then(setReady)
      .finally(() => setChecking(false));
    fetch("/api/processing/jobs", { cache: "no-store" })
      .then(async (response) => {
        if (response.ok) setJob((await readReply(response)) as Job);
      })
      .catch(() => {});
  }, []);

  // While a job is running, poll its progress. `disposed` makes sure a reply
  // that arrives after the job stopped (or the page closed) is ignored.
  useEffect(() => {
    if (!active) return;
    let disposed = false;
    const timer = setInterval(async () => {
      try {
        const response = await fetch("/api/processing/jobs", { cache: "no-store" });
        const reply = await readReply(response);
        if (disposed) return;
        // 404: the job expired or was cleared elsewhere, so forget it.
        if (response.status === 404) {
          setJob(null);
          setError(reply.error ?? "This job is no longer available.");
          return;
        }
        if (!response.ok) throw new Error(reply.error);
        setJob(reply as Job);
        setError("");
      } catch (e) {
        if (!disposed) setError(e instanceof Error && e.message ? e.message : "Cannot check the job right now.");
      }
    }, POLL_MS);
    return () => {
      disposed = true;
      clearInterval(timer);
    };
  }, [active]);

  // Every photo change starts over: nothing drawn, nothing marked, no consent yet.
  function selectFile(next: File | null) {
    setFile(next);
    setLoaded(false);
    setHasMask(false);
    setConsent(false);
  }

  // When a photo is chosen, draw its center square onto the 512 x 512 canvas.
  //
  // CROP MATH. `side` is the shorter of the photo's width and height. The
  // source square starts (width - side) / 2 from the left and
  // (height - side) / 2 from the top, which centres it; for a 4000 x 3000
  // photo that is the 3000 x 3000 square starting at x = 500, y = 0. drawImage
  // then scales that square to fill the whole 512 x 512 canvas.
  // The canvas is first filled with white so that any transparent pixels in a
  // PNG become white, as the worker would treat them.
  // The photo is opened through a temporary blob: URL that points at the local
  // file; it is released again when the photo changes.
  useEffect(() => {
    if (!file) return;
    let stopped = false;
    const url = URL.createObjectURL(file);
    const image = new Image();
    image.onload = () => {
      if (stopped) return;
      if (image.naturalWidth * image.naturalHeight > MAX_PIXELS) {
        setError("Choose a photo with fewer than 40 million pixels.");
        selectFile(null);
        return;
      }
      const target = photo.current?.getContext("2d");
      if (!target) return;
      const side = Math.min(image.naturalWidth, image.naturalHeight);
      target.fillStyle = "white";
      target.fillRect(0, 0, SIDE, SIDE);
      target.drawImage(image, (image.naturalWidth - side) / 2, (image.naturalHeight - side) / 2, side, side, 0, 0, SIDE, SIDE);
      mask.current?.getContext("2d")?.clearRect(0, 0, SIDE, SIDE);
      setLoaded(true);
    };
    image.onerror = () => {
      if (stopped) return;
      setError("This photo could not be read. Try another JPEG or PNG.");
      selectFile(null);
    };
    image.src = url;
    return () => {
      stopped = true;
      URL.revokeObjectURL(url);
    };
  }, [file]);

  /** Checks a chosen or dropped file's type and size before using it. */
  function choose(selected?: File) {
    if (!selected || locked) return;
    if (!["image/jpeg", "image/png"].includes(selected.type)) {
      setError("Choose a JPEG or PNG photo.");
      return;
    }
    if (selected.size > MAX_FILE_BYTES) {
      setError("Choose a photo smaller than 16 MB.");
      return;
    }
    setError("");
    selectFile(selected);
  }

  /**
   * Paints one point of a brush stroke on the mask layer.
   *
   * The canvas is drawn on screen at whatever size the layout gives it, but
   * its pixels are always 512 x 512, so the pointer position is scaled from
   * screen pixels to canvas pixels: (pointer x - canvas left) x 512 / on-screen
   * width, and the same for y.
   *
   * The first point of a stroke is a dot; each later point is a line from the
   * previous one, with round ends, so a fast stroke has no gaps. Painting
   * changes the mask, so any consent already given is withdrawn and must be
   * given again for the new mask.
   */
  function paint(e: PointerEvent<HTMLCanvasElement>) {
    if (!drawing.current || !loaded || locked) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const point = { x: ((e.clientX - rect.left) * SIDE) / rect.width, y: ((e.clientY - rect.top) * SIDE) / rect.height };
    const ctx = e.currentTarget.getContext("2d");
    if (!ctx) return;
    ctx.strokeStyle = "white";
    ctx.fillStyle = "white";
    ctx.lineWidth = BRUSH;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.beginPath();
    if (last.current) {
      ctx.moveTo(last.current.x, last.current.y);
      ctx.lineTo(point.x, point.y);
      ctx.stroke();
    } else {
      ctx.arc(point.x, point.y, BRUSH / 2, 0, Math.PI * 2);
      ctx.fill();
    }
    last.current = point;
    setHasMask(true);
    setConsent(false);
  }

  /** Ends the current stroke when the pointer is lifted or cancelled. */
  function stopPainting() {
    drawing.current = false;
    last.current = null;
  }

  /** Wipes the mask layer (and the consent that went with it). */
  function clearMask() {
    mask.current?.getContext("2d")?.clearRect(0, 0, SIDE, SIDE);
    setHasMask(false);
    setConsent(false);
  }

  /** Marks the middle quarter of the crop: a 256 x 256 square from (128, 128). */
  function markCenterRegion() {
    clearMask();
    const ctx = mask.current?.getContext("2d");
    if (!ctx) return;
    ctx.fillStyle = "white";
    ctx.fillRect(SIDE / 4, SIDE / 4, SIDE / 2, SIDE / 2);
    setHasMask(true);
  }

  /**
   * Sends the crop and mask to start a job. Runs only when everything is in
   * place: worker ready, photo loaded, mask painted, consent given.
   */
  async function start() {
    if (!ready || !loaded || !consent || !hasMask || locked) return;
    const base = photo.current;
    const layer = mask.current;
    if (!base || !layer) return;

    // Turn the soft brush strokes into a black-and-white mask (white = may be replaced).
    // A third, off-screen canvas is filled black and the transparent paint
    // layer is drawn over it, so unpainted pixels become black and painted ones
    // white or grey; normalizeEditMask then rounds the greys.
    const normalized = document.createElement("canvas");
    normalized.width = normalized.height = SIDE;
    const ctx = normalized.getContext("2d");
    if (!ctx) return;
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, SIDE, SIDE);
    ctx.drawImage(layer, 0, 0);
    const pixels = ctx.getImageData(0, 0, SIDE, SIDE);
    if (!normalizeEditMask(pixels.data)) {
      setError("Mark an area to protect, leaving some surrounding pixels unmarked.");
      return;
    }
    ctx.putImageData(pixels, 0, 0);

    setBusy(true);
    setError("");
    try {
      // toDataURL re-encodes the pixels, so no filename or EXIF data is sent.
      // It returns "data:image/png;base64,<data>"; split(",")[1] keeps only the
      // base64 data, which is what the gateway expects.
      const response = await fetch("/api/processing/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          recipe: PHOTO_GUARD_RECIPE,
          image_png: base.toDataURL("image/png").split(",")[1],
          mask_png: normalized.toDataURL("image/png").split(",")[1],
        }),
      });
      const reply = await readReply(response);
      if (!response.ok) throw new Error(reply.error);
      setJob(reply as Job);
      setConsent(false);
    } catch (e) {
      setError(e instanceof Error && e.message ? e.message : "The upload could not be confirmed.");
    } finally {
      setBusy(false);
    }
  }

  /**
   * Cancels and clears the job. When the server confirms it is gone, the photo
   * is removed from the page too; if it reports "cancelling", the page shows
   * that and keeps polling until the job is gone.
   */
  async function clearJob() {
    setBusy(true);
    setError("");
    try {
      const response = await fetch("/api/processing/jobs", { method: "DELETE" });
      const reply = await readReply(response);
      if (!response.ok) throw new Error(reply.error);
      if (reply.status === "cancelled") {
        setJob(null);
        selectFile(null);
      } else {
        setJob(reply as Job);
      }
    } catch (e) {
      setError(e instanceof Error && e.message ? e.message : "Deletion was not confirmed. Please retry.");
    } finally {
      setBusy(false);
    }
  }

  /**
   * Downloads the protected PNG. The file arrives as a blob in memory, a
   * temporary link to it is clicked to save it, and the blob is released 30
   * seconds later (long enough for the browser to finish saving).
   */
  async function download() {
    setBusy(true);
    setError("");
    try {
      const response = await fetch("/api/processing/output");
      if (!response.ok) throw new Error("Cannot download the result right now.");
      const blob = await response.blob();
      if (blob.type !== "image/png") throw new Error("The processor returned an invalid result.");
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "facade-photoguard.png";
      link.click();
      setTimeout(() => URL.revokeObjectURL(url), 30000);
    } catch (e) {
      setError(e instanceof Error && e.message ? e.message : "Download failed.");
    } finally {
      setBusy(false);
    }
  }

  // The layout: the photo panel (crop and mask brush) and the options panel
  // (server status, consent, upload, progress and download).
  return (
    <div className="workspace">
      <section className="photo-panel" aria-label="Photo preview">
        <div className="panel-top">
          <span>
            <ImageIcon size={18} /> Your photo
          </span>
          <span className="local-label">
            {job ? <Cloud size={14} /> : <LockKeyhole size={14} />} {job ? "Online job" : "Local preview"}
          </span>
        </div>

        <div
          className={"photo-stage " + (file ? "has-photo" : "")}
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault();
            choose(e.dataTransfer.files[0]);
          }}
        >
          {file ? (
            <>
              {/* The photo canvas with the transparent paint layer on top (see the file header). */}
              {/* The paint layer captures the pointer, so a stroke continues even if it leaves the canvas. */}
              <div className="mask-editor">
                <canvas ref={photo} width={SIDE} height={SIDE} aria-label="512 by 512 center crop of your photo" />
                <canvas
                  ref={mask}
                  width={SIDE}
                  height={SIDE}
                  className="mask-layer"
                  aria-label="Draw the area that might be replaced"
                  onPointerDown={(e) => {
                    e.currentTarget.setPointerCapture(e.pointerId);
                    drawing.current = true;
                    last.current = null;
                    paint(e);
                  }}
                  onPointerMove={paint}
                  onPointerUp={stopPainting}
                  onPointerCancel={stopPainting}
                />
              </div>
              <Button variant="secondary" className="remove-photo" disabled={locked} onClick={() => selectFile(null)} aria-label="Remove selected photo">
                <X size={16} />
              </Button>
            </>
          ) : (
            <div className="drop-content">
              <div className="upload-symbol">
                <Upload size={30} strokeWidth={1.5} />
              </div>
              <h2>Start with a photo</h2>
              <p>Drop it here, or choose one from your device.</p>
              <Button size="lg" disabled={locked} onClick={() => input.current?.click()}>
                Choose a photo
              </Button>
              <span className="file-hint">JPEG or PNG · up to 16 MB</span>
            </div>
          )}
          <input
            ref={input}
            type="file"
            tabIndex={-1}
            className="sr-only"
            accept="image/jpeg,image/png"
            aria-label="Choose photo"
            disabled={locked}
            onChange={(e) => {
              choose(e.target.files?.[0]);
              e.target.value = "";
            }}
          />
        </div>

        <div className="photo-caption">
          <span>{file ? "512\u00a0×\u00a0512 center crop. Only this crop is processed." : "Use a photo you own or have permission to use."}</span>
          {file && (
            <Button variant="ghost" disabled={locked} onClick={() => input.current?.click()}>
              Change photo
            </Button>
          )}
        </div>

        {loaded && (
          <div className="mask-instructions">
            <strong>Mark the area that could be replaced</strong>
            <p>
              Brush over the clothing or other area an editor might replace. PhotoGuard changes the surrounding pixels; the marked
              area stays unchanged in the protected PNG.
            </p>
            <Button variant="outline" disabled={locked} onClick={clearMask}>
              Clear marked area
            </Button>
            <Button variant="outline" disabled={locked} onClick={markCenterRegion}>
              Use center region
            </Button>
          </div>
        )}

        {error && (
          <p role="alert" className="error-message">
            {error}
          </p>
        )}
      </section>

      <aside className="options-panel">
        <p className="eyebrow">PROTECT IT ONLINE</p>
        <h2>PhotoGuard encoder</h2>

        <div className="service-card">
          <div className="service-title">
            <Cloud size={22} />
            <strong>Team GPU server</strong>
          </div>
          <p>
            Runs PhotoGuard&apos;s encoder method on the 512&nbsp;×&nbsp;512 crop, on the FACADE team&apos;s private GPU server. Only the crop and
            your marked area are sent, without the filename or photo metadata.
          </p>
          <span className={"availability" + (!checking && !ready ? " is-offline" : "")}>
            {!checking && !ready && <TriangleAlert size={14} aria-hidden />}
            {checking ? "Checking availability…" : ready ? "PhotoGuard encoder available" : "The GPU server is offline right now"}
          </span>
        </div>

        {job ? (
          <div className={"job-panel" + (job.status === "failed" ? " is-failed" : "")} aria-live="polite">
            <h3>
              {job.status === "failed" && <OctagonAlert size={18} aria-hidden />}
              {jobHeading(job.status)}
            </h3>
            <Progress value={(100 * job.iteration) / job.total_iterations} aria-label="Processing steps" />
            <p>
              {job.iteration} of {job.total_iterations} steps. This measures work completed, not protection strength.
            </p>
            {job.status === "completed" && (
              <Button disabled={busy} onClick={download}>
                <Download size={16} /> Download PNG
              </Button>
            )}
            <Button variant="outline" disabled={busy || job.status === "cancelling"} onClick={clearJob}>
              {active ? "Cancel and clear job" : "Clear online job"}
            </Button>
            <p>
              Save the PNG when it arrives. The server deletes the job about two minutes after your first download, and in any case
              30 minutes after upload. A run still in progress may take a little longer to stop.
            </p>
          </div>
        ) : (
          <div className="action-area">
            {ready && (
              <label className="upload-consent">
                <Checkbox checked={consent} onCheckedChange={(v) => setConsent(v === true)} disabled={!loaded || !hasMask || busy} />
                <span>
                  Send this crop and marked area to the FACADE team&apos;s private GPU server on AWS for this one job. The server keeps
                  them in memory only and deletes the job after processing: it holds one job for at most 30 minutes, less if you clear
                  it. Your photo is not used for training. The server uses Stable Diffusion&apos;s image encoder, so you
                  also agree to follow its{" "}
                  <a href={OPENRAIL_LICENSE_URL} target="_blank" rel="noreferrer">
                    use restrictions (CreativeML OpenRAIL-M)
                  </a>
                  .
                </span>
              </label>
            )}
            <Button className="process-button" size="lg" disabled={!ready || !loaded || !hasMask || !consent || busy} onClick={start}>
              {busy ? "Sending crop…" : ready ? "Apply PhotoGuard (experimental)" : "Online processing unavailable"}
            </Button>
            <p>
              {ready
                ? "Download the PNG when processing finishes. The result is experimental and may not resist other editors."
                : "You can still crop a photo and mark a region here. Nothing is uploaded until the server is ready and you choose to send it."}
            </p>
            {!ready && (
              <Button variant="ghost" disabled={checking} onClick={checkAvailability}>
                Check availability
              </Button>
            )}
          </div>
        )}

        <div className="app-note">
          <Smartphone size={20} />
          <p>
            <strong>Prefer to keep photos on your phone?</strong> The FACADE iOS app uses a separate, lighter method that runs
            entirely on the device. It is in testing; this website does not run it.
          </p>
        </div>

        <div className="honest-note">
          <Info size={20} />
          <p>Protection is experimental. Neither method is guaranteed to prevent editing or misuse.</p>
        </div>
      </aside>
    </div>
  );
}
