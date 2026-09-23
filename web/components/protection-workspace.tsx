"use client";

// The "Prepare a photo" workspace: choose a photo, crop it to 512 x 512 in the
// browser, mark the area an editor might replace, then (with consent) send the
// crop to the PhotoGuard worker through /api/processing.

import { useEffect, useRef, useState, type PointerEvent } from "react";
import { Cloud, Download, ImageIcon, LockKeyhole, ShieldCheck, Smartphone, Upload, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Progress } from "@/components/ui/progress";
import { normalizeEditMask } from "@/lib/protection-mask";
import { OPENRAIL_LICENSE_URL, PHOTO_GUARD_RECIPE } from "@/lib/protection-recipe";

const SIDE = 512; // the worker's fixed input size
const BRUSH = 36;
const MAX_FILE_BYTES = 16 * 1024 * 1024;
const MAX_PIXELS = 40_000_000;
const POLL_MS = 4000;

type Job = { id: string; status: string; iteration: number; total_iterations: number };
type ApiReply = Partial<Job> & { error?: string; ready?: boolean };

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

export function ProtectionWorkspace() {
  const [file, setFile] = useState<File | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [hasMask, setHasMask] = useState(false);
  const [consent, setConsent] = useState(false);
  const [ready, setReady] = useState(false);
  const [checking, setChecking] = useState(true);
  const [busy, setBusy] = useState(false);
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState("");

  const input = useRef<HTMLInputElement>(null);
  const photo = useRef<HTMLCanvasElement>(null);
  const mask = useRef<HTMLCanvasElement>(null);
  const drawing = useRef(false);
  const last = useRef<{ x: number; y: number } | null>(null);

  const active = job !== null && ["queued", "running", "cancelling"].includes(job.status);
  const locked = job !== null || busy;

  function checkAvailability() {
    setChecking(true);
    fetchReady()
      .then(setReady)
      .finally(() => setChecking(false));
  }

  // On load: check the worker, and pick up a job this browser already started.
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

  // While a job is running, poll its progress.
  useEffect(() => {
    if (!active) return;
    let disposed = false;
    const timer = setInterval(async () => {
      try {
        const response = await fetch("/api/processing/jobs", { cache: "no-store" });
        const reply = await readReply(response);
        if (disposed) return;
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

  function stopPainting() {
    drawing.current = false;
    last.current = null;
  }

  function clearMask() {
    mask.current?.getContext("2d")?.clearRect(0, 0, SIDE, SIDE);
    setHasMask(false);
    setConsent(false);
  }

  function markCenterRegion() {
    clearMask();
    const ctx = mask.current?.getContext("2d");
    if (!ctx) return;
    ctx.fillStyle = "white";
    ctx.fillRect(SIDE / 4, SIDE / 4, SIDE / 2, SIDE / 2);
    setHasMask(true);
  }

  async function start() {
    if (!ready || !loaded || !consent || !hasMask || locked) return;
    const base = photo.current;
    const layer = mask.current;
    if (!base || !layer) return;

    // Turn the soft brush strokes into a black-and-white mask (white = may be replaced).
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

  return (
    <div className="workspace">
      <section className="photo-panel" aria-label="Photo preview">
        <div className="panel-top">
          <span>
            <ImageIcon size={18} /> Your photo
          </span>
          <span className="local-label">
            <LockKeyhole size={14} /> {job ? "Online job" : "Local preview"}
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
          <span>{file ? "512 × 512 center crop. Only this crop is processed." : "Use a photo you own or have permission to use."}</span>
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
            Runs PhotoGuard&apos;s encoder method on the 512 × 512 crop, on the FACADE team&apos;s private GPU server. Only the crop and
            your marked area are sent, without the filename or photo metadata.
          </p>
          <span className="availability">
            {checking ? "Checking availability…" : ready ? "PhotoGuard encoder available" : "The GPU server is offline right now"}
          </span>
        </div>

        {job ? (
          <div className="job-panel" aria-live="polite">
            <h3>{jobHeading(job.status)}</h3>
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
          <ShieldCheck size={20} />
          <p>Protection is experimental. Neither method is guaranteed to prevent editing or misuse.</p>
        </div>
      </aside>
    </div>
  );
}
