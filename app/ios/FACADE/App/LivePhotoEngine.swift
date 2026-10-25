// LivePhotoEngine.swift
//
// The FACADE protection method as the app sees it: give it a photo, get back
// the original and a protected copy as PNGs.
//
// Where it sits: LivePhotoModel (the Protect screen's state) calls
// `LivePhotoEngine.shared.run`. This file prepares the photo, sets up the
// engine's shared settings, runs NativePipeline, checks the run was complete,
// and crops the result back to the photo's own size.
//
// Key ideas:
// 1. Preparation: the photo is decoded with its orientation applied, scaled
//    so its longer side is at most 1024 px, and placed in the centre of a
//    1024x1024 gray canvas. The engine always works on that square; the gray
//    padding is cut off again at the end.
// 2. Safety limits: a run stops if it takes longer than 180 seconds, if the
//    phone gets too hot, or if the user cancels. A background task checks
//    every 0.25 s; the engine also checks between steps. No partial result is
//    ever returned.
// 3. Only one run at a time. The engine's step schedule and gradient settings
//    are process-wide, so the actor refuses a second run and restores the
//    previous settings when a run ends.
// 4. Nothing is written to Photos or kept on disk. A temporary cache folder
//    used for progress checkpoints is deleted at the end of every run.
// 5. Protection is experimental. The checks here confirm the engine did all of
//    its work; they cannot show that a particular face-swap tool will fail.
import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import CoreML
import CNative

/// A progress update for the screen: stage name, step within it, the number of
/// steps per stage, and seconds since the run started.
struct LivePhotoProgress: Sendable {
    let stage: String
    let step: Int
    let totalSteps: Int
    let elapsedSeconds: Double
}

/// The finished run: both images as PNG at the photo's own size, plus timings
/// for preparation, model loading and the engine itself.
struct LivePhotoResult: Sendable {
    let originalPNG: Data
    let protectedPNG: Data
    let width: Int
    let height: Int
    let elapsedSeconds: Double
    let preparationSeconds: Double
    let loadSeconds: Double
    let pipelineSeconds: Double
}

/// Result of `preparationSelfCheck`, used by the host tests.
struct LivePhotoPreparationCheckReport: Codable, Sendable {
    let passed: Bool
    let assertions: Int
    let caseCount: Int
}

/// Shared stop flag for one run, read by the worker and the monitor task.
/// `@unchecked Sendable` is safe here because `reason` is only touched under
/// the lock and the other fields never change.
private final class LivePhotoControl: @unchecked Sendable {
    let started = facade_profile_now()
    /// The time limit for one photo.
    let budgetSeconds = 180.0
    private let lock = NSLock()
    private var reason: String?
    /// Seconds since the run started (facade_profile_now is a nanosecond clock).
    var elapsed: Double { Double(facade_profile_now() - started) / 1e9 }
    var stopReason: String? {
        lock.lock(); defer { lock.unlock() }
        return reason
    }
    /// Records the first stop reason; later ones are ignored.
    func stop(_ value: String) {
        lock.lock(); defer { lock.unlock() }
        if reason == nil { reason = value }
    }
    /// Updates the stop reason from the clock, the thermal state and cancellation,
    /// then throws a user-facing error if the run must stop.
    func check() throws {
        if elapsed >= budgetSeconds { stop("time-budget") }
        let thermal = ProcessInfo.processInfo.thermalState
        // iOS reports thermal state as nominal, fair, serious or critical. The run
        // stops at the two highest levels.
        if thermal == .serious || thermal == .critical { stop("thermal-stop") }
        if Task.isCancelled { stop("cancelled") }
        if let reason = stopReason {
            if reason == "cancelled" { throw CancellationError() }
            if reason == "thermal-stop" {
                throw NativeError.invalid("The phone needs to cool down. Please try again after it has cooled.")
            }
            throw NativeError.invalid("Processing reached its 180-second limit. No finished copy was produced.")
        }
    }
}

/// Runs FACADE protection, one photo at a time. Each run happens on a
/// detached worker task. Cancelling the caller cancels the worker, and the
/// actor accepts no new run until that worker has returned.
/// The 180-second limit is checked between steps, so one long model call can
/// run a little past it.
/// No source pixels, intermediate images or results are saved to disk.
actor LivePhotoEngine {
    static let shared = LivePhotoEngine()
    /// True while a run is in progress.
    private var busy = false
    private init() {}

    /// FACADE's on-device models are private and not in git. The app builds
    /// without them; app/ios/Models/README.md says which files go where.
    static let requiredResources = [
        "FACADEModelHandoff/manifest.json",
        "NativeModels/inswapper/graph.json", "NativeModels/inswapper/weights.f32",
        "NativeModels/simswap/graph.json", "NativeModels/simswap/weights.f32",
        "NativeModels/detector.mlpackage", "NativeModels/parser.mlpackage", "NativeModels/emap.f32",
    ]
    /// True when every required model file is in the app bundle. The screen uses
    /// this to show "models are not installed" instead of offering the method.
    static var resourcesAvailable: Bool {
        let root = Bundle.main.bundleURL
        return requiredResources.allSatisfy {
            FileManager.default.fileExists(atPath: root.appendingPathComponent($0).path)
        }
    }

    /// Protects the photo in `data` (any format ImageIO can read). `progress` is
    /// called from a background thread. Throws if the photo is unsuitable, a model
    /// is missing or does not match, or the run is stopped.
    func run(data: Data, progress: @escaping @Sendable (LivePhotoProgress) -> Void) async throws -> LivePhotoResult {
        guard !busy else { throw NativeError.invalid("Please wait for the current photo to finish stopping.") }
        busy = true
        defer { busy = false }
        let control = LivePhotoControl()
        // The heavy work runs off the actor so the actor stays responsive.
        let worker = Task.detached(priority: .userInitiated) {
            try await Self.execute(data: data, control: control, progress: progress)
        }
        // A watchdog that checks time and heat four times a second and cancels the
        // worker if either limit is reached.
        let monitor = Task.detached(priority: .utility) {
            while !Task.isCancelled {
                let thermal = ProcessInfo.processInfo.thermalState
                if thermal == .serious || thermal == .critical { control.stop("thermal-stop") }
                if control.elapsed >= control.budgetSeconds { control.stop("time-budget") }
                if control.stopReason != nil { worker.cancel(); return }
                do { try await Task.sleep(nanoseconds: 250_000_000) } catch { return }
            }
        }
        let outcome: Result<LivePhotoResult, Error>
        do {
            // If the caller is cancelled (the user tapped Cancel), pass it on.
            let result = try await withTaskCancellationHandler(operation: {
                try await worker.value
            }, onCancel: {
                control.stop("cancelled")
                worker.cancel()
            })
            outcome = .success(result)
        } catch {
            outcome = .failure(error)
        }
        // Stop the watchdog, then turn any stop reason into the right error before
        // looking at the worker's result.
        monitor.cancel()
        await monitor.value
        try control.check()
        let result = try outcome.get()
        // Includes preparation, model load, pipeline, PNG encoding and cleanup.
        return LivePhotoResult(originalPNG: result.originalPNG, protectedPNG: result.protectedPNG,
            width: result.width, height: result.height, elapsedSeconds: control.elapsed,
            preparationSeconds: result.preparationSeconds, loadSeconds: result.loadSeconds,
            pipelineSeconds: result.pipelineSeconds)
    }

    /// The 1024x1024 canvas and where the photo sits inside it.
    private struct Prepared {
        let clean: RGBImage
        let x: Int
        let y: Int
        let width: Int
        let height: Int
        /// Cuts the photo's rectangle back out of a 1024x1024 canvas, row by row.
        func cropped(_ image: RGBImage) throws -> RGBImage {
            guard image.width == 1024, image.height == 1024 else {
                throw NativeError.invalid("The prepared photo dimensions changed.")
            }
            var bytes = [UInt8](repeating: 0, count: width * height * 3)
            for row in 0..<height {
                let source = ((y + row) * 1024 + x) * 3
                let destination = row * width * 3
                bytes.replaceSubrange(destination..<(destination + width * 3),
                    with: image.bytes[source..<(source + width * 3)])
            }
            return try RGBImage(width: width, height: height, bytes: bytes)
        }
    }

    /// Decodes the photo and centres it on a 1024x1024 gray canvas. ImageIO's
    /// thumbnail path applies the EXIF orientation and limits the longer side to
    /// 1024 px in one step. A square canvas keeps the engine's 256 px first stage
    /// and the full-size stages in the same coordinates; only the padding is
    /// cropped off afterwards.
    private nonisolated static func prepare(_ data: Data) throws -> Prepared {
        guard !data.isEmpty,
              let source = CGImageSourceCreateWithData(data as CFData,
                [kCGImageSourceShouldCache: false] as CFDictionary),
              CGImageSourceGetCount(source) > 0,
              let image = CGImageSourceCreateThumbnailAtIndex(source, 0, [
                kCGImageSourceCreateThumbnailFromImageAlways: true,
                kCGImageSourceCreateThumbnailWithTransform: true,
                kCGImageSourceShouldCacheImmediately: true,
                kCGImageSourceThumbnailMaxPixelSize: 1024
              ] as CFDictionary) else {
            throw NativeError.invalid("Could not read this photo.")
        }
        // Photos under 128 px on a side are refused as too small.
        let width = image.width, height = image.height
        guard width >= 128, height >= 128, width <= 1024, height <= 1024 else {
            throw NativeError.invalid("Choose a larger photo with one clearly visible face.")
        }
        // Top-left corner that centres the photo on the canvas.
        let x = (1024 - width) / 2, y = (1024 - height) / 2
        var rgba = [UInt8](repeating: 0, count: 1024 * 1024 * 4)
        let drawn = rgba.withUnsafeMutableBytes { raw -> Bool in
            guard let context = CGContext(data: raw.baseAddress, width: 1024, height: 1024,
                bitsPerComponent: 8, bytesPerRow: 1024 * 4,
                space: CGColorSpace(name: CGColorSpace.sRGB)!,
                bitmapInfo: CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue) else { return false }
            // Mid-gray padding.
            context.setFillColor(red: 127.0 / 255, green: 127.0 / 255, blue: 127.0 / 255, alpha: 1)
            context.fill(CGRect(x: 0, y: 0, width: 1024, height: 1024))
            // The thumbnail is already the right size, so draw it 1:1 without smoothing.
            context.interpolationQuality = .none
            context.draw(image, in: CGRect(x: x, y: y, width: width, height: height))
            return true
        }
        guard drawn else { throw NativeError.invalid("Could not prepare the photo pixels.") }
        // Drop the alpha byte: RGBA -> RGB.
        var rgb = [UInt8](repeating: 0, count: 1024 * 1024 * 3)
        for pixel in 0..<(1024 * 1024) {
            rgb[3 * pixel] = rgba[4 * pixel]
            rgb[3 * pixel + 1] = rgba[4 * pixel + 1]
            rgb[3 * pixel + 2] = rgba[4 * pixel + 2]
        }
        return Prepared(clean: try RGBImage(width: 1024, height: 1024, bytes: rgb),
            x: x, y: y, width: width, height: height)
    }

    /// Sets up the shared profiler state for one live run: the deadline used by
    /// `Profile.boundary` and a temporary cache folder for progress checkpoints
    /// (stage names and times only, no pixels). The detailed C profiler stays
    /// off. Refuses to start if another operation is already using that state.
    private nonisolated static func beginScope(_ control: LivePhotoControl) throws -> URL {
        guard Profile.destination == nil, Profile.observers.isEmpty,
              NativeViewGradientPolicy.rows.isEmpty, Profile.captures.isEmpty else {
            throw NativeError.invalid("The engine is already being used by another operation. Please relaunch the app.")
        }
        let cache = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
        let out = cache.appendingPathComponent("FACADE-LIVE-SCALARS-" + UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
        Profile.lock.lock()
        Profile.destination = out
        Profile.start = control.started
        Profile.deadline = control.started + UInt64(control.budgetSeconds * 1e9)
        Profile.stopped = nil
        Profile.enabled = false
        Profile.events = []
        Profile.eventsClosed = false
        Profile.lock.unlock()
        facade_profile_enable(0)
        return out
    }

    /// Undoes `beginScope`: clears the shared state and deletes the cache folder.
    private nonisolated static func endScope(_ out: URL) {
        // Nothing else writes events during a live run, and the native calls have
        // all returned by the time this runs.
        Profile.lock.lock()
        Profile.eventsClosed = true
        Profile.destination = nil
        Profile.deadline = nil
        Profile.stopped = nil
        Profile.start = 0
        Profile.enabled = false
        Profile.events = []
        Profile.lock.unlock()
        facade_profile_enable(0)
        NativeViewGradientPolicy.rows.removeAll(keepingCapacity: false)
        Profile.captures.removeAll(keepingCapacity: false)
        try? FileManager.default.removeItem(at: out)
    }

    /// The worker: prepare, load models, run the engine, check, export.
    private nonisolated static func execute(data: Data, control: LivePhotoControl,
        progress: @escaping @Sendable (LivePhotoProgress) -> Void) async throws -> LivePhotoResult {
        /// Sends a progress update. 16 = steps per stage in the 10+6 schedule.
        func update(_ stage: String, _ step: Int = 0) {
            progress(LivePhotoProgress(stage: stage, step: step, totalSteps: 16, elapsedSeconds: control.elapsed))
        }
        try control.check()
        let scope = try beginScope(control)
        // Remember the process-wide settings and restore them however the run ends.
        let previousCoarse = NativeSchedule.halfSteps, previousFine = NativeSchedule.fineSteps
        let previousPolicy = NativeViewGradientPolicy.arm
        defer {
            endScope(scope)
            NativeSchedule.halfSteps = previousCoarse
            NativeSchedule.fineSteps = previousFine
            NativeViewGradientPolicy.arm = previousPolicy
        }
        // 10 coarse + 6 fine steps, and the gradient policy that alternates the two
        // compressed upload conditions (see NativeViewGradientPolicy).
        NativeSchedule.configure(.shorter)
        NativeViewGradientPolicy.arm = "reweighted_views"
        update("preparing")
        let preparationStarted = facade_profile_now()
        let prepared = try autoreleasepool { try prepare(data) }
        // The engine gets the full canvas; the returned original is the photo alone.
        let inputPNG = try prepared.clean.png()
        let originalPNG = try prepared.cropped(prepared.clean).png()
        let preparationSeconds = Double(facade_profile_now() - preparationStarted) / 1e9
        try control.check()
        update("models.load")
        let loadStarted = facade_profile_now()
        // Models are loaded for every run and released when it ends. `.all` lets
        // Core ML put the face detector on the GPU or Neural Engine; the parser
        // and the two C++ recognition networks run on the CPU.
        let engine = try NativePipeline(root: Bundle.main.bundleURL,
            inswapper: "inswapper/graph.json", simswap: "simswap/graph.json",
            detectorComputeUnits: .all)
        let loadSeconds = Double(facade_profile_now() - loadStarted) / 1e9
        try control.check()
        let pipelineStarted = facade_profile_now()
        // Re-detect the face every second step in the fine phase.
        let result = try await engine.run(data: inputPNG, clean: prepared.clean, cadenceInterval: 2) { stage, step in
            update(stage, step)
        }
        let pipelineSeconds = Double(facade_profile_now() - pipelineStarted) / 1e9
        try control.check()
        // Completeness check with the 10+6 schedule:
        // - 102 trace rows = 6 stages x 17 steps (steps 0...16),
        // - 85 gradient-policy rows = 5 stages x 17 (the 256 px stage has none),
        // - 204 face-alignment requests = 4 stages x 3 conditions x 17, of which 168
        //   were fresh detections and 36 reused an alignment.
        // Any other count means a step was skipped, so no copy is produced.
        guard result.trace.count == 102, NativeViewGradientPolicy.rows.count == 85,
              result.cadence.halfSteps == 10, result.cadence.totalSteps == 16,
              result.cadence.interval == 2, result.cadence.requests == 204,
              result.cadence.refreshes == 168, result.cadence.reuses == 36,
              result.cadence.shadowDetections == 0 else {
            throw NativeError.invalid("The live engine did not complete its configured schedule.")
        }
        update("exporting")
        // Cut the protected canvas back to the photo's own size.
        let protectedPNG = try prepared.cropped(result.image).png()
        try control.check()
        return LivePhotoResult(originalPNG: originalPNG, protectedPNG: protectedPNG,
            width: prepared.width, height: prepared.height, elapsedSeconds: control.elapsed,
            preparationSeconds: preparationSeconds, loadSeconds: loadSeconds,
            pipelineSeconds: pipelineSeconds)
    }

    /// Tests `prepare` without any real photo. A synthetic image with four
    /// coloured quadrants is saved with each of the eight EXIF orientations, in
    /// landscape and portrait. Each must come out the right size, centred, with
    /// the colours in the right corners, and survive a second prepare unchanged.
    nonisolated static func preparationSelfCheck() throws -> LivePhotoPreparationCheckReport {
        // Quadrant colours: top-left red, top-right green, bottom-left blue,
        // bottom-right yellow.
        let palette: [[UInt8]] = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]]
        // For EXIF orientation 1...8: which palette colour should appear at the
        // top-left, top-right, bottom-left and bottom-right after the orientation is
        // applied (1 = as stored, 2 = mirrored, 3 = rotated 180, and so on).
        let corners = [[0, 1, 2, 3], [1, 0, 3, 2], [3, 2, 1, 0], [2, 3, 0, 1],
                       [0, 2, 1, 3], [2, 0, 3, 1], [3, 1, 2, 0], [1, 3, 0, 2]]
        var assertions = 0, cases = 0
        /// Counts an assertion and throws with its number if it fails.
        func require(_ value: Bool) throws {
            assertions += 1
            guard value else { throw NativeError.invalid("Live photo preparation self-check failed at assertion \(assertions).") }
        }
        for (width, height) in [(1024, 512), (512, 1024)] {
            var bytes = [UInt8](repeating: 0, count: width * height * 3)
            for row in 0..<height {
                for column in 0..<width {
                    let color = palette[(row >= height / 2 ? 2 : 0) + (column >= width / 2 ? 1 : 0)]
                    let offset = (row * width + column) * 3
                    for channel in 0..<3 { bytes[offset + channel] = color[channel] }
                }
            }
            let image = try RGBImage(width: width, height: height, bytes: bytes)
            let png = try image.png()
            guard let source = CGImageSourceCreateWithData(png as CFData, nil),
                  let cg = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
                throw NativeError.invalid("Self-check image could not be decoded.")
            }
            for orientation in 1...8 {
                let encoded = NSMutableData()
                guard let destination = CGImageDestinationCreateWithData(encoded, UTType.tiff.identifier as CFString, 1, nil) else {
                    throw NativeError.invalid("Self-check encoder is unavailable.")
                }
                CGImageDestinationAddImage(destination, cg, [kCGImagePropertyOrientation: orientation] as CFDictionary)
                guard CGImageDestinationFinalize(destination) else { throw NativeError.invalid("Self-check encoding failed.") }
                let prepared = try prepare(encoded as Data)
                // Orientations 5-8 include a quarter turn, so width and height swap.
                let expectedWidth = orientation <= 4 ? width : height
                let expectedHeight = orientation <= 4 ? height : width
                try require(prepared.width == expectedWidth && prepared.height == expectedHeight)
                try require(prepared.clean.width == 1024 && prepared.clean.height == 1024)
                try require(prepared.x == (1024 - expectedWidth) / 2 && prepared.y == (1024 - expectedHeight) / 2)
                let crop = try prepared.cropped(prepared.clean)
                for (index, point) in [(crop.width / 4, crop.height / 4),
                                       (3 * crop.width / 4, crop.height / 4),
                                       (crop.width / 4, 3 * crop.height / 4),
                                       (3 * crop.width / 4, 3 * crop.height / 4)].enumerated() {
                    let offset = (point.1 * crop.width + point.0) * 3
                    try require(Array(crop.bytes[offset..<(offset + 3)]) == palette[corners[orientation - 1][index]])
                }
                // Preparing the prepared photo again must give identical pixels.
                let again = try prepare(crop.png())
                try require(try again.cropped(again.clean).bytes == crop.bytes)
                cases += 1
            }
        }
        return LivePhotoPreparationCheckReport(passed: true, assertions: assertions, caseCount: cases)
    }
}


// Prepares a photo without running the engine, so the screen can show the
// photo exactly as the engine will see it. It uses the same private `prepare`
// as `run`.
extension LivePhotoEngine {
    /// The prepared photo as PNG, with its width and height.
    nonisolated static func fixturePreparedOriginal(_ data: Data) throws -> (png: Data, width: Int, height: Int) {
        let prepared = try prepare(data)
        return (try prepared.cropped(prepared.clean).png(), prepared.width, prepared.height)
    }
}
