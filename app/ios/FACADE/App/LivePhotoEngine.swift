import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import CoreML
import CNative

struct LivePhotoProgress: Sendable {
    let stage: String
    let step: Int
    let totalSteps: Int
    let elapsedSeconds: Double
}

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

struct LivePhotoPreparationCheckReport: Codable, Sendable {
    let passed: Bool
    let assertions: Int
    let caseCount: Int
}

private final class LivePhotoControl: @unchecked Sendable {
    let started = facade_profile_now()
    let budgetSeconds = 180.0
    private let lock = NSLock()
    private var reason: String?
    var elapsed: Double { Double(facade_profile_now() - started) / 1e9 }
    var stopReason: String? {
        lock.lock(); defer { lock.unlock() }
        return reason
    }
    func stop(_ value: String) {
        lock.lock(); defer { lock.unlock() }
        if reason == nil { reason = value }
    }
    func check() throws {
        if elapsed >= budgetSeconds { stop("time-budget") }
        let thermal = ProcessInfo.processInfo.thermalState
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

/// A single live worker owns the native engine's process-wide schedule and
/// observation state. The caller's cancellation propagates to the detached
/// worker, and the actor does not admit another run until that worker returns.
/// The 180-second limit is cooperative; one synchronous call can overshoot.
/// This adapter never persists source pixels, intermediate images or results.
actor LivePhotoEngine {
    static let shared = LivePhotoEngine()
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
    static var resourcesAvailable: Bool {
        let root = Bundle.main.bundleURL
        return requiredResources.allSatisfy {
            FileManager.default.fileExists(atPath: root.appendingPathComponent($0).path)
        }
    }

    func run(data: Data, progress: @escaping @Sendable (LivePhotoProgress) -> Void) async throws -> LivePhotoResult {
        guard !busy else { throw NativeError.invalid("Please wait for the current photo to finish stopping.") }
        busy = true
        defer { busy = false }
        let control = LivePhotoControl()
        let worker = Task.detached(priority: .userInitiated) {
            try await Self.execute(data: data, control: control, progress: progress)
        }
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

    private struct Prepared {
        let clean: RGBImage
        let x: Int
        let y: Int
        let width: Int
        let height: Int
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

    /// ImageIO applies EXIF orientation while decoding a bounded thumbnail.
    /// A square canvas keeps the engine's 256px coverage image and native
    /// residual lift in the same coordinates. Only padding is cropped afterward.
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
        let width = image.width, height = image.height
        guard width >= 128, height >= 128, width <= 1024, height <= 1024 else {
            throw NativeError.invalid("Choose a larger photo with one clearly visible face.")
        }
        let x = (1024 - width) / 2, y = (1024 - height) / 2
        var rgba = [UInt8](repeating: 0, count: 1024 * 1024 * 4)
        let drawn = rgba.withUnsafeMutableBytes { raw -> Bool in
            guard let context = CGContext(data: raw.baseAddress, width: 1024, height: 1024,
                bitsPerComponent: 8, bytesPerRow: 1024 * 4,
                space: CGColorSpace(name: CGColorSpace.sRGB)!,
                bitmapInfo: CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue) else { return false }
            context.setFillColor(red: 127.0 / 255, green: 127.0 / 255, blue: 127.0 / 255, alpha: 1)
            context.fill(CGRect(x: 0, y: 0, width: 1024, height: 1024))
            context.interpolationQuality = .none
            context.draw(image, in: CGRect(x: x, y: y, width: width, height: height))
            return true
        }
        guard drawn else { throw NativeError.invalid("Could not prepare the photo pixels.") }
        var rgb = [UInt8](repeating: 0, count: 1024 * 1024 * 3)
        for pixel in 0..<(1024 * 1024) {
            rgb[3 * pixel] = rgba[4 * pixel]
            rgb[3 * pixel + 1] = rgba[4 * pixel + 1]
            rgb[3 * pixel + 2] = rgba[4 * pixel + 2]
        }
        return Prepared(clean: try RGBImage(width: 1024, height: 1024, bytes: rgb),
            x: x, y: y, width: width, height: height)
    }

    /// No notification observers or pixel files are created. Existing boundary
    /// checks use a temporary app-cache directory for scalar lifecycle events.
    /// The detailed C profiler stays disabled and its cumulative counters are
    /// never represented as measurements of an individual live photo.
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

    private nonisolated static func endScope(_ out: URL) {
        // The live scope installs no observers and has no concurrent event
        // writers. This executes after the native call stack has unwound.
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

    private nonisolated static func execute(data: Data, control: LivePhotoControl,
        progress: @escaping @Sendable (LivePhotoProgress) -> Void) async throws -> LivePhotoResult {
        func update(_ stage: String, _ step: Int = 0) {
            progress(LivePhotoProgress(stage: stage, step: step, totalSteps: 16, elapsedSeconds: control.elapsed))
        }
        try control.check()
        let scope = try beginScope(control)
        let previousCoarse = NativeSchedule.halfSteps, previousFine = NativeSchedule.fineSteps
        let previousPolicy = NativeViewGradientPolicy.arm
        defer {
            endScope(scope)
            NativeSchedule.halfSteps = previousCoarse
            NativeSchedule.fineSteps = previousFine
            NativeViewGradientPolicy.arm = previousPolicy
        }
        NativeSchedule.configure(.shorter)
        NativeViewGradientPolicy.arm = "reweighted_views"
        update("preparing")
        let preparationStarted = facade_profile_now()
        let prepared = try autoreleasepool { try prepare(data) }
        let inputPNG = try prepared.clean.png()
        let originalPNG = try prepared.cropped(prepared.clean).png()
        let preparationSeconds = Double(facade_profile_now() - preparationStarted) / 1e9
        try control.check()
        update("models.load")
        let loadStarted = facade_profile_now()
        let engine = try NativePipeline(root: Bundle.main.bundleURL,
            inswapper: "inswapper/graph.json", simswap: "simswap/graph.json",
            detectorComputeUnits: .all)
        let loadSeconds = Double(facade_profile_now() - loadStarted) / 1e9
        try control.check()
        let pipelineStarted = facade_profile_now()
        let result = try await engine.run(data: inputPNG, clean: prepared.clean, cadenceInterval: 2) { stage, step in
            update(stage, step)
        }
        let pipelineSeconds = Double(facade_profile_now() - pipelineStarted) / 1e9
        try control.check()
        guard result.trace.count == 102, NativeViewGradientPolicy.rows.count == 85,
              result.cadence.halfSteps == 10, result.cadence.totalSteps == 16,
              result.cadence.interval == 2, result.cadence.requests == 204,
              result.cadence.refreshes == 168, result.cadence.reuses == 36,
              result.cadence.shadowDetections == 0 else {
            throw NativeError.invalid("The live engine did not complete its configured schedule.")
        }
        update("exporting")
        let protectedPNG = try prepared.cropped(result.image).png()
        try control.check()
        return LivePhotoResult(originalPNG: originalPNG, protectedPNG: protectedPNG,
            width: prepared.width, height: prepared.height, elapsedSeconds: control.elapsed,
            preparationSeconds: preparationSeconds, loadSeconds: loadSeconds,
            pipelineSeconds: pipelineSeconds)
    }

    /// Synthetic pixels only. Exercises all eight EXIF orientations, both
    /// aspect ratios, canvas/crop dimensions and crop-to-PNG round-trip pixels.
    nonisolated static func preparationSelfCheck() throws -> LivePhotoPreparationCheckReport {
        let palette: [[UInt8]] = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]]
        let corners = [[0, 1, 2, 3], [1, 0, 3, 2], [3, 2, 1, 0], [2, 3, 0, 1],
                       [0, 2, 1, 3], [2, 0, 3, 1], [3, 1, 2, 0], [1, 3, 0, 2]]
        var assertions = 0, cases = 0
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
                let again = try prepare(crop.png())
                try require(try again.cropped(again.clean).bytes == crop.bytes)
                cases += 1
            }
        }
        return LivePhotoPreparationCheckReport(passed: true, assertions: assertions, caseCount: cases)
    }
}


// Fixture-only access bridge. Existing LivePhotoEngine bytes above this marker
// are unchanged; both prepare and run use the exact same private prepare method.
extension LivePhotoEngine {
    nonisolated static func fixturePreparedOriginal(_ data: Data) throws -> (png: Data, width: Int, height: Int) {
        let prepared = try prepare(data)
        return (try prepared.cropped(prepared.clean).png(), prepared.width, prepared.height)
    }
}
