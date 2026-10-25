// MiniPhotoGuardEngine.swift
//
// The "Mini PhotoGuard" protection method: a small, very experimental phone
// version of PhotoGuard (Salman et al., MIT Madry Lab). It is aimed at AI
// photo editors built on Stable Diffusion, such as clothing-edit tools.
//
// Where it sits: LivePhotoModel calls `MiniPhotoGuardEngine.shared.run` when
// the user picks the Mini PhotoGuard method. The result is returned as the
// same LivePhotoResult type the FACADE engine (LivePhotoEngine.swift) uses, so
// the Protect screen shows either method the same way.
//
// Key ideas:
// 1. The Core ML model is Stable Diffusion 1.5's image encoder (the VAE
//    encoder) cut down to 256 x 256 input, with the loss and its gradient built
//    into the model. It takes one input, "image", and returns "loss",
//    "gradient" and "latent_mean". The loss measures how far the image's
//    latent (the encoder's compressed description of it) is from the latent of
//    a flat gray image. Lowering it makes an editor "see" a gray picture
//    instead of the real one, so edits tend to come out broken.
// 2. The attack is projected gradient descent (PGD): 200 steps, each moving
//    every channel value by 1 level (out of 255) against the sign of the
//    gradient, while never letting any value drift more than 6 levels from
//    the original. The output is a 256 x 256 PNG.
// 3. Before trusting the model, the engine checks the files against SHA-256
//    hashes pinned in this file and runs a "parity check": it feeds a stored
//    reference input through the model and compares the outputs with stored
//    reference outputs. If either check fails, it refuses to run rather than
//    guess. Passing checks are remembered so later runs are faster.
//
// This is a prototype. It works only at 256 x 256, and like every method in
// the app it reduces, but does not guarantee to prevent, AI misuse.
import Foundation
import CoreML
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import CryptoKit
import Darwin

/// The one error type for Mini PhotoGuard. Every failure carries a message that
/// the Protect screen can show to the user as it is.
///
/// Notes for callers: this is the encoder-only, 256-pixel, 32-bit float
/// version of PhotoGuard's gray-latent objective, not the full diffusion
/// attack and not a face verifier. Run it only when no other heavy model engine
/// is running, and wait for a cancelled run to finish before starting another
/// engine. No loaded model survives a run; only the compiled model files are
/// kept on disk for reuse (see MiniPhotoGuardModelCache).
enum MiniPhotoGuardError: LocalizedError {
    /// A human-readable reason the run stopped.
    case invalid(String)
    /// Lets `error.localizedDescription` return the message itself.
    var errorDescription: String? { if case .invalid(let message) = self { return message }; return nil }
}

/// The safety brake shared by the worker thread and a watchdog task. It stops
/// the run on cancellation, after 10 minutes, when the phone gets hot, or when
/// the app uses more than 3 GB of memory. The first reason recorded wins.
/// `@unchecked Sendable` is safe here because the only mutable field is
/// protected by the lock.
private final class MiniPhotoGuardControl: @unchecked Sendable {
    /// Guards `reason`, which two threads may write at once.
    private let lock = NSLock()
    private var reason: String?
    /// When the run began (seconds since boot, so a clock change cannot affect it).
    let started = ProcessInfo.processInfo.systemUptime
    /// Seconds since the run began.
    var elapsed: Double { ProcessInfo.processInfo.systemUptime - started }
    /// Records why the run must stop. Later reasons are ignored.
    func stop(_ message: String) { lock.lock(); defer { lock.unlock() }; if reason == nil { reason = message } }
    /// Tests every limit, then throws if any stop reason has been recorded.
    /// Called between model calls; a single Core ML call cannot be interrupted.
    func check() throws {
        if Task.isCancelled { stop("miniPhotoGuard was cancelled.") }
        // 600 seconds = the 10-minute time limit.
        if elapsed > 600 { stop("miniPhotoGuard reached its 10-minute time limit.") }
        let thermal = ProcessInfo.processInfo.thermalState
        // iOS slows a hot phone down anyway; stopping avoids making it hotter.
        if thermal == .serious || thermal == .critical { stop("Let this device cool before trying miniPhotoGuard again.") }
        // A 3 GB cap on the app's memory footprint.
        if Self.memoryBytes() > 3_000_000_000 { stop("miniPhotoGuard reached its memory limit.") }
        lock.lock(); let message = reason; lock.unlock()
        if let message { throw MiniPhotoGuardError.invalid(message + " Elapsed: \(String(format: "%.1f", elapsed)) seconds.") }
    }
    /// The app's current memory footprint in bytes, the same figure iOS uses
    /// when deciding whether to end an app for using too much memory.
    static func memoryBytes() -> UInt64 {
        var info = task_vm_info_data_t()
        // task_info wants the structure size counted in 32-bit words, not bytes.
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let status = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        // If the query fails, report "huge" so the run stops instead of guessing.
        return status == KERN_SUCCESS ? info.phys_footprint : UInt64.max
    }
}

/// Pixel helpers that do not touch the model: preparing the 256 x 256 image,
/// converting between bytes and model values, the PGD step, and PNG encoding.
/// The host tests (Tests/MiniPhotoGuardEngineTests.swift) call these directly.
/// Byte images are RGB interleaved (R,G,B,R,G,B,...). Model arrays are NCHW
/// (all red values, then all green, then all blue), scaled to [-1, 1].
enum MiniPhotoGuardPixels {
    /// The model's fixed image size: 256 x 256 pixels.
    static let side = 256
    /// Number of channel values in one image: 256 x 256 x 3 = 196,608.
    static let count = side * side * 3
    /// The largest allowed change per channel value. On the [-1, 1] scale one
    /// 8-bit level is 2/255, so 12/255 here is 6 levels out of 255.
    static let epsilon: Float = 12.0 / 255.0
    /// How far each PGD step moves a value: 2/255 on the [-1, 1] scale = 1 level.
    static let step: Float = 2.0 / 255.0

    /// Converts interleaved RGB bytes (0...255) to the model's NCHW floats in [-1, 1].
    static func normalize(_ rgb: [UInt8]) throws -> [Float] {
        guard rgb.count == count else { throw MiniPhotoGuardError.invalid("Invalid miniPhotoGuard raster.") }
        let area = side * side
        var result = [Float](repeating: 0, count: count)
        // Byte index pixel*3+channel moves to plane index channel*area+pixel.
        // 0 maps to -1 and 255 maps to +1.
        for pixel in 0..<area { for channel in 0..<3 {
            result[channel * area + pixel] = Float(rgb[pixel * 3 + channel]) * (2.0 / 255.0) - 1
        } }
        return result
    }
    /// One PGD step, in place: move each value one step against the sign of the
    /// gradient (downhill on the loss), then clamp it back into both the
    /// original +/- epsilon box and the valid [-1, 1] range.
    static func project(_ image: inout [Float], original: [Float], gradient: [Float]) throws {
        guard image.count == count, original.count == count, gradient.count == count else {
            throw MiniPhotoGuardError.invalid("Invalid miniPhotoGuard gradient shape.")
        }
        for i in image.indices {
            // A NaN or infinity here would mean the model misbehaved; stop rather than save it.
            guard image[i].isFinite, original[i].isFinite, (-1...1).contains(original[i]), gradient[i].isFinite else {
                throw MiniPhotoGuardError.invalid("Nonfinite miniPhotoGuard calculation.")
            }
            // Only the sign of the gradient is used, so every value moves the same amount.
            let sign: Float = gradient[i] > 0 ? 1 : (gradient[i] < 0 ? -1 : 0)
            // Read inside out: take the step, raise it to at least the lower bound,
            // then cap it at the upper bound.
            image[i] = min(min(1, original[i] + epsilon), max(max(-1, original[i] - epsilon), image[i] - step * sign))
        }
    }
    /// Converts the finished float image back to 8-bit RGB bytes and checks, value
    /// by value, that it still respects the limits. Any violation throws, so a
    /// result that breaks the 6-level promise is never saved.
    static func quantize(_ image: [Float], originalRGB: [UInt8]) throws -> [UInt8] {
        let original = try normalize(originalRGB)
        guard image.count == count else { throw MiniPhotoGuardError.invalid("Invalid output shape.") }
        let area = side * side
        var rgb = originalRGB
        for pixel in 0..<area { for channel in 0..<3 {
            let i = channel * area + pixel
            let value = image[i]
            // Float rounding can land a hair past the limit, hence the ulpOfOne slack.
            guard value.isFinite, (-1...1).contains(value), abs(value - original[i]) <= epsilon + Float.ulpOfOne else {
                throw MiniPhotoGuardError.invalid("miniPhotoGuard projection check failed.")
            }
            // Inverse of normalize: -1 -> 0 and +1 -> 255, rounded to the nearest level.
            let quantized = Int(((value + 1) * 127.5).rounded())
            guard (0...255).contains(quantized), abs(quantized - Int(originalRGB[pixel * 3 + channel])) <= 6 else {
                throw MiniPhotoGuardError.invalid("miniPhotoGuard PNG pixel bound failed.")
            }
            rgb[pixel * 3 + channel] = UInt8(quantized)
        } }
        return rgb
    }
    /// Encodes 256 x 256 RGB bytes as a lossless sRGB PNG. PNG keeps every pixel
    /// exactly; JPEG would blur away the small changes the attack made.
    static func encode(_ rgb: [UInt8]) throws -> Data {
        guard rgb.count == count, let space = CGColorSpace(name: CGColorSpace.sRGB),
              let provider = CGDataProvider(data: Data(rgb) as CFData),
              let image = CGImage(width: side, height: side, bitsPerComponent: 8, bitsPerPixel: 24,
                bytesPerRow: side * 3, space: space, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent) else {
            throw MiniPhotoGuardError.invalid("Could not create the miniPhotoGuard image.")
        }
        // ImageIO writes the PNG into this in-memory buffer.
        let output = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(output, UTType.png.identifier as CFString, 1, nil) else {
            throw MiniPhotoGuardError.invalid("Could not encode the miniPhotoGuard PNG.")
        }
        CGImageDestinationAddImage(destination, image, [:] as CFDictionary)
        guard CGImageDestinationFinalize(destination) else { throw MiniPhotoGuardError.invalid("PNG encoding failed.") }
        return output as Data
    }
    /// Draws a CGImage into a 256 x 256 sRGB canvas and returns its RGB bytes.
    /// With `fit` the image is scaled to fit and centred on mid-gray (128)
    /// padding, so a photo that is not square keeps its shape. Without `fit`
    /// the image must already be 256 x 256 and is copied without smoothing.
    static func raster(_ image: CGImage, fit: Bool) throws -> [UInt8] {
        guard let space = CGColorSpace(name: CGColorSpace.sRGB) else { throw MiniPhotoGuardError.invalid("sRGB unavailable.") }
        // Core Graphics needs 4 bytes per pixel; the alpha byte is dropped at the end.
        var rgba = [UInt8](repeating: 255, count: side * side * 4)
        let success = rgba.withUnsafeMutableBytes { bytes -> Bool in
            guard let context = CGContext(data: bytes.baseAddress, width: side, height: side, bitsPerComponent: 8,
                bytesPerRow: side * 4, space: space,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue) else { return false }
            // Mid-gray background for the padding around a non-square photo.
            context.setFillColor(CGColor(colorSpace: space, components: [128.0/255, 128.0/255, 128.0/255, 1])!)
            context.fill(CGRect(x: 0, y: 0, width: side, height: side))
            // The smaller ratio makes the whole photo fit inside the square.
            let scale = min(Double(side) / Double(image.width), Double(side) / Double(image.height))
            let width = fit ? max(1, Int((Double(image.width) * scale).rounded())) : side
            let height = fit ? max(1, Int((Double(image.height) * scale).rounded())) : side
            context.interpolationQuality = fit ? .high : .none
            context.draw(image, in: CGRect(x: (side - width)/2, y: (side - height)/2, width: width, height: height))
            return true
        }
        guard success else { throw MiniPhotoGuardError.invalid("Could not prepare the miniPhotoGuard raster.") }
        var rgb = [UInt8](repeating: 0, count: count)
        // Keep R, G and B from each RGBA pixel.
        for pixel in 0..<(side * side) { for channel in 0..<3 { rgb[pixel*3+channel] = rgba[pixel*4+channel] } }
        return rgb
    }
    /// Turns the user's photo file (JPEG, HEIC, PNG...) into the 256 x 256 RGB
    /// bytes the model works on. Size limits are checked from the file header
    /// before anything is decoded, so an oversized file is refused cheaply.
    static func prepare(_ data: Data) throws -> [UInt8] {
        // Limits: at most a 32 MB file, 32,768 pixels per side and 160 megapixels.
        guard !data.isEmpty, data.count <= 32 * 1024 * 1024,
              let source = CGImageSourceCreateWithData(data as CFData, [kCGImageSourceShouldCache: false] as CFDictionary),
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let width = properties[kCGImagePropertyPixelWidth] as? Int,
              let height = properties[kCGImagePropertyPixelHeight] as? Int,
              width > 0, height > 0, width <= 32_768, height <= 32_768,
              Double(width) * Double(height) <= 160_000_000 else {
            throw MiniPhotoGuardError.invalid("This photo could not be opened within miniPhotoGuard's image limits.")
        }
        // Let ImageIO decode straight to a thumbnail no larger than 256 pixels, with
        // the photo's orientation flag applied so it comes out upright.
        let options: [CFString: Any] = [kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true, kCGImageSourceThumbnailMaxPixelSize: side,
            kCGImageSourceShouldCacheImmediately: true]
        guard let image = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary) else {
            throw MiniPhotoGuardError.invalid("This photo could not be prepared.")
        }
        return try raster(image, fit: true)
    }
    /// Reads one of this engine's own PNGs back into RGB bytes. Used to confirm
    /// that what was saved is exactly what was computed.
    static func decodePNG(_ data: Data) throws -> [UInt8] {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil), CGImageSourceGetCount(source) == 1,
              CGImageSourceGetType(source) as String? == UTType.png.identifier,
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil), image.width == side, image.height == side else {
            throw MiniPhotoGuardError.invalid("The saved miniPhotoGuard PNG has the wrong format.")
        }
        return try raster(image, fit: false)
    }
}

/// Reads a Core ML output as [Float]. A float32 array laid out in contiguous
/// row-major order is copied in bulk; that yields the same IEEE bits as the
/// per-element NSNumber read, which remains the path for every other layout.
/// The bulk copy is much faster for the 196,608-value gradient.
enum MiniPhotoGuardOutput {
    /// The strides a tightly packed row-major array of this shape would have.
    /// Example: shape [1, 3, 256, 256] gives strides [196608, 65536, 256, 1].
    static func contiguousStrides(_ shape: [Int]) -> [Int] {
        var strides = [Int](repeating: 1, count: shape.count)
        var running = 1
        for axis in shape.indices.reversed() { strides[axis] = running; running *= shape[axis] }
        return strides
    }
    /// True when the array is 32-bit float and tightly packed, so its memory can
    /// be copied directly. A padded or non-float32 array fails this.
    static func isContiguousFloat32(_ array: MLMultiArray) -> Bool {
        array.dataType == .float32 && array.strides.map(\.intValue) == contiguousStrides(array.shape.map(\.intValue))
    }
    /// The slow but always-correct path: read every element one at a time.
    static func perElement(_ array: MLMultiArray) -> [Float] { (0..<array.count).map { array[$0].floatValue } }
    /// The fast path: one memory copy, or nil if the layout does not allow it.
    static func bulk(_ array: MLMultiArray) -> [Float]? {
        guard isContiguousFloat32(array) else { return nil }
        let count = array.count
        return array.withUnsafeBufferPointer(ofType: Float.self) { buffer -> [Float]? in
            guard let base = buffer.baseAddress, buffer.count >= count else { return nil }
            return Array(UnsafeBufferPointer(start: base, count: count))
        }
    }
    /// Fast path when possible, slow path otherwise. Both give identical values.
    static func floats(_ array: MLMultiArray) -> [Float] { bulk(array) ?? perElement(array) }
}

/// Compiled-model cache. The bundled .mlpackage is compiled once per verified
/// model identity and OS build and kept in Caches (excluded from backup; iOS may
/// purge it when storage is low, which only costs one recompile); the stable
/// path also lets Core ML reuse its own device cache. Only a
/// complete compile is ever found under an identity folder: the compiler output
/// is staged beside it and renamed into place. A cache that fails to load is
/// deleted and rebuilt once. If the cache cannot be written, the run compiles to
/// a temporary folder that the caller deletes afterwards.
///
/// Why bother: an .mlpackage must be compiled to an .mlmodelc before Core ML
/// can load it, and compiling this 214 MB model takes noticeable time.
enum MiniPhotoGuardModelCache {
    /// Name of the compiled model folder inside each cache folder.
    static let compiledName = "model.mlmodelc"
    /// The loaded model plus how it was obtained: `compiled` is true when this run
    /// compiled it, `temporary` is a folder the caller must delete (only when the
    /// cache could not be used), and `identity` names the compiled folder that was
    /// loaded; every compile gets a new one.
    struct Loaded { let model: MLModel; let compiled: Bool; let temporary: URL?; let identity: String }

    /// A fingerprint for one compiled folder on disk: its path, file-system file
    /// number and creation time. A recompile at the same path gets a new file
    /// number or creation time, so the parity check knows to run again.
    static func identity(_ url: URL) -> String {
        let attributes = try? FileManager.default.attributesOfItem(atPath: url.path)
        let number = (attributes?[.systemFileNumber] as? NSNumber)?.uint64Value ?? 0
        let created = (attributes?[.creationDate] as? Date)?.timeIntervalSinceReferenceDate.bitPattern ?? 0
        return "\(url.path):\(number):\(created)"
    }

    /// The cache's parent folder, inside the app's Caches directory.
    static func root() -> URL? {
        try? FileManager.default.url(for: .cachesDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
            .appendingPathComponent("MiniPhotoGuardCompiledModel", isDirectory: true)
    }
    /// The cache folder for one model version on one OS build: the first 16 hex
    /// digits of each hash plus the OS version string, with anything that is not
    /// a letter, digit or dot replaced by "_" so it is a safe folder name.
    static func folderName(manifestSHA: String, exportSHA: String, system: String) -> String {
        let os = String(String.UnicodeScalarView(system.unicodeScalars.map {
            ($0.isASCII && (CharacterSet.alphanumerics.contains($0) || $0 == ".")) ? $0 : "_" }))
        return "\(manifestSHA.prefix(16))-\(exportSHA.prefix(16))-\(os)"
    }
    /// Loads the model, from the cache if a good compiled copy is there, otherwise
    /// by compiling the package (and trying to cache the result).
    /// `afterCompile` runs right after compiling so the caller can stop the run
    /// (for example on cancel) before the slow model load.
    static func load(package: URL, folder: URL?, configuration: MLModelConfiguration,
                     afterCompile: () throws -> Void) throws -> Loaded {
        let files = FileManager.default
        if let folder {
            let cached = folder.appendingPathComponent(compiledName, isDirectory: true)
            if files.fileExists(atPath: cached.path) {
                if let model = try? MLModel(contentsOf: cached, configuration: configuration) {
                    return Loaded(model: model, compiled: false, temporary: nil, identity: identity(cached))
                }
                try? files.removeItem(at: folder) // Unloadable cache: discard it and rebuild once.
            }
        }
        // Core ML compiles into a temporary folder it chooses; publish tries to move it into the cache.
        let compiled = try MLModel.compileModel(at: package)
        let published = folder.flatMap { publish(compiled, to: $0) }
        // If publishing failed, the compiled copy stays temporary and is deleted later.
        let temporary = published == nil ? compiled : nil
        do { try afterCompile() } catch {
            if let temporary { try? files.removeItem(at: temporary) }
            throw error
        }
        do {
            let model = try MLModel(contentsOf: published ?? compiled, configuration: configuration)
            return Loaded(model: model, compiled: true, temporary: temporary, identity: identity(published ?? compiled))
        } catch {
            // A compiled model that will not load is removed, wherever it lives.
            if let temporary { try? files.removeItem(at: temporary) } else if let folder { try? files.removeItem(at: folder) }
            throw error
        }
    }
    /// Moves a fresh compile into the cache folder. Returns the cached model's
    /// URL, or nil with the compiler output left where it was.
    /// The move goes through a hidden staging folder and a final rename, so the
    /// cache folder either holds a complete model or does not exist.
    private static func publish(_ compiled: URL, to folder: URL) -> URL? {
        let files = FileManager.default
        var root = folder.deletingLastPathComponent()
        let staging = root.appendingPathComponent(".staging-" + UUID().uuidString, isDirectory: true)
        let staged = staging.appendingPathComponent(compiledName, isDirectory: true)
        do {
            try files.createDirectory(at: root, withIntermediateDirectories: true)
            var values = URLResourceValues(); values.isExcludedFromBackup = true
            try root.setResourceValues(values)
            // Clear out everything else: caches for earlier model versions or OS
            // builds, and stagings left by interrupted runs. Only one copy is kept.
            for entry in try files.contentsOfDirectory(at: root, includingPropertiesForKeys: nil) { try? files.removeItem(at: entry) }
            try files.createDirectory(at: staging, withIntermediateDirectories: false)
            try files.moveItem(at: compiled, to: staged)
        } catch { try? files.removeItem(at: staging); return nil }
        do { try files.moveItem(at: staging, to: folder) } catch { // Same-directory rename: all or nothing.
            try? files.moveItem(at: staged, to: compiled)
            try? files.removeItem(at: staging)
            return nil
        }
        // Success: the complete compiled model now sits in the cache folder.
        return folder.appendingPathComponent(compiledName, isDirectory: true)
    }
}

/// Once-per-install records in UserDefaults: digests of the build numbers, the
/// bundle and executable, and each listed file's size and modification date. A
/// mismatch only repeats the full check; a failed check never records anything.
enum MiniPhotoGuardMarkers {
    /// UserDefaults key: the install fingerprint whose files were all hashed.
    static let verifiedInstall = "MiniPhotoGuard.verifiedInstall.v1"
    /// UserDefaults key: the fingerprint of the last setup that passed the parity check.
    static let passedParity = "MiniPhotoGuard.passedParity.v1"
    /// SHA-256 of the parts joined with newlines, as lowercase hex.
    static func digest(_ parts: [String]) -> String {
        SHA256.hash(data: Data(parts.joined(separator: "\n").utf8)).map { String(format: "%02x", $0) }.joined()
    }
    /// A cheap "has this file changed?" stamp: its size and modification time.
    /// Missing files give "unavailable", which simply never matches a stored value.
    static func stamp(_ url: URL?) -> String {
        guard let url, let attributes = try? FileManager.default.attributesOfItem(atPath: url.path),
              let size = attributes[.size] as? NSNumber, let date = attributes[.modificationDate] as? Date else { return "unavailable" }
        return "\(size.int64Value):\(date.timeIntervalSinceReferenceDate.bitPattern)"
    }
}

/// The Mini PhotoGuard engine. It is an `actor` so its `busy` flag can only be
/// touched by one caller at a time; the heavy work itself runs on a detached
/// task so the screen stays responsive.
actor MiniPhotoGuardEngine {
    /// The single shared engine; the private init stops anyone making a second one.
    static let shared = MiniPhotoGuardEngine()
    /// True while a run is in progress; a second run is refused.
    private var busy = false
    private init() {}
    /// SHA-256 identifying the exported Core ML encoder this code was checked with.
    private static let exportSHA = "b474f21b584feec1f003b88634e5b13d6444c3d205e845108e10783bc0a7730e"
    /// SHA-256 of the exact manifest.json this app version accepts. Any other
    /// manifest, even a harmless edit, is refused.
    private static let frozenManifestSHA = "174bf74e484ecfa546ef108fffe835080cba4a64fd5cff931f3dca883412036d"
    /// One file listed in the manifest: path, expected SHA-256 and size in bytes.
    private struct Binding: Decodable { let path: String; let sha256: String; let bytes: Int }
    /// The decoded manifest.json: the model files, the four reference ("fixture")
    /// files for the parity check, the attack settings, and the parity tolerances.
    private struct Manifest: Decodable {
        let schemaVersion: Int; let side: Int; let modelRelativePath: String; let modelFiles: [Binding]
        let fixtures: [String: Binding]; let exportSHA256: String
        let epsilonPixels: Double; let stepPixels: Double; let maxSteps: Int
        let gradientCosineMinimum: Double; let gradientRelativeL2Maximum: Double; let latentRelativeL2Maximum: Double
        let lossAbsoluteTolerance: Double; let lossRelativeTolerance: Double
    }
    /// The checked manifest plus where the payload folder and model package are.
    private struct Payload { let manifest: Manifest; let root: URL; let model: URL }
    /// One model call's three outputs.
    private struct Evaluation { let loss: Float; let gradient: [Float]; let latent: [Float] }

    /// Whether the model is installed and its files have the right sizes. The UI
    /// uses this to enable or grey out the method. It does not hash the files.
    nonisolated static var resourcesAvailable: Bool { (try? payload(fullVerification: false)) != nil }
    /// Makes the 256 x 256 "original" preview the screen shows before protection,
    /// prepared exactly as the run will prepare it.
    nonisolated static func preparePhoto(data: Data) throws -> (png: Data, width: Int, height: Int) {
        let rgb = try MiniPhotoGuardPixels.prepare(data)
        let png = try MiniPhotoGuardPixels.encode(rgb)
        // Round-trip check: the PNG must decode to exactly the same bytes.
        guard try MiniPhotoGuardPixels.decodePNG(png) == rgb else { throw MiniPhotoGuardError.invalid("Prepared PNG pixel check failed.") }
        return (png, 256, 256)
    }
    /// Protects one photo. Reports progress through `progress`, and returns the
    /// 256 x 256 original and protected PNGs with timing figures.
    func run(data: Data, progress: @escaping @Sendable (LivePhotoProgress) -> Void) async throws -> LivePhotoResult {
        guard !busy else { throw MiniPhotoGuardError.invalid("miniPhotoGuard is already running.") }
        busy = true
        defer { busy = false }
        let control = MiniPhotoGuardControl()
        try control.check()
        // The heavy work runs off the actor. The autorelease pool makes sure Core ML's
        // temporary objects (and the model) are freed when the work ends.
        let worker = Task.detached(priority: .userInitiated) {
            try autoreleasepool { try Self.execute(data: data, control: control, progress: progress) }
        }
        // Watchdog: checks the limits every 250 ms while the worker is busy inside a
        // long Core ML call, and latches the first limit that trips. The worker
        // then stops at its next check.
        let monitor = Task.detached {
            while !Task.isCancelled {
                do { try control.check() } catch { break }
                try? await Task.sleep(for: .milliseconds(250))
            }
        }
        defer { monitor.cancel() }
        // Core ML predictions are synchronous: cancellation joins the worker and
        // takes effect between calls, never falsely signals resource release.
        return try await withTaskCancellationHandler {
            let result = try await worker.value
            try control.check()
            // Include model teardown in elapsed time; the worker's autorelease
            // scope has ended before another engine may acquire the UI slot.
            return LivePhotoResult(originalPNG: result.originalPNG, protectedPNG: result.protectedPNG,
                width: result.width, height: result.height, elapsedSeconds: control.elapsed,
                preparationSeconds: result.preparationSeconds, loadSeconds: result.loadSeconds,
                pipelineSeconds: result.pipelineSeconds)
        } onCancel: {
            control.stop("miniPhotoGuard was cancelled.")
            worker.cancel()
        }
    }
    /// Turns a path from the manifest into a URL inside `root`, refusing absolute
    /// paths, backslashes, empty parts, "." and "..", and symbolic links, so the
    /// manifest cannot point outside the app's own files.
    private nonisolated static func safe(_ relative: String, root: URL) throws -> URL {
        guard !relative.isEmpty, !relative.hasPrefix("/"), !relative.contains("\\"),
              relative.split(separator: "/", omittingEmptySubsequences: false).allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." }) else {
            throw MiniPhotoGuardError.invalid("Invalid miniPhotoGuard resource path.")
        }
        let result = root.appendingPathComponent(relative).standardizedFileURL
        guard result.resolvingSymlinksInPath().path == result.path else { throw MiniPhotoGuardError.invalid("Symlinked miniPhotoGuard resource.") }
        return result
    }
    /// SHA-256 of some bytes, as lowercase hex.
    private nonisolated static func digest(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
    /// Checks one listed file: it must be a regular file of the stated size, and
    /// with `full` its SHA-256 must match too. Returns its URL.
    private nonisolated static func verify(_ binding: Binding, root: URL, full: Bool) throws -> URL {
        guard binding.bytes > 0, binding.sha256.range(of: "^[0-9a-f]{64}$", options: .regularExpression) != nil else {
            throw MiniPhotoGuardError.invalid("Invalid miniPhotoGuard resource binding.")
        }
        let url = try safe(binding.path, root: root)
        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        guard attributes[.type] as? FileAttributeType == .typeRegular,
              (attributes[.size] as? NSNumber)?.intValue == binding.bytes else { throw MiniPhotoGuardError.invalid("miniPhotoGuard resource is missing or changed.") }
        if full {
            let handle = try FileHandle(forReadingFrom: url)
            defer { try? handle.close() }
            var hash = SHA256()
            // Hash in 4 MB pieces so the 214 MB weights file is never fully in memory.
            while let data = try handle.read(upToCount: 4 * 1024 * 1024), !data.isEmpty { hash.update(data: data) }
            guard hash.finalize().map({ String(format: "%02x", $0) }).joined() == binding.sha256 else {
                throw MiniPhotoGuardError.invalid("miniPhotoGuard resource hash failed.")
            }
        }
        return url
    }
    /// Convenience form: hash everything, or nothing, regardless of past checks.
    private nonisolated static func payload(fullVerification: Bool) throws -> Payload {
        try payload(full: { _ in fullVerification }).payload
    }
    /// Finds and checks the model bundle. `full` receives this install's
    /// fingerprint and decides whether to hash every file (true) or only check
    /// sizes (false). Returns the payload and that fingerprint.
    private nonisolated static func payload(full: (String) -> Bool) throws -> (payload: Payload, install: String) {
        // The model bundle is private and not in git; see app/ios/Models/README.md.
        let root = Bundle.main.bundleURL.appendingPathComponent("MiniPhotoGuardPayload")
        guard let data = try? Data(contentsOf: root.appendingPathComponent("manifest.json")) else {
            throw MiniPhotoGuardError.invalid("The miniPhotoGuard model is not installed in this build.")
        }
        guard digest(data) == frozenManifestSHA else {
            throw MiniPhotoGuardError.invalid("The verified miniPhotoGuard model is not installed.")
        }
        let manifest = try JSONDecoder().decode(Manifest.self, from: data)
        // The manifest must describe exactly the recipe this code implements:
        // 256 pixels, a 6-level limit, 1-level steps, 200 steps, the four reference
        // files, and no file listed twice.
        guard manifest.schemaVersion == 1, manifest.side == 256, manifest.exportSHA256 == exportSHA,
              manifest.modelRelativePath == "photoguard_encoder_gradient_256_fp32.mlpackage",
              manifest.epsilonPixels == 6, manifest.stepPixels == 1, manifest.maxSteps == 200,
              Set(manifest.fixtures.keys) == Set(["input", "loss", "gradient", "latent_mean"]),
              !manifest.modelFiles.isEmpty, Set(manifest.modelFiles.map(\.path)).count == manifest.modelFiles.count else {
            throw MiniPhotoGuardError.invalid("The miniPhotoGuard recipe differs from the verified model.")
        }
        let model = try safe(manifest.modelRelativePath, root: Bundle.main.bundleURL)
        let info = Bundle.main.infoDictionary
        // The install fingerprint: app location, executable stamp, build numbers,
        // manifest hash, then a size/date stamp for every listed file. Any update
        // of the app or the model changes it.
        var parts = [Bundle.main.bundleURL.path, MiniPhotoGuardMarkers.stamp(Bundle.main.executableURL),
                     info?["CFBundleVersion"] as? String ?? "", info?["CFBundleShortVersionString"] as? String ?? "", frozenManifestSHA]
        for binding in manifest.modelFiles { parts.append(binding.path + "=" + MiniPhotoGuardMarkers.stamp(try? safe(binding.path, root: model))) }
        for (name, binding) in manifest.fixtures.sorted(by: { $0.key < $1.key }) {
            parts.append(name + "=" + MiniPhotoGuardMarkers.stamp(try? safe(binding.path, root: root)))
        }
        let install = MiniPhotoGuardMarkers.digest(parts)
        let fullVerification = full(install)
        // Size checks always run; hashes only when `full` said so.
        for binding in manifest.modelFiles { _ = try verify(binding, root: model, full: fullVerification) }
        for binding in manifest.fixtures.values { _ = try verify(binding, root: root, full: fullVerification) }
        return (Payload(manifest: manifest, root: root, model: model), install)
    }
    /// Reads a reference file of `count` little-endian 32-bit floats.
    private nonisolated static func floats(_ binding: Binding, root: URL, count: Int) throws -> [Float] {
        let data = try Data(contentsOf: safe(binding.path, root: root))
        guard data.count == count * 4 else { throw MiniPhotoGuardError.invalid("Invalid fixture shape.") }
        let result = data.withUnsafeBytes { bytes in (0..<count).map { Float(bitPattern: UInt32(littleEndian: bytes.loadUnaligned(fromByteOffset: $0*4, as: UInt32.self))) } }
        guard result.allSatisfy(\.isFinite) else { throw MiniPhotoGuardError.invalid("Nonfinite reference fixture.") }
        return result
    }
    /// Runs the model once on an image (NCHW floats in [-1, 1]) and returns the
    /// loss, the gradient of the loss for every input value, and the latent.
    private nonisolated static func evaluate(_ model: MLModel, image: [Float]) throws -> Evaluation {
        try autoreleasepool {
            guard image.count == MiniPhotoGuardPixels.count, image.allSatisfy({ $0.isFinite && (-1...1).contains($0) }) else { throw MiniPhotoGuardError.invalid("Invalid model input.") }
            // Model input "image": batch 1, 3 colour planes, 256 x 256.
            let array = try MLMultiArray(shape: [1, 3, 256, 256], dataType: .float32)
            image.withUnsafeBufferPointer { array.dataPointer.copyMemory(from: $0.baseAddress!, byteCount: image.count * 4) }
            let output = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(multiArray: array)]))
            // Reads one named output and checks its type, shape and that it has no NaNs.
            func values(_ name: String, _ shape: [Int]) throws -> [Float] {
                guard let array = output.featureValue(for: name)?.multiArrayValue, array.dataType == .float32,
                      array.shape.map(\.intValue) == shape else { throw MiniPhotoGuardError.invalid("Invalid model output: \(name).") }
                let values = MiniPhotoGuardOutput.floats(array)
                guard values.allSatisfy(\.isFinite) else { throw MiniPhotoGuardError.invalid("Nonfinite model output.") }
                return values
            }
            // Outputs: "loss" (one number), "gradient" (same shape as the input) and
            // "latent_mean" (4 x 32 x 32: the encoder shrinks each side by 8).
            return try Evaluation(loss: values("loss", [1])[0], gradient: values("gradient", [1,3,256,256]), latent: values("latent_mean", [1,4,32,32]))
        }
    }
    /// Euclidean length (L2 norm) of a vector, summed in Double for accuracy.
    private nonisolated static func norm(_ values: [Float]) -> Double { sqrt(values.reduce(0) { $0 + Double($1) * Double($1) }) }
    /// Relative L2 error: |a - b| / |b|. The 1e-30 floor avoids dividing by zero.
    private nonisolated static func relative(_ a: [Float], _ b: [Float]) -> Double {
        sqrt(zip(a,b).reduce(0) { $0 + pow(Double($1.0) - Double($1.1), 2) }) / max(norm(b), 1e-30)
    }
    /// The whole run, on the worker thread: check files, load the model, run the
    /// parity check if needed, prepare the photo, take 200 PGD steps, and save.
    private nonisolated static func execute(data: Data, control: MiniPhotoGuardControl,
        progress: @escaping @Sendable (LivePhotoProgress) -> Void) throws -> LivePhotoResult {
        // Every progress update is also a stop check.
        func report(_ stage: String, _ step: Int = 0) throws {
            try control.check()
            progress(LivePhotoProgress(stage: stage, step: step, totalSteps: 200, elapsedSeconds: control.elapsed))
        }
        try report("Checking miniPhotoGuard resources")
        // Every file is hashed on the first run of each installed build; later
        // runs of the same signed, read-only install check sizes only.
        let defaults = UserDefaults.standard
        let (checked, install) = try payload(full: { defaults.string(forKey: MiniPhotoGuardMarkers.verifiedInstall) != $0 })
        if defaults.string(forKey: MiniPhotoGuardMarkers.verifiedInstall) != install {
            defaults.set(install, forKey: MiniPhotoGuardMarkers.verifiedInstall)
        }
        try control.check()
        let manifest = checked.manifest
        let loadStart = control.elapsed
        try report("Loading miniPhotoGuard")
        // CPU and GPU only, no Neural Engine. The setting is part of the parity
        // fingerprint below, so changing it forces a fresh parity check.
        let configuration = MLModelConfiguration(); configuration.computeUnits = .cpuAndGPU
        let system = ProcessInfo.processInfo.operatingSystemVersionString
        let folder = MiniPhotoGuardModelCache.root()?.appendingPathComponent(
            MiniPhotoGuardModelCache.folderName(manifestSHA: frozenManifestSHA, exportSHA: exportSHA, system: system), isDirectory: true)
        let loaded = try MiniPhotoGuardModelCache.load(package: checked.model, folder: folder, configuration: configuration) {
            try control.check()
        }
        defer { if let temporary = loaded.temporary { try? FileManager.default.removeItem(at: temporary) } } // Only this call's compiler output, never images.
        let model = loaded.model
        try control.check()
        let loadSeconds = control.elapsed - loadStart
        try report("Checking the phone gradient")
        // The parity check runs once per install, OS build, compute units and
        // compiled model folder, so again after any new compile. Only a pass is recorded.
        let parity = MiniPhotoGuardMarkers.digest([install, system, "computeUnits=\(configuration.computeUnits.rawValue)", loaded.identity])
        if defaults.string(forKey: MiniPhotoGuardMarkers.passedParity) != parity {
            let fixture = try floats(manifest.fixtures["input"]!, root: checked.root, count: MiniPhotoGuardPixels.count)
            let expectedGradient = try floats(manifest.fixtures["gradient"]!, root: checked.root, count: MiniPhotoGuardPixels.count)
            let expectedLatent = try floats(manifest.fixtures["latent_mean"]!, root: checked.root, count: 4*32*32)
            let expectedLoss = try floats(manifest.fixtures["loss"]!, root: checked.root, count: 1)[0]
            let observed = try evaluate(model, image: fixture)
            try control.check()
            // Cosine similarity: 1.0 means the phone's gradient points in exactly the
            // same direction as the reference gradient.
            let cosine = zip(observed.gradient, expectedGradient).reduce(0.0) { $0 + Double($1.0)*Double($1.1) }
                / max(norm(observed.gradient)*norm(expectedGradient), 1e-30)
            // Pass only if the gradient direction, gradient size, latent and loss all
            // match the reference within the manifest's tolerances (cosine >= 0.99,
            // gradient within 5%, latent within 1%, loss within 1%).
            guard norm(expectedGradient) > 1e-12, norm(expectedLatent) > 1e-12,
                  cosine >= manifest.gradientCosineMinimum,
                  relative(observed.gradient, expectedGradient) <= manifest.gradientRelativeL2Maximum,
                  relative(observed.latent, expectedLatent) <= manifest.latentRelativeL2Maximum,
                  abs(Double(observed.loss)-Double(expectedLoss)) <= max(manifest.lossAbsoluteTolerance, abs(Double(expectedLoss))*manifest.lossRelativeTolerance) else {
                throw MiniPhotoGuardError.invalid("This device did not pass the miniPhotoGuard gradient check.")
            }
            defaults.set(parity, forKey: MiniPhotoGuardMarkers.passedParity)
        }
        try report("Preparing the 256-pixel image")
        let prepareStart = control.elapsed
        // The original, prepared exactly like the preview, and its [-1, 1] form.
        let originalRGB = try MiniPhotoGuardPixels.prepare(data)
        let originalPNG = try MiniPhotoGuardPixels.encode(originalRGB)
        guard try MiniPhotoGuardPixels.decodePNG(originalPNG) == originalRGB else { throw MiniPhotoGuardError.invalid("Prepared PNG pixel check failed.") }
        let original = try MiniPhotoGuardPixels.normalize(originalRGB)
        let preparationSeconds = control.elapsed - prepareStart
        let pipelineStart = control.elapsed
        var image = original
        // Each step uses the gradient from the previous model call, then calls the
        // model again on the moved image: 201 model calls in total.
        var evaluation = try evaluate(model, image: image)
        for step in 1...200 {
            try report("Running miniPhotoGuard", step - 1)
            try MiniPhotoGuardPixels.project(&image, original: original, gradient: evaluation.gradient)
            evaluation = try evaluate(model, image: image)
            try report("Running miniPhotoGuard", step)
        }
        // Back to bytes, to PNG, and a final check that the saved PNG decodes to
        // those bytes and that no value moved more than 6 levels.
        let rgb = try MiniPhotoGuardPixels.quantize(image, originalRGB: originalRGB)
        let protectedPNG = try MiniPhotoGuardPixels.encode(rgb)
        let decoded = try MiniPhotoGuardPixels.decodePNG(protectedPNG)
        guard decoded == rgb, zip(decoded, originalRGB).allSatisfy({ abs(Int($0.0)-Int($0.1)) <= 6 }) else {
            throw MiniPhotoGuardError.invalid("Saved PNG did not pass the miniPhotoGuard pixel check.")
        }
        try report("miniPhotoGuard image ready", 200)
        return LivePhotoResult(originalPNG: originalPNG, protectedPNG: protectedPNG, width: 256, height: 256,
            elapsedSeconds: control.elapsed, preparationSeconds: preparationSeconds, loadSeconds: loadSeconds,
            pipelineSeconds: control.elapsed - pipelineStart)
    }
}
