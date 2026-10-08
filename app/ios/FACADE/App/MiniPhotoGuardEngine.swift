import Foundation
import CoreML
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import CryptoKit
import Darwin

// Experimental encoder-only PhotoGuard adapter. This is the frozen 256 FP32
// gray-latent objective, not the full diffusion method or a face verifier.
// Caller must serialize this actor with all other heavy model engines and join
// cancellation before starting another engine. No loaded model survives a run;
// only the compiled model files are kept on disk for reuse (see below).
enum MiniPhotoGuardError: LocalizedError {
    case invalid(String)
    var errorDescription: String? { if case .invalid(let message) = self { return message }; return nil }
}

private final class MiniPhotoGuardControl: @unchecked Sendable {
    private let lock = NSLock()
    private var reason: String?
    let started = ProcessInfo.processInfo.systemUptime
    var elapsed: Double { ProcessInfo.processInfo.systemUptime - started }
    func stop(_ message: String) { lock.lock(); defer { lock.unlock() }; if reason == nil { reason = message } }
    func check() throws {
        if Task.isCancelled { stop("miniPhotoGuard was cancelled.") }
        if elapsed > 600 { stop("miniPhotoGuard reached its 10-minute time limit.") }
        let thermal = ProcessInfo.processInfo.thermalState
        if thermal == .serious || thermal == .critical { stop("Let this device cool before trying miniPhotoGuard again.") }
        if Self.memoryBytes() > 3_000_000_000 { stop("miniPhotoGuard reached its memory limit.") }
        lock.lock(); let message = reason; lock.unlock()
        if let message { throw MiniPhotoGuardError.invalid(message + " Elapsed: \(String(format: "%.1f", elapsed)) seconds.") }
    }
    static func memoryBytes() -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let status = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return status == KERN_SUCCESS ? info.phys_footprint : UInt64.max
    }
}

// Pure raster/projection helpers shared by the adapter and the focused tests.
// Byte order is RGB interleaved; model order is NCHW, normalized to [-1, 1].
enum MiniPhotoGuardPixels {
    static let side = 256
    static let count = side * side * 3
    static let epsilon: Float = 12.0 / 255.0
    static let step: Float = 2.0 / 255.0

    static func normalize(_ rgb: [UInt8]) throws -> [Float] {
        guard rgb.count == count else { throw MiniPhotoGuardError.invalid("Invalid miniPhotoGuard raster.") }
        let area = side * side
        var result = [Float](repeating: 0, count: count)
        for pixel in 0..<area { for channel in 0..<3 {
            result[channel * area + pixel] = Float(rgb[pixel * 3 + channel]) * (2.0 / 255.0) - 1
        } }
        return result
    }
    static func project(_ image: inout [Float], original: [Float], gradient: [Float]) throws {
        guard image.count == count, original.count == count, gradient.count == count else {
            throw MiniPhotoGuardError.invalid("Invalid miniPhotoGuard gradient shape.")
        }
        for i in image.indices {
            guard image[i].isFinite, original[i].isFinite, (-1...1).contains(original[i]), gradient[i].isFinite else {
                throw MiniPhotoGuardError.invalid("Nonfinite miniPhotoGuard calculation.")
            }
            let sign: Float = gradient[i] > 0 ? 1 : (gradient[i] < 0 ? -1 : 0)
            image[i] = min(min(1, original[i] + epsilon), max(max(-1, original[i] - epsilon), image[i] - step * sign))
        }
    }
    static func quantize(_ image: [Float], originalRGB: [UInt8]) throws -> [UInt8] {
        let original = try normalize(originalRGB)
        guard image.count == count else { throw MiniPhotoGuardError.invalid("Invalid output shape.") }
        let area = side * side
        var rgb = originalRGB
        for pixel in 0..<area { for channel in 0..<3 {
            let i = channel * area + pixel
            let value = image[i]
            guard value.isFinite, (-1...1).contains(value), abs(value - original[i]) <= epsilon + Float.ulpOfOne else {
                throw MiniPhotoGuardError.invalid("miniPhotoGuard projection check failed.")
            }
            let quantized = Int(((value + 1) * 127.5).rounded())
            guard (0...255).contains(quantized), abs(quantized - Int(originalRGB[pixel * 3 + channel])) <= 6 else {
                throw MiniPhotoGuardError.invalid("miniPhotoGuard PNG pixel bound failed.")
            }
            rgb[pixel * 3 + channel] = UInt8(quantized)
        } }
        return rgb
    }
    static func encode(_ rgb: [UInt8]) throws -> Data {
        guard rgb.count == count, let space = CGColorSpace(name: CGColorSpace.sRGB),
              let provider = CGDataProvider(data: Data(rgb) as CFData),
              let image = CGImage(width: side, height: side, bitsPerComponent: 8, bitsPerPixel: 24,
                bytesPerRow: side * 3, space: space, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent) else {
            throw MiniPhotoGuardError.invalid("Could not create the miniPhotoGuard image.")
        }
        let output = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(output, UTType.png.identifier as CFString, 1, nil) else {
            throw MiniPhotoGuardError.invalid("Could not encode the miniPhotoGuard PNG.")
        }
        CGImageDestinationAddImage(destination, image, [:] as CFDictionary)
        guard CGImageDestinationFinalize(destination) else { throw MiniPhotoGuardError.invalid("PNG encoding failed.") }
        return output as Data
    }
    static func raster(_ image: CGImage, fit: Bool) throws -> [UInt8] {
        guard let space = CGColorSpace(name: CGColorSpace.sRGB) else { throw MiniPhotoGuardError.invalid("sRGB unavailable.") }
        var rgba = [UInt8](repeating: 255, count: side * side * 4)
        let success = rgba.withUnsafeMutableBytes { bytes -> Bool in
            guard let context = CGContext(data: bytes.baseAddress, width: side, height: side, bitsPerComponent: 8,
                bytesPerRow: side * 4, space: space,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue) else { return false }
            context.setFillColor(CGColor(colorSpace: space, components: [128.0/255, 128.0/255, 128.0/255, 1])!)
            context.fill(CGRect(x: 0, y: 0, width: side, height: side))
            let scale = min(Double(side) / Double(image.width), Double(side) / Double(image.height))
            let width = fit ? max(1, Int((Double(image.width) * scale).rounded())) : side
            let height = fit ? max(1, Int((Double(image.height) * scale).rounded())) : side
            context.interpolationQuality = fit ? .high : .none
            context.draw(image, in: CGRect(x: (side - width)/2, y: (side - height)/2, width: width, height: height))
            return true
        }
        guard success else { throw MiniPhotoGuardError.invalid("Could not prepare the miniPhotoGuard raster.") }
        var rgb = [UInt8](repeating: 0, count: count)
        for pixel in 0..<(side * side) { for channel in 0..<3 { rgb[pixel*3+channel] = rgba[pixel*4+channel] } }
        return rgb
    }
    static func prepare(_ data: Data) throws -> [UInt8] {
        guard !data.isEmpty, data.count <= 32 * 1024 * 1024,
              let source = CGImageSourceCreateWithData(data as CFData, [kCGImageSourceShouldCache: false] as CFDictionary),
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let width = properties[kCGImagePropertyPixelWidth] as? Int,
              let height = properties[kCGImagePropertyPixelHeight] as? Int,
              width > 0, height > 0, width <= 32_768, height <= 32_768,
              Double(width) * Double(height) <= 160_000_000 else {
            throw MiniPhotoGuardError.invalid("This photo could not be opened within miniPhotoGuard's image limits.")
        }
        let options: [CFString: Any] = [kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true, kCGImageSourceThumbnailMaxPixelSize: side,
            kCGImageSourceShouldCacheImmediately: true]
        guard let image = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary) else {
            throw MiniPhotoGuardError.invalid("This photo could not be prepared.")
        }
        return try raster(image, fit: true)
    }
    static func decodePNG(_ data: Data) throws -> [UInt8] {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil), CGImageSourceGetCount(source) == 1,
              CGImageSourceGetType(source) as String? == UTType.png.identifier,
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil), image.width == side, image.height == side else {
            throw MiniPhotoGuardError.invalid("The saved miniPhotoGuard PNG has the wrong format.")
        }
        return try raster(image, fit: false)
    }
}

// Reads a Core ML output as [Float]. A float32 array laid out in contiguous
// row-major order is copied in bulk; that yields the same IEEE bits as the
// per-element NSNumber read, which remains the path for every other layout.
enum MiniPhotoGuardOutput {
    static func contiguousStrides(_ shape: [Int]) -> [Int] {
        var strides = [Int](repeating: 1, count: shape.count)
        var running = 1
        for axis in shape.indices.reversed() { strides[axis] = running; running *= shape[axis] }
        return strides
    }
    static func isContiguousFloat32(_ array: MLMultiArray) -> Bool {
        array.dataType == .float32 && array.strides.map(\.intValue) == contiguousStrides(array.shape.map(\.intValue))
    }
    static func perElement(_ array: MLMultiArray) -> [Float] { (0..<array.count).map { array[$0].floatValue } }
    static func bulk(_ array: MLMultiArray) -> [Float]? {
        guard isContiguousFloat32(array) else { return nil }
        let count = array.count
        return array.withUnsafeBufferPointer(ofType: Float.self) { buffer -> [Float]? in
            guard let base = buffer.baseAddress, buffer.count >= count else { return nil }
            return Array(UnsafeBufferPointer(start: base, count: count))
        }
    }
    static func floats(_ array: MLMultiArray) -> [Float] { bulk(array) ?? perElement(array) }
}

// Compiled-model cache. The bundled .mlpackage is compiled once per verified
// model identity and OS build and kept in Caches (excluded from backup; iOS may
// purge it when storage is low, which only costs one recompile); the stable
// path also lets Core ML reuse its own device cache. Only a
// complete compile is ever found under an identity folder: the compiler output
// is staged beside it and renamed into place. A cache that fails to load is
// deleted and rebuilt once. If the cache cannot be written, the run compiles to
// a temporary folder that the caller deletes afterwards, as before.
enum MiniPhotoGuardModelCache {
    static let compiledName = "model.mlmodelc"
    // `identity` names the compiled folder that was loaded; every compile gets a new one.
    struct Loaded { let model: MLModel; let compiled: Bool; let temporary: URL?; let identity: String }

    static func identity(_ url: URL) -> String {
        let attributes = try? FileManager.default.attributesOfItem(atPath: url.path)
        let number = (attributes?[.systemFileNumber] as? NSNumber)?.uint64Value ?? 0
        let created = (attributes?[.creationDate] as? Date)?.timeIntervalSinceReferenceDate.bitPattern ?? 0
        return "\(url.path):\(number):\(created)"
    }

    static func root() -> URL? {
        try? FileManager.default.url(for: .cachesDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
            .appendingPathComponent("MiniPhotoGuardCompiledModel", isDirectory: true)
    }
    static func folderName(manifestSHA: String, exportSHA: String, system: String) -> String {
        let os = String(String.UnicodeScalarView(system.unicodeScalars.map {
            ($0.isASCII && (CharacterSet.alphanumerics.contains($0) || $0 == ".")) ? $0 : "_" }))
        return "\(manifestSHA.prefix(16))-\(exportSHA.prefix(16))-\(os)"
    }
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
        let compiled = try MLModel.compileModel(at: package)
        let published = folder.flatMap { publish(compiled, to: $0) }
        let temporary = published == nil ? compiled : nil
        do { try afterCompile() } catch {
            if let temporary { try? files.removeItem(at: temporary) }
            throw error
        }
        do {
            let model = try MLModel(contentsOf: published ?? compiled, configuration: configuration)
            return Loaded(model: model, compiled: true, temporary: temporary, identity: identity(published ?? compiled))
        } catch {
            if let temporary { try? files.removeItem(at: temporary) } else if let folder { try? files.removeItem(at: folder) }
            throw error
        }
    }
    // Returns the published model, or nil with the compiler output left in place.
    private static func publish(_ compiled: URL, to folder: URL) -> URL? {
        let files = FileManager.default
        var root = folder.deletingLastPathComponent()
        let staging = root.appendingPathComponent(".staging-" + UUID().uuidString, isDirectory: true)
        let staged = staging.appendingPathComponent(compiledName, isDirectory: true)
        do {
            try files.createDirectory(at: root, withIntermediateDirectories: true)
            var values = URLResourceValues(); values.isExcludedFromBackup = true
            try root.setResourceValues(values)
            // Earlier model identities or OS builds, and interrupted stagings.
            for entry in try files.contentsOfDirectory(at: root, includingPropertiesForKeys: nil) { try? files.removeItem(at: entry) }
            try files.createDirectory(at: staging, withIntermediateDirectories: false)
            try files.moveItem(at: compiled, to: staged)
        } catch { try? files.removeItem(at: staging); return nil }
        do { try files.moveItem(at: staging, to: folder) } catch { // Same-directory rename: all or nothing.
            try? files.moveItem(at: staged, to: compiled)
            try? files.removeItem(at: staging)
            return nil
        }
        return folder.appendingPathComponent(compiledName, isDirectory: true)
    }
}

// Once-per-install records in UserDefaults: digests of the build numbers, the
// bundle and executable, and each bound file's size and modification date. A
// mismatch only repeats the full check; a failed check never records anything.
enum MiniPhotoGuardMarkers {
    static let verifiedInstall = "MiniPhotoGuard.verifiedInstall.v1"
    static let passedParity = "MiniPhotoGuard.passedParity.v1"
    static func digest(_ parts: [String]) -> String {
        SHA256.hash(data: Data(parts.joined(separator: "\n").utf8)).map { String(format: "%02x", $0) }.joined()
    }
    static func stamp(_ url: URL?) -> String {
        guard let url, let attributes = try? FileManager.default.attributesOfItem(atPath: url.path),
              let size = attributes[.size] as? NSNumber, let date = attributes[.modificationDate] as? Date else { return "unavailable" }
        return "\(size.int64Value):\(date.timeIntervalSinceReferenceDate.bitPattern)"
    }
}

actor MiniPhotoGuardEngine {
    static let shared = MiniPhotoGuardEngine()
    private var busy = false
    private init() {}
    private static let exportSHA = "b474f21b584feec1f003b88634e5b13d6444c3d205e845108e10783bc0a7730e"
    private static let frozenManifestSHA = "174bf74e484ecfa546ef108fffe835080cba4a64fd5cff931f3dca883412036d"
    private struct Binding: Decodable { let path: String; let sha256: String; let bytes: Int }
    private struct Manifest: Decodable {
        let schemaVersion: Int; let side: Int; let modelRelativePath: String; let modelFiles: [Binding]
        let fixtures: [String: Binding]; let exportSHA256: String
        let epsilonPixels: Double; let stepPixels: Double; let maxSteps: Int
        let gradientCosineMinimum: Double; let gradientRelativeL2Maximum: Double; let latentRelativeL2Maximum: Double
        let lossAbsoluteTolerance: Double; let lossRelativeTolerance: Double
    }
    private struct Payload { let manifest: Manifest; let root: URL; let model: URL }
    private struct Evaluation { let loss: Float; let gradient: [Float]; let latent: [Float] }

    nonisolated static var resourcesAvailable: Bool { (try? payload(fullVerification: false)) != nil }
    nonisolated static func preparePhoto(data: Data) throws -> (png: Data, width: Int, height: Int) {
        let rgb = try MiniPhotoGuardPixels.prepare(data)
        let png = try MiniPhotoGuardPixels.encode(rgb)
        guard try MiniPhotoGuardPixels.decodePNG(png) == rgb else { throw MiniPhotoGuardError.invalid("Prepared PNG pixel check failed.") }
        return (png, 256, 256)
    }
    func run(data: Data, progress: @escaping @Sendable (LivePhotoProgress) -> Void) async throws -> LivePhotoResult {
        guard !busy else { throw MiniPhotoGuardError.invalid("miniPhotoGuard is already running.") }
        busy = true
        defer { busy = false }
        let control = MiniPhotoGuardControl()
        try control.check()
        let worker = Task.detached(priority: .userInitiated) {
            try autoreleasepool { try Self.execute(data: data, control: control, progress: progress) }
        }
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
    private nonisolated static func safe(_ relative: String, root: URL) throws -> URL {
        guard !relative.isEmpty, !relative.hasPrefix("/"), !relative.contains("\\"),
              relative.split(separator: "/", omittingEmptySubsequences: false).allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." }) else {
            throw MiniPhotoGuardError.invalid("Invalid miniPhotoGuard resource path.")
        }
        let result = root.appendingPathComponent(relative).standardizedFileURL
        guard result.resolvingSymlinksInPath().path == result.path else { throw MiniPhotoGuardError.invalid("Symlinked miniPhotoGuard resource.") }
        return result
    }
    private nonisolated static func digest(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
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
            while let data = try handle.read(upToCount: 4 * 1024 * 1024), !data.isEmpty { hash.update(data: data) }
            guard hash.finalize().map({ String(format: "%02x", $0) }).joined() == binding.sha256 else {
                throw MiniPhotoGuardError.invalid("miniPhotoGuard resource hash failed.")
            }
        }
        return url
    }
    private nonisolated static func payload(fullVerification: Bool) throws -> Payload {
        try payload(full: { _ in fullVerification }).payload
    }
    // `full` receives this install's key and decides whether to hash every file.
    private nonisolated static func payload(full: (String) -> Bool) throws -> (payload: Payload, install: String) {
        // The payload is private and not in git; see app/ios/Models/README.md.
        let root = Bundle.main.bundleURL.appendingPathComponent("MiniPhotoGuardPayload")
        guard let data = try? Data(contentsOf: root.appendingPathComponent("manifest.json")) else {
            throw MiniPhotoGuardError.invalid("The miniPhotoGuard model is not installed in this build.")
        }
        guard digest(data) == frozenManifestSHA else {
            throw MiniPhotoGuardError.invalid("The verified miniPhotoGuard model is not installed.")
        }
        let manifest = try JSONDecoder().decode(Manifest.self, from: data)
        guard manifest.schemaVersion == 1, manifest.side == 256, manifest.exportSHA256 == exportSHA,
              manifest.modelRelativePath == "photoguard_encoder_gradient_256_fp32.mlpackage",
              manifest.epsilonPixels == 6, manifest.stepPixels == 1, manifest.maxSteps == 200,
              Set(manifest.fixtures.keys) == Set(["input", "loss", "gradient", "latent_mean"]),
              !manifest.modelFiles.isEmpty, Set(manifest.modelFiles.map(\.path)).count == manifest.modelFiles.count else {
            throw MiniPhotoGuardError.invalid("The miniPhotoGuard recipe differs from the verified model.")
        }
        let model = try safe(manifest.modelRelativePath, root: Bundle.main.bundleURL)
        let info = Bundle.main.infoDictionary
        var parts = [Bundle.main.bundleURL.path, MiniPhotoGuardMarkers.stamp(Bundle.main.executableURL),
                     info?["CFBundleVersion"] as? String ?? "", info?["CFBundleShortVersionString"] as? String ?? "", frozenManifestSHA]
        for binding in manifest.modelFiles { parts.append(binding.path + "=" + MiniPhotoGuardMarkers.stamp(try? safe(binding.path, root: model))) }
        for (name, binding) in manifest.fixtures.sorted(by: { $0.key < $1.key }) {
            parts.append(name + "=" + MiniPhotoGuardMarkers.stamp(try? safe(binding.path, root: root)))
        }
        let install = MiniPhotoGuardMarkers.digest(parts)
        let fullVerification = full(install)
        for binding in manifest.modelFiles { _ = try verify(binding, root: model, full: fullVerification) }
        for binding in manifest.fixtures.values { _ = try verify(binding, root: root, full: fullVerification) }
        return (Payload(manifest: manifest, root: root, model: model), install)
    }
    private nonisolated static func floats(_ binding: Binding, root: URL, count: Int) throws -> [Float] {
        let data = try Data(contentsOf: safe(binding.path, root: root))
        guard data.count == count * 4 else { throw MiniPhotoGuardError.invalid("Invalid fixture shape.") }
        let result = data.withUnsafeBytes { bytes in (0..<count).map { Float(bitPattern: UInt32(littleEndian: bytes.loadUnaligned(fromByteOffset: $0*4, as: UInt32.self))) } }
        guard result.allSatisfy(\.isFinite) else { throw MiniPhotoGuardError.invalid("Nonfinite reference fixture.") }
        return result
    }
    private nonisolated static func evaluate(_ model: MLModel, image: [Float]) throws -> Evaluation {
        try autoreleasepool {
            guard image.count == MiniPhotoGuardPixels.count, image.allSatisfy({ $0.isFinite && (-1...1).contains($0) }) else { throw MiniPhotoGuardError.invalid("Invalid model input.") }
            let array = try MLMultiArray(shape: [1, 3, 256, 256], dataType: .float32)
            image.withUnsafeBufferPointer { array.dataPointer.copyMemory(from: $0.baseAddress!, byteCount: image.count * 4) }
            let output = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(multiArray: array)]))
            func values(_ name: String, _ shape: [Int]) throws -> [Float] {
                guard let array = output.featureValue(for: name)?.multiArrayValue, array.dataType == .float32,
                      array.shape.map(\.intValue) == shape else { throw MiniPhotoGuardError.invalid("Invalid model output: \(name).") }
                let values = MiniPhotoGuardOutput.floats(array)
                guard values.allSatisfy(\.isFinite) else { throw MiniPhotoGuardError.invalid("Nonfinite model output.") }
                return values
            }
            return try Evaluation(loss: values("loss", [1])[0], gradient: values("gradient", [1,3,256,256]), latent: values("latent_mean", [1,4,32,32]))
        }
    }
    private nonisolated static func norm(_ values: [Float]) -> Double { sqrt(values.reduce(0) { $0 + Double($1) * Double($1) }) }
    private nonisolated static func relative(_ a: [Float], _ b: [Float]) -> Double {
        sqrt(zip(a,b).reduce(0) { $0 + pow(Double($1.0) - Double($1.1), 2) }) / max(norm(b), 1e-30)
    }
    private nonisolated static func execute(data: Data, control: MiniPhotoGuardControl,
        progress: @escaping @Sendable (LivePhotoProgress) -> Void) throws -> LivePhotoResult {
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
            let cosine = zip(observed.gradient, expectedGradient).reduce(0.0) { $0 + Double($1.0)*Double($1.1) }
                / max(norm(observed.gradient)*norm(expectedGradient), 1e-30)
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
        let originalRGB = try MiniPhotoGuardPixels.prepare(data)
        let originalPNG = try MiniPhotoGuardPixels.encode(originalRGB)
        guard try MiniPhotoGuardPixels.decodePNG(originalPNG) == originalRGB else { throw MiniPhotoGuardError.invalid("Prepared PNG pixel check failed.") }
        let original = try MiniPhotoGuardPixels.normalize(originalRGB)
        let preparationSeconds = control.elapsed - prepareStart
        let pipelineStart = control.elapsed
        var image = original
        var evaluation = try evaluate(model, image: image)
        for step in 1...200 {
            try report("Running miniPhotoGuard", step - 1)
            try MiniPhotoGuardPixels.project(&image, original: original, gradient: evaluation.gradient)
            evaluation = try evaluate(model, image: image)
            try report("Running miniPhotoGuard", step)
        }
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
