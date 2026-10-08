// Standalone host check with the real private Mini PhotoGuard model. Compile
// with -D MINI_PHOTOGUARD_MODEL_TEST, App/MiniPhotoGuardEngine.swift and this
// file, then run it with the path to app/ios/Models (run_host_tests.sh skips it
// when the model is not installed). It loads the model with Core ML on this Mac
// and checks the bulk output read against the per-element read bit for bit,
// and the compiled-model cache. No user photos; no protection efficacy evaluated.
#if MINI_PHOTOGUARD_MODEL_TEST
import Foundation
import CoreML

struct LivePhotoProgress: Sendable { let stage: String; let step: Int; let totalSteps: Int; let elapsedSeconds: Double }
struct LivePhotoResult: Sendable {
    let originalPNG: Data; let protectedPNG: Data; let width: Int; let height: Int
    let elapsedSeconds: Double; let preparationSeconds: Double; let loadSeconds: Double; let pipelineSeconds: Double
}

@main struct MiniPhotoGuardModelTests {
    static var checks = 0
    static func require(_ value: Bool, _ message: String) throws {
        checks += 1
        if !value { throw MiniPhotoGuardError.invalid("Test failed: " + message) }
    }
    static func bits(_ values: [Float]) -> [UInt32] { values.map(\.bitPattern) }
    static func now() -> Double { ProcessInfo.processInfo.systemUptime }
    static func entries(_ url: URL) -> [String] {
        ((try? FileManager.default.contentsOfDirectory(atPath: url.path)) ?? []).sorted()
    }
    static let outputs: [(String, [Int])] = [("loss", [1]), ("gradient", [1, 3, 256, 256]), ("latent_mean", [1, 4, 32, 32])]

    static func main() throws {
        let files = FileManager.default
        let models = URL(fileURLWithPath: CommandLine.arguments.dropFirst().first ?? "app/ios/Models", isDirectory: true)
        let package = models.appendingPathComponent("photoguard_encoder_gradient_256_fp32.mlpackage", isDirectory: true)
        let fixture = try Data(contentsOf: models.appendingPathComponent("MiniPhotoGuardPayload/fixture_input_f32.bin"))
        try require(fixture.count == MiniPhotoGuardPixels.count * 4, "fixture size")
        let input = try MLMultiArray(shape: [1, 3, 256, 256], dataType: .float32)
        let values = fixture.withUnsafeBytes { bytes in (0..<MiniPhotoGuardPixels.count).map {
            Float(bitPattern: UInt32(littleEndian: bytes.loadUnaligned(fromByteOffset: $0 * 4, as: UInt32.self))) } }
        values.withUnsafeBufferPointer { input.dataPointer.copyMemory(from: $0.baseAddress!, byteCount: values.count * 4) }
        let features = try MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(multiArray: input)])

        let scratch = files.temporaryDirectory.appendingPathComponent("MiniPhotoGuardModelTests-" + UUID().uuidString, isDirectory: true)
        defer { try? files.removeItem(at: scratch) }
        let root = scratch.appendingPathComponent("cache", isDirectory: true)
        let system = ProcessInfo.processInfo.operatingSystemVersionString
        let name = MiniPhotoGuardModelCache.folderName(manifestSHA: String(repeating: "a", count: 64),
                                                      exportSHA: String(repeating: "b", count: 64), system: system)
        try require(name.hasPrefix("aaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-") && !name.contains("/") && !name.contains(" "), "cache folder name")
        let folder = root.appendingPathComponent(name, isDirectory: true)
        let compiledModel = folder.appendingPathComponent(MiniPhotoGuardModelCache.compiledName, isDirectory: true)
        let configuration = MLModelConfiguration(); configuration.computeUnits = .cpuAndGPU

        // Compile once, then reuse.
        var start = now()
        let first = try MiniPhotoGuardModelCache.load(package: package, folder: folder, configuration: configuration) {}
        let firstLoad = now() - start
        try require(first.compiled && first.temporary == nil, "first load compiles into the cache")
        try require(files.fileExists(atPath: compiledModel.path), "compiled model is in its identity folder")
        try require(entries(root) == [name], "no staging folder is left behind")
        try require((try root.resourceValues(forKeys: [.isExcludedFromBackupKey])).isExcludedFromBackup == true, "cache excluded from backup")
        start = now()
        let second = try MiniPhotoGuardModelCache.load(package: package, folder: folder, configuration: configuration) {
            throw MiniPhotoGuardError.invalid("A cached load must not compile.")
        }
        let secondLoad = now() - start
        try require(!second.compiled && second.temporary == nil, "second load comes from the cache folder")
        try require(second.identity == first.identity && first.identity.hasPrefix(compiledModel.path), "cached load keeps the compiled-folder identity")

        // Bulk read equals the per-element read, bit for bit, for all outputs.
        let a = try first.model.prediction(from: features)
        let b = try second.model.prediction(from: features)
        var reference: [String: [UInt32]] = [:]
        for (output, shape) in outputs {
            let x = a.featureValue(for: output)!.multiArrayValue!, y = b.featureValue(for: output)!.multiArrayValue!
            try require(x.dataType == .float32 && x.shape.map(\.intValue) == shape, "\(output) shape and type")
            print("  \(output): shape \(shape) strides \(x.strides.map(\.intValue))")
            try require(MiniPhotoGuardOutput.isContiguousFloat32(x), "\(output) takes the bulk path")
            let old = bits(MiniPhotoGuardOutput.perElement(x))
            try require(MiniPhotoGuardOutput.bulk(x).map(bits) == old, "\(output) bulk read equals per-element read")
            try require(bits(MiniPhotoGuardOutput.floats(x)) == old, "\(output) engine read equals per-element read")
            try require(bits(MiniPhotoGuardOutput.floats(y)) == old, "\(output) from the cached load equals the first load")
            reference[output] = old
        }

        // Per-step read cost: all three outputs, old path vs new path.
        let arrays = outputs.map { a.featureValue(for: $0.0)!.multiArrayValue! }
        let repeats = 20
        var sink: Float = 0
        start = now()
        for _ in 0..<repeats { for array in arrays { sink += MiniPhotoGuardOutput.perElement(array).last! } }
        let oldRead = (now() - start) / Double(repeats)
        start = now()
        for _ in 0..<repeats { for array in arrays { sink += MiniPhotoGuardOutput.floats(array).last! } }
        let newRead = (now() - start) / Double(repeats)
        start = now()
        for _ in 0..<5 { sink += Float(try second.model.prediction(from: features).featureValue(for: "loss")!.multiArrayValue![0].floatValue) }
        let predict = (now() - start) / 5
        try require(sink.isFinite, "timing results used")

        // Any other layout or type keeps the per-element path.
        var storage = [Float](repeating: -7, count: 8)
        for row in 0..<2 { for column in 0..<3 { storage[row * 4 + column] = Float(row * 10 + column) } }
        try storage.withUnsafeMutableBytes { raw in
            let padded = try MLMultiArray(dataPointer: raw.baseAddress!, shape: [2, 3], dataType: .float32, strides: [4, 1])
            try require(!MiniPhotoGuardOutput.isContiguousFloat32(padded) && MiniPhotoGuardOutput.bulk(padded) == nil, "padded strides are not bulk-read")
            try require(bits(MiniPhotoGuardOutput.floats(padded)) == bits(MiniPhotoGuardOutput.perElement(padded)), "padded strides use the per-element path")
        }
        let double = try MLMultiArray(shape: [3], dataType: .double)
        for i in 0..<3 { double[i] = NSNumber(value: Double(i) + 0.1) }
        try require(MiniPhotoGuardOutput.bulk(double) == nil, "non-float32 is not bulk-read")
        try require(bits(MiniPhotoGuardOutput.floats(double)) == bits(MiniPhotoGuardOutput.perElement(double)), "non-float32 uses the per-element path")
        let special: [Float] = [-0.0, .leastNonzeroMagnitude, -.leastNormalMagnitude, .greatestFiniteMagnitude, 1.0 / 3.0, -2.5]
        let plain = try MLMultiArray(shape: [2, 3], dataType: .float32)
        special.withUnsafeBufferPointer { plain.dataPointer.copyMemory(from: $0.baseAddress!, byteCount: special.count * 4) }
        try require(MiniPhotoGuardOutput.bulk(plain).map(bits) == bits(special)
                    && bits(MiniPhotoGuardOutput.perElement(plain)) == bits(special), "signed zero and subnormals keep their bits")
        try require(MiniPhotoGuardOutput.contiguousStrides([1, 3, 256, 256]) == [196_608, 65_536, 256, 1], "row-major strides")

        // An unloadable cache is deleted and rebuilt once; stale entries are pruned.
        try files.removeItem(at: compiledModel)
        try files.createDirectory(at: compiledModel, withIntermediateDirectories: true)
        try Data("not a model".utf8).write(to: compiledModel.appendingPathComponent("coremldata.bin"))
        try files.createDirectory(at: root.appendingPathComponent(".staging-interrupted/model.mlmodelc"), withIntermediateDirectories: true)
        try files.createDirectory(at: root.appendingPathComponent("earlier-identity/model.mlmodelc"), withIntermediateDirectories: true)
        start = now()
        let rebuilt = try MiniPhotoGuardModelCache.load(package: package, folder: folder, configuration: configuration) {}
        let rebuildLoad = now() - start
        try require(rebuilt.compiled && rebuilt.temporary == nil, "unloadable cache is rebuilt into the cache")
        try require(rebuilt.identity != first.identity, "a rebuilt cache has a new identity, so parity runs again")
        try require(entries(root) == [name], "stale stagings and other identities are removed")
        let c = try rebuilt.model.prediction(from: features)
        for (output, _) in outputs {
            try require(bits(MiniPhotoGuardOutput.floats(c.featureValue(for: output)!.multiArrayValue!)) == reference[output]!, "\(output) after rebuild equals the first load")
        }

        // An unwritable cache falls back to a temporary compile the caller deletes.
        let blocker = scratch.appendingPathComponent("blocker")
        try Data().write(to: blocker)
        let fallback = try MiniPhotoGuardModelCache.load(package: package, folder: blocker.appendingPathComponent(name), configuration: configuration) {}
        try require(fallback.compiled && fallback.temporary != nil && files.fileExists(atPath: fallback.temporary!.path), "unwritable cache uses a temporary compile")
        try require(fallback.identity.hasPrefix(fallback.temporary!.path), "temporary compile identity")
        let d = try fallback.model.prediction(from: features)
        for (output, _) in outputs {
            try require(bits(MiniPhotoGuardOutput.floats(d.featureValue(for: output)!.multiArrayValue!)) == reference[output]!, "\(output) from the temporary compile equals the cache")
        }
        try files.removeItem(at: fallback.temporary!)
        let cancelled = MiniPhotoGuardError.invalid("cancelled")
        do {
            _ = try MiniPhotoGuardModelCache.load(package: package, folder: blocker.appendingPathComponent(name), configuration: configuration) { throw cancelled }
            try require(false, "a failed check after compiling propagates")
        } catch MiniPhotoGuardError.invalid(let message) where message == "cancelled" { checks += 1 }

        // Install markers change with a file's modification date.
        let stamped = scratch.appendingPathComponent("stamped.bin")
        try Data(count: 16).write(to: stamped)
        let before = MiniPhotoGuardMarkers.stamp(stamped)
        try files.setAttributes([.modificationDate: Date(timeIntervalSinceNow: -3600)], ofItemAtPath: stamped.path)
        try require(MiniPhotoGuardMarkers.stamp(stamped) != before && before.hasPrefix("16:"), "stamp has size and modification date")
        try require(MiniPhotoGuardMarkers.stamp(nil) == "unavailable" && MiniPhotoGuardMarkers.stamp(scratch.appendingPathComponent("missing")) == "unavailable", "missing files stamp as unavailable")
        try require(MiniPhotoGuardMarkers.digest(["a", "b"]) == MiniPhotoGuardMarkers.digest(["a", "b"])
                    && MiniPhotoGuardMarkers.digest(["a", "b"]) != MiniPhotoGuardMarkers.digest(["b", "a"]), "marker digest")

        print(String(format: "  compile+load %.2f s, cached load %.2f s, rebuild %.2f s, prediction %.3f s", firstLoad, secondLoad, rebuildLoad, predict))
        print(String(format: "  output read per step: per-element %.2f ms, bulk %.3f ms", oldRead * 1000, newRead * 1000))
        print("MiniPhotoGuardModelTests: \(checks) checks passed with the real model on this Mac; no protection efficacy evaluated.")
    }
}
#endif
