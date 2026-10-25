// MiniPhotoGuardModelTests.swift
//
// Tests the parts of Mini PhotoGuard that need the real Core ML model: how the
// model's outputs are read, and how the compiled model is cached on disk.
//
// Where it sits: run_host_tests.sh runs this last, and only if the private
// model is installed in app/ios/Models (see app/ios/Models/README.md).
//
// Key ideas:
// 1. The engine reads model outputs with a fast bulk copy when it can. This
//    test proves, bit for bit, that the fast read gives exactly the same
//    numbers as the slow element-by-element read, and that unusual layouts
//    fall back to the slow read.
// 2. It walks the compiled-model cache through its cases: first compile,
//    cached load, a broken cache that must be rebuilt, and a cache folder that
//    cannot be written.
// 3. It also prints timings. It does not measure whether protection works.
//
// Build: compile with -D MINI_PHOTOGUARD_MODEL_TEST, App/MiniPhotoGuardEngine.swift
// and this file, then run it with the path to app/ios/Models. No user photos
// are used.
//
// Compiled only with the test flag, so it never ends up in the app.
#if MINI_PHOTOGUARD_MODEL_TEST
import Foundation
import CoreML

// Copies of the two types from LivePhotoEngine.swift that the engine file needs.
struct LivePhotoProgress: Sendable { let stage: String; let step: Int; let totalSteps: Int; let elapsedSeconds: Double }
struct LivePhotoResult: Sendable {
    let originalPNG: Data; let protectedPNG: Data; let width: Int; let height: Int
    let elapsedSeconds: Double; let preparationSeconds: Double; let loadSeconds: Double; let pipelineSeconds: Double
}

/// The test program. Its one argument is the path to app/ios/Models.
@main struct MiniPhotoGuardModelTests {
    /// Passing-check counter and the check helper (throws on the first failure).
    static var checks = 0
    static func require(_ value: Bool, _ message: String) throws {
        checks += 1
        if !value { throw MiniPhotoGuardError.invalid("Test failed: " + message) }
    }
    /// Raw IEEE bit patterns, so comparisons are exact (they even tell -0.0 from 0.0).
    static func bits(_ values: [Float]) -> [UInt32] { values.map(\.bitPattern) }
    /// Seconds since boot, for timings.
    static func now() -> Double { ProcessInfo.processInfo.systemUptime }
    /// Sorted names in a folder, used to check that nothing extra is left behind.
    static func entries(_ url: URL) -> [String] {
        ((try? FileManager.default.contentsOfDirectory(atPath: url.path)) ?? []).sorted()
    }
    /// The model's three outputs and their shapes.
    static let outputs: [(String, [Int])] = [("loss", [1]), ("gradient", [1, 3, 256, 256]), ("latent_mean", [1, 4, 32, 32])]

    /// Runs every check in order.
    static func main() throws {
        let files = FileManager.default
        let models = URL(fileURLWithPath: CommandLine.arguments.dropFirst().first ?? "app/ios/Models", isDirectory: true)
        let package = models.appendingPathComponent("photoguard_encoder_gradient_256_fp32.mlpackage", isDirectory: true)
        // The stored reference input from the model bundle: 196,608 little-endian floats.
        let fixture = try Data(contentsOf: models.appendingPathComponent("MiniPhotoGuardPayload/fixture_input_f32.bin"))
        try require(fixture.count == MiniPhotoGuardPixels.count * 4, "fixture size")
        let input = try MLMultiArray(shape: [1, 3, 256, 256], dataType: .float32)
        let values = fixture.withUnsafeBytes { bytes in (0..<MiniPhotoGuardPixels.count).map {
            Float(bitPattern: UInt32(littleEndian: bytes.loadUnaligned(fromByteOffset: $0 * 4, as: UInt32.self))) } }
        values.withUnsafeBufferPointer { input.dataPointer.copyMemory(from: $0.baseAddress!, byteCount: values.count * 4) }
        // The model's one input is named "image".
        let features = try MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(multiArray: input)])

        // Everything this test writes goes in a throwaway folder that is deleted at the end.
        let scratch = files.temporaryDirectory.appendingPathComponent("MiniPhotoGuardModelTests-" + UUID().uuidString, isDirectory: true)
        defer { try? files.removeItem(at: scratch) }
        let root = scratch.appendingPathComponent("cache", isDirectory: true)
        let system = ProcessInfo.processInfo.operatingSystemVersionString
        // Fake hashes ("aaa...", "bbb...") make the expected folder name easy to check.
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
        // The callback only runs after a compile, so throwing here proves no compile happened.
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
        // `sink` keeps the compiler from skipping work whose result is never used.
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
        // A 2 x 3 array stored with rows 4 apart (one padding slot per row):
        // not tightly packed, so it must not be bulk-copied.
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
        // Edge-case floats (negative zero, tiny subnormals, the largest float) must
        // survive both read paths unchanged.
        let special: [Float] = [-0.0, .leastNonzeroMagnitude, -.leastNormalMagnitude, .greatestFiniteMagnitude, 1.0 / 3.0, -2.5]
        let plain = try MLMultiArray(shape: [2, 3], dataType: .float32)
        special.withUnsafeBufferPointer { plain.dataPointer.copyMemory(from: $0.baseAddress!, byteCount: special.count * 4) }
        try require(MiniPhotoGuardOutput.bulk(plain).map(bits) == bits(special)
                    && bits(MiniPhotoGuardOutput.perElement(plain)) == bits(special), "signed zero and subnormals keep their bits")
        try require(MiniPhotoGuardOutput.contiguousStrides([1, 3, 256, 256]) == [196_608, 65_536, 256, 1], "row-major strides")

        // An unloadable cache is deleted and rebuilt once; stale entries are pruned.
        // Replace the cached model with junk and add two stale folders; loading must
        // rebuild the cache and remove the stale folders.
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
        // A plain file named "blocker" cannot contain folders, so the cache cannot
        // be created under it and the loader must fall back to a temporary compile.
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
        // An error thrown by the after-compile callback (as a cancel would) must
        // reach the caller instead of being swallowed.
        let cancelled = MiniPhotoGuardError.invalid("cancelled")
        do {
            _ = try MiniPhotoGuardModelCache.load(package: package, folder: blocker.appendingPathComponent(name), configuration: configuration) { throw cancelled }
            try require(false, "a failed check after compiling propagates")
        } catch MiniPhotoGuardError.invalid(let message) where message == "cancelled" { checks += 1 }

        // Install markers change with a file's modification date.
        let stamped = scratch.appendingPathComponent("stamped.bin")
        try Data(count: 16).write(to: stamped)
        let before = MiniPhotoGuardMarkers.stamp(stamped)
        // Changing only the modification date must change the stamp.
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
