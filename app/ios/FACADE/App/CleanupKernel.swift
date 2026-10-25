// CleanupKernel.swift
//
// The "tight cleanup" step at the end of the FACADE engine. After the
// optimiser has changed the photo, this step keeps those changes only on the
// person and in a narrow band around them, and puts the original pixels back
// everywhere else.
//
// Where it sits: NativePipeline calls `TightCleanup.alpha` with the subject
// mask produced by the face parser, then `TightCleanup.composite` to blend the
// protected image back into the clean one. `TightCleanup.sha` is also used by
// NativePreparationLoader and NativeTensor to check model file checksums.
//
// Key ideas:
// 1. `alpha` is a per-pixel weight from 0 to 1. It is 1 on the subject and
//    within `margin` pixels of it, then fades smoothly to 0 over the next
//    `feather` pixels, so there is no hard edge where the change stops.
// 2. Distances come from an exact Euclidean distance transform (the
//    Felzenszwalb-Huttenlocher method): one pass along every row, then one
//    pass along every column, each in linear time.
// 3. The blend rounds half up in Double precision so its output matches,
//    byte for byte, the reference images it was checked against.
// 4. The fixture types and `check` below re-run the kernel against stored
//    reference files. The app's protection path does not call them.
import Foundation
import CryptoKit

/// One stored test case for the cleanup kernel: a folder of raw input files,
/// their expected SHA-256 checksums, and the checksum of the correct output.
struct CandidateFixture: Codable, Identifiable {
    let id: String
    /// Folder name, relative to the root passed to `check`.
    let folder: String
    /// Image size in pixels.
    let width: Int
    let height: Int
    /// File name -> expected SHA-256 (hex). Every file is checked before use.
    let files: [String: String]
    /// SHA-256 of the correct cleaned-up RGB output.
    let tightRGBSHA256: String
}
/// A list of cleanup fixtures plus the margin and feather they were made with.
struct CandidateManifest: Codable {
    let candidate: String
    let sourceCommit: String
    let scope: String
    /// Pixels of full-strength band around the subject, then the fade width.
    let margin: Float
    let feather: Float
    let fixtures: [CandidateFixture]
}
/// Errors from this file. The text is shown as-is, so it is kept short.
enum CleanupError: Error, LocalizedError {
    case invalid(String)
    var errorDescription: String? { if case .invalid(let text) = self { return text }; return nil }
}

// Exact cleanup kernel; the native pipeline supplies its clean-image subject mask.
/// Namespace for the cleanup functions. An enum with no cases cannot be
/// created, so it works as a container for static functions.
enum TightCleanup {
    /// Lower-case hex SHA-256 of `data`. Used here for fixtures and elsewhere for
    /// model files.
    static func sha(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    // Lower envelope of squared-distance parabolas. Finite sentinel avoids
    // infinity subtraction for rows containing no subject pixels.
    /// One-dimensional squared distance transform.
    ///
    /// For each position q it returns min over v of (q - v)^2 + f[v]. Each v
    /// contributes a parabola centred at v and lifted by f[v]; the answer is the
    /// lowest parabola at q (the "lower envelope").
    /// - `sites[0...k]` are the centres of the parabolas that make up the envelope.
    /// - `boundaries[i]` is where parabola `sites[i]` starts being the lowest one.
    /// The first loop builds the envelope; the second reads it off left to right.
    private static func distanceLine(_ f: [Double]) -> [Double] {
        let n = f.count
        var sites = [Int](repeating: 0, count: n)
        var boundaries = [Double](repeating: 0, count: n + 1)
        var out = [Double](repeating: 0, count: n)
        var k = 0
        boundaries[0] = -.infinity; boundaries[1] = .infinity
        for q in 1..<n {
            var s: Double
            repeat {
                let v = sites[k]
                // x-coordinate where the parabola at q crosses the parabola at v.
                s = ((f[q] + Double(q*q)) - (f[v] + Double(v*v))) / Double(2*(q-v))
                // If the crossing is right of where v starts, v keeps part of the envelope.
                // Otherwise v is completely hidden by q, so drop it and try the previous one.
                if s > boundaries[k] || k == 0 { break }
                k -= 1
            } while true
            // Add q as the new right-most parabola of the envelope.
            k += 1; sites[k] = q; boundaries[k] = s; boundaries[k+1] = .infinity
        }
        // Walk left to right, moving to the next parabola once q passes its start.
        k = 0
        for q in 0..<n {
            while boundaries[k+1] < Double(q) { k += 1 }
            let d = q-sites[k]
            out[q] = Double(d*d) + f[sites[k]]
        }
        return out
    }

    /// The blend weight for every pixel, row-major, one Float per pixel.
    ///
    /// - `subject`: one byte per pixel, nonzero = part of the person.
    /// - `margin`: distance in pixels over which the weight stays at 1.
    /// - `feather`: distance in pixels over which it then fades to 0.
    /// Throws if the mask is empty, the wrong size, or the geometry is invalid, so
    /// the caller never blends with a meaningless mask.
    static func alpha(subject: [UInt8], width: Int, height: Int,
                      margin: Float = 8, feather: Float = 16) throws -> [Float] {
        // 8192 px is a sanity limit far above any photo the app prepares.
        guard width > 1, height > 1, width <= 8192, height <= 8192,
              subject.count == width*height, subject.contains(where: { $0 != 0 }),
              margin.isFinite, margin >= 0, feather.isFinite, feather > 0 else {
            throw CleanupError.invalid("Invalid subject mask or cleanup geometry")
        }
        // Pass 1, rows: squared distance along the row to the nearest subject pixel.
        // Non-subject pixels start at 1e12 (a large finite "very far" value) rather
        // than infinity, because a row with no subject pixels would otherwise
        // compute infinity minus infinity, which is NaN.
        var distance = [Double](repeating: 0, count: width*height)
        for y in 0..<height {
            let row = (0..<width).map { subject[y*width+$0] == 0 ? 1e12 : 0.0 }
            let squared = distanceLine(row)
            for x in 0..<width { distance[y*width+x] = squared[x] }
        }
        // Pass 2, columns: running the same transform over the row results gives the
        // exact 2-D squared Euclidean distance.
        var result = [Float](repeating: 0, count: width*height)
        for x in 0..<width {
            let column = (0..<height).map { distance[$0*width+x] }
            let squared = distanceLine(column)
            for y in 0..<height {
                let p = y*width+x
                // Distance in pixels from this pixel to the nearest subject pixel.
                let d = Float(squared[y].squareRoot())
                // 0 inside the margin, rising to 1 at margin + feather.
                let t = min(Float(1), max(Float(0), (d-margin)/feather))
                // Smoothstep 3t^2 - 2t^3 gives a soft fade; subject pixels always get 1.
                result[p] = subject[p] == 0 ? 1-(t*t)*(3-2*t) : 1
            }
        }
        return result
    }

    /// Blends `protected` into `clean` with the per-pixel weight `alpha`:
    /// out = clean + alpha * (protected - clean). Weight 1 keeps the protected
    /// pixel, weight 0 restores the original. Rounding is half up in Double
    /// precision so the bytes match the reference images exactly.
    /// Both images are packed RGB, three bytes per pixel.
    static func composite(clean: [UInt8], protected: [UInt8], alpha: [Float]) throws -> [UInt8] {
        guard !alpha.isEmpty, clean.count == protected.count, clean.count == alpha.count*3,
              alpha.allSatisfy({ $0.isFinite && $0 >= 0 && $0 <= 1 }) else {
            throw CleanupError.invalid("Invalid RGB composite shape or alpha")
        }
        var out = clean
        for p in alpha.indices {
            let a = Double(alpha[p])
            for c in 0..<3 {
                // Byte index of channel c (R, G or B) of pixel p.
                let i = 3*p+c
                let value = Double(clean[i]) + a*Double(Int(protected[i])-Int(clean[i]))
                out[i] = UInt8(min(255, max(0, floor(value+0.5))))
            }
        }
        return out
    }

    /// Re-runs the kernel on one stored fixture and reports whether the output is
    /// byte-identical to the stored answer. Not used by the protection path.
    ///
    /// Fixture files: clean.rgb (original), previous.rgb (the protected image
    /// before cleanup), subject.u8 (mask), alpha.f32 (reference weights).
    static func check(_ fixture: CandidateFixture, root: URL, margin: Float, feather: Float) throws -> CleanupRow {
        let folder = root.appendingPathComponent(fixture.folder)
        /// Reads a fixture file and refuses it if its checksum is not the expected one.
        func read(_ name: String) throws -> Data {
            let data = try Data(contentsOf: folder.appendingPathComponent(name))
            guard sha(data) == fixture.files[name] else { throw CleanupError.invalid("Checksum mismatch: \(fixture.id)/\(name)") }
            return data
        }
        let clean = [UInt8](try read("clean.rgb"))
        let previous = [UInt8](try read("previous.rgb"))
        let subject = [UInt8](try read("subject.u8"))
        let referenceData = try read("alpha.f32")
        // Four bytes per Float, one Float per pixel.
        guard referenceData.count == fixture.width*fixture.height*4 else { throw CleanupError.invalid("Invalid alpha reference") }
        let reference: [Float] = referenceData.withUnsafeBytes { bytes in
            // Little-endian 32-bit floats, read without assuming alignment.
            (0..<(bytes.count/4)).map { Float(bitPattern: UInt32(littleEndian: bytes.loadUnaligned(fromByteOffset: $0*4, as: UInt32.self))) }
        }
        let start = ProcessInfo.processInfo.systemUptime
        let a = try alpha(subject: subject, width: fixture.width, height: fixture.height, margin: margin, feather: feather)
        let maskMS = (ProcessInfo.processInfo.systemUptime-start)*1000
        let compositeStart = ProcessInfo.processInfo.systemUptime
        let output = try composite(clean: clean, protected: previous, alpha: a)
        let compositeMS = (ProcessInfo.processInfo.systemUptime-compositeStart)*1000
        let hash = sha(Data(output))
        let maxAlphaDifference = zip(a,reference).map { abs($0-$1) }.max() ?? 0
        // Three sanity counts: the subject must be untouched by cleanup, pixels with
        // weight 0 must be restored exactly, and the largest change from the clean
        // image (L-infinity, in 0-255 units) must stay within the pixel budget.
        var foregroundChanges = 0, restoredChanges = 0, linf = 0
        for p in a.indices {
            for c in 0..<3 {
                let i = 3*p+c
                if subject[p] != 0 && output[i] != previous[i] { foregroundChanges += 1 }
                if a[p] == 0 && output[i] != clean[i] { restoredChanges += 1 }
                linf = max(linf, abs(Int(output[i])-Int(clean[i])))
            }
        }
        return CleanupRow(id: fixture.id, pixelExact: hash == fixture.tightRGBSHA256,
                          maximumAlphaError: maxAlphaDifference, foregroundChangedChannels: foregroundChanges,
                          restoredChangedChannels: restoredChanges, linf: linf, maskMilliseconds: maskMS,
                          compositeMilliseconds: compositeMS, outputRGBSHA256: hash)
    }
}

/// The result of `check` for one fixture.
struct CleanupRow: Codable {
    let id: String
    /// True when the output checksum equals the stored answer.
    let pixelExact: Bool
    /// Largest difference between computed and stored weights.
    let maximumAlphaError: Float
    /// Subject channels that cleanup changed (should be 0).
    let foregroundChangedChannels: Int
    /// Weight-0 channels that do not equal the clean image (should be 0).
    let restoredChangedChannels: Int
    /// Largest per-channel change from the clean image, 0-255 units.
    let linf: Int
    /// Time to build the weights and to blend, for profiling only.
    let maskMilliseconds: Double
    let compositeMilliseconds: Double
    let outputRGBSHA256: String
    /// A fixture passes only if every check holds. 5 is the per-channel change
    /// limit (5 out of 255) the protection is designed to stay within.
    var passed: Bool { pixelExact && maximumAlphaError <= 1e-6 && foregroundChangedChannels == 0 && restoredChangedChannels == 0 && linf <= 5 }
}
/// A full set of `check` results with where and when they were run.
struct CleanupReport: Codable {
    let schemaVersion: Int
    let createdAt: String
    let environment: String
    let manifestSHA256: String
    let scope: String
    let rows: [CleanupRow]
    let passed: Bool
}
