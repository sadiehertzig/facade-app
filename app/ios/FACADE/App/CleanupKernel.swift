import Foundation
import CryptoKit

struct CandidateFixture: Codable, Identifiable {
    let id: String
    let folder: String
    let width: Int
    let height: Int
    let files: [String: String]
    let tightRGBSHA256: String
}
struct CandidateManifest: Codable {
    let candidate: String
    let sourceCommit: String
    let scope: String
    let margin: Float
    let feather: Float
    let fixtures: [CandidateFixture]
}
enum CleanupError: Error, LocalizedError {
    case invalid(String)
    var errorDescription: String? { if case .invalid(let text) = self { return text }; return nil }
}

// Exact cleanup kernel; the native pipeline supplies its clean-image subject mask.
enum TightCleanup {
    static func sha(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    // Lower envelope of squared-distance parabolas. Finite sentinel avoids
    // infinity subtraction for rows containing no subject pixels.
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
                s = ((f[q] + Double(q*q)) - (f[v] + Double(v*v))) / Double(2*(q-v))
                if s > boundaries[k] || k == 0 { break }
                k -= 1
            } while true
            k += 1; sites[k] = q; boundaries[k] = s; boundaries[k+1] = .infinity
        }
        k = 0
        for q in 0..<n {
            while boundaries[k+1] < Double(q) { k += 1 }
            let d = q-sites[k]
            out[q] = Double(d*d) + f[sites[k]]
        }
        return out
    }

    static func alpha(subject: [UInt8], width: Int, height: Int,
                      margin: Float = 8, feather: Float = 16) throws -> [Float] {
        guard width > 1, height > 1, width <= 8192, height <= 8192,
              subject.count == width*height, subject.contains(where: { $0 != 0 }),
              margin.isFinite, margin >= 0, feather.isFinite, feather > 0 else {
            throw CleanupError.invalid("Invalid subject mask or cleanup geometry")
        }
        var distance = [Double](repeating: 0, count: width*height)
        for y in 0..<height {
            let row = (0..<width).map { subject[y*width+$0] == 0 ? 1e12 : 0.0 }
            let squared = distanceLine(row)
            for x in 0..<width { distance[y*width+x] = squared[x] }
        }
        var result = [Float](repeating: 0, count: width*height)
        for x in 0..<width {
            let column = (0..<height).map { distance[$0*width+x] }
            let squared = distanceLine(column)
            for y in 0..<height {
                let p = y*width+x
                let d = Float(squared[y].squareRoot())
                let t = min(Float(1), max(Float(0), (d-margin)/feather))
                result[p] = subject[p] == 0 ? 1-(t*t)*(3-2*t) : 1
            }
        }
        return result
    }

    // Round-half-up in float64, matching the research's final stored RGB image.
    static func composite(clean: [UInt8], protected: [UInt8], alpha: [Float]) throws -> [UInt8] {
        guard !alpha.isEmpty, clean.count == protected.count, clean.count == alpha.count*3,
              alpha.allSatisfy({ $0.isFinite && $0 >= 0 && $0 <= 1 }) else {
            throw CleanupError.invalid("Invalid RGB composite shape or alpha")
        }
        var out = clean
        for p in alpha.indices {
            let a = Double(alpha[p])
            for c in 0..<3 {
                let i = 3*p+c
                let value = Double(clean[i]) + a*Double(Int(protected[i])-Int(clean[i]))
                out[i] = UInt8(min(255, max(0, floor(value+0.5))))
            }
        }
        return out
    }

    static func check(_ fixture: CandidateFixture, root: URL, margin: Float, feather: Float) throws -> CleanupRow {
        let folder = root.appendingPathComponent(fixture.folder)
        func read(_ name: String) throws -> Data {
            let data = try Data(contentsOf: folder.appendingPathComponent(name))
            guard sha(data) == fixture.files[name] else { throw CleanupError.invalid("Checksum mismatch: \(fixture.id)/\(name)") }
            return data
        }
        let clean = [UInt8](try read("clean.rgb"))
        let previous = [UInt8](try read("previous.rgb"))
        let subject = [UInt8](try read("subject.u8"))
        let referenceData = try read("alpha.f32")
        guard referenceData.count == fixture.width*fixture.height*4 else { throw CleanupError.invalid("Invalid alpha reference") }
        let reference: [Float] = referenceData.withUnsafeBytes { bytes in
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

struct CleanupRow: Codable {
    let id: String
    let pixelExact: Bool
    let maximumAlphaError: Float
    let foregroundChangedChannels: Int
    let restoredChangedChannels: Int
    let linf: Int
    let maskMilliseconds: Double
    let compositeMilliseconds: Double
    let outputRGBSHA256: String
    var passed: Bool { pixelExact && maximumAlphaError <= 1e-6 && foregroundChangedChannels == 0 && restoredChangedChannels == 0 && linf <= 5 }
}
struct CleanupReport: Codable {
    let schemaVersion: Int
    let createdAt: String
    let environment: String
    let manifestSHA256: String
    let scope: String
    let rows: [CleanupRow]
    let passed: Bool
}
