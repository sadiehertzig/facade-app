// Standalone focused test: compile with -D MINI_PHOTOGUARD_STANDALONE_TEST,
// App/MiniPhotoGuardEngine.swift, and this file. No model loads or user photos.
#if MINI_PHOTOGUARD_STANDALONE_TEST
import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

struct LivePhotoProgress: Sendable { let stage: String; let step: Int; let totalSteps: Int; let elapsedSeconds: Double }
struct LivePhotoResult: Sendable {
    let originalPNG: Data; let protectedPNG: Data; let width: Int; let height: Int
    let elapsedSeconds: Double; let preparationSeconds: Double; let loadSeconds: Double; let pipelineSeconds: Double
}

@main @MainActor struct MiniPhotoGuardEngineTests {
    static var checks = 0
    static func require(_ value: Bool, _ message: String) throws {
        checks += 1
        if !value { throw MiniPhotoGuardError.invalid("Test failed: " + message) }
    }
    static func mustThrow(_ message: String, _ body: () throws -> Void) throws {
        do { try body() } catch { checks += 1; return }
        throw MiniPhotoGuardError.invalid("Expected rejection: " + message)
    }
    static func fixture(width: Int, height: Int, orientation: Int = 1, alpha: UInt8 = 255) throws -> Data {
        var rgba = [UInt8](repeating: alpha, count: width*height*4)
        for y in 0..<height { for x in 0..<width {
            let i = (y*width+x)*4
            rgba[i] = UInt8((x*3+y*5)%256); rgba[i+1] = UInt8((x*7+y*11)%256); rgba[i+2] = UInt8((x*13+y*17)%256)
            if alpha == 0 { rgba[i] = 0; rgba[i+1] = 0; rgba[i+2] = 0 }
        } }
        let image = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
            bytesPerRow: width*4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
            provider: CGDataProvider(data: Data(rgba) as CFData)!, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
        let data = NSMutableData()
        let destination = CGImageDestinationCreateWithData(data, UTType.tiff.identifier as CFString, 1, nil)!
        CGImageDestinationAddImage(destination, image, [kCGImagePropertyOrientation: orientation,
            kCGImagePropertyExifDictionary: [kCGImagePropertyExifUserComment: "not retained"],
            kCGImagePropertyGPSDictionary: [kCGImagePropertyGPSLatitude: 1.0]] as CFDictionary)
        guard CGImageDestinationFinalize(destination) else { throw MiniPhotoGuardError.invalid("Fixture encoding failed") }
        return data as Data
    }
    static func main() async throws {
        var rgb = [UInt8](repeating: 0, count: MiniPhotoGuardPixels.count)
        for i in rgb.indices { rgb[i] = UInt8((i*29+i/77)%256) }
        let normalized = try MiniPhotoGuardPixels.normalize(rgb)
        try require(normalized.count == 196_608, "NCHW shape")
        try require(normalized[65_536+123] == Float(rgb[123*3+1])*(2.0/255)-1, "NCHW channel mapping")
        try require(try MiniPhotoGuardPixels.quantize(normalized, originalRGB: rgb) == rgb, "all-byte normalization round trip")
        let png = try MiniPhotoGuardPixels.encode(rgb)
        try require(try MiniPhotoGuardPixels.decodePNG(png) == rgb, "PNG lossless RGB round trip")
        let preview = try MiniPhotoGuardEngine.preparePhoto(data: png)
        try require(preview.width == 256 && preview.height == 256, "preview size")
        try require(try MiniPhotoGuardPixels.decodePNG(preview.png) == rgb, "already-prepared image preserves exact pixels")
        try require(try MiniPhotoGuardEngine.preparePhoto(data: preview.png).png == preview.png, "preview bytes idempotent when processing")
        var image = normalized
        let gradient = (0..<image.count).map { Float($0 % 3 - 1) }
        for _ in 0..<200 { try MiniPhotoGuardPixels.project(&image, original: normalized, gradient: gradient) }
        let protected = try MiniPhotoGuardPixels.quantize(image, originalRGB: rgb)
        try require(zip(protected, rgb).allSatisfy { abs(Int($0.0)-Int($0.1)) <= 6 }, "200-step quantized L-infinity bound")
        try require(protected[0] == min(255, Int(rgb[0])+6), "negative gradient descends upward")
        try require(protected[3] == rgb[3], "zero gradient does not change its corresponding pixel")
        var one = normalized
        try MiniPhotoGuardPixels.project(&one, original: normalized, gradient: [Float](repeating: 1, count: one.count))
        try require(one[100] <= normalized[100], "positive gradient descends")
        var badGradient = gradient; badGradient[4] = .nan
        try mustThrow("nonfinite gradient") { var bad = normalized; try MiniPhotoGuardPixels.project(&bad, original: normalized, gradient: badGradient) }
        try mustThrow("gradient shape") { var bad = normalized; try MiniPhotoGuardPixels.project(&bad, original: normalized, gradient: [0]) }
        try mustThrow("output out of bounds") { var bad = normalized; bad[0] = 2; _ = try MiniPhotoGuardPixels.quantize(bad, originalRGB: rgb) }
        try mustThrow("input decoding") { _ = try MiniPhotoGuardEngine.preparePhoto(data: Data([1,2,3])) }
        try mustThrow("oversize encoded input") { _ = try MiniPhotoGuardEngine.preparePhoto(data: Data(count: 32*1024*1024+1)) }
        let photo = try fixture(width: 100, height: 200)
        let prepared = try MiniPhotoGuardEngine.preparePhoto(data: photo)
        let portrait = try MiniPhotoGuardPixels.decodePNG(prepared.png)
        try require(portrait[0...2].allSatisfy { $0 == 128 }, "aspect-fit neutral padding")
        let source = CGImageSourceCreateWithData(prepared.png as CFData, nil)!
        let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil)! as NSDictionary
        // ImageIO synthesizes color-space/dimension properties for a new RGB
        // CGImage; no source EXIF, GPS, orientation, or text is forwarded.
        let exif = properties[kCGImagePropertyExifDictionary] as? NSDictionary
        let allowedExif = Set([kCGImagePropertyExifColorSpace as String,
            kCGImagePropertyExifPixelXDimension as String, kCGImagePropertyExifPixelYDimension as String])
        let exifKeys = Set(exif?.allKeys.compactMap { $0 as? String } ?? [])
        try require(exifKeys.isSubset(of: allowedExif) && properties[kCGImagePropertyGPSDictionary] == nil
                    && exif?[kCGImagePropertyExifUserComment] == nil, "original EXIF/GPS is not copied")
        let rotated = try MiniPhotoGuardEngine.preparePhoto(data: fixture(width: 100, height: 200, orientation: 6))
        let rotatedRGB = try MiniPhotoGuardPixels.decodePNG(rotated.png)
        try require(Array(rotatedRGB[(128*256)*3..<(128*256)*3+3]) != [128,128,128], "orientation applied before aspect fit")
        let transparent = try MiniPhotoGuardEngine.preparePhoto(data: fixture(width: 256, height: 256, alpha: 0))
        try require(try MiniPhotoGuardPixels.decodePNG(transparent.png).allSatisfy { $0 == 128 }, "transparent input composites on neutral gray")
        try require(!MiniPhotoGuardEngine.resourcesAvailable, "standalone executable cannot claim missing model readiness")
        let cancelled = Task { () throws -> LivePhotoResult in
            withUnsafeCurrentTask { $0?.cancel() }
            return try await MiniPhotoGuardEngine.shared.run(data: png) { _ in }
        }
        do { _ = try await cancelled.value; throw MiniPhotoGuardError.invalid("Cancellation unexpectedly succeeded") }
        catch { try require(error.localizedDescription.contains("cancelled"), "pre-cancel stops before model access") }
        print("MiniPhotoGuardEngineTests: \(checks) checks passed; no model loaded, no protection efficacy evaluated.")
    }
}
#endif
