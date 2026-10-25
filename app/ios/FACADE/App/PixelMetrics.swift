// PixelMetrics.swift
//
// Computes the "See what changed" numbers on the Protect screen: the largest
// single-channel change, the root-mean-square change, PSNR, and an amplified
// difference image.
//
// Where it sits: the Protect screen (PhotoAcquisitionViews.swift) calls
// `measure` on a background task after a protected copy is ready. It only
// reads the two PNGs; it never feeds anything back into an engine.
//
// Key ideas:
// 1. These numbers describe how different the pixels are. They say nothing
//    about how well the protection works, and the screen says so.
// 2. Both PNGs are decoded to 8-bit RGBA so every pixel can be compared.
// 3. The difference image multiplies each channel difference by 16 so changes
//    of a few levels become visible; the alpha channel is left fully opaque.
import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

/// Display-only pixel diagnostics. Never changes the protected image or engine.
struct PixelMetrics: Sendable {
    /// Largest absolute change in any single R, G or B channel value (0...255).
    let maximumChange: Int
    /// Root mean square of the per-channel differences, in 0...255 units.
    let rmsChange: Double
    /// Peak signal-to-noise ratio in dB; infinity when the images are identical.
    let psnr: Double
    /// PNG of the amplified difference, shown as "Pixel difference · amplified ×16".
    let deltaPNG: Data

    /// Decodes both PNGs and compares them pixel by pixel.
    static func measure(_ original: Data, _ protected: Data) throws -> PixelMetrics {
        let a = try decode(original), b = try decode(protected)
        // The two images must be the same size; the protected copy is a crop of the
        // same canvas as the original, so this only fails on corrupted input.
        guard a.width == b.width, a.height == b.height else { throw PixelError.dimensions }
        // Starts at 255 so the alpha channel (every fourth byte) stays opaque.
        var delta = [UInt8](repeating: 255, count: a.bytes.count)
        var maximum = 0, sumSquared = 0.0
        for pixel in 0..<(a.width * a.height) {
            for channel in 0..<3 {
                // RGBA layout: 4 bytes per pixel; channels 0, 1, 2 are R, G, B.
                let i = pixel * 4 + channel
                let difference = abs(Int(a.bytes[i]) - Int(b.bytes[i]))
                maximum = max(maximum, difference)
                sumSquared += Double(difference * difference)
                // Amplify by 16 and cap at 255 for the visible difference image.
                delta[i] = UInt8(min(255, difference * 16))
            }
        }
        // Mean squared error over all R, G, B samples.
        let mse = sumSquared / Double(a.width * a.height * 3)
        let png = try encode(delta, width: a.width, height: a.height)
        return PixelMetrics(maximumChange: maximum, rmsChange: sqrt(mse),
            // Standard PSNR for 8-bit images: 10 * log10(255^2 / MSE).
            psnr: mse == 0 ? .infinity : 10 * log10(255 * 255 / mse), deltaPNG: png)
    }

    /// Internal failure reasons; the screen only shows a generic "unavailable" note.
    private enum PixelError: Error { case decode, dimensions, encode }
    /// Decodes an image into 8-bit RGBA bytes with a 1024 px size cap.
    private static func decode(_ data: Data) throws -> (bytes: [UInt8], width: Int, height: Int) {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil),
              // The engines never produce anything larger than 1024 px on a side.
              image.width > 0, image.height > 0, image.width <= 1024, image.height <= 1024 else {
            throw PixelError.decode
        }
        // Draw into a bitmap we own so the bytes are in a known layout.
        var bytes = [UInt8](repeating: 0, count: image.width * image.height * 4)
        let ok = bytes.withUnsafeMutableBytes { storage -> Bool in
            guard let context = CGContext(data: storage.baseAddress, width: image.width, height: image.height,
                bitsPerComponent: 8, bytesPerRow: image.width * 4,
                space: CGColorSpace(name: CGColorSpace.sRGB)!,
                bitmapInfo: CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue) else { return false }
            context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
            return true
        }
        guard ok else { throw PixelError.decode }
        return (bytes, image.width, image.height)
    }
    /// Encodes RGBA bytes as a PNG (used for the difference image).
    private static func encode(_ bytes: [UInt8], width: Int, height: Int) throws -> Data {
        guard let provider = CGDataProvider(data: Data(bytes) as CFData),
              let image = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
                bytesPerRow: width * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                bitmapInfo: CGBitmapInfo(rawValue: CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue),
                provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent) else { throw PixelError.encode }
        let data = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(data, UTType.png.identifier as CFString, 1, nil) else { throw PixelError.encode }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else { throw PixelError.encode }
        return data as Data
    }
}
