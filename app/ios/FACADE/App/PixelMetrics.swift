import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

/// Display-only pixel diagnostics. Never changes the protected image or engine.
struct PixelMetrics: Sendable {
    let maximumChange: Int
    let rmsChange: Double
    let psnr: Double
    let deltaPNG: Data

    static func measure(_ original: Data, _ protected: Data) throws -> PixelMetrics {
        let a = try decode(original), b = try decode(protected)
        guard a.width == b.width, a.height == b.height else { throw PixelError.dimensions }
        var delta = [UInt8](repeating: 255, count: a.bytes.count)
        var maximum = 0, sumSquared = 0.0
        for pixel in 0..<(a.width * a.height) {
            for channel in 0..<3 {
                let i = pixel * 4 + channel
                let difference = abs(Int(a.bytes[i]) - Int(b.bytes[i]))
                maximum = max(maximum, difference)
                sumSquared += Double(difference * difference)
                delta[i] = UInt8(min(255, difference * 16))
            }
        }
        let mse = sumSquared / Double(a.width * a.height * 3)
        let png = try encode(delta, width: a.width, height: a.height)
        return PixelMetrics(maximumChange: maximum, rmsChange: sqrt(mse),
            psnr: mse == 0 ? .infinity : 10 * log10(255 * 255 / mse), deltaPNG: png)
    }

    private enum PixelError: Error { case decode, dimensions, encode }
    private static func decode(_ data: Data) throws -> (bytes: [UInt8], width: Int, height: Int) {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil),
              image.width > 0, image.height > 0, image.width <= 1024, image.height <= 1024 else {
            throw PixelError.decode
        }
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
