import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

@main struct PixelMetricsTests {
    static func png(_ bytes: [UInt8], width: Int, height: Int) throws -> Data {
        let provider = CGDataProvider(data: Data(bytes) as CFData)!
        let image = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 24,
            bytesPerRow: width * 3, space: CGColorSpace(name: CGColorSpace.sRGB)!,
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue), provider: provider,
            decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
        let data = NSMutableData()
        let destination = CGImageDestinationCreateWithData(data, UTType.png.identifier as CFString, 1, nil)!
        CGImageDestinationAddImage(destination, image, nil)
        precondition(CGImageDestinationFinalize(destination))
        return data as Data
    }
    static func main() throws {
        let original = try png(Array(repeating: 100, count: 12), width: 2, height: 2)
        var changed = Array(repeating: UInt8(100), count: 12)
        changed[0] = 103; changed[1] = 104
        let altered = try png(changed, width: 2, height: 2)
        let measured = try PixelMetrics.measure(original, altered)
        precondition(measured.maximumChange == 4)
        precondition(abs(measured.rmsChange - sqrt(25.0 / 12.0)) < 1e-9)
        precondition(abs(measured.psnr - 10 * log10(65025 / (25.0 / 12.0))) < 1e-9)
        let same = try PixelMetrics.measure(original, original)
        precondition(same.maximumChange == 0 && same.rmsChange == 0 && same.psnr == .infinity)
        do {
            _ = try PixelMetrics.measure(original, png([100,100,100], width: 1, height: 1))
            preconditionFailure("Different dimensions must be rejected")
        } catch {}
        do { _ = try PixelMetrics.measure(Data(), altered); preconditionFailure("Invalid PNG must fail") } catch {}
        print("PixelMetrics: known differences, identical pixels, dimension mismatch and invalid input passed")
    }
}
