// PixelMetricsTests.swift
//
// A small test for PixelMetrics.swift, the code that measures how much the
// protected copy differs from the original (largest change, RMS change, PSNR).
//
// Where it sits: app/ios/scripts/run_host_tests.sh compiles this file together
// with App/PixelMetrics.swift into a tiny Mac program and runs it. No
// simulator, iPhone or model is needed.
//
// Key ideas:
// 1. The test builds two 2 x 2 images whose difference is known exactly, so
//    each metric can be checked against a value worked out by hand.
// 2. `precondition` stops the program with a non-zero exit on any failure,
//    which makes the test script report it.
import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

/// The test program. `@main` makes `main()` the entry point.
@main struct PixelMetricsTests {
    /// Encodes raw RGB bytes (3 per pixel, no alpha) as a PNG, the format the app
    /// passes to PixelMetrics.
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
    /// Runs every check in order.
    static func main() throws {
        // 2 x 2 pixels x 3 channels = 12 values, all 100.
        let original = try png(Array(repeating: 100, count: 12), width: 2, height: 2)
        var changed = Array(repeating: UInt8(100), count: 12)
        // Change two values, by +3 and +4. The largest change is 4, and the
        // sum of squares is 9 + 16 = 25 over 12 values, so the mean squared error
        // (MSE) is 25/12.
        changed[0] = 103; changed[1] = 104
        let altered = try png(changed, width: 2, height: 2)
        let measured = try PixelMetrics.measure(original, altered)
        precondition(measured.maximumChange == 4)
        // RMS change = square root of the MSE.
        precondition(abs(measured.rmsChange - sqrt(25.0 / 12.0)) < 1e-9)
        // PSNR = 10 log10(255^2 / MSE); 255^2 = 65025.
        precondition(abs(measured.psnr - 10 * log10(65025 / (25.0 / 12.0))) < 1e-9)
        // Identical images: no change, and PSNR is infinite (MSE is zero).
        let same = try PixelMetrics.measure(original, original)
        precondition(same.maximumChange == 0 && same.rmsChange == 0 && same.psnr == .infinity)
        // Images of different sizes cannot be compared and must be refused.
        do {
            _ = try PixelMetrics.measure(original, png([100,100,100], width: 1, height: 1))
            preconditionFailure("Different dimensions must be rejected")
        } catch {}
        // Bytes that are not a PNG must be refused too.
        do { _ = try PixelMetrics.measure(Data(), altered); preconditionFailure("Invalid PNG must fail") } catch {}
        print("PixelMetrics: known differences, identical pixels, dimension mismatch and invalid input passed")
    }
}
