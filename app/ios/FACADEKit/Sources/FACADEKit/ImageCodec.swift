// ImageCodec.swift
//
// Turning the user's photo file into the fixed 256x256 working frame.
//
// The models in the bundle all expect a square 256x256 RGB image with values
// in 0...1. Photos from the camera roll are rarely square and often carry an
// EXIF orientation tag. `FACADECoreImagePNGCodec` uses Core Image to:
//
//   1. Decode the file (PNG, JPEG, HEIC - whatever Core Image can read),
//      applying the EXIF orientation so the pixels are upright.
//   2. Scale the image down so its longer side is 256, keeping aspect ratio.
//   3. Centre it on a neutral grey square (letter-boxing), so nothing is
//      cropped away and the face detector sees the whole photo.
//   4. Read the result back as 8-bit RGBA and convert to planar floats.
//
// Core Image does the decoding, rotation and Lanczos scaling, so no image
// file parsing or resampling code lives in this package.

import CoreGraphics
import CoreImage
import Foundation

/// Core Image implementation of `FACADELocalVerificationImageCodec`.
///
/// Despite "PNG" in its name it reads any format Core Image understands; it
/// only decodes and prepares, it never writes files.
public final class FACADECoreImagePNGCodec: @unchecked Sendable, FACADELocalVerificationImageCodec {
    /// Side length of the square output frame. Only 256 is supported.
    public let side: Int
    /// Core Image rendering context, created once and reused for every photo.
    private let context: CIContext

    /// Creates a codec for the given frame size. `decodeAndPrepare` refuses
    /// any size other than 256, because that is what every model expects.
    public init(side: Int = 256) {
        self.side = side
        // Intermediate caching is off: each photo is processed exactly once,
        // so caching would only hold memory.
        self.context = CIContext(options: [.cacheIntermediates: false])
    }

    /// Decodes `data` and returns it as a 256x256 letter-boxed RGB float image.
    public func decodeAndPrepare(_ data: Data) throws -> FACADEFloatImage {
        // `.applyOrientationProperty` bakes the EXIF rotation into the pixels.
        guard side == 256,
              let source = CIImage(
                  data: data,
                  options: [.applyOrientationProperty: true]
              ) else {
            throw FACADEProtectionError.invalidInput("the input image could not be decoded")
        }
        let extent = source.extent
        guard extent.width > 0, extent.height > 0,
              extent.width.isFinite, extent.height.isFinite else {
            throw FACADEProtectionError.invalidInput("the input image has invalid dimensions")
        }
        // Uniform scale so the longer side becomes exactly `side`.
        let scale = min(CGFloat(side) / extent.width, CGFloat(side) / extent.height)
        // Move the image so its origin is at (0, 0) before scaling.
        let normalized = source.transformed(
            by: CGAffineTransform(translationX: -extent.origin.x, y: -extent.origin.y)
        )
        let resized = normalized.applyingFilter(
            "CILanczosScaleTransform",
            parameters: [
                kCIInputScaleKey: scale,
                kCIInputAspectRatioKey: 1.0,
            ]
        )
        // Centre the scaled image inside the square.
        let translated = resized.transformed(
            by: CGAffineTransform(
                translationX: (CGFloat(side) - resized.extent.width) * 0.5 - resized.extent.origin.x,
                y: (CGFloat(side) - resized.extent.height) * 0.5 - resized.extent.origin.y
            )
        )
        let bounds = CGRect(x: 0, y: 0, width: side, height: side)
        // Mid-grey (127/255) fills whatever the photo does not cover.
        let background = CIImage(
            color: CIColor(red: 127.0 / 255.0, green: 127.0 / 255.0, blue: 127.0 / 255.0)
        ).cropped(to: bounds)
        let prepared = translated.composited(over: background).cropped(to: bounds)
        // Render to an interleaved 8-bit RGBA buffer in sRGB.
        var bytes = [UInt8](repeating: 0, count: side * side * 4)
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
        bytes.withUnsafeMutableBytes { buffer in
            guard let baseAddress = buffer.baseAddress else { return }
            context.render(
                prepared,
                toBitmap: baseAddress,
                rowBytes: side * 4,
                bounds: bounds,
                format: .RGBA8,
                colorSpace: colorSpace
            )
        }

        // De-interleave RGBA bytes into three float planes (alpha is dropped).
        let plane = side * side
        var values = [Float](repeating: 0, count: 3 * plane)
        for pixel in 0..<plane {
            values[pixel] = Float(bytes[pixel * 4]) / 255.0
            values[plane + pixel] = Float(bytes[pixel * 4 + 1]) / 255.0
            values[2 * plane + pixel] = Float(bytes[pixel * 4 + 2]) / 255.0
        }
        return try FACADEFloatImage(width: side, height: side, values: values)
    }
}
