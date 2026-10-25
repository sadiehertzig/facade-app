// FaceAlignment.swift
//
// Finding the face and lining it up the way face-recognition models expect.
//
// The ArcFace recogniser in the model bundle was trained on 112x112 crops in
// which the eyes, nose and mouth corners sit at fixed positions. To score a
// photo the way that model expects, FACADEKit:
//
//   1. Asks Apple Vision for face landmarks (`FACADEVisionSingleFaceDetector`).
//      Exactly one face must be present, otherwise the photo is refused.
//   2. Fits a similarity transform (rotation + uniform scale + translation)
//      that moves the five detected points onto the standard template
//      (`FACADEFaceCropTransform.init(sourceLandmarks:)`).
//   3. Resamples the photo through that transform with bilinear
//      interpolation to produce the aligned crop (`alignedImage(from:)`).
//
// The protection loop also needs to go the other way: the gradient model
// reports how the score changes per pixel of the aligned crop, and the app
// needs that per pixel of the full frame. `backProject(alignedGradient:...)`
// is the exact mathematical transpose of the resampler, so a gradient pushed
// through it lands on the right source pixels with the right weights.
//
// Where it fits: the app's NativePipeline asks the detector for one transform
// per photo, then calls `alignedImage` and `backProject` on every step of its
// protection loop. (The SimSwap recogniser uses a different 224-pixel crop
// that the app computes itself.)
//
// `FACADEApplePixelBridge` is a small helper that turns a float image into a
// `CGImage`, which is what Vision consumes.

import CoreGraphics
import Foundation
import Vision

/// Anything that can find the single face in a prepared frame and return the
/// transform that aligns it. The app holds the Vision implementation behind
/// this protocol.
public protocol FACADEExactlyOneFaceDetector: Sendable {
    /// Returns the alignment transform for the one face in `image`, or throws
    /// if there is not exactly one face with complete landmarks.
    func alignmentTransform(in image: FACADEFloatImage) throws -> FACADEFaceCropTransform
}

/// Face detection and landmark extraction using Apple's Vision framework.
///
/// The rule is deliberately strict: Vision must find exactly one face, that
/// face must be at least `minimumFaceFraction` of the frame, and all of the
/// eye, nose and lip landmark groups must be present. Both numbers come from
/// the model manifest so the app runs the same policy the models were
/// validated with.
public final class FACADEVisionSingleFaceDetector: @unchecked Sendable, FACADEExactlyOneFaceDetector {
    /// Vision request revision to use (from the manifest).
    private let requestRevision: Int
    /// Smallest accepted face, as a fraction of the shorter frame side.
    private let minimumFaceFraction: Float

    /// Validates the two policy numbers before storing them.
    public init(requestRevision: Int, minimumFaceFraction: Double) throws {
        guard requestRevision >= 1,
              minimumFaceFraction.isFinite,
              minimumFaceFraction > 0,
              minimumFaceFraction <= 1 else {
            throw FACADEProtectionError.modelUnavailable("invalid Vision face policy")
        }
        self.requestRevision = requestRevision
        self.minimumFaceFraction = Float(minimumFaceFraction)
    }

    /// Runs a landmark request and builds the alignment transform from the
    /// five standard points: left eye centre, right eye centre, nose centre,
    /// left mouth corner, right mouth corner.
    public func alignmentTransform(in image: FACADEFloatImage) throws -> FACADEFaceCropTransform {
        let cgImage = try FACADEApplePixelBridge.makeCGImage(image)
        let request = VNDetectFaceLandmarksRequest()
        request.revision = requestRevision
        do {
            try VNImageRequestHandler(cgImage: cgImage, orientation: .up).perform([request])
        } catch {
            throw FACADEProtectionError.verificationUnavailable(
                "Vision landmark detection failed: \(error)"
            )
        }
        // One face, big enough, with every landmark group we need.
        guard let observations = request.results, observations.count == 1,
              let observation = observations.first,
              Float(min(observation.boundingBox.width, observation.boundingBox.height)) >=
                minimumFaceFraction,
              let landmarks = observation.landmarks,
              let visionLeftEye = landmarks.leftEye,
              let visionRightEye = landmarks.rightEye,
              let nose = landmarks.nose,
              let lips = landmarks.outerLips else {
            throw FACADEProtectionError.verificationRejected(
                "exactly one face with complete five-point landmarks is required"
            )
        }

        // Vision returns a contour of several points per feature; the centre
        // of each contour is used as that feature's single landmark.
        let leftEye = try meanPoint(
            try imagePoints(visionLeftEye, faceBox: observation.boundingBox, image: image)
        )
        let rightEye = try meanPoint(
            try imagePoints(visionRightEye, faceBox: observation.boundingBox, image: image)
        )
        // Vision's "left" is the subject's left; the template wants the eye
        // with the smaller x first, so sort by x rather than trusting labels.
        let orderedEyes = [leftEye, rightEye].sorted { $0.x < $1.x }
        let nosePoint = try meanPoint(
            try imagePoints(nose, faceBox: observation.boundingBox, image: image)
        )
        // Mouth corners are the leftmost and rightmost points of the outer lip contour.
        let lipPoints = try imagePoints(lips, faceBox: observation.boundingBox, image: image)
        guard let leftMouth = lipPoints.min(by: { $0.x < $1.x }),
              let rightMouth = lipPoints.max(by: { $0.x < $1.x }) else {
            throw FACADEProtectionError.verificationRejected(
                "Vision did not return both mouth corners"
            )
        }
        return try FACADEFaceCropTransform(
            sourceLandmarks: [
                orderedEyes[0], orderedEyes[1], nosePoint, leftMouth, rightMouth,
            ]
        )
    }

    /// Converts a Vision landmark region to pixel coordinates.
    ///
    /// Vision reports points relative to the face box, and the face box
    /// relative to the image, both with the origin at the bottom-left. The
    /// image tensor has its origin at the top-left, hence the `1 - y` flip.
    private func imagePoints(
        _ region: VNFaceLandmarkRegion2D,
        faceBox: CGRect,
        image: FACADEFloatImage
    ) throws -> [FACADEImagePoint] {
        let points = region.normalizedPoints
        guard !points.isEmpty else {
            throw FACADEProtectionError.verificationRejected("Vision returned an empty landmark region")
        }
        return points.map { point in
            let normalizedX = Float(faceBox.minX + point.x * faceBox.width)
            let normalizedYTop = 1 - Float(faceBox.minY + point.y * faceBox.height)
            return FACADEImagePoint(
                x: normalizedX * Float(image.width),
                y: normalizedYTop * Float(image.height)
            )
        }
    }

    /// Centre of a set of points.
    private func meanPoint(_ points: [FACADEImagePoint]) throws -> FACADEImagePoint {
        guard !points.isEmpty else {
            throw FACADEProtectionError.verificationRejected("Vision returned no landmark points")
        }
        return FACADEImagePoint(
            x: points.reduce(0) { $0 + $1.x } / Float(points.count),
            y: points.reduce(0) { $0 + $1.y } / Float(points.count)
        )
    }
}

/// A point in pixel coordinates, origin top-left, y increasing downwards.
public struct FACADEImagePoint: Sendable, Equatable {
    /// Horizontal position in pixels, 0 at the left edge.
    public let x: Float
    /// Vertical position in pixels, 0 at the top edge.
    public let y: Float

    /// Stores the two coordinates as given.
    public init(x: Float, y: Float) {
        self.x = x
        self.y = y
    }
}

/// The transform that maps a face in the source frame onto the standard
/// 112x112 recogniser crop, plus the resampling in both directions.
public struct FACADEFaceCropTransform: Sendable, Equatable {
    /// Side length of the square crop (112 for the bundled recognisers).
    public let outputSide: Int
    /// Row-major 2x3 affine matrix mapping an output pixel (x, y) to the
    /// source pixel it should sample:
    ///
    ///     sx = a[0]*x + a[1]*y + a[2]
    ///     sy = a[3]*x + a[4]*y + a[5]
    private let outputToSourceAffine: [Float]

    /// Fits a similarity transform from the five detected landmarks to the
    /// ArcFace template and stores its inverse.
    ///
    /// The template coordinates are the standard ArcFace reference points for
    /// a 112x112 crop (scaled if `outputSide` differs). The fit is a
    /// closed-form least-squares solution for a transform of the form
    /// `[a -b; b a] * p + t`, which allows rotation and uniform scale but no
    /// shear, so the face is never distorted.
    public init(sourceLandmarks: [FACADEImagePoint], outputSide: Int = 112) throws {
        let target: [FACADEImagePoint] = [
            FACADEImagePoint(x: 38.2946, y: 51.6963),
            FACADEImagePoint(x: 73.5318, y: 51.5014),
            FACADEImagePoint(x: 56.0252, y: 71.7366),
            FACADEImagePoint(x: 41.5493, y: 92.3655),
            FACADEImagePoint(x: 70.7299, y: 92.2041),
        ].map { point in
            let scale = Float(outputSide) / 112.0
            return FACADEImagePoint(x: point.x * scale, y: point.y * scale)
        }
        guard outputSide > 0, sourceLandmarks.count == target.count,
              sourceLandmarks.allSatisfy({ $0.x.isFinite && $0.y.isFinite }) else {
            throw FACADEProtectionError.verificationRejected(
                "five finite landmarks are required for ArcFace alignment"
            )
        }
        // Centre both point sets, then solve for (a, b) in closed form.
        let sourceCenter = Self.mean(sourceLandmarks)
        let targetCenter = Self.mean(target)
        var denominator: Float = 0
        var aNumerator: Float = 0
        var bNumerator: Float = 0
        for (source, destination) in zip(sourceLandmarks, target) {
            let x = source.x - sourceCenter.x
            let y = source.y - sourceCenter.y
            let u = destination.x - targetCenter.x
            let v = destination.y - targetCenter.y
            denominator += x * x + y * y
            aNumerator += x * u + y * v
            bNumerator += x * v - y * u
        }
        // All five landmarks on one point would give a zero denominator.
        guard denominator > 1e-6 else {
            throw FACADEProtectionError.verificationRejected("the face landmarks are degenerate")
        }
        let a = aNumerator / denominator
        let b = bNumerator / denominator
        let determinant = a * a + b * b
        guard determinant.isFinite, determinant > 1e-8 else {
            throw FACADEProtectionError.verificationRejected(
                "the ArcFace alignment transform is not invertible"
            )
        }
        let targetOffsetX = targetCenter.x - a * sourceCenter.x + b * sourceCenter.y
        let targetOffsetY = targetCenter.y - b * sourceCenter.x - a * sourceCenter.y
        // We sample the source for each output pixel, so what is stored is the
        // inverse of target = [a -b; b a] * source + offset.
        let inverse = [
            a / determinant,
            b / determinant,
            (-a * targetOffsetX - b * targetOffsetY) / determinant,
            -b / determinant,
            a / determinant,
            (b * targetOffsetX - a * targetOffsetY) / determinant,
        ]
        self.outputSide = outputSide
        self.outputToSourceAffine = inverse
    }

    /// Produces the aligned crop by bilinear sampling: for every output pixel,
    /// find the source position it maps to and blend the four neighbouring
    /// source pixels by their distance weights.
    public func alignedImage(from source: FACADEFloatImage) throws -> FACADEFloatImage {
        var output = [Float](repeating: 0, count: 3 * outputSide * outputSide)
        let sourcePlane = source.width * source.height
        let outputPlane = outputSide * outputSide
        for y in 0..<outputSide {
            for x in 0..<outputSide {
                let sourcePoint = mappedSourcePoint(x: x, y: y)
                let sample = Self.bilinearSupport(
                    x: sourcePoint.x,
                    y: sourcePoint.y,
                    width: source.width,
                    height: source.height
                )
                for channel in 0..<3 {
                    var value: Float = 0
                    for item in sample {
                        value += source.values[channel * sourcePlane + item.index] * item.weight
                    }
                    output[channel * outputPlane + y * outputSide + x] = value
                }
            }
        }
        return try FACADEFloatImage(width: outputSide, height: outputSide, values: output)
    }

    /// Exact transpose of `alignedImage`'s bilinear sampler.
    ///
    /// `alignedImage` computes each crop pixel as a weighted sum of four source
    /// pixels. The derivative of that sum with respect to each source pixel is
    /// just its weight, so to carry a gradient from the crop back to the
    /// source we add `gradient * weight` into the same four source pixels.
    /// Source pixels that several crop pixels touch accumulate all of their
    /// contributions. The app uses this to turn the gradient model's 112x112
    /// output into a gradient over the full 256x256 frame.
    public func backProject(
        alignedGradient: FACADEFloatImage,
        sourceWidth: Int,
        sourceHeight: Int
    ) throws -> FACADEFloatImage {
        guard alignedGradient.width == outputSide,
              alignedGradient.height == outputSide else {
            throw FACADEProtectionError.candidateGenerationFailed(
                "the ArcFace input-gradient shape does not match the aligned face size"
            )
        }
        // Building the empty frame through `FACADEFloatImage` is also the size
        // check: a zero or mismatched frame size throws here instead of
        // reaching the index arithmetic below.
        let emptySource = try FACADEFloatImage(
            width: sourceWidth,
            height: sourceHeight,
            values: [Float](repeating: 0, count: 3 * sourceWidth * sourceHeight)
        )
        var output = emptySource.values
        let sourcePlane = sourceWidth * sourceHeight
        let alignedPlane = outputSide * outputSide
        for y in 0..<outputSide {
            for x in 0..<outputSide {
                let sourcePoint = mappedSourcePoint(x: x, y: y)
                let support = Self.bilinearSupport(
                    x: sourcePoint.x,
                    y: sourcePoint.y,
                    width: sourceWidth,
                    height: sourceHeight
                )
                for channel in 0..<3 {
                    let gradient = alignedGradient.values[channel * alignedPlane + y * outputSide + x]
                    for item in support {
                        output[channel * sourcePlane + item.index] += gradient * item.weight
                    }
                }
            }
        }
        return try FACADEFloatImage(width: sourceWidth, height: sourceHeight, values: output)
    }

    /// Applies the stored affine matrix to an output pixel index.
    private func mappedSourcePoint(x: Int, y: Int) -> (x: Float, y: Float) {
        let affine = outputToSourceAffine
        let outputX = Float(x)
        let outputY = Float(y)
        return (
            affine[0] * outputX + affine[1] * outputY + affine[2],
            affine[3] * outputX + affine[4] * outputY + affine[5]
        )
    }

    /// Centre of a set of points.
    private static func mean(_ points: [FACADEImagePoint]) -> FACADEImagePoint {
        FACADEImagePoint(
            x: points.reduce(0) { $0 + $1.x } / Float(points.count),
            y: points.reduce(0) { $0 + $1.y } / Float(points.count)
        )
    }

    /// One of the four source pixels a sample touches, with its weight.
    private struct Support {
        /// Index into a single colour plane (`y * width + x`).
        let index: Int
        let weight: Float
    }

    /// The four pixels around a fractional position and their bilinear
    /// weights. Positions outside the image are clamped to the edge, so the
    /// weights always sum to one and no index is out of range.
    private static func bilinearSupport(
        x: Float,
        y: Float,
        width: Int,
        height: Int
    ) -> [Support] {
        let clampedX = min(Float(width - 1), max(0, x))
        let clampedY = min(Float(height - 1), max(0, y))
        let x0 = Int(floor(clampedX))
        let y0 = Int(floor(clampedY))
        let x1 = min(width - 1, x0 + 1)
        let y1 = min(height - 1, y0 + 1)
        let fx = clampedX - Float(x0)
        let fy = clampedY - Float(y0)
        return [
            Support(index: y0 * width + x0, weight: (1 - fx) * (1 - fy)),
            Support(index: y0 * width + x1, weight: fx * (1 - fy)),
            Support(index: y1 * width + x0, weight: (1 - fx) * fy),
            Support(index: y1 * width + x1, weight: fx * fy),
        ]
    }
}

/// Converts a `FACADEFloatImage` into a `CGImage` so Apple frameworks can
/// consume it.
enum FACADEApplePixelBridge {
    /// Builds an 8-bit RGBA sRGB bitmap. Each float is scaled by 255, rounded
    /// to nearest, clamped to 0...255; alpha is set to fully opaque.
    static func makeCGImage(_ image: FACADEFloatImage) throws -> CGImage {
        let plane = image.width * image.height
        // Interleaved RGBA, 4 bytes per pixel. The planar float layout is
        // re-interleaved here because CGImage wants pixels, not planes.
        var bytes = [UInt8](repeating: 255, count: plane * 4)
        for pixel in 0..<plane {
            for channel in 0..<3 {
                bytes[pixel * 4 + channel] = UInt8(
                    min(255, max(0, Int((image.values[channel * plane + pixel] * 255).rounded())))
                )
            }
        }
        guard let provider = CGDataProvider(data: Data(bytes) as CFData),
              let cgImage = CGImage(
                  width: image.width,
                  height: image.height,
                  bitsPerComponent: 8,
                  bitsPerPixel: 32,
                  bytesPerRow: image.width * 4,
                  space: CGColorSpace(name: CGColorSpace.sRGB)!,
                  // `.last` + big-endian byte order means the bytes are read as
                  // R, G, B, A in memory order, matching the buffer above.
                  bitmapInfo: [
                      CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
                      .byteOrder32Big,
                  ],
                  provider: provider,
                  decode: nil,
                  shouldInterpolate: false,
                  intent: .defaultIntent
              ) else {
            throw FACADEProtectionError.verificationUnavailable("could not create a verification bitmap")
        }
        return cgImage
    }
}
