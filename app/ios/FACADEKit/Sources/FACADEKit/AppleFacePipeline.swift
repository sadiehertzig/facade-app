import CoreGraphics
import CoreML
import Foundation
import ImageIO
import Metal
import MetalPerformanceShadersGraph
import UniformTypeIdentifiers
import Vision

public struct FACADEFaceRegion: Sendable, Equatable {
    /// Normalized top-left coordinates in the prepared 256x256 frame.
    public let x: Float
    public let y: Float
    public let width: Float
    public let height: Float

    public init(x: Float, y: Float, width: Float, height: Float) throws {
        guard [x, y, width, height].allSatisfy(\.isFinite),
              x >= 0, y >= 0, width > 0, height > 0,
              x + width <= 1.000001, y + height <= 1.000001 else {
            throw FACADEProtectionError.verificationUnavailable("Vision returned an invalid face region")
        }
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    }
}

public protocol FACADEExactlyOneFaceDetector: Sendable {
    func exactlyOneFace(in image: FACADEFloatImage) throws -> FACADEFaceRegion
    func alignmentTransform(in image: FACADEFloatImage) throws -> FACADEFaceCropTransform
}

/// The phone-v1 alignment rule is intentionally explicit: Vision must find
/// exactly one face and all five required landmark groups. Eye centers, nose
/// center, and lip corners are fitted to the standard 112x112 ArcFace template
/// with a least-squares similarity transform. The research manifest must name
/// this same rule; changing it is a new parity candidate.
public final class FACADEVisionSingleFaceDetector: @unchecked Sendable, FACADEExactlyOneFaceDetector {
    public static let coordinateConvention =
        "vision_5pt_to_arcface112_similarity_top_left_bilinear_v1"

    private let requestRevision: Int
    private let minimumFaceFraction: Float

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

    public func exactlyOneFace(in image: FACADEFloatImage) throws -> FACADEFaceRegion {
        let cgImage = try FACADEApplePixelBridge.makeCGImage(image)
        let request = VNDetectFaceRectanglesRequest()
        request.revision = requestRevision
        do {
            try VNImageRequestHandler(cgImage: cgImage, orientation: .up).perform([request])
        } catch {
            throw FACADEProtectionError.verificationUnavailable(
                "Vision face detection failed: \(error)"
            )
        }
        guard let observations = request.results, observations.count == 1,
              let face = observations.first else {
            throw FACADEProtectionError.verificationRejected(
                "exactly one face is required; no output was issued"
            )
        }

        let box = face.boundingBox
        guard Float(min(box.width, box.height)) >= minimumFaceFraction else {
            throw FACADEProtectionError.verificationRejected(
                "the detected face is smaller than the declared minimum"
            )
        }

        let centerX = Float(box.midX)
        let centerYTop = 1.0 - Float(box.midY)
        let desiredSide = min(1.0, Float(max(box.width, box.height)) * 1.25)
        let originX = min(max(0, centerX - desiredSide * 0.5), 1.0 - desiredSide)
        let originY = min(max(0, centerYTop - desiredSide * 0.5), 1.0 - desiredSide)
        return try FACADEFaceRegion(
            x: originX,
            y: originY,
            width: desiredSide,
            height: desiredSide
        )
    }

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

        let leftEye = try meanPoint(
            try imagePoints(visionLeftEye, faceBox: observation.boundingBox, image: image)
        )
        let rightEye = try meanPoint(
            try imagePoints(visionRightEye, faceBox: observation.boundingBox, image: image)
        )
        let orderedEyes = [leftEye, rightEye].sorted { $0.x < $1.x }
        let nosePoint = try meanPoint(
            try imagePoints(nose, faceBox: observation.boundingBox, image: image)
        )
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

public struct FACADEImagePoint: Sendable, Equatable {
    public let x: Float
    public let y: Float

    public init(x: Float, y: Float) {
        self.x = x
        self.y = y
    }
}

public struct FACADEFaceCropTransform: Sendable, Equatable {
    public let outputSide: Int
    private let mapping: Mapping

    private enum Mapping: Sendable, Equatable {
        case normalizedRegion(FACADEFaceRegion)
        /// Row-major 2x3 affine mapping from output pixels to source pixels.
        case outputToSourceAffine([Float])
    }

    public init(region: FACADEFaceRegion, outputSide: Int = 112) throws {
        guard outputSide > 0 else {
            throw FACADEProtectionError.invalidInput("face crop side must be positive")
        }
        self.outputSide = outputSide
        self.mapping = .normalizedRegion(region)
    }

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
        // Inverse of target = [a -b; b a] * source + offset.
        let inverse = [
            a / determinant,
            b / determinant,
            (-a * targetOffsetX - b * targetOffsetY) / determinant,
            -b / determinant,
            a / determinant,
            (b * targetOffsetX - a * targetOffsetY) / determinant,
        ]
        self.outputSide = outputSide
        self.mapping = .outputToSourceAffine(inverse)
    }

    public func alignedImage(from source: FACADEFloatImage) throws -> FACADEFloatImage {
        var output = [Float](repeating: 0, count: 3 * outputSide * outputSide)
        let sourcePlane = source.width * source.height
        let outputPlane = outputSide * outputSide
        for y in 0..<outputSide {
            for x in 0..<outputSide {
                let sourcePoint = mappedSourcePoint(x: x, y: y, source: source)
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

    /// Exact transpose of `alignedImage`'s bilinear sampler. This maps an
    /// aligned-crop input gradient back into the prepared 256x256 frame.
    public func backProject(
        alignedGradient: FACADEFloatImage,
        sourceWidth: Int,
        sourceHeight: Int
    ) throws -> FACADEFloatImage {
        guard alignedGradient.width == outputSide,
              alignedGradient.height == outputSide else {
            throw FACADEProtectionError.candidateGenerationFailed(
                "the ArcFace input-gradient shape does not match the alignment contract"
            )
        }
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
                let sourcePoint = mappedSourcePoint(x: x, y: y, source: emptySource)
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

    private func mappedSourcePoint(
        x: Int,
        y: Int,
        source: FACADEFloatImage
    ) -> (x: Float, y: Float) {
        switch mapping {
        case .normalizedRegion(let region):
            let normalizedX = region.x + (Float(x) + 0.5) / Float(outputSide) * region.width
            let normalizedY = region.y + (Float(y) + 0.5) / Float(outputSide) * region.height
            return (
                normalizedX * Float(source.width) - 0.5,
                normalizedY * Float(source.height) - 0.5
            )
        case .outputToSourceAffine(let affine):
            let outputX = Float(x)
            let outputY = Float(y)
            return (
                affine[0] * outputX + affine[1] * outputY + affine[2],
                affine[3] * outputX + affine[4] * outputY + affine[5]
            )
        }
    }

    private static func mean(_ points: [FACADEImagePoint]) -> FACADEImagePoint {
        FACADEImagePoint(
            x: points.reduce(0) { $0 + $1.x } / Float(points.count),
            y: points.reduce(0) { $0 + $1.y } / Float(points.count)
        )
    }

    private struct Support {
        let index: Int
        let weight: Float
    }

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

public enum FACADEEmbeddingNormalization: Sendable {
    case arcFaceMinusOneToOne
    case imageNetRGB
}

public final class FACADECoreMLEmbeddingModel: @unchecked Sendable {
    public let sourceWeightsSHA256: String
    private let model: MLModel
    private let inputName: String
    private let outputName: String
    private let normalization: FACADEEmbeddingNormalization

    public init(
        model: MLModel,
        artifact: FACADEModelArtifact,
        expectedRole: FACADEModelRole,
        inputName: String,
        outputName: String,
        normalization: FACADEEmbeddingNormalization
    ) throws {
        guard artifact.role == expectedRole,
              artifact.storagePrecision == .int8,
              artifact.gradientSemantics == .none,
              artifact.inputs.count == 1,
              artifact.outputs.count == 1,
              artifact.inputs[0].name == inputName,
              artifact.inputs[0].shape == [1, 3, 112, 112],
              artifact.outputs[0].name == outputName,
              artifact.outputs[0].shape == [1, 512] else {
            throw FACADEProtectionError.modelUnavailable(
                "the \(expectedRole.rawValue) Core ML contract is incompatible"
            )
        }
        self.model = model
        self.sourceWeightsSHA256 = artifact.sourceWeightsSHA256
        self.inputName = inputName
        self.outputName = outputName
        self.normalization = normalization
    }

    public func embedding(for alignedRGB: FACADEFloatImage) throws -> [Float] {
        guard alignedRGB.width == 112, alignedRGB.height == 112 else {
            throw FACADEProtectionError.verificationUnavailable("embedding input must be 112x112 RGB")
        }
        let input = try MLMultiArray(
            shape: [1, 3, 112, 112] as [NSNumber],
            dataType: .float32
        )
        let destination = input.dataPointer.bindMemory(to: Float.self, capacity: input.count)
        let plane = 112 * 112
        for channel in 0..<3 {
            for pixel in 0..<plane {
                let value = alignedRGB.values[channel * plane + pixel]
                destination[channel * plane + pixel] = normalized(value, channel: channel)
            }
        }
        let provider = try MLDictionaryFeatureProvider(dictionary: [inputName: input])
        let prediction: MLFeatureProvider
        do {
            prediction = try model.prediction(from: provider)
        } catch {
            throw FACADEProtectionError.verificationUnavailable(
                "\(outputName) prediction failed: \(error)"
            )
        }
        guard let output = prediction.featureValue(for: outputName)?.multiArrayValue,
              output.shape.map(\.intValue) == [1, 512], output.count == 512 else {
            throw FACADEProtectionError.verificationUnavailable(
                "\(outputName) returned an incompatible tensor"
            )
        }
        let values = (0..<512).map { output[$0].floatValue }
        guard values.allSatisfy(\.isFinite) else {
            throw FACADEProtectionError.verificationUnavailable(
                "\(outputName) returned non-finite values"
            )
        }
        let norm = sqrt(values.reduce(0) { $0 + $1 * $1 })
        guard norm.isFinite, abs(norm - 1) <= 0.02 else {
            throw FACADEProtectionError.verificationUnavailable(
                "\(outputName) is not an L2-normalized embedding"
            )
        }
        return values
    }

    private func normalized(_ value: Float, channel: Int) -> Float {
        switch normalization {
        case .arcFaceMinusOneToOne:
            return value * 2 - 1
        case .imageNetRGB:
            let mean: [Float] = [0.485, 0.456, 0.406]
            let standardDeviation: [Float] = [0.229, 0.224, 0.225]
            return (value - mean[channel]) / standardDeviation[channel]
        }
    }
}

public struct FACADEAlignedGradientResult: Sendable, Equatable {
    public let score: Float
    public let gradientMinusOneToOne: FACADEFloatImage

    public init(score: Float, gradientMinusOneToOne: FACADEFloatImage) throws {
        guard score.isFinite,
              gradientMinusOneToOne.width == 112,
              gradientMinusOneToOne.height == 112 else {
            throw FACADEProtectionError.candidateGenerationFailed(
                "the MPSGraph score/input-gradient result is invalid"
            )
        }
        self.score = score
        self.gradientMinusOneToOne = gradientMinusOneToOne
    }
}

public protocol FACADEArcFaceGradientExecutor: Sendable {
    func execute(
        alignedCandidateMinusOneToOne: FACADEFloatImage,
        referenceEmbedding: [Float]
    ) throws -> FACADEAlignedGradientResult
}

/// Executes a host-materialized backward graph as a normal Core ML model. The
/// model has actual frozen ArcFace-family weights and explicit score/gradient
/// outputs; Core ML does not perform runtime autodiff.
public final class FACADECoreMLStaticScoreGradientExecutor: @unchecked Sendable,
    FACADEArcFaceGradientExecutor {
    private let model: MLModel

    public init(model: MLModel, artifact: FACADEModelArtifact) throws {
        guard artifact.role == .arcFaceScoreGradient,
              artifact.format == .coreMLPackage || artifact.format == .compiledCoreML,
              artifact.gradientSemantics == .staticGradientOutput,
              artifact.inputs.map(\.name) == [
                  "aligned_rgb_nchw_minus1_to1", "reference_embedding_512",
              ],
              artifact.inputs.map(\.shape) == [[1, 3, 112, 112], [1, 512]],
              artifact.outputs.map(\.name) == ["score", "candidate_input_gradient_nchw"],
              artifact.outputs.map(\.shape) == [[1], [1, 3, 112, 112]],
              artifact.executionPrecision != .int8 else {
            throw FACADEProtectionError.modelUnavailable(
                "the static ArcFace score-gradient Core ML contract is incompatible"
            )
        }
        self.model = model
    }

    public func execute(
        alignedCandidateMinusOneToOne: FACADEFloatImage,
        referenceEmbedding: [Float]
    ) throws -> FACADEAlignedGradientResult {
        guard alignedCandidateMinusOneToOne.width == 112,
              alignedCandidateMinusOneToOne.height == 112,
              referenceEmbedding.count == 512,
              referenceEmbedding.allSatisfy(\.isFinite) else {
            throw FACADEProtectionError.candidateGenerationFailed(
                "the static score-gradient inputs are invalid"
            )
        }
        let candidate = try MLMultiArray(
            shape: [1, 3, 112, 112] as [NSNumber],
            dataType: .float32
        )
        let reference = try MLMultiArray(shape: [1, 512] as [NSNumber], dataType: .float32)
        let candidatePointer = candidate.dataPointer.bindMemory(
            to: Float.self,
            capacity: candidate.count
        )
        let referencePointer = reference.dataPointer.bindMemory(
            to: Float.self,
            capacity: reference.count
        )
        alignedCandidateMinusOneToOne.values.withUnsafeBufferPointer { source in
            candidatePointer.update(from: source.baseAddress!, count: source.count)
        }
        referenceEmbedding.withUnsafeBufferPointer { source in
            referencePointer.update(from: source.baseAddress!, count: source.count)
        }
        let provider = try MLDictionaryFeatureProvider(
            dictionary: [
                "aligned_rgb_nchw_minus1_to1": candidate,
                "reference_embedding_512": reference,
            ]
        )
        let prediction: MLFeatureProvider
        do {
            prediction = try model.prediction(from: provider)
        } catch {
            throw FACADEProtectionError.candidateGenerationFailed(
                "static ArcFace score-gradient prediction failed: \(error)"
            )
        }
        guard let scoreOutput = prediction.featureValue(for: "score")?.multiArrayValue,
              scoreOutput.count == 1,
              let gradientOutput = prediction.featureValue(
                  for: "candidate_input_gradient_nchw"
              )?.multiArrayValue,
              gradientOutput.shape.map(\.intValue) == [1, 3, 112, 112],
              gradientOutput.count == 3 * 112 * 112 else {
            throw FACADEProtectionError.candidateGenerationFailed(
                "static ArcFace score-gradient output contract changed"
            )
        }
        return try FACADEAlignedGradientResult(
            score: scoreOutput[0].floatValue,
            gradientMinusOneToOne: FACADEFloatImage(
                width: 112,
                height: 112,
                values: (0..<gradientOutput.count).map {
                    gradientOutput[$0].floatValue
                }
            )
        )
    }
}

/// Executes the delivered actual-weight MPSGraph package. Argument and result
/// ordering is frozen by the manifest: candidate, reference; score, gradient.
public final class FACADEMPSGraphScoreGradientExecutor: @unchecked Sendable,
    FACADEArcFaceGradientExecutor {
    private let executable: MPSGraphExecutable
    private let graphDevice: MPSGraphDevice
    private let commandQueue: any MTLCommandQueue

    public init(packageURL: URL, artifact: FACADEModelArtifact) throws {
        guard artifact.role == .arcFaceScoreGradient,
              artifact.format == .mpsGraphPackage,
              artifact.gradientSemantics != .none,
              artifact.inputs.map(\.name) == [
                  "aligned_rgb_nchw_minus1_to1", "reference_embedding_512",
              ],
              artifact.inputs.map(\.shape) == [[1, 3, 112, 112], [1, 512]],
              artifact.outputs.map(\.name) == ["score", "candidate_input_gradient_nchw"],
              artifact.outputs.map(\.shape) == [[1], [1, 3, 112, 112]],
              artifact.executionPrecision == .float16 else {
            throw FACADEProtectionError.modelUnavailable(
                "the ArcFace score-gradient package contract is incompatible"
            )
        }
        guard let metalDevice = MTLCreateSystemDefaultDevice(),
              let queue = metalDevice.makeCommandQueue() else {
            throw FACADEProtectionError.modelUnavailable("Metal is unavailable")
        }
        self.executable = MPSGraphExecutable(package: packageURL, descriptor: nil)
        self.graphDevice = MPSGraphDevice(mtlDevice: metalDevice)
        self.commandQueue = queue
        self.executable.options = .synchronizeResults
    }

    public func execute(
        alignedCandidateMinusOneToOne: FACADEFloatImage,
        referenceEmbedding: [Float]
    ) throws -> FACADEAlignedGradientResult {
        guard alignedCandidateMinusOneToOne.width == 112,
              alignedCandidateMinusOneToOne.height == 112,
              referenceEmbedding.count == 512,
              referenceEmbedding.allSatisfy(\.isFinite) else {
            throw FACADEProtectionError.candidateGenerationFailed(
                "the MPSGraph score-gradient inputs are invalid"
            )
        }
        let candidate = alignedCandidateMinusOneToOne.values.map(Float16.init)
        let reference = referenceEmbedding.map(Float16.init)
        let candidateData = candidate.withUnsafeBytes { Data($0) }
        let referenceData = reference.withUnsafeBytes { Data($0) }
        let candidateTensor = MPSGraphTensorData(
            device: graphDevice,
            data: candidateData,
            shape: [1, 3, 112, 112] as [NSNumber],
            dataType: .float16
        )
        let referenceTensor = MPSGraphTensorData(
            device: graphDevice,
            data: referenceData,
            shape: [1, 512] as [NSNumber],
            dataType: .float16
        )
        let outputs = executable.run(
            with: commandQueue,
            inputs: [candidateTensor, referenceTensor],
            results: nil,
            executionDescriptor: nil
        )
        guard outputs.count == 2,
              outputs[0].shape.map(\.intValue) == [1],
              outputs[1].shape.map(\.intValue) == [1, 3, 112, 112] else {
            throw FACADEProtectionError.candidateGenerationFailed(
                "the MPSGraph package returned incompatible results"
            )
        }
        var score = [Float16](repeating: 0, count: 1)
        var gradient = [Float16](repeating: 0, count: 3 * 112 * 112)
        score.withUnsafeMutableBytes { buffer in
            outputs[0].mpsndarray().readBytes(buffer.baseAddress!, strideBytes: nil)
        }
        gradient.withUnsafeMutableBytes { buffer in
            outputs[1].mpsndarray().readBytes(buffer.baseAddress!, strideBytes: nil)
        }
        return try FACADEAlignedGradientResult(
            score: Float(score[0]),
            gradientMinusOneToOne: FACADEFloatImage(
                width: 112,
                height: 112,
                values: gradient.map(Float.init)
            )
        )
    }
}

public actor FACADEAppleS1BInputGradientObjective: FACADEInputGradientObjective {
    private let detector: any FACADEExactlyOneFaceDetector
    private let arcFace: FACADECoreMLEmbeddingModel
    private let executor: any FACADEArcFaceGradientExecutor
    private let identityMap: FACADEIdentityMapMatrix
    private var cachedClean: FACADEFloatImage?
    private var cachedTransform: FACADEFaceCropTransform?
    private var cachedReferenceEmbedding: [Float]?

    public init(
        detector: any FACADEExactlyOneFaceDetector,
        arcFace: FACADECoreMLEmbeddingModel,
        executor: any FACADEArcFaceGradientExecutor,
        identityMap: FACADEIdentityMapMatrix
    ) {
        self.detector = detector
        self.arcFace = arcFace
        self.executor = executor
        self.identityMap = identityMap
    }

    public func evaluate(
        clean: FACADEFloatImage,
        candidate: FACADEFloatImage
    ) async throws -> FACADEObjectiveGradient {
        let objectiveStarted = DispatchTime.now().uptimeNanoseconds
        var stageLatencyMilliseconds: [String: Double] = [:]
        let referenceStarted = DispatchTime.now().uptimeNanoseconds
        let transform: FACADEFaceCropTransform
        let referenceEmbedding: [Float]
        if cachedClean == clean,
           let existingTransform = cachedTransform,
           let existingReference = cachedReferenceEmbedding {
            transform = existingTransform
            referenceEmbedding = existingReference
        } else {
            transform = try detector.alignmentTransform(in: clean)
            // r_hat is the forward verifier's L2-normalized clean embedding;
            // the graph's runtime reference is v = normalize(E E^T r_hat), the
            // vector InSwapper actually consumes. +v is passed unchanged and
            // the engine descends the score. A clean crop scores roughly
            // 0.24-0.34 against v, not 1.0.
            let rHat = try arcFace.embedding(for: transform.alignedImage(from: clean))
            referenceEmbedding = try identityMap.projectedReference(rHat)
            cachedClean = clean
            cachedTransform = transform
            cachedReferenceEmbedding = referenceEmbedding
        }
        stageLatencyMilliseconds["reference_setup"] = Self.milliseconds(
            since: referenceStarted
        )

        let alignmentStarted = DispatchTime.now().uptimeNanoseconds
        let alignedCandidate = try transform.alignedImage(from: candidate)
        let minusOneToOne = try FACADEFloatImage(
            width: 112,
            height: 112,
            values: alignedCandidate.values.map { $0 * 2 - 1 }
        )
        stageLatencyMilliseconds["candidate_alignment"] = Self.milliseconds(
            since: alignmentStarted
        )
        let executorStarted = DispatchTime.now().uptimeNanoseconds
        let result = try executor.execute(
            alignedCandidateMinusOneToOne: minusOneToOne,
            referenceEmbedding: referenceEmbedding
        )
        stageLatencyMilliseconds["static_gradient_execute"] = Self.milliseconds(
            since: executorStarted
        )
        // d(x*2-1)/dx = 2 before applying the transpose of the crop sampler.
        let backProjectionStarted = DispatchTime.now().uptimeNanoseconds
        let zeroToOneGradient = try FACADEFloatImage(
            width: 112,
            height: 112,
            values: result.gradientMinusOneToOne.values.map { $0 * 2 }
        )
        let fullGradient = try transform.backProject(
            alignedGradient: zeroToOneGradient,
            sourceWidth: candidate.width,
            sourceHeight: candidate.height
        )
        stageLatencyMilliseconds["gradient_backprojection"] = Self.milliseconds(
            since: backProjectionStarted
        )
        stageLatencyMilliseconds["total_objective"] = Self.milliseconds(
            since: objectiveStarted
        )
        return try FACADEObjectiveGradient(
            score: result.score,
            inputGradient: fullGradient,
            stageLatencyMilliseconds: stageLatencyMilliseconds
        )
    }

    private static func milliseconds(since started: UInt64) -> Double {
        Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000.0
    }
}

enum FACADEApplePixelBridge {
    static func makeCGImage(_ image: FACADEFloatImage) throws -> CGImage {
        let plane = image.width * image.height
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

    static func encode(
        _ image: FACADEFloatImage,
        typeIdentifier: String,
        quality: Double
    ) throws -> Data {
        let output = NSMutableData()
        let supportedTypes = CGImageDestinationCopyTypeIdentifiers() as NSArray
        guard supportedTypes.contains(typeIdentifier),
              let destination = CGImageDestinationCreateWithData(
                  output,
                  typeIdentifier as CFString,
                  1,
                  nil
              ) else {
            throw FACADEProtectionError.verificationUnavailable(
                "the required local transform encoder is unavailable: \(typeIdentifier)"
            )
        }
        CGImageDestinationAddImage(
            destination,
            try makeCGImage(image),
            [kCGImageDestinationLossyCompressionQuality: quality] as CFDictionary
        )
        guard CGImageDestinationFinalize(destination) else {
            throw FACADEProtectionError.verificationUnavailable(
                "the required local transform could not be encoded: \(typeIdentifier)"
            )
        }
        return output as Data
    }
}
