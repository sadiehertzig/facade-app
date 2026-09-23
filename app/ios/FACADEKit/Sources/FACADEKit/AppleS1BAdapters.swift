import CoreGraphics
import CoreImage
import CoreML
import Foundation
import ImageIO
import UniformTypeIdentifiers

public final class FACADECoreMLR6InitializerBank: @unchecked Sendable, FACADEInitializerResidualBank {
    public let sourceCheckpointSHA256: String

    private let model: MLModel
    private let inputName: String
    private let outputName: String

    public init(
        model: MLModel,
        handoff: FACADELoadedModelHandoff,
        inputName: String = "image_nchw_0_1",
        outputName: String = "residual_bank_n24chw"
    ) throws {
        guard let artifact = handoff.manifest.artifacts.first(where: { $0.role == .protectionEngine }),
              artifact.sourceWeightsSHA256 == FACADEExperimentalS1BEngine.frozenR6CheckpointSHA256,
              artifact.storagePrecision == .int8,
              artifact.inputs.count == 1,
              artifact.outputs.count == 1,
              artifact.inputs[0].name == inputName,
              artifact.inputs[0].shape == [1, 3, 256, 256],
              artifact.outputs[0].name == outputName,
              artifact.outputs[0].shape == [1, 24, 256, 256] else {
            throw FACADEProtectionError.modelUnavailable(
                "the installed protection artifact does not match the frozen r6 Core ML contract"
            )
        }
        self.model = model
        self.sourceCheckpointSHA256 = artifact.sourceWeightsSHA256
        self.inputName = inputName
        self.outputName = outputName
    }

    public func residuals(for image: FACADEFloatImage) async throws -> [FACADEFloatImage] {
        guard image.width == 256, image.height == 256 else {
            throw FACADEProtectionError.invalidInput("the Core ML r6 initializer requires 256x256 RGB")
        }
        let input = try MLMultiArray(
            shape: [1, 3, 256, 256] as [NSNumber],
            dataType: .float32
        )
        guard input.count == image.values.count else {
            throw FACADEProtectionError.modelUnavailable("Core ML allocated an unexpected input tensor")
        }
        let destination = input.dataPointer.bindMemory(to: Float.self, capacity: input.count)
        image.values.withUnsafeBufferPointer { source in
            guard let address = source.baseAddress else { return }
            destination.update(from: address, count: source.count)
        }

        let provider = try MLDictionaryFeatureProvider(dictionary: [inputName: input])
        let prediction: MLFeatureProvider
        do {
            prediction = try await model.prediction(from: provider)
        } catch {
            throw FACADEProtectionError.candidateGenerationFailed(
                "r6 Core ML initializer prediction failed: \(error)"
            )
        }
        guard let output = prediction.featureValue(for: outputName)?.multiArrayValue,
              output.shape.map(\.intValue) == [1, 24, 256, 256],
              output.count == 24 * 256 * 256 else {
            throw FACADEProtectionError.modelUnavailable("r6 Core ML initializer output contract changed")
        }

        let plane = 256 * 256
        var heads: [FACADEFloatImage] = []
        heads.reserveCapacity(8)
        for head in 0..<8 {
            var values = [Float](repeating: 0, count: 3 * plane)
            for index in 0..<values.count {
                let value = output[head * 3 * plane + index].floatValue
                guard value.isFinite else {
                    throw FACADEProtectionError.candidateGenerationFailed(
                        "r6 Core ML initializer returned a non-finite residual"
                    )
                }
                values[index] = value
            }
            heads.append(try FACADEFloatImage(width: 256, height: 256, values: values))
        }
        return heads
    }
}

public final class FACADECoreImagePNGCodec: @unchecked Sendable, FACADELocalVerificationImageCodec {
    public let side: Int
    private let context: CIContext

    public init(side: Int = 256) {
        self.side = side
        self.context = CIContext(options: [.cacheIntermediates: false])
    }

    public func decodeAndPrepare(_ data: Data) throws -> FACADEFloatImage {
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
        let scale = min(CGFloat(side) / extent.width, CGFloat(side) / extent.height)
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
        let translated = resized.transformed(
            by: CGAffineTransform(
                translationX: (CGFloat(side) - resized.extent.width) * 0.5 - resized.extent.origin.x,
                y: (CGFloat(side) - resized.extent.height) * 0.5 - resized.extent.origin.y
            )
        )
        let bounds = CGRect(x: 0, y: 0, width: side, height: side)
        let background = CIImage(
            color: CIColor(red: 127.0 / 255.0, green: 127.0 / 255.0, blue: 127.0 / 255.0)
        ).cropped(to: bounds)
        let prepared = translated.composited(over: background).cropped(to: bounds)
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

        let plane = side * side
        var values = [Float](repeating: 0, count: 3 * plane)
        for pixel in 0..<plane {
            values[pixel] = Float(bytes[pixel * 4]) / 255.0
            values[plane + pixel] = Float(bytes[pixel * 4 + 1]) / 255.0
            values[2 * plane + pixel] = Float(bytes[pixel * 4 + 2]) / 255.0
        }
        return try FACADEFloatImage(width: side, height: side, values: values)
    }

    public func encodePNG(_ image: FACADEFloatImage) throws -> Data {
        guard image.width == side, image.height == side else {
            throw FACADEProtectionError.candidateGenerationFailed("the protected image has the wrong dimensions")
        }
        let plane = side * side
        var bytes = [UInt8](repeating: 255, count: plane * 4)
        for pixel in 0..<plane {
            for channel in 0..<3 {
                let value = image.values[channel * plane + pixel]
                bytes[pixel * 4 + channel] = UInt8(
                    min(255, max(0, Int((value * 255.0).rounded())))
                )
            }
        }
        guard let provider = CGDataProvider(data: Data(bytes) as CFData),
              let image = CGImage(
                  width: side,
                  height: side,
                  bitsPerComponent: 8,
                  bitsPerPixel: 32,
                  bytesPerRow: side * 4,
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
            throw FACADEProtectionError.candidateGenerationFailed("the protected bitmap could not be created")
        }
        let output = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(
            output,
            UTType.png.identifier as CFString,
            1,
            nil
        ) else {
            throw FACADEProtectionError.candidateGenerationFailed("the PNG encoder is unavailable")
        }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else {
            throw FACADEProtectionError.candidateGenerationFailed("the protected PNG could not be encoded")
        }
        return output as Data
    }

    public func jpegQ70RoundTrip(
        _ image: FACADEFloatImage
    ) throws -> FACADEVerificationRoundTrip {
        try verificationRoundTrip(
            image,
            typeIdentifier: UTType.jpeg.identifier,
            quality: 0.70
        )
    }

    public func heicQ80RoundTrip(
        _ image: FACADEFloatImage
    ) throws -> FACADEVerificationRoundTrip {
        try verificationRoundTrip(
            image,
            typeIdentifier: UTType.heic.identifier,
            quality: 0.80
        )
    }

    private func verificationRoundTrip(
        _ image: FACADEFloatImage,
        typeIdentifier: String,
        quality: Double
    ) throws -> FACADEVerificationRoundTrip {
        let encoded = try FACADEApplePixelBridge.encode(
            image,
            typeIdentifier: typeIdentifier,
            quality: quality
        )
        return FACADEVerificationRoundTrip(
            image: try decodeAndPrepare(encoded),
            encodedByteCount: encoded.count
        )
    }
}
