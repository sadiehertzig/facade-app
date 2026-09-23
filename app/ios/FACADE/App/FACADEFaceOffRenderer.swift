import CoreImage
import CoreML
import CryptoKit
import Foundation
import ImageIO
import Vision

struct FACADEFaceOffRenderedResult: Equatable, Sendable {
    let syntheticTargetData: Data
    let attemptedSwapData: Data
    let sourceEmbedding: [Float]
    /// Recognizer vector of the swapped face (rows 0..<112, columns 8..<120 of the
    /// 128 px render), for a display-only similarity to the source.
    let swapEmbedding: [Float]
    /// The 128 px swapped face before compositing, for a close-up view.
    let alignedSwapFaceData: Data
    let runtimeSeconds: Double
}

protocol FACADEFaceOffRendering: Sendable {
    func render(sourceImageData: Data) async throws -> FACADEFaceOffRenderedResult
}

enum FACADEFaceOffError: LocalizedError {
    case missingResource(String)
    case invalidResource(String)
    case faceCount(label: String, count: Int)
    case missingLandmarks(String)
    case invalidAlignment(String)
    case modelFailure(String)
    case renderFailure(String)

    var errorDescription: String? {
        switch self {
        case .missingResource(let name):
            "The local Face-Off resource \(name) is missing."
        case .invalidResource(let reason):
            "A local Face-Off resource is invalid: \(reason)"
        case .faceCount(let label, let count):
            "Face-Off needs exactly one clear face in the \(label); it found \(count)."
        case .missingLandmarks(let label):
            "Face-Off could not find stable eye and nose landmarks in the \(label)."
        case .invalidAlignment(let label):
            "Face-Off could not align the \(label) safely."
        case .modelFailure(let reason):
            "The local Face-Off model stopped: \(reason)"
        case .renderFailure(let reason):
            "Face-Off could not create the attempted swap image: \(reason)"
        }
    }
}

struct FACADEUnavailableFaceOffRenderer: FACADEFaceOffRendering {
    let reason: String

    func render(sourceImageData: Data) async throws -> FACADEFaceOffRenderedResult {
        throw FACADEFaceOffError.missingResource(reason)
    }
}

actor FACADEOnDeviceFaceOffRenderer: FACADEFaceOffRendering {
    static let targetSHA256 =
        "f27ac119abfa60e77d5e6cb82935f66fe7b0b3f0a86f9de3b0a6fa50e9263e8a"
    /// Preferred Face-Off target when Models/FaceOffTarget/target.png is bundled.
    static let preferredTargetSHA256 =
        "874e23015c4958393b6cc232cf4b2be161efd3907173339fa43e660510dbae3a"
    static let identityMapSHA256 =
        "370af5bf707dafdbea8a40448d697d9697610bd223ecf92887af9c9cc7055ac8"

    private struct Alignment {
        let alignedImage: CIImage
        let sourceToCanonical: CGAffineTransform
    }

    private enum ImageNormalization {
        case minusOneToOne
        case zeroToOne
    }

    private let recognizerURL: URL
    private let rendererURL: URL
    private let identityMapURL: URL
    private let targetURL: URL
    private let expectedTargetSHA256: String
    private let context = CIContext(options: [
        .cacheIntermediates: false,
        .workingColorSpace: CGColorSpace(name: CGColorSpace.sRGB) as Any,
        .outputColorSpace: CGColorSpace(name: CGColorSpace.sRGB) as Any,
    ])
    private var recognizer: MLModel?
    private var renderer: MLModel?
    private var identityMap: [Float32]?

    static func load(bundle: Bundle = .main) throws -> FACADEOnDeviceFaceOffRenderer {
        guard let recognizerURL = bundle.url(
            forResource: "recognizer-float32",
            withExtension: "mlmodelc",
            subdirectory: "FaceOffModels"
        ) else {
            throw FACADEFaceOffError.missingResource("recognizer-float32.mlmodelc")
        }
        guard let rendererURL = bundle.url(
            forResource: "renderer",
            withExtension: "mlmodelc",
            subdirectory: "FaceOffModels"
        ) else {
            throw FACADEFaceOffError.missingResource("renderer.mlmodelc")
        }
        guard let identityMapURL = bundle.url(
            forResource: "identity-map-float32",
            withExtension: "bin",
            subdirectory: "FaceOffModels"
        ) else {
            throw FACADEFaceOffError.missingResource("identity-map-float32.bin")
        }
        if let slideTargetURL = bundle.url(
            forResource: "target",
            withExtension: "png",
            subdirectory: "FaceOffTarget"
        ) {
            return FACADEOnDeviceFaceOffRenderer(
                recognizerURL: recognizerURL,
                rendererURL: rendererURL,
                identityMapURL: identityMapURL,
                targetURL: slideTargetURL,
                targetSHA256: preferredTargetSHA256
            )
        }
        guard let targetURL = bundle.url(
            forResource: "synthetic_blonde_woman_faceoff_target",
            withExtension: "png",
            subdirectory: "FaceOffAssets"
        ) else {
            throw FACADEFaceOffError.missingResource(
                "FaceOffAssets/synthetic_blonde_woman_faceoff_target.png"
            )
        }
        return load(
            recognizerURL: recognizerURL,
            rendererURL: rendererURL,
            identityMapURL: identityMapURL,
            targetURL: targetURL
        )
    }

    static func load(
        recognizerURL: URL,
        rendererURL: URL,
        identityMapURL: URL,
        targetURL: URL
    ) -> FACADEOnDeviceFaceOffRenderer {
        FACADEOnDeviceFaceOffRenderer(
            recognizerURL: recognizerURL,
            rendererURL: rendererURL,
            identityMapURL: identityMapURL,
            targetURL: targetURL,
            targetSHA256: targetSHA256
        )
    }

    private init(
        recognizerURL: URL,
        rendererURL: URL,
        identityMapURL: URL,
        targetURL: URL,
        targetSHA256: String
    ) {
        self.recognizerURL = recognizerURL
        self.rendererURL = rendererURL
        self.identityMapURL = identityMapURL
        self.targetURL = targetURL
        self.expectedTargetSHA256 = targetSHA256
    }

    func render(sourceImageData: Data) async throws -> FACADEFaceOffRenderedResult {
        let clock = ContinuousClock()
        let started = clock.now
        try Task.checkCancellation()

        guard !sourceImageData.isEmpty else {
            throw FACADEFaceOffError.invalidResource("the source image is empty")
        }
        let targetData = try Data(contentsOf: targetURL, options: [.mappedIfSafe])
        let targetDigest = SHA256.hash(data: targetData)
            .map { String(format: "%02x", $0) }
            .joined()
        guard targetDigest == expectedTargetSHA256 else {
            throw FACADEFaceOffError.invalidResource(
                "the synthetic target checksum does not match"
            )
        }

        let sourceImage = try decodeImage(
            sourceImageData,
            label: "source selfie"
        )
        let targetImage = try decodeImage(targetData, label: "synthetic target")
        let sourceAlignment = try align(
            sourceImage,
            label: "source selfie",
            size: 112,
            horizontalOffset: 0
        )
        let targetAlignment = try align(
            targetImage,
            label: "synthetic target",
            size: 128,
            horizontalOffset: 8
        )

        try Task.checkCancellation()
        let models = try loadModelsIfNeeded()
        let sourceTensor = try imageTensor(
            from: sourceAlignment.alignedImage,
            size: 112,
            normalization: .minusOneToOne
        )
        let sourceEmbedding = try recognize(
            sourceTensor,
            with: models.recognizer
        )
        let rendererIdentity = try mapIdentity(
            sourceEmbedding,
            with: loadIdentityMapIfNeeded()
        )
        let targetTensor = try imageTensor(
            from: targetAlignment.alignedImage,
            size: 128,
            normalization: .zeroToOne
        )
        let renderedTensor = try renderFace(
            target: targetTensor,
            sourceIdentity: rendererIdentity,
            with: models.renderer
        )
        try Task.checkCancellation()

        let alignedRenderedFace = try image(from: renderedTensor, size: 128)
        // Same face window the desktop graders score: top 112 rows, columns 8..<120.
        let swapCrop = alignedRenderedFace
            .cropped(to: CGRect(x: 8, y: 16, width: 112, height: 112))
            .transformed(by: CGAffineTransform(translationX: -8, y: -16))
        let swapEmbedding = try recognize(
            try imageTensor(from: swapCrop, size: 112, normalization: .minusOneToOne),
            with: models.recognizer
        )
        guard let sRGB = CGColorSpace(name: CGColorSpace.sRGB),
              let alignedSwapFaceData = context.pngRepresentation(
                of: alignedRenderedFace, format: .RGBA8, colorSpace: sRGB) else {
            throw FACADEFaceOffError.renderFailure("the swapped face could not be encoded")
        }
        let composited = try composite(
            alignedRenderedFace: alignedRenderedFace,
            targetImage: targetImage,
            targetAlignment: targetAlignment
        )
        let elapsed = started.duration(to: clock.now)
        let seconds = Double(elapsed.components.seconds) +
            Double(elapsed.components.attoseconds) / 1_000_000_000_000_000_000
        return FACADEFaceOffRenderedResult(
            syntheticTargetData: targetData,
            attemptedSwapData: composited,
            sourceEmbedding: embeddingValues(sourceEmbedding),
            swapEmbedding: embeddingValues(swapEmbedding),
            alignedSwapFaceData: alignedSwapFaceData,
            runtimeSeconds: seconds
        )
    }

    private func embeddingValues(_ embedding: MLMultiArray) -> [Float] {
        let pointer = embedding.dataPointer.bindMemory(
            to: Float32.self,
            capacity: embedding.count
        )
        return Array(UnsafeBufferPointer(start: pointer, count: embedding.count))
    }

    private func loadModelsIfNeeded() throws -> (recognizer: MLModel, renderer: MLModel) {
        if let recognizer, let renderer {
            return (recognizer, renderer)
        }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all
        do {
            let loadedRecognizer = try MLModel(
                contentsOf: recognizerURL,
                configuration: configuration
            )
            let loadedRenderer = try MLModel(
                contentsOf: rendererURL,
                configuration: configuration
            )
            recognizer = loadedRecognizer
            renderer = loadedRenderer
            return (loadedRecognizer, loadedRenderer)
        } catch {
            throw FACADEFaceOffError.modelFailure(error.localizedDescription)
        }
    }

    private func loadIdentityMapIfNeeded() throws -> [Float32] {
        if let identityMap {
            return identityMap
        }
        let data: Data
        do {
            data = try Data(contentsOf: identityMapURL, options: [.mappedIfSafe])
        } catch {
            throw FACADEFaceOffError.invalidResource(
                "the InSwapper identity map cannot be read"
            )
        }
        guard data.count == 512 * 512 * MemoryLayout<Float32>.size else {
            throw FACADEFaceOffError.invalidResource(
                "the InSwapper identity map has an unexpected byte count"
            )
        }
        let digest = SHA256.hash(data: data)
            .map { String(format: "%02x", $0) }
            .joined()
        guard digest == Self.identityMapSHA256 else {
            throw FACADEFaceOffError.invalidResource(
                "the InSwapper identity-map checksum does not match"
            )
        }

        var values = [Float32]()
        values.reserveCapacity(512 * 512)
        data.withUnsafeBytes { buffer in
            guard let baseAddress = buffer.baseAddress else { return }
            let bytes = baseAddress.assumingMemoryBound(to: UInt8.self)
            for index in 0..<(512 * 512) {
                let offset = index * 4
                let bitPattern = UInt32(bytes[offset]) |
                    UInt32(bytes[offset + 1]) << 8 |
                    UInt32(bytes[offset + 2]) << 16 |
                    UInt32(bytes[offset + 3]) << 24
                values.append(Float32(bitPattern: bitPattern))
            }
        }
        guard values.count == 512 * 512, values.allSatisfy({ $0.isFinite }) else {
            throw FACADEFaceOffError.invalidResource(
                "the InSwapper identity map contains invalid values"
            )
        }
        identityMap = values
        return values
    }

    private func decodeImage(_ data: Data, label: String) throws -> CIImage {
        guard let decoded = CIImage(
            data: data,
            options: [.applyOrientationProperty: true]
        ) else {
            throw FACADEFaceOffError.invalidResource("the \(label) cannot be decoded")
        }
        let extent = decoded.extent.integral
        guard extent.width >= 128, extent.height >= 128 else {
            throw FACADEFaceOffError.invalidResource("the \(label) is too small")
        }
        return decoded.transformed(
            by: CGAffineTransform(translationX: -extent.minX, y: -extent.minY)
        )
    }

    private func align(
        _ image: CIImage,
        label: String,
        size: CGFloat,
        horizontalOffset: CGFloat
    ) throws -> Alignment {
        let request = VNDetectFaceLandmarksRequest()
        let handler = VNImageRequestHandler(ciImage: image, orientation: .up)
        do {
            try handler.perform([request])
        } catch {
            throw FACADEFaceOffError.invalidAlignment(label)
        }
        let faces = request.results ?? []
        guard faces.count == 1, let face = faces.first else {
            throw FACADEFaceOffError.faceCount(label: label, count: faces.count)
        }
        guard let landmarks = face.landmarks,
              let leftEye = landmarks.leftEye,
              let rightEye = landmarks.rightEye,
              let nose = landmarks.nose else {
            throw FACADEFaceOffError.missingLandmarks(label)
        }

        let source = [leftEye, rightEye, nose].map {
            imagePoint(for: average($0.normalizedPoints), face: face, image: image)
        }
        // Eye and nose points of the standard ArcFace template (InsightFace, MIT-licensed code).
        let destination = [
            CGPoint(x: 38.2946 + horizontalOffset, y: size - 51.6963),
            CGPoint(x: 73.5318 + horizontalOffset, y: size - 51.5014),
            CGPoint(x: 56.0252 + horizontalOffset, y: size - 71.7366),
        ]
        let transform = try affineTransform(
            source: source,
            destination: destination,
            label: label
        )
        let bounds = CGRect(x: 0, y: 0, width: size, height: size)
        let aligned = image.transformed(by: transform).cropped(to: bounds)
        return Alignment(alignedImage: aligned, sourceToCanonical: transform)
    }

    private func average(_ points: [CGPoint]) -> CGPoint {
        guard !points.isEmpty else { return .zero }
        let sum = points.reduce(CGPoint.zero) {
            CGPoint(x: $0.x + $1.x, y: $0.y + $1.y)
        }
        return CGPoint(x: sum.x / CGFloat(points.count), y: sum.y / CGFloat(points.count))
    }

    private func imagePoint(
        for point: CGPoint,
        face: VNFaceObservation,
        image: CIImage
    ) -> CGPoint {
        let box = face.boundingBox
        let normalized = CGPoint(
            x: box.minX + point.x * box.width,
            y: box.minY + point.y * box.height
        )
        return CGPoint(
            x: normalized.x * image.extent.width,
            y: normalized.y * image.extent.height
        )
    }

    private func affineTransform(
        source: [CGPoint],
        destination: [CGPoint],
        label: String
    ) throws -> CGAffineTransform {
        guard source.count == 3, destination.count == 3 else {
            throw FACADEFaceOffError.invalidAlignment(label)
        }
        let sourceX1 = source[1].x - source[0].x
        let sourceY1 = source[1].y - source[0].y
        let sourceX2 = source[2].x - source[0].x
        let sourceY2 = source[2].y - source[0].y
        let destinationX1 = destination[1].x - destination[0].x
        let destinationY1 = destination[1].y - destination[0].y
        let destinationX2 = destination[2].x - destination[0].x
        let destinationY2 = destination[2].y - destination[0].y
        let determinant = sourceX1 * sourceY2 - sourceX2 * sourceY1
        guard determinant.isFinite, abs(determinant) > 0.000_001 else {
            throw FACADEFaceOffError.invalidAlignment(label)
        }

        let a = (destinationX1 * sourceY2 - destinationX2 * sourceY1) /
            determinant
        let c = (-destinationX1 * sourceX2 + destinationX2 * sourceX1) /
            determinant
        let b = (destinationY1 * sourceY2 - destinationY2 * sourceY1) /
            determinant
        let d = (-destinationY1 * sourceX2 + destinationY2 * sourceX1) /
            determinant
        let tx = destination[0].x - a * source[0].x - c * source[0].y
        let ty = destination[0].y - b * source[0].x - d * source[0].y
        let transform = CGAffineTransform(a: a, b: b, c: c, d: d, tx: tx, ty: ty)
        guard transform.a.isFinite, transform.b.isFinite,
              transform.c.isFinite, transform.d.isFinite,
              transform.tx.isFinite, transform.ty.isFinite else {
            throw FACADEFaceOffError.invalidAlignment(label)
        }
        return transform
    }

    private func imageTensor(
        from image: CIImage,
        size: Int,
        normalization: ImageNormalization
    ) throws -> MLMultiArray {
        let pixels = try rgbaPixels(from: image, size: size)
        let tensor = try MLMultiArray(
            shape: [1, 3, NSNumber(value: size), NSNumber(value: size)],
            dataType: .float32
        )
        let strides = tensor.strides.map(\.intValue)
        let pointer = tensor.dataPointer.bindMemory(
            to: Float32.self,
            capacity: tensor.count
        )
        for y in 0..<size {
            for x in 0..<size {
                let pixelOffset = (y * size + x) * 4
                for channel in 0..<3 {
                    let tensorOffset = channel * strides[1] + y * strides[2] + x * strides[3]
                    let value = Float32(pixels[pixelOffset + channel])
                    switch normalization {
                    case .minusOneToOne:
                        pointer[tensorOffset] = value / 127.5 - 1
                    case .zeroToOne:
                        pointer[tensorOffset] = value / 255
                    }
                }
            }
        }
        return tensor
    }

    private func rgbaPixels(from image: CIImage, size: Int) throws -> [UInt8] {
        var pixels = [UInt8](repeating: 0, count: size * size * 4)
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)
        pixels.withUnsafeMutableBytes { buffer in
            context.render(
                image,
                toBitmap: buffer.baseAddress!,
                rowBytes: size * 4,
                bounds: CGRect(x: 0, y: 0, width: size, height: size),
                format: .RGBA8,
                colorSpace: colorSpace
            )
        }
        return pixels
    }

    private func recognize(_ input: MLMultiArray, with model: MLModel) throws -> MLMultiArray {
        do {
            let provider = try MLDictionaryFeatureProvider(dictionary: ["input_0": input])
            let prediction = try model.prediction(from: provider)
            guard let raw = prediction.featureValue(for: "input")?.multiArrayValue,
                  raw.count == 512 else {
                throw FACADEFaceOffError.modelFailure(
                    "the recognizer returned an unexpected output"
                )
            }
            let norm = sqrt((0..<512).reduce(0.0) { partial, index in
                let value = raw[[0, NSNumber(value: index)]].doubleValue
                return partial + value * value
            })
            guard norm.isFinite, norm > 0.000_001 else {
                throw FACADEFaceOffError.modelFailure(
                    "the recognizer returned an invalid identity vector"
                )
            }
            let normalized = try MLMultiArray(shape: [1, 512], dataType: .float32)
            let pointer = normalized.dataPointer.bindMemory(
                to: Float32.self,
                capacity: normalized.count
            )
            for index in 0..<512 {
                pointer[index] = Float32(
                    raw[[0, NSNumber(value: index)]].doubleValue / norm
                )
            }
            return normalized
        } catch let error as FACADEFaceOffError {
            throw error
        } catch {
            throw FACADEFaceOffError.modelFailure(error.localizedDescription)
        }
    }

    private func renderFace(
        target: MLMultiArray,
        sourceIdentity: MLMultiArray,
        with model: MLModel
    ) throws -> MLMultiArray {
        do {
            let provider = try MLDictionaryFeatureProvider(dictionary: [
                "input_0": target,
                "input_1": sourceIdentity,
            ])
            let prediction = try model.prediction(from: provider)
            guard let output = prediction.featureValue(for: "var_351")?.multiArrayValue,
                  output.count == 3 * 128 * 128 else {
                throw FACADEFaceOffError.modelFailure(
                    "the renderer returned an unexpected output"
                )
            }
            return output
        } catch let error as FACADEFaceOffError {
            throw error
        } catch {
            throw FACADEFaceOffError.modelFailure(error.localizedDescription)
        }
    }

    private func mapIdentity(
        _ sourceEmbedding: MLMultiArray,
        with identityMap: [Float32]
    ) throws -> MLMultiArray {
        guard sourceEmbedding.count == 512, identityMap.count == 512 * 512 else {
            throw FACADEFaceOffError.modelFailure(
                "the InSwapper identity adapter has unexpected dimensions"
            )
        }
        let source = sourceEmbedding.dataPointer.bindMemory(
            to: Float32.self,
            capacity: 512
        )
        var mapped = [Double](repeating: 0, count: 512)
        for column in 0..<512 {
            var sum = 0.0
            for row in 0..<512 {
                sum += Double(source[row]) * Double(identityMap[row * 512 + column])
            }
            mapped[column] = sum
        }
        let norm = sqrt(mapped.reduce(0) { $0 + $1 * $1 })
        guard norm.isFinite, norm > 0.000_001 else {
            throw FACADEFaceOffError.modelFailure(
                "the InSwapper identity adapter returned an invalid vector"
            )
        }
        let normalized = try MLMultiArray(shape: [1, 512], dataType: .float32)
        let destination = normalized.dataPointer.bindMemory(
            to: Float32.self,
            capacity: normalized.count
        )
        for index in 0..<512 {
            destination[index] = Float32(mapped[index] / norm)
        }
        return normalized
    }

    private func image(from tensor: MLMultiArray, size: Int) throws -> CIImage {
        var pixels = [UInt8](repeating: 255, count: size * size * 4)
        for y in 0..<size {
            for x in 0..<size {
                let pixelOffset = (y * size + x) * 4
                for channel in 0..<3 {
                    let value = tensor[
                        [0, NSNumber(value: channel), NSNumber(value: y), NSNumber(value: x)]
                    ].doubleValue
                    guard value.isFinite else {
                        throw FACADEFaceOffError.modelFailure(
                            "the renderer returned non-finite pixels"
                        )
                    }
                    pixels[pixelOffset + channel] = UInt8(
                        max(0, min(255, (value * 255).rounded()))
                    )
                }
            }
        }
        let data = Data(pixels) as CFData
        guard let provider = CGDataProvider(data: data),
              let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
              let image = CGImage(
                width: size,
                height: size,
                bitsPerComponent: 8,
                bitsPerPixel: 32,
                bytesPerRow: size * 4,
                space: colorSpace,
                bitmapInfo: CGBitmapInfo(
                    rawValue: CGImageAlphaInfo.premultipliedLast.rawValue
                ),
                provider: provider,
                decode: nil,
                shouldInterpolate: true,
                intent: .defaultIntent
              ) else {
            throw FACADEFaceOffError.renderFailure(
                "the renderer output could not become an image"
            )
        }
        return CIImage(cgImage: image)
    }

    private func composite(
        alignedRenderedFace: CIImage,
        targetImage: CIImage,
        targetAlignment: Alignment
    ) throws -> Data {
        let inverse = targetAlignment.sourceToCanonical.inverted()
        guard inverse.a.isFinite, inverse.d.isFinite else {
            throw FACADEFaceOffError.invalidAlignment("synthetic target")
        }
        let bounds = CGRect(x: 0, y: 0, width: 128, height: 128)
        guard let gradient = CIFilter(
            name: "CIRadialGradient",
            parameters: [
                kCIInputCenterKey: CIVector(x: 64, y: 62),
                "inputRadius0": 43,
                "inputRadius1": 62,
                "inputColor0": CIColor(red: 1, green: 1, blue: 1, alpha: 1),
                "inputColor1": CIColor(red: 0, green: 0, blue: 0, alpha: 0),
            ]
        )?.outputImage else {
            throw FACADEFaceOffError.renderFailure("the local blend mask failed")
        }
        let faceLayer = alignedRenderedFace.cropped(to: bounds).transformed(by: inverse)
        let maskLayer = gradient.cropped(to: bounds).transformed(by: inverse)
        let composited = faceLayer.applyingFilter(
            "CIBlendWithMask",
            parameters: [
                kCIInputBackgroundImageKey: targetImage,
                kCIInputMaskImageKey: maskLayer,
            ]
        ).cropped(to: targetImage.extent)
        guard let outputCGImage = context.createCGImage(
            composited,
            from: targetImage.extent,
            format: .RGBA8,
            colorSpace: CGColorSpace(name: CGColorSpace.sRGB)
        ) else {
            throw FACADEFaceOffError.renderFailure("the local composite could not be encoded")
        }
        let png = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(
            png,
            "public.png" as CFString,
            1,
            nil
        ) else {
            throw FACADEFaceOffError.renderFailure("the local PNG destination failed")
        }
        CGImageDestinationAddImage(destination, outputCGImage, nil)
        guard CGImageDestinationFinalize(destination) else {
            throw FACADEFaceOffError.renderFailure("the local PNG encoding failed")
        }
        return png as Data
    }
}
