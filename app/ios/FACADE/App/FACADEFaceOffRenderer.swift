// FACADEFaceOffRenderer.swift
//
// "Face-Off": a real face swapper that runs on the phone. It takes the user's
// photo (original or protected), reads the person's identity from it, and
// paints that identity onto a fixed, fictional target face. The app shows the
// two swaps side by side so the user can see what a swap tool does with each.
//
// Where it sits: LivePhotoModel.renderFaceOff creates one
// FACADEOnDeviceFaceOffRenderer per comparison and calls `render` twice, once
// for the original photo and once for the protected copy. Face-Off is only a
// demonstration; it never changes or saves the user's photo.
//
// Key ideas:
// 1. The swapper is built from InSwapper's parts, converted to Core ML:
//    - a recognizer turns a 112 x 112 face into a 512-number identity vector;
//    - an "identity map" (a fixed 512 x 512 matrix) converts that vector into
//      the form the renderer expects;
//    - the renderer takes a 128 x 128 target face plus that converted identity
//      and draws the target face with the source person's identity.
//    This is the same recognizer-through-identity-map path the FACADE
//    protection engine attacks. That makes Face-Off the most favourable test
//    for FACADE; other swap tools may cope better with a protected photo.
// 2. Faces are "aligned" first: Apple's Vision framework finds the eyes and
//    nose, and an affine transform moves those three points onto the standard
//    ArcFace face template, so every face reaches the models upright, centred
//    and at the same scale.
// 3. Core Image uses a bottom-left origin (y grows upward), while the face
//    template and the models use a top-left origin. That is why many y
//    values below are written as `size - y`.
//
// The similarity scores the app shows come from this swapper's own
// recognizer. They illustrate one swap; they are not proof that protection
// works against other tools.
import CoreImage
import CoreML
import CryptoKit
import Foundation
import ImageIO
import Vision

/// Everything one Face-Off swap produces.
struct FACADEFaceOffRenderedResult: Equatable, Sendable {
    /// The fictional target photo's file bytes, unchanged.
    let syntheticTargetData: Data
    /// The full target photo with the swapped face blended in, as PNG.
    let attemptedSwapData: Data
    /// The recognizer's 512-number identity vector for the user's photo, scaled to length 1.
    let sourceEmbedding: [Float]
    /// Recognizer vector of the swapped face (rows 0..<112, columns 8..<120 of the
    /// 128 px render), for a display-only similarity to the source.
    let swapEmbedding: [Float]
    /// The 128 px swapped face before compositing, for a close-up view.
    let alignedSwapFaceData: Data
    /// How long this one swap took, in seconds.
    let runtimeSeconds: Double
}

/// Anything that can perform a Face-Off swap. The on-device renderer below and
/// the "unavailable" placeholder both conform.
protocol FACADEFaceOffRendering: Sendable {
    /// Always throws: this renderer exists only to report why Face-Off is unavailable.
    func render(sourceImageData: Data) async throws -> FACADEFaceOffRenderedResult
}

/// Every way a swap can fail. Each case has a plain-English message that the
/// Protect screen shows to the user.
enum FACADEFaceOffError: LocalizedError {
    // A model or target file is not in the app (the models are private and
    // not in git; see app/ios/Models/README.md).
    case missingResource(String)
    // A file is present but has the wrong size, hash or contents.
    case invalidResource(String)
    // Face-Off needs exactly one face; zero or several are refused.
    case faceCount(label: String, count: Int)
    // Vision found a face but not both eyes and the nose.
    case missingLandmarks(String)
    // The eye and nose points gave a transform that cannot be used.
    case invalidAlignment(String)
    // Core ML failed, or a model returned something unexpected.
    case modelFailure(String)
    // Turning the result into an image or PNG failed.
    case renderFailure(String)

    /// The message shown to the user for each case.
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

/// A stand-in renderer that always fails with a "missing" message. Nothing in
/// the app currently creates one; LivePhotoModel uses the on-device renderer.
struct FACADEUnavailableFaceOffRenderer: FACADEFaceOffRendering {
    let reason: String

    func render(sourceImageData: Data) async throws -> FACADEFaceOffRenderedResult {
        throw FACADEFaceOffError.missingResource(reason)
    }
}

/// The real on-device swapper. It is an `actor`, so only one swap at a time can
/// use its cached models.
actor FACADEOnDeviceFaceOffRenderer: FACADEFaceOffRendering {
    /// SHA-256 of the default fictional target face,
    /// FaceOffAssets/synthetic_blonde_woman_faceoff_target.png. A different file is refused.
    static let targetSHA256 =
        "f27ac119abfa60e77d5e6cb82935f66fe7b0b3f0a86f9de3b0a6fa50e9263e8a"
    /// Preferred Face-Off target when Models/FaceOffTarget/target.png is bundled.
    static let preferredTargetSHA256 =
        "874e23015c4958393b6cc232cf4b2be161efd3907173339fa43e660510dbae3a"
    /// SHA-256 of identity-map-float32.bin, the 512 x 512 identity map.
    static let identityMapSHA256 =
        "370af5bf707dafdbea8a40448d697d9697610bd223ecf92887af9c9cc7055ac8"

    /// An aligned face crop, plus the transform that produced it. The transform
    /// is kept so the swapped face can be put back exactly where it came from.
    private struct Alignment {
        let alignedImage: CIImage
        let sourceToCanonical: CGAffineTransform
    }

    /// How pixel values are scaled for a model: the recognizer expects [-1, 1],
    /// the renderer expects [0, 1].
    private enum ImageNormalization {
        case minusOneToOne
        case zeroToOne
    }

    /// Where the model files and target face are inside the app, and the hash the
    /// target must have.
    private let recognizerURL: URL
    // The renderer (the face-drawing network), compiled for Core ML.
    private let rendererURL: URL
    // The 512 x 512 identity map as raw float32 bytes.
    private let identityMapURL: URL
    // The fictional target face photo, and the SHA-256 it must have.
    private let targetURL: URL
    private let expectedTargetSHA256: String
    /// One Core Image context for all drawing, working in sRGB throughout so colours
    /// are not converted along the way. Intermediate caching is off to save memory.
    private let context = CIContext(options: [
        .cacheIntermediates: false,
        .workingColorSpace: CGColorSpace(name: CGColorSpace.sRGB) as Any,
        .outputColorSpace: CGColorSpace(name: CGColorSpace.sRGB) as Any,
    ])
    /// Models and identity map, loaded on first use and then kept for the second
    /// swap of the same comparison.
    private var recognizer: MLModel?
    private var renderer: MLModel?
    private var identityMap: [Float32]?

    /// Finds the Face-Off files inside the app bundle. Throws `missingResource`
    /// naming the first file that is absent. Nothing is loaded or checked yet;
    /// that happens in `render`.
    static func load(bundle: Bundle = .main) throws -> FACADEOnDeviceFaceOffRenderer {
        // The three model files live in the FaceOffModels folder inside the app.
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
        // If the optional FaceOffTarget/target.png is bundled, it is used as the
        // target face (with its own pinned hash); otherwise the default target is used.
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

    /// Builds a renderer from explicit file locations, using the default target's hash.
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

    /// Stores the locations only. Use one of the `load` functions.
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

    /// Performs one swap: the identity in `sourceImageData` is painted onto the
    /// fictional target face. Steps: check the target's hash, decode both
    /// images, align both faces, read the source identity, convert it with the
    /// identity map, render the swapped face, measure it, and blend it into the
    /// target photo.
    func render(sourceImageData: Data) async throws -> FACADEFaceOffRenderedResult {
        let clock = ContinuousClock()
        let started = clock.now
        // Stop early if the user already cancelled the comparison.
        try Task.checkCancellation()

        guard !sourceImageData.isEmpty else {
            throw FACADEFaceOffError.invalidResource("the source image is empty")
        }
        // Check the target face against its pinned hash before every swap, so a
        // replaced or damaged target file is never used.
        let targetData = try Data(contentsOf: targetURL, options: [.mappedIfSafe])
        let targetDigest = SHA256.hash(data: targetData)
            .map { String(format: "%02x", $0) }
            .joined()
        guard targetDigest == expectedTargetSHA256 else {
            throw FACADEFaceOffError.invalidResource(
                "the synthetic target checksum does not match"
            )
        }

        // Decode both photos upright.
        let sourceImage = try decodeImage(
            sourceImageData,
            label: "source selfie"
        )
        let targetImage = try decodeImage(targetData, label: "synthetic target")
        // The source face is aligned to 112 x 112, the recognizer's input size.
        let sourceAlignment = try align(
            sourceImage,
            label: "source selfie",
            size: 112,
            horizontalOffset: 0
        )
        // The target face is aligned to 128 x 128, the renderer's input size. For a
        // 128-pixel crop InsightFace uses the same template shifted 8 pixels to the right.
        let targetAlignment = try align(
            targetImage,
            label: "synthetic target",
            size: 128,
            horizontalOffset: 8
        )

        try Task.checkCancellation()
        // Load the models only after both faces were found, so a photo with no clear
        // face fails fast without the cost of loading.
        let models = try loadModelsIfNeeded()
        let sourceTensor = try imageTensor(
            from: sourceAlignment.alignedImage,
            size: 112,
            normalization: .minusOneToOne
        )
        // Who is in the source photo, as a 512-number identity vector.
        let sourceEmbedding = try recognize(
            sourceTensor,
            with: models.recognizer
        )
        // Convert the identity vector into the form the renderer takes.
        let rendererIdentity = try mapIdentity(
            sourceEmbedding,
            with: loadIdentityMapIfNeeded()
        )
        // The aligned target face as renderer input, scaled to [0, 1].
        let targetTensor = try imageTensor(
            from: targetAlignment.alignedImage,
            size: 128,
            normalization: .zeroToOne
        )
        // Draw the target face with the source person's identity.
        let renderedTensor = try renderFace(
            target: targetTensor,
            sourceIdentity: rendererIdentity,
            with: models.renderer
        )
        // Rendering is the slow part; check for a cancel before finishing up.
        try Task.checkCancellation()

        // The renderer's output array as a 128 x 128 image.
        let alignedRenderedFace = try image(from: renderedTensor, size: 128)
        // Re-measure the swapped face with the same recognizer, so the app can say how
        // much it still looks like the source person. The window is the top 112 rows
        // and columns 8..<120 of the 128-pixel render. In Core Image's bottom-left
        // coordinates, the top 112 rows start at y = 16; the shift moves the crop to
        // the origin.
        let swapCrop = alignedRenderedFace
            .cropped(to: CGRect(x: 8, y: 16, width: 112, height: 112))
            .transformed(by: CGAffineTransform(translationX: -8, y: -16))
        let swapEmbedding = try recognize(
            try imageTensor(from: swapCrop, size: 112, normalization: .minusOneToOne),
            with: models.recognizer
        )
        // The 128-pixel swapped face on its own, for the close-up view.
        guard let sRGB = CGColorSpace(name: CGColorSpace.sRGB),
              let alignedSwapFaceData = context.pngRepresentation(
                of: alignedRenderedFace, format: .RGBA8, colorSpace: sRGB) else {
            throw FACADEFaceOffError.renderFailure("the swapped face could not be encoded")
        }
        // Paste the swapped face back into the full target photo.
        let composited = try composite(
            alignedRenderedFace: alignedRenderedFace,
            targetImage: targetImage,
            targetAlignment: targetAlignment
        )
        // ContinuousClock keeps counting while the phone sleeps, so this is wall time.
        let elapsed = started.duration(to: clock.now)
        // A Duration is whole seconds plus attoseconds (10^-18 s).
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

    /// Copies a 512-value float32 Core ML array into a plain Swift array.
    private func embeddingValues(_ embedding: MLMultiArray) -> [Float] {
        let pointer = embedding.dataPointer.bindMemory(
            to: Float32.self,
            capacity: embedding.count
        )
        return Array(UnsafeBufferPointer(start: pointer, count: embedding.count))
    }

    /// Loads the recognizer and renderer on first use and keeps them for later swaps.
    private func loadModelsIfNeeded() throws -> (recognizer: MLModel, renderer: MLModel) {
        // Already loaded by the first swap of this comparison.
        if let recognizer, let renderer {
            return (recognizer, renderer)
        }
        let configuration = MLModelConfiguration()
        // Let Core ML use the CPU, GPU or Neural Engine, whichever it picks. The swap is
        // only a demonstration, so small numeric differences between them do not matter here.
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
            // Keep both only when both loaded, so a half-loaded state never exists.
            recognizer = loadedRecognizer
            renderer = loadedRenderer
            return (loadedRecognizer, loadedRenderer)
        // Core ML's error text is passed on inside a Face-Off message.
        } catch {
            throw FACADEFaceOffError.modelFailure(error.localizedDescription)
        }
    }

    /// Reads the 512 x 512 identity map on first use. The file must be exactly
    /// 512 x 512 little-endian 32-bit floats (1 MB), match its pinned SHA-256,
    /// and contain only finite numbers.
    private func loadIdentityMapIfNeeded() throws -> [Float32] {
        if let identityMap {
            return identityMap
        }
        let data: Data
        do {
            // Memory-mapped when possible, so the file is not copied just to be hashed.
            data = try Data(contentsOf: identityMapURL, options: [.mappedIfSafe])
        } catch {
            throw FACADEFaceOffError.invalidResource(
                "the InSwapper identity map cannot be read"
            )
        }
        // 512 x 512 values x 4 bytes = 1,048,576 bytes.
        guard data.count == 512 * 512 * MemoryLayout<Float32>.size else {
            throw FACADEFaceOffError.invalidResource(
                "the InSwapper identity map has an unexpected byte count"
            )
        }
        let digest = SHA256.hash(data: data)
            .map { String(format: "%02x", $0) }
            .joined()
        // A different map would produce different swaps, so only the pinned one is accepted.
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
                // Assemble each float from its four bytes, lowest byte first (little-endian),
                // so the result does not depend on how the phone stores numbers.
                let bitPattern = UInt32(bytes[offset]) |
                    UInt32(bytes[offset + 1]) << 8 |
                    UInt32(bytes[offset + 2]) << 16 |
                    UInt32(bytes[offset + 3]) << 24
                values.append(Float32(bitPattern: bitPattern))
            }
        }
        // A NaN or infinity in the map would spoil every swap.
        guard values.count == 512 * 512, values.allSatisfy({ $0.isFinite }) else {
            throw FACADEFaceOffError.invalidResource(
                "the InSwapper identity map contains invalid values"
            )
        }
        identityMap = values
        return values
    }

    /// Decodes an image file into a CIImage, turned upright using its EXIF
    /// orientation. Images smaller than 128 x 128 are refused.
    private func decodeImage(_ data: Data, label: String) throws -> CIImage {
        guard let decoded = CIImage(
            data: data,
            // Apply the EXIF orientation flag, so a phone photo taken sideways comes out upright.
            options: [.applyOrientationProperty: true]
        ) else {
            throw FACADEFaceOffError.invalidResource("the \(label) cannot be decoded")
        }
        let extent = decoded.extent.integral
        // The renderer works at 128 pixels; a smaller image cannot hold a usable face.
        guard extent.width >= 128, extent.height >= 128 else {
            throw FACADEFaceOffError.invalidResource("the \(label) is too small")
        }
        // Move the image so its corner sits at (0, 0); later math assumes this.
        return decoded.transformed(
            by: CGAffineTransform(translationX: -extent.minX, y: -extent.minY)
        )
    }

    /// Finds the single face, takes the centre of each eye and of the nose, and
    /// warps the image so those points land on the ArcFace template. Returns a
    /// `size` x `size` crop and the transform used.
    private func align(
        _ image: CIImage,
        label: String,
        size: CGFloat,
        horizontalOffset: CGFloat
    ) throws -> Alignment {
        // Vision's face-landmark detector. The image was already made upright, so
        // it is passed with orientation `.up`.
        let request = VNDetectFaceLandmarksRequest()
        let handler = VNImageRequestHandler(ciImage: image, orientation: .up)
        do {
            // Runs face detection and landmark finding together.
            try handler.perform([request])
        } catch {
            throw FACADEFaceOffError.invalidAlignment(label)
        }
        let faces = request.results ?? []
        // Exactly one face: with several, Face-Off could swap the wrong person.
        guard faces.count == 1, let face = faces.first else {
            throw FACADEFaceOffError.faceCount(label: label, count: faces.count)
        }
        guard let landmarks = face.landmarks,
              let leftEye = landmarks.leftEye,
              let rightEye = landmarks.rightEye,
              let nose = landmarks.nose else {
            throw FACADEFaceOffError.missingLandmarks(label)
        }

        // Each landmark is a group of points (an eye outline, the nose); use the
        // average point of each group, converted to image pixels.
        let source = [leftEye, rightEye, nose].map {
            imagePoint(for: average($0.normalizedPoints), face: face, image: image)
        }
        // Eye and nose points of the standard 112 x 112 ArcFace template (InsightFace,
        // MIT-licensed code). The template has five points; Face-Off uses three: the
        // two eyes and the nose tip. y is flipped (`size - y`) for Core Image.
        let destination = [
            CGPoint(x: 38.2946 + horizontalOffset, y: size - 51.6963),
            CGPoint(x: 73.5318 + horizontalOffset, y: size - 51.5014),
            CGPoint(x: 56.0252 + horizontalOffset, y: size - 71.7366),
        ]
        // The transform that carries the photo's eye and nose points onto the template.
        let transform = try affineTransform(
            source: source,
            destination: destination,
            label: label
        )
        let bounds = CGRect(x: 0, y: 0, width: size, height: size)
        // Warp the whole image, then keep only the size x size square at the origin.
        let aligned = image.transformed(by: transform).cropped(to: bounds)
        return Alignment(alignedImage: aligned, sourceToCanonical: transform)
    }

    /// The average of a list of points (zero for an empty list).
    private func average(_ points: [CGPoint]) -> CGPoint {
        guard !points.isEmpty else { return .zero }
        let sum = points.reduce(CGPoint.zero) {
            CGPoint(x: $0.x + $1.x, y: $0.y + $1.y)
        }
        return CGPoint(x: sum.x / CGFloat(points.count), y: sum.y / CGFloat(points.count))
    }

    /// Converts a Vision landmark point to image pixels. Vision gives landmark
    /// points as fractions of the face's bounding box, and the box as fractions
    /// of the image, both with a bottom-left origin like Core Image.
    private func imagePoint(
        for point: CGPoint,
        face: VNFaceObservation,
        image: CIImage
    ) -> CGPoint {
        // Landmark point (fractions of the box) -> fractions of the image -> pixels.
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

    /// Solves for the one affine transform that maps the three source points
    /// exactly onto the three destination points.
    ///
    /// Method: subtract point 0 from the others, so only the 2 x 2 linear part is
    /// left, and solve it with Cramer's rule; then pick the translation so point
    /// 0 lands correctly. A determinant near zero means the three points are
    /// (almost) in a line, so no reliable transform exists and alignment is refused.
    ///
    /// Note: InsightFace fits a similarity transform (rotation, scale, shift only)
    /// to five points. An exact three-point affine fit can also shear slightly, so
    /// alignments can differ a little from InsightFace's.
    private func affineTransform(
        source: [CGPoint],
        destination: [CGPoint],
        label: String
    ) throws -> CGAffineTransform {
        // Exactly three point pairs are needed to fix an affine transform.
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
        // Twice the signed area of the source triangle.
        let determinant = sourceX1 * sourceY2 - sourceX2 * sourceY1
        guard determinant.isFinite, abs(determinant) > 0.000_001 else {
            throw FACADEFaceOffError.invalidAlignment(label)
        }

        // CGAffineTransform maps (x, y) to (a x + c y + tx, b x + d y + ty).
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
        // Last safety check: never return a transform with NaN or infinite entries.
        guard transform.a.isFinite, transform.b.isFinite,
              transform.c.isFinite, transform.d.isFinite,
              transform.tx.isFinite, transform.ty.isFinite else {
            throw FACADEFaceOffError.invalidAlignment(label)
        }
        return transform
    }

    /// Renders a CIImage into a [1, 3, size, size] float32 Core ML array (NCHW:
    /// all red values, then green, then blue), scaled as the model expects.
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
        // Use the array's own strides instead of assuming tight packing, in case
        // Core ML pads rows.
        let strides = tensor.strides.map(\.intValue)
        let pointer = tensor.dataPointer.bindMemory(
            to: Float32.self,
            capacity: tensor.count
        )
        for y in 0..<size {
            for x in 0..<size {
                // RGBA bytes: 4 per pixel, row by row; the alpha byte is ignored.
                let pixelOffset = (y * size + x) * 4
                for channel in 0..<3 {
                    let tensorOffset = channel * strides[1] + y * strides[2] + x * strides[3]
                    let value = Float32(pixels[pixelOffset + channel])
                    switch normalization {
                    // 0...255 -> -1...1 for the recognizer; 0...255 -> 0...1 for the renderer.
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

    /// Draws the `size` x `size` region at the origin of a CIImage into RGBA
    /// bytes, 4 per pixel, in sRGB.
    private func rgbaPixels(from image: CIImage, size: Int) throws -> [UInt8] {
        var pixels = [UInt8](repeating: 0, count: size * size * 4)
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)
        pixels.withUnsafeMutableBytes { buffer in
            // The buffer is filled in standard image order, top row first, so buffer
            // row 0 is the top of the face, as the models expect.
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

    /// Runs the recognizer on a 112 x 112 face and returns its identity vector,
    /// scaled to length 1 so that comparing two vectors gives a cosine similarity.
    private func recognize(_ input: MLMultiArray, with model: MLModel) throws -> MLMultiArray {
        do {
            // "input_0" and "input" are the names the model conversion gave the
            // recognizer's input and output; they are not meaningful words.
            let provider = try MLDictionaryFeatureProvider(dictionary: ["input_0": input])
            let prediction = try model.prediction(from: provider)
            // The recognizer must return exactly 512 numbers.
            guard let raw = prediction.featureValue(for: "input")?.multiArrayValue,
                  raw.count == 512 else {
                throw FACADEFaceOffError.modelFailure(
                    "the recognizer returned an unexpected output"
                )
            }
            // Length of the vector, used to scale it to length 1.
            let norm = sqrt((0..<512).reduce(0.0) { partial, index in
                let value = raw[[0, NSNumber(value: index)]].doubleValue
                return partial + value * value
            })
            // A zero or broken vector has no direction and cannot be compared.
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
            // Divide every value by the length, so the vector has length 1.
            for index in 0..<512 {
                pointer[index] = Float32(
                    raw[[0, NSNumber(value: index)]].doubleValue / norm
                )
            }
            return normalized
        // Our own errors pass through; anything else is wrapped as a model failure.
        } catch let error as FACADEFaceOffError {
            throw error
        } catch {
            throw FACADEFaceOffError.modelFailure(error.localizedDescription)
        }
    }

    /// Runs the renderer: a 128 x 128 target face plus a 512-number identity in,
    /// a 3 x 128 x 128 swapped face out (values 0...1).
    private func renderFace(
        target: MLMultiArray,
        sourceIdentity: MLMultiArray,
        with model: MLModel
    ) throws -> MLMultiArray {
        do {
            let provider = try MLDictionaryFeatureProvider(dictionary: [
                // "input_0" = target face, "input_1" = converted identity. "var_351" is the
                // automatically generated name of the output layer.
                "input_0": target,
                "input_1": sourceIdentity,
            ])
            // This one call is the actual face swap.
            let prediction = try model.prediction(from: provider)
            // The output must hold 3 x 128 x 128 values.
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

    /// Converts a recognizer vector into the renderer's identity input: multiply
    /// the vector by the 512 x 512 identity map, then scale to length 1. This is
    /// how InSwapper itself prepares the source identity.
    private func mapIdentity(
        _ sourceEmbedding: MLMultiArray,
        with identityMap: [Float32]
    ) throws -> MLMultiArray {
        guard sourceEmbedding.count == 512, identityMap.count == 512 * 512 else {
            throw FACADEFaceOffError.modelFailure(
                "the InSwapper identity adapter has unexpected dimensions"
            )
        }
        // Read the recognizer vector directly from Core ML's memory.
        let source = sourceEmbedding.dataPointer.bindMemory(
            to: Float32.self,
            capacity: 512
        )
        var mapped = [Double](repeating: 0, count: 512)
        // Vector-times-matrix, in Double for accuracy: mapped[column] is the sum
        // over rows of source[row] x map[row][column]. The map is stored row by row.
        for column in 0..<512 {
            var sum = 0.0
            for row in 0..<512 {
                sum += Double(source[row]) * Double(identityMap[row * 512 + column])
            }
            mapped[column] = sum
        }
        // Scale the result to length 1, as InSwapper does.
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

    /// Turns the renderer's 3 x size x size output (0...1) into an opaque CIImage.
    private func image(from tensor: MLMultiArray, size: Int) throws -> CIImage {
        // Start every byte at 255 so the alpha channel is fully opaque.
        var pixels = [UInt8](repeating: 255, count: size * size * 4)
        for y in 0..<size {
            for x in 0..<size {
                let pixelOffset = (y * size + x) * 4
                for channel in 0..<3 {
                    // Element [0, channel, y, x], read one at a time; 128 x 128 x 3 reads is quick.
                    let value = tensor[
                        [0, NSNumber(value: channel), NSNumber(value: y), NSNumber(value: x)]
                    ].doubleValue
                    // A NaN pixel means the model misbehaved; refuse rather than show garbage.
                    guard value.isFinite else {
                        throw FACADEFaceOffError.modelFailure(
                            "the renderer returned non-finite pixels"
                        )
                    }
                    // Scale 0...1 to 0...255, round, and clamp in case the model overshoots.
                    pixels[pixelOffset + channel] = UInt8(
                        max(0, min(255, (value * 255).rounded()))
                    )
                }
            }
        }
        // Wrap the bytes as an 8-bit RGBA sRGB CGImage, then as a CIImage.
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

    /// Blends the swapped 128-pixel face back into the full target photo. The
    /// inverse of the target's alignment transform carries the face back to where
    /// it came from. A soft round mask keeps the centre of the swapped face and
    /// fades to the original photo at the edges, which hides the square crop.
    private func composite(
        alignedRenderedFace: CIImage,
        targetImage: CIImage,
        targetAlignment: Alignment
    ) throws -> Data {
        // The alignment mapped photo -> 128-pixel crop; its inverse maps crop -> photo.
        let inverse = targetAlignment.sourceToCanonical.inverted()
        // CGAffineTransform.inverted() returns the input unchanged when it cannot
        // invert, so check the result is at least finite.
        guard inverse.a.isFinite, inverse.d.isFinite else {
            throw FACADEFaceOffError.invalidAlignment("synthetic target")
        }
        let bounds = CGRect(x: 0, y: 0, width: 128, height: 128)
        // Mask: fully opaque within 43 pixels of (64, 62) in the 128-pixel crop,
        // fading to transparent at 62 pixels.
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
        // Move both the face and its mask back into the target photo's coordinates.
        let faceLayer = alignedRenderedFace.cropped(to: bounds).transformed(by: inverse)
        let maskLayer = gradient.cropped(to: bounds).transformed(by: inverse)
        // Where the mask is white show the swapped face, where it is clear show the
        // target photo, and blend in between. Crop to the target photo's size.
        let composited = faceLayer.applyingFilter(
            "CIBlendWithMask",
            parameters: [
                kCIInputBackgroundImageKey: targetImage,
                kCIInputMaskImageKey: maskLayer,
            ]
        ).cropped(to: targetImage.extent)
        // Render the composite to 8-bit sRGB pixels.
        guard let outputCGImage = context.createCGImage(
            composited,
            from: targetImage.extent,
            format: .RGBA8,
            colorSpace: CGColorSpace(name: CGColorSpace.sRGB)
        ) else {
            throw FACADEFaceOffError.renderFailure("the local composite could not be encoded")
        }
        // Encode the result as PNG in memory.
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
