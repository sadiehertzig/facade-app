// CoreMLModels.swift
//
// Thin, checked wrappers around the two Core ML models the app runs through
// FACADEKit:
//
//   * `FACADECoreMLEmbeddingModel` wraps a face recogniser. Given an aligned
//     112x112 face crop it returns a 512-number "embedding", a fingerprint of
//     the identity. The app computes this once for the clean photo and uses it
//     as the reference the protection tries to move away from.
//   * `FACADECoreMLR6InitializerBank` wraps the protection model. Given the
//     256x256 prepared photo it returns eight candidate residuals (small
//     per-pixel adjustments). The app takes the first as the starting point
//     for its optimisation loop.
//
// Both wrappers refuse to be built unless the manifest entry for the model
// describes exactly the input and output names and shapes the code below
// hard-codes. That way a model swapped for one with a different interface is
// caught at load time rather than producing garbage at run time.
//
// Core ML detail: a model is driven by a dictionary of named inputs
// (`MLDictionaryFeatureProvider`) and returns named outputs. The names below
// are the ones baked into the exported models.

import CoreML
import Foundation

/// How a 0...1 RGB crop must be rescaled before the recogniser sees it.
public enum FACADEEmbeddingNormalization: Sendable {
    /// ArcFace expects values in -1...1: `value * 2 - 1`, the same for all
    /// three colour channels.
    case arcFaceMinusOneToOne
}

/// A face recogniser exported to Core ML.
///
/// Input: one 112x112 RGB crop, channel-first, already aligned by
/// `FACADEFaceCropTransform`. Output: a 512-dimensional unit-length vector.
public final class FACADECoreMLEmbeddingModel: @unchecked Sendable {
    /// The loaded Core ML recogniser.
    private let model: MLModel
    /// Core ML input feature name, for example "aligned_rgb_nchw_minus1_to1".
    private let inputName: String
    /// Core ML output feature name, for example "l2_embedding_512".
    private let outputName: String
    /// Input scaling the recogniser was trained with.
    private let normalization: FACADEEmbeddingNormalization

    /// Checks the manifest's description of the model against what this
    /// wrapper assumes: the expected role, 8-bit weights, no gradient output,
    /// a single [1, 3, 112, 112] input and a single [1, 512] output with the
    /// given names.
    public init(
        model: MLModel,
        artifact: FACADEModelArtifact,
        expectedRole: FACADEModelRole,
        inputName: String,
        outputName: String,
        normalization: FACADEEmbeddingNormalization
    ) throws {
        // The bundle's recognisers are 8-bit, forward-only models with one
        // input and one output. Anything else means the wrong file was paired
        // with this wrapper, so it is refused before it can run.
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
                "the \(expectedRole.rawValue) Core ML model does not have the inputs and outputs this app expects"
            )
        }
        self.model = model
        self.inputName = inputName
        self.outputName = outputName
        self.normalization = normalization
    }

    /// Runs the recogniser on an aligned crop and returns its 512-number
    /// embedding, after checking the output is finite and unit length.
    public func embedding(for alignedRGB: FACADEFloatImage) throws -> [Float] {
        // The model's input is fixed at 112x112; the alignment step always
        // produces that size, so anything else is a programming error.
        guard alignedRGB.width == 112, alignedRGB.height == 112 else {
            throw FACADEProtectionError.verificationUnavailable("embedding input must be 112x112 RGB")
        }
        // Allocate the NCHW input tensor and fill it plane by plane, applying
        // the model's expected normalisation to each value on the way in.
        let input = try MLMultiArray(
            shape: [1, 3, 112, 112] as [NSNumber],
            dataType: .float32
        )
        let destination = input.dataPointer.bindMemory(to: Float.self, capacity: input.count)
        let plane = 112 * 112
        for channel in 0..<3 {
            for pixel in 0..<plane {
                let value = alignedRGB.values[channel * plane + pixel]
                destination[channel * plane + pixel] = normalized(value)
            }
        }
        // Core ML takes inputs as a name -> value dictionary.
        let provider = try MLDictionaryFeatureProvider(dictionary: [inputName: input])
        let prediction: MLFeatureProvider
        do {
            prediction = try model.prediction(from: provider)
        } catch {
            throw FACADEProtectionError.verificationUnavailable(
                "\(outputName) prediction failed: \(error)"
            )
        }
        // The output must be exactly one 512-number vector.
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
        // The exported model normalises its own output, so the length should
        // be 1. A 2% tolerance absorbs 8-bit quantisation error; anything
        // further off means the wrong model or a broken conversion.
        let norm = sqrt(values.reduce(0) { $0 + $1 * $1 })
        guard norm.isFinite, abs(norm - 1) <= 0.02 else {
            throw FACADEProtectionError.verificationUnavailable(
                "\(outputName) is not an L2-normalized embedding"
            )
        }
        return values
    }

    /// Applies the chosen input normalisation to one pixel value.
    private func normalized(_ value: Float) -> Float {
        switch normalization {
        case .arcFaceMinusOneToOne:
            return value * 2 - 1
        }
    }
}

/// The protection model exported to Core ML.
///
/// Input: the 256x256 prepared photo, channel-first, values 0...1.
/// Output: a [1, 24, 256, 256] tensor, which is eight RGB residual images
/// stacked along the channel axis (8 heads x 3 colours = 24 planes). Each
/// residual is a small per-pixel adjustment that, added to the photo, is a
/// starting guess at a protective perturbation.
///
/// The short code in the class name marks the manifest version this app was
/// validated with; the wrapper pins the model to that version by checking the
/// digest of its source weights.
public final class FACADECoreMLR6InitializerBank: @unchecked Sendable, FACADEInitializerResidualBank {
    /// SHA-256 of the weights this app's protection model was converted from.
    /// A bundle whose protection model declares different source weights is
    /// refused, even if its interface matches.
    static let expectedSourceWeightsSHA256 =
        "c1a1bc2c61da8811f1681a2bf35134411414a2c7ae98b76b0df802b9df5e1c20"

    /// The loaded Core ML protection model.
    private let model: MLModel
    /// Core ML input feature name ("image_nchw_0_1" in the shipped model).
    private let inputName: String
    /// Core ML output feature name ("residual_bank_n24chw" in the shipped model).
    private let outputName: String

    /// Locates the protection model's manifest entry and checks it: the
    /// expected source weights, 8-bit storage, one [1, 3, 256, 256] input and
    /// one [1, 24, 256, 256] output with the given names.
    public init(
        model: MLModel,
        handoff: FACADELoadedModelHandoff,
        // Feature names baked into the exported model: the input is a 0...1
        // NCHW image; the output is the bank of residuals, N x 24 x H x W.
        inputName: String = "image_nchw_0_1",
        outputName: String = "residual_bank_n24chw"
    ) throws {
        // The manifest entry has already been hash-checked by the loader; here
        // it must also describe the exact interface this wrapper hard-codes,
        // and come from the one set of source weights the app was tested with.
        guard let artifact = handoff.manifest.artifacts.first(where: { $0.role == .protectionEngine }),
              artifact.sourceWeightsSHA256 == Self.expectedSourceWeightsSHA256,
              artifact.storagePrecision == .int8,
              artifact.inputs.count == 1,
              artifact.outputs.count == 1,
              artifact.inputs[0].name == inputName,
              artifact.inputs[0].shape == [1, 3, 256, 256],
              artifact.outputs[0].name == outputName,
              artifact.outputs[0].shape == [1, 24, 256, 256] else {
            throw FACADEProtectionError.modelUnavailable(
                "the installed protection model does not match the Core ML model this app was validated with"
            )
        }
        self.model = model
        self.inputName = inputName
        self.outputName = outputName
    }

    /// Runs the model once and splits its 24-plane output into eight
    /// three-plane residual images.
    public func residuals(for image: FACADEFloatImage) async throws -> [FACADEFloatImage] {
        // The model's input shape is fixed; any other size would not fit.
        guard image.width == 256, image.height == 256 else {
            throw FACADEProtectionError.invalidInput("the Core ML r6 initializer requires 256x256 RGB")
        }
        let input = try MLMultiArray(
            shape: [1, 3, 256, 256] as [NSNumber],
            dataType: .float32
        )
        // Guards the raw copy below: Core ML must have allocated exactly
        // 3 x 256 x 256 floats, or the copy would over- or under-run.
        guard input.count == image.values.count else {
            throw FACADEProtectionError.modelUnavailable("Core ML allocated an unexpected input tensor")
        }
        // The float image is already in NCHW order, so it is one memcpy.
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
        // The output must be the full eight-candidate bank at full resolution.
        guard let output = prediction.featureValue(for: outputName)?.multiArrayValue,
              output.shape.map(\.intValue) == [1, 24, 256, 256],
              output.count == 24 * 256 * 256 else {
            throw FACADEProtectionError.modelUnavailable("the protection model's outputs are not the shape this app expects")
        }

        // Head h occupies planes 3h, 3h+1, 3h+2 of the output. Copy each
        // head's three planes out as its own image, rejecting NaN/infinity.
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
