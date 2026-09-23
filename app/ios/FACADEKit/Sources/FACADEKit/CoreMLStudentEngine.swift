import CoreML
import Foundation

/// Exact Core ML contract for a future one-pass bounded-residual student.
/// Research can replace the model bytes and manifest without changing the
/// acquisition, verification, or save layers of the app.
public final class FACADECoreMLOnePassResidualEngine: @unchecked Sendable,
    FACADEProtectionEngine {
    public static let inputName = "image_nchw_0_1"
    public static let outputName = "residual_nchw_m16_16_over_255"
    public static let spatialSize = 256
    public static let residualLimit: Float = 16.0 / 255.0

    public let engineIdentifier: String
    public let modelBundleIdentifier: String

    private let model: MLModel
    private let codec: any FACADEExperimentalImageCodec
    private let safetyGate: FACADEDeviceSafetyGate

    public init(
        model: MLModel,
        handoff: FACADELoadedModelHandoff,
        codec: any FACADEExperimentalImageCodec,
        safetyGate: FACADEDeviceSafetyGate = FACADEDeviceSafetyGate()
    ) throws {
        guard let artifact = handoff.manifest.artifacts.first(where: {
            $0.role == .protectionEngine
        }), Self.matchesArtifactContract(artifact),
              artifact.sourceWeightsSHA256 == handoff.manifest.provenance.sourceCheckpointSHA256 else {
            throw FACADEProtectionError.modelUnavailable(
                "the installed protection model does not match the one-pass student contract"
            )
        }
        self.model = model
        self.codec = codec
        self.safetyGate = safetyGate
        self.modelBundleIdentifier = handoff.manifest.bundleIdentifier
        self.engineIdentifier = "facade.student.one-pass.\(artifact.sourceWeightsSHA256.prefix(12))"
    }

    public func generateCandidate(
        for input: FACADEProtectionInput,
        policy: FACADEProtectionPolicy
    ) async throws -> FACADEProtectionCandidate {
        try safetyGate.requireSafeForIntensiveWork("Local student inference")
        try Task.checkCancellation()
        guard policy.mode == .experimentalDemo,
              policy.requiredEvidenceTier == .experimentalProxyOnly else {
            throw FACADEProtectionError.evidenceTierRejected(
                "an unpromoted one-pass student can only run in experimental mode"
            )
        }
        guard policy.headCount == 1 else {
            throw FACADEProtectionError.invalidPolicy(
                "the one-pass student supports only k=1; k=3 and k=8 remain iterative-demo benchmarks"
            )
        }
        try Self.requireL16ResearchComparatorPolicy(policy)

        let clean = try codec.decodeAndPrepare(input.originalData)
        guard clean.width == Self.spatialSize, clean.height == Self.spatialSize else {
            throw FACADEProtectionError.invalidInput(
                "the one-pass student requires a 256x256 prepared RGB frame"
            )
        }
        let tensor = try Self.multiArray(from: clean)
        let provider = try MLDictionaryFeatureProvider(dictionary: [Self.inputName: tensor])
        let prediction: MLFeatureProvider
        do {
            prediction = try await model.prediction(from: provider)
        } catch {
            throw FACADEProtectionError.candidateGenerationFailed(
                "one-pass Core ML student prediction failed: \(error)"
            )
        }
        try safetyGate.requireSafeForIntensiveWork("Local student inference")
        try Task.checkCancellation()
        guard let output = prediction.featureValue(for: Self.outputName)?.multiArrayValue,
              output.shape.map(\.intValue) == [1, 3, Self.spatialSize, Self.spatialSize],
              output.count == clean.values.count else {
            throw FACADEProtectionError.modelUnavailable(
                "the one-pass Core ML student output contract changed"
            )
        }

        var protectedValues = [Float](repeating: 0, count: clean.values.count)
        for index in protectedValues.indices {
            let residual = output[index].floatValue
            guard residual.isFinite, abs(residual) <= Self.residualLimit + 1e-6 else {
                throw FACADEProtectionError.candidateGenerationFailed(
                    "the one-pass student returned a non-finite or out-of-bound residual"
                )
            }
            protectedValues[index] = min(1, max(0, clean.values[index] + residual))
        }
        let protected = try FACADEFloatImage(
            width: Self.spatialSize,
            height: Self.spatialSize,
            values: protectedValues
        )
        return FACADEProtectionCandidate(
            data: try codec.encodePNG(protected),
            uniformTypeIdentifier: "public.png",
            engineIdentifier: engineIdentifier,
            modelBundleIdentifier: modelBundleIdentifier,
            evidenceTier: .experimentalProxyOnly
        )
    }

    static func matchesArtifactContract(_ artifact: FACADEModelArtifact) -> Bool {
        guard artifact.role == .protectionEngine,
              artifact.format == .coreMLPackage || artifact.format == .compiledCoreML,
              artifact.storagePrecision == .int8,
              artifact.gradientSemantics == .none,
              artifact.inputs.count == 1,
              artifact.outputs.count == 1 else {
            return false
        }
        let input = artifact.inputs[0]
        let output = artifact.outputs[0]
        // The manifest/schema pins the mathematical Double value 16/255. The
        // runtime bound is Float, whose widened value differs by ~3.7e-9.
        // Validate metadata against the schema value rather than that widened
        // runtime approximation.
        let declaredResidualLimit = 16.0 / 255.0
        return input.name == Self.inputName &&
            input.shape == [1, 3, Self.spatialSize, Self.spatialSize] &&
            input.dataType == "float32" && input.layout == "NCHW" &&
            input.valueRange == [0, 1] &&
            output.name == Self.outputName &&
            output.shape == [1, 3, Self.spatialSize, Self.spatialSize] &&
            output.dataType == "float32" && output.layout == "NCHW" &&
            output.valueRange.count == 2 &&
            abs(output.valueRange[0] + declaredResidualLimit) < 1e-12 &&
            abs(output.valueRange[1] - declaredResidualLimit) < 1e-12
    }

    static func requireL16ResearchComparatorPolicy(
        _ policy: FACADEProtectionPolicy
    ) throws {
        guard policy.uint8LInfinityMaximum ==
                FACADEProtectionPolicy.legacyDCTLowResearchComparatorUInt8LInfinityMaximum else {
            throw FACADEProtectionError.invalidPolicy(
                "the one-pass student is fixed at uint8 L-infinity 16 and may only run as the explicit L16 research comparator"
            )
        }
    }

    private static func multiArray(from image: FACADEFloatImage) throws -> MLMultiArray {
        let result = try MLMultiArray(
            shape: [1, 3, Self.spatialSize, Self.spatialSize] as [NSNumber],
            dataType: .float32
        )
        guard result.count == image.values.count else {
            throw FACADEProtectionError.modelUnavailable(
                "Core ML allocated an unexpected one-pass input tensor"
            )
        }
        let destination = result.dataPointer.bindMemory(to: Float.self, capacity: result.count)
        image.values.withUnsafeBufferPointer { source in
            guard let address = source.baseAddress else { return }
            destination.update(from: address, count: source.count)
        }
        return result
    }
}
