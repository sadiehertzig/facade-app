// NativePreparationLoader.swift
//
// Loads and checks the FACADE model bundle (the "FACADEModelHandoff" folder)
// that the 256-pixel first stage of the engine needs.
//
// Where it sits: NativePipeline calls `load` once per run, before loading the
// other native models. The FACADEKit package does the detailed manifest
// validation; this file wires its results into the engine's types.
//
// Key ideas:
// 1. The bundle's manifest.json is pinned to one SHA-256. A different or
//    re-exported bundle is refused rather than used, because the engine's
//    expected step counts and model shapes were only checked against this one.
// 2. Two Core ML models come out of it: a model that proposes a starting
//    perturbation for the 256-pixel stage, and an ArcFace-style face
//    recogniser used to compute the reference identity vector.
// 3. Core ML packages (.mlpackage) are compiled on the phone on first use;
//    already-compiled models (.mlmodelc) are loaded directly.
import Foundation
import CoreML
import FACADEKit

/// Everything the 256-pixel stage needs, already validated and loaded.
struct NativePreparationRuntime {
    /// Decodes a photo and resizes it to the 256 x 256 float image the stage uses.
    let codec: any FACADELocalVerificationImageCodec
    /// The Core ML model that proposes the starting perturbation ("residuals").
    let initializer: any FACADEInitializerResidualBank
    /// Apple Vision face detector that must find exactly one face and gives the
    /// alignment transform for the recogniser crop.
    let detector: any FACADEExactlyOneFaceDetector
    /// The ArcFace-style recogniser that produces 512-number identity vectors.
    let arcFace: FACADECoreMLEmbeddingModel
}
/// Loader for the model bundle.
enum NativePreparationLoader {
    // Only the exact manifest version this app was validated with is accepted.
    static let manifestSHA256 = "f5bb3d07302de5116b7d207e5c128070ef7f30e28fce86bb7c2b00e916ce8648"
    /// Reads, verifies and loads the bundle at `rootURL`.
    static func load(rootURL: URL) throws -> NativePreparationRuntime {
        // Thermal/time checkpoints only run during a live protection run.
        if Profile.deadline != nil {try Profile.boundary("preparation.validation.before")}
        let bytes = try Data(contentsOf: rootURL.appendingPathComponent("manifest.json"))
        // First gate: the raw manifest bytes must hash to the pinned value.
        guard TightCleanup.sha(bytes) == manifestSHA256 else {
            throw NativeError.invalid("Preparation manifest differs from frozen validated contract")
        }
        // FACADEKit checks every artifact listed in the manifest (sizes, hashes,
        // roles, formats) and resolves their URLs.
        let handoff = try Profile.measure("loader.validate.all") {
            try FACADEModelHandoffLoader.load(rootURL: rootURL, mode: .experimentalDemo)
        }
        // Second gate: the loader's own hash must agree with the pre-check above.
        guard handoff.manifestSHA256 == manifestSHA256 else {
            throw NativeError.invalid("Preparation manifest changed during validation")
        }
        if Profile.deadline != nil {try Profile.boundary("preparation.validation.after")}
        // The 512 x 512 identity-map parse and bounds check is kept as a validation
        // step even though the native pipeline loads its own copy of that matrix.
        _ = try FACADEIdentityMapMatrix.load(from: handoff)
        /// Finds the manifest entry for a model role or throws.
        func artifact(_ role: FACADEModelRole) throws -> FACADEModelArtifact {
            guard let a = handoff.manifest.artifacts.first(where: {$0.role == role}) else {
                throw NativeError.invalid("Missing preparation artifact")
            }
            return a
        }
        /// Loads one Core ML model by role, compiling it first if it is a package.
        func load(_ role: FACADEModelRole) throws -> MLModel {
            if Profile.deadline != nil {try Profile.boundary("preparation.model.before."+role.rawValue)}
            let a = try artifact(role)
            guard let source = handoff.artifactURLs[role] else {
                throw NativeError.invalid("Missing validated preparation URL")
            }
            let runtime: URL
            // Already compiled: use it in place.
            switch a.format {
            // A source package: Core ML compiles it to a temporary .mlmodelc.
            case .compiledCoreML: runtime = source
            case .coreMLPackage:
                runtime = try Profile.measure("loader.compile."+role.rawValue) {
                    try MLModel.compileModel(at: source)
                }
            // The bundle can describe MPSGraph models, but this stage only uses Core ML.
            case .mpsGraphPackage: throw NativeError.invalid("Preparation model is not Core ML")
            }
            if Profile.deadline != nil {try Profile.boundary("preparation.compile.after."+role.rawValue)}
            // Let Core ML choose CPU, GPU or Neural Engine for these two models.
            let configuration = MLModelConfiguration()
            configuration.computeUnits = .all
            let model = try Profile.measure("loader.load."+role.rawValue) {
                try MLModel(contentsOf: runtime, configuration: configuration)
            }
            if Profile.deadline != nil {try Profile.boundary("preparation.model.after."+role.rawValue)}
            return model
        }
        // The two models this stage needs.
        let protection = try load(.protectionEngine)
        let arc = try load(.arcFaceForwardVerifier)
        // 256-pixel square PNG codec for the stage's working image.
        let codec = FACADECoreImagePNGCodec(side: 256)
        // Face detection settings (Vision request revision, minimum face size) come
        // from the manifest so they match how the bundle was validated.
        let detector = try FACADEVisionSingleFaceDetector(
            requestRevision: handoff.manifest.faceDetectionAlignment.requestRevision,
            minimumFaceFraction: handoff.manifest.faceDetectionAlignment.minimumFaceFraction)
        // Input name "aligned_rgb_nchw_minus1_to1": a 1x3x112x112 tensor, channels
        // first, scaled to -1...1. Output "l2_embedding_512": a unit-length vector.
        let arcFace = try FACADECoreMLEmbeddingModel(
            model: arc, artifact: artifact(.arcFaceForwardVerifier),
            expectedRole: .arcFaceForwardVerifier, inputName: "aligned_rgb_nchw_minus1_to1",
            outputName: "l2_embedding_512", normalization: .arcFaceMinusOneToOne)
        // Wraps the perturbation model with the manifest's fixture checks.
        let initializer = try FACADECoreMLR6InitializerBank(model: protection, handoff: handoff)
        return NativePreparationRuntime(codec: codec, initializer: initializer,
                                        detector: detector, arcFace: arcFace)
    }
}
