import Foundation
import CoreML
import FACADEKit

struct NativePreparationRuntime {
    let codec: any FACADELocalVerificationImageCodec
    let initializer: any FACADEInitializerResidualBank
    let detector: any FACADEExactlyOneFaceDetector
    let arcFace: FACADECoreMLEmbeddingModel
}
enum NativePreparationLoader {
    // This experiment supports only the already accepted exact R6 manifest.
    static let manifestSHA256 = "f5bb3d07302de5116b7d207e5c128070ef7f30e28fce86bb7c2b00e916ce8648"
    static func load(rootURL: URL) throws -> NativePreparationRuntime {
        if Profile.deadline != nil {try Profile.boundary("preparation.validation.before")}
        let bytes = try Data(contentsOf: rootURL.appendingPathComponent("manifest.json"))
        guard TightCleanup.sha(bytes) == manifestSHA256 else {
            throw NativeError.invalid("Preparation manifest differs from frozen validated contract")
        }
        let handoff = try Profile.measure("loader.validate.all") {
            try FACADEModelHandoffLoader.load(rootURL: rootURL, mode: .experimentalDemo)
        }
        guard handoff.manifestSHA256 == manifestSHA256 else {
            throw NativeError.invalid("Preparation manifest changed during validation")
        }
        if Profile.deadline != nil {try Profile.boundary("preparation.validation.after")}
        // Retain the legacy identity-map parsing/bounds check although the
        // native pipeline loads its own exact identity-map representation.
        _ = try FACADEIdentityMapMatrix.load(from: handoff)
        func artifact(_ role: FACADEModelRole) throws -> FACADEModelArtifact {
            guard let a = handoff.manifest.artifacts.first(where: {$0.role == role}) else {
                throw NativeError.invalid("Missing preparation artifact")
            }
            return a
        }
        func load(_ role: FACADEModelRole) throws -> MLModel {
            if Profile.deadline != nil {try Profile.boundary("preparation.model.before."+role.rawValue)}
            let a = try artifact(role)
            guard let source = handoff.artifactURLs[role] else {
                throw NativeError.invalid("Missing validated preparation URL")
            }
            let runtime: URL
            switch a.format {
            case .compiledCoreML: runtime = source
            case .coreMLPackage:
                runtime = try Profile.measure("loader.compile."+role.rawValue) {
                    try MLModel.compileModel(at: source)
                }
            case .mpsGraphPackage: throw NativeError.invalid("Preparation model is not Core ML")
            }
            if Profile.deadline != nil {try Profile.boundary("preparation.compile.after."+role.rawValue)}
            let configuration = MLModelConfiguration()
            configuration.computeUnits = .all
            let model = try Profile.measure("loader.load."+role.rawValue) {
                try MLModel(contentsOf: runtime, configuration: configuration)
            }
            if Profile.deadline != nil {try Profile.boundary("preparation.model.after."+role.rawValue)}
            return model
        }
        let protection = try load(.protectionEngine)
        let arc = try load(.arcFaceForwardVerifier)
        let codec = FACADECoreImagePNGCodec(side: 256)
        let detector = try FACADEVisionSingleFaceDetector(
            requestRevision: handoff.manifest.faceDetectionAlignment.requestRevision,
            minimumFaceFraction: handoff.manifest.faceDetectionAlignment.minimumFaceFraction)
        let arcFace = try FACADECoreMLEmbeddingModel(
            model: arc, artifact: artifact(.arcFaceForwardVerifier),
            expectedRole: .arcFaceForwardVerifier, inputName: "aligned_rgb_nchw_minus1_to1",
            outputName: "l2_embedding_512", normalization: .arcFaceMinusOneToOne)
        let initializer = try FACADECoreMLR6InitializerBank(model: protection, handoff: handoff)
        return NativePreparationRuntime(codec: codec, initializer: initializer,
                                        detector: detector, arcFace: arcFace)
    }
}
