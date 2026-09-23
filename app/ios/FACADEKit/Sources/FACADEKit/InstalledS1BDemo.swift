import CoreML
import Foundation

public enum FACADEInstalledEngineKind: String, Sendable {
    case iterativeS1B = "iterative_s1b"
    case onePassStudent = "one_pass_student"
}

public struct FACADECoreMLRoleComputePlan: @unchecked Sendable {
    public let defaultComputeUnits: MLComputeUnits
    public let simSwapComputeUnits: MLComputeUnits

    public init(
        defaultComputeUnits: MLComputeUnits,
        simSwapComputeUnits: MLComputeUnits? = nil
    ) {
        self.defaultComputeUnits = defaultComputeUnits
        self.simSwapComputeUnits = simSwapComputeUnits ?? defaultComputeUnits
    }

    public func computeUnits(for role: FACADEModelRole) -> MLComputeUnits {
        role == .simSwapForwardVerifier ? simSwapComputeUnits : defaultComputeUnits
    }

    public var evidenceIdentifier: String {
        if defaultComputeUnits == simSwapComputeUnits {
            return "uniform_\(Self.label(defaultComputeUnits))"
        }
        return [
            "protection=\(Self.label(defaultComputeUnits))",
            "arcface=\(Self.label(defaultComputeUnits))",
            "gradient=\(Self.label(defaultComputeUnits))",
            "simswap=\(Self.label(simSwapComputeUnits))",
        ].joined(separator: ";")
    }

    private static func label(_ computeUnits: MLComputeUnits) -> String {
        switch computeUnits {
        case .all: "all"
        case .cpuOnly: "cpu_only"
        case .cpuAndGPU: "cpu_and_gpu"
        case .cpuAndNeuralEngine: "cpu_and_neural_engine"
        @unknown default: "unknown"
        }
    }
}

/// Exact component instances used by the installed iterative engine. They are
/// exposed only so the scalar-only stage-isolation harness can execute the
/// same codec, initializer, and objective without changing optimizer behavior.
public struct FACADEInstalledS1BStageIsolationRuntime: Sendable {
    public let codec: any FACADELocalVerificationImageCodec
    public let initializer: any FACADEInitializerResidualBank
    public let objective: any FACADEInputGradientObjective
    public let detector: any FACADEExactlyOneFaceDetector
    public let arcFace: FACADECoreMLEmbeddingModel
    public let simSwap: FACADECoreMLEmbeddingModel

    public init(
        codec: any FACADELocalVerificationImageCodec,
        initializer: any FACADEInitializerResidualBank,
        objective: any FACADEInputGradientObjective,
        detector: any FACADEExactlyOneFaceDetector,
        arcFace: FACADECoreMLEmbeddingModel,
        simSwap: FACADECoreMLEmbeddingModel
    ) {
        self.codec = codec
        self.initializer = initializer
        self.objective = objective
        self.detector = detector
        self.arcFace = arcFace
        self.simSwap = simSwap
    }
}

public struct FACADEInstalledS1BDemo: Sendable {
    public let engine: any FACADEProtectionEngine
    public let verifier: any FACADELocalVerifier
    public let handoff: FACADELoadedModelHandoff
    /// Exact model locations executed by Core ML. Device evidence records
    /// deterministic hashes for these in addition to source package hashes.
    public let runtimeCoreMLURLs: [FACADEModelRole: URL]
    public let engineKind: FACADEInstalledEngineKind
    public let supportedHeadCounts: [Int]
    public let stageIsolationRuntime: FACADEInstalledS1BStageIsolationRuntime?
    public let requestedComputePlanIdentifier: String
    public let optimizerUpdateRuleIdentifier: String?

    public init(
        engine: any FACADEProtectionEngine,
        verifier: any FACADELocalVerifier,
        handoff: FACADELoadedModelHandoff,
        runtimeCoreMLURLs: [FACADEModelRole: URL],
        engineKind: FACADEInstalledEngineKind,
        supportedHeadCounts: [Int],
        stageIsolationRuntime: FACADEInstalledS1BStageIsolationRuntime?,
        requestedComputePlanIdentifier: String,
        optimizerUpdateRuleIdentifier: String?
    ) {
        self.engine = engine
        self.verifier = verifier
        self.handoff = handoff
        self.runtimeCoreMLURLs = runtimeCoreMLURLs
        self.engineKind = engineKind
        self.supportedHeadCounts = supportedHeadCounts
        self.stageIsolationRuntime = stageIsolationRuntime
        self.requestedComputePlanIdentifier = requestedComputePlanIdentifier
        self.optimizerUpdateRuleIdentifier = optimizerUpdateRuleIdentifier
    }
}

public enum FACADEInstalledS1BDemoLoader {
    public static let preprocessingFormula =
        "r6:rgb_0_1;arcface:rgb_0_1_times_2_minus_1;simswap:imagenet_rgb"
    public static let resizeKernel = "bilinear_half_pixel_centers"
    public static let pixelRounding = "protected_png_round_to_nearest_clamp_uint8"

    /// Loads the current frozen r6 student and every fail-closed dependency.
    /// The method intentionally has no partial-success return: one missing
    /// model, hash, threshold, document, fixture, or gradient graph rejects the
    /// complete demo runtime.
    public static func load(
        rootURL: URL,
        computeUnits: MLComputeUnits = .all,
        simSwapComputeUnits: MLComputeUnits? = nil,
        compiledModelCacheRootURL: URL? = nil,
        optimizerUpdateRule: FACADEOptimizerUpdateRule = .currentMaxNormalizedDCTLowV1
    ) throws -> FACADEInstalledS1BDemo {
        let computePlan = FACADECoreMLRoleComputePlan(
            defaultComputeUnits: computeUnits,
            simSwapComputeUnits: simSwapComputeUnits
        )
        let handoff = try FACADEModelHandoffLoader.load(
            rootURL: rootURL,
            mode: .experimentalDemo
        )
        let engineKind = try validateDemoContract(handoff.manifest)

        let protectionArtifact = try artifact(.protectionEngine, in: handoff)
        let arcFaceArtifact = try artifact(.arcFaceForwardVerifier, in: handoff)
        let simSwapArtifact = try artifact(.simSwapForwardVerifier, in: handoff)
        let gradientArtifact = try artifact(.arcFaceScoreGradient, in: handoff)

        let protectionLoad = try loadCoreML(
            artifact: protectionArtifact,
            handoffURL: try artifactURL(.protectionEngine, in: handoff),
            computeUnits: computePlan.computeUnits(for: .protectionEngine),
            compiledModelCacheRootURL: compiledModelCacheRootURL
        )
        let arcFaceLoad = try loadCoreML(
            artifact: arcFaceArtifact,
            handoffURL: try artifactURL(.arcFaceForwardVerifier, in: handoff),
            computeUnits: computePlan.computeUnits(for: .arcFaceForwardVerifier),
            compiledModelCacheRootURL: compiledModelCacheRootURL
        )
        let simSwapLoad = try loadCoreML(
            artifact: simSwapArtifact,
            handoffURL: try artifactURL(.simSwapForwardVerifier, in: handoff),
            computeUnits: computePlan.computeUnits(for: .simSwapForwardVerifier),
            compiledModelCacheRootURL: compiledModelCacheRootURL
        )

        let codec = FACADECoreImagePNGCodec(side: 256)
        let detector = try FACADEVisionSingleFaceDetector(
            requestRevision: handoff.manifest.faceDetectionAlignment.requestRevision,
            minimumFaceFraction: handoff.manifest.faceDetectionAlignment.minimumFaceFraction
        )
        let arcFace = try FACADECoreMLEmbeddingModel(
            model: arcFaceLoad.model,
            artifact: arcFaceArtifact,
            expectedRole: .arcFaceForwardVerifier,
            inputName: "aligned_rgb_nchw_minus1_to1",
            outputName: "l2_embedding_512",
            normalization: .arcFaceMinusOneToOne
        )
        let simSwap = try FACADECoreMLEmbeddingModel(
            model: simSwapLoad.model,
            artifact: simSwapArtifact,
            expectedRole: .simSwapForwardVerifier,
            inputName: "imagenet_normalized_rgb_112",
            outputName: "l2_simswap_embedding_512",
            normalization: .imageNetRGB
        )
        let gradientURL = try artifactURL(.arcFaceScoreGradient, in: handoff)
        var gradientExecutor: (any FACADEArcFaceGradientExecutor)?
        var runtimeCoreMLURLs: [FACADEModelRole: URL] = [
            .protectionEngine: protectionLoad.runtimeURL,
            .arcFaceForwardVerifier: arcFaceLoad.runtimeURL,
            .simSwapForwardVerifier: simSwapLoad.runtimeURL,
        ]
        switch gradientArtifact.format {
        case .coreMLPackage, .compiledCoreML:
            let gradientLoad = try loadCoreML(
                artifact: gradientArtifact,
                handoffURL: gradientURL,
                computeUnits: computePlan.computeUnits(for: .arcFaceScoreGradient),
                compiledModelCacheRootURL: compiledModelCacheRootURL
            )
            gradientExecutor = try FACADECoreMLStaticScoreGradientExecutor(
                model: gradientLoad.model,
                artifact: gradientArtifact
            )
            runtimeCoreMLURLs[.arcFaceScoreGradient] = gradientLoad.runtimeURL
        case .mpsGraphPackage:
            gradientExecutor = try FACADEMPSGraphScoreGradientExecutor(
                packageURL: gradientURL,
                artifact: gradientArtifact
            )
        }
        let engine: any FACADEProtectionEngine
        let supportedHeadCounts: [Int]
        let stageIsolationRuntime: FACADEInstalledS1BStageIsolationRuntime?
        switch engineKind {
        case .iterativeS1B:
            guard let gradientExecutor else {
                throw FACADEProtectionError.modelUnavailable(
                    "the S1-B runtime requires its installed score-gradient model"
                )
            }
            let initializer = try FACADECoreMLR6InitializerBank(
                model: protectionLoad.model,
                handoff: handoff
            )
            let identityMap = try FACADEIdentityMapMatrix.load(from: handoff)
            let objective = FACADEAppleS1BInputGradientObjective(
                detector: detector,
                arcFace: arcFace,
                executor: gradientExecutor,
                identityMap: identityMap
            )
            engine = try FACADEExperimentalS1BEngine(
                engineIdentifier: optimizerUpdateRule.engineIdentifier,
                modelBundleIdentifier: handoff.manifest.bundleIdentifier,
                codec: codec,
                initializer: initializer,
                objective: objective,
                optimizerUpdateRule: optimizerUpdateRule
            )
            supportedHeadCounts = [1, 3, 8]
            stageIsolationRuntime = FACADEInstalledS1BStageIsolationRuntime(
                codec: codec,
                initializer: initializer,
                objective: objective,
                detector: detector,
                arcFace: arcFace,
                simSwap: simSwap
            )
        case .onePassStudent:
            engine = try FACADECoreMLOnePassResidualEngine(
                model: protectionLoad.model,
                handoff: handoff,
                codec: codec
            )
            supportedHeadCounts = [1]
            stageIsolationRuntime = nil
        }
        let verifier = try FACADECoreMLLocalVerifier(
            handoff: handoff,
            codec: codec,
            detector: detector,
            arcFace: arcFace,
            simSwap: simSwap
        )
        return FACADEInstalledS1BDemo(
            engine: engine,
            verifier: verifier,
            handoff: handoff,
            runtimeCoreMLURLs: runtimeCoreMLURLs,
            engineKind: engineKind,
            supportedHeadCounts: supportedHeadCounts,
            stageIsolationRuntime: stageIsolationRuntime,
            requestedComputePlanIdentifier: computePlan.evidenceIdentifier,
            optimizerUpdateRuleIdentifier: engineKind == .iterativeS1B
                ? optimizerUpdateRule.rawValue
                : nil
        )
    }

    private static func validateDemoContract(
        _ manifest: FACADEModelHandoffManifest
    ) throws -> FACADEInstalledEngineKind {
        guard manifest.evidence.tier == .experimentalProxyOnly,
              manifest.preprocessing.channelOrder == "RGB",
              manifest.preprocessing.cropSize == [112, 112],
              [
                preprocessingFormula,
                "protection:rgb_0_1;arcface:rgb_0_1_times_2_minus_1;simswap:imagenet_rgb",
              ].contains(manifest.preprocessing.normalization),
              manifest.preprocessing.resizeKernel == resizeKernel,
              manifest.preprocessing.pixelRounding == pixelRounding,
              manifest.faceDetectionAlignment.coordinateConvention ==
                FACADEVisionSingleFaceDetector.coordinateConvention,
              let protection = manifest.artifacts.first(where: { $0.role == .protectionEngine }),
              manifest.provenance.sourceCheckpointSHA256 == protection.sourceWeightsSHA256 else {
            throw FACADEProtectionError.modelUnavailable(
                "the handoff does not match an accepted experimental phone engine contract"
            )
        }
        let isR6 = protection.sourceWeightsSHA256 ==
            FACADEExperimentalS1BEngine.frozenR6CheckpointSHA256 &&
            protection.inputs.map(\.name) == ["image_nchw_0_1"] &&
            protection.inputs.map(\.shape) == [[1, 3, 256, 256]] &&
            protection.outputs.map(\.name) == ["residual_bank_n24chw"] &&
            protection.outputs.map(\.shape) == [[1, 24, 256, 256]]
        if isR6 {
            return .iterativeS1B
        }
        if FACADECoreMLOnePassResidualEngine.matchesArtifactContract(protection) {
            return .onePassStudent
        }
        throw FACADEProtectionError.modelUnavailable(
            "the protection artifact is neither the frozen r6/S1-B engine nor the exact one-pass student contract"
        )
    }

    private static func artifact(
        _ role: FACADEModelRole,
        in handoff: FACADELoadedModelHandoff
    ) throws -> FACADEModelArtifact {
        guard let artifact = handoff.manifest.artifacts.first(where: { $0.role == role }) else {
            throw FACADEProtectionError.modelUnavailable("missing \(role.rawValue) artifact")
        }
        return artifact
    }

    private static func artifactURL(
        _ role: FACADEModelRole,
        in handoff: FACADELoadedModelHandoff
    ) throws -> URL {
        guard let url = handoff.artifactURLs[role] else {
            throw FACADEProtectionError.modelUnavailable("missing \(role.rawValue) artifact URL")
        }
        return url
    }

    private struct LoadedCoreML {
        let model: MLModel
        let runtimeURL: URL
    }

    private static func loadCoreML(
        artifact: FACADEModelArtifact,
        handoffURL: URL,
        computeUnits: MLComputeUnits,
        compiledModelCacheRootURL: URL?
    ) throws -> LoadedCoreML {
        let runtimeURL = try materializeCoreMLRuntime(
            artifact: artifact,
            handoffURL: handoffURL,
            cacheRootURL: compiledModelCacheRootURL
        )
        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits
        do {
            return LoadedCoreML(
                model: try MLModel(contentsOf: runtimeURL, configuration: configuration),
                runtimeURL: runtimeURL
            )
        } catch {
            throw FACADEProtectionError.modelUnavailable(
                "Core ML could not load \(artifact.role.rawValue): \(error)"
            )
        }
    }

    /// When a cache root is supplied, the exact compiled package used by Core
    /// ML is materialized once under a source-artifact-derived name and never
    /// overwritten. This lets physical-device evidence preserve the runtime
    /// bytes across the separate Instruments launches needed for all phases.
    private static func materializeCoreMLRuntime(
        artifact: FACADEModelArtifact,
        handoffURL: URL,
        cacheRootURL: URL?
    ) throws -> URL {
        if let cacheRootURL {
            guard cacheRootURL.isFileURL else {
                throw FACADEProtectionError.modelUnavailable(
                    "compiled Core ML cache must be a local directory"
                )
            }
            try FileManager.default.createDirectory(
                at: cacheRootURL,
                withIntermediateDirectories: true
            )
            let rootValues = try cacheRootURL.resourceValues(
                forKeys: [.isDirectoryKey, .isSymbolicLinkKey]
            )
            guard rootValues.isDirectory == true, rootValues.isSymbolicLink != true else {
                throw FACADEProtectionError.modelUnavailable(
                    "compiled Core ML cache is not a safe local directory"
                )
            }
            let cachedURL = cacheRootURL.appendingPathComponent(
                "\(artifact.role.rawValue)-\(artifact.sha256).mlmodelc",
                isDirectory: true
            )
            var cachedIsDirectory: ObjCBool = false
            if FileManager.default.fileExists(
                atPath: cachedURL.path,
                isDirectory: &cachedIsDirectory
            ) {
                guard cachedIsDirectory.boolValue,
                      (try cachedURL.resourceValues(forKeys: [.isSymbolicLinkKey]))
                        .isSymbolicLink != true else {
                    throw FACADEProtectionError.modelUnavailable(
                        "cached Core ML runtime is not a safe directory"
                    )
                }
                return cachedURL
            }

            let generatedURL = try compileCoreMLRuntime(
                artifact: artifact,
                handoffURL: handoffURL
            )
            let stagingURL = cacheRootURL.appendingPathComponent(
                ".staging-\(UUID().uuidString).mlmodelc",
                isDirectory: true
            )
            do {
                try FileManager.default.copyItem(at: generatedURL, to: stagingURL)
                try FileManager.default.moveItem(at: stagingURL, to: cachedURL)
            } catch {
                try? FileManager.default.removeItem(at: stagingURL)
                throw FACADEProtectionError.modelUnavailable(
                    "could not preserve compiled \(artifact.role.rawValue) runtime: \(error)"
                )
            }
            return cachedURL
        }
        return try compileCoreMLRuntime(artifact: artifact, handoffURL: handoffURL)
    }

    private static func compileCoreMLRuntime(
        artifact: FACADEModelArtifact,
        handoffURL: URL
    ) throws -> URL {
        switch artifact.format {
        case .compiledCoreML:
            return handoffURL
        case .coreMLPackage:
            do {
                return try MLModel.compileModel(at: handoffURL)
            } catch {
                throw FACADEProtectionError.modelUnavailable(
                    "Core ML could not compile \(artifact.role.rawValue): \(error)"
                )
            }
        case .mpsGraphPackage:
            throw FACADEProtectionError.modelUnavailable(
                "\(artifact.role.rawValue) was declared as Core ML but supplied as MPSGraph"
            )
        }
    }
}

/// Stable app-facing name. The legacy S1-B loader remains source-compatible,
/// while this entry point can load either the temporary iterative demo or an
/// exact-contract one-pass student handoff.
public enum FACADEInstalledDemoLoader {
    public static func load(
        rootURL: URL,
        computeUnits: MLComputeUnits = .all,
        simSwapComputeUnits: MLComputeUnits? = nil,
        compiledModelCacheRootURL: URL? = nil,
        optimizerUpdateRule: FACADEOptimizerUpdateRule = .currentMaxNormalizedDCTLowV1
    ) throws -> FACADEInstalledS1BDemo {
        try FACADEInstalledS1BDemoLoader.load(
            rootURL: rootURL,
            computeUnits: computeUnits,
            simSwapComputeUnits: simSwapComputeUnits,
            compiledModelCacheRootURL: compiledModelCacheRootURL,
            optimizerUpdateRule: optimizerUpdateRule
        )
    }
}
