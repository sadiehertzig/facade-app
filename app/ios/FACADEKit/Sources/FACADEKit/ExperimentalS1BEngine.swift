import Foundation

public struct FACADEFloatImage: Sendable, Equatable {
    public let width: Int
    public let height: Int
    /// Channel-first RGB values, contiguous as C,H,W and expected in [0, 1].
    public let values: [Float]

    public init(width: Int, height: Int, values: [Float]) throws {
        guard width > 0, height > 0, values.count == 3 * width * height,
              values.allSatisfy(\.isFinite) else {
            throw FACADEProtectionError.invalidInput("invalid channel-first RGB tensor")
        }
        self.width = width
        self.height = height
        self.values = values
    }
}

public struct FACADEObjectiveGradient: Sendable, Equatable {
    public let score: Float
    public let inputGradient: FACADEFloatImage
    /// Scalar-only timings reported by the concrete objective. The engine
    /// aggregates these across evaluations; no tensor or pixel data is logged.
    public let stageLatencyMilliseconds: [String: Double]

    public init(
        score: Float,
        inputGradient: FACADEFloatImage,
        stageLatencyMilliseconds: [String: Double] = [:]
    ) throws {
        guard score.isFinite else {
            throw FACADEProtectionError.candidateGenerationFailed("the local objective score is non-finite")
        }
        guard stageLatencyMilliseconds.values.allSatisfy({ $0.isFinite && $0 >= 0 }) else {
            throw FACADEProtectionError.candidateGenerationFailed(
                "the local objective returned an invalid scalar latency"
            )
        }
        self.score = score
        self.inputGradient = inputGradient
        self.stageLatencyMilliseconds = stageLatencyMilliseconds
    }
}

public struct FACADEEngineRunDiagnostics: Codable, Sendable, Equatable {
    public let schemaVersion: Int
    /// Zero-based index into the frozen eight-head initializer bank.
    public let chosenHeadIndex: Int
    /// Zero-based evaluation index within the chosen head; the final
    /// post-update evaluation is `searchSteps + qualitySteps`.
    public let chosenEvaluationIndex: Int
    public let objectiveEvaluationCount: Int
    public let objectiveScore: Double
    /// Score of the untouched clean frame against the same reference vector.
    /// The engine fails closed unless `objectiveScore` is strictly below it;
    /// diagnostics serialized before the sign guard existed carry `nil`.
    public let cleanObjectiveScore: Double?
    public let requestedUInt8LInfinityMaximum: Int
    public let finalAchievedUInt8LInfinity: Int
    public let residualRMSUInt8: Double
    public let residualEnergyUInt8Squared: Double
    public let changedCoordinateCount: Int
    public let coordinateCount: Int
    public let changedCoordinateDensity: Double
    /// Fraction of stored RGB coordinates whose exact uint8 difference equals
    /// `requestedUInt8LInfinityMaximum`.
    public let storedClampSaturationFraction: Double
    /// Fraction of selected floating-point delta coordinates at the optimizer
    /// clamp before output-image clipping and uint8 rounding.
    public let optimizerClampSaturationFraction: Double
    public let stageLatencyMilliseconds: [String: Double]

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case chosenHeadIndex = "chosen_head_index_zero_based"
        case chosenEvaluationIndex = "chosen_evaluation_index_zero_based"
        case objectiveEvaluationCount = "objective_evaluation_count"
        case objectiveScore = "objective_score"
        case cleanObjectiveScore = "clean_objective_score"
        case requestedUInt8LInfinityMaximum = "requested_uint8_linf_maximum"
        case finalAchievedUInt8LInfinity = "final_achieved_uint8_linf"
        case residualRMSUInt8 = "residual_rms_uint8"
        case residualEnergyUInt8Squared = "residual_energy_uint8_squared"
        case changedCoordinateCount = "changed_coordinate_count"
        case coordinateCount = "coordinate_count"
        case changedCoordinateDensity = "changed_coordinate_density"
        case storedClampSaturationFraction = "stored_clamp_saturation_fraction"
        case optimizerClampSaturationFraction = "optimizer_clamp_saturation_fraction"
        case stageLatencyMilliseconds = "stage_latency_ms"
    }

    public init(
        schemaVersion: Int,
        chosenHeadIndex: Int,
        chosenEvaluationIndex: Int,
        objectiveEvaluationCount: Int,
        objectiveScore: Double,
        cleanObjectiveScore: Double? = nil,
        requestedUInt8LInfinityMaximum: Int =
            FACADEProtectionPolicy.legacyDCTLowResearchComparatorUInt8LInfinityMaximum,
        finalAchievedUInt8LInfinity: Int,
        residualRMSUInt8: Double,
        residualEnergyUInt8Squared: Double,
        changedCoordinateCount: Int,
        coordinateCount: Int,
        changedCoordinateDensity: Double,
        storedClampSaturationFraction: Double,
        optimizerClampSaturationFraction: Double,
        stageLatencyMilliseconds: [String: Double]
    ) {
        self.schemaVersion = schemaVersion
        self.chosenHeadIndex = chosenHeadIndex
        self.chosenEvaluationIndex = chosenEvaluationIndex
        self.objectiveEvaluationCount = objectiveEvaluationCount
        self.objectiveScore = objectiveScore
        self.cleanObjectiveScore = cleanObjectiveScore
        self.requestedUInt8LInfinityMaximum = requestedUInt8LInfinityMaximum
        self.finalAchievedUInt8LInfinity = finalAchievedUInt8LInfinity
        self.residualRMSUInt8 = residualRMSUInt8
        self.residualEnergyUInt8Squared = residualEnergyUInt8Squared
        self.changedCoordinateCount = changedCoordinateCount
        self.coordinateCount = coordinateCount
        self.changedCoordinateDensity = changedCoordinateDensity
        self.storedClampSaturationFraction = storedClampSaturationFraction
        self.optimizerClampSaturationFraction = optimizerClampSaturationFraction
        self.stageLatencyMilliseconds = stageLatencyMilliseconds
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let requestedUInt8LInfinityMaximum: Int
        if container.contains(.requestedUInt8LInfinityMaximum) {
            requestedUInt8LInfinityMaximum = try container.decode(
                Int.self,
                forKey: .requestedUInt8LInfinityMaximum
            )
        } else {
            requestedUInt8LInfinityMaximum =
                FACADEProtectionPolicy.legacyDCTLowResearchComparatorUInt8LInfinityMaximum
        }
        self.init(
            schemaVersion: try container.decode(Int.self, forKey: .schemaVersion),
            chosenHeadIndex: try container.decode(Int.self, forKey: .chosenHeadIndex),
            chosenEvaluationIndex: try container.decode(
                Int.self,
                forKey: .chosenEvaluationIndex
            ),
            objectiveEvaluationCount: try container.decode(
                Int.self,
                forKey: .objectiveEvaluationCount
            ),
            objectiveScore: try container.decode(Double.self, forKey: .objectiveScore),
            cleanObjectiveScore: try container.decodeIfPresent(
                Double.self,
                forKey: .cleanObjectiveScore
            ),
            requestedUInt8LInfinityMaximum: requestedUInt8LInfinityMaximum,
            finalAchievedUInt8LInfinity: try container.decode(
                Int.self,
                forKey: .finalAchievedUInt8LInfinity
            ),
            residualRMSUInt8: try container.decode(Double.self, forKey: .residualRMSUInt8),
            residualEnergyUInt8Squared: try container.decode(
                Double.self,
                forKey: .residualEnergyUInt8Squared
            ),
            changedCoordinateCount: try container.decode(
                Int.self,
                forKey: .changedCoordinateCount
            ),
            coordinateCount: try container.decode(Int.self, forKey: .coordinateCount),
            changedCoordinateDensity: try container.decode(
                Double.self,
                forKey: .changedCoordinateDensity
            ),
            storedClampSaturationFraction: try container.decode(
                Double.self,
                forKey: .storedClampSaturationFraction
            ),
            optimizerClampSaturationFraction: try container.decode(
                Double.self,
                forKey: .optimizerClampSaturationFraction
            ),
            stageLatencyMilliseconds: try container.decode(
                [String: Double].self,
                forKey: .stageLatencyMilliseconds
            )
        )
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(schemaVersion, forKey: .schemaVersion)
        try container.encode(chosenHeadIndex, forKey: .chosenHeadIndex)
        try container.encode(chosenEvaluationIndex, forKey: .chosenEvaluationIndex)
        try container.encode(objectiveEvaluationCount, forKey: .objectiveEvaluationCount)
        try container.encode(objectiveScore, forKey: .objectiveScore)
        try container.encodeIfPresent(cleanObjectiveScore, forKey: .cleanObjectiveScore)
        try container.encode(
            requestedUInt8LInfinityMaximum,
            forKey: .requestedUInt8LInfinityMaximum
        )
        try container.encode(finalAchievedUInt8LInfinity, forKey: .finalAchievedUInt8LInfinity)
        try container.encode(residualRMSUInt8, forKey: .residualRMSUInt8)
        try container.encode(residualEnergyUInt8Squared, forKey: .residualEnergyUInt8Squared)
        try container.encode(changedCoordinateCount, forKey: .changedCoordinateCount)
        try container.encode(coordinateCount, forKey: .coordinateCount)
        try container.encode(changedCoordinateDensity, forKey: .changedCoordinateDensity)
        try container.encode(
            storedClampSaturationFraction,
            forKey: .storedClampSaturationFraction
        )
        try container.encode(
            optimizerClampSaturationFraction,
            forKey: .optimizerClampSaturationFraction
        )
        try container.encode(stageLatencyMilliseconds, forKey: .stageLatencyMilliseconds)
    }
}

public struct FACADEInstrumentedProtectionCandidate: Sendable, Equatable {
    public let candidate: FACADEProtectionCandidate
    public let diagnostics: FACADEEngineRunDiagnostics

    public init(
        candidate: FACADEProtectionCandidate,
        diagnostics: FACADEEngineRunDiagnostics
    ) {
        self.candidate = candidate
        self.diagnostics = diagnostics
    }
}

public protocol FACADEExperimentalImageCodec: Sendable {
    func decodeAndPrepare(_ data: Data) throws -> FACADEFloatImage
    func encodePNG(_ image: FACADEFloatImage) throws -> Data
}

public struct FACADEVerificationRoundTrip: Sendable {
    public let image: FACADEFloatImage
    public let encodedByteCount: Int

    public init(image: FACADEFloatImage, encodedByteCount: Int) {
        self.image = image
        self.encodedByteCount = encodedByteCount
    }
}

/// The two frozen lossy transforms used by the independent local verifier.
/// Exposing them as separate calls lets the scalar-only harness bracket their
/// latency without changing transform bytes or production verifier behavior.
public protocol FACADELocalVerificationImageCodec: FACADEExperimentalImageCodec {
    func jpegQ70RoundTrip(_ image: FACADEFloatImage) throws -> FACADEVerificationRoundTrip
    func heicQ80RoundTrip(_ image: FACADEFloatImage) throws -> FACADEVerificationRoundTrip
}

public protocol FACADEInitializerResidualBank: Sendable {
    var sourceCheckpointSHA256: String { get }
    func residuals(for image: FACADEFloatImage) async throws -> [FACADEFloatImage]
}

/// The concrete phone backend owns face detection/alignment, the fixed clean
/// reference embedding, the actual-weight gradient graph, and its declared
/// objective. The engine owns only the frozen bounded DCT update mechanics.
public protocol FACADEInputGradientObjective: Sendable {
    func evaluate(
        clean: FACADEFloatImage,
        candidate: FACADEFloatImage
    ) async throws -> FACADEObjectiveGradient
}

/// Prospectively frozen P03 optimizer arms. The default remains the installed
/// max-normalized update; callers must opt in explicitly to sign-PGD.
public enum FACADEOptimizerUpdateRule: String, Codable, Sendable, CaseIterable {
    case currentMaxNormalizedDCTLowV1 = "current_max_normalized_dct_low_v1"
    case signPGDDCTLowV1 = "sign_pgd_dct_low_v1"

    public var engineIdentifier: String {
        switch self {
        case .currentMaxNormalizedDCTLowV1:
            "facade.s1b.r6.dct-low.experimental"
        case .signPGDDCTLowV1:
            "facade.s1b.r6.dct-low.sign-pgd.experimental"
        }
    }

    func updateDirection(projectedGradient: [Float]) throws -> [Float] {
        let maximum = projectedGradient.lazy.map { abs($0) }.max() ?? 0
        guard maximum.isFinite, maximum > 1e-12,
              projectedGradient.allSatisfy(\.isFinite) else {
            throw FACADEProtectionError.candidateGenerationFailed(
                "the on-device input gradient is empty or non-finite"
            )
        }
        switch self {
        case .currentMaxNormalizedDCTLowV1:
            return projectedGradient.map { $0 / maximum }
        case .signPGDDCTLowV1:
            return projectedGradient.map { value in
                if value > 0 { return 1 }
                if value < 0 { return -1 }
                return 0
            }
        }
    }

    func updatedResidual(
        residual: [Float],
        projectedGradient: [Float],
        stepSize: Float,
        clampLimit: Float
    ) throws -> [Float] {
        guard residual.count == projectedGradient.count else {
            throw FACADEProtectionError.candidateGenerationFailed(
                "the optimizer residual and gradient shapes do not match"
            )
        }
        let maximum = projectedGradient.lazy.map { abs($0) }.max() ?? 0
        guard maximum.isFinite, maximum > 1e-12,
              projectedGradient.allSatisfy(\.isFinite) else {
            throw FACADEProtectionError.candidateGenerationFailed(
                "the on-device input gradient is empty or non-finite"
            )
        }
        switch self {
        case .currentMaxNormalizedDCTLowV1:
            // Preserve the installed expression and floating-point operation
            // order exactly; this is the non-negotiable P03 baseline arm.
            return zip(residual, projectedGradient).map { value, gradient in
                min(clampLimit, max(-clampLimit, value - stepSize * gradient / maximum))
            }
        case .signPGDDCTLowV1:
            return zip(residual, projectedGradient).map { value, gradient in
                let direction: Float = gradient > 0 ? 1 : (gradient < 0 ? -1 : 0)
                return min(clampLimit, max(-clampLimit, value - stepSize * direction))
            }
        }
    }
}

public struct FACADEExperimentalS1BEngine: FACADEInstrumentedProtectionEngine {
    public static let frozenR6CheckpointSHA256 =
        "c1a1bc2c61da8811f1681a2bf35134411414a2c7ae98b76b0df802b9df5e1c20"

    public let engineIdentifier: String
    public let modelBundleIdentifier: String

    private let codec: any FACADEExperimentalImageCodec
    private let initializer: any FACADEInitializerResidualBank
    private let objective: any FACADEInputGradientObjective
    private let safetyGate: FACADEDeviceSafetyGate
    private let searchStep: Float
    private let qualityStep: Float
    public let optimizerUpdateRule: FACADEOptimizerUpdateRule

    public init(
        engineIdentifier: String = "facade.s1b.r6.dct-low.experimental",
        modelBundleIdentifier: String,
        codec: any FACADEExperimentalImageCodec,
        initializer: any FACADEInitializerResidualBank,
        objective: any FACADEInputGradientObjective,
        safetyGate: FACADEDeviceSafetyGate = FACADEDeviceSafetyGate(),
        searchStepPixels: Float = 1.0,
        qualityStepPixels: Float = 0.1,
        optimizerUpdateRule: FACADEOptimizerUpdateRule = .currentMaxNormalizedDCTLowV1
    ) throws {
        guard !modelBundleIdentifier.isEmpty else {
            throw FACADEProtectionError.modelUnavailable("the S1-B model bundle identifier is missing")
        }
        guard initializer.sourceCheckpointSHA256 == Self.frozenR6CheckpointSHA256 else {
            throw FACADEProtectionError.modelUnavailable("the installed initializer is not the frozen r6 student")
        }
        guard searchStepPixels.isFinite, searchStepPixels > 0,
              qualityStepPixels.isFinite, qualityStepPixels >= 0 else {
            throw FACADEProtectionError.invalidPolicy("S1-B pixel steps must be finite and nonnegative")
        }
        self.engineIdentifier = engineIdentifier
        self.modelBundleIdentifier = modelBundleIdentifier
        self.codec = codec
        self.initializer = initializer
        self.objective = objective
        self.safetyGate = safetyGate
        self.searchStep = searchStepPixels / 255.0
        self.qualityStep = qualityStepPixels / 255.0
        self.optimizerUpdateRule = optimizerUpdateRule
    }

    public func generateCandidate(
        for input: FACADEProtectionInput,
        policy: FACADEProtectionPolicy
    ) async throws -> FACADEProtectionCandidate {
        try await generateInstrumentedCandidate(for: input, policy: policy).candidate
    }

    public func generateInstrumentedCandidate(
        for input: FACADEProtectionInput,
        policy: FACADEProtectionPolicy
    ) async throws -> FACADEInstrumentedProtectionCandidate {
        let engineStarted = DispatchTime.now().uptimeNanoseconds
        var stageLatencyMilliseconds: [String: Double] = [:]
        try safetyGate.requireSafeForIntensiveWork("Local protection")
        guard policy.mode == .experimentalDemo,
              policy.requiredEvidenceTier == .experimentalProxyOnly else {
            throw FACADEProtectionError.evidenceTierRejected(
                "the current S1-B r6 student is experimental-only and cannot enter release mode"
            )
        }
        let decodeStarted = DispatchTime.now().uptimeNanoseconds
        let clean = try codec.decodeAndPrepare(input.originalData)
        stageLatencyMilliseconds["decode_prepare"] = Self.milliseconds(since: decodeStarted)
        guard clean.width == 256, clean.height == 256 else {
            throw FACADEProtectionError.invalidInput("S1-B requires the frozen 256x256 prepared frame")
        }
        let initializerStarted = DispatchTime.now().uptimeNanoseconds
        let allInitializers = try await initializer.residuals(for: clean)
        stageLatencyMilliseconds["initializer_inference"] = Self.milliseconds(
            since: initializerStarted
        )
        guard allInitializers.count == 8 else {
            throw FACADEProtectionError.modelUnavailable("the r6 initializer must expose exactly eight frozen heads")
        }

        let clampLimit = Float(policy.uint8LInfinityMaximum) / 255.0
        var objectiveEvaluationCount = 0
        // One extra evaluation on the untouched clean frame, with the same
        // reference vector the loop uses, anchors the fail-closed sign guard
        // below. The loop itself is unchanged.
        let cleanObjectiveStarted = DispatchTime.now().uptimeNanoseconds
        let cleanEvaluation = try await objective.evaluate(clean: clean, candidate: clean)
        stageLatencyMilliseconds["clean_objective_evaluation"] = Self.milliseconds(
            since: cleanObjectiveStarted
        )
        Self.accumulateObjectiveLatencies(
            cleanEvaluation.stageLatencyMilliseconds,
            into: &stageLatencyMilliseconds
        )
        objectiveEvaluationCount += 1
        try Self.requireSameShape(cleanEvaluation.inputGradient, clean)
        let cleanScore = cleanEvaluation.score
        var best: (
            score: Float,
            image: FACADEFloatImage,
            delta: FACADEFloatImage,
            headIndex: Int,
            evaluationIndex: Int
        )?
        for (headIndex, head) in allInitializers.prefix(policy.headCount).enumerated() {
            try safetyGate.requireSafeForIntensiveWork("Local protection")
            try Self.requireSameShape(head, clean)
            let initialProjectionStarted = DispatchTime.now().uptimeNanoseconds
            var delta = try FACADEDCTLowProjector.project(
                Self.clampedResidual(head, limit: clampLimit)
            )
            delta = Self.clampedResidual(delta, limit: clampLimit)
            Self.accumulateLatency(
                "initial_projection",
                since: initialProjectionStarted,
                into: &stageLatencyMilliseconds
            )

            let totalSteps = policy.searchSteps + policy.qualitySteps
            for step in 0..<totalSteps {
                try Task.checkCancellation()
                try safetyGate.requireSafeForIntensiveWork("Local protection")
                let candidate = try Self.adding(clean, delta: delta)
                let objectiveStarted = DispatchTime.now().uptimeNanoseconds
                let evaluation = try await objective.evaluate(clean: clean, candidate: candidate)
                Self.accumulateLatency(
                    "iterative_objective_evaluations",
                    since: objectiveStarted,
                    into: &stageLatencyMilliseconds
                )
                Self.accumulateObjectiveLatencies(
                    evaluation.stageLatencyMilliseconds,
                    into: &stageLatencyMilliseconds
                )
                objectiveEvaluationCount += 1
                try Self.requireSameShape(evaluation.inputGradient, clean)
                if best == nil || evaluation.score < best!.score {
                    best = (evaluation.score, candidate, delta, headIndex, step)
                }

                let updateStarted = DispatchTime.now().uptimeNanoseconds
                let projectedGradient = try FACADEDCTLowProjector.project(
                    evaluation.inputGradient
                )
                let stepSize = step < policy.searchSteps ? searchStep : qualityStep
                let updated = try optimizerUpdateRule.updatedResidual(
                    residual: delta.values,
                    projectedGradient: projectedGradient.values,
                    stepSize: stepSize,
                    clampLimit: clampLimit
                )
                delta = try FACADEDCTLowProjector.project(
                    FACADEFloatImage(width: clean.width, height: clean.height, values: updated)
                )
                delta = Self.clampedResidual(delta, limit: clampLimit)
                Self.accumulateLatency(
                    "gradient_projection_and_update",
                    since: updateStarted,
                    into: &stageLatencyMilliseconds
                )
            }

            try safetyGate.requireSafeForIntensiveWork("Local protection")
            let finalCandidate = try Self.adding(clean, delta: delta)
            let finalObjectiveStarted = DispatchTime.now().uptimeNanoseconds
            let finalEvaluation = try await objective.evaluate(clean: clean, candidate: finalCandidate)
            Self.accumulateLatency(
                "final_objective_evaluations",
                since: finalObjectiveStarted,
                into: &stageLatencyMilliseconds
            )
            Self.accumulateObjectiveLatencies(
                finalEvaluation.stageLatencyMilliseconds,
                into: &stageLatencyMilliseconds
            )
            objectiveEvaluationCount += 1
            if best == nil || finalEvaluation.score < best!.score {
                best = (
                    finalEvaluation.score,
                    finalCandidate,
                    delta,
                    headIndex,
                    totalSteps
                )
            }
        }

        guard let best else {
            throw FACADEProtectionError.candidateGenerationFailed("S1-B produced no candidate")
        }
        // The engine descends the score, so a correctly signed reference must
        // leave the selected candidate strictly below the clean baseline. A
        // reference-vector sign error would drive the score up instead; that
        // must never be saved as protection.
        guard best.score < cleanScore else {
            throw FACADEProtectionError.candidateGenerationFailed(
                "selected candidate did not reduce the objective below the clean image's score " +
                "(candidate \(best.score), clean \(cleanScore)); check reference-vector sign"
            )
        }
        let scalarDiagnosticsStarted = DispatchTime.now().uptimeNanoseconds
        let stored = Self.storedResidualScalars(
            clean: clean,
            candidate: best.image,
            saturationLimit: policy.uint8LInfinityMaximum
        )
        guard stored.lInfinity <= policy.uint8LInfinityMaximum else {
            throw FACADEProtectionError.candidateGenerationFailed(
                "the stored candidate exceeded the requested uint8 L-infinity budget"
            )
        }
        let optimizerSaturatedCount = best.delta.values.reduce(0) { count, value in
            count + (abs(abs(value) - clampLimit) <= 1e-7 ? 1 : 0)
        }
        stageLatencyMilliseconds["scalar_diagnostics"] = Self.milliseconds(
            since: scalarDiagnosticsStarted
        )
        let encodeStarted = DispatchTime.now().uptimeNanoseconds
        let encoded = try codec.encodePNG(best.image)
        stageLatencyMilliseconds["png_encode"] = Self.milliseconds(since: encodeStarted)
        stageLatencyMilliseconds["total_engine"] = Self.milliseconds(since: engineStarted)
        let candidate = FACADEProtectionCandidate(
            data: encoded,
            uniformTypeIdentifier: "public.png",
            engineIdentifier: engineIdentifier,
            modelBundleIdentifier: modelBundleIdentifier,
            evidenceTier: .experimentalProxyOnly
        )
        let diagnostics = FACADEEngineRunDiagnostics(
            schemaVersion: 1,
            chosenHeadIndex: best.headIndex,
            chosenEvaluationIndex: best.evaluationIndex,
            objectiveEvaluationCount: objectiveEvaluationCount,
            objectiveScore: Double(best.score),
            cleanObjectiveScore: Double(cleanScore),
            requestedUInt8LInfinityMaximum: policy.uint8LInfinityMaximum,
            finalAchievedUInt8LInfinity: stored.lInfinity,
            residualRMSUInt8: stored.rms,
            residualEnergyUInt8Squared: stored.energy,
            changedCoordinateCount: stored.changedCount,
            coordinateCount: stored.coordinateCount,
            changedCoordinateDensity: Double(stored.changedCount) / Double(stored.coordinateCount),
            storedClampSaturationFraction: Double(stored.saturatedCount) /
                Double(stored.coordinateCount),
            optimizerClampSaturationFraction: Double(optimizerSaturatedCount) /
                Double(best.delta.values.count),
            stageLatencyMilliseconds: stageLatencyMilliseconds
        )
        return FACADEInstrumentedProtectionCandidate(
            candidate: candidate,
            diagnostics: diagnostics
        )
    }

    private static func milliseconds(since started: UInt64) -> Double {
        Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000.0
    }

    private static func accumulateLatency(
        _ name: String,
        since started: UInt64,
        into values: inout [String: Double]
    ) {
        values[name, default: 0] += milliseconds(since: started)
    }

    private static func accumulateObjectiveLatencies(
        _ objectiveValues: [String: Double],
        into values: inout [String: Double]
    ) {
        for (name, value) in objectiveValues {
            values["objective_\(name)", default: 0] += value
        }
    }

    private static func storedResidualScalars(
        clean: FACADEFloatImage,
        candidate: FACADEFloatImage,
        saturationLimit: Int
    ) -> (
        lInfinity: Int,
        rms: Double,
        energy: Double,
        changedCount: Int,
        saturatedCount: Int,
        coordinateCount: Int
    ) {
        var lInfinity = 0
        var energy = 0.0
        var changedCount = 0
        var saturatedCount = 0
        for (cleanValue, candidateValue) in zip(clean.values, candidate.values) {
            let cleanByte = min(255, max(0, Int((cleanValue * 255).rounded())))
            let candidateByte = min(255, max(0, Int((candidateValue * 255).rounded())))
            let difference = abs(candidateByte - cleanByte)
            lInfinity = max(lInfinity, difference)
            energy += Double(difference * difference)
            changedCount += difference == 0 ? 0 : 1
            saturatedCount += difference == saturationLimit ? 1 : 0
        }
        let coordinateCount = clean.values.count
        return (
            lInfinity,
            sqrt(energy / Double(coordinateCount)),
            energy,
            changedCount,
            saturatedCount,
            coordinateCount
        )
    }

    private static func requireSameShape(
        _ left: FACADEFloatImage,
        _ right: FACADEFloatImage
    ) throws {
        guard left.width == right.width, left.height == right.height,
              left.values.count == right.values.count else {
            throw FACADEProtectionError.candidateGenerationFailed("S1-B tensor shapes do not match")
        }
    }

    private static func adding(
        _ clean: FACADEFloatImage,
        delta: FACADEFloatImage
    ) throws -> FACADEFloatImage {
        try requireSameShape(clean, delta)
        return try FACADEFloatImage(
            width: clean.width,
            height: clean.height,
            values: zip(clean.values, delta.values).map { min(1, max(0, $0 + $1)) }
        )
    }

    private static func clampedResidual(
        _ residual: FACADEFloatImage,
        limit: Float
    ) -> FACADEFloatImage {
        // Construction is safe because the input shape was already validated.
        try! FACADEFloatImage(
            width: residual.width,
            height: residual.height,
            values: residual.values.map { min(limit, max(-limit, $0)) }
        )
    }
}

public enum FACADEDCTLowProjector {
    /// Exact CPU reference for the S1 8x8 orthonormal block-DCT low band
    /// (u + v <= 3), including the frozen global mean-absolute rescaling.
    public static func project(_ image: FACADEFloatImage) throws -> FACADEFloatImage {
        guard image.width.isMultiple(of: 8), image.height.isMultiple(of: 8) else {
            throw FACADEProtectionError.candidateGenerationFailed(
                "DCT-low requires dimensions divisible by eight"
            )
        }
        let matrix: [[Float]] = (0..<8).map { frequency in
            (0..<8).map { position in
                let normalization = frequency == 0 ? 1.0 / sqrt(8.0) : sqrt(2.0 / 8.0)
                return Float(normalization * cos(Double.pi / 8.0 * (Double(position) + 0.5) * Double(frequency)))
            }
        }
        var reconstructed = [Float](repeating: 0, count: image.values.count)
        let plane = image.width * image.height

        for channel in 0..<3 {
            for blockY in stride(from: 0, to: image.height, by: 8) {
                for blockX in stride(from: 0, to: image.width, by: 8) {
                    var coefficients = [Float](repeating: 0, count: 64)
                    for u in 0..<8 {
                        for v in 0..<8 where u + v <= 3 {
                            var value: Float = 0
                            for x in 0..<8 {
                                for y in 0..<8 {
                                    let index = channel * plane + (blockY + x) * image.width + blockX + y
                                    value += matrix[u][x] * image.values[index] * matrix[v][y]
                                }
                            }
                            coefficients[u * 8 + v] = value
                        }
                    }
                    for x in 0..<8 {
                        for y in 0..<8 {
                            var value: Float = 0
                            for u in 0..<8 {
                                for v in 0..<8 where u + v <= 3 {
                                    value += matrix[u][x] * coefficients[u * 8 + v] * matrix[v][y]
                                }
                            }
                            let index = channel * plane + (blockY + x) * image.width + blockX + y
                            reconstructed[index] = value
                        }
                    }
                }
            }
        }

        let originalMean = image.values.reduce(0) { $0 + abs($1) } / Float(image.values.count)
        let reconstructedMean = reconstructed.reduce(0) { $0 + abs($1) } / Float(reconstructed.count)
        let scale = originalMean / max(reconstructedMean, 1e-12)
        return try FACADEFloatImage(
            width: image.width,
            height: image.height,
            values: reconstructed.map { $0 * scale }
        )
    }
}
