import Foundation

public final class FACADECoreMLLocalVerifier: @unchecked Sendable, FACADELocalVerifier {
    public let verifierIdentifier: String

    private let handoff: FACADELoadedModelHandoff
    private let codec: any FACADELocalVerificationImageCodec
    private let detector: any FACADEExactlyOneFaceDetector
    private let arcFace: FACADECoreMLEmbeddingModel
    private let simSwap: FACADECoreMLEmbeddingModel

    public init(
        verifierIdentifier: String = "facade.local.arcface-simswap.experimental.v1",
        handoff: FACADELoadedModelHandoff,
        codec: any FACADELocalVerificationImageCodec,
        detector: any FACADEExactlyOneFaceDetector,
        arcFace: FACADECoreMLEmbeddingModel,
        simSwap: FACADECoreMLEmbeddingModel
    ) throws {
        guard handoff.manifest.evidence.tier == .experimentalProxyOnly,
              handoff.manifest.localVerification.independentFromProtectionEngine,
              handoff.manifest.localVerification.missingOrUnscorableIsFailure else {
            throw FACADEProtectionError.modelUnavailable(
                "the local verification manifest is not experimental and fail-closed"
            )
        }
        self.verifierIdentifier = verifierIdentifier
        self.handoff = handoff
        self.codec = codec
        self.detector = detector
        self.arcFace = arcFace
        self.simSwap = simSwap
    }

    public func verify(
        original: FACADEProtectionInput,
        candidate: FACADEProtectionCandidate,
        policy: FACADEProtectionPolicy
    ) async throws -> FACADEVerificationReport {
        let verifierStarted = DispatchTime.now().uptimeNanoseconds
        var stageLatencyMilliseconds: [String: Double] = [:]
        guard policy.mode == .experimentalDemo,
              candidate.modelBundleIdentifier == handoff.manifest.bundleIdentifier,
              candidate.evidenceTier == .experimentalProxyOnly else {
            throw FACADEProtectionError.verificationRejected(
                "the candidate does not match the installed experimental verifier bundle"
            )
        }

        let originalDataBefore = original.originalData
        let decodeStarted = DispatchTime.now().uptimeNanoseconds
        let clean = try codec.decodeAndPrepare(original.originalData)
        let rawCandidate = try codec.decodeAndPrepare(candidate.data)
        stageLatencyMilliseconds["decode_prepare"] = Self.milliseconds(since: decodeStarted)
        guard clean.width == 256, clean.height == 256,
              rawCandidate.width == 256, rawCandidate.height == 256 else {
            throw FACADEProtectionError.verificationUnavailable(
                "verification requires the frozen 256x256 prepared frame"
            )
        }

        let referenceStarted = DispatchTime.now().uptimeNanoseconds
        let referenceTransform = try detector.alignmentTransform(in: clean)
        let referenceAligned = try referenceTransform.alignedImage(from: clean)
        let referenceArcFace = try arcFace.embedding(for: referenceAligned)
        let referenceSimSwap = try simSwap.embedding(for: referenceAligned)
        stageLatencyMilliseconds["reference_alignment_and_embeddings"] = Self.milliseconds(
            since: referenceStarted
        )

        let jpegStarted = DispatchTime.now().uptimeNanoseconds
        let jpegCandidate = try codec.jpegQ70RoundTrip(rawCandidate).image
        stageLatencyMilliseconds["jpeg_q70_roundtrip"] = Self.milliseconds(
            since: jpegStarted
        )
        let heicStarted = DispatchTime.now().uptimeNanoseconds
        let heicCandidate = try codec.heicQ80RoundTrip(rawCandidate).image
        stageLatencyMilliseconds["heic_q80_roundtrip"] = Self.milliseconds(
            since: heicStarted
        )
        let transformedCandidates: [(name: String, image: FACADEFloatImage)] = [
            ("raw", rawCandidate),
            ("jpeg_q70", jpegCandidate),
            ("heic_q80", heicCandidate),
        ]

        var scores: [String: Float] = [:]
        for transformed in transformedCandidates {
            let scoringStarted = DispatchTime.now().uptimeNanoseconds
            _ = try detector.exactlyOneFace(in: transformed.image)
            let aligned = try referenceTransform.alignedImage(from: transformed.image)
            scores["arcface_\(transformed.name)"] = Self.cosineSimilarity(
                referenceArcFace,
                try arcFace.embedding(for: aligned)
            )
            scores["simswap_\(transformed.name)"] = Self.cosineSimilarity(
                referenceSimSwap,
                try simSwap.embedding(for: aligned)
            )
            stageLatencyMilliseconds["\(transformed.name)_scoring"] = Self.milliseconds(
                since: scoringStarted
            )
        }

        let decisionStarted = DispatchTime.now().uptimeNanoseconds
        let thresholdByName = Dictionary(
            uniqueKeysWithValues: handoff.manifest.localVerification.thresholds.map {
                ($0.check, $0)
            }
        )
        let arcFacePass = ["raw", "jpeg_q70", "heic_q80"].allSatisfy { transform in
            Self.passes(
                score: scores["arcface_\(transform)"],
                threshold: thresholdByName["arcface_\(transform)"]
            )
        }
        let simSwapPass = ["raw", "jpeg_q70", "heic_q80"].allSatisfy { transform in
            Self.passes(
                score: scores["simswap_\(transform)"],
                threshold: thresholdByName["simswap_\(transform)"]
            )
        }
        let maximumDifference = Self.maximumStoredUInt8Difference(clean, rawCandidate)
        let boundPass = Self.exactUInt8BoundPasses(
            actual: maximumDifference,
            manifestMaximum: handoff.manifest.localVerification.exactUInt8LInfinityMaximum,
            policy: policy
        )

        var checks = Dictionary(
            uniqueKeysWithValues: FACADEVerificationCheck.allCases.map { ($0, FACADECheckStatus.passed) }
        )
        checks[.exactUInt8Bound] = boundPass ? .passed : .failed
        checks[.arcFaceLocal] = arcFacePass ? .passed : .failed
        checks[.simSwapLocal] = simSwapPass ? .passed : .failed
        checks[.originalPreserved] = original.originalData == originalDataBefore ? .passed : .failed
        let accepted = policy.requiredChecks.allSatisfy { checks[$0] == .passed }
        let orderedScores = scores.keys.sorted().map { key in
            let value = String(format: "%.7f", scores[key]!)
            return "\(key)=\(value)"
        }
        stageLatencyMilliseconds["decision_and_uint8_bound"] = Self.milliseconds(
            since: decisionStarted
        )
        stageLatencyMilliseconds["total_verifier"] = Self.milliseconds(
            since: verifierStarted
        )
        return FACADEVerificationReport(
            verifierIdentifier: verifierIdentifier,
            modelBundleIdentifier: handoff.manifest.bundleIdentifier,
            decision: accepted ? .acceptedForSave : .rejected,
            evidenceTier: .experimentalProxyOnly,
            checks: checks,
            notes: [
                "maximum_stored_uint8_difference=\(maximumDifference)",
                "Local embedding scores are experimental proxies, not rendered-attack evidence.",
            ] + orderedScores,
            scalarScores: scores.mapValues(Double.init),
            stageLatencyMilliseconds: stageLatencyMilliseconds
        )
    }

    private static func milliseconds(since started: UInt64) -> Double {
        Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000.0
    }

    private static func maximumStoredUInt8Difference(
        _ original: FACADEFloatImage,
        _ candidate: FACADEFloatImage
    ) -> Int {
        zip(original.values, candidate.values).reduce(0) { maximum, pair in
            let left = min(255, max(0, Int((pair.0 * 255).rounded())))
            let right = min(255, max(0, Int((pair.1 * 255).rounded())))
            return max(maximum, abs(left - right))
        }
    }

    static func exactUInt8BoundPasses(
        actual: Int,
        manifestMaximum: Int,
        policy: FACADEProtectionPolicy
    ) -> Bool {
        actual <= min(manifestMaximum, policy.uint8LInfinityMaximum)
    }

    private static func cosineSimilarity(_ left: [Float], _ right: [Float]) -> Float {
        zip(left, right).reduce(0) { $0 + $1.0 * $1.1 }
    }

    private static func passes(
        score: Float?,
        threshold: FACADEVerificationThreshold?
    ) -> Bool {
        guard let score, score.isFinite, let threshold else { return false }
        switch threshold.comparison {
        case "less_than_or_equal":
            return Double(score) <= threshold.threshold - threshold.calibratedMargin
        case "greater_than_or_equal":
            return Double(score) >= threshold.threshold + threshold.calibratedMargin
        default:
            return false
        }
    }
}
