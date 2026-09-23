import Foundation

public enum FACADEEvidenceTier: String, Codable, Sendable, CaseIterable {
    /// Local proxy verification may run, but rendered-attack efficacy is unconfirmed.
    case experimentalProxyOnly = "experimental_proxy_only"

    /// V0 and the final student independently passed the frozen rendered-attack protocol.
    case renderedAttackValidated = "rendered_attack_validated"
}

public enum FACADEProtectionMode: String, Codable, Sendable {
    case experimentalDemo = "experimental_demo"
    case releaseCandidate = "release_candidate"
}

public enum FACADEVerificationCheck: String, Codable, Sendable, CaseIterable {
    case modelIntegrity = "model_integrity"
    case outputDecodes = "output_decodes"
    case exactlyOneFace = "exactly_one_face"
    case exactUInt8Bound = "exact_uint8_bound"
    case arcFaceLocal = "arcface_local"
    case simSwapLocal = "simswap_local"
    case rawTransform = "raw_transform"
    case jpegQ70Transform = "jpeg_q70_transform"
    case heicQ80Transform = "heic_q80_transform"
    case originalPreserved = "original_preserved"
}

public enum FACADECheckStatus: String, Codable, Sendable {
    case passed
    case failed
    case unavailable
}

public enum FACADEVerificationDecision: String, Codable, Sendable {
    case acceptedForSave = "accepted_for_save"
    case rejected
    case unavailable
}

public struct FACADEProtectionPolicy: Codable, Sendable, Equatable {
    public static let requiredExperimentalLabel =
        "EXPERIMENTAL - visibly filtered; protection efficacy is not confirmed."

    /// Intended appearance-first engineering baseline for the iterative
    /// DCT-low engine. Its scientific freeze is owned by the separate DCT6
    /// baseline task; this value alone is not an efficacy or shipping claim.
    public static let intendedAppearanceFirstDCTLowUInt8LInfinityMaximum = 5

    /// Historical L16 behavior retained only for explicit research
    /// comparisons and compatibility with policies serialized before the
    /// runtime bound was parameterized.
    public static let legacyDCTLowResearchComparatorUInt8LInfinityMaximum = 16

    public let mode: FACADEProtectionMode
    public let requiredChecks: Set<FACADEVerificationCheck>
    public let requiredEvidenceTier: FACADEEvidenceTier
    public let experimentalLabel: String?
    public let headCount: Int
    public let searchSteps: Int
    public let qualitySteps: Int
    /// Exact integer constraint applied to stored uint8 channel values.
    /// The installed model manifest may impose a stricter upper bound.
    public let uint8LInfinityMaximum: Int

    enum CodingKeys: String, CodingKey {
        case mode
        case requiredChecks
        case requiredEvidenceTier
        case experimentalLabel
        case headCount
        case searchSteps
        case qualitySteps
        case uint8LInfinityMaximum
    }

    public init(
        mode: FACADEProtectionMode,
        requiredChecks: Set<FACADEVerificationCheck>,
        requiredEvidenceTier: FACADEEvidenceTier,
        experimentalLabel: String?,
        headCount: Int,
        searchSteps: Int,
        qualitySteps: Int,
        uint8LInfinityMaximum: Int =
            FACADEProtectionPolicy.intendedAppearanceFirstDCTLowUInt8LInfinityMaximum
    ) throws {
        guard [1, 3, 8].contains(headCount) else {
            throw FACADEProtectionError.invalidPolicy("headCount must be one of 1, 3, or 8")
        }
        guard searchSteps > 0, qualitySteps >= 0 else {
            throw FACADEProtectionError.invalidPolicy("step budgets must be non-negative and searchSteps must be positive")
        }
        guard (1...Self.legacyDCTLowResearchComparatorUInt8LInfinityMaximum)
            .contains(uint8LInfinityMaximum) else {
            throw FACADEProtectionError.invalidPolicy(
                "uint8LInfinityMaximum must be between 1 and 16 inclusive"
            )
        }
        if mode == .experimentalDemo,
           experimentalLabel != Self.requiredExperimentalLabel {
            throw FACADEProtectionError.invalidPolicy("the frozen experimental label is required")
        }
        if mode == .releaseCandidate,
           requiredEvidenceTier != .renderedAttackValidated {
            throw FACADEProtectionError.invalidPolicy("releaseCandidate requires rendered-attack-validated evidence")
        }

        self.mode = mode
        self.requiredChecks = requiredChecks
        self.requiredEvidenceTier = requiredEvidenceTier
        self.experimentalLabel = experimentalLabel
        self.headCount = headCount
        self.searchSteps = searchSteps
        self.qualitySteps = qualitySteps
        self.uint8LInfinityMaximum = uint8LInfinityMaximum
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let uint8LInfinityMaximum: Int
        if container.contains(.uint8LInfinityMaximum) {
            uint8LInfinityMaximum = try container.decode(
                Int.self,
                forKey: .uint8LInfinityMaximum
            )
        } else {
            // Missing means the pre-parameterization L16 behavior. Do not
            // reinterpret historical policies as the new DCT6 default.
            uint8LInfinityMaximum =
                Self.legacyDCTLowResearchComparatorUInt8LInfinityMaximum
        }
        try self.init(
            mode: container.decode(FACADEProtectionMode.self, forKey: .mode),
            requiredChecks: container.decode(
                Set<FACADEVerificationCheck>.self,
                forKey: .requiredChecks
            ),
            requiredEvidenceTier: container.decode(
                FACADEEvidenceTier.self,
                forKey: .requiredEvidenceTier
            ),
            experimentalLabel: container.decodeIfPresent(
                String.self,
                forKey: .experimentalLabel
            ),
            headCount: container.decode(Int.self, forKey: .headCount),
            searchSteps: container.decode(Int.self, forKey: .searchSteps),
            qualitySteps: container.decode(Int.self, forKey: .qualitySteps),
            uint8LInfinityMaximum: uint8LInfinityMaximum
        )
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(mode, forKey: .mode)
        try container.encode(requiredChecks, forKey: .requiredChecks)
        try container.encode(requiredEvidenceTier, forKey: .requiredEvidenceTier)
        try container.encodeIfPresent(experimentalLabel, forKey: .experimentalLabel)
        try container.encode(headCount, forKey: .headCount)
        try container.encode(searchSteps, forKey: .searchSteps)
        try container.encode(qualitySteps, forKey: .qualitySteps)
        try container.encode(uint8LInfinityMaximum, forKey: .uint8LInfinityMaximum)
    }

    public static func experimentalDemo(
        headCount: Int,
        searchSteps: Int,
        qualitySteps: Int,
        uint8LInfinityMaximum: Int =
            FACADEProtectionPolicy.intendedAppearanceFirstDCTLowUInt8LInfinityMaximum
    ) throws -> Self {
        try Self(
            mode: .experimentalDemo,
            requiredChecks: Set(FACADEVerificationCheck.allCases),
            requiredEvidenceTier: .experimentalProxyOnly,
            experimentalLabel: requiredExperimentalLabel,
            headCount: headCount,
            searchSteps: searchSteps,
            qualitySteps: qualitySteps,
            uint8LInfinityMaximum: uint8LInfinityMaximum
        )
    }

    /// Explicit source-level opt-in for reproducing the historical DCT-low
    /// L16 research comparator. This does not promote it to release mode or
    /// attach any efficacy claim.
    public static func experimentalDCTLowL16ResearchComparator(
        headCount: Int,
        searchSteps: Int,
        qualitySteps: Int
    ) throws -> Self {
        try experimentalDemo(
            headCount: headCount,
            searchSteps: searchSteps,
            qualitySteps: qualitySteps,
            uint8LInfinityMaximum:
                legacyDCTLowResearchComparatorUInt8LInfinityMaximum
        )
    }
}

public struct FACADEProtectionInput: Sendable, Equatable {
    public enum Source: Sendable, Equatable {
        /// The source asset is only read. The package has no API for editing or deleting it.
        case imported(readOnlyAssetIdentifier: String)

        /// A camera capture must first be saved as an original asset before processing begins.
        case captured(preservedOriginalAssetIdentifier: String)
    }

    public let id: UUID
    public let originalData: Data
    public let uniformTypeIdentifier: String
    public let source: Source

    public init(
        id: UUID = UUID(),
        originalData: Data,
        uniformTypeIdentifier: String,
        source: Source
    ) throws {
        guard !originalData.isEmpty else {
            throw FACADEProtectionError.invalidInput("the original image is empty")
        }
        guard !uniformTypeIdentifier.isEmpty else {
            throw FACADEProtectionError.invalidInput("the image type is missing")
        }
        switch source {
        case .imported(let identifier), .captured(let identifier):
            guard !identifier.isEmpty else {
                throw FACADEProtectionError.invalidInput("an immutable original asset identifier is required")
            }
        }

        self.id = id
        self.originalData = originalData
        self.uniformTypeIdentifier = uniformTypeIdentifier
        self.source = source
    }
}

public struct FACADEProtectionCandidate: Sendable, Equatable {
    public let data: Data
    public let uniformTypeIdentifier: String
    public let engineIdentifier: String
    public let modelBundleIdentifier: String
    public let evidenceTier: FACADEEvidenceTier

    public init(
        data: Data,
        uniformTypeIdentifier: String,
        engineIdentifier: String,
        modelBundleIdentifier: String,
        evidenceTier: FACADEEvidenceTier
    ) {
        self.data = data
        self.uniformTypeIdentifier = uniformTypeIdentifier
        self.engineIdentifier = engineIdentifier
        self.modelBundleIdentifier = modelBundleIdentifier
        self.evidenceTier = evidenceTier
    }
}

public struct FACADEVerificationReport: Codable, Sendable, Equatable {
    public let verifierIdentifier: String
    public let modelBundleIdentifier: String
    public let decision: FACADEVerificationDecision
    public let evidenceTier: FACADEEvidenceTier
    public let checks: [FACADEVerificationCheck: FACADECheckStatus]
    public let notes: [String]
    /// Structured scalar proxy scores. No image pixels, crops, embeddings, or
    /// other personal-device content may be stored here.
    public let scalarScores: [String: Double]?
    public let stageLatencyMilliseconds: [String: Double]?

    public init(
        verifierIdentifier: String,
        modelBundleIdentifier: String,
        decision: FACADEVerificationDecision,
        evidenceTier: FACADEEvidenceTier,
        checks: [FACADEVerificationCheck: FACADECheckStatus],
        notes: [String] = [],
        scalarScores: [String: Double]? = nil,
        stageLatencyMilliseconds: [String: Double]? = nil
    ) {
        self.verifierIdentifier = verifierIdentifier
        self.modelBundleIdentifier = modelBundleIdentifier
        self.decision = decision
        self.evidenceTier = evidenceTier
        self.checks = checks
        self.notes = notes
        self.scalarScores = scalarScores
        self.stageLatencyMilliseconds = stageLatencyMilliseconds
    }
}

/// This value has no public initializer. It can only be issued by a successful
/// `FACADEProtectionWorkflow` verification transaction.
public struct FACADEVerifiedProtectedCopy: Sendable, Equatable {
    public let transactionIdentifier: UUID
    public let sourceIdentifier: UUID
    public let data: Data
    public let uniformTypeIdentifier: String
    public let originalSHA256: String
    public let protectedSHA256: String
    public let report: FACADEVerificationReport
    public let experimentalLabel: String?

    init(
        transactionIdentifier: UUID,
        sourceIdentifier: UUID,
        data: Data,
        uniformTypeIdentifier: String,
        originalSHA256: String,
        protectedSHA256: String,
        report: FACADEVerificationReport,
        experimentalLabel: String?
    ) {
        self.transactionIdentifier = transactionIdentifier
        self.sourceIdentifier = sourceIdentifier
        self.data = data
        self.uniformTypeIdentifier = uniformTypeIdentifier
        self.originalSHA256 = originalSHA256
        self.protectedSHA256 = protectedSHA256
        self.report = report
        self.experimentalLabel = experimentalLabel
    }
}

public struct FACADESaveReceipt: Codable, Sendable, Equatable {
    public let transactionIdentifier: UUID
    public let newAssetIdentifier: String
    public let protectedSHA256: String
    public let originalWasModified: Bool

    public init(
        transactionIdentifier: UUID,
        newAssetIdentifier: String,
        protectedSHA256: String,
        originalWasModified: Bool
    ) {
        self.transactionIdentifier = transactionIdentifier
        self.newAssetIdentifier = newAssetIdentifier
        self.protectedSHA256 = protectedSHA256
        self.originalWasModified = originalWasModified
    }
}

public enum FACADEProtectionError: Error, Sendable, Equatable {
    case invalidInput(String)
    case invalidPolicy(String)
    case modelUnavailable(String)
    case candidateGenerationFailed(String)
    case candidateInvalid(String)
    case verificationUnavailable(String)
    case verificationRejected(String)
    case evidenceTierRejected(String)
    case deviceSafetyBlocked(String)
    case originalMutationDetected
    case saveFailed(String)
}

extension FACADEProtectionError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .invalidInput(let message),
             .invalidPolicy(let message),
             .modelUnavailable(let message),
             .candidateGenerationFailed(let message),
             .candidateInvalid(let message),
             .verificationUnavailable(let message),
             .verificationRejected(let message),
             .evidenceTierRejected(let message),
             .deviceSafetyBlocked(let message),
             .saveFailed(let message):
            return message
        case .originalMutationDetected:
            return "The original image changed during processing; no copy was issued."
        }
    }
}
