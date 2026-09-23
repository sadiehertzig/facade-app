import CryptoKit
import Foundation

public enum FACADEModelRole: String, Codable, Sendable, CaseIterable {
    case protectionEngine = "protection_engine"
    case arcFaceForwardVerifier = "arcface_forward_verifier"
    case simSwapForwardVerifier = "simswap_forward_verifier"
    case arcFaceScoreGradient = "arcface_score_gradient"
}

public enum FACADEArtifactFormat: String, Codable, Sendable {
    case coreMLPackage = "coreml_mlpackage"
    case compiledCoreML = "coreml_mlmodelc"
    case mpsGraphPackage = "mpsgraphpackage"
}

public enum FACADEArtifactPrecision: String, Codable, Sendable {
    case int8
    case float16
    case float32
    case mixed
}

public enum FACADEGradientSemantics: String, Codable, Sendable {
    case none
    case runtimeAutomaticDifferentiation = "runtime_automatic_differentiation"
    case staticGradientOutput = "static_gradient_output"
}

public enum FACADEComputeUnit: String, Codable, Sendable {
    case cpu
    case gpu
    case ane
}

public enum FACADEEvidenceStatus: String, Codable, Sendable {
    case pending
    case passed
    case failed
}

public struct FACADETensorContract: Codable, Sendable, Equatable {
    public let name: String
    public let shape: [Int]
    public let dataType: String
    public let layout: String
    public let valueRange: [Double]

    public init(
        name: String,
        shape: [Int],
        dataType: String,
        layout: String,
        valueRange: [Double]
    ) {
        self.name = name
        self.shape = shape
        self.dataType = dataType
        self.layout = layout
        self.valueRange = valueRange
    }

    enum CodingKeys: String, CodingKey {
        case name
        case shape
        case dataType = "data_type"
        case layout
        case valueRange = "value_range"
    }
}

public struct FACADEModelArtifact: Codable, Sendable, Equatable {
    public let role: FACADEModelRole
    public let relativePath: String
    public let sha256: String
    public let sha256Kind: String
    public let format: FACADEArtifactFormat
    public let modelFamily: String
    public let sourceWeightsSHA256: String
    public let storagePrecision: FACADEArtifactPrecision
    public let executionPrecision: FACADEArtifactPrecision
    public let gradientSemantics: FACADEGradientSemantics
    public let computeUnits: [FACADEComputeUnit]
    public let inputs: [FACADETensorContract]
    public let outputs: [FACADETensorContract]
    public let quantization: FACADEQuantizationContract
    public let modelCardPath: String

    enum CodingKeys: String, CodingKey {
        case role
        case relativePath = "relative_path"
        case sha256
        case sha256Kind = "sha256_kind"
        case format
        case modelFamily = "model_family"
        case sourceWeightsSHA256 = "source_weights_sha256"
        case storagePrecision = "storage_precision"
        case executionPrecision = "execution_precision"
        case gradientSemantics = "gradient_semantics"
        case computeUnits = "compute_units"
        case inputs
        case outputs
        case quantization
        case modelCardPath = "model_card_path"
    }
}

public struct FACADEQuantizationContract: Codable, Sendable, Equatable {
    public let scheme: String
    public let granularity: String
    public let calibrationManifestSHA256: String

    enum CodingKeys: String, CodingKey {
        case scheme
        case granularity
        case calibrationManifestSHA256 = "calibration_manifest_sha256"
    }
}

public struct FACADEEvidenceReference: Codable, Sendable, Equatable {
    public let status: FACADEEvidenceStatus
    public let artifactIdentifier: String?
    public let sha256: String?

    enum CodingKeys: String, CodingKey {
        case status
        case artifactIdentifier = "artifact_identifier"
        case sha256
    }
}

public struct FACADEHandoffEvidence: Codable, Sendable, Equatable {
    public let tier: FACADEEvidenceTier
    public let v0RenderedSwap: FACADEEvidenceReference
    public let studentRenderedSwap: FACADEEvidenceReference
    public let syntheticAdultsOnly: Bool
    public let developmentIdentities: String
    public let reserveSealedIdentities: String
    public let consumerAppearanceDecision: String
    public let proxyScoresAreNotEfficacyClaims: Bool

    enum CodingKeys: String, CodingKey {
        case tier
        case v0RenderedSwap = "v0_rendered_swap"
        case studentRenderedSwap = "student_rendered_swap"
        case syntheticAdultsOnly = "synthetic_adults_only"
        case developmentIdentities = "development_identities"
        case reserveSealedIdentities = "reserve_sealed_identities"
        case consumerAppearanceDecision = "consumer_appearance_decision"
        case proxyScoresAreNotEfficacyClaims = "proxy_scores_are_not_efficacy_claims"
    }
}

public struct FACADEDataPolicy: Codable, Sendable, Equatable {
    public let personalPhotosUsed: Bool
    public let networkRequired: Bool
    public let trainingScope: String
    public let modelProvenanceDocument: String

    enum CodingKeys: String, CodingKey {
        case personalPhotosUsed = "personal_photos_used"
        case networkRequired = "network_required"
        case trainingScope = "training_scope"
        case modelProvenanceDocument = "model_provenance_document"
    }
}

public struct FACADEVerificationThreshold: Codable, Sendable, Equatable {
    public let check: String
    public let comparison: String
    public let threshold: Double
    public let calibratedMargin: Double

    enum CodingKeys: String, CodingKey {
        case check
        case comparison
        case threshold
        case calibratedMargin = "calibrated_margin"
    }
}

public struct FACADELocalVerificationContract: Codable, Sendable, Equatable {
    public let facePolicy: String
    public let requiredTransforms: [String]
    public let exactUInt8LInfinityMaximum: Int
    public let thresholds: [FACADEVerificationThreshold]
    public let missingOrUnscorableIsFailure: Bool
    public let thresholdProtocolSHA256: String
    public let independentFromProtectionEngine: Bool

    enum CodingKeys: String, CodingKey {
        case facePolicy = "face_policy"
        case requiredTransforms = "required_transforms"
        case exactUInt8LInfinityMaximum = "exact_uint8_linf_maximum"
        case thresholds
        case missingOrUnscorableIsFailure = "missing_or_unscorable_is_failure"
        case thresholdProtocolSHA256 = "threshold_protocol_sha256"
        case independentFromProtectionEngine = "independent_from_protection_engine"
    }
}

public struct FACADEPreprocessingContract: Codable, Sendable, Equatable {
    public let colorSpace: String
    public let orientationPolicy: String
    public let cropSize: [Int]
    public let landmarkOrder: [String]
    public let channelOrder: String
    public let normalization: String
    public let resizeKernel: String
    public let pixelRounding: String

    enum CodingKeys: String, CodingKey {
        case colorSpace = "color_space"
        case orientationPolicy = "orientation_policy"
        case cropSize = "crop_size"
        case landmarkOrder = "landmark_order"
        case channelOrder = "channel_order"
        case normalization
        case resizeKernel = "resize_kernel"
        case pixelRounding = "pixel_rounding"
    }
}

public struct FACADEFaceDetectionAlignment: Codable, Sendable, Equatable {
    public let runtime: String
    public let requestRevision: Int
    public let minimumFaceFraction: Double
    public let coordinateConvention: String
    public let faceSelectionPolicy: String
    public let failurePolicy: String

    enum CodingKeys: String, CodingKey {
        case runtime
        case requestRevision = "request_revision"
        case minimumFaceFraction = "minimum_face_fraction"
        case coordinateConvention = "coordinate_convention"
        case faceSelectionPolicy = "face_selection_policy"
        case failurePolicy = "failure_policy"
    }
}

public struct FACADEQuantizedParity: Codable, Sendable, Equatable {
    public let protocolSHA256: String
    public let fixtureManifestSHA256: String
    public let floatVsQuantizedMetrics: [String: Double]
    public let unchangedJudgeResult: String
    public let visualReviewResult: String

    enum CodingKeys: String, CodingKey {
        case protocolSHA256 = "protocol_sha256"
        case fixtureManifestSHA256 = "fixture_manifest_sha256"
        case floatVsQuantizedMetrics = "float_vs_quantized_metrics"
        case unchangedJudgeResult = "unchanged_judge_result"
        case visualReviewResult = "visual_review_result"
    }
}

public struct FACADEBenchmarkFixture: Codable, Sendable, Equatable {
    public let relativePath: String
    public let sha256: String
    public let syntheticAdult: Bool
    public let expectedInputSHA256: String

    enum CodingKeys: String, CodingKey {
        case relativePath = "relative_path"
        case sha256
        case syntheticAdult = "synthetic_adult"
        case expectedInputSHA256 = "expected_input_sha256"
    }
}

public struct FACADEModelProvenance: Codable, Sendable, Equatable {
    public let researchProtocolSHA256: String
    public let sourceCheckpointSHA256: String
    public let exportScriptSHA256: String
    public let licenses: [String]
    public let generatedBy: String

    enum CodingKeys: String, CodingKey {
        case researchProtocolSHA256 = "research_protocol_sha256"
        case sourceCheckpointSHA256 = "source_checkpoint_sha256"
        case exportScriptSHA256 = "export_script_sha256"
        case licenses
        case generatedBy = "generated_by"
    }
}

public enum FACADEAdapterResourceRole: String, Codable, Sendable, CaseIterable {
    /// E E^T built from InSwapper's 512x512 `buff2fs` identity-map
    /// initializer. The gradient graph's runtime reference is
    /// v = normalize(E E^T r_hat), not r_hat itself.
    case inSwapperIdentityMapEET = "inswapper_identity_map_eet"
}

/// Non-model bytes the runtime needs beside the Core ML artifacts. They are
/// path-checked, size-checked, and hashed like artifacts but are deliberately
/// not `FACADEModelArtifact`s so the fixed model-role set stays fixed.
public struct FACADEAdapterResource: Codable, Sendable, Equatable {
    public static let identityMapEETDimension = 512
    public static let identityMapEETByteCount = 512 * 512 * 4
    public static let identityMapEETFormat = "row-major-little-endian-float32"

    public let role: FACADEAdapterResourceRole
    public let relativePath: String
    public let sha256: String
    public let byteCount: Int
    public let shape: [Int]
    public let format: String
    public let sourceModelSHA256: String
    public let sourceInitializer: String

    enum CodingKeys: String, CodingKey {
        case role
        case relativePath = "relative_path"
        case sha256
        case byteCount = "byte_count"
        case shape
        case format
        case sourceModelSHA256 = "source_model_sha256"
        case sourceInitializer = "source_initializer"
    }
}

public struct FACADEModelHandoffManifest: Codable, Sendable, Equatable {
    public let schemaVersion: Int
    public let bundleIdentifier: String
    public let bundleVersion: String
    public let createdAt: String
    public let minimumIOSVersion: String
    public let coremltoolsVersion: String
    public let xcodeVersion: String
    public let swiftPackageAPIVersion: String
    public let evidence: FACADEHandoffEvidence
    public let dataPolicy: FACADEDataPolicy
    public let preprocessing: FACADEPreprocessingContract
    public let faceDetectionAlignment: FACADEFaceDetectionAlignment
    public let localVerification: FACADELocalVerificationContract
    public let artifacts: [FACADEModelArtifact]
    public let adapterResources: [FACADEAdapterResource]
    public let quantizedParity: FACADEQuantizedParity
    public let benchmarkFixture: FACADEBenchmarkFixture
    public let provenance: FACADEModelProvenance

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case bundleIdentifier = "bundle_identifier"
        case bundleVersion = "bundle_version"
        case createdAt = "created_at"
        case minimumIOSVersion = "minimum_ios_version"
        case coremltoolsVersion = "coremltools_version"
        case xcodeVersion = "xcode_version"
        case swiftPackageAPIVersion = "swift_package_api_version"
        case evidence
        case dataPolicy = "data_policy"
        case preprocessing
        case faceDetectionAlignment = "face_detection_alignment"
        case localVerification = "local_verification"
        case artifacts
        case adapterResources = "adapter_resources"
        case quantizedParity = "quantized_parity"
        case benchmarkFixture = "benchmark_fixture"
        case provenance
    }
}

public struct FACADELoadedModelHandoff: Sendable {
    public let rootURL: URL
    public let manifest: FACADEModelHandoffManifest
    public let manifestSHA256: String
    public let artifactURLs: [FACADEModelRole: URL]
    public let adapterResourceURLs: [FACADEAdapterResourceRole: URL]
}

/// The 512x512 E E^T identity-map matrix, re-read and re-hashed at load so a
/// file swapped after manifest validation cannot reach the objective.
public struct FACADEIdentityMapMatrix: Sendable, Equatable {
    public static let dimension = FACADEAdapterResource.identityMapEETDimension

    public let sha256: String
    /// Row-major [512][512] float32 values. The matrix is symmetric, so the
    /// row/column orientation is immaterial.
    public let values: [Float]

    public init(sha256: String, values: [Float]) throws {
        guard values.count == Self.dimension * Self.dimension,
              values.allSatisfy(\.isFinite) else {
            throw FACADEProtectionError.modelUnavailable(
                "the identity-map matrix must be 512x512 finite float32 values"
            )
        }
        self.sha256 = sha256
        self.values = values
    }

    public static func load(from handoff: FACADELoadedModelHandoff) throws -> FACADEIdentityMapMatrix {
        guard let resource = handoff.manifest.adapterResources.first(
                  where: { $0.role == .inSwapperIdentityMapEET }
              ),
              let url = handoff.adapterResourceURLs[.inSwapperIdentityMapEET] else {
            throw FACADEProtectionError.modelUnavailable(
                "the handoff does not resolve the InSwapper identity-map resource"
            )
        }
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            throw FACADEProtectionError.modelUnavailable(
                "the InSwapper identity-map resource is unreadable: \(error)"
            )
        }
        guard data.count == resource.byteCount,
              data.count == FACADEAdapterResource.identityMapEETByteCount else {
            throw FACADEProtectionError.modelUnavailable(
                "the InSwapper identity-map resource has \(data.count) bytes, expected " +
                "\(FACADEAdapterResource.identityMapEETByteCount)"
            )
        }
        let actualHash = SHA256.hash(data: data)
            .map { String(format: "%02x", $0) }
            .joined()
        guard actualHash == resource.sha256 else {
            throw FACADEProtectionError.modelUnavailable(
                "identity-map resource hash mismatch: \(actualHash) != \(resource.sha256)"
            )
        }
        let count = Self.dimension * Self.dimension
        var values = [Float](repeating: 0, count: count)
        data.withUnsafeBytes { raw in
            for index in 0..<count {
                let bits = raw.loadUnaligned(fromByteOffset: index * 4, as: UInt32.self)
                values[index] = Float(bitPattern: UInt32(littleEndian: bits))
            }
        }
        return try FACADEIdentityMapMatrix(sha256: actualHash, values: values)
    }

    /// v = normalize(E E^T r_hat) for the L2-normalized forward embedding
    /// r_hat. The caller passes +v to the gradient graph unchanged; the
    /// engine descends the resulting score.
    public func projectedReference(_ rHat: [Float]) throws -> [Float] {
        guard rHat.count == Self.dimension, rHat.allSatisfy(\.isFinite) else {
            throw FACADEProtectionError.candidateGenerationFailed(
                "the reference embedding must be 512 finite values"
            )
        }
        var projected = [Float](repeating: 0, count: Self.dimension)
        for row in 0..<Self.dimension {
            let base = row * Self.dimension
            var sum: Float = 0
            for column in 0..<Self.dimension {
                sum += values[base + column] * rHat[column]
            }
            projected[row] = sum
        }
        let norm = sqrt(projected.reduce(0) { $0 + $1 * $1 })
        guard norm.isFinite, norm > 0, projected.allSatisfy(\.isFinite) else {
            throw FACADEProtectionError.candidateGenerationFailed(
                "the projected reference vector has zero or non-finite norm"
            )
        }
        return projected.map { $0 / norm }
    }
}

public enum FACADEModelHandoffLoader {
    public static func load(
        rootURL: URL,
        mode: FACADEProtectionMode
    ) throws -> FACADELoadedModelHandoff {
        let manifestURL = rootURL.appendingPathComponent("manifest.json", isDirectory: false)
        let data: Data
        do {
            data = try Data(contentsOf: manifestURL, options: [.mappedIfSafe])
        } catch {
            throw FACADEProtectionError.modelUnavailable("manifest.json is unreadable: \(error)")
        }

        try validateJSONEnvelope(data)

        let manifest: FACADEModelHandoffManifest
        do {
            manifest = try JSONDecoder().decode(FACADEModelHandoffManifest.self, from: data)
        } catch {
            throw FACADEProtectionError.modelUnavailable("manifest.json is invalid: \(error)")
        }

        try validate(manifest, mode: mode)

        #if os(iOS)
        guard runtimeMeetsMinimumIOS(manifest.minimumIOSVersion) else {
            throw FACADEProtectionError.modelUnavailable(
                "this model bundle requires iOS \(manifest.minimumIOSVersion) or later"
            )
        }
        #endif

        var artifactURLs: [FACADEModelRole: URL] = [:]
        for artifact in manifest.artifacts {
            let url = try safeArtifactURL(rootURL: rootURL, relativePath: artifact.relativePath)
            let actualHash = try deterministicSHA256(of: url)
            guard actualHash == artifact.sha256 else {
                throw FACADEProtectionError.modelUnavailable(
                    "artifact hash mismatch for \(artifact.role.rawValue): " +
                    "\(actualHash) != \(artifact.sha256)"
                )
            }
            guard artifactURLs.updateValue(url, forKey: artifact.role) == nil else {
                throw FACADEProtectionError.modelUnavailable(
                    "duplicate artifact role \(artifact.role.rawValue)"
                )
            }
        }

        var adapterResourceURLs: [FACADEAdapterResourceRole: URL] = [:]
        for resource in manifest.adapterResources {
            let url = try safeArtifactURL(rootURL: rootURL, relativePath: resource.relativePath)
            let values: URLResourceValues
            do {
                values = try url.resourceValues(
                    forKeys: [.isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey]
                )
            } catch {
                throw FACADEProtectionError.modelUnavailable(
                    "adapter resource \(resource.role.rawValue) is missing: \(error)"
                )
            }
            guard values.isRegularFile == true, values.isSymbolicLink != true,
                  values.fileSize == resource.byteCount else {
                throw FACADEProtectionError.modelUnavailable(
                    "adapter resource \(resource.role.rawValue) is not a regular file of " +
                    "\(resource.byteCount) bytes"
                )
            }
            let actualHash = try deterministicSHA256(of: url)
            guard actualHash == resource.sha256 else {
                throw FACADEProtectionError.modelUnavailable(
                    "adapter resource hash mismatch for \(resource.role.rawValue): " +
                    "\(actualHash) != \(resource.sha256)"
                )
            }
            guard adapterResourceURLs.updateValue(url, forKey: resource.role) == nil else {
                throw FACADEProtectionError.modelUnavailable(
                    "duplicate adapter resource role \(resource.role.rawValue)"
                )
            }
        }

        let fixtureURL = try safeArtifactURL(
            rootURL: rootURL,
            relativePath: manifest.benchmarkFixture.relativePath
        )
        guard try deterministicSHA256(of: fixtureURL) == manifest.benchmarkFixture.sha256 else {
            throw FACADEProtectionError.modelUnavailable("benchmark fixture hash mismatch")
        }

        let documentationPaths = [manifest.dataPolicy.modelProvenanceDocument]
            + manifest.artifacts.map(\.modelCardPath)
            + manifest.provenance.licenses
        for relativePath in documentationPaths {
            let url = try safeArtifactURL(rootURL: rootURL, relativePath: relativePath)
            let values = try url.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey])
            guard values.isRegularFile == true, values.isSymbolicLink != true else {
                throw FACADEProtectionError.modelUnavailable(
                    "required handoff document is missing or unsafe: \(relativePath)"
                )
            }
        }

        let requiredRoles = Set(FACADEModelRole.allCases)
        guard Set(artifactURLs.keys) == requiredRoles else {
            throw FACADEProtectionError.modelUnavailable("the handoff does not contain every required model role")
        }

        return FACADELoadedModelHandoff(
            rootURL: rootURL,
            manifest: manifest,
            manifestSHA256: SHA256.hash(data: data)
                .map { String(format: "%02x", $0) }
                .joined(),
            artifactURLs: artifactURLs,
            adapterResourceURLs: adapterResourceURLs
        )
    }

    private static func validate(
        _ manifest: FACADEModelHandoffManifest,
        mode: FACADEProtectionMode
    ) throws {
        guard manifest.schemaVersion == 1 else {
            throw FACADEProtectionError.modelUnavailable("unsupported handoff schema version")
        }
        guard !manifest.bundleIdentifier.isEmpty,
              manifest.bundleIdentifier.utf8.allSatisfy({ byte in
                  (48...57).contains(byte) || (65...90).contains(byte) ||
                  (97...122).contains(byte) || byte == 45 || byte == 46
              }),
              !manifest.bundleVersion.isEmpty,
              isISO8601Date(manifest.createdAt),
              isMajorMinorVersion(manifest.minimumIOSVersion),
              !manifest.coremltoolsVersion.isEmpty,
              !manifest.xcodeVersion.isEmpty,
              manifest.swiftPackageAPIVersion == "1.0.0" else {
            throw FACADEProtectionError.modelUnavailable("bundle identity is incomplete")
        }
        guard manifest.evidence.syntheticAdultsOnly,
              manifest.evidence.proxyScoresAreNotEfficacyClaims else {
            throw FACADEProtectionError.modelUnavailable("the initial handoff must remain synthetic-adult-only")
        }
        guard manifest.evidence.developmentIdentities == "0240-0271:evaluation-only" else {
            throw FACADEProtectionError.modelUnavailable("development identity policy does not match the frozen boundary")
        }
        guard manifest.evidence.reserveSealedIdentities == "0272-0303:closed" else {
            throw FACADEProtectionError.modelUnavailable("reserve/sealed identity policy does not match the frozen boundary")
        }
        guard ["no_go", "pass"].contains(manifest.evidence.consumerAppearanceDecision),
              evidenceReferenceIsValid(manifest.evidence.v0RenderedSwap),
              evidenceReferenceIsValid(manifest.evidence.studentRenderedSwap) else {
            throw FACADEProtectionError.modelUnavailable("rendered-attack evidence metadata is invalid")
        }
        guard !manifest.dataPolicy.personalPhotosUsed,
              !manifest.dataPolicy.networkRequired,
              manifest.dataPolicy.trainingScope == "synthetic_adults_only",
              isSafeRelativePath(manifest.dataPolicy.modelProvenanceDocument) else {
            throw FACADEProtectionError.modelUnavailable("the handoff violates the offline synthetic-only data policy")
        }
        guard manifest.preprocessing.colorSpace == "sRGB",
              manifest.preprocessing.orientationPolicy == "apply_exif_then_strip_orientation",
              manifest.preprocessing.cropSize.count == 2,
              manifest.preprocessing.cropSize.allSatisfy({ $0 > 0 }),
              manifest.preprocessing.landmarkOrder == [
                  "left_eye", "right_eye", "nose", "left_mouth", "right_mouth",
              ],
              ["RGB", "BGR"].contains(manifest.preprocessing.channelOrder),
              !manifest.preprocessing.normalization.isEmpty,
              !manifest.preprocessing.resizeKernel.isEmpty,
              !manifest.preprocessing.pixelRounding.isEmpty else {
            throw FACADEProtectionError.modelUnavailable("preprocessing metadata is incomplete or incompatible")
        }
        guard manifest.faceDetectionAlignment.runtime == "Apple Vision",
              manifest.faceDetectionAlignment.requestRevision >= 1,
              manifest.faceDetectionAlignment.minimumFaceFraction > 0,
              manifest.faceDetectionAlignment.minimumFaceFraction <= 1,
              !manifest.faceDetectionAlignment.coordinateConvention.isEmpty,
              manifest.faceDetectionAlignment.faceSelectionPolicy == "exactly_one_face",
              manifest.faceDetectionAlignment.failurePolicy == "fail_closed_no_output" else {
            throw FACADEProtectionError.modelUnavailable("face detection/alignment is not fail-closed")
        }
        guard manifest.localVerification.facePolicy == "exactly_one_face",
              manifest.localVerification.missingOrUnscorableIsFailure,
              manifest.localVerification.independentFromProtectionEngine,
              manifest.localVerification.requiredTransforms == ["raw", "jpeg_q70", "heic_q80"],
              (0...16).contains(manifest.localVerification.exactUInt8LInfinityMaximum),
              isNonPlaceholderSHA256(manifest.localVerification.thresholdProtocolSHA256),
              verificationThresholdsAreValid(manifest.localVerification.thresholds) else {
            throw FACADEProtectionError.modelUnavailable("local verification is not fail-closed")
        }

        guard isNonPlaceholderSHA256(manifest.quantizedParity.protocolSHA256),
              isNonPlaceholderSHA256(manifest.quantizedParity.fixtureManifestSHA256),
              ["pending", "pass", "fail"].contains(manifest.quantizedParity.unchangedJudgeResult),
              ["pending", "pass", "fail"].contains(manifest.quantizedParity.visualReviewResult),
              manifest.benchmarkFixture.syntheticAdult,
              isSafeRelativePath(manifest.benchmarkFixture.relativePath),
              isNonPlaceholderSHA256(manifest.benchmarkFixture.sha256),
              isNonPlaceholderSHA256(manifest.benchmarkFixture.expectedInputSHA256),
              isNonPlaceholderSHA256(manifest.provenance.researchProtocolSHA256),
              isNonPlaceholderSHA256(manifest.provenance.sourceCheckpointSHA256),
              isNonPlaceholderSHA256(manifest.provenance.exportScriptSHA256),
              !manifest.provenance.licenses.isEmpty,
              manifest.provenance.licenses.allSatisfy(isSafeRelativePath),
              !manifest.provenance.generatedBy.isEmpty else {
            throw FACADEProtectionError.modelUnavailable("parity, fixture, or provenance metadata is incomplete")
        }

        if mode == .releaseCandidate {
            guard manifest.evidence.tier == .renderedAttackValidated,
                  manifest.evidence.v0RenderedSwap.status == .passed,
                  manifest.evidence.studentRenderedSwap.status == .passed,
                  manifest.evidence.consumerAppearanceDecision == "pass",
                  manifest.quantizedParity.unchangedJudgeResult == "pass",
                  manifest.quantizedParity.visualReviewResult == "pass",
                  !manifest.quantizedParity.floatVsQuantizedMetrics.isEmpty else {
                throw FACADEProtectionError.evidenceTierRejected(
                    "release mode requires V0 and student rendered-swap passes plus consumer appearance pass"
                )
            }
        }

        for artifact in manifest.artifacts {
            guard isSHA256(artifact.sha256), isNonPlaceholderSHA256(artifact.sourceWeightsSHA256),
                  artifact.sha256Kind == "facade_sorted_tree_v1_or_file_bytes",
                  isSafeRelativePath(artifact.relativePath),
                  isSafeRelativePath(artifact.modelCardPath),
                  !artifact.modelFamily.isEmpty,
                  !artifact.computeUnits.isEmpty,
                  Set(artifact.computeUnits).count == artifact.computeUnits.count,
                  !artifact.inputs.isEmpty,
                  !artifact.outputs.isEmpty,
                  artifact.inputs.allSatisfy(tensorIsValid),
                  artifact.outputs.allSatisfy(tensorIsValid),
                  !artifact.quantization.scheme.isEmpty,
                  !artifact.quantization.granularity.isEmpty,
                  isNonPlaceholderSHA256(artifact.quantization.calibrationManifestSHA256) else {
                throw FACADEProtectionError.modelUnavailable("artifact hashes must be lowercase SHA-256 values")
            }
            if artifact.role == .arcFaceForwardVerifier || artifact.role == .simSwapForwardVerifier {
                guard artifact.storagePrecision == .int8,
                      artifact.gradientSemantics == .none,
                      artifact.format == .coreMLPackage || artifact.format == .compiledCoreML else {
                    throw FACADEProtectionError.modelUnavailable("forward verifiers must be real INT8 Core ML artifacts")
                }
            }
            if artifact.role == .arcFaceScoreGradient {
                let isMPSGraphGradient = artifact.format == .mpsGraphPackage &&
                    artifact.gradientSemantics != .none
                let isStaticCoreMLGradient =
                    (artifact.format == .coreMLPackage || artifact.format == .compiledCoreML) &&
                    artifact.gradientSemantics == .staticGradientOutput
                guard isMPSGraphGradient || isStaticCoreMLGradient else {
                    throw FACADEProtectionError.modelUnavailable(
                        "the ArcFace gradient must be runtime MPSGraph autodiff or an explicit static Core ML gradient-output graph"
                    )
                }
                if artifact.executionPrecision == .int8,
                   artifact.gradientSemantics != .staticGradientOutput {
                    throw FACADEProtectionError.modelUnavailable(
                        "INT8 backward may only be claimed for an exported static-gradient output graph"
                    )
                }
                if artifact.gradientSemantics == .staticGradientOutput,
                   artifact.executionPrecision == .int8 {
                    throw FACADEProtectionError.modelUnavailable(
                        "INT8 weight storage must not be declared as INT8 static-gradient execution"
                    )
                }
            }
        }

        guard manifest.artifacts.count == FACADEModelRole.allCases.count,
              Set(manifest.artifacts.map(\.role)) == Set(FACADEModelRole.allCases),
              manifest.artifacts.first(where: { $0.role == .arcFaceForwardVerifier })?.sourceWeightsSHA256 ==
                manifest.artifacts.first(where: { $0.role == .arcFaceScoreGradient })?.sourceWeightsSHA256 else {
            throw FACADEProtectionError.modelUnavailable(
                "ArcFace forward and gradient artifacts must derive from the same source weights"
            )
        }

        for resource in manifest.adapterResources {
            guard isNonPlaceholderSHA256(resource.sha256),
                  isNonPlaceholderSHA256(resource.sourceModelSHA256),
                  isSafeRelativePath(resource.relativePath),
                  !resource.format.isEmpty,
                  !resource.sourceInitializer.isEmpty,
                  resource.byteCount > 0,
                  !resource.shape.isEmpty,
                  resource.shape.allSatisfy({ $0 > 0 }) else {
                throw FACADEProtectionError.modelUnavailable(
                    "adapter resource metadata is incomplete or its hashes are not lowercase SHA-256 values"
                )
            }
            switch resource.role {
            case .inSwapperIdentityMapEET:
                guard resource.byteCount == FACADEAdapterResource.identityMapEETByteCount,
                      resource.shape == [
                          FACADEAdapterResource.identityMapEETDimension,
                          FACADEAdapterResource.identityMapEETDimension,
                      ],
                      resource.format == FACADEAdapterResource.identityMapEETFormat else {
                    throw FACADEProtectionError.modelUnavailable(
                        "the InSwapper identity-map resource must be a 512x512 row-major little-endian float32 matrix"
                    )
                }
            }
        }
        guard manifest.adapterResources.count == FACADEAdapterResourceRole.allCases.count,
              Set(manifest.adapterResources.map(\.role)) == Set(FACADEAdapterResourceRole.allCases) else {
            throw FACADEProtectionError.modelUnavailable(
                "the handoff does not contain every required adapter resource"
            )
        }
    }

    private static func validateJSONEnvelope(_ data: Data) throws {
        let root: Any
        do {
            root = try JSONSerialization.jsonObject(with: data)
        } catch {
            throw FACADEProtectionError.modelUnavailable("manifest.json is not valid JSON: \(error)")
        }
        guard let object = root as? [String: Any] else {
            throw FACADEProtectionError.modelUnavailable("manifest.json root must be an object")
        }
        try requireExactKeys(object, [
            "schema_version", "bundle_identifier", "bundle_version", "created_at",
            "minimum_ios_version", "coremltools_version", "xcode_version",
            "swift_package_api_version", "evidence", "data_policy", "preprocessing",
            "face_detection_alignment", "local_verification", "artifacts",
            "adapter_resources", "quantized_parity", "benchmark_fixture", "provenance",
        ], context: "manifest")

        let evidence = try requireObject(object["evidence"], context: "evidence")
        try requireExactKeys(evidence, [
            "tier", "v0_rendered_swap", "student_rendered_swap", "synthetic_adults_only",
            "development_identities", "reserve_sealed_identities",
            "consumer_appearance_decision", "proxy_scores_are_not_efficacy_claims",
        ], context: "evidence")
        for key in ["v0_rendered_swap", "student_rendered_swap"] {
            try requireExactKeys(
                try requireObject(evidence[key], context: "evidence.\(key)"),
                ["status", "artifact_identifier", "sha256"],
                context: "evidence.\(key)"
            )
        }

        try requireExactKeys(
            try requireObject(object["data_policy"], context: "data_policy"),
            ["personal_photos_used", "network_required", "training_scope", "model_provenance_document"],
            context: "data_policy"
        )
        try requireExactKeys(
            try requireObject(object["preprocessing"], context: "preprocessing"),
            [
                "color_space", "orientation_policy", "crop_size", "landmark_order",
                "channel_order", "normalization", "resize_kernel", "pixel_rounding",
            ],
            context: "preprocessing"
        )
        try requireExactKeys(
            try requireObject(object["face_detection_alignment"], context: "face_detection_alignment"),
            [
                "runtime", "request_revision", "minimum_face_fraction", "coordinate_convention",
                "face_selection_policy", "failure_policy",
            ],
            context: "face_detection_alignment"
        )

        let localVerification = try requireObject(object["local_verification"], context: "local_verification")
        try requireExactKeys(localVerification, [
            "face_policy", "required_transforms", "exact_uint8_linf_maximum", "thresholds",
            "missing_or_unscorable_is_failure", "threshold_protocol_sha256",
            "independent_from_protection_engine",
        ], context: "local_verification")
        for (index, threshold) in try requireArray(localVerification["thresholds"], context: "local_verification.thresholds").enumerated() {
            try requireExactKeys(
                try requireObject(threshold, context: "threshold[\(index)]"),
                ["check", "comparison", "threshold", "calibrated_margin"],
                context: "threshold[\(index)]"
            )
        }

        for (index, artifactValue) in try requireArray(object["artifacts"], context: "artifacts").enumerated() {
            let artifact = try requireObject(artifactValue, context: "artifact[\(index)]")
            try requireExactKeys(artifact, [
                "role", "relative_path", "sha256", "sha256_kind", "format", "model_family",
                "source_weights_sha256", "storage_precision", "execution_precision",
                "gradient_semantics", "compute_units", "inputs", "outputs", "quantization",
                "model_card_path",
            ], context: "artifact[\(index)]")
            for tensorKey in ["inputs", "outputs"] {
                for (tensorIndex, tensorValue) in try requireArray(artifact[tensorKey], context: "artifact[\(index)].\(tensorKey)").enumerated() {
                    try requireExactKeys(
                        try requireObject(tensorValue, context: "artifact[\(index)].\(tensorKey)[\(tensorIndex)]"),
                        ["name", "shape", "data_type", "layout", "value_range"],
                        context: "artifact[\(index)].\(tensorKey)[\(tensorIndex)]"
                    )
                }
            }
            try requireExactKeys(
                try requireObject(artifact["quantization"], context: "artifact[\(index)].quantization"),
                ["scheme", "granularity", "calibration_manifest_sha256"],
                context: "artifact[\(index)].quantization"
            )
        }

        for (index, resourceValue) in try requireArray(object["adapter_resources"], context: "adapter_resources").enumerated() {
            try requireExactKeys(
                try requireObject(resourceValue, context: "adapter_resource[\(index)]"),
                [
                    "role", "relative_path", "sha256", "byte_count", "shape", "format",
                    "source_model_sha256", "source_initializer",
                ],
                context: "adapter_resource[\(index)]"
            )
        }

        try requireExactKeys(
            try requireObject(object["quantized_parity"], context: "quantized_parity"),
            [
                "protocol_sha256", "fixture_manifest_sha256", "float_vs_quantized_metrics",
                "unchanged_judge_result", "visual_review_result",
            ],
            context: "quantized_parity"
        )
        try requireExactKeys(
            try requireObject(object["benchmark_fixture"], context: "benchmark_fixture"),
            ["relative_path", "sha256", "synthetic_adult", "expected_input_sha256"],
            context: "benchmark_fixture"
        )
        try requireExactKeys(
            try requireObject(object["provenance"], context: "provenance"),
            [
                "research_protocol_sha256", "source_checkpoint_sha256", "export_script_sha256",
                "licenses", "generated_by",
            ],
            context: "provenance"
        )
    }

    private static func requireObject(_ value: Any?, context: String) throws -> [String: Any] {
        guard let object = value as? [String: Any] else {
            throw FACADEProtectionError.modelUnavailable("\(context) must be an object")
        }
        return object
    }

    private static func requireArray(_ value: Any?, context: String) throws -> [Any] {
        guard let array = value as? [Any] else {
            throw FACADEProtectionError.modelUnavailable("\(context) must be an array")
        }
        return array
    }

    private static func requireExactKeys(
        _ object: [String: Any],
        _ expected: Set<String>,
        context: String
    ) throws {
        let actual = Set(object.keys)
        guard actual == expected else {
            let missing = expected.subtracting(actual).sorted().joined(separator: ",")
            let unexpected = actual.subtracting(expected).sorted().joined(separator: ",")
            throw FACADEProtectionError.modelUnavailable(
                "\(context) keys do not match the handoff schema; missing=[\(missing)] unexpected=[\(unexpected)]"
            )
        }
    }

    private static func evidenceReferenceIsValid(_ reference: FACADEEvidenceReference) -> Bool {
        switch reference.status {
        case .pending:
            return reference.artifactIdentifier == nil && reference.sha256 == nil
        case .passed, .failed:
            return !(reference.artifactIdentifier ?? "").isEmpty &&
                reference.sha256.map(isNonPlaceholderSHA256) == true
        }
    }

    private static func verificationThresholdsAreValid(
        _ thresholds: [FACADEVerificationThreshold]
    ) -> Bool {
        let expectedChecks = Set([
            "arcface_raw", "arcface_jpeg_q70", "arcface_heic_q80",
            "simswap_raw", "simswap_jpeg_q70", "simswap_heic_q80",
        ])
        return thresholds.count == expectedChecks.count &&
            Set(thresholds.map(\.check)) == expectedChecks &&
            thresholds.allSatisfy { threshold in
                ["less_than_or_equal", "greater_than_or_equal"].contains(threshold.comparison) &&
                    threshold.threshold.isFinite && threshold.calibratedMargin.isFinite &&
                    threshold.calibratedMargin >= 0
            }
    }

    private static func tensorIsValid(_ tensor: FACADETensorContract) -> Bool {
        !tensor.name.isEmpty && !tensor.shape.isEmpty && tensor.shape.allSatisfy({ $0 > 0 }) &&
            ["float16", "float32", "int32", "image"].contains(tensor.dataType) &&
            !tensor.layout.isEmpty && tensor.valueRange.count == 2 &&
            tensor.valueRange.allSatisfy(\.isFinite) && tensor.valueRange[0] <= tensor.valueRange[1]
    }

    private static func isISO8601Date(_ value: String) -> Bool {
        let standard = ISO8601DateFormatter()
        if standard.date(from: value) != nil {
            return true
        }
        let fractional = ISO8601DateFormatter()
        fractional.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return fractional.date(from: value) != nil
    }

    private static func isMajorMinorVersion(_ value: String) -> Bool {
        let pieces = value.split(separator: ".", omittingEmptySubsequences: false)
        return pieces.count == 2 && pieces.allSatisfy { !$0.isEmpty && $0.allSatisfy(\.isNumber) }
    }

    private static func runtimeMeetsMinimumIOS(_ minimum: String) -> Bool {
        let pieces = minimum.split(separator: ".")
        guard pieces.count == 2,
              let major = Int(pieces[0]),
              let minor = Int(pieces[1]) else {
            return false
        }
        let current = ProcessInfo.processInfo.operatingSystemVersion
        return current.majorVersion > major ||
            (current.majorVersion == major && current.minorVersion >= minor)
    }

    private static func isSafeRelativePath(_ value: String) -> Bool {
        guard !value.isEmpty, !value.hasPrefix("/"), URL(string: value)?.scheme == nil else {
            return false
        }
        let components = value.split(separator: "/", omittingEmptySubsequences: false)
        return !components.contains("..") && !components.contains(".") && !components.contains("")
    }

    private static func safeArtifactURL(rootURL: URL, relativePath: String) throws -> URL {
        guard isSafeRelativePath(relativePath) else {
            throw FACADEProtectionError.modelUnavailable("artifact paths must be local and relative")
        }

        let standardizedRoot = rootURL.standardizedFileURL
        let candidate = standardizedRoot.appendingPathComponent(relativePath).standardizedFileURL
        guard candidate.path.hasPrefix(standardizedRoot.path + "/") else {
            throw FACADEProtectionError.modelUnavailable("artifact escaped the model bundle")
        }
        var componentURL = standardizedRoot
        for component in relativePath.split(separator: "/") {
            componentURL.appendPathComponent(String(component))
            let values = try componentURL.resourceValues(forKeys: [.isSymbolicLinkKey])
            guard values.isSymbolicLink != true else {
                throw FACADEProtectionError.modelUnavailable("symbolic links are prohibited in model bundles")
            }
        }
        return candidate
    }

    private static func deterministicSHA256(of url: URL) throws -> String {
        let values = try url.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
        guard values.isSymbolicLink != true else {
            throw FACADEProtectionError.modelUnavailable("symbolic links are prohibited in model artifacts")
        }

        if values.isDirectory == true {
            let keys: Set<URLResourceKey> = [.isDirectoryKey, .isSymbolicLinkKey]
            guard let enumerator = FileManager.default.enumerator(
                at: url,
                includingPropertiesForKeys: Array(keys),
                options: [.skipsHiddenFiles]
            ) else {
                throw FACADEProtectionError.modelUnavailable("could not enumerate model artifact")
            }
            var files: [URL] = []
            for case let child as URL in enumerator {
                let childValues = try child.resourceValues(forKeys: keys)
                guard childValues.isSymbolicLink != true else {
                    throw FACADEProtectionError.modelUnavailable("symbolic links are prohibited in model artifacts")
                }
                if childValues.isDirectory != true {
                    files.append(child)
                }
            }
            files.sort { $0.path < $1.path }

            var hasher = SHA256()
            for file in files {
                let rootComponents = url.standardizedFileURL.pathComponents
                let fileComponents = file.standardizedFileURL.pathComponents
                guard fileComponents.starts(with: rootComponents),
                      fileComponents.count > rootComponents.count else {
                    throw FACADEProtectionError.modelUnavailable(
                        "model artifact contains a path outside its root"
                    )
                }
                let relative = fileComponents
                    .dropFirst(rootComponents.count)
                    .joined(separator: "/")
                hasher.update(data: Data(relative.utf8))
                hasher.update(data: Data([0]))
                hasher.update(data: try Data(contentsOf: file, options: [.mappedIfSafe]))
                hasher.update(data: Data([0]))
            }
            return hasher.finalize().map { String(format: "%02x", $0) }.joined()
        }

        return SHA256.hash(data: try Data(contentsOf: url, options: [.mappedIfSafe]))
            .map { String(format: "%02x", $0) }
            .joined()
    }

    private static func isSHA256(_ value: String) -> Bool {
        value.utf8.count == 64 && value.utf8.allSatisfy { byte in
            (48...57).contains(byte) || (97...102).contains(byte)
        }
    }

    private static func isNonPlaceholderSHA256(_ value: String) -> Bool {
        isSHA256(value) && value != String(repeating: "0", count: 64)
    }
}
