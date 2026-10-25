// ModelHandoff.swift
//
// Everything to do with the model bundle on disk and its `manifest.json`.
//
// The FACADE app does not trust a model file just because it is present. The
// bundle ships with a manifest that lists every model, every supporting data
// file, and every document, each with a SHA-256 digest and a description of
// its inputs and outputs. This file:
//
//   1. Declares the Swift types that `manifest.json` decodes into
//      (`FACADEModelHandoffManifest` and the structs it contains). The
//      `CodingKeys` on each struct map snake_case JSON names to Swift names.
//   2. Implements `FACADEModelHandoffLoader`, which reads the manifest, checks
//      that its keys are exactly the ones expected, validates every field,
//      then walks the bundle and re-hashes every referenced file. Any mismatch
//      means nothing is loaded.
//   3. Implements `FACADEIdentityMapMatrix`, which re-reads and re-hashes the
//      one non-model data file (a 512x512 matrix) at the moment it is used.
//
// Key idea: the manifest is the single source of truth about what the app is
// allowed to run. The app additionally pins the SHA-256 of the manifest file
// itself (see `NativePreparationLoader`), so the whole chain from "bytes on
// disk" to "model in memory" is checked against values fixed at build time.
//
// Naming note: type names in this file contain the word "Handoff"; it simply
// means the model bundle, and the comments call it that.

import CryptoKit
import Foundation

// MARK: - Manifest vocabulary

/// The four models every bundle must contain, keyed by the manifest's `role`.
///
/// The app's protection loop uses the first two through FACADEKit. The
/// SimSwap verifier and the gradient model are loaded by the app's own native
/// code, but the bundle is still required to contain and hash-check them.
public enum FACADEModelRole: String, Codable, Sendable, CaseIterable {
    /// The network that proposes starting residuals (see `FACADEInitializerResidualBank`).
    case protectionEngine = "protection_engine"
    /// ArcFace-family recogniser used to compute the clean photo's identity embedding.
    case arcFaceForwardVerifier = "arcface_forward_verifier"
    /// SimSwap's recogniser, a second independent opinion on identity.
    case simSwapForwardVerifier = "simswap_forward_verifier"
    /// ArcFace score plus its gradient with respect to the input pixels.
    case arcFaceScoreGradient = "arcface_score_gradient"
}

/// File format of a model artifact.
public enum FACADEArtifactFormat: String, Codable, Sendable {
    /// A Core ML package (`.mlpackage`) that must be compiled before loading.
    case coreMLPackage = "coreml_mlpackage"
    /// An already compiled Core ML model (`.mlmodelc`).
    case compiledCoreML = "coreml_mlmodelc"
    /// A Metal Performance Shaders Graph package. Allowed by the schema but not
    /// used by the shipped bundle.
    case mpsGraphPackage = "mpsgraphpackage"
}

/// Numeric precision, used both for how weights are stored and how they run.
public enum FACADEArtifactPrecision: String, Codable, Sendable {
    case int8
    case float16
    case float32
    case mixed
}

/// Whether, and how, a model exposes a gradient.
public enum FACADEGradientSemantics: String, Codable, Sendable {
    /// Forward inference only.
    case none
    /// The runtime differentiates the graph itself (MPSGraph style).
    case runtimeAutomaticDifferentiation = "runtime_automatic_differentiation"
    /// The gradient was baked into the exported model as an extra output.
    case staticGradientOutput = "static_gradient_output"
}

/// Processor types a model is permitted to run on.
public enum FACADEComputeUnit: String, Codable, Sendable {
    case cpu
    case gpu
    /// Apple Neural Engine.
    case ane
}

/// Result of an external validation step recorded in the manifest.
public enum FACADEEvidenceStatus: String, Codable, Sendable {
    case pending
    case passed
    case failed
}

// MARK: - Manifest schema

/// Description of one tensor a model reads or writes: its name as Core ML
/// knows it, its shape, element type, layout label and expected value range.
public struct FACADETensorContract: Codable, Sendable, Equatable {
    /// Feature name used when calling the Core ML model.
    public let name: String
    /// Dimensions, for example [1, 3, 112, 112] for one RGB 112x112 image.
    public let shape: [Int]
    /// Element type: "float32", "float16", "int32" or "image".
    public let dataType: String
    /// Human-readable axis description such as "NCHW" (batch, channel, height, width).
    public let layout: String
    /// [minimum, maximum] a well-formed value is expected to lie within.
    public let valueRange: [Double]

    /// Plain memberwise initialiser; decoding from JSON does not use it.
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

    /// Maps each Swift property to its snake_case key in manifest.json.
    enum CodingKeys: String, CodingKey {
        case name
        case shape
        case dataType = "data_type"
        case layout
        case valueRange = "value_range"
    }
}

/// One model in the bundle: where it lives, what it is, how it was made, and
/// the digest its bytes must hash to.
public struct FACADEModelArtifact: Codable, Sendable, Equatable {
    /// Which of the four model slots this entry fills.
    public let role: FACADEModelRole
    /// Path relative to the bundle root, for example `Models/protection_engine.mlpackage`.
    public let relativePath: String
    /// Expected SHA-256 of the artifact (see `sha256Kind` for how a directory is hashed).
    public let sha256: String
    /// Names the hashing rule: a plain file hashes its bytes; a package
    /// directory hashes its sorted file list (see `deterministicSHA256`).
    public let sha256Kind: String
    /// Package, compiled model or MPSGraph package (see `FACADEArtifactFormat`).
    public let format: FACADEArtifactFormat
    /// Human-readable family name, for example "InsightFace Buffalo-L w600k_r50".
    public let modelFamily: String
    /// Digest of the original weights this artifact was converted from.
    public let sourceWeightsSHA256: String
    /// How the weights are stored on disk (8-bit for the shipped recognisers).
    public let storagePrecision: FACADEArtifactPrecision
    /// Precision the model computes in at run time.
    public let executionPrecision: FACADEArtifactPrecision
    /// Whether the model also outputs a gradient (see `FACADEGradientSemantics`).
    public let gradientSemantics: FACADEGradientSemantics
    /// Processors the model may run on; duplicates are rejected.
    public let computeUnits: [FACADEComputeUnit]
    /// Every input tensor, in the order the model declares them.
    public let inputs: [FACADETensorContract]
    /// Every output tensor, in the order the model declares them.
    public let outputs: [FACADETensorContract]
    /// How the weights were reduced to fewer bits.
    public let quantization: FACADEQuantizationContract
    /// Path to the model's documentation card inside the bundle.
    public let modelCardPath: String

    /// Maps each Swift property to its snake_case key in manifest.json.
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

/// How a model's weights were shrunk to 8-bit (or not), and the digest of the
/// data used to calibrate that step.
public struct FACADEQuantizationContract: Codable, Sendable, Equatable {
    /// Quantisation method, for example "linear_symmetric_int8_weight_only".
    public let scheme: String
    /// Whether scales were chosen per tensor, per channel, and so on.
    public let granularity: String
    /// Digest of the list of calibration inputs used when choosing scales.
    public let calibrationManifestSHA256: String

    /// Maps each Swift property to its snake_case key in manifest.json.
    enum CodingKeys: String, CodingKey {
        case scheme
        case granularity
        case calibrationManifestSHA256 = "calibration_manifest_sha256"
    }
}

/// A pointer to an external test result: its status and, once it exists, the
/// identifier and digest of the record that proves it.
public struct FACADEEvidenceReference: Codable, Sendable, Equatable {
    /// Whether the test is still to run, passed, or failed.
    public let status: FACADEEvidenceStatus
    /// Name of the record holding the result; nil while `status` is pending.
    public let artifactIdentifier: String?
    /// Digest of that record; nil while `status` is pending.
    public let sha256: String?

    /// Maps each Swift property to its snake_case key in manifest.json.
    enum CodingKeys: String, CodingKey {
        case status
        case artifactIdentifier = "artifact_identifier"
        case sha256
    }
}

/// The manifest's `evidence` section: how far the bundle has been tested and
/// which data was used to test it.
public struct FACADEHandoffEvidence: Codable, Sendable, Equatable {
    /// Overall testing level of the bundle (see `FACADEEvidenceTier`).
    public let tier: FACADEEvidenceTier
    /// Result of a rendered face-swap test against the original full-size model.
    public let v0RenderedSwap: FACADEEvidenceReference
    /// Result of the same test against the smaller on-device model.
    public let studentRenderedSwap: FACADEEvidenceReference
    /// True when only synthetic adult faces were used for evaluation.
    public let syntheticAdultsOnly: Bool
    /// Identifier range of the synthetic faces used during development.
    public let developmentIdentities: String
    /// Identifier range of synthetic faces held back and never looked at.
    public let reserveSealedIdentities: String
    /// "pass" or "no_go": whether the visible change was judged acceptable.
    public let consumerAppearanceDecision: String
    /// Acknowledges that recogniser scores are stand-ins, not proof of protection.
    public let proxyScoresAreNotEfficacyClaims: Bool

    /// Maps each Swift property to its snake_case key in manifest.json.
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

/// Privacy commitments the bundle makes: no personal photos in training and
/// no network access at run time. The loader refuses a bundle that says otherwise.
public struct FACADEDataPolicy: Codable, Sendable, Equatable {
    /// Must be false.
    public let personalPhotosUsed: Bool
    /// Must be false: everything runs on the phone.
    public let networkRequired: Bool
    /// Must be "synthetic_adults_only".
    public let trainingScope: String
    /// Bundle-relative path of the document describing where the models came from.
    public let modelProvenanceDocument: String

    /// Maps each Swift property to its snake_case key in manifest.json.
    enum CodingKeys: String, CodingKey {
        case personalPhotosUsed = "personal_photos_used"
        case networkRequired = "network_required"
        case trainingScope = "training_scope"
        case modelProvenanceDocument = "model_provenance_document"
    }
}

/// One pass/fail rule for a recogniser score, for example
/// "arcface_jpeg_q70 must be less than or equal to 0.95".
public struct FACADEVerificationThreshold: Codable, Sendable, Equatable {
    /// Which score: "<recogniser>_<save format>", for example "simswap_heic_q80".
    public let check: String
    /// "less_than_or_equal" or "greater_than_or_equal".
    public let comparison: String
    /// The cut-off value.
    public let threshold: Double
    /// Extra safety margin applied on top of the cut-off; never negative.
    public let calibratedMargin: Double

    /// Maps each Swift property to its snake_case key in manifest.json.
    enum CodingKeys: String, CodingKey {
        case check
        case comparison
        case threshold
        case calibratedMargin = "calibrated_margin"
    }
}

/// The manifest's `local_verification` section: the rules an on-device check
/// of a protected photo must follow. The shipped app performs its own scoring
/// natively, but the loader still validates that this section is well formed.
public struct FACADELocalVerificationContract: Codable, Sendable, Equatable {
    /// "exactly_one_face": a check on a photo with zero or several faces fails.
    public let facePolicy: String
    /// Save formats the protected photo must survive: "raw", "jpeg_q70", "heic_q80".
    public let requiredTransforms: [String]
    /// Largest allowed per-channel change, in 0...255 units, after saving.
    public let exactUInt8LInfinityMaximum: Int
    /// One pass/fail rule per recogniser per save format (six in total).
    public let thresholds: [FACADEVerificationThreshold]
    /// A photo that cannot be scored counts as a failure, never as a pass.
    public let missingOrUnscorableIsFailure: Bool
    /// Digest of the written procedure that produced the thresholds.
    public let thresholdProtocolSHA256: String
    /// The checking models must not be the model that made the protection.
    public let independentFromProtectionEngine: Bool

    /// Maps each Swift property to its snake_case key in manifest.json.
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

/// The manifest's `preprocessing` section: how pixels must be prepared before
/// they reach each model (colour space, orientation, crop, normalisation).
public struct FACADEPreprocessingContract: Codable, Sendable, Equatable {
    /// Working colour space; must be "sRGB".
    public let colorSpace: String
    /// How camera rotation metadata is handled; must be "apply_exif_then_strip_orientation".
    public let orientationPolicy: String
    /// Side lengths of the aligned face crop, [112, 112] for the bundled recognisers.
    public let cropSize: [Int]
    /// The five landmarks in the order the alignment template expects.
    public let landmarkOrder: [String]
    /// "RGB" or "BGR".
    public let channelOrder: String
    /// Per-model input scaling, written as a short formula string.
    public let normalization: String
    /// Resampling filter used when the photo is scaled.
    public let resizeKernel: String
    /// How float pixels are converted back to 8-bit when a photo is saved.
    public let pixelRounding: String

    /// Maps each Swift property to its snake_case key in manifest.json.
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

/// The manifest's `face_detection_alignment` section: which face detector to
/// use and the rule for accepting a photo. The app reads `requestRevision`
/// and `minimumFaceFraction` from here when it builds the Vision detector.
public struct FACADEFaceDetectionAlignment: Codable, Sendable, Equatable {
    /// Detector framework; must be "Apple Vision".
    public let runtime: String
    /// Apple Vision request revision number to use.
    public let requestRevision: Int
    /// Smallest face (as a fraction of the shorter frame side) that is accepted.
    public let minimumFaceFraction: Double
    /// Name of the landmark-to-crop mapping rule (see `FACADEVisionSingleFaceDetector`).
    public let coordinateConvention: String
    /// Must be "exactly_one_face".
    public let faceSelectionPolicy: String
    /// Must be "fail_closed_no_output": no face means no protected photo.
    public let failurePolicy: String

    /// Maps each Swift property to its snake_case key in manifest.json.
    enum CodingKeys: String, CodingKey {
        case runtime
        case requestRevision = "request_revision"
        case minimumFaceFraction = "minimum_face_fraction"
        case coordinateConvention = "coordinate_convention"
        case faceSelectionPolicy = "face_selection_policy"
        case failurePolicy = "failure_policy"
    }
}

/// Numbers recorded when the 8-bit models were compared with their full
/// precision originals, plus the digests of the procedure and test data.
public struct FACADEQuantizedParity: Codable, Sendable, Equatable {
    /// Digest of the written comparison procedure.
    public let protocolSHA256: String
    /// Digest of the list of test inputs used in the comparison.
    public let fixtureManifestSHA256: String
    /// Measured differences, keyed by name (for example "arcface.max_abs").
    public let floatVsQuantizedMetrics: [String: Double]
    /// "pending", "pass" or "fail".
    public let unchangedJudgeResult: String
    /// "pending", "pass" or "fail".
    public let visualReviewResult: String

    /// Maps each Swift property to its snake_case key in manifest.json.
    enum CodingKeys: String, CodingKey {
        case protocolSHA256 = "protocol_sha256"
        case fixtureManifestSHA256 = "fixture_manifest_sha256"
        case floatVsQuantizedMetrics = "float_vs_quantized_metrics"
        case unchangedJudgeResult = "unchanged_judge_result"
        case visualReviewResult = "visual_review_result"
    }
}

/// A synthetic test photo shipped in the bundle, hash-checked like a model.
public struct FACADEBenchmarkFixture: Codable, Sendable, Equatable {
    /// Path of the photo inside the bundle.
    public let relativePath: String
    /// Digest the photo's bytes must match.
    public let sha256: String
    /// Must be true: the test photo is a generated adult face, not a real person.
    public let syntheticAdult: Bool
    /// Digest the photo should have after the app's own preparation step.
    public let expectedInputSHA256: String

    /// Maps each Swift property to its snake_case key in manifest.json.
    enum CodingKeys: String, CodingKey {
        case relativePath = "relative_path"
        case sha256
        case syntheticAdult = "synthetic_adult"
        case expectedInputSHA256 = "expected_input_sha256"
    }
}

/// Where the models came from: digests of the procedure, source checkpoint and
/// export script, plus the licence files that must ship alongside them.
public struct FACADEModelProvenance: Codable, Sendable, Equatable {
    /// Digest of the written procedure the models were produced under.
    public let researchProtocolSHA256: String
    /// Digest of the trained weights the protection model was exported from.
    public let sourceCheckpointSHA256: String
    /// Digest of the script that converted the models to Core ML.
    public let exportScriptSHA256: String
    /// Bundle-relative paths of licence and notice files; must all exist.
    public let licenses: [String]
    /// Free-text description of how the bundle was assembled.
    public let generatedBy: String

    /// Maps each Swift property to its snake_case key in manifest.json.
    enum CodingKeys: String, CodingKey {
        case researchProtocolSHA256 = "research_protocol_sha256"
        case sourceCheckpointSHA256 = "source_checkpoint_sha256"
        case exportScriptSHA256 = "export_script_sha256"
        case licenses
        case generatedBy = "generated_by"
    }
}

/// Roles for supporting data files that are not models.
public enum FACADEAdapterResourceRole: String, Codable, Sendable, CaseIterable {
    /// The 512x512 matrix E E^T taken from the InSwapper face-swapper's
    /// identity-map initialiser. The swapper does not consume a raw ArcFace
    /// embedding r; it consumes normalize(E E^T r). The app multiplies by this
    /// matrix so the protection targets the vector the swapper actually sees.
    case inSwapperIdentityMapEET = "inswapper_identity_map_eet"
}

/// A non-model data file the app needs. It is path-checked, size-checked and
/// hashed like a model but kept in its own list so the set of model roles
/// stays exactly four.
public struct FACADEAdapterResource: Codable, Sendable, Equatable {
    /// The identity-map matrix is always 512 x 512 ...
    public static let identityMapEETDimension = 512
    /// ... of 4-byte floats, so exactly 1 MiB on disk ...
    public static let identityMapEETByteCount = 512 * 512 * 4
    /// ... stored row by row, little-endian, as the manifest must declare.
    public static let identityMapEETFormat = "row-major-little-endian-float32"

    /// What the file is for (today only the identity-map matrix).
    public let role: FACADEAdapterResourceRole
    /// Path of the file inside the bundle.
    public let relativePath: String
    /// Digest the file's bytes must match.
    public let sha256: String
    /// Exact size the file must have on disk, in bytes.
    public let byteCount: Int
    /// Matrix dimensions, [512, 512] for the identity map.
    public let shape: [Int]
    /// Byte layout name; must equal `identityMapEETFormat`.
    public let format: String
    /// Digest of the model file the matrix was extracted from.
    public let sourceModelSHA256: String
    /// Name of the tensor inside that model (for example "buff2fs").
    public let sourceInitializer: String

    /// Maps each Swift property to its snake_case key in manifest.json.
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

/// The whole `manifest.json`. Every field is required; `validateJSONEnvelope`
/// additionally rejects a file with any extra or missing key so a manifest
/// written for a different schema cannot be partially understood.
public struct FACADEModelHandoffManifest: Codable, Sendable, Equatable {
    /// Manifest layout version; this code understands version 1 only.
    public let schemaVersion: Int
    /// Reverse-DNS style name of the bundle.
    public let bundleIdentifier: String
    /// Free-form version string of the bundle.
    public let bundleVersion: String
    /// ISO-8601 timestamp of when the bundle was assembled.
    public let createdAt: String
    /// Lowest iOS version the models run on, as "major.minor".
    public let minimumIOSVersion: String
    /// Version of Apple's coremltools used for the conversion.
    public let coremltoolsVersion: String
    /// Xcode version the bundle was checked with.
    public let xcodeVersion: String
    /// Version of this package's API the manifest was written for; must be "1.0.0".
    public let swiftPackageAPIVersion: String
    /// Testing status and data-use declarations.
    public let evidence: FACADEHandoffEvidence
    /// Privacy commitments.
    public let dataPolicy: FACADEDataPolicy
    /// Pixel preparation rules.
    public let preprocessing: FACADEPreprocessingContract
    /// Face detector settings.
    public let faceDetectionAlignment: FACADEFaceDetectionAlignment
    /// Rules for checking a protected photo on the device.
    public let localVerification: FACADELocalVerificationContract
    /// The four models.
    public let artifacts: [FACADEModelArtifact]
    /// Supporting data files (the identity-map matrix).
    public let adapterResources: [FACADEAdapterResource]
    /// 8-bit versus full-precision comparison results.
    public let quantizedParity: FACADEQuantizedParity
    /// The bundled synthetic test photo.
    public let benchmarkFixture: FACADEBenchmarkFixture
    /// Origin of the models and their licences.
    public let provenance: FACADEModelProvenance

    /// Maps each Swift property to its snake_case key in manifest.json.
    /// This is also the exact list of top-level keys `validateJSONEnvelope` demands.
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

// MARK: - Loaded bundle

/// What `FACADEModelHandoffLoader.load` returns once every check has passed:
/// the decoded manifest, the digest of the manifest bytes, and the verified
/// on-disk location of each model and data file.
public struct FACADELoadedModelHandoff: Sendable {
    /// Directory that contains `manifest.json`.
    public let rootURL: URL
    /// The decoded and fully validated manifest.
    public let manifest: FACADEModelHandoffManifest
    /// SHA-256 of the raw `manifest.json` bytes. The app compares this with a
    /// value fixed in its source so only the one validated manifest is accepted.
    public let manifestSHA256: String
    /// Hash-verified file locations, one per model role.
    public let artifactURLs: [FACADEModelRole: URL]
    /// Hash-verified file locations, one per supporting data file.
    public let adapterResourceURLs: [FACADEAdapterResourceRole: URL]
}

/// The 512x512 identity-map matrix, read from disk and hashed again at the
/// moment it is used. Re-hashing here (and not only in the loader) means a
/// file swapped between validation and use would still be caught.
///
/// The app parses its own copy of this matrix for the arithmetic; it calls
/// `load(from:)` purely as an extra integrity check on the bytes.
public struct FACADEIdentityMapMatrix: Sendable, Equatable {
    /// Number of rows and of columns: 512.
    public static let dimension = FACADEAdapterResource.identityMapEETDimension

    /// Digest of the bytes that were actually read.
    public let sha256: String
    /// Row-major [512][512] float32 values. The matrix is symmetric, so the
    /// row/column orientation is immaterial.
    public let values: [Float]

    /// Accepts only a complete, finite 512x512 matrix.
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

    /// Reads the matrix file named in the manifest, checks its size and
    /// SHA-256 against the manifest, and decodes the little-endian floats.
    public static func load(from handoff: FACADELoadedModelHandoff) throws -> FACADEIdentityMapMatrix {
        guard let resource = handoff.manifest.adapterResources.first(
                  where: { $0.role == .inSwapperIdentityMapEET }
              ),
              let url = handoff.adapterResourceURLs[.inSwapperIdentityMapEET] else {
            throw FACADEProtectionError.modelUnavailable(
                "the model bundle does not include the InSwapper identity-map resource"
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
        // Both the manifest's declared size and the package constant must agree.
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
        // Byte layout: 262,144 consecutive IEEE-754 float32 values, little
        // endian. `loadUnaligned` is used because Data gives no alignment
        // guarantee; `UInt32(littleEndian:)` makes the decode correct on any
        // host byte order.
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
}

// MARK: - Loader

/// Reads and verifies a model bundle. There is no partial success: one bad
/// key, one unexpected value, one missing document or one hash mismatch and
/// `load` throws without returning anything.
public enum FACADEModelHandoffLoader {
    /// Loads the bundle rooted at `rootURL` for the given mode.
    ///
    /// Steps, in order:
    ///   1. Read `manifest.json` and confirm its key set matches the schema exactly.
    ///   2. Decode it into `FACADEModelHandoffManifest`.
    ///   3. Validate every field's value (`validate`).
    ///   4. On iOS, confirm the device meets the manifest's minimum OS version.
    ///   5. For every model: resolve a safe path, hash the file or package, and
    ///      compare with the manifest.
    ///   6. Do the same for every supporting data file, also checking size.
    ///   7. Hash-check the test photo and confirm every document and licence exists.
    ///   8. Confirm all four model roles were present.
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

        // Exact-key check first, before Codable, because JSONDecoder silently
        // ignores unknown keys and would let an unexpected field slip through.
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

        // Hash every model. A duplicate role is rejected so a second entry
        // cannot quietly override the first.
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

        // Supporting data files: must be regular files (not symlinks) of the
        // declared size, and must hash to the declared digest.
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

        // The bundled test photo is hashed too.
        let fixtureURL = try safeArtifactURL(
            rootURL: rootURL,
            relativePath: manifest.benchmarkFixture.relativePath
        )
        guard try deterministicSHA256(of: fixtureURL) == manifest.benchmarkFixture.sha256 else {
            throw FACADEProtectionError.modelUnavailable("benchmark fixture hash mismatch")
        }

        // Documentation and licence files must exist as plain files. They are
        // not hashed, but a bundle that drops its licences is refused.
        let documentationPaths = [manifest.dataPolicy.modelProvenanceDocument]
            + manifest.artifacts.map(\.modelCardPath)
            + manifest.provenance.licenses
        for relativePath in documentationPaths {
            let url = try safeArtifactURL(rootURL: rootURL, relativePath: relativePath)
            let values = try url.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey])
            guard values.isRegularFile == true, values.isSymbolicLink != true else {
                throw FACADEProtectionError.modelUnavailable(
                    "a required model-bundle document is missing or has an unsafe path: \(relativePath)"
                )
            }
        }

        let requiredRoles = Set(FACADEModelRole.allCases)
        guard Set(artifactURLs.keys) == requiredRoles else {
            throw FACADEProtectionError.modelUnavailable("the model bundle does not contain every required model role")
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

    /// Checks the value of every manifest field against what this app was
    /// built to accept. Most rules are plain sanity checks (non-empty strings,
    /// real SHA-256 digests, safe relative paths); a few pin exact strings so
    /// a bundle produced under different privacy or preprocessing rules is
    /// refused even if it is internally consistent.
    private static func validate(
        _ manifest: FACADEModelHandoffManifest,
        mode: FACADEProtectionMode
    ) throws {
        guard manifest.schemaVersion == 1 else {
            throw FACADEProtectionError.modelUnavailable("unsupported model-bundle manifest version")
        }
        // Bundle identity: identifier limited to [0-9A-Za-z.-], a parseable
        // creation date, a "major.minor" iOS version, and the package API
        // version this code understands.
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
        // Testing policy: synthetic adults only, with the exact identity ranges
        // this app's models were accepted with.
        guard manifest.evidence.syntheticAdultsOnly,
              manifest.evidence.proxyScoresAreNotEfficacyClaims else {
            throw FACADEProtectionError.modelUnavailable("the model bundle must declare that it was tested on synthetic adults only")
        }
        guard manifest.evidence.developmentIdentities == "0240-0271:evaluation-only" else {
            throw FACADEProtectionError.modelUnavailable("the manifest's development identity policy is not the one this app expects")
        }
        guard manifest.evidence.reserveSealedIdentities == "0272-0303:closed" else {
            throw FACADEProtectionError.modelUnavailable("the manifest's reserve/sealed identity policy is not the one this app expects")
        }
        guard ["no_go", "pass"].contains(manifest.evidence.consumerAppearanceDecision),
              evidenceReferenceIsValid(manifest.evidence.v0RenderedSwap),
              evidenceReferenceIsValid(manifest.evidence.studentRenderedSwap) else {
            throw FACADEProtectionError.modelUnavailable("the manifest's face-swap test-result metadata is invalid")
        }
        // Privacy: no personal photos, no network.
        guard !manifest.dataPolicy.personalPhotosUsed,
              !manifest.dataPolicy.networkRequired,
              manifest.dataPolicy.trainingScope == "synthetic_adults_only",
              isSafeRelativePath(manifest.dataPolicy.modelProvenanceDocument) else {
            throw FACADEProtectionError.modelUnavailable("the model bundle does not declare the offline, synthetic-only data policy")
        }
        // Preprocessing: sRGB, orientation baked in, and the five landmarks in
        // the order the alignment code assumes.
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
        // Face detection: Apple Vision, exactly one face, stop on failure.
        guard manifest.faceDetectionAlignment.runtime == "Apple Vision",
              manifest.faceDetectionAlignment.requestRevision >= 1,
              manifest.faceDetectionAlignment.minimumFaceFraction > 0,
              manifest.faceDetectionAlignment.minimumFaceFraction <= 1,
              !manifest.faceDetectionAlignment.coordinateConvention.isEmpty,
              manifest.faceDetectionAlignment.faceSelectionPolicy == "exactly_one_face",
              manifest.faceDetectionAlignment.failurePolicy == "fail_closed_no_output" else {
            throw FACADEProtectionError.modelUnavailable("the manifest does not require face detection to refuse ambiguous photos")
        }
        // Local verification rules: the three save formats and six thresholds.
        guard manifest.localVerification.facePolicy == "exactly_one_face",
              manifest.localVerification.missingOrUnscorableIsFailure,
              manifest.localVerification.independentFromProtectionEngine,
              manifest.localVerification.requiredTransforms == ["raw", "jpeg_q70", "heic_q80"],
              (0...16).contains(manifest.localVerification.exactUInt8LInfinityMaximum),
              isNonPlaceholderSHA256(manifest.localVerification.thresholdProtocolSHA256),
              verificationThresholdsAreValid(manifest.localVerification.thresholds) else {
            throw FACADEProtectionError.modelUnavailable("the manifest does not require local verification to refuse on any failure")
        }

        // Parity, test photo and provenance: all digests real, licences listed.
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

        // Release mode would need every external test to have passed. The
        // shipped app loads with `.experimentalDemo`, so this branch is not
        // taken today; it stays so a stricter bundle can be demanded later.
        if mode == .releaseCandidate {
            guard manifest.evidence.tier == .renderedAttackValidated,
                  manifest.evidence.v0RenderedSwap.status == .passed,
                  manifest.evidence.studentRenderedSwap.status == .passed,
                  manifest.evidence.consumerAppearanceDecision == "pass",
                  manifest.quantizedParity.unchangedJudgeResult == "pass",
                  manifest.quantizedParity.visualReviewResult == "pass",
                  !manifest.quantizedParity.floatVsQuantizedMetrics.isEmpty else {
                throw FACADEProtectionError.evidenceTierRejected(
                    "release mode requires every rendered face-swap test and the appearance test to have passed"
                )
            }
        }

        // Per-model rules.
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
            // The two recognisers must be 8-bit forward-only Core ML models.
            if artifact.role == .arcFaceForwardVerifier || artifact.role == .simSwapForwardVerifier {
                guard artifact.storagePrecision == .int8,
                      artifact.gradientSemantics == .none,
                      artifact.format == .coreMLPackage || artifact.format == .compiledCoreML else {
                    throw FACADEProtectionError.modelUnavailable("forward verifiers must be real INT8 Core ML artifacts")
                }
            }
            // The gradient model must either be an MPSGraph package that
            // differentiates at run time, or a Core ML model whose gradient was
            // exported as an explicit output. An 8-bit gradient is never accepted
            // because quantised backward passes are too inaccurate to trust.
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

        // Exactly one model per role, and the ArcFace forward and gradient
        // models must come from the same weights (otherwise the gradient
        // would be steering against a different recogniser than the one that
        // produced the reference embedding).
        guard manifest.artifacts.count == FACADEModelRole.allCases.count,
              Set(manifest.artifacts.map(\.role)) == Set(FACADEModelRole.allCases),
              manifest.artifacts.first(where: { $0.role == .arcFaceForwardVerifier })?.sourceWeightsSHA256 ==
                manifest.artifacts.first(where: { $0.role == .arcFaceScoreGradient })?.sourceWeightsSHA256 else {
            throw FACADEProtectionError.modelUnavailable(
                "ArcFace forward and gradient artifacts must derive from the same source weights"
            )
        }

        // Supporting data files: generic checks, then the exact shape, size
        // and byte format the identity map must have.
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
                "the model bundle does not contain every required adapter resource"
            )
        }
    }

    /// Parses the manifest as plain JSON and insists that every object has
    /// exactly the expected keys: nothing missing, nothing extra. This runs
    /// before Codable decoding, which would otherwise ignore unknown keys.
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
        // Top level: the eighteen sections every manifest must have.
        try requireExactKeys(object, [
            "schema_version", "bundle_identifier", "bundle_version", "created_at",
            "minimum_ios_version", "coremltools_version", "xcode_version",
            "swift_package_api_version", "evidence", "data_policy", "preprocessing",
            "face_detection_alignment", "local_verification", "artifacts",
            "adapter_resources", "quantized_parity", "benchmark_fixture", "provenance",
        ], context: "manifest")

        // Testing status, including the two nested test-result records.
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

        // Privacy, preprocessing and face-detection sections are flat objects.
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

        // Verification rules, including each threshold object in the array.
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

        // Each model entry, its input/output tensor descriptions, and its
        // quantisation record.
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

        // Each supporting data file.
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

        // The remaining three sections are flat objects.
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

    /// Unwraps a JSON value as an object or throws naming where it was found.
    private static func requireObject(_ value: Any?, context: String) throws -> [String: Any] {
        guard let object = value as? [String: Any] else {
            throw FACADEProtectionError.modelUnavailable("\(context) must be an object")
        }
        return object
    }

    /// Unwraps a JSON value as an array or throws naming where it was found.
    private static func requireArray(_ value: Any?, context: String) throws -> [Any] {
        guard let array = value as? [Any] else {
            throw FACADEProtectionError.modelUnavailable("\(context) must be an array")
        }
        return array
    }

    /// Throws unless the object's key set equals `expected`; the message lists
    /// what was missing and what was unexpected.
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
                "\(context) keys do not match the manifest schema; missing=[\(missing)] unexpected=[\(unexpected)]"
            )
        }
    }

    /// A "pending" result must carry no identifier or digest; a decided one
    /// must carry both.
    private static func evidenceReferenceIsValid(_ reference: FACADEEvidenceReference) -> Bool {
        switch reference.status {
        case .pending:
            return reference.artifactIdentifier == nil && reference.sha256 == nil
        case .passed, .failed:
            return !(reference.artifactIdentifier ?? "").isEmpty &&
                reference.sha256.map(isNonPlaceholderSHA256) == true
        }
    }

    /// Requires exactly the six named checks (two recognisers x three save
    /// formats), each with a known comparison and finite, non-negative margin.
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

    /// A tensor description needs a name, a positive shape, a known element
    /// type and a sensible [min, max] range.
    private static func tensorIsValid(_ tensor: FACADETensorContract) -> Bool {
        !tensor.name.isEmpty && !tensor.shape.isEmpty && tensor.shape.allSatisfy({ $0 > 0 }) &&
            ["float16", "float32", "int32", "image"].contains(tensor.dataType) &&
            !tensor.layout.isEmpty && tensor.valueRange.count == 2 &&
            tensor.valueRange.allSatisfy(\.isFinite) && tensor.valueRange[0] <= tensor.valueRange[1]
    }

    /// Accepts ISO-8601 timestamps with or without fractional seconds.
    private static func isISO8601Date(_ value: String) -> Bool {
        let standard = ISO8601DateFormatter()
        if standard.date(from: value) != nil {
            return true
        }
        let fractional = ISO8601DateFormatter()
        fractional.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return fractional.date(from: value) != nil
    }

    /// True for strings of the form "18.0" (two numeric parts).
    private static func isMajorMinorVersion(_ value: String) -> Bool {
        let pieces = value.split(separator: ".", omittingEmptySubsequences: false)
        return pieces.count == 2 && pieces.allSatisfy { !$0.isEmpty && $0.allSatisfy(\.isNumber) }
    }

    /// Compares the running OS version with the manifest's minimum.
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

    /// A path from the manifest may not be absolute, may not carry a URL
    /// scheme, and may not contain ".", ".." or empty components. This stops a
    /// manifest from pointing outside its own bundle.
    private static func isSafeRelativePath(_ value: String) -> Bool {
        guard !value.isEmpty, !value.hasPrefix("/"), URL(string: value)?.scheme == nil else {
            return false
        }
        let components = value.split(separator: "/", omittingEmptySubsequences: false)
        return !components.contains("..") && !components.contains(".") && !components.contains("")
    }

    /// Resolves a manifest path under the bundle root and proves the result
    /// stays inside it, checking each path component for symbolic links on
    /// the way down (a link anywhere in the chain could redirect elsewhere).
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

    /// SHA-256 of a file, or of a directory such as an `.mlpackage`.
    ///
    /// A directory is hashed in a way that does not depend on file-system
    /// enumeration order: every regular file inside it is collected, sorted by
    /// path, and then for each file the hasher absorbs
    ///
    ///     <relative path bytes> 0x00 <file bytes> 0x00
    ///
    /// Hidden files are skipped and symbolic links are refused. This is the
    /// rule the manifest calls `facade_sorted_tree_v1_or_file_bytes`; the
    /// tool that produced the manifest applies the same rule, so the digests
    /// agree across machines.
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

    /// 64 lowercase hex characters.
    private static func isSHA256(_ value: String) -> Bool {
        value.utf8.count == 64 && value.utf8.allSatisfy { byte in
            (48...57).contains(byte) || (97...102).contains(byte)
        }
    }

    /// A real digest: well formed and not the all-zero placeholder a tool
    /// might emit before a value is known.
    private static func isNonPlaceholderSHA256(_ value: String) -> Bool {
        isSHA256(value) && value != String(repeating: "0", count: 64)
    }
}
