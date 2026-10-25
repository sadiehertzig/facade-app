// ProtectionTypes.swift
//
// The small set of shared building blocks that every other file in FACADEKit
// relies on:
//
//   * `FACADEFloatImage` - the one in-memory picture format the package passes
//     between its stages (a planar RGB float tensor).
//   * `FACADEProtectionError` - the single error type thrown by the package.
//   * `FACADEEvidenceTier` - the label in the model manifest that says how
//     far the bundled models have been tested - and `FACADEProtectionMode`,
//     which the app passes to the loader to say how strict that test level
//     must be.
//   * Two tiny protocols (`FACADELocalVerificationImageCodec` and
//     `FACADEInitializerResidualBank`) that let the app hold the image decoder
//     and the protection model behind an interface instead of a concrete class.
//
// Where it fits: FACADEKit is the library the FACADE app uses to (1) check the
// model bundle on disk against its hash-pinned manifest, (2) decode and square the
// user's photo, (3) find and align the face, and (4) run the two Core ML models
// the protection loop needs. The app's `NativePreparationLoader` builds these
// pieces and `NativePipeline` drives them.

import Foundation

/// A picture held as 32-bit floats, one plane per colour channel.
///
/// Layout is "channel first": all red values for every pixel, then all green
/// values, then all blue values. Within a plane, pixels run row by row from the
/// top-left corner. Every value is expected to lie in the range 0...1 for a
/// real image; residuals (small adjustments added to an image) use the same
/// container but may be negative.
///
/// This is the same memory order Core ML expects for an NCHW tensor, so the
/// values can be copied straight into a model input without reshuffling.
public struct FACADEFloatImage: Sendable, Equatable {
    /// Width in pixels.
    public let width: Int
    /// Height in pixels.
    public let height: Int
    /// Exactly `3 * width * height` finite floats in C,H,W order.
    public let values: [Float]

    /// Builds an image, refusing anything whose size does not match its data
    /// or that contains NaN or infinity. Catching bad tensors here means the
    /// rest of the package can trust every image it receives.
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

/// How thoroughly the models in a bundle have been tested, as declared by the
/// manifest's `evidence.tier` field.
///
/// The bundle this app ships declares `experimental_proxy_only`: the models
/// were scored against on-device face recognisers, but a full rendered
/// face-swap test was not part of the bundle's acceptance.
public enum FACADEEvidenceTier: String, Codable, Sendable, CaseIterable {
    /// Scored only against local face-recognition models.
    case experimentalProxyOnly = "experimental_proxy_only"

    /// Also passed an end-to-end rendered face-swap test.
    case renderedAttackValidated = "rendered_attack_validated"
}

/// The mode the app asks the manifest loader to validate for.
///
/// `experimentalDemo` is what the shipped app uses. `releaseCandidate` demands
/// a stricter manifest (every validation result marked "pass"); it exists so
/// a future, more fully tested bundle can be held to a higher bar without
/// changing the loader.
public enum FACADEProtectionMode: String, Codable, Sendable {
    /// Accepts the shipped bundle's test level; used by the app.
    case experimentalDemo = "experimental_demo"
    /// Also requires every external test in the manifest to have passed.
    case releaseCandidate = "release_candidate"
}

/// The decoder that turns the user's photo bytes into the fixed-size working
/// frame the models consume.
///
/// The app stores the decoder as `any FACADELocalVerificationImageCodec` so
/// the concrete Core Image implementation (`FACADECoreImagePNGCodec`) could be
/// swapped for another without touching the pipeline.
public protocol FACADELocalVerificationImageCodec: Sendable {
    /// Decodes an image file (PNG, JPEG, HEIC...) and returns it as a square
    /// RGB float image, letter-boxed onto neutral grey if the photo was not
    /// already square.
    func decodeAndPrepare(_ data: Data) throws -> FACADEFloatImage
}

/// The protection model: given the prepared photo, it proposes a small set of
/// starting residuals (candidate perturbations) that the optimisation loop in
/// the app then refines.
///
/// "Bank" refers to the model producing several alternative starting points
/// in one pass; the app currently keeps the first one.
public protocol FACADEInitializerResidualBank: Sendable {
    /// Runs the model once and returns every proposed residual as its own
    /// image, in the order the model emits them.
    func residuals(for image: FACADEFloatImage) async throws -> [FACADEFloatImage]
}

/// Every failure FACADEKit can report.
///
/// The associated string is a human-readable explanation that the app shows
/// or logs. Grouping all failures under one type lets the app treat "the
/// protection feature is unavailable" uniformly, whatever the root cause.
public enum FACADEProtectionError: Error, Sendable, Equatable {
    /// The caller passed data the package cannot work with (empty image,
    /// wrong tensor size, undecodable bytes).
    case invalidInput(String)
    /// The model bundle is missing, tampered with, or does not match what
    /// this app was built to run. Nothing is loaded in that case.
    case modelUnavailable(String)
    /// A model ran but produced something unusable (wrong shape, NaN...).
    case candidateGenerationFailed(String)
    /// A system service the package depends on (Vision, Core ML) failed.
    case verificationUnavailable(String)
    /// The photo itself was judged unsuitable, for example because it does not
    /// contain exactly one face.
    case verificationRejected(String)
    /// The manifest's validation level is too low for the requested mode.
    case evidenceTierRejected(String)
}

/// Lets the explanation string show up through Swift's standard error description.
extension FACADEProtectionError: LocalizedError {
    /// Surfaces the explanation string through the standard `localizedDescription`.
    public var errorDescription: String? {
        switch self {
        case .invalidInput(let message),
             .modelUnavailable(let message),
             .candidateGenerationFailed(let message),
             .verificationUnavailable(let message),
             .verificationRejected(let message),
             .evidenceTierRejected(let message):
            return message
        }
    }
}
