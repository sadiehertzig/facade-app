import CryptoKit
import Foundation

public protocol FACADEProtectionEngine: Sendable {
    var engineIdentifier: String { get }

    func generateCandidate(
        for input: FACADEProtectionInput,
        policy: FACADEProtectionPolicy
    ) async throws -> FACADEProtectionCandidate
}

public protocol FACADEInstrumentedProtectionEngine: FACADEProtectionEngine {
    func generateInstrumentedCandidate(
        for input: FACADEProtectionInput,
        policy: FACADEProtectionPolicy
    ) async throws -> FACADEInstrumentedProtectionCandidate
}

public protocol FACADELocalVerifier: Sendable {
    var verifierIdentifier: String { get }

    func verify(
        original: FACADEProtectionInput,
        candidate: FACADEProtectionCandidate,
        policy: FACADEProtectionPolicy
    ) async throws -> FACADEVerificationReport
}

public struct FACADEWorkflowRunDiagnostics: Codable, Sendable, Equatable {
    public let schemaVersion: Int
    public let engine: FACADEEngineRunDiagnostics
    public let verificationDecision: FACADEVerificationDecision
    public let verificationChecks: [String: FACADECheckStatus]
    public let verifierScores: [String: Double]
    public let verifierStageLatencyMilliseconds: [String: Double]
    public let workflowStageLatencyMilliseconds: [String: Double]

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case engine
        case verificationDecision = "verification_decision"
        case verificationChecks = "verification_checks"
        case verifierScores = "verifier_scores"
        case verifierStageLatencyMilliseconds = "verifier_stage_latency_ms"
        case workflowStageLatencyMilliseconds = "workflow_stage_latency_ms"
    }

    public init(
        schemaVersion: Int = 1,
        engine: FACADEEngineRunDiagnostics,
        verificationDecision: FACADEVerificationDecision,
        verificationChecks: [String: FACADECheckStatus],
        verifierScores: [String: Double],
        verifierStageLatencyMilliseconds: [String: Double],
        workflowStageLatencyMilliseconds: [String: Double]
    ) {
        self.schemaVersion = schemaVersion
        self.engine = engine
        self.verificationDecision = verificationDecision
        self.verificationChecks = verificationChecks
        self.verifierScores = verifierScores
        self.verifierStageLatencyMilliseconds = verifierStageLatencyMilliseconds
        self.workflowStageLatencyMilliseconds = workflowStageLatencyMilliseconds
    }
}

public struct FACADEInstrumentedProtectedCopy: Sendable, Equatable {
    public let copy: FACADEVerifiedProtectedCopy
    public let diagnostics: FACADEWorkflowRunDiagnostics

    public init(
        copy: FACADEVerifiedProtectedCopy,
        diagnostics: FACADEWorkflowRunDiagnostics
    ) {
        self.copy = copy
        self.diagnostics = diagnostics
    }
}

public protocol FACADEProtectedCopySaver: Sendable {
    /// Implementations must create a new asset. Editing or replacing the source
    /// asset is outside this protocol and therefore outside the package API.
    func saveCopy(_ copy: FACADEVerifiedProtectedCopy) async throws -> FACADESaveReceipt
}

public actor FACADEProtectionWorkflow {
    private let engine: any FACADEProtectionEngine
    private let verifier: any FACADELocalVerifier

    public init(
        engine: any FACADEProtectionEngine,
        verifier: any FACADELocalVerifier
    ) {
        self.engine = engine
        self.verifier = verifier
    }

    /// Generates and verifies an in-memory candidate. Nothing is persisted by
    /// this method. Every error path fails before a save-capable value exists.
    public func prepareProtectedCopy(
        from input: FACADEProtectionInput,
        policy: FACADEProtectionPolicy
    ) async throws -> FACADEVerifiedProtectedCopy {
        let result = try await prepareProtectedCopyInternal(
            from: input,
            policy: policy,
            captureInstrumentation: false,
            requireAcceptedVerification: true
        )
        guard let copy = result.copy else {
            throw FACADEProtectionError.verificationUnavailable(
                "an accepted verification transaction did not produce a copy"
            )
        }
        return copy
    }

    /// Runs the same in-memory fail-closed transaction while retaining only
    /// scalar timing/reachability diagnostics for the synthetic P01 fixture.
    public func prepareProtectedCopyWithDiagnostics(
        from input: FACADEProtectionInput,
        policy: FACADEProtectionPolicy
    ) async throws -> FACADEInstrumentedProtectedCopy {
        let result = try await prepareProtectedCopyInternal(
            from: input,
            policy: policy,
            captureInstrumentation: true,
            requireAcceptedVerification: true
        )
        guard let copy = result.copy else {
            throw FACADEProtectionError.verificationUnavailable(
                "an accepted P01 transaction did not produce a save-capable copy"
            )
        }
        return FACADEInstrumentedProtectedCopy(
            copy: copy,
            diagnostics: try scalarDiagnostics(from: result)
        )
    }

    /// Benchmark-only scalar path. It performs generation and all independent
    /// verifier transforms but never returns candidate bytes or a save-capable
    /// value. A threshold rejection is recorded rather than discarded so P01
    /// can measure optimizer reachability without weakening the save workflow.
    public func runScalarDiagnostics(
        from input: FACADEProtectionInput,
        policy: FACADEProtectionPolicy
    ) async throws -> FACADEWorkflowRunDiagnostics {
        let result = try await prepareProtectedCopyInternal(
            from: input,
            policy: policy,
            captureInstrumentation: true,
            requireAcceptedVerification: false
        )
        return try scalarDiagnostics(from: result)
    }

    private struct PreparedCopyResult {
        let copy: FACADEVerifiedProtectedCopy?
        let report: FACADEVerificationReport
        let engineDiagnostics: FACADEEngineRunDiagnostics?
        let workflowStageLatencyMilliseconds: [String: Double]
    }

    private func prepareProtectedCopyInternal(
        from input: FACADEProtectionInput,
        policy: FACADEProtectionPolicy,
        captureInstrumentation: Bool,
        requireAcceptedVerification: Bool
    ) async throws -> PreparedCopyResult {
        let workflowStarted = DispatchTime.now().uptimeNanoseconds
        var workflowStageLatencyMilliseconds: [String: Double] = [:]
        try Task.checkCancellation()
        let inputHashStarted = DispatchTime.now().uptimeNanoseconds
        let originalHash = Self.sha256(input.originalData)
        workflowStageLatencyMilliseconds["input_hash"] = Self.milliseconds(
            since: inputHashStarted
        )

        let candidate: FACADEProtectionCandidate
        var engineDiagnostics: FACADEEngineRunDiagnostics?
        let candidateGenerationStarted = DispatchTime.now().uptimeNanoseconds
        do {
            if captureInstrumentation {
                guard let instrumentedEngine = engine as? any FACADEInstrumentedProtectionEngine else {
                    throw FACADEProtectionError.modelUnavailable(
                        "the installed engine does not implement P01 scalar diagnostics"
                    )
                }
                let result = try await instrumentedEngine.generateInstrumentedCandidate(
                    for: input,
                    policy: policy
                )
                candidate = result.candidate
                engineDiagnostics = result.diagnostics
            } else {
                candidate = try await engine.generateCandidate(for: input, policy: policy)
            }
        } catch is CancellationError {
            throw CancellationError()
        } catch let error as FACADEProtectionError {
            throw error
        } catch {
            throw FACADEProtectionError.candidateGenerationFailed(String(describing: error))
        }
        workflowStageLatencyMilliseconds["candidate_generation"] = Self.milliseconds(
            since: candidateGenerationStarted
        )
        try Task.checkCancellation()

        let candidateValidationStarted = DispatchTime.now().uptimeNanoseconds
        guard !candidate.data.isEmpty else {
            throw FACADEProtectionError.candidateInvalid("the engine returned an empty image")
        }
        guard candidate.data != input.originalData else {
            throw FACADEProtectionError.candidateInvalid("the engine returned the unmodified original")
        }
        guard !candidate.uniformTypeIdentifier.isEmpty else {
            throw FACADEProtectionError.candidateInvalid("the candidate image type is missing")
        }
        guard !candidate.modelBundleIdentifier.isEmpty else {
            throw FACADEProtectionError.candidateInvalid("the candidate model bundle identifier is missing")
        }
        guard candidate.engineIdentifier == engine.engineIdentifier else {
            throw FACADEProtectionError.candidateInvalid("the engine identifier does not match the loaded engine")
        }
        try Self.enforceEvidenceTier(candidate.evidenceTier, for: policy)
        workflowStageLatencyMilliseconds["candidate_validation"] = Self.milliseconds(
            since: candidateValidationStarted
        )

        let report: FACADEVerificationReport
        let verifierStarted = DispatchTime.now().uptimeNanoseconds
        do {
            report = try await verifier.verify(
                original: input,
                candidate: candidate,
                policy: policy
            )
        } catch is CancellationError {
            throw CancellationError()
        } catch let error as FACADEProtectionError {
            throw error
        } catch {
            throw FACADEProtectionError.verificationUnavailable(String(describing: error))
        }
        workflowStageLatencyMilliseconds["local_verification"] = Self.milliseconds(
            since: verifierStarted
        )
        try Task.checkCancellation()

        let reportValidationStarted = DispatchTime.now().uptimeNanoseconds
        guard report.verifierIdentifier == verifier.verifierIdentifier else {
            throw FACADEProtectionError.verificationRejected("the verifier identifier does not match the loaded verifier")
        }
        guard report.modelBundleIdentifier == candidate.modelBundleIdentifier else {
            throw FACADEProtectionError.verificationRejected("the engine and verifier model bundles do not match")
        }
        guard report.evidenceTier == candidate.evidenceTier else {
            throw FACADEProtectionError.verificationRejected("the engine and verifier evidence tiers do not match")
        }
        if requireAcceptedVerification, report.decision != .acceptedForSave {
            switch report.decision {
            case .unavailable:
                throw FACADEProtectionError.verificationUnavailable("local verification did not complete")
            case .rejected:
                throw FACADEProtectionError.verificationRejected("local verification rejected the candidate")
            case .acceptedForSave:
                preconditionFailure("unreachable")
            }
        }
        try Self.enforceEvidenceTier(report.evidenceTier, for: policy)

        var allRequiredChecksPassed = true
        for requiredCheck in policy.requiredChecks {
            guard let status = report.checks[requiredCheck] else {
                throw FACADEProtectionError.verificationUnavailable(
                    "required check \(requiredCheck.rawValue) was not reported"
                )
            }
            switch status {
            case .passed:
                continue
            case .failed:
                allRequiredChecksPassed = false
                if requireAcceptedVerification {
                    throw FACADEProtectionError.verificationRejected(
                        "required check \(requiredCheck.rawValue) failed"
                    )
                }
            case .unavailable:
                allRequiredChecksPassed = false
                if requireAcceptedVerification {
                    throw FACADEProtectionError.verificationUnavailable(
                        "required check \(requiredCheck.rawValue) was unavailable"
                    )
                }
            }
        }
        workflowStageLatencyMilliseconds["verification_report_validation"] = Self.milliseconds(
            since: reportValidationStarted
        )

        let finalIntegrityStarted = DispatchTime.now().uptimeNanoseconds
        guard Self.sha256(input.originalData) == originalHash else {
            throw FACADEProtectionError.originalMutationDetected
        }

        let copy: FACADEVerifiedProtectedCopy?
        if requireAcceptedVerification,
           report.decision == .acceptedForSave,
           allRequiredChecksPassed {
            copy = FACADEVerifiedProtectedCopy(
                transactionIdentifier: UUID(),
                sourceIdentifier: input.id,
                data: candidate.data,
                uniformTypeIdentifier: candidate.uniformTypeIdentifier,
                originalSHA256: originalHash,
                protectedSHA256: Self.sha256(candidate.data),
                report: report,
                experimentalLabel: policy.experimentalLabel
            )
        } else {
            copy = nil
        }
        workflowStageLatencyMilliseconds["final_integrity_and_hashing"] = Self.milliseconds(
            since: finalIntegrityStarted
        )
        workflowStageLatencyMilliseconds["total_workflow"] = Self.milliseconds(
            since: workflowStarted
        )
        return PreparedCopyResult(
            copy: copy,
            report: report,
            engineDiagnostics: engineDiagnostics,
            workflowStageLatencyMilliseconds: workflowStageLatencyMilliseconds
        )
    }

    private func scalarDiagnostics(
        from result: PreparedCopyResult
    ) throws -> FACADEWorkflowRunDiagnostics {
        guard let engineDiagnostics = result.engineDiagnostics,
              let verifierScores = result.report.scalarScores,
              let verifierStageLatencyMilliseconds =
                result.report.stageLatencyMilliseconds else {
            throw FACADEProtectionError.verificationUnavailable(
                "the P01 scalar diagnostics were not produced by the installed runtime"
            )
        }
        let requiredScoreNames = Set([
            "arcface_raw", "arcface_jpeg_q70", "arcface_heic_q80",
            "simswap_raw", "simswap_jpeg_q70", "simswap_heic_q80",
        ])
        guard Set(verifierScores.keys) == requiredScoreNames,
              verifierScores.values.allSatisfy(\.isFinite),
              verifierStageLatencyMilliseconds.values.allSatisfy({
                  $0.isFinite && $0 >= 0
              }) else {
            throw FACADEProtectionError.verificationUnavailable(
                "the P01 verifier scalar set is incomplete or invalid"
            )
        }
        return FACADEWorkflowRunDiagnostics(
            engine: engineDiagnostics,
            verificationDecision: result.report.decision,
            verificationChecks: Dictionary(
                uniqueKeysWithValues: result.report.checks.map { check, status in
                    (check.rawValue, status)
                }
            ),
            verifierScores: verifierScores,
            verifierStageLatencyMilliseconds: verifierStageLatencyMilliseconds,
            workflowStageLatencyMilliseconds: result.workflowStageLatencyMilliseconds
        )
    }

    private static func enforceEvidenceTier(
        _ tier: FACADEEvidenceTier,
        for policy: FACADEProtectionPolicy
    ) throws {
        switch (policy.requiredEvidenceTier, tier) {
        case (.experimentalProxyOnly, _),
             (.renderedAttackValidated, .renderedAttackValidated):
            return
        case (.renderedAttackValidated, .experimentalProxyOnly):
            throw FACADEProtectionError.evidenceTierRejected(
                "release mode cannot use a proxy-only model bundle"
            )
        }
    }

    private static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private static func milliseconds(since started: UInt64) -> Double {
        Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000.0
    }
}

public actor FACADEFailClosedSaveCoordinator {
    private let saver: any FACADEProtectedCopySaver

    public init(saver: any FACADEProtectedCopySaver) {
        self.saver = saver
    }

    public func save(_ copy: FACADEVerifiedProtectedCopy) async throws -> FACADESaveReceipt {
        let receipt: FACADESaveReceipt
        do {
            receipt = try await saver.saveCopy(copy)
        } catch let error as FACADEProtectionError {
            throw error
        } catch {
            throw FACADEProtectionError.saveFailed(String(describing: error))
        }

        guard receipt.transactionIdentifier == copy.transactionIdentifier,
              receipt.protectedSHA256 == copy.protectedSHA256,
              !receipt.newAssetIdentifier.isEmpty,
              receipt.originalWasModified == false else {
            throw FACADEProtectionError.saveFailed(
                "the saver did not prove creation of a new copy with the original preserved"
            )
        }
        return receipt
    }
}

public struct FACADEUnavailableProtectionEngine: FACADEProtectionEngine {
    public let engineIdentifier: String
    public let reason: String

    public init(
        engineIdentifier: String = "unavailable",
        reason: String = "No research-approved model bundle is installed."
    ) {
        self.engineIdentifier = engineIdentifier
        self.reason = reason
    }

    public func generateCandidate(
        for input: FACADEProtectionInput,
        policy: FACADEProtectionPolicy
    ) async throws -> FACADEProtectionCandidate {
        throw FACADEProtectionError.modelUnavailable(reason)
    }
}

public struct FACADEUnavailableLocalVerifier: FACADELocalVerifier {
    public let verifierIdentifier: String
    public let reason: String

    public init(
        verifierIdentifier: String = "unavailable",
        reason: String = "No research-approved local verifier is installed."
    ) {
        self.verifierIdentifier = verifierIdentifier
        self.reason = reason
    }

    public func verify(
        original: FACADEProtectionInput,
        candidate: FACADEProtectionCandidate,
        policy: FACADEProtectionPolicy
    ) async throws -> FACADEVerificationReport {
        throw FACADEProtectionError.verificationUnavailable(reason)
    }
}
