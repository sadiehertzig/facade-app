// LivePhotoPreparationTest.swift
//
// Tests the first part of the FACADE engine: turning a photo file into the
// 1024 x 1024 working image, for all eight EXIF orientations. It also checks
// that a run which fails leaves the engine's shared state clean.
//
// Where it sits: run_host_tests.sh compiles this file with
// LivePhotoPreparationStubs.swift and the real App/LivePhotoEngine.swift,
// App/NativeImageIO.swift and App/NativeSchedule.swift, then runs it on the Mac.
//
// Key ideas:
// 1. `LivePhotoEngine.preparationSelfCheck()` builds synthetic four-colour
//    images, saves them with each orientation flag, prepares them, and checks
//    sizes, placement and the colour in each corner. 2 shapes x 8 orientations
//    = 16 cases, with 8 checks each = 128 checks.
// 2. The engine is then run twice with a stub pipeline that refuses to load
//    models. Each run must fail with the stub's message and must put back
//    every shared setting it changed, so the next run starts clean.
// 3. No model is loaded and no protection result is measured.
import Foundation

/// The test program; `main` is its entry point.
@main struct LivePhotoPreparationTest {
    /// Runs the preparation self-check, then the two failing runs.
    static func main() async throws {
        let report = try LivePhotoEngine.preparationSelfCheck()
        // Also check the counts, so a self-check that silently skipped cases fails.
        guard report.passed, report.assertions == 128, report.caseCount == 16 else {
            throw NativeError.invalid("Unexpected preparation check denominator")
        }
        // A plain mid-gray 256 x 256 test image.
        let png = try RGBImage(width: 256, height: 256,
            bytes: [UInt8](repeating: 127, count: 256 * 256 * 3)).png()
        // Remember the shared settings the engine is allowed to change during a run.
        let priorCoarse = NativeSchedule.halfSteps, priorFine = NativeSchedule.fineSteps
        let priorPolicy = NativeViewGradientPolicy.arm
        // Twice, to prove the first failure did not leave the engine stuck.
        for _ in 0..<2 {
            do {
                _ = try await LivePhotoEngine.shared.run(data: png) { _ in }
                throw NativeError.invalid("The model-free stub unexpectedly returned a result")
            } catch {
                // Any other error means the engine failed somewhere unexpected.
                guard error.localizedDescription == ModelFreeStub.refusal else { throw error }
            }
            // After the failure, the profiler state must be reset and the step
            // schedule and gradient setting must be back to their earlier values.
            guard Profile.destination == nil, Profile.deadline == nil, Profile.stopped == nil,
                  Profile.observers.isEmpty, Profile.events.isEmpty, Profile.eventsClosed,
                  Profile.captures.isEmpty, NativeViewGradientPolicy.rows.isEmpty,
                  NativeSchedule.halfSteps == priorCoarse, NativeSchedule.fineSteps == priorFine,
                  NativeViewGradientPolicy.arm == priorPolicy else {
                throw NativeError.invalid("The live worker did not restore state after failure")
            }
        }
        // Both runs must have reached the pipeline; if the second was refused, the
        // engine's "busy" flag was not cleared after the first failure.
        guard ModelFreeStub.constructorCalls == 2 else { throw NativeError.invalid("Repeat was not admitted") }
        // Print a short JSON summary of what was checked.
        let result: [String: Any] = [
            "passed": true, "preparationAssertions": report.assertions,
            "preparationCases": report.caseCount, "repeatedFailureCleanupCases": 2,
            "stubConstructorRefusals": ModelFreeStub.constructorCalls,
            "actualModelsLoaded": 0, "rendererCalls": 0, "deviceCalls": 0,
            "scope": "Actual unchanged LivePhotoEngine.swift and NativeImageIO.swift. Model-free NativePipeline stub refuses at construction."
        ]
        let json = try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys])
        print(String(decoding: json, as: UTF8.self))
    }
}
