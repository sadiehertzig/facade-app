import Foundation

@main struct LivePhotoPreparationTest {
    static func main() async throws {
        let report = try LivePhotoEngine.preparationSelfCheck()
        guard report.passed, report.assertions == 128, report.caseCount == 16 else {
            throw NativeError.invalid("Unexpected preparation check denominator")
        }
        let png = try RGBImage(width: 256, height: 256,
            bytes: [UInt8](repeating: 127, count: 256 * 256 * 3)).png()
        let priorCoarse = NativeSchedule.halfSteps, priorFine = NativeSchedule.fineSteps
        let priorPolicy = NativeViewGradientPolicy.arm
        for _ in 0..<2 {
            do {
                _ = try await LivePhotoEngine.shared.run(data: png) { _ in }
                throw NativeError.invalid("The model-free stub unexpectedly returned a result")
            } catch {
                guard error.localizedDescription == ModelFreeStub.refusal else { throw error }
            }
            guard Profile.destination == nil, Profile.deadline == nil, Profile.stopped == nil,
                  Profile.observers.isEmpty, Profile.events.isEmpty, Profile.eventsClosed,
                  Profile.captures.isEmpty, NativeViewGradientPolicy.rows.isEmpty,
                  NativeSchedule.halfSteps == priorCoarse, NativeSchedule.fineSteps == priorFine,
                  NativeViewGradientPolicy.arm == priorPolicy else {
                throw NativeError.invalid("The live worker did not restore state after failure")
            }
        }
        guard ModelFreeStub.constructorCalls == 2 else { throw NativeError.invalid("Repeat was not admitted") }
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
