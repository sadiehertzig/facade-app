// LivePhotoPreparationStubs.swift
//
// Stand-ins ("stubs") for the parts of the FACADE engine that
// LivePhotoPreparationTest does not want to compile or run: the C/C++
// network, the profiler, and the real NativePipeline.
//
// Where it sits: run_host_tests.sh compiles this file with the real
// App/LivePhotoEngine.swift, App/NativeImageIO.swift and
// App/NativeSchedule.swift, so the test exercises the unchanged photo
// preparation code while everything that needs models is replaced.
//
// Key ideas:
// 1. Each stub has the same name and shape as the real type it replaces
//    (for example RGBImage, Profile, NativePipeline), so LivePhotoEngine.swift
//    compiles against it without changes.
// 2. The stub NativePipeline always throws in its initializer, so no model can
//    ever be loaded or run. The test then checks that the engine cleaned up its
//    shared state after that failure.
import Foundation
import CoreML
import CNative

/// Same error type as the real engine (NativeTensor.swift), so error
/// messages read the same way.
enum NativeError: Error, LocalizedError {
    case invalid(String)
    var errorDescription: String? { if case .invalid(let message) = self { return message }; return nil }
}
/// A minimal copy of the real RGBImage (NativeTensor.swift): 8-bit RGB
/// pixels with the same 2048-pixel size limit. NativeImageIO.swift adds `png()`.
struct RGBImage {
    let width: Int, height: Int
    var bytes: [UInt8]
    init(width: Int, height: Int, bytes: [UInt8]) throws {
        guard width > 0, height > 0, width <= 2048, height <= 2048,
              bytes.count == width * height * 3 else { throw NativeError.invalid("Invalid RGB image") }
        self.width = width; self.height = height; self.bytes = bytes
    }
}
// The two C profiler functions the engine calls, provided here in Swift.
// `@_cdecl` exports them under the C names, so no C code has to be linked.
@_cdecl("facade_profile_now") func test_profile_now() -> UInt64 { DispatchTime.now().uptimeNanoseconds }
@_cdecl("facade_profile_enable") func test_profile_enable(_ value: Int32) {}
/// Stand-in for the real profiler state (NativeProfile.swift). The engine sets
/// these fields when a run starts and must reset them when it ends; the test
/// reads them afterwards to prove that clean-up happened.
enum Profile {
    static var enabled = false
    static var destination: URL?
    static var start: UInt64 = 0
    static var deadline: UInt64?
    static var stopped: String?
    static let lock = NSLock()
    static var events = [[String: Any]]()
    static var observers = [NSObjectProtocol]()
    static var eventsClosed = false
    static var captures = [String: [String: String]]()
    // Timing calls do nothing in the test.
    static func begin(_ name: String) -> FacadeProfileToken { FacadeProfileToken(wall: 0, cpu: 0, thread: 0) }
    static func end(_ name: String, _ token: FacadeProfileToken) {}
}
/// Stand-in for NativeViewGradientPolicy.swift: only the two fields the engine
/// sets and restores.
enum NativeViewGradientPolicy {
    static var arm = "full_views"
    static var rows = [Int]()
}
/// Placeholder result types so the stub `run` has a return type with the
/// fields LivePhotoEngine reads. They are never actually created.
struct TestCadence {
    let halfSteps: Int, totalSteps: Int, interval: Int, requests: Int, refreshes: Int, reuses: Int, shadowDetections: Int
}
struct TestPipelineOutput {
    let image: RGBImage
    let trace: [Int]
    let cadence: TestCadence
}
/// Counts how many times the engine tried to build a pipeline, and holds the
/// exact error message the stub throws.
enum ModelFreeStub {
    static var constructorCalls = 0
    static let refusal = "MODEL_FREE_STUB_NO_MODEL_EXECUTION"
}
/// The stub pipeline. Construction always fails, so the engine stops right
/// where it would start loading models.
final class NativePipeline {
    init(root: URL, inswapper: String, simswap: String, detectorComputeUnits: MLComputeUnits) throws {
        ModelFreeStub.constructorCalls += 1
        throw NativeError.invalid(ModelFreeStub.refusal)
    }
    /// Never reached, because construction already failed; it throws anyway.
    func run(data: Data, clean: RGBImage, cadenceInterval: Int,
             progress: @escaping (String, Int) -> Void) async throws -> TestPipelineOutput {
        throw NativeError.invalid(ModelFreeStub.refusal)
    }
}
