// Model-free test stubs for LivePhotoPreparationTest. NativePipeline can never load or run a model.
import Foundation
import CoreML
import CNative

enum NativeError: Error, LocalizedError {
    case invalid(String)
    var errorDescription: String? { if case .invalid(let message) = self { return message }; return nil }
}
struct RGBImage {
    let width: Int, height: Int
    var bytes: [UInt8]
    init(width: Int, height: Int, bytes: [UInt8]) throws {
        guard width > 0, height > 0, width <= 2048, height <= 2048,
              bytes.count == width * height * 3 else { throw NativeError.invalid("Invalid RGB image") }
        self.width = width; self.height = height; self.bytes = bytes
    }
}
@_cdecl("facade_profile_now") func test_profile_now() -> UInt64 { DispatchTime.now().uptimeNanoseconds }
@_cdecl("facade_profile_enable") func test_profile_enable(_ value: Int32) {}
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
    static func begin(_ name: String) -> FacadeProfileToken { FacadeProfileToken(wall: 0, cpu: 0, thread: 0) }
    static func end(_ name: String, _ token: FacadeProfileToken) {}
}
enum NativeViewGradientPolicy {
    static var arm = "full_views"
    static var rows = [Int]()
}
struct TestCadence {
    let halfSteps: Int, totalSteps: Int, interval: Int, requests: Int, refreshes: Int, reuses: Int, shadowDetections: Int
}
struct TestPipelineOutput {
    let image: RGBImage
    let trace: [Int]
    let cadence: TestCadence
}
enum ModelFreeStub {
    static var constructorCalls = 0
    static let refusal = "MODEL_FREE_STUB_NO_MODEL_EXECUTION"
}
final class NativePipeline {
    init(root: URL, inswapper: String, simswap: String, detectorComputeUnits: MLComputeUnits) throws {
        ModelFreeStub.constructorCalls += 1
        throw NativeError.invalid(ModelFreeStub.refusal)
    }
    func run(data: Data, clean: RGBImage, cadenceInterval: Int,
             progress: @escaping (String, Int) -> Void) async throws -> TestPipelineOutput {
        throw NativeError.invalid(ModelFreeStub.refusal)
    }
}
