import Dispatch
import Foundation

// Fixed-slot parallel map. Slot i always holds body(i), so every caller below
// still consumes results in declaration order and no reducer sees a new order.
// Each body must touch only its own inputs; the native networks, OpenCV helpers
// and geometry kernels used here hold no shared mutable state.
func facadeParallelMap<T>(_ count: Int, _ body: (Int) throws -> T) throws -> [T] {
    guard count > 1 else { return try (0..<count).map(body) }
    let slots = UnsafeMutablePointer<Result<T, Error>?>.allocate(capacity: count)
    slots.initialize(repeating: nil, count: count)
    defer { slots.deinitialize(count: count); slots.deallocate() }
    withoutActuallyEscaping(body) { escaping in
        DispatchQueue.concurrentPerform(iterations: count) { i in
            do { slots[i] = .success(try escaping(i)) } catch { slots[i] = .failure(error) }
        }
    }
    var out = [T](); out.reserveCapacity(count)
    for i in 0..<count {
        guard let slot = slots[i] else { throw NativeError.invalid("Parallel branch produced no result") }
        out.append(try slot.get())
    }
    return out
}
