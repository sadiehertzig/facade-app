// NativeParallel.swift
//
// A tiny helper that runs a few independent pieces of work on several CPU
// cores at once and returns their results in a fixed order.
//
// Where it sits: NativeVision.swift uses it so the three "upload conditions"
// (original, JPEG, JPEG plus half-size) and the two face-recognition models
// can be evaluated in parallel during each optimisation step.
//
// Key ideas:
// 1. Result order must not depend on which thread finishes first. Slot i is
//    always the result of body(i), so every later sum or average sees the same
//    order as a plain serial loop and produces the same floating-point bits.
// 2. Each body must only touch its own inputs. The C networks, the OpenCV
//    helpers and the geometry kernels keep no shared mutable state, which is
//    what makes this safe.
// 3. An error inside any slot is stored and re-thrown after all slots finish,
//    so a thrown error never leaves other threads running.
import Dispatch
import Foundation

/// Fixed-slot parallel map: returns `[body(0), body(1), ..., body(count-1)]`.
/// Slot i always holds body(i), so callers consume results in declaration
/// order and no reducer ever sees a different order than a serial loop would.
/// Each body must touch only its own inputs; the native networks, OpenCV
/// helpers and geometry kernels used here hold no shared mutable state.
func facadeParallelMap<T>(_ count: Int, _ body: (Int) throws -> T) throws -> [T] {
    // Zero or one items: nothing to parallelise, so run it inline.
    guard count > 1 else { return try (0..<count).map(body) }
    // One optional Result per slot. Raw memory is used because the slots are
    // written from several threads at once; each thread writes only its own.
    let slots = UnsafeMutablePointer<Result<T, Error>?>.allocate(capacity: count)
    slots.initialize(repeating: nil, count: count)
    defer { slots.deinitialize(count: count); slots.deallocate() }
    // `body` is non-escaping, but concurrentPerform needs an escaping closure.
    // This wrapper promises the compiler the closure does not outlive the call.
    withoutActuallyEscaping(body) { escaping in
        DispatchQueue.concurrentPerform(iterations: count) { i in
            // Store either the value or the thrown error for slot i.
            do { slots[i] = .success(try escaping(i)) } catch { slots[i] = .failure(error) }
        }
    }
    var out = [T](); out.reserveCapacity(count)
    for i in 0..<count {
        // A nil slot would mean a thread never ran, which should be impossible.
        guard let slot = slots[i] else { throw NativeError.invalid("Parallel branch produced no result") }
        out.append(try slot.get())
    // Re-throw the first stored error, in slot order.
    }
    return out
}
