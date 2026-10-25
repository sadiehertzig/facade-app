// NativeAlignmentCadence.swift
//
// Decides when the engine re-runs face detection during an optimisation stage
// and when it may reuse the face alignment it found a step earlier.
//
// Where it sits: NativeOptimizer asks NativeCadenceRun for one
// NativeAlignmentCadence per stage, and NativeAlignedProxies.evaluate calls
// `resolve` for each of the three upload conditions on every step.
// LivePhotoEngine checks the totals in the final report after the run.
//
// Key ideas:
// 1. Detecting the face again on every step is the safest choice, because the
//    nudged pixels can shift the landmarks slightly. It is also slow, so in
//    the fine phase (small steps, after step 10) the app re-detects only every
//    `interval` steps ("K2" = every second step) and reuses the last
//    alignment in between.
// 2. During the coarse phase and on the final step, the face is always
//    detected fresh.
// 3. The cache is strict: it only accepts requests in the exact expected order
//    (step by step, condition 0, 1, 2), and it never carries an alignment into
//    another stage or another photo. Anything unexpected throws instead of
//    silently using an old alignment.
// 4. Only these step layouts are supported: 10+10 steps with interval 1, 2
//    or 4, and 10+6 steps with interval 1 or 2. They are the configurations
//    the engine was measured with.
import Foundation

/// Thrown when the cadence is misconfigured or used out of order.
enum NativeCadenceError: Error {
    case invalid(String)
}

/// One alignment request: at which stage/step/condition, whether it was a
/// "refresh" (fresh detection) or a "reuse", and the step whose alignment was
/// used (`anchorStep`).
struct NativeCadenceEvent: Codable {
    let stage: String
    let step: Int
    let condition: Int
    let action: String
    let anchorStep: Int
}
/// Totals and the full event list for one stage.
struct NativeCadenceStageReport: Codable {
    let stage: String
    let interval: Int
    let width: Int
    let height: Int
    let requests: Int
    let refreshes: Int
    let reuses: Int
    let events: [NativeCadenceEvent]
}
/// Totals for a whole run. `shadowDetections` is always 0 in this code: no
/// extra comparison detections are run on reuse steps.
struct NativeCadenceReport: Codable {
    let schemaVersion: Int
    let mode: String
    let interval: Int
    let halfSteps: Int
    let totalSteps: Int
    let requests: Int
    let refreshes: Int
    let reuses: Int
    let shadowDetections: Int
    let stages: [NativeCadenceStageReport]
}

/// The alignment cache for one stage. Requests for the three upload
/// conditions arrive one after another; only the network work inside each
/// condition runs in parallel. Each cached value is the pair of matrices from
/// NativeFaceDetector.alignments. Clean-photo references, score-only calls
/// and the first native stage (which uses fixed alignments) never come here.
final class NativeAlignmentCadence {
    /// The four stages that re-detect the face, in the order they run.
    static let eligibleStages = ["native_realign8", "energy80", "masked_compensation", "balanced_continue"]
    /// A cached alignment and the step at which it was detected.
    private struct Anchor {
        let matrices: [[Float]]
        let step: Int
    }
    let stage: String, interval: Int, width: Int, height: Int
    let halfSteps: Int, totalSteps: Int
    /// The latest alignment for each upload condition.
    private var anchors = [Anchor?](repeating: nil, count: 3)
    /// Every request so far, in order.
    private(set) var events = [NativeCadenceEvent]()

    /// Refuses any stage name, interval or step layout it was not built for.
    init(stage: String, interval: Int, width: Int, height: Int, halfSteps: Int, totalSteps: Int) throws {
        guard Self.eligibleStages.contains(stage), [1, 2, 4].contains(interval),
              width > 0, height > 0, halfSteps == 10,
              (totalSteps == 20 || (totalSteps == 16 && [1, 2].contains(interval))) else {
            throw NativeCadenceError.invalid("Unsupported frozen cadence configuration")
        }
        self.stage = stage; self.interval = interval; self.width = width; self.height = height
        self.halfSteps = halfSteps; self.totalSteps = totalSteps
        events.reserveCapacity(3 * (totalSteps + 1))
    }
    /// Returns the alignment matrices for `condition` at `step`: either fresh ones
    /// from `fresh()` or the cached ones, following the cadence rule.
    func resolve(condition: Int, step: Int, width: Int, height: Int,
                 fresh: () throws -> [[Float]]) throws -> [[Float]] {
        guard (0..<3).contains(condition), (0...totalSteps).contains(step),
              width == self.width, height == self.height,
              // Requests must arrive in order: step 0 conditions 0,1,2, then step 1, ...
              // so the number of earlier events tells us exactly which request this must be.
              events.count == 3 * step + condition else {
            throw NativeCadenceError.invalid("Cadence stage/view order or dimensions changed")
        }
        // Fresh detection during the coarse phase (steps 0...10), on the last step,
        // and every `interval` steps in the fine phase. With 10+6 steps and interval 2
        // that is steps 0-10, 12, 14 and 16; steps 11, 13 and 15 reuse.
        let refresh = step <= halfSteps || step == totalSteps || (step - halfSteps) % interval == 0
        if refresh {
            // Do not retain a stale fallback if fresh detection or alignment fails.
            anchors[condition] = nil
            let matrices = try fresh()
            guard matrices.count == 2, matrices.allSatisfy({$0.count == 6 && $0.allSatisfy(\.isFinite)}) else {
                throw NativeCadenceError.invalid("Fresh cadence matrices are invalid")
            }
            anchors[condition] = Anchor(matrices: matrices, step: step)
            events.append(NativeCadenceEvent(stage: stage, step: step, condition: condition, action: "refresh", anchorStep: step))
            return matrices
        }
        // A reuse is allowed only from an alignment for the same condition that was
        // detected at step 10 or later and fewer than `interval` steps ago.
        guard let anchor = anchors[condition], anchor.step >= halfSteps,
              anchor.step < step, step - anchor.step < interval else {
            throw NativeCadenceError.invalid("Missing or stale fine-phase cadence anchor")
        }
        events.append(NativeCadenceEvent(stage: stage, step: step, condition: condition, action: "reuse", anchorStep: anchor.step))
        return anchor.matrices
    }
    /// Summarises this stage's requests.
    func report() -> NativeCadenceStageReport {
        let refreshes = events.filter{$0.action == "refresh"}.count
        return NativeCadenceStageReport(stage: stage, interval: interval, width: width, height: height,
            requests: events.count, refreshes: refreshes, reuses: events.count - refreshes, events: events)
    }
}

/// Collects the per-stage caches for one engine run. Each stage gets a new,
/// empty cache; nothing is shared between stages or between photos.
final class NativeCadenceRun {
    let interval: Int, halfSteps: Int, totalSteps: Int
    private var stages = [NativeAlignmentCadence]()
    init(interval: Int, halfSteps: Int, totalSteps: Int) throws {
        guard [1, 2, 4].contains(interval), halfSteps == 10,
              (totalSteps == 20 || (totalSteps == 16 && [1, 2].contains(interval))) else {
            throw NativeCadenceError.invalid("Cadence requires frozen 10+10 K1/K2/K4 or 10+6 K1/K2")
        }
        self.interval = interval; self.halfSteps = halfSteps; self.totalSteps = totalSteps
    }
    /// Creates the cache for the next stage. Stages must arrive in the order of
    /// `eligibleStages`.
    func newStage(_ stage: String, width: Int, height: Int) throws -> NativeAlignmentCadence {
        guard stages.count < NativeAlignmentCadence.eligibleStages.count,
              NativeAlignmentCadence.eligibleStages[stages.count] == stage else {
            throw NativeCadenceError.invalid("Cadence optimizer stages changed order")
        }
        let cache = try NativeAlignmentCadence(stage: stage, interval: interval, width: width,
            height: height, halfSteps: halfSteps, totalSteps: totalSteps)
        stages.append(cache)
        return cache
    }
    /// Totals for the run. Throws unless all four stages ran and each made exactly
    /// 3 x (steps + 1) requests, i.e. no step was skipped.
    func report() throws -> NativeCadenceReport {
        let rows = stages.map{$0.report()}
        guard rows.map(\.stage) == NativeAlignmentCadence.eligibleStages,
              rows.allSatisfy({$0.requests == 3 * (totalSteps + 1)}) else {
            throw NativeCadenceError.invalid("Incomplete cadence observations")
        }
        return NativeCadenceReport(schemaVersion: 1, mode: "actual_fine_phase_matrix_reuse",
            interval: interval, halfSteps: halfSteps, totalSteps: totalSteps,
            requests: rows.reduce(0){$0 + $1.requests}, refreshes: rows.reduce(0){$0 + $1.refreshes},
            reuses: rows.reduce(0){$0 + $1.reuses}, shadowDetections: 0, stages: rows)
    }
}
