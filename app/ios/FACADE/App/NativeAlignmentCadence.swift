import Foundation

enum NativeCadenceError: Error {
    case invalid(String)
}

struct NativeCadenceEvent: Codable {
    let stage: String
    let step: Int
    let condition: Int
    let action: String
    let anchorStep: Int
}
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

/// One successful optimizer-stage invocation owns one cache. All three
/// conditions remain serial at this boundary; only proxy model work is parallel.
/// Stored values are the two Float matrices produced by the ordinary detector.
/// Clean references, score-only calls and fixed-alignment EOT never enter here.
final class NativeAlignmentCadence {
    static let eligibleStages = ["native_realign8", "energy80", "masked_compensation", "balanced_continue"]
    private struct Anchor {
        let matrices: [[Float]]
        let step: Int
    }
    let stage: String, interval: Int, width: Int, height: Int
    let halfSteps: Int, totalSteps: Int
    private var anchors = [Anchor?](repeating: nil, count: 3)
    private(set) var events = [NativeCadenceEvent]()

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
    func resolve(condition: Int, step: Int, width: Int, height: Int,
                 fresh: () throws -> [[Float]]) throws -> [[Float]] {
        guard (0..<3).contains(condition), (0...totalSteps).contains(step),
              width == self.width, height == self.height,
              events.count == 3 * step + condition else {
            throw NativeCadenceError.invalid("Cadence stage/view order or dimensions changed")
        }
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
        guard let anchor = anchors[condition], anchor.step >= halfSteps,
              anchor.step < step, step - anchor.step < interval else {
            throw NativeCadenceError.invalid("Missing or stale fine-phase cadence anchor")
        }
        events.append(NativeCadenceEvent(stage: stage, step: step, condition: condition, action: "reuse", anchorStep: anchor.step))
        return anchor.matrices
    }
    func report() -> NativeCadenceStageReport {
        let refreshes = events.filter{$0.action == "refresh"}.count
        return NativeCadenceStageReport(stage: stage, interval: interval, width: width, height: height,
            requests: events.count, refreshes: refreshes, reuses: events.count - refreshes, events: events)
    }
}

/// Run-owned recording only. Caches are created inside individual stage calls
/// and are never looked up or carried into another stage/source/pipeline run.
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
