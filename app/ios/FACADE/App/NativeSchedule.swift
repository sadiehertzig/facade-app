// NativeSchedule.swift
//
// How many optimisation steps the FACADE engine takes, split into a "coarse"
// phase (bigger pixel moves) and a "fine" phase (ten times smaller moves).
//
// Where it sits: NativeOptimizer and NativePipeline read `NativeSchedule.steps`
// and `NativeSchedule.halfSteps` in their loops. LivePhotoEngine sets the
// schedule to `.shorter` (10 coarse + 6 fine) before every run and restores
// the previous values afterwards.
//
// Key ideas:
// 1. The schedule is process-wide state, not a parameter, because the C and
//    Swift engine code was written around one shared setting.
// 2. The coarse phase is always 10 steps. Only the fine phase length differs
//    between the two named schedules.
// 3. The derived counts (trace rows, policy rows, cadence requests) are the
//    exact numbers of records a complete run must produce; LivePhotoEngine
//    checks them after each run to confirm nothing was skipped.
import Foundation

/// The two named step schedules. "10+10" was the original measured schedule;
/// "10+6" is the one the app ships with. The coarse phase is fixed at 10 steps
/// and a shorter fine phase must never move that boundary.
enum NativeStepSchedule: String, CaseIterable, Codable {
    /// 10 coarse + 10 fine steps.
    case control = "10+10"
    /// 10 coarse + 6 fine steps (what the app runs).
    case shorter = "10+6"
    /// Steps taken with the larger move size (1/255 per step).
    var coarseSteps: Int { 10 }
    /// Steps taken with the smaller move size (0.1/255 per step).
    var fineSteps: Int { self == .control ? 10 : 6 }
    /// Total optimisation steps; each stage evaluates steps 0...totalSteps.
    var totalSteps: Int { coarseSteps + fineSteps }
    /// Expected number of optimisation-trace rows: 6 stages x (steps + 1).
    var traceRows: Int { 6 * (totalSteps + 1) }
    /// Expected number of gradient-policy rows: 5 stages x (steps + 1).
    var policyRows: Int { 5 * (totalSteps + 1) }
    /// Face-alignment requests: 4 cached stages x 3 upload conditions x (steps + 1).
    var cadenceRequests: Int { 12 * (totalSteps + 1) }
    /// Alignment refreshes with the every-second-step ("K2") cadence: every coarse
    /// step plus the final step, and every other fine step, per condition.
    var cadenceRefreshesK2: Int { 12 * (coarseSteps + 1 + fineSteps / 2) }
    /// The remaining requests reuse a cached alignment.
    var cadenceReusesK2: Int { cadenceRequests - cadenceRefreshesK2 }
}

/// The process-wide schedule in effect. Set once before model execution.
/// A nil `fineSteps` means "same as halfSteps" (the older symmetric schedules,
/// including command-line tools that only set halfSteps).
enum NativeSchedule {
    /// Coarse steps. The optimisers compare the step index against this to pick
    /// the move size.
    static var halfSteps = 25
    /// Fine steps, or nil for a symmetric schedule.
    static var fineSteps: Int?
    /// Total steps = coarse + fine.
    static var steps: Int { halfSteps + (fineSteps ?? halfSteps) }

    /// Switch to a named schedule.
    static func configure(_ schedule: NativeStepSchedule) {
        halfSteps = schedule.coarseSteps
        fineSteps = schedule.fineSteps
    }

    /// Runs `body` under a schedule and then restores both values, whether the
    /// body succeeds or throws, so the next operation sees the previous schedule.
    static func withConfiguration<T>(_ schedule: NativeStepSchedule, _ body: () throws -> T) rethrows -> T {
        let oldCoarse = halfSteps, oldFine = fineSteps
        defer { halfSteps = oldCoarse; fineSteps = oldFine }
        configure(schedule)
        return try body()
    }
}
