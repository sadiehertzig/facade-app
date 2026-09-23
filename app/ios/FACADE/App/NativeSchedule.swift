import Foundation

/// Named experimental schedules. "halfSteps" remains the legacy coarse boundary;
/// a shorter fine phase must never move that boundary from step 10.
enum NativeStepSchedule: String, CaseIterable, Codable {
    case control = "10+10"
    case shorter = "10+6"
    var coarseSteps: Int { 10 }
    var fineSteps: Int { self == .control ? 10 : 6 }
    var totalSteps: Int { coarseSteps + fineSteps }
    var traceRows: Int { 6 * (totalSteps + 1) }
    var policyRows: Int { 5 * (totalSteps + 1) }
    var cadenceRequests: Int { 12 * (totalSteps + 1) }
    var cadenceRefreshesK2: Int { 12 * (coarseSteps + 1 + fineSteps / 2) }
    var cadenceReusesK2: Int { cadenceRequests - cadenceRefreshesK2 }
}

// Set once before model execution. A nil fine override retains every legacy
// symmetric schedule, including old command-line clients that set halfSteps.
enum NativeSchedule {
    static var halfSteps = 25
    static var fineSteps: Int?
    static var steps: Int { halfSteps + (fineSteps ?? halfSteps) }

    static func configure(_ schedule: NativeStepSchedule) {
        halfSteps = schedule.coarseSteps
        fineSteps = schedule.fineSteps
    }

    /// Qualification checks may exercise a schedule, but must restore both
    /// values on success and failure before the next operation uses them.
    static func withConfiguration<T>(_ schedule: NativeStepSchedule, _ body: () throws -> T) rethrows -> T {
        let oldCoarse = halfSteps, oldFine = fineSteps
        defer { halfSteps = oldCoarse; fineSteps = oldFine }
        configure(schedule)
        return try body()
    }
}
