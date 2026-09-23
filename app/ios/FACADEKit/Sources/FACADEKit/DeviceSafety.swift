import Foundation

public enum FACADEDeviceThermalState: String, Codable, Sendable, CaseIterable {
    case nominal
    case fair
    case serious
    case critical
    case unknown

    public var requiresCooldown: Bool {
        self == .serious || self == .critical || self == .unknown
    }
}

/// A small, dependency-free safety gate shared by the demo engine and the
/// benchmark harness. Intensive work fails closed when iOS reports serious or
/// critical thermal pressure. Unknown future states also fail closed.
public struct FACADEDeviceSafetyGate: Sendable {
    private let thermalStateProvider: @Sendable () -> FACADEDeviceThermalState

    public init() {
        self.thermalStateProvider = { Self.systemThermalState }
    }

    init(
        thermalStateProvider: @escaping @Sendable () -> FACADEDeviceThermalState
    ) {
        self.thermalStateProvider = thermalStateProvider
    }

    public var currentThermalState: FACADEDeviceThermalState {
        thermalStateProvider()
    }

    public func requireSafeForIntensiveWork(_ context: String) throws {
        let state = currentThermalState
        guard !state.requiresCooldown else {
            throw FACADEProtectionError.deviceSafetyBlocked(
                "\(context) stopped because the device thermal state is \(state.rawValue). " +
                    "No protected copy was issued; let the phone cool naturally before retrying."
            )
        }
    }

    public static var systemThermalState: FACADEDeviceThermalState {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: .nominal
        case .fair: .fair
        case .serious: .serious
        case .critical: .critical
        @unknown default: .unknown
        }
    }
}
