import Foundation

enum ProtectionMethod: String, CaseIterable, Identifiable, Sendable {
    case facade, miniPhotoGuard
    var id: String { rawValue }
    var title: String { self == .facade ? "FACADE" : "Mini PhotoGuard" }
}
