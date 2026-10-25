// ProtectionMethod.swift
//
// The two protection methods a user can choose on the Protect screen.
//
// Where it sits: LivePhotoModel stores the chosen case; the Protect screen
// shows one radio-style row per case; the model picks the engine to run.
//
// Key idea: this is only a label. The actual work lives in LivePhotoEngine
// (for `.facade`) and MiniPhotoGuardEngine (for `.miniPhotoGuard`).
import Foundation

/// `facade` runs the FACADE engine, aimed at face-swap tools.
/// `miniPhotoGuard` runs the small PhotoGuard encoder attack, aimed at AI
/// clothing edits (still a prototype).
/// The raw string value is only used as a stable identifier for SwiftUI lists.
enum ProtectionMethod: String, CaseIterable, Identifiable, Sendable {
    case facade, miniPhotoGuard
    /// Identifier for `ForEach`; the raw value is unique per case.
    var id: String { rawValue }
    /// The name shown in the user interface for each method.
    var title: String { self == .facade ? "FACADE" : "Mini PhotoGuard" }
}
