// LocalDataManagement.swift
//
// The "Local Data" sheet: explains what the app stores on the phone and offers
// a button that clears leftover temporary files.
//
// Where it sits: opened from the "Local data controls" button inside the
// "About this engine" section of the Protect screen.
//
// Key ideas:
// 1. A FACADE run writes a small folder of timing records into the app's
//    Caches directory (named FACADE-LIVE-SCALARS-<uuid>) and deletes it when
//    the run ends. Only a run interrupted by the app being killed leaves one
//    behind. No photo pixels are ever written there.
// 2. The app has add-only Photos permission, so nothing here can touch the
//    user's photo library.
// 3. Deleting the app removes its whole private container anyway.
#if os(iOS)
import Combine
import Foundation
import SwiftUI

/// State for the sheet: just a status line that reports what the last clear did.
@MainActor
public final class FACADELocalDataViewModel: ObservableObject {
    /// Shown under the explanatory text; updated after each clear attempt.
    @Published public private(set) var status =
        "Nothing has been cleared yet. Removing the app always removes its private container."

    public init() {}

    /// Removes the temporary working folders a protection run keeps in the app's
    /// Caches folder. They are normally deleted when the run ends and are left
    /// behind only if the app was closed mid-run. Photos are never touched.
    public func clearTemporaryFiles() {
        do {
            let caches = try FileManager.default.url(
                for: .cachesDirectory,
                in: .userDomainMask,
                appropriateFor: nil,
                create: true
            )
            var removed = 0
            for item in try FileManager.default.contentsOfDirectory(
                at: caches,
                includingPropertiesForKeys: nil
            // Only folders with the engine's own prefix are removed; anything else in
            // Caches (for example the compiled Mini PhotoGuard model) is left alone.
            ) where item.lastPathComponent.hasPrefix("FACADE-LIVE-SCALARS-") {
                try FileManager.default.removeItem(at: item)
                removed += 1
            }
            status = removed == 0
                ? "There were no temporary files to clear. Photos were not edited or deleted."
                : "Cleared FACADE's temporary files. Photos were not edited or deleted."
        } catch {
            status = "Clearing temporary files failed: \(error.localizedDescription). Deleting the app still removes its complete private container."
        }
    }
}

/// The sheet itself: three explanatory sections and a confirmed clear button.
public struct FACADELocalDataScreen: View {
    @StateObject private var model = FACADELocalDataViewModel()
    @State private var showingConfirmation = false

    public init() {}

    /// A standard grouped form with a confirmation dialog before clearing.
    public var body: some View {
        NavigationStack {
            Form {
                Section("Stored locally") {
                    Text("While it protects a photo, FACADE keeps temporary working files inside this app's private container and removes them when it finishes. The app has no delete or edit permission for Photos.")
                    Text(model.status).font(.caption)
                }
                Section("Cleanup") {
                    // The destructive role shows the button in red; the actual deletion only
                    // happens after the confirmation below.
                    Button("Clear temporary files", role: .destructive) {
                        showingConfirmation = true
                    }
                }
                Section("Complete removal") {
                    Text("To remove the app and its entire private container, press and hold its Home Screen icon, choose Remove App, then Delete App. Originals and copies already saved in Photos remain under the phone owner's control.")
                }
            }
            .navigationTitle("Local Data")
        }
        .confirmationDialog(
            "Clear FACADE's temporary files?",
            isPresented: $showingConfirmation,
            titleVisibility: .visible
        ) {
            Button("Clear temporary files", role: .destructive) {
                model.clearTemporaryFiles()
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("This does not delete or modify any photo in the Photos library.")
        }
    }
}
#endif
