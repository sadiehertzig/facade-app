#if os(iOS)
import Combine
import Foundation
import SwiftUI

@MainActor
public final class FACADELocalDataViewModel: ObservableObject {
    @Published public private(set) var status =
        "Nothing has been cleared yet. Removing the app always removes its private container."

    public init() {}

    /// Removes the temporary working folders a protection run keeps in the
    /// app's Caches folder (normally deleted when the run ends; left behind only
    /// if the app was closed mid-run). Photos are never touched.
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

public struct FACADELocalDataScreen: View {
    @StateObject private var model = FACADELocalDataViewModel()
    @State private var showingConfirmation = false

    public init() {}

    public var body: some View {
        NavigationStack {
            Form {
                Section("Stored locally") {
                    Text("While it protects a photo, FACADE keeps temporary working files inside this app's private container and removes them when it finishes. The app has no delete or edit permission for Photos.")
                    Text(model.status).font(.caption)
                }
                Section("Cleanup") {
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
