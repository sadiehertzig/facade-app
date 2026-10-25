// PhoneApp.swift
//
// The app's entry point. iOS starts here and builds the whole user interface
// from this one scene: a tab bar with two tabs.
//
// Where it sits: this is the top of the view tree. Everything else in App/ is
// reached from one of the two tabs.
//
// Key ideas:
// 1. The Protect tab (FACADEDemoScreen) is driven by one LivePhotoModel, which
//    holds the chosen photo, the protected copy and the current progress.
// 2. The Get Help tab (HelpScreen) is a static list of official help links.
// 3. The app is deliberately light-mode only and tinted with the navy brand
//    colour defined in DemoAppearance.swift.
import SwiftUI

/// The SwiftUI `App`. `@main` tells Swift to start the program from here.
/// Note that the type name still says "K2" for historical reasons; the product
/// name shown to users is just FACADE.
@main struct FACADEK2PhoneApp: App {
    /// One window containing a two-tab bar.
    var body: some Scene {
        WindowGroup {
            TabView {
                // Tab 1: pick or take a photo and make a protected copy. The model is created
                // once here and owned by the screen as a @StateObject.
                FACADEDemoScreen(model: LivePhotoModel())
                    .tabItem { Label("Protect", systemImage: "shield.lefthalf.filled") }
                // Tab 2: links to official services for people whose images were misused.
                HelpScreen()
                    .tabItem { Label("Get Help", systemImage: "heart.text.square") }
            }
            // Accent colour for selected tabs and controls.
            .tint(Color.facadeNavy)
            // The colour tokens were only designed and checked for light mode.
            .preferredColorScheme(.light)
        }
    }
}
