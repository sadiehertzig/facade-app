import SwiftUI

@main struct FACADEK2PhoneApp: App {
    var body: some Scene {
        WindowGroup {
            TabView {
                FACADEDemoScreen(model: LivePhotoModel())
                    .tabItem { Label("Protect", systemImage: "shield.lefthalf.filled") }
                HelpScreen()
                    .tabItem { Label("Get Help", systemImage: "heart.text.square") }
            }
            .tint(Color.facadeViolet)
            .preferredColorScheme(.light)
        }
    }
}
