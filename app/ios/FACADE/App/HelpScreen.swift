// HelpScreen.swift
//
// The Get Help tab. It is for people whose images have been misused, or who
// are being threatened with them, and it points them to official services.
//
// Where it sits: PhoneApp.swift shows this as the second tab. The services
// themselves (names and fixed web addresses) live in HelpResources.swift; this
// file decides which ones to show for which situation and how links open.
//
// Key ideas:
// 1. The user picks the situation that fits ("I was under 18 in the image",
//    "Someone is threatening me", ...) and only then sees the two or so
//    services that apply, instead of a long list of hotlines.
// 2. FACADE never sends anything to these services. Each button opens a fixed
//    public address; no photo, name or other data goes with it.
// 3. Links open in the user's browser. Only if iOS refuses to open the browser
//    does the app show the page in an in-app Safari sheet (HelpBrowser).
import SwiftUI
import UIKit
import SafariServices

/// One situation on the Get Help screen and the fixed resources it points to.
/// `Identifiable` lets SwiftUI's `ForEach` tell the situations apart by `id`.
private struct HelpSituation: Identifiable {
    /// A short stable key (for example "minor"), also used to remember which
    /// situation is expanded.
    let id: String
    /// SF Symbols icon name, heading text, and one supportive sentence shown when
    /// the situation is opened.
    let icon: String
    let title: String
    let reassurance: String
    /// The services to show, each paired with a sentence explaining why it fits
    /// this particular situation.
    let resources: [(FACADEHelpResource, String)]
}

/// The four situations, in the order they appear on screen. The same service
/// can appear under more than one situation with a different explanation
/// (for example, the CyberTipline for minors and for threats).
private let helpSituations: [HelpSituation] = [
    HelpSituation(id: "minor", icon: "person.fill", title: "I was under 18 in the image",
        reassurance: "You can get help even if you're an adult now, and even if the image is AI-made.",
        resources: [(.takeItDown, "Free help getting the image removed from participating sites. The image stays on your phone."),
                    (.cyberTipline, "Report threats or exploitation. You don't need to have the image.")]),
    HelpSituation(id: "adult", icon: "person.fill.checkmark", title: "I'm an adult in the image",
        reassurance: "This includes deepfakes and AI-edited images. You have options.",
        resources: [(.stopNCII, "Free help stopping the image from being shared on participating platforms. It never leaves your phone."),
                    (.takeItDownAct, "Your rights under the TAKE IT DOWN Act, including how to ask a site to remove it.")]),
    HelpSituation(id: "threat", icon: "exclamationmark.bubble.fill", title: "Someone is threatening or blackmailing me",
        reassurance: "It's not your fault. You don't have to pay or do what they ask to get help.",
        resources: [(.cyberTipline, "If you're under 18, report it here, even if you don't have the image."),
                    (.stopNCII, "If you're 18 or older, help stop an image from being shared.")]),
    HelpSituation(id: "site", icon: "globe", title: "A website won't take it down",
        reassurance: "Covered platforms must remove qualifying images within 48 hours of a valid request.",
        resources: [(.takeItDownAct, "The FTC explains how to make a request and report a platform that doesn't comply.")])
]

/// The Get Help screen: a heading, the emergency banner, the expandable
/// situations, a "contact your representative" card, and a note on how the
/// links work.
struct HelpScreen: View {
    @Environment(\.openURL) private var openURL
    /// The service shown in the in-app Safari sheet; nil means no sheet. It is
    /// only set when opening the browser failed.
    @State private var browserResource: FACADEHelpResource?
    /// The `id` of the situation currently expanded. Only one is open at a time.
    @State private var openSituation: String?

    /// One scrolling column, at most 680 points wide so lines stay readable on
    /// wide screens, where it is centred.
    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Get help").font(.largeTitle.weight(.bold)).foregroundStyle(Color.facadeText)
                        Text("What happened is not your fault.").font(.title3.weight(.semibold)).foregroundStyle(Color.facadeText)
                        Text("You don't have to handle this alone. Free, confidential help is available, and you never need to show the image to FACADE.")
                            .foregroundStyle(Color.facadeMuted).fixedSize(horizontal: false, vertical: true)
                    }
                    // Calm and high-contrast. Red would read as "something failed".
                    StatusBanner(kind: .urgentHelp,
                                 message: "If you are in immediate danger, call 911. For emotional support at any time, call or text 988.")
                    VStack(alignment: .leading, spacing: 10) {
                        Text("What's happening?").font(.headline).foregroundStyle(Color.facadeText)
                        Text("Choose what fits. We'll show you where to start.")
                            .font(.subheadline).foregroundStyle(Color.facadeMuted)
                        ForEach(helpSituations) { situation($0) }
                    }
                    // Below the situations: contacting a member of Congress, then the
                    // collapsible note about privacy and limits.
                    congressCard
                    DisclosureGroup("How these links work") {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Each link opens an official website in your browser. No photo, address or personal information is sent by FACADE. Links need Internet access.")
                            Text("Don't download, forward or ask for an intimate image to use these services. If you don't have it, the CyberTipline can still help.")
                            Text("Fingerprint (hash) services work with participating platforms and can't guarantee removal everywhere. FACADE can't remove images already online.")
                            Text("Information checked September 22, 2026.").font(.caption)
                        }.font(.footnote).foregroundStyle(Color.facadeMuted).padding(.top, 8)
                    }.font(.subheadline.weight(.semibold)).tint(Color.facadeNavy)
                }.padding(20).frame(maxWidth: 680).frame(maxWidth: .infinity)
            }.background(Color.facadeBackground).toolbar(.hidden, for: .navigationBar)
            .sheet(item: $browserResource) { item in
                HelpBrowser(url: item.destinationURL) { browserResource = nil }
                    .ignoresSafeArea()
            }
        }
    }

    /// One situation card. Tapping the header expands it to show the reassurance
    /// sentence and the matching services; tapping again collapses it.
    private func situation(_ item: HelpSituation) -> some View {
        let expanded = openSituation == item.id
        return VStack(alignment: .leading, spacing: 12) {
            Button {
                // Opening one situation replaces whichever was open before.
                withAnimation(.easeInOut(duration: 0.2)) { openSituation = expanded ? nil : item.id }
            } label: {
                HStack(spacing: 12) {
                    IconBadge(systemImage: item.icon)
                    Text(item.title).font(.subheadline.weight(.semibold)).foregroundStyle(Color.facadeText)
                        .multilineTextAlignment(.leading).fixedSize(horizontal: false, vertical: true)
                    Spacer(minLength: 8)
                    Image(systemName: "chevron.down").font(.footnote.weight(.semibold))
                        .foregroundStyle(Color.facadeMuted).rotationEffect(.degrees(expanded ? 180 : 0))
                        .accessibilityHidden(true)
                }.contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            // The chevron is hidden from VoiceOver, so the open/closed state is spoken
            // as the button's value instead.
            .accessibilityValue(expanded ? "Expanded" : "Collapsed")
            if expanded {
                Text(item.reassurance).font(.subheadline).foregroundStyle(Color.facadeText)
                    .fixedSize(horizontal: false, vertical: true)
                // `\.0` uses the service itself (the first item of each pair) as the
                // identity. That is safe because a situation never lists a service twice.
                ForEach(item.resources, id: \.0) { resource, reason in
                    resourceRow(resource, reason: reason)
                }
            }
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.facadeSurface, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous)
            .strokeBorder(expanded ? Color.facadeNavy : Color.facadeHairline, lineWidth: expanded ? 1.5 : 1))
    }

    /// One service inside an open situation: its name, why it fits, and a button
    /// that opens its website.
    private func resourceRow(_ item: FACADEHelpResource, reason: String) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(item.title).font(.subheadline.weight(.semibold)).foregroundStyle(Color.facadeText)
            Text(reason).font(.footnote).foregroundStyle(Color.facadeMuted)
                .fixedSize(horizontal: false, vertical: true)
            openButton(item, label: "Open \(item.title)")
        }
        .padding(12).frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.facadeBackground, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
    }

    /// The "Still need help?" card, which links to the House's official
    /// find-your-representative page.
    private var congressCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 12) {
                IconBadge(systemImage: "building.columns.fill")
                Text("Still need help?").font(.headline)
            }
            Text("Reach out to your member of Congress. Their office has caseworkers whose job is to help people in their district, including connecting you with the right agencies. Your story also helps lawmakers understand why this matters.")
                .font(.subheadline).foregroundStyle(Color.facadeMuted)
                .fixedSize(horizontal: false, vertical: true)
            openButton(.findRepresentative, label: "Find your representative")
        }.fastCard()
    }

    /// A button that opens one service's fixed web address. It tries three ways in
    /// order: SwiftUI's `openURL`, then UIKit's `UIApplication.open`, and only if
    /// both fail, the in-app Safari sheet.
    private func openButton(_ item: FACADEHelpResource, label: String) -> some View {
        Button {
            let destination = item.destinationURL
            openURL(destination) { accepted in
                guard !accepted else { return }
                // In some app states SwiftUI's opener reports failure even when a
                // browser is available. Retry the same fixed public address
                // through UIKit's `UIApplication.open`.
                UIApplication.shared.open(destination, options: [:]) { opened in
                    if !opened {
                        // The completion closure is not marked as main-actor code, so the state
                        // change that shows the sheet is explicitly run on the main actor.
                        Task { @MainActor in browserResource = item }
                    }
                }
            }
        } label: {
            Label(label, systemImage: "arrow.up.right.square")
        }.buttonStyle(SecondaryButtonStyle())
            // The hint names the website's host (for example "takeitdown.ncmec.org"),
            // so VoiceOver users know where the link goes before tapping it.
            .accessibilityHint("Opens \(item.destinationURL.host ?? "the official website") in your browser. Internet access is required.")
    }
}

/// An in-app Safari page, shown only after the user tapped a service and the
/// system browser could not be opened.
/// `SFSafariViewController` is Apple's own browser view: the app cannot read
/// the page, its cookies or anything typed into it, and adds no scripts.
struct HelpBrowser: UIViewControllerRepresentable {
    /// The page to open, and what to do when the user taps Done.
    let url: URL
    let onDismiss: () -> Void

    /// SwiftUI calls these `UIViewControllerRepresentable` methods to wrap a
    /// UIKit view controller. The coordinator receives Safari's "Done" callback.
    func makeCoordinator() -> Coordinator { Coordinator(onDismiss: onDismiss) }

    /// Creates the Safari view once, with a "Done" button to close it.
    func makeUIViewController(context: Context) -> SFSafariViewController {
        let controller = SFSafariViewController(url: url)
        controller.delegate = context.coordinator
        controller.dismissButtonStyle = .done
        if #unavailable(iOS 26) {
            // iOS 26 ignores this setting (deprecated there); older versions get navy controls.
            controller.preferredControlTintColor = UIColor(Color.facadeNavy)
        }
        return controller
    }

    /// Nothing to update: the address never changes while the sheet is open.
    func updateUIViewController(_ controller: SFSafariViewController, context: Context) {}

    /// Forwards Safari's "Done" tap to `onDismiss`, which closes the sheet.
    final class Coordinator: NSObject, SFSafariViewControllerDelegate {
        let onDismiss: () -> Void
        init(onDismiss: @escaping () -> Void) { self.onDismiss = onDismiss }
        func safariViewControllerDidFinish(_ controller: SFSafariViewController) { onDismiss() }
    }
}
