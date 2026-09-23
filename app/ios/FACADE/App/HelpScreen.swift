import SwiftUI
import UIKit
import SafariServices

/// One situation on the Get Help screen and the fixed resources it points to.
private struct HelpSituation: Identifiable {
    let id: String
    let icon: String
    let title: String
    let reassurance: String
    let resources: [(FACADEHelpResource, String)]
}

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

struct HelpScreen: View {
    @Environment(\.openURL) private var openURL
    @State private var browserResource: FACADEHelpResource?
    @State private var openSituation: String?

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Get help").font(.largeTitle.weight(.bold))
                        Text("What happened is not your fault.").font(.title3.weight(.semibold))
                        Text("You don't have to handle this alone. Free, confidential help is available, and you never need to show the image to FACADE.")
                            .foregroundStyle(Color.facadeMuted).fixedSize(horizontal: false, vertical: true)
                    }
                    Label("If you are in immediate danger, call 911. For emotional support at any time, call or text 988.", systemImage: "phone.fill")
                        .font(.footnote.weight(.semibold)).foregroundStyle(Color.facadeCoral)
                        .fixedSize(horizontal: false, vertical: true)
                        .padding(12).frame(maxWidth: .infinity, alignment: .leading)
                        .background(Color.facadeCoral.opacity(0.08), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
                    VStack(alignment: .leading, spacing: 10) {
                        Text("What's happening?").font(.headline)
                        Text("Choose what fits. We'll show you where to start.")
                            .font(.subheadline).foregroundStyle(Color.facadeMuted)
                        ForEach(helpSituations) { situation($0) }
                    }
                    congressCard
                    DisclosureGroup("How these links work") {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Each link opens an official website in your browser. No photo, address or personal information is sent by FACADE. Links need Internet access.")
                            Text("Don't download, forward or ask for an intimate image to use these services. If you don't have it, the CyberTipline can still help.")
                            Text("Fingerprint (hash) services work with participating platforms and can't guarantee removal everywhere. FACADE can't remove images already online.")
                            Text("Information checked September 22, 2026.").font(.caption)
                        }.font(.footnote).foregroundStyle(Color.facadeMuted).padding(.top, 8)
                    }.font(.subheadline.weight(.semibold)).tint(Color.facadeViolet)
                }.padding(20).frame(maxWidth: 680).frame(maxWidth: .infinity)
            }.background(Color.facadeBackground).toolbar(.hidden, for: .navigationBar)
            .sheet(item: $browserResource) { item in
                HelpBrowser(url: item.destinationURL) { browserResource = nil }
                    .ignoresSafeArea()
            }
        }
    }

    private func situation(_ item: HelpSituation) -> some View {
        let expanded = openSituation == item.id
        return VStack(alignment: .leading, spacing: 12) {
            Button {
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
            .accessibilityValue(expanded ? "Expanded" : "Collapsed")
            if expanded {
                Text(item.reassurance).font(.subheadline).foregroundStyle(Color.facadeMint)
                    .fixedSize(horizontal: false, vertical: true)
                ForEach(item.resources, id: \.0) { resource, reason in
                    resourceRow(resource, reason: reason)
                }
            }
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.facadeSurface, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous)
            .strokeBorder(expanded ? Color.facadeViolet : Color.facadeHairline, lineWidth: expanded ? 1.5 : 1))
    }

    private func resourceRow(_ item: FACADEHelpResource, reason: String) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(item.title).font(.subheadline.weight(.semibold))
            Text(reason).font(.footnote).foregroundStyle(Color.facadeMuted)
                .fixedSize(horizontal: false, vertical: true)
            openButton(item, label: "Open \(item.title)")
        }
        .padding(12).frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.facadeBackground, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
    }

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

    private func openButton(_ item: FACADEHelpResource, label: String) -> some View {
        Button {
            let destination = item.destinationURL
            openURL(destination) { accepted in
                guard !accepted else { return }
                // Some active iOS scenes reject SwiftUI's route even when
                // the browser is available. Retry only this fixed public
                // destination through the system application API.
                UIApplication.shared.open(destination, options: [:]) { opened in
                    if !opened {
                        Task { @MainActor in browserResource = item }
                    }
                }
            }
        } label: {
            Label(label, systemImage: "arrow.up.right.square")
        }.buttonStyle(SecondaryButtonStyle())
            .accessibilityHint("Opens \(item.destinationURL.host ?? "the official website") in your browser. Internet access is required.")
    }
}

/// Created only after an explicit resource tap and failed external opening.
/// Safari owns navigation and form contents; no script or page-data bridge is added.
struct HelpBrowser: UIViewControllerRepresentable {
    let url: URL
    let onDismiss: () -> Void

    func makeCoordinator() -> Coordinator { Coordinator(onDismiss: onDismiss) }

    func makeUIViewController(context: Context) -> SFSafariViewController {
        let controller = SFSafariViewController(url: url)
        controller.delegate = context.coordinator
        controller.dismissButtonStyle = .done
        return controller
    }

    func updateUIViewController(_ controller: SFSafariViewController, context: Context) {}

    final class Coordinator: NSObject, SFSafariViewControllerDelegate {
        let onDismiss: () -> Void
        init(onDismiss: @escaping () -> Void) { self.onDismiss = onDismiss }
        func safariViewControllerDidFinish(_ controller: SFSafariViewController) { onDismiss() }
    }
}
