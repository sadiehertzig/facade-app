import SwiftUI
import UIKit

/// Flat navy, midnight while pressed. Disabled is grey with an outline border so
/// it never looks like an enabled secondary button, even in greyscale.
struct PrimaryButtonStyle: ButtonStyle {
    @Environment(\.isEnabled) private var enabled
    func makeBody(configuration: Configuration) -> some View {
        let shape = RoundedRectangle(cornerRadius: 14, style: .continuous)
        configuration.label.font(.headline).frame(maxWidth: .infinity, minHeight: 52)
            .padding(.horizontal, 12)
            .foregroundStyle(enabled ? Color.white : Color.facadeMuted)
            .background(enabled ? (configuration.isPressed ? Color.facadeMidnight : Color.facadeNavy)
                                : Color.facadeSurfaceRaised, in: shape)
            .overlay(shape.strokeBorder(enabled ? Color.clear : Color.facadeOutline, lineWidth: 1))
            .scaleEffect(configuration.isPressed ? 0.98 : 1)
            .animation(.easeOut(duration: 0.12), value: configuration.isPressed)
    }
}

/// Navy label on sky tint; grey when pressed or disabled.
struct SecondaryButtonStyle: ButtonStyle {
    @Environment(\.isEnabled) private var enabled
    func makeBody(configuration: Configuration) -> some View {
        configuration.label.font(.subheadline.weight(.semibold)).frame(maxWidth: .infinity, minHeight: 48)
            .padding(.horizontal, 12)
            .foregroundStyle(enabled ? Color.facadeNavy : Color.facadeMuted)
            .background(enabled && !configuration.isPressed ? Color.facadeSkyTint : Color.facadeSurfaceRaised,
                        in: RoundedRectangle(cornerRadius: 14, style: .continuous))
    }
}

extension LinearGradient {
    /// Used only on the header badge and the intro card, never on buttons or results.
    static let facadeBrand = LinearGradient(colors: [.facadeNavy, .facadeMidnight],
                                            startPoint: .topLeading, endPoint: .bottomTrailing)
}

extension View {
    func fastCard() -> some View {
        self.padding(18).frame(maxWidth: .infinity, alignment: .leading)
            .background(Color.facadeSurface, in: RoundedRectangle(cornerRadius: 20, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 20, style: .continuous).strokeBorder(Color.facadeHairline, lineWidth: 1))
            .shadow(color: Color.facadeNavy.opacity(0.06), radius: 12, y: 4)
            .foregroundStyle(Color.facadeText)
    }
}

/// Tinted rounded-square icon used on section and resource headers.
struct IconBadge: View {
    let systemImage: String
    var tint: Color = .facadeNavy
    var body: some View {
        Image(systemName: systemImage).font(.headline).foregroundStyle(tint)
            .frame(width: 38, height: 38)
            .background(tint.opacity(0.12), in: RoundedRectangle(cornerRadius: 11, style: .continuous))
            .accessibilityHidden(true)
    }
}

/// Numbered section title used on the Protect flow.
struct StepTitle: View {
    let number: Int
    let title: String
    var body: some View {
        HStack(spacing: 10) {
            Text("\(number)").font(.subheadline.weight(.bold)).foregroundStyle(.white)
                .frame(width: 26, height: 26).background(Color.facadeNavy, in: Circle())
                .accessibilityHidden(true)
            Text(title).font(.headline)
        }.accessibilityElement(children: .combine).accessibilityLabel("Step \(number): \(title)")
    }
}

/// Four-step progress strip: Choose, Review, Protect, Save.
struct StepIndicator: View {
    let current: Int
    private let names = ["Choose", "Review", "Protect", "Save"]
    var body: some View {
        HStack(spacing: 6) {
            ForEach(names.indices, id: \.self) { index in
                VStack(spacing: 6) {
                    Capsule().fill(index < current ? Color.facadeNavy : Color.facadeOutline).frame(height: 4)
                    Text(names[index]).font(.caption2.weight(index + 1 == current ? .bold : .medium))
                        .foregroundStyle(index < current ? Color.facadeText : Color.facadeMuted)
                        .lineLimit(1).minimumScaleFactor(0.7)
                }
            }
        }
        .animation(.easeInOut(duration: 0.25), value: current)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Step \(current) of \(names.count): \(names[max(0, min(current, names.count) - 1)])")
    }
}

/// Before/after slider. Both images share one pixel grid; the protected copy is
/// revealed from the left as the handle moves. Display only; no pixels change.
struct ComparisonSlider: View {
    let original: UIImage
    let protected: UIImage
    @State private var position: CGFloat = 0.5

    var body: some View {
        let ratio = original.size.height > 0 ? original.size.width / original.size.height : 1
        GeometryReader { geometry in
            let width = geometry.size.width
            ZStack(alignment: .leading) {
                Image(uiImage: original).resizable().scaledToFit()
                Image(uiImage: protected).resizable().scaledToFit()
                    .mask(alignment: .leading) { Rectangle().frame(width: width * position) }
                Rectangle().fill(.white).frame(width: 2).shadow(radius: 2)
                    .offset(x: width * position - 1)
                Image(systemName: "chevron.left.chevron.right").font(.caption.weight(.bold))
                    .foregroundStyle(Color.facadeNavy).frame(width: 36, height: 36)
                    .background(.white, in: Circle()).shadow(color: .black.opacity(0.2), radius: 4)
                    .offset(x: width * position - 18)
                tag("Protected", alignment: .topLeading)
                tag("Original", alignment: .topTrailing)
            }
            .frame(width: width, height: geometry.size.height)
            .contentShape(Rectangle())
            .gesture(DragGesture(minimumDistance: 8).onChanged { value in
                position = min(max(value.location.x / max(width, 1), 0), 1)
            })
            .onTapGesture { location in
                withAnimation(.easeOut(duration: 0.2)) { position = min(max(location.x / max(width, 1), 0), 1) }
            }
        }
        .aspectRatio(ratio, contentMode: .fit)
        .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
        .privacySensitive()
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Original and protected comparison")
        .accessibilityValue("\(Int((position * 100).rounded())) percent protected copy shown")
        .accessibilityAdjustableAction { direction in
            switch direction {
            case .increment: position = min(position + 0.1, 1)
            case .decrement: position = max(position - 0.1, 0)
            @unknown default: break
            }
        }
    }

    private func tag(_ text: String, alignment: Alignment) -> some View {
        Text(text).font(.caption2.weight(.bold)).foregroundStyle(.white)
            .padding(.horizontal, 8).padding(.vertical, 4)
            .background(Color.facadeMidnight.opacity(0.8), in: Capsule())
            .padding(8)
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: alignment)
            .allowsHitTesting(false)
    }
}

/// A notice on a solid fill. Each kind has its own colour and glyph, and the
/// words carry the meaning, so colour is never the only cue.
struct StatusBanner: View {
    enum Kind { case info, warning, error, urgentHelp }

    let kind: Kind
    let message: String

    /// Picks the kind for a status message shown by LivePhotoModel or the
    /// Protect screen. When you add or edit one of those messages, update this
    /// table. An unknown message shows as an error, so it is never under-warned.
    static func kind(for message: String) -> Kind {
        switch message {
        case "Choose a photo of consenting adults.",
             "Confirm that you have permission to use this photo.",
             "Processing stopped. Nothing was saved.",
             "The in-memory preview was cleared when the app left the foreground.",
             "The comparison was stopped. Nothing was saved.",
             "The comparison was stopped.":
            return .info
        case "FACADE's on-device models are not installed in this build.",
             "No sample photo is included in this build. Choose your own photo instead.",
             "This photo could not be prepared. Choose a clear photo with one face.":
            return .warning
        default:
            return .error
        }
    }

    var body: some View {
        let shape = RoundedRectangle(cornerRadius: 16, style: .continuous)
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: glyph)
                .font(kind == .urgentHelp ? .footnote.weight(.semibold) : .callout.weight(.semibold))
                .foregroundStyle(kind == .urgentHelp ? Color.facadeNavy : tint)
            Text(message)
                .font(kind == .urgentHelp ? .footnote.weight(.semibold) : .callout)
                .foregroundStyle(tint)
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(14)
        .background(fill, in: shape)
        .overlay(shape.strokeBorder(kind == .urgentHelp ? Color.facadeOutline : Color.clear, lineWidth: 1))
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("\(spokenKind): \(message)")
    }

    private var glyph: String {
        switch kind {
        case .info: "info.circle.fill"
        case .warning: "exclamationmark.triangle.fill"
        case .error: "exclamationmark.octagon.fill"
        case .urgentHelp: "phone.fill"
        }
    }

    private var tint: Color {
        switch kind {
        case .info: .facadeNavy
        case .warning: .facadeWarning
        case .error: .facadeError
        case .urgentHelp: .facadeText
        }
    }

    private var fill: Color {
        switch kind {
        case .info: .facadeSkyTint
        case .warning: .facadeWarningFill
        case .error: .facadeErrorFill
        case .urgentHelp: .facadeSurface
        }
    }

    private var spokenKind: String {
        switch kind {
        case .info: "Note"
        case .warning: "Warning"
        case .error: "Error"
        case .urgentHelp: "Urgent help"
        }
    }
}

#Preview("Status banners") {
    ScrollView {
        VStack(spacing: 12) {
            ForEach([
                "Choose a photo of consenting adults.",
                "Confirm that you have permission to use this photo.",
                "Processing stopped. Nothing was saved.",
                "The in-memory preview was cleared when the app left the foreground.",
                "The comparison was stopped. Nothing was saved.",
                "The comparison was stopped.",
                "FACADE's on-device models are not installed in this build.",
                "No sample photo is included in this build. Choose your own photo instead.",
                "This photo could not be prepared. Choose a clear photo with one face.",
                "This photo could not be read.",
                "This photo could not be read. Please try another photo.",
                "The new copy was not saved: Photos access was denied.",
                "An unexpected engine error.",
            ], id: \.self) { message in
                StatusBanner(kind: StatusBanner.kind(for: message), message: message)
            }
            StatusBanner(kind: .urgentHelp,
                         message: "If you are in immediate danger, call 911. For emotional support at any time, call or text 988.")
        }.padding(20)
    }.background(Color.facadeBackground)
}
