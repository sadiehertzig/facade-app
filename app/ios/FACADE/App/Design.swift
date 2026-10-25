// Design.swift
//
// The shared visual pieces of the app: the two button styles, the card look,
// the icon badge, the numbered step title, the four-step progress strip, the
// before/after comparison slider, and the status banner.
//
// Where it sits: the Protect screen (PhotoAcquisitionViews.swift) and the Get
// Help screen (HelpScreen.swift) are built from these pieces. The colours they
// use come from DemoAppearance.swift.
//
// Key ideas:
// 1. Buttons stay readable when disabled: a disabled button turns grey and
//    gains a border, so it never looks like a different, enabled button.
// 2. Results are never decorated as a "success". The brand gradient and green
//    are kept off result cards because protection is partial and not
//    guaranteed.
// 3. The pieces set up VoiceOver: decorative icons are hidden, and composite
//    views (the step strip, the slider, the banner) read as one sentence.
import SwiftUI
import UIKit

/// Flat navy, midnight while pressed. Disabled is grey with an outline border so
/// it never looks like an enabled secondary button, even in greyscale.
/// Used for the main action on each step (Choose photo, Protect, Save).
struct PrimaryButtonStyle: ButtonStyle {
    /// Read from SwiftUI's environment, so `.disabled(true)` on the button
    /// automatically switches the style to its grey look.
    @Environment(\.isEnabled) private var enabled
    /// Draws the label full width, at least 52 points tall (comfortably above
    /// Apple's 44-point minimum touch target).
    func makeBody(configuration: Configuration) -> some View {
        let shape = RoundedRectangle(cornerRadius: 14, style: .continuous)
        configuration.label.font(.headline).frame(maxWidth: .infinity, minHeight: 52)
            .padding(.horizontal, 12)
            .foregroundStyle(enabled ? Color.white : Color.facadeMuted)
            .background(enabled ? (configuration.isPressed ? Color.facadeMidnight : Color.facadeNavy)
                                : Color.facadeSurfaceRaised, in: shape)
            .overlay(shape.strokeBorder(enabled ? Color.clear : Color.facadeOutline, lineWidth: 1))
            // A slight shrink while the finger is down gives tactile feedback.
            .scaleEffect(configuration.isPressed ? 0.98 : 1)
            .animation(.easeOut(duration: 0.12), value: configuration.isPressed)
    }
}

/// The style for second-choice actions (Cancel, Choose another photo, links):
/// navy text on a light sky-blue fill; grey when pressed or disabled.
struct SecondaryButtonStyle: ButtonStyle {
    /// Whether the button is enabled, read from SwiftUI's environment.
    @Environment(\.isEnabled) private var enabled
    /// Full width, at least 48 points tall, slightly smaller text than the
    /// primary style so the two never look the same.
    func makeBody(configuration: Configuration) -> some View {
        configuration.label.font(.subheadline.weight(.semibold)).frame(maxWidth: .infinity, minHeight: 48)
            .padding(.horizontal, 12)
            .foregroundStyle(enabled ? Color.facadeNavy : Color.facadeMuted)
            .background(enabled && !configuration.isPressed ? Color.facadeSkyTint : Color.facadeSurfaceRaised,
                        in: RoundedRectangle(cornerRadius: 14, style: .continuous))
    }
}

/// The navy-to-midnight brand gradient.
extension LinearGradient {
    /// Used only on the header badge and the intro card, never on buttons or results.
    static let facadeBrand = LinearGradient(colors: [.facadeNavy, .facadeMidnight],
                                            startPoint: .topLeading, endPoint: .bottomTrailing)
}

/// A shared card look for every section of both screens.
extension View {
    /// Wraps any view in a padded white card with a hairline border, a faint navy
    /// shadow and the standard text colour. ("fast" is left over from an older
    /// app name; the look is simply the standard FACADE card.)
    func fastCard() -> some View {
        self.padding(18).frame(maxWidth: .infinity, alignment: .leading)
            .background(Color.facadeSurface, in: RoundedRectangle(cornerRadius: 20, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 20, style: .continuous).strokeBorder(Color.facadeHairline, lineWidth: 1))
            .shadow(color: Color.facadeNavy.opacity(0.06), radius: 12, y: 4)
            .foregroundStyle(Color.facadeText)
    }
}

/// A small rounded square holding an SF Symbols icon, used at the start of
/// section and resource headers. The background is the icon's tint at 12%.
struct IconBadge: View {
    /// The SF Symbols name, and the colour (navy unless a caller overrides it).
    let systemImage: String
    var tint: Color = .facadeNavy
    /// Hidden from VoiceOver: the header text next to it already says what the
    /// section is.
    var body: some View {
        Image(systemName: systemImage).font(.headline).foregroundStyle(tint)
            .frame(width: 38, height: 38)
            .background(tint.opacity(0.12), in: RoundedRectangle(cornerRadius: 11, style: .continuous))
            .accessibilityHidden(true)
    }
}

/// A section heading with its step number in a navy circle, such as
/// "(1) Choose your photo" on the Protect screen.
struct StepTitle: View {
    /// The step number shown in the circle and the heading text beside it.
    let number: Int
    let title: String
    /// VoiceOver reads the pair as one phrase, "Step 1: Choose your photo",
    /// instead of a lone number followed by the title.
    var body: some View {
        HStack(spacing: 10) {
            Text("\(number)").font(.subheadline.weight(.bold)).foregroundStyle(.white)
                .frame(width: 26, height: 26).background(Color.facadeNavy, in: Circle())
                .accessibilityHidden(true)
            Text(title).font(.headline)
        }.accessibilityElement(children: .combine).accessibilityLabel("Step \(number): \(title)")
    }
}

/// The four-step progress strip at the top of the Protect screen: Choose,
/// Review, Protect, Save. Each step is a thin bar with its name underneath.
struct StepIndicator: View {
    /// The step the user is on, counted from 1. The Protect screen passes 5 after
    /// a successful save so that all four bars are filled.
    let current: Int
    private let names = ["Choose", "Review", "Protect", "Save"]
    /// Bars before the current step (and the current step itself) are filled
    /// navy; the current step's name is bold.
    var body: some View {
        HStack(spacing: 6) {
            // `index` counts from 0 while `current` counts from 1, so `index < current`
            // is true for every step up to and including the current one, and
            // `index + 1 == current` picks out the current step's name.
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
        // VoiceOver reads one summary, such as "Step 2 of 4: Review", instead of
        // four separate words.
        .accessibilityElement(children: .ignore)
        // The index is clamped into 0...3, so a `current` of 5 (after saving) still
        // names the last step instead of reading past the end of the array.
        .accessibilityLabel("Step \(current) of \(names.count): \(names[max(0, min(current, names.count) - 1)])")
    }
}

/// Before/after slider. Both images share one pixel grid; the protected copy is
/// revealed from the left as the handle moves. Display only; no pixels change.
///
/// Dragging or tapping moves the divider. VoiceOver users can swipe up or
/// down to move it in 10% steps.
struct ComparisonSlider: View {
    /// The two images to compare. They must be the same size, because they are
    /// drawn on top of each other and only one mask decides which one shows.
    let original: UIImage
    let protected: UIImage
    /// Where the divider is, as a fraction of the width: 0 = all original,
    /// 1 = all protected. It starts in the middle.
    @State private var position: CGFloat = 0.5

    /// Draws the original, then the protected copy masked to the left
    /// `position` of the width, then the white divider line and its round handle.
    var body: some View {
        // Width-to-height ratio of the photo, so the slider is exactly as tall as
        // the picture. Falls back to a square if the height is somehow zero.
        let ratio = original.size.height > 0 ? original.size.width / original.size.height : 1
        GeometryReader { geometry in
            let width = geometry.size.width
            ZStack(alignment: .leading) {
                Image(uiImage: original).resizable().scaledToFit()
                Image(uiImage: protected).resizable().scaledToFit()
                    .mask(alignment: .leading) { Rectangle().frame(width: width * position) }
                // The divider is 2 points wide, so it is shifted 1 point left to centre it
                // on `position`. The 36-point handle below is shifted by half its width (18).
                Rectangle().fill(.white).frame(width: 2).shadow(radius: 2)
                    .offset(x: width * position - 1)
                Image(systemName: "chevron.left.chevron.right").font(.caption.weight(.bold))
                    .foregroundStyle(Color.facadeNavy).frame(width: 36, height: 36)
                    .background(.white, in: Circle()).shadow(color: .black.opacity(0.2), radius: 4)
                    .offset(x: width * position - 18)
                // The protected copy is on the left of the divider, the original on the
                // right, so the tags sit in those corners.
                tag("Protected", alignment: .topLeading)
                tag("Original", alignment: .topTrailing)
            }
            .frame(width: width, height: geometry.size.height)
            .contentShape(Rectangle())
            // A drag must move 8 points before it counts, so a vertical scroll of the
            // page is not mistaken for moving the slider. The position is clamped to
            // 0...1; `max(width, 1)` avoids dividing by zero before layout.
            .gesture(DragGesture(minimumDistance: 8).onChanged { value in
                position = min(max(value.location.x / max(width, 1), 0), 1)
            })
            .onTapGesture { location in
                withAnimation(.easeOut(duration: 0.2)) { position = min(max(location.x / max(width, 1), 0), 1) }
            }
        }
        .aspectRatio(ratio, contentMode: .fit)
        .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
        // Marks both photos as private: SwiftUI hides them if the view is drawn
        // with privacy redaction turned on.
        .privacySensitive()
        // To VoiceOver the whole slider is one adjustable control whose value is
        // how much of the protected copy is showing.
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

    /// A small dark label pinned to one corner of the slider. It ignores touches
    /// so it never blocks a drag that starts on top of it.
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
///
/// The Protect screen shows one of these for every blocked or stopped state,
/// and the Get Help screen uses the `.urgentHelp` kind for its emergency line.
struct StatusBanner: View {
    /// How serious the message is. `.urgentHelp` is deliberately calm (dark text
    /// on white with an outline) rather than red, because red would read as
    /// "something in the app failed".
    enum Kind { case info, warning, error, urgentHelp }

    let kind: Kind
    let message: String

    /// Picks the kind for a status message shown by LivePhotoModel or the
    /// Protect screen. When you add or edit one of those messages, update this
    /// table. An unknown message shows as an error, so it is never under-warned.
    /// Messages are matched by their exact text, which is why the table has to
    /// be kept in step with the messages themselves.
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
        // Includes photo-read errors, save failures and any error text passed up
        // from the engines.
        default:
            return .error
        }
    }

    /// Glyph and text side by side on the kind's fill colour. VoiceOver reads the
    /// kind first, for example "Warning: ...", so the meaning does not depend on
    /// seeing the colour.
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

    /// The SF Symbols icon for each kind. Each kind has a different shape, not
    /// just a different colour.
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

/// An Xcode preview (not part of the running app) that shows every known
/// message with the kind the table above assigns it, including several that
/// are not in the table and so show the error fallback.
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
