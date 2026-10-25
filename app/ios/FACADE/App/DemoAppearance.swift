// DemoAppearance.swift
//
// The app's colour palette ("Midnight Harbor") plus four small, reusable
// SwiftUI building blocks: a card style, a status card, an image preview tile
// and a label/value row.
//
// Where it sits: every screen reads its colours from the `Color.facade...`
// constants defined here. Design.swift builds the buttons, cards and banners
// the screens actually use on top of these colours.
//
// Key ideas:
// 1. Colours are named by their job ("text", "muted", "outline"), not by their
//    hue, so a screen asks for "the colour of secondary text" and the palette
//    can change in one place.
// 2. Colour is never the only signal. Success, warning and error states are
//    always shown together with an icon and words, so they still read in
//    greyscale or for colour-blind users.
// 3. The values were chosen and contrast-checked for light mode only, which is
//    why PhoneApp.swift forces light mode.
// 4. The whole file is wrapped in `#if os(iOS)` because it uses UIKit types
//    (`UIImage`, `Color(.separator)`) that do not exist on every Apple platform.
#if os(iOS)
import SwiftUI
import UIKit

/// Midnight Harbor colour tokens: one navy and light-blue family shared with
/// the app icon and the website. Every colour in the app comes from here.
/// Each constant is written as RGB fractions (0...1). The trailing comment on
/// each line gives the same colour as a hex code and says where it is used.
public extension Color {
    // Neutrals: page background, card surfaces, text and borders.
    // "Hairline" is a light border for decoration only; "outline" is darker and
    // is used where the border itself carries meaning (for example, a disabled
    // button or an unselected option), so it must stay visible.
    // Neutrals
    static let facadeBackground = Color(red: 0.953, green: 0.961, blue: 0.976)   // #F3F5F9 page
    static let facadeSurface = Color(red: 1, green: 1, blue: 1)                    // #FFFFFF cards
    static let facadeSurfaceRaised = Color(red: 0.906, green: 0.929, blue: 0.961)  // #E7EDF5 pressed, disabled, insets
    static let facadeText = Color(red: 0.067, green: 0.122, blue: 0.204)           // #111F34 ink
    static let facadeMuted = Color(red: 0.318, green: 0.365, blue: 0.431)          // #515D6E secondary text
    static let facadeHairline = Color(red: 0.824, green: 0.847, blue: 0.882)       // #D2D8E1 decorative borders only
    static let facadeOutline = Color(red: 0.439, green: 0.486, blue: 0.561)        // #707C8F borders that show state

    // Brand blues. Navy marks every tappable control; midnight is the darker
    // shade used for pressed buttons and the icon background. The light "sky"
    // blue is only legible on a dark navy or midnight background.
    // Brand blues
    static let facadeNavy = Color(red: 0.141, green: 0.294, blue: 0.573)           // #244B92 everything you tap
    static let facadeMidnight = Color(red: 0.020, green: 0.031, blue: 0.271)       // #050845 the icon's background
    static let facadeSkyTint = Color(red: 0.839, green: 0.910, blue: 0.988)        // #D6E8FC light-blue fills
    static let facadeSky = Color(red: 0.651, green: 0.827, blue: 1)                // #A6D3FF only on navy or midnight

    // Each state has a strong colour for text and icons and, for warning and
    // error, a pale "fill" colour for the banner background behind it.
    // States. Always shown with a glyph and words, never by colour alone.
    // Success is never used on a protection result.
    static let facadeSuccess = Color(red: 0.012, green: 0.439, blue: 0.384)        // #037062
    static let facadeWarning = Color(red: 0.561, green: 0.322, blue: 0)            // #8F5200
    static let facadeWarningFill = Color(red: 0.957, green: 0.933, blue: 0.902)    // #F4EEE6
    static let facadeError = Color(red: 0.706, green: 0.180, blue: 0.322)          // #B42E52
    static let facadeErrorFill = Color(red: 0.973, green: 0.918, blue: 0.933)      // #F8EAEE
}

/// A `GroupBox` style that draws a white rounded card with a hairline border
/// and a bold heading. It is public and kept for reuse, but the current
/// screens draw their cards with `fastCard()` in Design.swift instead.
public struct FACADEDemoCardStyle: GroupBoxStyle {
    /// No settings: the style always looks the same.
    public init() {}

    /// Stacks the group box's label (as a headline) above its content, then pads
    /// the stack and puts it on a rounded white card.
    public func makeBody(configuration: Configuration) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            configuration.label
                .font(.headline)
                .foregroundStyle(Color.facadeText)
            configuration.content
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            Color.facadeSurface,
            in: RoundedRectangle(cornerRadius: 20, style: .continuous)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 20, style: .continuous)
                .strokeBorder(Color.facadeHairline, lineWidth: 1)
        )
    }
}

/// A one-line status message on a lightly tinted rounded background, with
/// either an SF Symbol icon or a spinning activity indicator on the left.
/// Like the card style above, it is public but not used by the current
/// screens, which show status with `StatusBanner` in Design.swift.
public struct FACADEDemoStatusCard: View {
    /// What the card shows: the text, the SF Symbols icon name, the accent
    /// colour, and whether to show a spinner in place of the icon.
    private let message: String
    private let systemImage: String
    private let tint: Color
    private let showsProgress: Bool

    /// `showsProgress` defaults to false, so most callers only pass the message,
    /// icon and tint.
    public init(message: String, systemImage: String, tint: Color, showsProgress: Bool = false) {
        self.message = message
        self.systemImage = systemImage
        self.tint = tint
        self.showsProgress = showsProgress
    }

    /// Icon (or spinner) on the left, message filling the rest of the row.
    public var body: some View {
        HStack(alignment: .top, spacing: 12) {
            if showsProgress {
                ProgressView()
                    .tint(tint)
            } else {
                Image(systemName: systemImage)
                    .font(.title3)
                    .foregroundStyle(tint)
                    // The icon is decorative; VoiceOver reads the message text instead.
                    .accessibilityHidden(true)
            }
            Text(message)
                .font(.subheadline)
                // A grey `.secondary` tint would make the message too faint to read, so
                // in that one case the text falls back to the normal primary colour.
                .foregroundStyle(tint == .secondary ? Color.primary : tint)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(16)
        .background(
            // The background is the tint at 10% strength, so the text on top keeps
            // enough contrast.
            tint.opacity(0.10),
            in: RoundedRectangle(cornerRadius: 18, style: .continuous)
        )
    }
}

/// A rounded image preview with a caption underneath. If the image data is
/// missing or cannot be decoded, a grey placeholder square is shown instead.
/// Public, but not used by the current screens (they use the `photo` helper in
/// PhotoAcquisitionViews.swift).
public struct FACADEDemoPreviewTile: View {
    /// The caption, the encoded image bytes (PNG or JPEG), and the description
    /// VoiceOver reads for the image.
    private let title: String
    private let data: Data?
    private let accessibilityLabel: String

    /// Stores the three values; nothing is decoded until the view is drawn.
    public init(title: String, data: Data?, accessibilityLabel: String) {
        self.title = title
        self.data = data
        self.accessibilityLabel = accessibilityLabel
    }

    public var body: some View {
        VStack(spacing: 6) {
            if let data, let image = UIImage(data: data) {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFit()
                    .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                    .overlay(
                        RoundedRectangle(cornerRadius: 12, style: .continuous)
                            .strokeBorder(Color(.separator).opacity(0.5), lineWidth: 0.5)
                    )
                    // Marks the photo as private: SwiftUI hides it if this view is ever drawn
                    // with privacy redaction turned on. Normal display is unaffected.
                    .privacySensitive()
                    .accessibilityLabel(accessibilityLabel)
            } else {
                // Fallback: a square grey placeholder with a "preview unavailable" note.
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .fill(Color(.tertiarySystemGroupedBackground))
                    .aspectRatio(1, contentMode: .fit)
                    .overlay {
                        VStack(spacing: 4) {
                            Image(systemName: "photo.badge.exclamationmark")
                                .font(.title3)
                            Text("Preview unavailable")
                                .font(.caption2)
                        }
                        .foregroundStyle(.secondary)
                    }
            }
            Text(title)
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
    }
}

/// A label on the left and a value on the right, with digits drawn at a fixed
/// width so numbers line up from row to row. Public, but not used by the
/// current screens (they use `detailRow` in PhotoAcquisitionViews.swift).
public struct FACADEDemoMetricRow: View {
    /// The row's name, its value text, and the colour of the value.
    private let label: String
    private let value: String
    private let valueTint: Color

    /// The label is unnamed at the call site, so a row reads like
    /// `FACADEDemoMetricRow("PSNR", value: "40 dB")`.
    public init(_ label: String, value: String, valueTint: Color = .primary) {
        self.label = label
        self.value = value
        self.valueTint = valueTint
    }

    /// Lines the label and value up on the first line of text and keeps at
    /// least 12 points between them.
    public var body: some View {
        HStack(alignment: .firstTextBaseline) {
            Text(label)
                .foregroundStyle(.secondary)
            Spacer(minLength: 12)
            Text(value)
                // Every digit gets the same width, so values in a column align.
                .monospacedDigit()
                .foregroundStyle(valueTint)
                .multilineTextAlignment(.trailing)
        }
        .font(.caption)
    }
}
#endif
