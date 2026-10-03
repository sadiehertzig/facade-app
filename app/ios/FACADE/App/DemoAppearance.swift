#if os(iOS)
import SwiftUI
import UIKit

/// Midnight Harbor colour tokens: one navy and light-blue family shared with
/// the app icon and the website. Every colour in the app comes from here.
public extension Color {
    // Neutrals
    static let facadeBackground = Color(red: 0.953, green: 0.961, blue: 0.976)   // #F3F5F9 page
    static let facadeSurface = Color(red: 1, green: 1, blue: 1)                    // #FFFFFF cards
    static let facadeSurfaceRaised = Color(red: 0.906, green: 0.929, blue: 0.961)  // #E7EDF5 pressed, disabled, insets
    static let facadeText = Color(red: 0.067, green: 0.122, blue: 0.204)           // #111F34 ink
    static let facadeMuted = Color(red: 0.318, green: 0.365, blue: 0.431)          // #515D6E secondary text
    static let facadeHairline = Color(red: 0.824, green: 0.847, blue: 0.882)       // #D2D8E1 decorative borders only
    static let facadeOutline = Color(red: 0.439, green: 0.486, blue: 0.561)        // #707C8F borders that show state

    // Brand blues
    static let facadeNavy = Color(red: 0.141, green: 0.294, blue: 0.573)           // #244B92 everything you tap
    static let facadeMidnight = Color(red: 0.020, green: 0.031, blue: 0.271)       // #050845 the icon's background
    static let facadeSkyTint = Color(red: 0.839, green: 0.910, blue: 0.988)        // #D6E8FC light-blue fills
    static let facadeSky = Color(red: 0.651, green: 0.827, blue: 1)                // #A6D3FF only on navy or midnight

    // States. Always shown with a glyph and words, never by colour alone.
    // Success is never used on a protection result.
    static let facadeSuccess = Color(red: 0.012, green: 0.439, blue: 0.384)        // #037062
    static let facadeWarning = Color(red: 0.561, green: 0.322, blue: 0)            // #8F5200
    static let facadeWarningFill = Color(red: 0.957, green: 0.933, blue: 0.902)    // #F4EEE6
    static let facadeError = Color(red: 0.706, green: 0.180, blue: 0.322)          // #B42E52
    static let facadeErrorFill = Color(red: 0.973, green: 0.918, blue: 0.933)      // #F8EAEE
}

/// Card presentation shared by every section of the demo screen.
public struct FACADEDemoCardStyle: GroupBoxStyle {
    public init() {}

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

/// Status line with an icon, a tint, and an optional activity indicator.
public struct FACADEDemoStatusCard: View {
    private let message: String
    private let systemImage: String
    private let tint: Color
    private let showsProgress: Bool

    public init(message: String, systemImage: String, tint: Color, showsProgress: Bool = false) {
        self.message = message
        self.systemImage = systemImage
        self.tint = tint
        self.showsProgress = showsProgress
    }

    public var body: some View {
        HStack(alignment: .top, spacing: 12) {
            if showsProgress {
                ProgressView()
                    .tint(tint)
            } else {
                Image(systemName: systemImage)
                    .font(.title3)
                    .foregroundStyle(tint)
                    .accessibilityHidden(true)
            }
            Text(message)
                .font(.subheadline)
                .foregroundStyle(tint == .secondary ? Color.primary : tint)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(16)
        .background(
            tint.opacity(0.10),
            in: RoundedRectangle(cornerRadius: 18, style: .continuous)
        )
    }
}

/// Rounded, privacy-sensitive image tile with a caption.
public struct FACADEDemoPreviewTile: View {
    private let title: String
    private let data: Data?
    private let accessibilityLabel: String

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
                    .privacySensitive()
                    .accessibilityLabel(accessibilityLabel)
            } else {
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

/// Label on the left, monospaced value on the right.
public struct FACADEDemoMetricRow: View {
    private let label: String
    private let value: String
    private let valueTint: Color

    public init(_ label: String, value: String, valueTint: Color = .primary) {
        self.label = label
        self.value = value
        self.valueTint = valueTint
    }

    public var body: some View {
        HStack(alignment: .firstTextBaseline) {
            Text(label)
                .foregroundStyle(.secondary)
            Spacer(minLength: 12)
            Text(value)
                .monospacedDigit()
                .foregroundStyle(valueTint)
                .multilineTextAlignment(.trailing)
        }
        .font(.caption)
    }
}
#endif
