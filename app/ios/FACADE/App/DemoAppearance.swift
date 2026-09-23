#if os(iOS)
import SwiftUI
import UIKit

/// Light navy visual tokens for the FACADE interface.
public extension Color {
    static let facadeBackground = Color(red: 0.953, green: 0.961, blue: 0.976)
    static let facadeSurface = Color(red: 1, green: 1, blue: 1)
    static let facadeSurfaceRaised = Color(red: 0.906, green: 0.929, blue: 0.961)
    static let facadeViolet = Color(red: 0.067, green: 0.165, blue: 0.29)
    static let facadeVioletBright = Color(red: 0.122, green: 0.275, blue: 0.455)
    static let facadeMint = Color(red: 0.047, green: 0.392, blue: 0.294)
    static let facadeCoral = Color(red: 0.69, green: 0.176, blue: 0.153)
    static let facadeAmber = Color(red: 0.502, green: 0.302, blue: 0.035)
    static let facadeText = Color(red: 0.067, green: 0.122, blue: 0.204)
    static let facadeMuted = Color(red: 0.318, green: 0.365, blue: 0.431)
    static let facadeHairline = Color(red: 0.824, green: 0.847, blue: 0.882)
    static let facadeHairlineBright = Color.white.opacity(0.14)
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
