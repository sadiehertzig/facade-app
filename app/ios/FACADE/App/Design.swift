import SwiftUI
import UIKit

struct PrimaryButtonStyle: ButtonStyle {
    @Environment(\.isEnabled) private var enabled
    func makeBody(configuration: Configuration) -> some View {
        configuration.label.font(.headline).frame(maxWidth: .infinity, minHeight: 52)
            .padding(.horizontal, 12).foregroundStyle(.white)
            .background(LinearGradient.facadeBrand.opacity(enabled ? (configuration.isPressed ? 0.8 : 1) : 0.4),
                        in: RoundedRectangle(cornerRadius: 14, style: .continuous))
            .scaleEffect(configuration.isPressed ? 0.98 : 1)
            .animation(.easeOut(duration: 0.12), value: configuration.isPressed)
    }
}

struct SecondaryButtonStyle: ButtonStyle {
    @Environment(\.isEnabled) private var enabled
    func makeBody(configuration: Configuration) -> some View {
        configuration.label.font(.subheadline.weight(.semibold)).frame(maxWidth: .infinity, minHeight: 48)
            .padding(.horizontal, 12)
            .foregroundStyle(Color.facadeViolet.opacity(enabled ? 1 : 0.4))
            .background(Color.facadeSurfaceRaised.opacity(configuration.isPressed ? 0.6 : 1),
                        in: RoundedRectangle(cornerRadius: 14, style: .continuous))
    }
}

extension LinearGradient {
    static let facadeBrand = LinearGradient(colors: [.facadeVioletBright, .facadeViolet],
                                            startPoint: .topLeading, endPoint: .bottomTrailing)
}

extension View {
    func fastCard() -> some View {
        self.padding(18).frame(maxWidth: .infinity, alignment: .leading)
            .background(Color.facadeSurface, in: RoundedRectangle(cornerRadius: 20, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 20, style: .continuous).strokeBorder(Color.facadeHairline, lineWidth: 1))
            .shadow(color: Color.facadeViolet.opacity(0.06), radius: 12, y: 4)
            .foregroundStyle(Color.facadeText)
    }
}

/// Tinted rounded-square icon used on section and resource headers.
struct IconBadge: View {
    let systemImage: String
    var tint: Color = .facadeViolet
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
                .frame(width: 26, height: 26).background(Color.facadeViolet, in: Circle())
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
                    Capsule().fill(index < current ? Color.facadeViolet : Color.facadeHairline).frame(height: 4)
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
                    .foregroundStyle(Color.facadeViolet).frame(width: 36, height: 36)
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
            .background(.black.opacity(0.45), in: Capsule())
            .padding(8)
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: alignment)
            .allowsHitTesting(false)
    }
}
