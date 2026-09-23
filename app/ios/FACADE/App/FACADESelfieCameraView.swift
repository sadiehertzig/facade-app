#if os(iOS)
import SwiftUI
import UniformTypeIdentifiers
import UIKit

/// Front-camera capture presented as a sheet. It hands back only the captured
/// still as normalized JPEG bytes via `onCapture`; nothing is written to Photos
/// and the original selfie is never saved. The delivered `Data` flows into the
/// SAME protect-only path as a system-picker selection
/// (`FACADEDemoViewModel.startImportedPhoto`), so the camera adds a capture
/// source, not a new data path.
struct FACADESelfieCameraView: UIViewControllerRepresentable {
    let onCapture: (Data) -> Void
    @Environment(\.dismiss) private var dismiss

    func makeCoordinator() -> Coordinator {
        Coordinator(onCapture: onCapture, dismiss: dismiss)
    }

    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        // Presenting a picker with an unavailable `.camera` source raises
        // NSInvalidArgumentException and crashes; only set it when the camera
        // source actually exists (it never does on the Simulator, and may be
        // restricted by Screen Time / MDM). Callers also gate the button on this.
        if UIImagePickerController.isSourceTypeAvailable(.camera) {
            picker.sourceType = .camera
            picker.cameraCaptureMode = .photo
            if UIImagePickerController.isCameraDeviceAvailable(.front) {
                picker.cameraDevice = .front
            }
        }
        picker.mediaTypes = [UTType.image.identifier]
        picker.allowsEditing = false
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(
        _ uiViewController: UIImagePickerController,
        context: Context
    ) {}

    final class Coordinator: NSObject,
        UINavigationControllerDelegate,
        UIImagePickerControllerDelegate
    {
        private let onCapture: (Data) -> Void
        private let dismiss: DismissAction

        init(onCapture: @escaping (Data) -> Void, dismiss: DismissAction) {
            self.onCapture = onCapture
            self.dismiss = dismiss
        }

        func imagePickerController(
            _ picker: UIImagePickerController,
            didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]
        ) {
            defer { dismiss() }
            guard let image = info[.originalImage] as? UIImage,
                  let data = Self.normalizedJPEG(from: image) else {
                return
            }
            onCapture(data)
        }

        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            dismiss()
        }

        private static func normalizedJPEG(from image: UIImage) -> Data? {
            let maximumDimension: CGFloat = 2_048
            let longestDimension = max(image.size.width, image.size.height)
            let scale = longestDimension > maximumDimension
                ? maximumDimension / longestDimension
                : 1
            let outputSize = CGSize(
                width: max(1, (image.size.width * scale).rounded()),
                height: max(1, (image.size.height * scale).rounded())
            )
            let format = UIGraphicsImageRendererFormat.default()
            format.scale = 1
            format.opaque = true
            let rendered = UIGraphicsImageRenderer(
                size: outputSize,
                format: format
            ).image { _ in
                image.draw(in: CGRect(origin: .zero, size: outputSize))
            }
            return rendered.jpegData(compressionQuality: 0.95)
        }
    }
}
#endif
