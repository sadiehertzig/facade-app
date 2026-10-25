// FACADESelfieCameraView.swift
//
// The "Take a selfie" camera. It shows Apple's built-in camera screen, set to
// the front camera, and hands the captured photo back to the Protect screen.
//
// Where it sits: PhotoAcquisitionViews.swift presents this as a sheet. The
// photo it returns goes to LivePhotoModel.startImportedPhoto, exactly like a
// photo chosen from the library.
//
// Key ideas:
// 1. SwiftUI has no camera view of its own, so this wraps UIKit's
//    UIImagePickerController with `UIViewControllerRepresentable`. A small
//    Coordinator object receives UIKit's delegate callbacks.
// 2. The selfie is never saved to Photos by this screen. Only the bytes are
//    passed on, after being redrawn upright and shrunk to at most 2048 pixels.
// 3. The whole file is compiled only for iOS (`#if os(iOS)`), because
//    UIImagePickerController's camera does not exist on the Mac.
#if os(iOS)
import SwiftUI
import UniformTypeIdentifiers
import UIKit

/// Front-camera capture presented as a sheet. It hands back only the captured
/// still as normalized JPEG bytes via `onCapture`; nothing is written to Photos
/// and the original selfie is never saved. The delivered `Data` flows into the
/// SAME protect-only path as a system-picker selection
/// (`LivePhotoModel.startImportedPhoto`), so the camera adds a capture
/// source, not a new data path.
struct FACADESelfieCameraView: UIViewControllerRepresentable {
    /// Called once with the JPEG bytes of the captured photo.
    let onCapture: (Data) -> Void
    /// Closes the sheet; passed to the Coordinator so it can close it from UIKit.
    @Environment(\.dismiss) private var dismiss

    /// SwiftUI calls this once to create the object that will act as the picker's delegate.
    func makeCoordinator() -> Coordinator {
        Coordinator(onCapture: onCapture, dismiss: dismiss)
    }

    /// Builds and configures the UIKit camera screen.
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
        // Still photos only (no video), and no built-in crop step.
        picker.mediaTypes = [UTType.image.identifier]
        picker.allowsEditing = false
        picker.delegate = context.coordinator
        return picker
    }

    /// Required by the protocol. Nothing changes after creation, so it is empty.
    func updateUIViewController(
        _ uiViewController: UIImagePickerController,
        context: Context
    ) {}

    /// Receives the picker's callbacks (photo taken, or cancelled). UIKit's
    /// delegate protocols need an NSObject subclass, which is why this is a class.
    final class Coordinator: NSObject,
        UINavigationControllerDelegate,
        UIImagePickerControllerDelegate
    {
        /// Copies of the two closures from the SwiftUI view.
        private let onCapture: (Data) -> Void
        private let dismiss: DismissAction

        init(onCapture: @escaping (Data) -> Void, dismiss: DismissAction) {
            self.onCapture = onCapture
            self.dismiss = dismiss
        }

        /// The user took a photo and tapped "Use Photo".
        func imagePickerController(
            _ picker: UIImagePickerController,
            didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]
        ) {
            // The sheet closes in every case, even if the photo cannot be converted.
            defer { dismiss() }
            // If conversion fails nothing is passed on; the user simply sees no new photo.
            guard let image = info[.originalImage] as? UIImage,
                  let data = Self.normalizedJPEG(from: image) else {
                return
            }
            onCapture(data)
        }

        /// The user tapped Cancel: close the sheet without a photo.
        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            dismiss()
        }

        /// Redraws the photo upright, scales it so its longest side is at most 2048
        /// pixels, and encodes it as JPEG. Redrawing bakes the camera's orientation
        /// flag into the pixels, so later code never has to handle rotated selfies.
        private static func normalizedJPEG(from image: UIImage) -> Data? {
            // 2048 matches RGBImage's size limit in NativeTensor.swift.
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
            // Scale 1 means one output pixel per point, so outputSize is in pixels.
            // Opaque: a photo has no transparency, and this drops the alpha channel.
            format.scale = 1
            format.opaque = true
            let rendered = UIGraphicsImageRenderer(
                size: outputSize,
                format: format
            ).image { _ in
                image.draw(in: CGRect(origin: .zero, size: outputSize))
            }
            // Quality 0.95 keeps compression loss small before protection starts.
            return rendered.jpegData(compressionQuality: 0.95)
        }
    }
}
#endif
