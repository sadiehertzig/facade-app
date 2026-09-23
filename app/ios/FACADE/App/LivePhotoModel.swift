import Foundation
import Combine
import UIKit
import Photos
import UniformTypeIdentifiers

/// Adapts the copied demo screens to the new engine. Personal pixels remain in
/// memory; only an explicit save adds a new Photos asset.
@MainActor final class LivePhotoModel: ObservableObject {
    enum Stage: Equatable {
        case idle, photoReady, processingLocally, savingProtectedCopy
        case processingBundledFictionalAdult, validatingBundledFixture
        case bundledFictionalAdultReady, bundledFixtureValidated(String)
        case verifiedAwaitingExplicitSave, saved, blocked(String)
    }
    enum FaceOffState: Equatable { case idle, rendering, ready, failed(String) }

    @Published private(set) var selectedMethod: ProtectionMethod = .facade
    @Published private(set) var resultMethod: ProtectionMethod = .facade
    var selectedMethodAvailable: Bool {
        selectedMethod == .facade ? LivePhotoEngine.resourcesAvailable : MiniPhotoGuardEngine.resourcesAvailable
    }

    @Published private(set) var stage: Stage = .idle
    @Published private(set) var previewData: Data?
    @Published private(set) var demoResult: LivePhotoResult?
    @Published private(set) var verifiedCopy: Data?
    @Published private(set) var faceOffState: FaceOffState = .idle
    @Published private(set) var faceOffUnprotectedSwap: Data?
    @Published private(set) var faceOffProtectedSwap: Data?
    @Published private(set) var faceOffRuntimeSeconds: Double?
    /// Close-ups of the two swapped faces and display-only similarity of each swap
    /// to the unprotected source, from the swap tool's own recognizer.
    @Published private(set) var faceOffUnprotectedFace: Data?
    @Published private(set) var faceOffProtectedFace: Data?
    @Published private(set) var faceOffUnprotectedMatch: Double?
    @Published private(set) var faceOffProtectedMatch: Double?
    @Published private(set) var busy = false
    @Published private(set) var elapsedSeconds = 0.0
    @Published private(set) var progressText = "Ready"
    let runtimeReady = true
    let runtimeStatus = "FACADE experimental engine · 10 coarse + 6 fine steps · K2 cadence · on-device"
    let experimentalLabel = "Experimental — protection is not guaranteed."
    let limitations = "The shorter engine is an experimental choice. The original full comparison was incomplete. This live InSwapper view is one local example, not a general protection test. Your photos are never uploaded."
    let bundledFixtureValidationConfigured = false
    let bundledFixtureValidationAvailable = false
    let bundledFictionalAdultAvailable = true
    private var preparedInput: Data?
    private var preparedIsSample = false
    @Published private(set) var preparedSize = ""
    private var operation: Task<Void, Never>?
    private var clockTask: Task<Void, Never>?
    private var operationID = UUID()
    private var cancellationRequested = false
    private var saving = false

    func startImportedPhoto(data: Data, uniformTypeIdentifier: String,
                            userPickedPhoto: Bool, adultSubjectsConsented: Bool) {
        guard !busy else { return }
        guard userPickedPhoto, adultSubjectsConsented else {
            block("Choose a photo of consenting adults."); return
        }
        prepareForReview(data: data, sample: false)
    }

    /// Review uses the same bounded, oriented preparation as the unchanged engine.
    /// Original bytes are retained only in memory until explicit Protect or discard.
    private func prepareForReview(data: Data, sample: Bool) {
        guard !busy else { return }
        clearImages()
        do {
            let prepared = try selectedMethod == .facade
                ? LivePhotoEngine.fixturePreparedOriginal(data)
                : MiniPhotoGuardEngine.preparePhoto(data: data)
            preparedInput = data
            preparedIsSample = sample
            previewData = prepared.png
            preparedSize = "\(prepared.width) × \(prepared.height)"
            stage = .photoReady
            progressText = "Ready to protect"
        } catch { block("This photo could not be prepared. Choose a clear photo with one face.") }
    }

    func prepareSampleForReview() {
        guard !busy else { return }
        do { prepareForReview(data: try Self.interactiveSamplePNG(), sample: true) }
        catch { block(error.localizedDescription) }
    }

    func selectMethod(_ method: ProtectionMethod) {
        guard !busy, demoResult == nil else { return }
        selectedMethod = method
        if let data = preparedInput {
            let sample = preparedIsSample
            prepareForReview(data: data, sample: sample)
        }
    }

    func protectPreparedPhoto() {
        guard !busy, selectedMethodAvailable, stage == .photoReady, let data = preparedInput else { return }
        let sample = preparedIsSample
        start(data: data, sample: sample)
    }

    /// "Try a sample" face. The sample image is private and not in git; put
    /// one at app/ios/Models/SampleFace/sample.png (see Models/README.md).
    private static func interactiveSamplePNG() throws -> Data {
        guard let url = Bundle.main.url(forResource: "sample", withExtension: "png", subdirectory: "SampleFace"),
              let data = try? Data(contentsOf: url), !data.isEmpty else {
            throw NativeError.invalid("No sample photo is included in this build. Choose your own photo instead.")
        }
        return data
    }

    private func beginOperation(_ text: String) -> UUID {
        clearImages()
        operationID = UUID()
        cancellationRequested = false
        busy = true
        elapsedSeconds = 0
        progressText = text
        UIApplication.shared.isIdleTimerDisabled = true
        let identifier = operationID
        let start = ProcessInfo.processInfo.systemUptime
        clockTask = Task { [weak self] in
            while !Task.isCancelled {
                guard let self, self.busy, self.operationID == identifier else { return }
                self.elapsedSeconds = ProcessInfo.processInfo.systemUptime - start
                do { try await Task.sleep(nanoseconds: 250_000_000) } catch { return }
            }
        }
        return identifier
    }

    private func start(data: Data, sample: Bool) {
        guard !busy else { return }
        let chosenMethod = selectedMethod
        guard chosenMethod != .facade || LivePhotoEngine.resourcesAvailable else {
            block("FACADE's on-device models are not installed in this build."); return
        }
        let identifier = beginOperation("Preparing photo")
        stage = sample ? .processingBundledFictionalAdult : .processingLocally
        operation = Task { [weak self] in
            guard let self else { return }
            do {
                let reportProgress: @Sendable (LivePhotoProgress) -> Void = { [weak self] progress in
                    Task { @MainActor [weak self] in
                        guard let self, self.busy, self.operationID == identifier,
                              !self.cancellationRequested else { return }
                        self.progressText = Self.stageName(progress.stage)
                        if progress.step > 0 {
                            self.progressText += " · step \(progress.step) of \(progress.totalSteps)"
                        }
                        self.elapsedSeconds = progress.elapsedSeconds
                    }
                }
                let result: LivePhotoResult
                if chosenMethod == .facade {
                    result = try await LivePhotoEngine.shared.run(data: data, progress: reportProgress)
                } else {
                    result = try await MiniPhotoGuardEngine.shared.run(data: data, progress: reportProgress)
                }
                try Task.checkCancellation()
                guard operationID == identifier, !cancellationRequested else { throw CancellationError() }
                resultMethod = chosenMethod
                demoResult = result
                previewData = result.protectedPNG
                verifiedCopy = sample ? nil : result.protectedPNG
                elapsedSeconds = result.elapsedSeconds
                progressText = "Completed"
                stage = sample ? .bundledFictionalAdultReady : .verifiedAwaitingExplicitSave
            } catch {
                clearImages()
                stage = .blocked(cancellationRequested || error is CancellationError
                    ? "Processing stopped. Nothing was saved."
                    : error.localizedDescription)
            }
            finishOperation(identifier)
        }
    }

    private func finishOperation(_ identifier: UUID) {
        guard operationID == identifier else { return }
        clockTask?.cancel()
        clockTask = nil
        operation = nil
        busy = false
        UIApplication.shared.isIdleTimerDisabled = false
    }

    func cancelProcessing() {
        guard busy, !saving else { return }
        cancellationRequested = true
        progressText = "Stopping after the current model call…"
        operation?.cancel()
        // Busy remains true until the adapter has joined its cancelled worker.
    }

    func cancelForBackgrounding() {
        if saving { return }
        if busy { cancelProcessing() }
        else if demoResult != nil || preparedInput != nil {
            clearImages()
            stage = .blocked("The in-memory preview was cleared when the app left the foreground.")
        }
    }

    func discardCandidate() {
        guard !busy else { cancelProcessing(); return }
        clearImages()
        stage = .idle
        progressText = "Ready"
        elapsedSeconds = 0
    }

    func block(_ message: String) {
        guard !busy else { return }
        clearImages()
        stage = .blocked(message)
    }

    private func clearImages() {
        preparedInput = nil
        preparedIsSample = false
        preparedSize = ""
        previewData = nil
        demoResult = nil
        verifiedCopy = nil
        faceOffState = .idle
        faceOffUnprotectedSwap = nil
        faceOffProtectedSwap = nil
        faceOffRuntimeSeconds = nil
        faceOffUnprotectedFace = nil
        faceOffProtectedFace = nil
        faceOffUnprotectedMatch = nil
        faceOffProtectedMatch = nil
    }

    private static func cosine(_ a: [Float], _ b: [Float]) -> Double? {
        guard a.count == b.count, !a.isEmpty else { return nil }
        var dot = 0.0, na = 0.0, nb = 0.0
        for i in a.indices {
            dot += Double(a[i]) * Double(b[i]); na += Double(a[i]) * Double(a[i]); nb += Double(b[i]) * Double(b[i])
        }
        let value = dot / (na.squareRoot() * nb.squareRoot())
        return value.isFinite ? value : nil
    }

    func saveVerifiedCopy(userConfirmedSave: Bool) async {
        guard !busy, userConfirmedSave, let png = verifiedCopy else { return }
        saving = true
        busy = true
        stage = .savingProtectedCopy
        defer { saving = false; busy = false }
        do {
            let access = await PHPhotoLibrary.requestAuthorization(for: .addOnly)
            guard access == .authorized || access == .limited else {
                throw NativeError.invalid("Allow Photos access to add a new copy.")
            }
            try await PHPhotoLibrary.shared().performChanges {
                let options = PHAssetResourceCreationOptions()
                options.originalFilename = "FACADE-\(UUID().uuidString).png"
                PHAssetCreationRequest.forAsset().addResource(with: .photo, data: png, options: options)
            }
            clearImages()
            stage = .saved
        } catch {
            // Retain the in-memory result so permission can be retried or it can be discarded.
            stage = .blocked("The new copy was not saved: \(error.localizedDescription)")
        }
    }

    func renderFaceOff(for result: LivePhotoResult) {
        guard !busy, demoResult != nil, faceOffState != .rendering else { return }
        let identifier = UUID()
        operationID = identifier
        busy = true
        cancellationRequested = false
        faceOffState = .rendering
        faceOffUnprotectedSwap = nil
        faceOffProtectedSwap = nil
        faceOffRuntimeSeconds = nil
        faceOffUnprotectedFace = nil
        faceOffProtectedFace = nil
        faceOffUnprotectedMatch = nil
        faceOffProtectedMatch = nil
        UIApplication.shared.isIdleTimerDisabled = true
        let original = result.originalPNG, protected = result.protectedPNG
        operation = Task { [weak self] in
            guard let self else { return }
            let work = Task.detached(priority: .userInitiated) { () throws -> (FACADEFaceOffRenderedResult, FACADEFaceOffRenderedResult) in
                try Task.checkCancellation()
                let renderer = try FACADEOnDeviceFaceOffRenderer.load()
                let originalSwap = try await renderer.render(sourceImageData: original)
                try Task.checkCancellation()
                let protectedSwap = try await renderer.render(sourceImageData: protected)
                try Task.checkCancellation()
                return (originalSwap, protectedSwap)
            }
            do {
                let rendered = try await withTaskCancellationHandler {
                    try await work.value
                } onCancel: { work.cancel() }
                try Task.checkCancellation()
                guard operationID == identifier, !cancellationRequested else { throw CancellationError() }
                let (clean, shielded) = rendered
                faceOffUnprotectedSwap = clean.attemptedSwapData
                faceOffProtectedSwap = shielded.attemptedSwapData
                faceOffUnprotectedFace = clean.alignedSwapFaceData
                faceOffProtectedFace = shielded.alignedSwapFaceData
                faceOffUnprotectedMatch = Self.cosine(clean.sourceEmbedding, clean.swapEmbedding)
                faceOffProtectedMatch = Self.cosine(clean.sourceEmbedding, shielded.swapEmbedding)
                faceOffRuntimeSeconds = clean.runtimeSeconds + shielded.runtimeSeconds
                faceOffState = .ready
            } catch {
                faceOffState = .failed(cancellationRequested || error is CancellationError
                    ? "The comparison was stopped." : error.localizedDescription)
                if cancellationRequested { clearImages(); stage = .blocked("The comparison was stopped. Nothing was saved.") }
            }
            finishOperation(identifier)
        }
    }

    static func stageName(_ stage: String) -> String {
        switch stage {
        case "prepare", "preparation": return "Preparing photo"
        case "models.load", "load": return "Loading on-device models"
        case "coverage": return "Building initial protection"
        case "native_lift": return "Restoring image detail"
        case "native_eot8": return "Refining image"
        case "native_realign8": return "Refining alignment"
        case "energy80": return "Balancing image changes"
        case "tight": return "Refining compression handling"
        case "masked": return "Refining face detail"
        case "balanced_continue": return "Finishing image"
        case "export", "encode": return "Preparing PNG"
        default: return stage.replacingOccurrences(of: "_", with: " ").replacingOccurrences(of: ".", with: " ").capitalized
        }
    }
}
