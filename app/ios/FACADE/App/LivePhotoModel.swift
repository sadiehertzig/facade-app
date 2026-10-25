// LivePhotoModel.swift
//
// The state and logic behind the Protect tab: which photo is loaded, which
// protection method is chosen, whether something is running, the protected
// result, the optional face-swap comparison, and saving to Photos.
//
// Where it sits: FACADEDemoScreen (PhotoAcquisitionViews.swift) owns one of
// these and redraws whenever a `@Published` property changes. The model in
// turn calls the two engines, LivePhotoEngine (FACADE) and
// MiniPhotoGuardEngine (Mini PhotoGuard), and the Face-Off renderer
// (FACADEFaceOffRenderer.swift).
//
// Key ideas:
// 1. Privacy: the chosen photo and its results are held in memory. Nothing
//    is uploaded, and the only thing added to the photo library is a new
//    picture, when the user taps Save. Leaving the app clears everything.
// 2. One job at a time. `busy` blocks new work while a protection run, a
//    face-swap comparison or a save is in progress.
// 3. Every job gets a fresh `operationID`. Results and progress updates that
//    arrive from an older, cancelled job carry the old ID and are ignored, so
//    a slow background task can never overwrite what the user is now doing.
// 4. The engines run off the main thread and report progress through a
//    callback; the model hops each update back onto the main actor before
//    touching any published state.
import Foundation
import Combine
import UIKit
import Photos
import UniformTypeIdentifiers

/// The Protect tab's view model. It runs on the main actor (the UI thread), so
/// SwiftUI can read its published properties safely. Personal pixels remain in
/// memory; only an explicit save adds a new Photos asset.
@MainActor final class LivePhotoModel: ObservableObject {
    /// Where the Protect flow is. The screen picks which cards to show from this.
    /// - `idle`: nothing loaded. `photoReady`: a photo is prepared and shown for
    ///   review. `processingLocally`: protecting the user's own photo.
    /// - `processingBundledFictionalAdult` / `bundledFictionalAdultReady`: the
    ///   same, for the built-in fictional sample face. Sample results cannot be
    ///   saved.
    /// - `verifiedAwaitingExplicitSave`: a protected copy of the user's photo is
    ///   ready and waits for the user to tap Save. `savingProtectedCopy`, then
    ///   `saved`.
    /// - `blocked(message)`: stopped, with a message for the status banner.
    /// - `validatingBundledFixture` and `bundledFixtureValidated` are never set
    ///   by the current code.
    enum Stage: Equatable {
        case idle, photoReady, processingLocally, savingProtectedCopy
        case processingBundledFictionalAdult, validatingBundledFixture
        case bundledFictionalAdultReady, bundledFixtureValidated(String)
        case verifiedAwaitingExplicitSave, saved, blocked(String)
    }
    /// Progress of the optional face-swap comparison, separate from `stage`
    /// because it runs after a result already exists.
    enum FaceOffState: Equatable { case idle, rendering, ready, failed(String) }

    // Which method the user picked, and which method produced the result on
    // screen. The result card reads `resultMethod`, so it always names the
    // method that actually ran. `private(set)` means views can read these
    // but only the model can change them.
    @Published private(set) var selectedMethod: ProtectionMethod = .facade
    @Published private(set) var resultMethod: ProtectionMethod = .facade
    /// True if the chosen method's model files are installed in this build. The
    /// Protect button is disabled otherwise.
    var selectedMethodAvailable: Bool {
        selectedMethod == .facade ? LivePhotoEngine.resourcesAvailable : MiniPhotoGuardEngine.resourcesAvailable
    }

    @Published private(set) var stage: Stage = .idle
    /// PNG shown in the review card: the prepared photo before protection, then
    /// the protected copy afterwards.
    @Published private(set) var previewData: Data?
    /// The finished result (original PNG, protected PNG, size and timings), or
    /// nil when there is none.
    @Published private(set) var demoResult: LivePhotoResult?
    /// The PNG that Save will write. It is nil for the fictional sample, which is
    /// why the screen offers no Save button for it.
    @Published private(set) var verifiedCopy: Data?
    // Face-swap comparison: its state, the full swapped photos made from the
    // original and from the protected copy, and the total time both swaps took.
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
    /// True while a protection run, comparison or save is in progress. Most
    /// actions return early while it is set.
    @Published private(set) var busy = false
    @Published private(set) var elapsedSeconds = 0.0
    /// A short description of what the engine is doing, built by `stageName`.
    @Published private(set) var progressText = "Ready"
    // Fixed descriptive values. None of these is read by the current screens;
    // the screens write their own wording.
    let runtimeReady = true
    let runtimeStatus = "FACADE experimental engine · 10 coarse + 6 fine steps · K2 cadence · on-device"
    let experimentalLabel = "Experimental — protection is not guaranteed."
    let limitations = "The shorter engine is an experimental choice. The original full comparison was incomplete. This live InSwapper view is one local example, not a general protection test. Your photos are never uploaded."
    let bundledFixtureValidationConfigured = false
    let bundledFixtureValidationAvailable = false
    let bundledFictionalAdultAvailable = true
    // The photo the user chose, exactly as it was loaded (not the prepared
    // version). It is kept so that switching methods can prepare it again, and
    // so the engine can prepare it itself when Protect is tapped.
    private var preparedInput: Data?
    private var preparedIsSample = false
    /// Prepared size as text, for example "768 × 1024".
    @Published private(set) var preparedSize = ""
    // Bookkeeping for the current job: the task doing the work, the task that
    // ticks the elapsed-time clock, the job's ID, and whether the user asked to
    // stop. `saving` makes a save impossible to cancel halfway.
    private var operation: Task<Void, Never>?
    private var clockTask: Task<Void, Never>?
    private var operationID = UUID()
    private var cancellationRequested = false
    private var saving = false

    /// Entry point for a photo from the photo picker or the selfie camera. Both
    /// flags must be true: the user picked the photo themselves and confirmed
    /// that everyone in it is a consenting adult.
    /// `uniformTypeIdentifier` (the file type) is accepted but not used; the
    /// image decoder works the type out from the bytes.
    func startImportedPhoto(data: Data, uniformTypeIdentifier: String,
                            userPickedPhoto: Bool, adultSubjectsConsented: Bool) {
        guard !busy else { return }
        guard userPickedPhoto, adultSubjectsConsented else {
            block("Choose a photo of consenting adults."); return
        }
        prepareForReview(data: data, sample: false)
    }

    /// Prepares the photo exactly the way the chosen engine will, and shows it for
    /// review. The original bytes stay only in memory until Protect or discard.
    ///
    /// FACADE limits the long edge to 1,024 pixels and applies the photo's
    /// orientation; Mini PhotoGuard instead pads the photo to a 256 × 256 square.
    /// The review shows exactly what the chosen method will work on.
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
        // If preparation throws (the bytes are not a readable image, or for FACADE
        // the photo is under 128 pixels on a side), the catch shows one general message.
        } catch { block("This photo could not be prepared. Choose a clear photo with one face.") }
    }

    /// Loads the built-in fictional sample face and shows it for review, as if
    /// the user had chosen it. Builds without the sample show a warning instead.
    func prepareSampleForReview() {
        guard !busy else { return }
        do { prepareForReview(data: try Self.interactiveSamplePNG(), sample: true) }
        catch { block(error.localizedDescription) }
    }

    /// Switches between FACADE and Mini PhotoGuard. Locked while busy or after a
    /// result exists, so a result is always labelled with the method that made
    /// it. A loaded photo is prepared again, because the two methods prepare
    /// photos differently.
    func selectMethod(_ method: ProtectionMethod) {
        guard !busy, demoResult == nil else { return }
        selectedMethod = method
        if let data = preparedInput {
            let sample = preparedIsSample
            prepareForReview(data: data, sample: sample)
        }
    }

    /// Called by the Protect button. Only works when a reviewed photo is waiting
    /// and the chosen method's models are installed.
    func protectPreparedPhoto() {
        guard !busy, selectedMethodAvailable, stage == .photoReady, let data = preparedInput else { return }
        let sample = preparedIsSample
        start(data: data, sample: sample)
    }

    /// "Try a sample" face. The sample image is private and not in git; put
    /// one at app/ios/Models/SampleFace/sample.png (see Models/README.md).
    /// The build copies it into the app bundle's SampleFace folder when it exists
    /// in Models/.
    private static func interactiveSamplePNG() throws -> Data {
        guard let url = Bundle.main.url(forResource: "sample", withExtension: "png", subdirectory: "SampleFace"),
              let data = try? Data(contentsOf: url), !data.isEmpty else {
            throw NativeError.invalid("No sample photo is included in this build. Choose your own photo instead.")
        }
        return data
    }

    /// Common start of a protection run: clears old images, issues a new
    /// operation ID, marks the model busy and starts the elapsed-time clock.
    /// Returns the new ID so the caller can check later that it is still current.
    private func beginOperation(_ text: String) -> UUID {
        clearImages()
        operationID = UUID()
        cancellationRequested = false
        busy = true
        elapsedSeconds = 0
        progressText = text
        // Keeps the screen from auto-locking during a long run. Locking would send
        // the app to the background, which stops processing (see
        // `cancelForBackgrounding`).
        UIApplication.shared.isIdleTimerDisabled = true
        let identifier = operationID
        // `systemUptime` is a monotonic clock: it is not affected by the user or the
        // network changing the time of day.
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

    /// Runs the chosen protection method on `data` in a background task and
    /// publishes the result. The method is captured at the start, so the result
    /// is labelled with the method that actually ran.
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
                // The engines call this from their own background thread. Each update is
                // moved onto the main actor and dropped if this job has been cancelled or
                // replaced in the meantime.
                let reportProgress: @Sendable (LivePhotoProgress) -> Void = { [weak self] progress in
                    Task { @MainActor [weak self] in
                        guard let self, self.busy, self.operationID == identifier,
                              !self.cancellationRequested else { return }
                        self.progressText = Self.stageName(progress.stage)
                        // Step 0 means "no step count for this stage" (for example while loading
                        // models), so the "step n of m" suffix is shown only for real steps.
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
                // The engine has returned. Check once more that the user has not cancelled
                // and that no newer job has started before showing this result.
                try Task.checkCancellation()
                guard operationID == identifier, !cancellationRequested else { throw CancellationError() }
                resultMethod = chosenMethod
                demoResult = result
                previewData = result.protectedPNG
                // Only the user's own photo can be saved; the sample is for trying the app.
                verifiedCopy = sample ? nil : result.protectedPNG
                elapsedSeconds = result.elapsedSeconds
                progressText = "Completed"
                stage = sample ? .bundledFictionalAdultReady : .verifiedAwaitingExplicitSave
            } catch {
                // On any failure or cancel, nothing partial is kept on screen.
                clearImages()
                stage = .blocked(cancellationRequested || error is CancellationError
                    ? "Processing stopped. Nothing was saved."
                    : error.localizedDescription)
            }
            finishOperation(identifier)
        }
    }

    /// Common end of a job: stops the clock, clears the busy flag and lets the
    /// screen auto-lock again. Does nothing if a newer job has already started.
    private func finishOperation(_ identifier: UUID) {
        guard operationID == identifier else { return }
        clockTask?.cancel()
        clockTask = nil
        operation = nil
        busy = false
        UIApplication.shared.isIdleTimerDisabled = false
    }

    /// Asks the running job to stop. The engines check for cancellation between
    /// model calls, so the current call finishes first. Ignored during a save.
    func cancelProcessing() {
        guard busy, !saving else { return }
        cancellationRequested = true
        progressText = "Stopping after the current model call…"
        operation?.cancel()
        // `busy` stays true until the job's task has actually finished
        // (`finishOperation`), so a new job cannot start while the old one is still
        // using the models.
    }

    /// Called when the app goes to the background. A running job is stopped and
    /// any loaded photo or result is wiped from memory, so it is not left behind
    /// in a suspended app. A save that has started is allowed to finish.
    func cancelForBackgrounding() {
        if saving { return }
        if busy { cancelProcessing() }
        else if demoResult != nil || preparedInput != nil {
            clearImages()
            stage = .blocked("The in-memory preview was cleared when the app left the foreground.")
        }
    }

    /// "Choose another photo" / "Discard without saving": throws away the photo
    /// and any result and returns to the start. While busy it cancels instead.
    func discardCandidate() {
        guard !busy else { cancelProcessing(); return }
        clearImages()
        stage = .idle
        progressText = "Ready"
        elapsedSeconds = 0
    }

    /// Clears everything and shows `message` in the status banner. Ignored while
    /// busy, so a late message cannot interrupt a running job.
    func block(_ message: String) {
        guard !busy else { return }
        clearImages()
        stage = .blocked(message)
    }

    /// Drops every photo, result and comparison image the model holds, so they
    /// can be freed from memory.
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

    /// Cosine similarity of two face-recognition vectors: 1 means they point the
    /// same way (the recognizer thinks it is the same face), values near 0 mean
    /// unrelated. Returns nil for mismatched or empty vectors, or if a vector is
    /// all zeros (the division would not give a finite number).
    private static func cosine(_ a: [Float], _ b: [Float]) -> Double? {
        guard a.count == b.count, !a.isEmpty else { return nil }
        // Dot product and the two squared lengths, summed in Double for accuracy.
        var dot = 0.0, na = 0.0, nb = 0.0
        for i in a.indices {
            dot += Double(a[i]) * Double(b[i]); na += Double(a[i]) * Double(a[i]); nb += Double(b[i]) * Double(b[i])
        }
        let value = dot / (na.squareRoot() * nb.squareRoot())
        return value.isFinite ? value : nil
    }

    /// Adds the protected PNG to Photos as a new picture, after the user has
    /// confirmed in a dialog. The original photo in the library is never changed.
    func saveVerifiedCopy(userConfirmedSave: Bool) async {
        guard !busy, userConfirmedSave, let png = verifiedCopy else { return }
        saving = true
        busy = true
        stage = .savingProtectedCopy
        defer { saving = false; busy = false }
        do {
            // "Add only" permission lets the app add pictures without being able to see
            // the rest of the user's library.
            let access = await PHPhotoLibrary.requestAuthorization(for: .addOnly)
            guard access == .authorized || access == .limited else {
                throw NativeError.invalid("Allow Photos access to add a new copy.")
            }
            try await PHPhotoLibrary.shared().performChanges {
                let options = PHAssetResourceCreationOptions()
                // A random file name, so nothing about the person or the photo is in it.
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

    /// The optional "See it against a face swap" comparison. It runs the
    /// on-device face swapper on both the original and the protected copy, onto
    /// the same fictional target face, and measures how similar each swapped
    /// face is to the original person.
    ///
    /// The similarity comes from the swapper's own recognizer, which FACADE also
    /// targets, so it is a single illustration, not independent proof that
    /// protection works.
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
            // A detached task runs the heavy swapping off the main actor. Unlike a
            // normal child task it does not inherit cancellation, so the cancellation
            // handler below forwards a cancel to it by hand.
            let work = Task.detached(priority: .userInitiated) { () throws -> (FACADEFaceOffRenderedResult, FACADEFaceOffRenderedResult) in
                try Task.checkCancellation()
                // A fresh renderer is created for each comparison, so nothing from an
                // earlier comparison is reused.
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
                // Both scores compare against the recognizer vector of the original,
                // unprotected photo: "how much does each swap still look like you?"
                faceOffUnprotectedMatch = Self.cosine(clean.sourceEmbedding, clean.swapEmbedding)
                faceOffProtectedMatch = Self.cosine(clean.sourceEmbedding, shielded.swapEmbedding)
                faceOffRuntimeSeconds = clean.runtimeSeconds + shielded.runtimeSeconds
                faceOffState = .ready
            } catch {
                faceOffState = .failed(cancellationRequested || error is CancellationError
                    ? "The comparison was stopped." : error.localizedDescription)
                // A cancelled comparison is treated like leaving: everything is cleared.
                // A comparison that failed on its own keeps the protected result on screen.
                if cancellationRequested { clearImages(); stage = .blocked("The comparison was stopped. Nothing was saved.") }
            }
            finishOperation(identifier)
        }
    }

    /// Turns an engine's internal stage name into the short text shown on the
    /// progress card. Names not in the table (including Mini PhotoGuard's, which
    /// are already written as sentences) are tidied up instead: underscores and
    /// dots become spaces and each word is capitalised.
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
