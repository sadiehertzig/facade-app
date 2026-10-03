import PhotosUI
import SwiftUI
import UniformTypeIdentifiers
import UIKit

struct FACADEDemoScreen: View {
    @StateObject private var model: LivePhotoModel
    @Environment(\.scenePhase) private var scenePhase
    @Environment(\.openURL) private var openURL
    @Environment(\.dynamicTypeSize) private var dynamicTypeSize
    @State private var importedItem: PhotosPickerItem?
    @State private var importGeneration = UUID()
    @State private var importing = false
    @State private var consent = false
    @State private var showingCamera = false
    @State private var confirmSave = false
    private var garmentIntent: Bool { model.selectedMethod == .miniPhotoGuard }
    @State private var metrics: PixelMetrics?
    @State private var metricFailure = false
    @State private var showingData = false
    @State private var showingFullPhotoGuard = false
    /// The full PhotoGuard website. Set FACADE_WEBSITE_URL in your Local.xcconfig
    /// (it is empty in Config/Base.xcconfig); Info.plist passes it in as FACADEWebsiteURL.
    /// Without a valid https address the web link is hidden.
    private let fullPhotoGuardURL: URL? = {
        guard let text = Bundle.main.object(forInfoDictionaryKey: "FACADEWebsiteURL") as? String,
              let url = URL(string: text), url.scheme == "https", url.host != nil else { return nil }
        return url
    }()

    init(model: @autoclosure @escaping () -> LivePhotoModel) {
        _model = StateObject(wrappedValue: model())
    }

    private var stepNumber: Int {
        if model.stage == .saved { return 5 }
        if model.demoResult != nil { return 4 }
        if model.busy { return 3 }
        if model.stage == .photoReady { return 2 }
        return 1
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    header
                    StepIndicator(current: stepNumber)
                    statusNote
                    if let result = model.demoResult {
                        resultCard(result)
                        resultActions
                        comparisonCard(result)
                        if model.resultMethod == .miniPhotoGuard { miniResultNotice }
                    } else if model.busy {
                        progressCard
                    } else if model.stage == .photoReady, let data = model.previewData {
                        preparedCard(data)
                        modeCard
                        VStack(spacing: 10) {
                            Button { model.protectPreparedPhoto() } label: {
                                Label(model.selectedMethodAvailable ? "Protect with \(model.selectedMethod.title)" : "\(model.selectedMethod.title) models are not installed", systemImage: "shield.lefthalf.filled")
                            }.buttonStyle(PrimaryButtonStyle()).disabled(!model.selectedMethodAvailable)
                            Label("Runs on your iPhone. Your photo is never uploaded.", systemImage: "iphone")
                                .font(.footnote).foregroundStyle(Color.facadeMuted)
                            Button("Choose another photo") { model.discardCandidate() }
                                .buttonStyle(SecondaryButtonStyle())
                        }
                    } else {
                        if model.stage == .saved { savedCard } else { introduction }
                        captureCard
                        modeCard
                    }
                    privacyNote
                }
                .padding(20)
                .frame(maxWidth: 680)
                .frame(maxWidth: .infinity)
                .animation(.easeInOut(duration: 0.25), value: stepNumber)
            }
            .background(Color.facadeBackground)
            .toolbar(.hidden, for: .navigationBar)
            .sheet(isPresented: $showingCamera) {
                FACADESelfieCameraView { data in
                    cancelPendingImport()
                    consent = false
                    model.startImportedPhoto(data: data, uniformTypeIdentifier: UTType.jpeg.identifier,
                                             userPickedPhoto: true, adultSubjectsConsented: true)
                }.ignoresSafeArea()
            }
            .sheet(isPresented: $showingData) { FACADELocalDataScreen() }
            .sheet(isPresented: $showingFullPhotoGuard) {
                if let url = fullPhotoGuardURL {
                    HelpBrowser(url: url) { showingFullPhotoGuard = false }
                        .ignoresSafeArea()
                }
            }
            .confirmationDialog("Save a new protected copy?", isPresented: $confirmSave, titleVisibility: .visible) {
                Button("Save new PNG to Photos") {
                    Task { await model.saveVerifiedCopy(userConfirmedSave: true) }
                }
                Button("Cancel", role: .cancel) {}
            } message: { Text("Your original stays unchanged. Protection is experimental and is not guaranteed.") }
        }
        .task(id: importedItem) {
            guard let item = importedItem else { return }
            let generation = UUID()
            importGeneration = generation
            importing = true
            defer {
                if importGeneration == generation { importedItem = nil; consent = false; importing = false }
            }
            guard consent else { model.block("Confirm that you have permission to use this photo."); return }
            do {
                let loaded = try await item.loadTransferable(type: Data.self)
                try Task.checkCancellation()
                guard importGeneration == generation, importedItem == item, scenePhase == .active else { return }
                guard let data = loaded else {
                    model.block("This photo could not be read."); return
                }
                model.startImportedPhoto(data: data,
                    uniformTypeIdentifier: item.supportedContentTypes.first?.identifier ?? UTType.image.identifier,
                    userPickedPhoto: true, adultSubjectsConsented: true)
            } catch {
                if !Task.isCancelled, importGeneration == generation, scenePhase == .active {
                    model.block("This photo could not be read. Please try another photo.")
                }
            }
        }
        .onChange(of: scenePhase) { _, phase in
            if phase == .background { cancelPendingImport(); model.cancelForBackgrounding(); metrics = nil }
        }
        .onChange(of: model.demoResult == nil) { _, missing in
            if missing { metrics = nil; metricFailure = false }
        }
    }

    private func cancelPendingImport() {
        importGeneration = UUID()
        importedItem = nil
        importing = false
        consent = false
    }

    private var header: some View {
        HStack(alignment: .center, spacing: 12) {
            Image(systemName: "shield.lefthalf.filled")
                .font(.title3.weight(.bold)).foregroundStyle(.white)
                .frame(width: 42, height: 42)
                .background(LinearGradient.facadeBrand, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 2) {
                Text("FACADE").font(.title2.weight(.heavy)).tracking(1.5)
                Text("Your photo. Your choice.").font(.caption).foregroundStyle(Color.facadeMuted)
            }
            Spacer(minLength: 6)
            Label("On device", systemImage: "lock.fill").font(.caption2.weight(.bold)).textCase(.uppercase)
                .foregroundStyle(Color.facadeNavy)
                .padding(.horizontal, 10).padding(.vertical, 7)
                .background(Color.facadeSkyTint, in: Capsule())
        }.foregroundStyle(Color.facadeText)
        .accessibilityElement(children: .combine)
    }

    private var introduction: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Add a layer of protection before you share.")
                .font(.title.weight(.bold)).foregroundStyle(.white)
                .fixedSize(horizontal: false, vertical: true)
            Text("FACADE makes small, careful changes to your photo that are designed to confuse AI face-swap tools. You keep the original.")
                .font(.subheadline).foregroundStyle(Color.facadeSkyTint)
                .fixedSize(horizontal: false, vertical: true)
            ViewThatFits(in: .horizontal) {
                HStack(spacing: 8) { introPills }
                VStack(alignment: .leading, spacing: 8) { introPills }
            }
        }
        .padding(20).frame(maxWidth: .infinity, alignment: .leading)
        .background(LinearGradient.facadeBrand, in: RoundedRectangle(cornerRadius: 22, style: .continuous))
        .overlay(alignment: .topTrailing) {
            Image(systemName: "person.crop.square.filled.and.at.rectangle")
                .font(.system(size: 64)).foregroundStyle(.white.opacity(0.08))
                .padding(14).accessibilityHidden(true)
        }
    }

    @ViewBuilder private var introPills: some View {
        pill("No upload", systemImage: "icloud.slash")
        pill("Original kept", systemImage: "photo.stack")
        pill("Experimental", systemImage: "flask")
    }

    private func pill(_ text: String, systemImage: String) -> some View {
        Label(text, systemImage: systemImage).font(.caption.weight(.semibold)).foregroundStyle(.white)
            .padding(.horizontal, 10).padding(.vertical, 6)
            .background(.white.opacity(0.14), in: Capsule())
    }

    private var savedCard: some View {
        HStack(alignment: .top, spacing: 14) {
            IconBadge(systemImage: "checkmark.circle.fill")
            VStack(alignment: .leading, spacing: 6) {
                Text("Protected copy saved").font(.headline)
                Text("A new PNG is in your Photos library. Your original is unchanged. Share the protected copy instead of the original.")
                    .font(.subheadline).foregroundStyle(Color.facadeMuted)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }.fastCard()
    }

    private var captureCard: some View {
        VStack(alignment: .leading, spacing: 16) {
            StepTitle(number: 1, title: "Choose your photo")
            Toggle(isOn: $consent) {
                Text("I have permission to use this photo, and everyone shown is a consenting adult.")
                    .font(.footnote).fixedSize(horizontal: false, vertical: true)
            }
            .tint(Color.facadeNavy)
            .padding(12)
            .background(Color.facadeSurfaceRaised.opacity(0.6), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
            PhotosPicker(selection: $importedItem, matching: .images) {
                Label(importing ? "Opening photo…" : "Choose photo", systemImage: "photo.on.rectangle")
            }.buttonStyle(PrimaryButtonStyle()).disabled(!consent || model.busy || importing)
            if Bundle.main.object(forInfoDictionaryKey: "FACADECameraEnabled") as? Bool == true,
               UIImagePickerController.isSourceTypeAvailable(.camera) {
                Button { showingCamera = true } label: {
                    Label("Take selfie", systemImage: "camera")
                }.buttonStyle(SecondaryButtonStyle()).disabled(!consent || model.busy || importing)
            }
            if !consent {
                Text("Turn on the permission switch to choose a photo.")
                    .font(.caption).foregroundStyle(Color.facadeMuted)
            }
            Divider()
            Button { cancelPendingImport(); model.prepareSampleForReview() } label: {
                Label("No photo handy? Try a fictional sample", systemImage: "sparkles")
                    .font(.subheadline.weight(.semibold)).underline()
                    .frame(maxWidth: .infinity, minHeight: 44)
            }.buttonStyle(.plain).foregroundStyle(Color.facadeNavy)
        }.fastCard()
    }

    private var modeCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("What are you worried about?").font(.headline)
            methodOption(.facade, icon: "face.smiling",
                         title: "Face swaps", subtitle: LivePhotoEngine.resourcesAvailable
                            ? "FACADE · recommended default" : "FACADE · models not installed in this build")
            methodOption(.miniPhotoGuard, icon: "tshirt",
                         title: "AI clothing edits", subtitle: MiniPhotoGuardEngine.resourcesAvailable
                            ? "Mini PhotoGuard · research prototype" : "Mini PhotoGuard · unavailable in this build")
            Text(garmentIntent
                ? (model.selectedMethodAvailable ? "Mini PhotoGuard is a 256 × 256 full-image encoder prototype. It has no clothing mask and is not the full website method. Protection against clothing edits has not been demonstrated. Our test photo took about 6 minutes." : "PhotoGuard is suggested for this request, but its phone model is not available in this build. Choose Face swaps above to use FACADE.")
                : "FACADE targets the face-recognition step that face-swap apps rely on. Our test photo took about 1 minute on iPhone 17 Pro.")
                .font(.footnote).foregroundStyle(Color.facadeMuted)
                .fixedSize(horizontal: false, vertical: true)
        }.fastCard()
    }

    private func methodOption(_ method: ProtectionMethod, icon: String, title: String, subtitle: String) -> some View {
        let selected = model.selectedMethod == method
        return Button { model.selectMethod(method) } label: {
            HStack(spacing: 12) {
                IconBadge(systemImage: icon)
                VStack(alignment: .leading, spacing: 2) {
                    Text(title).font(.subheadline.weight(.semibold)).foregroundStyle(Color.facadeText)
                    Text(subtitle).font(.caption).foregroundStyle(Color.facadeMuted)
                }
                Spacer(minLength: 8)
                Image(systemName: selected ? "checkmark.circle.fill" : "circle")
                    .font(.title3).foregroundStyle(selected ? Color.facadeNavy : Color.facadeOutline)
                    .accessibilityHidden(true)
            }
            .padding(12)
            .background(selected ? Color.facadeSkyTint : Color.clear,
                        in: RoundedRectangle(cornerRadius: 14, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous)
                .strokeBorder(selected ? Color.facadeNavy : Color.facadeOutline, lineWidth: selected ? 1.5 : 1))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .disabled(model.busy)
        .accessibilityAddTraits(selected ? .isSelected : [])
        .accessibilityHint("Selects \(method.title)")
    }

    private func preparedCard(_ data: Data) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            StepTitle(number: 2, title: "Review your photo")
            photo(data, title: "Prepared preview · \(model.preparedSize)")
            Text(garmentIntent ? "Mini PhotoGuard exports this 256 × 256 square with gray padding. The prototype changes the full image; it does not use a clothing mask." : "This is exactly what will be protected. The longest edge is at most 1,024 pixels and the aspect ratio is preserved.")
                .font(.footnote).foregroundStyle(Color.facadeMuted)
                .fixedSize(horizontal: false, vertical: true)
        }.fastCard()
    }

    private var progressCard: some View {
        VStack(alignment: .leading, spacing: 18) {
            StepTitle(number: 3, title: "Protecting with \(model.selectedMethod.title)")
            HStack(spacing: 16) {
                Image(systemName: "shield.lefthalf.filled")
                    .font(.system(size: 34, weight: .semibold)).foregroundStyle(Color.facadeNavy)
                    .symbolEffect(.pulse, options: .repeating)
                    .frame(width: 64, height: 64)
                    .background(Color.facadeNavy.opacity(0.1), in: Circle())
                    .accessibilityHidden(true)
                VStack(alignment: .leading, spacing: 4) {
                    Text(model.progressText).font(.headline)
                        .fixedSize(horizontal: false, vertical: true)
                        .contentTransition(.opacity)
                    Text(elapsedText).font(.subheadline.monospacedDigit()).foregroundStyle(Color.facadeMuted)
                }
            }
            ProgressView().progressViewStyle(.linear).tint(Color.facadeNavy)
            Label("Keep FACADE open until it finishes. Leaving the app stops processing and clears the photo.", systemImage: "exclamationmark.circle")
                .font(.footnote).foregroundStyle(Color.facadeMuted)
                .fixedSize(horizontal: false, vertical: true)
            Text("Steps describe processing, not a protection score.")
                .font(.caption).foregroundStyle(Color.facadeMuted)
            Button("Cancel", role: .cancel) { model.cancelProcessing() }
                .buttonStyle(SecondaryButtonStyle())
        }.fastCard().accessibilityElement(children: .contain)
    }

    private var elapsedText: String {
        let seconds = Int(model.elapsedSeconds)
        return seconds < 60 ? "\(seconds) s elapsed" : String(format: "%d:%02d elapsed", seconds / 60, seconds % 60)
    }

    private func resultCard(_ result: LivePhotoResult) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(alignment: .top, spacing: 12) {
                // The half-lit shield means partial protection; never a checkmark or green here.
                IconBadge(systemImage: "shield.lefthalf.filled")
                VStack(alignment: .leading, spacing: 2) {
                    Text("Your protected copy is ready").font(.headline)
                    Text("\(model.resultMethod.title) · \(String(format: "%.1f", result.elapsedSeconds)) s · \(result.width) × \(result.height) PNG")
                        .font(.caption).foregroundStyle(Color.facadeMuted)
                }
            }
            Label("Protection is experimental. No method can guarantee that an image cannot be misused.", systemImage: "info.circle")
                .font(.footnote).foregroundStyle(Color.facadeText)
                .fixedSize(horizontal: false, vertical: true)
            if !dynamicTypeSize.isAccessibilitySize,
               let original = UIImage(data: result.originalPNG), let protected = UIImage(data: result.protectedPNG) {
                ComparisonSlider(original: original, protected: protected)
                Text(model.resultMethod == .facade ? "Drag to compare. The changes are designed to be hard to see." : "Drag to compare. This prototype visibly changes image texture.")
                    .font(.caption).foregroundStyle(Color.facadeMuted)
            } else {
                VStack(spacing: 12) {
                    photo(result.originalPNG, title: "Original")
                    photo(result.protectedPNG, title: "Protected copy")
                }
            }
            DisclosureGroup("See what changed") {
                VStack(alignment: .leading, spacing: 12) {
                    if let metrics {
                        photo(metrics.deltaPNG, title: "Pixel difference · amplified ×16")
                        Text("The difference is amplified for visibility. It is not an image of protection strength.")
                            .font(.caption).foregroundStyle(Color.facadeMuted)
                        detailRow("Largest channel change", "\(metrics.maximumChange) / 255")
                        detailRow("RMS pixel change", String(format: "%.2f / 255", metrics.rmsChange))
                        detailRow("PSNR", metrics.psnr.isFinite ? String(format: "%.2f dB", metrics.psnr) : "Identical pixels")
                        Text("These measure pixel differences, not protection effectiveness.")
                            .font(.caption).foregroundStyle(Color.facadeMuted)
                    } else if metricFailure {
                        StatusBanner(kind: .info, message: "Pixel details are unavailable. The original and protected previews remain available.")
                    } else { ProgressView("Calculating pixel details") }
                    Divider()
                    detailRow("Preparation", String(format: "%.2f s", result.preparationSeconds))
                    detailRow("Model loading", String(format: "%.2f s", result.loadSeconds))
                    detailRow("Protection engine", String(format: "%.2f s", result.pipelineSeconds))
                }.font(.footnote).padding(.top, 12)
            }.font(.subheadline.weight(.semibold)).tint(Color.facadeNavy)
        }.fastCard()
        .task(id: result.protectedPNG) {
            metrics = nil; metricFailure = false
            let clean = result.originalPNG, protected = result.protectedPNG
            let computed = await Task.detached(priority: .utility) { try? PixelMetrics.measure(clean, protected) }.value
            guard !Task.isCancelled, model.demoResult != nil else { return }
            metrics = computed; metricFailure = computed == nil
        }
    }

    @ViewBuilder private var resultActions: some View {
        if model.faceOffState != .rendering {
            VStack(spacing: 10) {
                if model.verifiedCopy != nil {
                    Button { confirmSave = true } label: {
                        Label("Save protected copy", systemImage: "square.and.arrow.down")
                    }.buttonStyle(PrimaryButtonStyle()).disabled(model.busy)
                } else if model.stage == .bundledFictionalAdultReady {
                    Label("Sample results cannot be saved. Choose your own photo to save a protected copy.", systemImage: "sparkles")
                        .font(.footnote).foregroundStyle(Color.facadeMuted)
                }
                Button(model.verifiedCopy == nil ? "Try another photo" : "Discard without saving") {
                    metrics = nil; model.discardCandidate()
                }.buttonStyle(SecondaryButtonStyle()).disabled(model.busy)
            }
        }
    }

    private var miniResultNotice: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("Mini PhotoGuard is very experimental", systemImage: "exclamationmark.triangle.fill")
                .font(.headline).foregroundStyle(Color.facadeWarning)
            Text("It can visibly change image texture. Protection against AI clothing edits has not been demonstrated. The comparison above tests face swaps only.")
                .font(.footnote).foregroundStyle(Color.facadeMuted)
            if let url = fullPhotoGuardURL {
                Text("Unhappy with the result? You can try the separate full PhotoGuard method on the web. It is also experimental and requires an available private demo session.")
                    .font(.footnote).foregroundStyle(Color.facadeMuted)
                Button {
                    openURL(url) { accepted in
                        guard !accepted else { return }
                        UIApplication.shared.open(url, options: [:]) { opened in
                            if !opened {
                                Task { @MainActor in showingFullPhotoGuard = true }
                            }
                        }
                    }
                } label: {
                    Label("Try full PhotoGuard on the web", systemImage: "arrow.up.right.square")
                }.buttonStyle(SecondaryButtonStyle()).disabled(model.busy)
                    .accessibilityHint("Opens the PhotoGuard website. No photo is sent. Internet access is required.")
                Text("Opening the website sends no photo. Uploading is a separate choice there.")
                    .font(.caption).foregroundStyle(Color.facadeMuted)
            }
        }.fixedSize(horizontal: false, vertical: true).fastCard()
    }

    private func comparisonCard(_ result: LivePhotoResult) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(spacing: 12) {
                IconBadge(systemImage: "theatermasks.fill")
                Text("See it against a face swap").font(.headline)
            }
            Text(model.resultMethod == .miniPhotoGuard
                 ? "Run the same on-device InSwapper attempt on the prepared 256 × 256 original and its Mini PhotoGuard copy. This tests face swaps, not protection against clothing edits. One example is not a general protection test."
                 : "Run the same on-device InSwapper face swap on both copies and compare. One example is not a general protection test; results vary by photo.")
                .font(.footnote).foregroundStyle(Color.facadeMuted)
                .fixedSize(horizontal: false, vertical: true)
            switch model.faceOffState {
            case .idle, .failed:
                Button { model.renderFaceOff(for: result) } label: {
                    Label("Run face-swap test", systemImage: "rectangle.on.rectangle")
                }.buttonStyle(SecondaryButtonStyle()).disabled(model.busy)
                if case .failed(let message) = model.faceOffState {
                    StatusBanner(kind: StatusBanner.kind(for: message), message: message)
                }
            case .rendering:
                ProgressView("Rendering both face swaps").tint(Color.facadeNavy)
                Button("Cancel test") { model.cancelProcessing() }.buttonStyle(SecondaryButtonStyle())
            case .ready:
                let columns = Array(repeating: GridItem(.flexible(), spacing: 12), count: dynamicTypeSize.isAccessibilitySize ? 1 : 2)
                if let clean = model.faceOffUnprotectedFace, let protected = model.faceOffProtectedFace {
                    LazyVGrid(columns: columns, alignment: .leading, spacing: 12) {
                        swapFace(clean, title: "Swap on original", match: model.faceOffUnprotectedMatch)
                        swapFace(protected, title: "Swap on protected", match: model.faceOffProtectedMatch)
                    }
                    Text("Match to you: similarity of each swapped face to the original person, measured by the swap tool's own face recognizer. Higher values mean more similar to this recognizer, not proof of identity or protection.")
                        .font(.caption).foregroundStyle(Color.facadeMuted)
                        .fixedSize(horizontal: false, vertical: true)
                }
                if model.resultMethod == .facade {
                    Text("FACADE also targets this recognizer. These scores are not independent evidence of protection.")
                        .font(.caption).foregroundStyle(Color.facadeMuted)
                        .fixedSize(horizontal: false, vertical: true)
                }
                if let clean = model.faceOffUnprotectedSwap, let protected = model.faceOffProtectedSwap {
                    DisclosureGroup("Show full swapped photos") {
                        LazyVGrid(columns: columns, spacing: 12) {
                            photo(clean, title: "Swap on original")
                            photo(protected, title: "Swap on protected")
                        }.padding(.top, 8)
                    }.font(.footnote.weight(.semibold)).tint(Color.facadeNavy)
                }
                if let seconds = model.faceOffRuntimeSeconds {
                    Text(String(format: "Both swaps ran on this iPhone in %.1f seconds.", seconds)).font(.caption).foregroundStyle(Color.facadeMuted)
                }
            }
        }.fastCard()
    }

    /// Both scores use the same ink: colour must not suggest which result is "good".
    private func swapFace(_ data: Data, title: String, match: Double?) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            if let image = UIImage(data: data) {
                Image(uiImage: image).resizable().interpolation(.high).scaledToFit()
                    .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
                    .privacySensitive()
                    .accessibilityLabel(title)
            }
            Text(title).font(.caption.weight(.semibold)).foregroundStyle(Color.facadeMuted)
            if let match {
                Text(String(format: "%.2f", match)).font(.title2.weight(.bold).monospacedDigit()).foregroundStyle(Color.facadeText)
                    .accessibilityLabel("Match to you \(String(format: "%.2f", match))")
                Text("match to you").font(.caption2).foregroundStyle(Color.facadeMuted)
            }
        }.frame(maxWidth: .infinity, alignment: .leading)
    }

    @ViewBuilder private var statusNote: some View {
        switch model.stage {
        case .blocked(let message):
            VStack(alignment: .leading, spacing: 6) {
                StatusBanner(kind: StatusBanner.kind(for: message), message: message)
                if message.hasPrefix("The new copy was not saved"),
                   let settings = URL(string: UIApplication.openSettingsURLString) {
                    Button("Open Settings to allow Photos access") { UIApplication.shared.open(settings) }
                        .font(.footnote.weight(.semibold)).frame(minHeight: 44)
                }
            }
        default: EmptyView()
        }
    }

    private var privacyNote: some View {
        VStack(alignment: .leading, spacing: 10) {
            Label("Private by design", systemImage: "lock.shield").font(.subheadline.weight(.semibold))
                .foregroundStyle(Color.facadeText)
            Text("Photos stay on this device. Leaving the app clears the preview. Only Save adds a new copy to Photos.")
                .font(.footnote).foregroundStyle(Color.facadeMuted)
                .fixedSize(horizontal: false, vertical: true)
            DisclosureGroup("About this engine") {
                VStack(alignment: .leading, spacing: 8) {
                    Text("FACADE · 10 coarse + 6 fine steps · K2 cadence. The measured engine is unchanged. Mini PhotoGuard is a separate 256-pixel encoder prototype with 200 iterations. Neither method guarantees protection.")
                        .font(.footnote).foregroundStyle(Color.facadeMuted)
                    Divider()
                    benchmarkSummary
                    Button("Local data controls") { showingData = true }.frame(minHeight: 44)
                }.padding(.top, 8)
            }.font(.footnote).tint(Color.facadeNavy)
        }.padding(.vertical, 6)
    }

    // Fixed numbers from the team's offline benchmark: FACADE outputs made on a
    // phone, scored with an independent FaceNet recognizer (not computed here).
    // They are separate from the live swapper's own-recognizer scores.
    private var benchmarkSummary: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("FACADE face-swap benchmarks").font(.subheadline.weight(.semibold))
                .foregroundStyle(Color.facadeText)
            Text("Pairs meeting the test criterion, measured with an independent FaceNet recognizer.")
                .foregroundStyle(Color.facadeMuted)
            benchmarkRow("Original PNG", simSwap: 176, inSwapper: 113)
            Divider()
            benchmarkRow("JPEG quality 75", simSwap: 170, inSwapper: 71)
            Divider()
            benchmarkRow("Half-size + JPEG quality 75", simSwap: 171, inSwapper: 69)
            Text("48 fictional development images × 4 fixed recipients = 192 pairs per condition. All clean swaps qualified; no pairs were missing.")
                .foregroundStyle(Color.facadeMuted)
            Text("Criterion: similarity to the source is at least 0.4 for the clean swap, below 0.4 for the protected swap, and drops by at least 0.1.")
                .foregroundStyle(Color.facadeMuted)
            Text("Exploratory results from a reused development cohort, not a held-out test or a guarantee. These are FACADE results, not Mini PhotoGuard results, and are separate from the single-photo comparison above.")
                .foregroundStyle(Color.facadeMuted)
        }
        .font(.footnote)
        .fixedSize(horizontal: false, vertical: true)
    }

    private func benchmarkRow(_ condition: String, simSwap: Int, inSwapper: Int) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(condition).font(.footnote.weight(.semibold)).foregroundStyle(Color.facadeText)
            let columns = Array(repeating: GridItem(.flexible(), alignment: .leading),
                                count: dynamicTypeSize.isAccessibilitySize ? 1 : 2)
            LazyVGrid(columns: columns, alignment: .leading, spacing: 8) {
                benchmarkValue("SimSwap", passes: simSwap)
                benchmarkValue("InSwapper", passes: inSwapper)
            }
        }
    }

    private func benchmarkValue(_ engine: String, passes: Int) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(engine).font(.caption.weight(.semibold)).foregroundStyle(Color.facadeText)
            Text(String(format: "%.1f%%", Double(passes) / 192 * 100))
                .font(.headline).monospacedDigit().foregroundStyle(Color.facadeText)
            Text("\(passes)/192 pairs").font(.caption).monospacedDigit()
                .foregroundStyle(Color.facadeMuted)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .accessibilityElement(children: .combine)
    }

    /// A label in ink with its value in muted grey.
    private func detailRow(_ label: String, _ value: String) -> some View {
        LabeledContent {
            Text(value).foregroundStyle(Color.facadeMuted)
        } label: {
            Text(label).foregroundStyle(Color.facadeText)
        }
    }

    private func photo(_ data: Data, title: String) -> some View {
        VStack(alignment: .leading, spacing: 7) {
            if let image = UIImage(data: data) {
                Image(uiImage: image).resizable().scaledToFit()
                    .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
                    .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous).strokeBorder(Color.facadeHairline, lineWidth: 1))
                    .privacySensitive()
                    .accessibilityLabel(title)
            }
            Text(title).font(.caption.weight(.medium)).foregroundStyle(Color.facadeMuted)
        }.frame(maxWidth: .infinity, alignment: .leading)
    }
}
