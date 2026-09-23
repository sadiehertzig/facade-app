# FACADE for iPhone

The FACADE iOS app. It has two tabs:

- **Protect**: choose a photo (or take a selfie), review it, and make a
  protected copy on the phone. Nothing is uploaded. The protected copy is saved
  to Photos only when you tap Save. Two methods are offered: **FACADE** (the
  default, aimed at face-swap tools) and **Mini PhotoGuard** (a small,
  very experimental prototype aimed at AI clothing edits). An optional
  face-swap comparison shows what a swap tool does to the original and to the
  protected copy.
- **Get Help**: links to official help services (Take It Down, StopNCII,
  CyberTipline, FTC, and finding your representative).

Protection is experimental and never guaranteed. Please keep the app's wording
honest about that when you change it.

## What you need

- A Mac with Xcode 26 or newer (the project was checked with Xcode 27.0).
  The simulator build needs an Apple Silicon Mac.
- CMake, to build the C/C++ libraries once: `brew install cmake`.
  Without Homebrew, `python3 -m pip install --user cmake` works too, but its
  `cmake` is usually not on your `PATH`; pass it to the build script instead:
  `CMAKE="$(python3 -m site --user-base)/bin/cmake" app/ios/scripts/build_native_deps.sh`.
- To run on a real iPhone (iOS 18 or newer): a free or paid Apple developer
  account signed in to Xcode.

## First-time setup

Run these from the repository root.

1. **Build the C/C++ libraries** (OpenCV, KleidiCV, SLEEF, libjpeg-turbo).
   The script downloads pinned source releases, checks their SHA-256, and
   builds them for the iPhone and the simulator. It takes a few minutes, needs
   about 500 MB of temporary space, and leaves about 120 MB (libraries plus the
   downloaded sources) in `app/ios/Vendor/`, which git ignores.

   ```bash
   app/ios/scripts/build_native_deps.sh
   ```

2. **Signing, only for a real iPhone.** Copy the example file and fill in your
   own team ID and a bundle identifier that is unique to you:

   ```bash
   cp app/ios/FACADE/Config/Local.xcconfig.example app/ios/FACADE/Config/Local.xcconfig
   ```

   `Local.xcconfig` is ignored by git. Do not choose a team in Xcode's
   *Signing & Capabilities* tab: Xcode would write your team ID into
   `project.pbxproj`, which everyone shares.

3. **Models, optional.** The app builds and runs without them, but the
   protection methods stay switched off until they are installed. See
   [`Models/README.md`](Models/README.md).

4. Open `app/ios/FACADE.xcodeproj`, choose the **FACADE** scheme and a
   simulator or your iPhone, and press Run.

Command-line build for the simulator (no signing needed):

```bash
cd app/ios
xcodebuild -project FACADE.xcodeproj -scheme FACADE -sdk iphonesimulator \
  -configuration Debug CODE_SIGNING_ALLOWED=NO build
```

## Tests

A few small tests run directly on your Mac, without a simulator or models:

```bash
app/ios/scripts/run_host_tests.sh
```

They cover the pixel-difference display, Mini PhotoGuard's image preparation
and pixel limits, and FACADE's photo preparation (all eight photo
orientations).

## How the project is organized

```
app/ios/
  FACADE.xcodeproj       Xcode project (relative paths only)
  FACADE/
    App/                 Swift source: screens, view models and the engines
    Native/              C and C++ code, reached from Swift as the CNative module
    Resources/           asset catalog (the app icon slot is empty for now)
    Config/              Base.xcconfig, Local.xcconfig.example, Info.plist,
                         PrivacyInfo.xcprivacy, FACADE.entitlements
  FACADEKit/             local Swift package: model-manifest checks and the
                         Apple Vision / Core ML adapters used by the engine
  Tests/                 standalone tests (see above)
  Models/                private models go here (not in git)
  scripts/               build_native_deps.sh, copy_models.sh, run_host_tests.sh
  Vendor/                built C/C++ libraries (generated, not in git)
```

Useful starting points:

- `App/PhoneApp.swift`: the app entry point and its two tabs.
- `App/PhotoAcquisitionViews.swift`: the Protect screen.
- `App/HelpScreen.swift` and `App/HelpResources.swift`: the Get Help screen.
- `App/LivePhotoModel.swift`: the state behind the Protect screen.
- `App/LivePhotoEngine.swift` and `App/MiniPhotoGuardEngine.swift`: the two
  protection methods.

`App/`, `Native/` and `Resources/` are folder-synchronized in Xcode: a file you
add to one of those folders is part of the app automatically, and the project
file does not change.

The `App/Native*.swift` files, `App/CleanupKernel.swift`,
`App/RegionalEnergyKernel.swift` and the `Native/` folder are the numerical
protection engine. It is compiled with optimization and strict floating point
in both Debug and Release so that every build produces the same pixels. Changing it changes the protection, so talk to
the team first and re-measure afterwards.

## Settings you may want to change

All in `FACADE/Config/Base.xcconfig` (shared) or your `Local.xcconfig`
(personal):

- `PRODUCT_BUNDLE_IDENTIFIER`: defaults to `com.example.facade`. Set your own in
  `Local.xcconfig`. To upgrade a copy of FACADE that is already installed on
  your phone, use the identifier that copy was installed with.
- `DEVELOPMENT_TEAM`: your Apple team ID, only in `Local.xcconfig`.
- `FACADE_WEBSITE_URL`: the page opened by "Try full PhotoGuard on the web".
- `MARKETING_VERSION` and `CURRENT_PROJECT_VERSION`: the version shown to users
  and the build number.

## Please do not commit

`Local.xcconfig`, anything in `Models/` except its README, `Vendor/`,
`xcuserdata/`, build folders, provisioning profiles or certificates. The
repository's `.gitignore` already covers these.
