#!/bin/sh
# Xcode build phase "Copy private models".
#
# Copies the private model folders listed in app/ios/Models/README.md into the
# app bundle. They are not in git. The app builds and runs without them; the
# features that need them report that the models are not installed.
set -eu
# Model folders may be symlinks (for example to a folder outside the repo);
# rsync -L copies the real files, because iOS apps cannot contain such links.

src="$SRCROOT/Models"
dest="$TARGET_BUILD_DIR/$UNLOCALIZED_RESOURCES_FOLDER_PATH"

for name in NativeModels FACADEModelHandoff FaceOffModels FaceOffAssets FaceOffTarget \
            SampleFace MiniPhotoGuardPayload photoguard_encoder_gradient_256_fp32.mlpackage; do
  if [ -e "$src/$name" ]; then
    echo "Copying Models/$name"
    rsync -aL --delete "$src/$name" "$dest/"
  else
    # Remove a copy left over from an earlier build.
    rm -rf "$dest/$name"
  fi
done
