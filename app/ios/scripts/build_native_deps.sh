#!/usr/bin/env bash
# Builds the four C/C++ libraries the FACADE iOS app links against, from pinned
# upstream source releases, for both the iPhone and the (Apple Silicon)
# simulator. Run it once after cloning, and again only if this file changes.
#
#   app/ios/scripts/build_native_deps.sh              # both device and simulator
#   app/ios/scripts/build_native_deps.sh iphonesimulator
#
# Output (ignored by git):
#   app/ios/Vendor/iphoneos/{include,lib}
#   app/ios/Vendor/iphonesimulator/{include,lib}
#
# Needs: Xcode, curl, and CMake 3.25 or newer (`brew install cmake`; see
# app/ios/README.md for pip). Set CMAKE=/path/to/cmake to use a specific one.
# Set FACADE_DOWNLOADS=/some/dir to reuse already-downloaded tarballs; every
# tarball is checked against the SHA-256 below before it is used. Set
# FACADE_KEEP_BUILD=1 to keep the intermediate build folders for debugging.
set -euo pipefail

IOS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR="$IOS_DIR/Vendor"
DOWNLOADS="${FACADE_DOWNLOADS:-$VENDOR/downloads}"
WORK="$VENDOR/.build"
CMAKE="${CMAKE:-cmake}"
DEPLOYMENT_TARGET=18.0
JOBS="$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"

# name | file | url | sha256
SOURCES=(
  "opencv|opencv-5.0.0.tar.gz|https://github.com/opencv/opencv/archive/refs/tags/5.0.0.tar.gz|b0528f5a1d379d59d4701cb28c36e22214cc51cf64594e5b56f2d3e6c0233095"
  "kleidicv|kleidicv-26.03.tar.gz|https://gitlab.arm.com/kleidi/kleidicv/-/archive/26.03/kleidicv-26.03.tar.gz|1bb4078fd215565f3906c33f76a00e7b843328965230d765afcd0d51141c434f"
  "sleef|sleef-5a1d179df9cf652951b59010a2d2075372d67f68.tar.gz|https://github.com/shibatch/sleef/archive/5a1d179df9cf652951b59010a2d2075372d67f68.tar.gz|afd1b92010ae7918e20eec3e5bb270b7ca4828b5cb19e4d820a0a4558a04ce63"
  "libjpeg-turbo|libjpeg-turbo-3.1.4.1.tar.gz|https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/3.1.4.1/libjpeg-turbo-3.1.4.1.tar.gz|ecae8008e2cc9ade2f2c1bb9d5e6d4fb73e7c433866a056bd82980741571a022"
)

PLATFORMS=("$@")
if [ ${#PLATFORMS[@]} -eq 0 ]; then PLATFORMS=(iphoneos iphonesimulator); fi
for p in "${PLATFORMS[@]}"; do
  case "$p" in iphoneos|iphonesimulator) ;; *) echo "Unknown platform: $p (use iphoneos or iphonesimulator)" >&2; exit 2 ;; esac
done

command -v "$CMAKE" >/dev/null || { echo "CMake not found. Install it (brew install cmake) or set CMAKE=/path/to/cmake." >&2; exit 1; }
command -v xcrun >/dev/null || { echo "Xcode command line tools not found." >&2; exit 1; }
if [ "$(uname -m)" != "arm64" ]; then
  echo "Note: the simulator libraries are built for arm64 only (Apple Silicon Macs)." >&2
fi

log() { printf '\n==> %s\n' "$*"; }

fetch() {  # fetch <file> <url> <sha256>; prints the verified path
  local file="$1" url="$2" sha="$3" path="$DOWNLOADS/$1"
  mkdir -p "$DOWNLOADS"
  if [ ! -f "$path" ]; then
    log "Downloading $file" >&2
    curl -fL --retry 3 -o "$path.part" "$url" >&2
    mv "$path.part" "$path"
  fi
  local got
  got="$(shasum -a 256 "$path" | awk '{print $1}')"
  if [ "$got" != "$sha" ]; then
    echo "SHA-256 mismatch for $file: expected $sha, got $got. Delete it and rerun." >&2
    exit 1
  fi
  echo "$path"
}

unpack() {  # unpack <tarball> <dest-dir>; prints the single top-level directory
  local tarball="$1" dest="$2"
  rm -rf "$dest"; mkdir -p "$dest"
  tar -xzf "$tarball" -C "$dest"
  local top
  top="$(find "$dest" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
  echo "$top"
}

cross_flags() {  # cross_flags <platform> <processor name the project expects>
  echo "-DCMAKE_SYSTEM_NAME=iOS -DCMAKE_SYSTEM_PROCESSOR=$2 -DCMAKE_OSX_SYSROOT=$1 \
-DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET=$DEPLOYMENT_TARGET \
-DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF"
}

# 1. Download and unpack the pinned sources.
SRC="$WORK/src"
for row in "${SOURCES[@]}"; do
  IFS='|' read -r name file url sha <<<"$row"
  tarball="$(fetch "$file" "$url" "$sha")"
  eval "TAR_${name//-/_}=\"\$tarball\""
  if [ "$name" != kleidicv ]; then  # OpenCV unpacks KleidiCV itself (step 3)
    top="$(unpack "$tarball" "$SRC/$name")"
    eval "SRC_${name//-/_}=\"\$top\""
  fi
done

# 2. SLEEF generates some of its sources with small host tools, so a macOS build
#    of those tools comes first.
SLEEF_FLAGS="-DSLEEF_BUILD_TESTS=OFF -DSLEEF_BUILD_DFT=OFF -DSLEEF_BUILD_QUAD=OFF \
-DSLEEF_BUILD_GNUABI_LIBS=OFF -DSLEEF_BUILD_SCALAR_LIB=OFF -DSLEEF_BUILD_INLINE_HEADERS=OFF \
-DSLEEF_BUILD_BENCH=OFF -DSLEEF_SHOW_CONFIG=OFF"
log "SLEEF host tools"
rm -rf "$WORK/sleef-host"
"$CMAKE" -S "$SRC_sleef" -B "$WORK/sleef-host" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF $SLEEF_FLAGS >/dev/null
"$CMAKE" --build "$WORK/sleef-host" -j "$JOBS" >/dev/null

for platform in "${PLATFORMS[@]}"; do
  OUT="$VENDOR/$platform"
  rm -rf "$OUT"; mkdir -p "$OUT/include" "$OUT/lib"
  B="$WORK/$platform"
  rm -rf "$B"

  # 3. OpenCV 5.0.0: only the core and imgproc modules, with Arm's KleidiCV
  #    acceleration. The optional Unicode text font is left out (no download).
  #    OpenCV fetches KleidiCV while configuring; it is handed the verified
  #    tarball through its download cache. (Passing KLEIDICV_SOURCE_PATH instead
  #    silently turns KleidiCV off in OpenCV 5.0.0.)
  log "OpenCV ($platform)"
  mkdir -p "$B/opencv-downloads/kleidicv"
  cp "$TAR_kleidicv" "$B/opencv-downloads/kleidicv/b85a745bfe0e87e67e30be9533eb6b24-kleidicv-26.03.tar.gz"
  "$CMAKE" -S "$SRC_opencv" -B "$B/opencv" $(cross_flags "$platform" arm64) \
    -DBUILD_LIST=core,imgproc -DBUILD_opencv_python3=OFF -DWITH_ITT=OFF \
    -DWITH_IPP=OFF -DWITH_LAPACK=OFF -DWITH_OPENCL=OFF -DWITH_TBB=OFF -DWITH_KLEIDICV=ON \
    -DWITH_UNIFONT=OFF -DCPU_DISPATCH="NEON_FP16;NEON_BF16;NEON_DOTPROD" \
    -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_opencv_apps=OFF \
    -DOPENCV_DOWNLOAD_PATH="$B/opencv-downloads" \
    -DCMAKE_INSTALL_PREFIX="$B/opencv-install" >/dev/null
  "$CMAKE" --build "$B/opencv" -j "$JOBS" >/dev/null
  "$CMAKE" --install "$B/opencv" >/dev/null
  opencv_include="$(dirname "$(find "$B/opencv-install/include" -type d -name opencv2 | head -n 1)")"
  cp -R "$opencv_include/opencv2" "$OUT/include/"
  for lib in opencv_core opencv_imgproc kleidicv kleidicv_hal kleidicv_thread zlib; do
    found="$(find "$B/opencv-install" "$B/opencv" -name "lib$lib.a" | head -n 1)"
    [ -n "$found" ] || { echo "OpenCV build did not produce lib$lib.a" >&2; exit 1; }
    cp "$found" "$OUT/lib/"
  done

  # 4. SLEEF vector math (pinned to the commit PyTorch uses).
  log "SLEEF ($platform)"
  "$CMAKE" -S "$SRC_sleef" -B "$B/sleef" $(cross_flags "$platform" aarch64) \
    -DNATIVE_BUILD_DIR="$WORK/sleef-host" $SLEEF_FLAGS >/dev/null
  "$CMAKE" --build "$B/sleef" -j "$JOBS" >/dev/null
  cp "$B/sleef/include/"*.h "$OUT/include/"
  cp "$B/sleef/lib/libsleef.a" "$OUT/lib/"

  # 5. libjpeg-turbo, plain C (no SIMD), static only.
  log "libjpeg-turbo ($platform)"
  "$CMAKE" -S "$SRC_libjpeg_turbo" -B "$B/jpeg" $(cross_flags "$platform" arm64) \
    -DENABLE_SHARED=OFF -DENABLE_STATIC=ON -DWITH_SIMD=OFF -DWITH_TURBOJPEG=OFF \
    -DWITH_TOOLS=OFF -DWITH_TESTS=OFF >/dev/null
  "$CMAKE" --build "$B/jpeg" --target jpeg-static -j "$JOBS" >/dev/null
  cp "$SRC_libjpeg_turbo/src/jpeglib.h" "$SRC_libjpeg_turbo/src/jmorecfg.h" \
     "$SRC_libjpeg_turbo/src/jerror.h" "$B/jpeg/jconfig.h" "$OUT/include/"
  cp "$B/jpeg/libjpeg.a" "$OUT/lib/"

  log "Installed $platform libraries in ${OUT#$IOS_DIR/}"
  ls -1 "$OUT/lib"
done

# The intermediate build trees are large; Xcode only needs Vendor/<platform>.
if [ "${FACADE_KEEP_BUILD:-0}" != "1" ]; then rm -rf "$WORK"; fi
echo
echo "Done."
