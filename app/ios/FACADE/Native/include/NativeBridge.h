// NativeBridge.h
//
// The single header Swift imports to reach the C and C++ engine. The module
// map next to this file (module.modulemap) turns it into a Swift module named
// CNative, so Swift files write `import CNative` and can then call every
// function declared in the four headers below as if it were a Swift function.
//
// The project does not turn on Swift's C++ interoperability, so Swift reads
// these headers as plain C. That is why each header wraps its functions in
// extern "C" (or is plain C already) and passes only pointers, numbers and an
// opaque `void *` handle. The C++ classes in NativeNetwork.cpp stay hidden
// behind that handle, and C++-only parts (ProfileScope in NativeProfile.h) sit
// inside #ifdef __cplusplus where Swift does not see them.
//
// The paths are relative ("../") because the headers live one folder up, in
// FACADE/Native/, next to their source files.
// 8-bit image operations: JPEG round trip and Pillow-exact resize.
#include "../NativePixels.h"
// Resizing, face-alignment math, float crop and DCT filter (with gradients).
#include "../NativeGeometry.h"
// The face-recognition network: load, forward, forward + backward.
#include "../NativeNetwork.h"
// Tiny profiler and the engine's nanosecond clock.
#include "../NativeProfile.h"
