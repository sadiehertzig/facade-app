// NativeImageIO.swift
//
// Turns an RGBImage (the engine's plain byte array of pixels, defined in
// NativeTensor.swift) into PNG file bytes.
//
// Where it sits: LivePhotoEngine calls `png()` to produce the "original" and
// "protected" previews that the Protect screen displays and that Save writes
// to Photos. The same function is used by the photo-preparation host test.
//
// Key ideas:
// 1. PNG is lossless, so the exact pixels the engine produced are what gets
//    saved. A JPEG here would undo part of the protection.
// 2. The image is 24-bit RGB with no alpha channel and an sRGB colour space.
// 3. Apple's ImageIO does the actual encoding; this file only wraps it.
import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
/// PNG export for the engine's image type.
extension RGBImage {
 /// Encodes the image as PNG bytes. Throws if any ImageIO step refuses.
 func png() throws ->Data {
        // Timing hook for the optional profiler (a no-op in the shipping app).
        let profileToken=Profile.begin("image.png");defer{Profile.end("image.png",profileToken)}

  // Wrap the pixel bytes so Core Graphics can read them without copying.
  let data=Data(bytes),provider=CGDataProvider(data:data as CFData)!
  // 3 bytes per pixel, 8 bits per channel, rows packed with no padding,
  // no alpha ("none"), sRGB. Any mismatch with the byte layout would yield nil.
  guard let cg=CGImage(width:width,height:height,bitsPerComponent:8,bitsPerPixel:24,bytesPerRow:width*3,space:CGColorSpace(name:CGColorSpace.sRGB)!,bitmapInfo:CGBitmapInfo(rawValue:CGImageAlphaInfo.none.rawValue),provider:provider,decode:nil,shouldInterpolate:false,intent:.defaultIntent) else {throw NativeError.invalid("Could not encode image")}
  // An in-memory PNG writer. `1` is the number of images the file will hold.
  let out=NSMutableData();guard let dest=CGImageDestinationCreateWithData(out,UTType.png.identifier as CFString,1,nil) else {throw NativeError.invalid("PNG encoder unavailable")}
  // Add the one image and write the file; `out` now holds the PNG bytes.
  CGImageDestinationAddImage(dest,cg,nil);guard CGImageDestinationFinalize(dest) else {throw NativeError.invalid("PNG export failed")};return out as Data
 }
}
