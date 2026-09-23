import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
extension RGBImage {
 func png() throws ->Data {
        let profileToken=Profile.begin("image.png");defer{Profile.end("image.png",profileToken)}

  let data=Data(bytes),provider=CGDataProvider(data:data as CFData)!
  guard let cg=CGImage(width:width,height:height,bitsPerComponent:8,bitsPerPixel:24,bytesPerRow:width*3,space:CGColorSpace(name:CGColorSpace.sRGB)!,bitmapInfo:CGBitmapInfo(rawValue:CGImageAlphaInfo.none.rawValue),provider:provider,decode:nil,shouldInterpolate:false,intent:.defaultIntent) else {throw NativeError.invalid("Could not encode image")}
  let out=NSMutableData();guard let dest=CGImageDestinationCreateWithData(out,UTType.png.identifier as CFString,1,nil) else {throw NativeError.invalid("PNG encoder unavailable")}
  CGImageDestinationAddImage(dest,cg,nil);guard CGImageDestinationFinalize(dest) else {throw NativeError.invalid("PNG export failed")};return out as Data
 }
}
