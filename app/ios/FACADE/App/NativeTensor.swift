// NativeTensor.swift
//
// The basic data types of the FACADE engine and thin Swift wrappers around
// the C/C++ code in FACADE/Native/.
//
// Where it sits: every other Native*.swift file uses these types. RGBImage is
// the 8-bit photo, TensorMath holds the float operations the optimiser needs
// (and their gradients), NativeModel runs a Core ML model, and
// NativeProxyModel runs one of the two face-recognition networks that stand in
// for a face swapper.
//
// Key ideas:
// 1. Two pixel layouts. RGBImage.bytes is packed (R,G,B,R,G,B,...), the way
//    images are stored. Float tensors are "channel first" (all red values, then
//    all green, then all blue), the way neural networks read them, with values
//    from 0 to 1. `tensor()` and `stored(_:)` convert between the two.
// 2. Most operations come in pairs: a forward function (crop, resize) and a
//    "backward" function that carries a gradient from the output back to the
//    input. That is how the engine learns which pixels to nudge.
// 3. Every boundary checks shapes and refuses NaN or infinite values, so a
//    broken model or a bad input stops the run instead of quietly producing a
//    damaged photo.
// 4. The `Profile.begin`/`Profile.end` pairs are timing hooks for an optional
//    profiler; they do not change any result.
import CNative
import Foundation
import CoreML

/// The engine's one error type. The message is shown to the user, so it is
/// written in plain words.
enum NativeError: Error, LocalizedError {
    case invalid(String)
    var errorDescription: String? { if case .invalid(let s) = self { return s }; return nil }
}
/// An 8-bit RGB image, three bytes per pixel, row by row, top row first.
/// The initialiser refuses empty images, images over 2048 px on a side, and a
/// byte count that does not match the size.
struct RGBImage {
    let width: Int, height: Int
    var bytes: [UInt8]
    init(width: Int, height: Int, bytes: [UInt8]) throws {
        guard width > 0, height > 0, width <= 2048, height <= 2048, bytes.count == width*height*3 else { throw NativeError.invalid("Invalid RGB image") }
        self.width=width; self.height=height; self.bytes=bytes
    }
    /// Number of pixels (the size of one channel plane in a float tensor).
    var plane: Int { width*height }
    /// Converts to a channel-first float tensor with values 0...1.
    func tensor() -> [Float] {
        let profileToken=Profile.begin("image.tensor");defer{Profile.end("image.tensor",profileToken)}

        var out=Profile.measure("allocation.swift.tensor"){[Float](repeating: 0,count: bytes.count)}
        // Packed index 3*p+c becomes planar index c*plane+p.
        for p in 0..<plane { for c in 0..<3 { out[c*plane+p]=Float(bytes[3*p+c])/255 } }
        return out
    }
    /// Converts a float tensor back to bytes, the way a saved image would store
    /// it: scale to 0...255, round half up, and clip. The result keeps this
    /// image's size. Refuses NaN or infinity, which would mean the optimiser broke.
    func stored(_ tensor: [Float]) throws -> RGBImage {
        let profileToken=Profile.begin("image.stored");defer{Profile.end("image.stored",profileToken)}

        guard tensor.count==bytes.count, tensor.allSatisfy(\.isFinite) else { throw NativeError.invalid("Non-finite optimizer state") }
        var out=bytes
        for p in 0..<plane { for c in 0..<3 { out[3*p+c]=UInt8(min(255,max(0,floorf(tensor[c*plane+p]*255+0.5)))) } }
        return try RGBImage(width:width,height:height,bytes:out)
    }
    /// Simulates saving as JPEG at `quality` (1-100) and opening it again, using
    /// libjpeg-turbo in memory. Nothing is written to disk.
    func jpeg(_ quality: Int) throws -> RGBImage {
        let profileToken=Profile.begin("image.jpeg");defer{Profile.end("image.jpeg",profileToken)}

        var out=bytes; var message=[CChar](repeating:0,count:1024)
        let status=facade_jpeg_roundtrip(bytes,Int32(width),Int32(height),Int32(quality),&out,&message,message.count)
        guard status==0 else { throw NativeError.invalid("JPEG processing failed: \(String(cString:message))") }
        return try RGBImage(width:width,height:height,bytes:out)
    }
    /// Resizes with Pillow-compatible filters: 1 = bilinear, 3 = Lanczos.
    func resize(_ w: Int,_ h: Int, filter: Int) throws -> RGBImage {
        let profileToken=Profile.begin("image.resize");defer{Profile.end("image.resize",profileToken)}

        var out=[UInt8](repeating:0,count:w*h*3)
        let status=facade_resize_rgb(bytes,Int32(width),Int32(height),&out,Int32(w),Int32(h),Int32(filter))
        guard status==0 else { throw NativeError.invalid("Image resize failed") }
        return try RGBImage(width:w,height:h,bytes:out)
    }
    /// The three "upload conditions" the protection is optimised against, a rough
    /// imitation of what happens when a photo is shared online:
    /// 0 = unchanged, 1 = JPEG quality 70, 2 = JPEG quality 75, shrunk to half
    /// size and enlarged back (both with Lanczos).
    func handled(_ condition: Int) throws -> RGBImage {
        switch condition {
        case 0:return self
        case 1:return try jpeg(70)
        case 2:return try jpeg(75).resize(width/2,height/2,filter:3).resize(width,height,filter:3)
        default:throw NativeError.invalid("Unknown upload condition")
        }
    }
    /// Cuts out a size x size face crop using OpenCV. `matrix` is a 2x3 affine
    /// transform from photo coordinates to crop coordinates (see NativeVision).
    /// This is the exact 8-bit crop, not differentiable.
    func crop(_ matrix: [Float], size: Int) throws -> RGBImage {
        let profileToken=Profile.begin("image.crop");defer{Profile.end("image.crop",profileToken)}

        guard matrix.count==6, matrix.allSatisfy(\.isFinite) else { throw NativeError.invalid("Invalid face alignment") }
        var out=[UInt8](repeating:0,count:size*size*3)
        guard facade_cv_warp_rgb(bytes,Int32(width),Int32(height),matrix,&out,Int32(size))==0 else { throw NativeError.invalid("Face crop failed") }
        return try RGBImage(width:size,height:size,bytes:out)
    }
}
/// Float operations on channel-first tensors, most of them thin calls into C.
enum TensorMath {
    /// Low-frequency filter: on every 8x8 block keeps only the 10 lowest DCT
    /// frequencies. The optimiser runs its changes through this so the
    /// protection pattern is smooth rather than speckled. With `rescale` on, the
    /// result is scaled so its mean absolute value matches the input's.
    static func project(_ x:[Float], width:Int,height:Int,rescale:Bool=true)->[Float] {
        let profileToken=Profile.begin("tensor.dct");defer{Profile.end("tensor.dct",profileToken)}

        // The filter works on whole 8x8 blocks; the 1024 px canvas and the 256 px
        // stage image are both multiples of 8.
        precondition(width%8==0 && height%8==0 && x.count%(width*height)==0)
        var out=[Float](repeating:0,count:x.count)
        facade_dct_low(x,&out,Int32(x.count/(width*height)),Int32(width),Int32(height),rescale ? 1:0)
        return out
    }
    /// Bilinear resize of a 3-channel float tensor. `antialias` widens the filter
    /// when shrinking, so fine detail is averaged instead of skipped.
    static func resize(_ x:[Float],width:Int,height:Int,outWidth:Int,outHeight:Int,antialias:Bool=false)->[Float] {
        let profileToken=Profile.begin("tensor.resize");defer{Profile.end("tensor.resize",profileToken)}

        precondition(x.count==3*width*height)
        var out=[Float](repeating:0,count:3*outWidth*outHeight)
        facade_resize_float(x,Int32(width),Int32(height),&out,Int32(outWidth),Int32(outHeight),antialias ? 1:0)
        return out
    }
    /// The gradient of `resize`: takes a gradient on the resized tensor and returns
    /// the gradient on the original-size tensor.
    static func resizeBackward(_ g:[Float],width:Int,height:Int,originalWidth:Int,originalHeight:Int,antialias:Bool=false)->[Float] {
        let profileToken=Profile.begin("tensor.resize.backward");defer{Profile.end("tensor.resize.backward",profileToken)}

        precondition(g.count==3*width*height)
        var out=[Float](repeating:0,count:3*originalWidth*originalHeight)
        facade_resize_backward(g,Int32(width),Int32(height),&out,Int32(originalWidth),Int32(originalHeight),antialias ? 1:0)
        return out
    }
    /// Element-wise average of equally sized arrays (used to combine gradients).
    static func mean(_ xs:[[Float]])->[Float] {
        let profileToken=Profile.begin("tensor.mean");defer{Profile.end("tensor.mean",profileToken)}

        precondition(!xs.isEmpty && xs.allSatisfy{$0.count==xs[0].count})
        var out=[Float](repeating:0,count:xs[0].count)
        for x in xs { for i in out.indices { out[i]+=x[i] } }
        let n=Float(xs.count);for i in out.indices { out[i]/=n };return out
    }
    /// Scales a vector to length 1. The 1e-12 floor avoids dividing by zero.
    static func normalized(_ x:[Float])->[Float] { let n=sqrt(x.reduce(Float(0)){$0+$1*$1});return x.map{$0/max(n,1e-12)} }
    /// Clips every value to -limit...limit (the per-pixel change limit).
    static func clamp(_ x:[Float], _ limit:Float)->[Float] { x.map{min(limit,max(-limit,$0))} }
    /// -1, 0 or 1. The optimiser steps by the sign of the gradient, not its size,
    /// so each step moves every pixel by the same small amount.
    static func sign(_ x:Float)->Float { x>0 ? 1 : x<0 ? -1 : 0 }
    /// Inverts a 2x3 affine matrix. Throws if it cannot be inverted (for example a
    /// face alignment that collapsed to a line).
    static func inverse(_ m:[Float]) throws ->[Float] { var n=[Float](repeating:0,count:6);guard m.count==6,facade_inverse(m,&n)==0 else { throw NativeError.invalid("Singular face alignment") };return n }
    /// Differentiable float version of `RGBImage.crop`. Note that it takes the
    /// INVERSE matrix (crop coordinates to photo coordinates).
    static func crop(_ x:[Float],width:Int,height:Int,inverse:[Float],size:Int)->[Float] {
        let profileToken=Profile.begin("tensor.crop");defer{Profile.end("tensor.crop",profileToken)}

        var out=[Float](repeating:0,count:3*size*size);facade_float_crop(x,Int32(width),Int32(height),inverse,&out,Int32(size));return out
    }
    /// The gradient of `crop`: spreads a crop-sized gradient back onto the full
    /// image. Pixels outside the face crop get zero.
    static func cropBackward(_ g:[Float],size:Int,inverse:[Float],width:Int,height:Int)->[Float] {
        let profileToken=Profile.begin("tensor.crop.backward");defer{Profile.end("tensor.crop.backward",profileToken)}

        var out=[Float](repeating:0,count:3*width*height);facade_crop_backward(g,Int32(size),inverse,&out,Int32(width),Int32(height));return out
    }
}
/// A loaded Core ML model with a simple "names in, float arrays out" interface.
/// Used for the face detector and the subject parser.
final class NativeModel {
    let model:MLModel
    /// Loads a model. A source package (.mlpackage) is compiled on the phone first;
    /// a compiled model (.mlmodelc) is used directly. CPU-only is the default
    /// because CPU results repeat exactly from run to run.
    init(url:URL,computeUnits:MLComputeUnits = .cpuOnly) throws {
        let profileToken=Profile.begin("model.coreml.load.total");defer{Profile.end("model.coreml.load.total",profileToken)}

        model=try autoreleasepool {
            let config=MLModelConfiguration();config.computeUnits = computeUnits
            let compiled=url.pathExtension=="mlmodelc" ? url : try Profile.measure("model.coreml.compile"){try MLModel.compileModel(at:url)}
            return try Profile.measure("model.coreml.load"){try MLModel(contentsOf:compiled,configuration:config)}
        }
    }
    /// Runs the model once. `inputs` maps each input name to its values and shape.
    /// Every output must be a float32 array and finite; the values are copied out
    /// so they stay valid after Core ML releases its buffers.
    func predict(_ inputs:[String:([Float],[Int])]) throws ->[String:[Float]] {
        let profileToken=Profile.begin("model.coreml.predict.total");defer{Profile.end("model.coreml.predict.total",profileToken)}

        // Copy outputs into Swift-owned arrays before draining Core ML temporaries.
        // This synchronous loop can run for minutes without a run-loop pool drain.
        return try autoreleasepool {
        var dictionary=[String:Any]()
        for (name,(values,shape)) in inputs {
            // The value count must equal the product of the shape, e.g. 1*3*640*640.
            guard values.count==shape.reduce(1,*),values.allSatisfy(\.isFinite) else { throw NativeError.invalid("Invalid model input") }
            let array=try MLMultiArray(shape:shape.map{NSNumber(value:$0)},dataType:.float32)
            values.withUnsafeBufferPointer { array.dataPointer.assumingMemoryBound(to:Float.self).update(from:$0.baseAddress!,count:values.count) }
            dictionary[name]=array
        }
        let prediction=try Profile.measure("model.coreml.forward"){try model.prediction(from:MLDictionaryFeatureProvider(dictionary:dictionary))}
        var result=[String:[Float]]()
        for name in prediction.featureNames {
            guard let array=prediction.featureValue(for:name)?.multiArrayValue,array.dataType == .float32 else { throw NativeError.invalid("Model output format changed") }
            // Core ML may pad rows in memory. If every stride matches a tightly packed
            // array, copy the memory in one go; otherwise read element by element.
            let dims=array.shape.map(\.intValue),strides=array.strides.map(\.intValue)
            var expected=1,contiguous=true
            for i in dims.indices.reversed() { if dims[i]>1 && strides[i] != expected { contiguous=false };expected*=dims[i] }
            let out: [Float]
            if contiguous { out=Array(UnsafeBufferPointer(start:array.dataPointer.assumingMemoryBound(to:Float.self),count:array.count)) }
            else { out=(0..<array.count).map{array[$0].floatValue} }
            guard out.allSatisfy(\.isFinite) else { throw NativeError.invalid("Non-finite model output") };result[name]=out
        }
        return result
        }
    }
}
/// One recognition result without a gradient: the 512-number identity vector
/// and its score against the reference.
struct ProxyForwardValue { let embedding:[Float],score:Float }
/// The same plus `gradient`: how the score changes with each input value of
/// the 3x112x112 crop.
struct ProxyValue { let embedding:[Float],score:Float,gradient:[Float] }
/// One of the two face-recognition networks the engine attacks ("proxies" for
/// what a face swapper reads). The score is the dot product of the crop's
/// identity vector with a reference vector; lower means the protected face
/// matches the original less.
///
/// The network can come from two places:
/// - graph.json + weights.f32: run by the C++ engine in Native/, which also
///   computes the exact gradient. This is what the app uses.
/// - a Core ML model with inputs "rgb01" and "reference" and outputs
///   "embedding", "score" and "gradient". The app does not load one today.
final class NativeProxyModel {
    /// Set only for the Core ML form.
    let model:NativeModel?
    /// Handle to the C++ network, set only for the graph.json form.
    var native:UnsafeMutableRawPointer?
    /// When true, the Swift side applies the standard ImageNet mean/std
    /// normalisation before calling the Core ML form (and corrects the gradient).
    let externalNormalization:Bool
    /// Loads a network. For graph.json, weights.f32 must sit in the same folder and
    /// match the SHA-256 written in graph.json, or loading is refused.
    init(url:URL,externalNormalization:Bool=false) throws {
        let profileToken=Profile.begin("model.proxy.load");defer{Profile.end("model.proxy.load",profileToken)}

        self.externalNormalization=externalNormalization
        if url.pathExtension=="json" {
            model=nil;var message=[CChar](repeating:0,count:2048)
            let weights=url.deletingLastPathComponent().appendingPathComponent("weights.f32")
            let config=try JSONSerialization.jsonObject(with:Data(contentsOf:url)) as? [String:Any]
            guard let expected=config?["weightsSHA256"] as? String,TightCleanup.sha(try Data(contentsOf:weights,options:.mappedIfSafe))==expected else {throw NativeError.invalid("Native model checksum failed")}
            native=facade_network_load(url.path,weights.path,&message,message.count)
            guard native != nil else {throw NativeError.invalid(String(cString:message))}
        }else {model=try NativeModel(url:url)}
    }
    /// Releases the C++ network.
    deinit {if let native {facade_network_free(native)}}
    /// Score and gradient for one 112x112 face crop (values 0...1).
    func evaluate(_ crop:[Float],reference:[Float]) throws ->ProxyValue {
        let profileToken=Profile.begin("proxy.evaluate");defer{Profile.end("proxy.evaluate",profileToken)}

        guard crop.count==3*112*112,reference.count==512 else {throw NativeError.invalid("Invalid face model shape")}
        if let native {
            // Output buffers: embedding (512), score (1), gradient (one per input value),
            // and room for an error message from C++.
            var e=[Float](repeating:0,count:512),s:Float=0,g=[Float](repeating:0,count:crop.count),message=[CChar](repeating:0,count:2048)
            guard facade_network_predict(native,crop,reference,&e,&s,&g,&message,message.count)==0,e.allSatisfy(\.isFinite),s.isFinite,g.allSatisfy(\.isFinite) else {throw NativeError.invalid("Native model evaluation failed: \(String(cString:message))")}
            return ProxyValue(embedding:e,score:s,gradient:g)
        }
        // Core ML form only. ImageNet channel means and standard deviations.
        var input=crop;let means:[Float]=[0.485,0.456,0.406],std:[Float]=[0.229,0.224,0.225]
        if externalNormalization {for c in 0..<3 {for p in 0..<(112*112) {let i=c*112*112+p;input[i]=(input[i]-means[c])/std[c]}}}
        let o=try model!.predict(["rgb01":(input,[1,3,112,112]),"reference":(reference,[1,512])])
        guard let e=o["embedding"],let s=o["score"],let g=o["gradient"],e.count==512,s.count==1,g.count==crop.count else {throw NativeError.invalid("Protection model outputs changed")}
        // Chain rule: the input was divided by std, so the gradient is too.
        // offset/(112*112) is the channel index of a planar position.
        let gradient=externalNormalization ? g.enumerated().map{$0.element/std[$0.offset/(112*112)]}:g
        return ProxyValue(embedding:e,score:s[0],gradient:gradient)
    }
    /// Score without a gradient, which is much cheaper in the C++ engine. The Core
    /// ML form has no score-only path, so it runs `evaluate` and drops the gradient.
    func forward(_ crop:[Float],reference:[Float]) throws ->ProxyForwardValue {
        let profileToken=Profile.begin("proxy.forward-only");defer{Profile.end("proxy.forward-only",profileToken)}
        guard crop.count==3*112*112,reference.count==512 else {throw NativeError.invalid("Invalid face model shape")}
        guard let native else {
            let value=try evaluate(crop,reference:reference)
            return ProxyForwardValue(embedding:value.embedding,score:value.score)
        }
        var e=[Float](repeating:0,count:512),s:Float=0,message=[CChar](repeating:0,count:2048)
        guard facade_network_forward(native,crop,reference,&e,&s,&message,message.count)==0,e.allSatisfy(\.isFinite),s.isFinite else {throw NativeError.invalid("Native forward failed: \(String(cString:message))")}
        return ProxyForwardValue(embedding:e,score:s)
    }
    /// Picks `evaluate` or `forward`. The gradient is empty when it was not asked for.
    func evaluate(_ crop:[Float],reference:[Float],needsGradient:Bool) throws ->ProxyValue {
        if needsGradient {return try evaluate(crop,reference:reference)}
        let value=try forward(crop,reference:reference)
        return ProxyValue(embedding:value.embedding,score:value.score,gradient:[])
    }
    /// The identity vector alone. A zero reference is passed only because the
    /// network call needs one; the score it produces is ignored.
    func embed(_ crop:[Float]) throws ->[Float] {try Profile.measure("proxy.embed"){try forward(crop,reference:[Float](repeating:0,count:512)).embedding}}
}
