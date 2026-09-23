import CNative
import Foundation
import CoreML

enum NativeError: Error, LocalizedError {
    case invalid(String)
    var errorDescription: String? { if case .invalid(let s) = self { return s }; return nil }
}
struct RGBImage {
    let width: Int, height: Int
    var bytes: [UInt8]
    init(width: Int, height: Int, bytes: [UInt8]) throws {
        guard width > 0, height > 0, width <= 2048, height <= 2048, bytes.count == width*height*3 else { throw NativeError.invalid("Invalid RGB image") }
        self.width=width; self.height=height; self.bytes=bytes
    }
    var plane: Int { width*height }
    func tensor() -> [Float] {
        let profileToken=Profile.begin("image.tensor");defer{Profile.end("image.tensor",profileToken)}

        var out=Profile.measure("allocation.swift.tensor"){[Float](repeating: 0,count: bytes.count)}
        for p in 0..<plane { for c in 0..<3 { out[c*plane+p]=Float(bytes[3*p+c])/255 } }
        return out
    }
    func stored(_ tensor: [Float]) throws -> RGBImage {
        let profileToken=Profile.begin("image.stored");defer{Profile.end("image.stored",profileToken)}

        guard tensor.count==bytes.count, tensor.allSatisfy(\.isFinite) else { throw NativeError.invalid("Non-finite optimizer state") }
        var out=bytes
        for p in 0..<plane { for c in 0..<3 { out[3*p+c]=UInt8(min(255,max(0,floorf(tensor[c*plane+p]*255+0.5)))) } }
        return try RGBImage(width:width,height:height,bytes:out)
    }
    func jpeg(_ quality: Int) throws -> RGBImage {
        let profileToken=Profile.begin("image.jpeg");defer{Profile.end("image.jpeg",profileToken)}

        var out=bytes; var message=[CChar](repeating:0,count:1024)
        let status=facade_jpeg_roundtrip(bytes,Int32(width),Int32(height),Int32(quality),&out,&message,message.count)
        guard status==0 else { throw NativeError.invalid("JPEG processing failed: \(String(cString:message))") }
        return try RGBImage(width:width,height:height,bytes:out)
    }
    func resize(_ w: Int,_ h: Int, filter: Int) throws -> RGBImage {
        let profileToken=Profile.begin("image.resize");defer{Profile.end("image.resize",profileToken)}

        var out=[UInt8](repeating:0,count:w*h*3)
        let status=facade_resize_rgb(bytes,Int32(width),Int32(height),&out,Int32(w),Int32(h),Int32(filter))
        guard status==0 else { throw NativeError.invalid("Image resize failed") }
        return try RGBImage(width:w,height:h,bytes:out)
    }
    func handled(_ condition: Int) throws -> RGBImage {
        switch condition {
        case 0:return self
        case 1:return try jpeg(70)
        case 2:return try jpeg(75).resize(width/2,height/2,filter:3).resize(width,height,filter:3)
        default:throw NativeError.invalid("Unknown upload condition")
        }
    }
    func crop(_ matrix: [Float], size: Int) throws -> RGBImage {
        let profileToken=Profile.begin("image.crop");defer{Profile.end("image.crop",profileToken)}

        guard matrix.count==6, matrix.allSatisfy(\.isFinite) else { throw NativeError.invalid("Invalid face alignment") }
        var out=[UInt8](repeating:0,count:size*size*3)
        guard facade_cv_warp_rgb(bytes,Int32(width),Int32(height),matrix,&out,Int32(size))==0 else { throw NativeError.invalid("Face crop failed") }
        return try RGBImage(width:size,height:size,bytes:out)
    }
}
enum TensorMath {
    static func project(_ x:[Float], width:Int,height:Int,rescale:Bool=true)->[Float] {
        let profileToken=Profile.begin("tensor.dct");defer{Profile.end("tensor.dct",profileToken)}

        precondition(width%8==0 && height%8==0 && x.count%(width*height)==0)
        var out=[Float](repeating:0,count:x.count)
        facade_dct_low(x,&out,Int32(x.count/(width*height)),Int32(width),Int32(height),rescale ? 1:0)
        return out
    }
    static func resize(_ x:[Float],width:Int,height:Int,outWidth:Int,outHeight:Int,antialias:Bool=false)->[Float] {
        let profileToken=Profile.begin("tensor.resize");defer{Profile.end("tensor.resize",profileToken)}

        precondition(x.count==3*width*height)
        var out=[Float](repeating:0,count:3*outWidth*outHeight)
        facade_resize_float(x,Int32(width),Int32(height),&out,Int32(outWidth),Int32(outHeight),antialias ? 1:0)
        return out
    }
    static func resizeBackward(_ g:[Float],width:Int,height:Int,originalWidth:Int,originalHeight:Int,antialias:Bool=false)->[Float] {
        let profileToken=Profile.begin("tensor.resize.backward");defer{Profile.end("tensor.resize.backward",profileToken)}

        precondition(g.count==3*width*height)
        var out=[Float](repeating:0,count:3*originalWidth*originalHeight)
        facade_resize_backward(g,Int32(width),Int32(height),&out,Int32(originalWidth),Int32(originalHeight),antialias ? 1:0)
        return out
    }
    static func mean(_ xs:[[Float]])->[Float] {
        let profileToken=Profile.begin("tensor.mean");defer{Profile.end("tensor.mean",profileToken)}

        precondition(!xs.isEmpty && xs.allSatisfy{$0.count==xs[0].count})
        var out=[Float](repeating:0,count:xs[0].count)
        for x in xs { for i in out.indices { out[i]+=x[i] } }
        let n=Float(xs.count);for i in out.indices { out[i]/=n };return out
    }
    static func normalized(_ x:[Float])->[Float] { let n=sqrt(x.reduce(Float(0)){$0+$1*$1});return x.map{$0/max(n,1e-12)} }
    static func clamp(_ x:[Float], _ limit:Float)->[Float] { x.map{min(limit,max(-limit,$0))} }
    static func sign(_ x:Float)->Float { x>0 ? 1 : x<0 ? -1 : 0 }
    static func inverse(_ m:[Float]) throws ->[Float] { var n=[Float](repeating:0,count:6);guard m.count==6,facade_inverse(m,&n)==0 else { throw NativeError.invalid("Singular face alignment") };return n }
    static func crop(_ x:[Float],width:Int,height:Int,inverse:[Float],size:Int)->[Float] {
        let profileToken=Profile.begin("tensor.crop");defer{Profile.end("tensor.crop",profileToken)}

        var out=[Float](repeating:0,count:3*size*size);facade_float_crop(x,Int32(width),Int32(height),inverse,&out,Int32(size));return out
    }
    static func cropBackward(_ g:[Float],size:Int,inverse:[Float],width:Int,height:Int)->[Float] {
        let profileToken=Profile.begin("tensor.crop.backward");defer{Profile.end("tensor.crop.backward",profileToken)}

        var out=[Float](repeating:0,count:3*width*height);facade_crop_backward(g,Int32(size),inverse,&out,Int32(width),Int32(height));return out
    }
}
final class NativeModel {
    let model:MLModel
    init(url:URL,computeUnits:MLComputeUnits = .cpuOnly) throws {
        let profileToken=Profile.begin("model.coreml.load.total");defer{Profile.end("model.coreml.load.total",profileToken)}

        model=try autoreleasepool {
            let config=MLModelConfiguration();config.computeUnits = computeUnits
            let compiled=url.pathExtension=="mlmodelc" ? url : try Profile.measure("model.coreml.compile"){try MLModel.compileModel(at:url)}
            return try Profile.measure("model.coreml.load"){try MLModel(contentsOf:compiled,configuration:config)}
        }
    }
    func predict(_ inputs:[String:([Float],[Int])]) throws ->[String:[Float]] {
        let profileToken=Profile.begin("model.coreml.predict.total");defer{Profile.end("model.coreml.predict.total",profileToken)}

        // Copy outputs into Swift-owned arrays before draining Core ML temporaries.
        // This synchronous loop can run for minutes without a run-loop pool drain.
        return try autoreleasepool {
        var dictionary=[String:Any]()
        for (name,(values,shape)) in inputs {
            guard values.count==shape.reduce(1,*),values.allSatisfy(\.isFinite) else { throw NativeError.invalid("Invalid model input") }
            let array=try MLMultiArray(shape:shape.map{NSNumber(value:$0)},dataType:.float32)
            values.withUnsafeBufferPointer { array.dataPointer.assumingMemoryBound(to:Float.self).update(from:$0.baseAddress!,count:values.count) }
            dictionary[name]=array
        }
        let prediction=try Profile.measure("model.coreml.forward"){try model.prediction(from:MLDictionaryFeatureProvider(dictionary:dictionary))}
        var result=[String:[Float]]()
        for name in prediction.featureNames {
            guard let array=prediction.featureValue(for:name)?.multiArrayValue,array.dataType == .float32 else { throw NativeError.invalid("Model output format changed") }
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
struct ProxyForwardValue { let embedding:[Float],score:Float }
struct ProxyValue { let embedding:[Float],score:Float,gradient:[Float] }
final class NativeProxyModel {
    let model:NativeModel?
    var native:UnsafeMutableRawPointer?
    let externalNormalization:Bool
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
    deinit {if let native {facade_network_free(native)}}
    func evaluate(_ crop:[Float],reference:[Float]) throws ->ProxyValue {
        let profileToken=Profile.begin("proxy.evaluate");defer{Profile.end("proxy.evaluate",profileToken)}

        guard crop.count==3*112*112,reference.count==512 else {throw NativeError.invalid("Invalid face model shape")}
        if let native {
            var e=[Float](repeating:0,count:512),s:Float=0,g=[Float](repeating:0,count:crop.count),message=[CChar](repeating:0,count:2048)
            guard facade_network_predict(native,crop,reference,&e,&s,&g,&message,message.count)==0,e.allSatisfy(\.isFinite),s.isFinite,g.allSatisfy(\.isFinite) else {throw NativeError.invalid("Native model evaluation failed: \(String(cString:message))")}
            return ProxyValue(embedding:e,score:s,gradient:g)
        }
        var input=crop;let means:[Float]=[0.485,0.456,0.406],std:[Float]=[0.229,0.224,0.225]
        if externalNormalization {for c in 0..<3 {for p in 0..<(112*112) {let i=c*112*112+p;input[i]=(input[i]-means[c])/std[c]}}}
        let o=try model!.predict(["rgb01":(input,[1,3,112,112]),"reference":(reference,[1,512])])
        guard let e=o["embedding"],let s=o["score"],let g=o["gradient"],e.count==512,s.count==1,g.count==crop.count else {throw NativeError.invalid("Protection model outputs changed")}
        let gradient=externalNormalization ? g.enumerated().map{$0.element/std[$0.offset/(112*112)]}:g
        return ProxyValue(embedding:e,score:s[0],gradient:gradient)
    }
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
    func evaluate(_ crop:[Float],reference:[Float],needsGradient:Bool) throws ->ProxyValue {
        if needsGradient {return try evaluate(crop,reference:reference)}
        let value=try forward(crop,reference:reference)
        return ProxyValue(embedding:value.embedding,score:value.score,gradient:[])
    }
    func embed(_ crop:[Float]) throws ->[Float] {try Profile.measure("proxy.embed"){try forward(crop,reference:[Float](repeating:0,count:512)).embedding}}
}
