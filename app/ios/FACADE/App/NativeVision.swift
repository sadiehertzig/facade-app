import CNative
import Foundation
import CoreML

struct DetectedFace { let box:[Float],landmarks:[Float],score:Float }
struct NativeDetectorInput { let tensor:[Float],scale:Float }
final class NativeFaceDetector {
    let model:NativeModel
    init(url:URL,computeUnits:MLComputeUnits = .cpuOnly) throws {model=try NativeModel(url:url,computeUnits:computeUnits)}
    func detect(_ image:RGBImage) throws ->DetectedFace {
        let profileToken=Profile.begin("vision.detect");defer{Profile.end("vision.detect",profileToken)}

        let input=try Self.input(image)
        return try Self.face(outputs:prediction(input),scale:input.scale)
    }
    // The paired diagnostic calls prediction twice on this same prepared tensor.
    // Normal pipeline calls retain one prediction and no diagnostic capture.
    static func input(_ image:RGBImage) throws ->NativeDetectorInput {
        let ratio=Double(image.height)/Double(image.width)
        let w=ratio>1 ? Int(640/ratio):640, h=ratio>1 ? 640:Int(640*ratio)
        var resized=[UInt8](repeating:0,count:w*h*3)
        guard facade_cv_resize_rgb(image.bytes,Int32(image.width),Int32(image.height),&resized,Int32(w),Int32(h))==0 else {throw NativeError.invalid("Detector resize failed")}
        var tensor=[Float](repeating:-127.5/128,count:3*640*640)
        for y in 0..<h {for x in 0..<w {for c in 0..<3 {tensor[c*640*640+y*640+x]=(Float(resized[(y*w+x)*3+c])-127.5)/128}}}
        return NativeDetectorInput(tensor:tensor,scale:Float(h)/Float(image.height))
    }
    func prediction(_ input:NativeDetectorInput) throws ->[String:[Float]] {
        try model.predict(["image":(input.tensor,[1,3,640,640])])
    }
    static func face(outputs:[String:[Float]],scale:Float) throws ->DetectedFace {
        var all=[DetectedFace]()
        for stride in [8,16,32] {
            let n=640/stride,count=n*n*2
            guard let scores=outputs["score\(stride)"],let boxes=outputs["box\(stride)"],let points=outputs["kps\(stride)"],scores.count==count,boxes.count==count*4,points.count==count*10 else {throw NativeError.invalid("Detector output shape changed")}
            for i in 0..<count where scores[i]>=0.5 {
                let x=Float((i/2)%n*stride),y=Float((i/2)/n*stride),s=Float(stride)
                let b=[(x-boxes[4*i]*s)/scale,(y-boxes[4*i+1]*s)/scale,(x+boxes[4*i+2]*s)/scale,(y+boxes[4*i+3]*s)/scale]
                var kp=[Float]();for k in 0..<5 {kp.append((x+points[10*i+2*k]*s)/scale);kp.append((y+points[10*i+2*k+1]*s)/scale)}
                all.append(DetectedFace(box:b,landmarks:kp,score:scores[i]))
            }
        }
        all.sort{$0.score>$1.score};var kept=[DetectedFace]()
        for face in all {
            let b=face.box,area=(b[2]-b[0]+1)*(b[3]-b[1]+1)
            let suppressed=kept.contains {existing in
                let a=existing.box,other=(a[2]-a[0]+1)*(a[3]-a[1]+1)
                let overlap=max(Float(0),min(a[2],b[2])-max(a[0],b[0])+1)*max(Float(0),min(a[3],b[3])-max(a[1],b[1])+1)
                return overlap/(area+other-overlap)>0.4
            }
            if !suppressed {kept.append(face)}
        }
        guard kept.count==1 else {throw NativeError.invalid("Use a photo with exactly one clearly visible face (found \(kept.count)).")}
        return kept[0]
    }
    func alignments(_ image:RGBImage) throws ->[[Float]] {
        let profileToken=Profile.begin("vision.alignments");defer{Profile.end("vision.alignments",profileToken)}

        return try Self.matrices(landmarks:detect(image).landmarks)
    }
    static func matrices(landmarks points:[Float]) throws ->[[Float]] {
        // Standard ArcFace 112 px five-point template (InsightFace, MIT-licensed code).
        let arc:[Float]=[38.2946,51.6963,73.5318,51.5014,56.0252,71.7366,41.5493,92.3655,70.7299,92.2041]
        let ffhq:[Float]=[192.98138,239.94708,318.90277,240.1936,256.63416,314.01935,201.26117,371.41043,313.08905,371.15118].map{$0*(224.0/512.0)}
        return try [arc,ffhq].map {destination in
            var matrix=[Float](repeating:0,count:6)
            guard facade_similarity(points,destination,&matrix)==0 else {throw NativeError.invalid("Could not align this face")};return matrix
        }
    }
}
final class NativeSubjectParser {
    let model:NativeModel
    init(url:URL) throws {model=try NativeModel(url:url,computeUnits:.cpuOnly)}
    func subject(_ image:RGBImage) throws ->[UInt8] {
        let profileToken=Profile.begin("vision.parser");defer{Profile.end("vision.parser",profileToken)}

        let resized=try image.resize(512,512,filter:1);var x=resized.tensor();let mean:[Float]=[0.485,0.456,0.406],std:[Float]=[0.229,0.224,0.225]
        for c in 0..<3 {for i in 0..<(512*512) {x[c*512*512+i]=(x[c*512*512+i]-mean[c])/std[c]}}
        guard let logits=try model.predict(["normalized_rgb":(x,[1,3,512,512])])["logits"],logits.count==19*512*512 else {throw NativeError.invalid("Subject parser output changed")}
        var small=[UInt8](repeating:0,count:512*512)
        for i in small.indices {var label=0,best=logits[i];for c in 1..<19 where logits[c*512*512+i]>best {label=c;best=logits[c*512*512+i]};small[i]=label==0 ? 0:1}
        var out=[UInt8](repeating:0,count:image.plane)
        for y in 0..<image.height {for x in 0..<image.width {out[y*image.width+x]=small[min(511,y*512/image.height)*512+min(511,x*512/image.width)]}}
        guard out.contains(1),out.contains(0) else {throw NativeError.invalid("Could not separate the subject from the background")};return out
    }
}
final class NativeAlignedProxies {
    let clean:RGBImage,detector:NativeFaceDetector,models:[NativeProxyModel]
    let exactCrop:Bool
    var alignments=[[[Float]]](),references=[[[Float]]]()
    init(clean:RGBImage,detector:NativeFaceDetector,models:[NativeProxyModel],exactCrop:Bool=true) throws {
        let profileToken=Profile.begin("proxy.clean.references");defer{Profile.end("proxy.clean.references",profileToken)}

        self.clean=clean;self.detector=detector;self.models=models;self.exactCrop=exactCrop
        for condition in 0..<3 {
            let image=try clean.handled(condition),matrices=try detector.alignments(image)
            alignments.append(matrices)
            references.append(try (0..<2).map {try models[$0].embed(Self.crop(image,matrix:matrices[$0],model:$0,exact:exactCrop))})
        }
    }
    static func crop(_ image:RGBImage,matrix:[Float],model:Int,exact:Bool=true) throws ->[Float] {
        let size=model==0 ? 112:224
        let tensor=try exact ? image.crop(matrix,size:size).tensor():TensorMath.crop(image.tensor(),width:image.width,height:image.height,inverse:TensorMath.inverse(matrix),size:size)
        return size==112 ? tensor:TensorMath.resize(tensor,width:224,height:224,outWidth:112,outHeight:112)
    }
    func scores(_ image:RGBImage) throws ->[Float] {
        let profileToken=Profile.begin("proxy.scores");defer{Profile.end("proxy.scores",profileToken)}

        // PARALLEL_VIEW_ROUTER_BEGIN: same conditions, same models, same order.
        let prepared=try facadeParallelMap(3){try image.handled($0)}
        var scores=[Float]()
        for condition in 0..<3 {
            let handled=prepared[condition],matrices=try exactCrop ? detector.alignments(handled):alignments[condition]
            let branch=try facadeParallelMap(2){model in try self.models[model].forward(Self.crop(handled,matrix:matrices[model],model:model,exact:self.exactCrop),reference:self.references[condition][model]).score}
            scores.append(contentsOf:branch)
        };return scores
        // PARALLEL_VIEW_ROUTER_END
    }
    func evaluate(_ state:[Float],realign:Bool,needsGradient:Bool=true,updateStep:Int=0,stage:String="qualification",alignmentCadence:NativeAlignmentCadence?=nil) throws ->([Float],[[Float]]) {
        let profileToken=Profile.begin("proxy.upload.evaluate");defer{Profile.end("proxy.upload.evaluate",profileToken)}

        let stored=try clean.stored(state);var scores=[Float](),gradients=[[Float]](),scoreOrder=[String](),gradientOrder=[String]()
        // PARALLEL_VIEW_ROUTER_BEGIN: the three upload conditions are independent
        // round trips and the two proxies own separate networks. Results are
        // collected into fixed slots, so scoreOrder/gradientOrder and the mean
        // reducer see exactly the sequence the serial loop produced.
        let prepared=try facadeParallelMap(3){try stored.handled($0)}
        for condition in 0..<3 {
            let image=prepared[condition]
            let matrices:[[Float]]
            if realign, let cadence=alignmentCadence {
                matrices=try cadence.resolve(condition:condition,step:updateStep,width:image.width,height:image.height){try detector.alignments(image)}
            } else {
                matrices=try realign ? detector.alignments(image):alignments[condition]
            }
            let branch=try facadeParallelMap(2){model->(Float,[Float]?) in
                let matrix=matrices[model],size=model==0 ? 112:224
                let selected=NativeViewGradientPolicy.includes(condition,step:updateStep,needsGradient:needsGradient)
                let value=try self.models[model].evaluate(Self.crop(image,matrix:matrix,model:model,exact:self.exactCrop),reference:self.references[condition][model],needsGradient:selected)
                guard selected else {return (value.score,nil)}
                let cropGradient=model==0 ? value.gradient:TensorMath.resizeBackward(value.gradient,width:112,height:112,originalWidth:224,originalHeight:224)
                var full=TensorMath.cropBackward(cropGradient,size:size,inverse:try TensorMath.inverse(matrix),width:self.clean.width,height:self.clean.height)
                if condition==2 {
                    let half=TensorMath.resizeBackward(full,width:self.clean.width,height:self.clean.height,originalWidth:self.clean.width/2,originalHeight:self.clean.height/2,antialias:true)
                    full=TensorMath.resizeBackward(half,width:self.clean.width/2,height:self.clean.height/2,originalWidth:self.clean.width,originalHeight:self.clean.height,antialias:true)
                }
                return (value.score,full)
            }
            for model in 0..<2 {
                scores.append(branch[model].0);scoreOrder.append("\(condition):\(model)")
                if let full=branch[model].1 {gradients.append(full);gradientOrder.append("\(condition):\(model)")}
            }
        }
        // PARALLEL_VIEW_ROUTER_END
        if stage != "qualification" {try NativeViewGradientPolicy.record(stage:stage,step:updateStep,scoreOrder:scoreOrder,gradientOrder:gradientOrder,gradients:gradients)}
        return (scores,gradients)
    }
}
