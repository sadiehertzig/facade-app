// NativeVision.swift
//
// Finding the face, finding the person, and scoring a candidate image with the
// two face-recognition networks.
//
// Where it sits: NativePipeline creates one NativeFaceDetector and one
// NativeSubjectParser when it loads, and builds NativeAlignedProxies for each
// optimisation stage. NativeOptimizer then calls `scores` and `evaluate` on
// every step.
//
// Key ideas:
// 1. The detector returns five landmarks (eyes, nose, mouth corners). They are
//    turned into an affine matrix that cuts out a standard, upright face crop,
//    one for each recognition network, because each was trained on its own
//    crop layout.
// 2. The photo must contain exactly one face. With zero or several, the engine
//    refuses to run rather than guess which face to protect.
// 3. Each candidate is scored under three upload conditions (unchanged, JPEG,
//    JPEG plus half-size; see RGBImage.handled) by two networks: six scores per
//    step. The scores compare against references taken from the clean photo
//    under the same condition.
// 4. Gradients flow back through the crop and resize steps exactly, but the
//    JPEG step and the 8-bit rounding are treated as if they passed the
//    gradient straight through. That is an approximation.
import CNative
import Foundation
import CoreML

/// One detection in photo pixel coordinates: box = [x1, y1, x2, y2],
/// landmarks = five (x, y) points flattened to 10 numbers, score = confidence 0-1.
struct DetectedFace { let box:[Float],landmarks:[Float],score:Float }
/// The 640x640 detector tensor plus the resize factor needed to map results back.
struct NativeDetectorInput { let tensor:[Float],scale:Float }
/// Runs the Core ML face detector (an InsightFace-style detector that predicts
/// faces on three grids of different coarseness).
final class NativeFaceDetector {
    let model:NativeModel
    init(url:URL,computeUnits:MLComputeUnits = .cpuOnly) throws {model=try NativeModel(url:url,computeUnits:computeUnits)}
    /// Finds the single face in `image`, or throws.
    func detect(_ image:RGBImage) throws ->DetectedFace {
        let profileToken=Profile.begin("vision.detect");defer{Profile.end("vision.detect",profileToken)}

        let input=try Self.input(image)
        return try Self.face(outputs:prediction(input),scale:input.scale)
    }
    /// Builds the detector input. It is separate from `detect` so a diagnostic can
    /// run `prediction` twice on one prepared tensor; the normal path runs it once.
    ///
    /// The image is shrunk so its longer side is 640 px, placed in the top-left
    /// corner of a 640x640 square, and scaled to roughly -1...1 with (v - 127.5) / 128.
    static func input(_ image:RGBImage) throws ->NativeDetectorInput {
        let ratio=Double(image.height)/Double(image.width)
        let w=ratio>1 ? Int(640/ratio):640, h=ratio>1 ? 640:Int(640*ratio)
        var resized=[UInt8](repeating:0,count:w*h*3)
        guard facade_cv_resize_rgb(image.bytes,Int32(image.width),Int32(image.height),&resized,Int32(w),Int32(h))==0 else {throw NativeError.invalid("Detector resize failed")}
        // Pre-filled with the value of a black pixel, so the unused area is black padding.
        var tensor=[Float](repeating:-127.5/128,count:3*640*640)
        for y in 0..<h {for x in 0..<w {for c in 0..<3 {tensor[c*640*640+y*640+x]=(Float(resized[(y*w+x)*3+c])-127.5)/128}}}
        return NativeDetectorInput(tensor:tensor,scale:Float(h)/Float(image.height))
    }
    /// One forward pass. Core ML input name "image", shape 1x3x640x640.
    func prediction(_ input:NativeDetectorInput) throws ->[String:[Float]] {
        try model.predict(["image":(input.tensor,[1,3,640,640])])
    }
    /// Turns the raw detector outputs into one face.
    ///
    /// For each grid (cells of 8, 16 and 32 px) the model gives, per cell and per
    /// each of its 2 anchors: a score, four box distances (left, top, right,
    /// bottom) and five landmark offsets, all in units of the cell size. Outputs
    /// are named "score8", "box8", "kps8", and so on.
    static func face(outputs:[String:[Float]],scale:Float) throws ->DetectedFace {
        var all=[DetectedFace]()
        for stride in [8,16,32] {
            // n x n cells, 2 anchors per cell.
            let n=640/stride,count=n*n*2
            guard let scores=outputs["score\(stride)"],let boxes=outputs["box\(stride)"],let points=outputs["kps\(stride)"],scores.count==count,boxes.count==count*4,points.count==count*10 else {throw NativeError.invalid("Detector output shape changed")}
            // Keep only detections with confidence at least 0.5.
            for i in 0..<count where scores[i]>=0.5 {
                // i/2 is the cell index (two anchors share a cell); (x, y) is the cell's
                // corner in detector pixels. Dividing by `scale` maps back to photo pixels.
                let x=Float((i/2)%n*stride),y=Float((i/2)/n*stride),s=Float(stride)
                let b=[(x-boxes[4*i]*s)/scale,(y-boxes[4*i+1]*s)/scale,(x+boxes[4*i+2]*s)/scale,(y+boxes[4*i+3]*s)/scale]
                var kp=[Float]();for k in 0..<5 {kp.append((x+points[10*i+2*k]*s)/scale);kp.append((y+points[10*i+2*k+1]*s)/scale)}
                all.append(DetectedFace(box:b,landmarks:kp,score:scores[i]))
            }
        }
        // Non-maximum suppression: go from most to least confident and drop any box
        // that overlaps an already kept one by more than 0.4 (intersection over union).
        // The +1 terms follow the usual pixel-inclusive box convention.
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
        // Exactly one face, or the run stops with a message the user can act on.
        guard kept.count==1 else {throw NativeError.invalid("Use a photo with exactly one clearly visible face (found \(kept.count)).")}
        return kept[0]
    }
    /// Detects the face and returns its two alignment matrices (see `matrices`).
    func alignments(_ image:RGBImage) throws ->[[Float]] {
        let profileToken=Profile.begin("vision.alignments");defer{Profile.end("vision.alignments",profileToken)}

        return try Self.matrices(landmarks:detect(image).landmarks)
    }
    /// Two 2x3 matrices from photo coordinates to crop coordinates:
    /// [0] the 112x112 ArcFace crop (read by the InSwapper-side network) and
    /// [1] a 224x224 FFHQ-style crop (read by the SimSwap-side network).
    /// Each is the rotation + uniform scale + shift that best fits the five
    /// landmarks onto the template's five points.
    static func matrices(landmarks points:[Float]) throws ->[[Float]] {
        // Standard ArcFace 112 px five-point template (InsightFace, MIT-licensed code).
        let arc:[Float]=[38.2946,51.6963,73.5318,51.5014,56.0252,71.7366,41.5493,92.3655,70.7299,92.2041]
        // The FFHQ five-point template is defined for 512 px; scale it to 224 px.
        let ffhq:[Float]=[192.98138,239.94708,318.90277,240.1936,256.63416,314.01935,201.26117,371.41043,313.08905,371.15118].map{$0*(224.0/512.0)}
        return try [arc,ffhq].map {destination in
            var matrix=[Float](repeating:0,count:6)
            guard facade_similarity(points,destination,&matrix)==0 else {throw NativeError.invalid("Could not align this face")};return matrix
        }
    }
}
/// Runs a face-parsing model that labels every pixel (skin, hair, clothes,
/// background and so on) and keeps a simple yes/no "subject" mask: every label
/// except background. The cleanup step uses this mask to decide where changes
/// may stay.
final class NativeSubjectParser {
    let model:NativeModel
    init(url:URL) throws {model=try NativeModel(url:url,computeUnits:.cpuOnly)}
    /// Returns one byte per pixel: 1 = subject, 0 = background.
    func subject(_ image:RGBImage) throws ->[UInt8] {
        let profileToken=Profile.begin("vision.parser");defer{Profile.end("vision.parser",profileToken)}

        // The model takes 512x512 input with ImageNet mean/std normalisation.
        let resized=try image.resize(512,512,filter:1);var x=resized.tensor();let mean:[Float]=[0.485,0.456,0.406],std:[Float]=[0.229,0.224,0.225]
        for c in 0..<3 {for i in 0..<(512*512) {x[c*512*512+i]=(x[c*512*512+i]-mean[c])/std[c]}}
        // Core ML input "normalized_rgb" (1x3x512x512); output "logits" holds 19
        // label planes of 512x512.
        guard let logits=try model.predict(["normalized_rgb":(x,[1,3,512,512])])["logits"],logits.count==19*512*512 else {throw NativeError.invalid("Subject parser output changed")}
        var small=[UInt8](repeating:0,count:512*512)
        // For each pixel pick the label with the highest logit; label 0 is background.
        for i in small.indices {var label=0,best=logits[i];for c in 1..<19 where logits[c*512*512+i]>best {label=c;best=logits[c*512*512+i]};small[i]=label==0 ? 0:1}
        // Scale the 512x512 mask back to the image size by nearest-neighbour lookup.
        var out=[UInt8](repeating:0,count:image.plane)
        for y in 0..<image.height {for x in 0..<image.width {out[y*image.width+x]=small[min(511,y*512/image.height)*512+min(511,x*512/image.width)]}}
        // A mask that is all subject or all background is useless; refuse to continue.
        guard out.contains(1),out.contains(0) else {throw NativeError.invalid("Could not separate the subject from the background")};return out
    }
}
/// The scoring set-up for one optimisation stage: the clean image, the
/// detector, the two networks, and for each upload condition the clean
/// alignment matrices and reference identity vectors.
final class NativeAlignedProxies {
    let clean:RGBImage,detector:NativeFaceDetector,models:[NativeProxyModel]
    /// true = cut crops with the exact 8-bit OpenCV crop and re-detect the face on
    /// every scoring call; false = use the differentiable float crop and the
    /// alignments found on the clean image (faster, used by the first native stage).
    let exactCrop:Bool
    /// Indexed [condition][model]: alignment matrices and 512-number references.
    var alignments=[[[Float]]](),references=[[[Float]]]()
    /// Computes the references: the clean photo is put through each upload
    /// condition, aligned, and embedded by both networks. Scores later compare a
    /// candidate under condition k with the clean photo under the same condition k.
    init(clean:RGBImage,detector:NativeFaceDetector,models:[NativeProxyModel],exactCrop:Bool=true) throws {
        let profileToken=Profile.begin("proxy.clean.references");defer{Profile.end("proxy.clean.references",profileToken)}

        self.clean=clean;self.detector=detector;self.models=models;self.exactCrop=exactCrop
        for condition in 0..<3 {
            let image=try clean.handled(condition),matrices=try detector.alignments(image)
            alignments.append(matrices)
            references.append(try (0..<2).map {try models[$0].embed(Self.crop(image,matrix:matrices[$0],model:$0,exact:exactCrop))})
        }
    }
    /// The 112x112 float crop that network `model` reads. Network 1 is aligned at
    /// 224 px and then shrunk to 112, which is how its crop was defined.
    static func crop(_ image:RGBImage,matrix:[Float],model:Int,exact:Bool=true) throws ->[Float] {
        let size=model==0 ? 112:224
        let tensor=try exact ? image.crop(matrix,size:size).tensor():TensorMath.crop(image.tensor(),width:image.width,height:image.height,inverse:TensorMath.inverse(matrix),size:size)
        return size==112 ? tensor:TensorMath.resize(tensor,width:224,height:224,outWidth:112,outHeight:112)
    }
    /// The six scores for an 8-bit candidate, no gradients. Order: condition 0
    /// (network 0, network 1), condition 1 (...), condition 2 (...).
    func scores(_ image:RGBImage) throws ->[Float] {
        let profileToken=Profile.begin("proxy.scores");defer{Profile.end("proxy.scores",profileToken)}

        // The three conditions are prepared in parallel and the two networks run in
        // parallel; results are put back in fixed slots, so the order is the same as
        // a plain loop's.
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
    /// Scores and gradients for a float candidate (channel first, 0...1).
    ///
    /// - realign: find the face again on each conditioned image (otherwise use the
    ///   clean alignments). With an `alignmentCadence`, a recent alignment may be
    ///   reused instead (see NativeAlignmentCadence).
    /// - needsGradient / updateStep: which conditions get a gradient is decided
    ///   by NativeViewGradientPolicy.
    /// - stage: "qualification" means a check outside a normal run, which records
    ///   no policy row.
    /// Returns the six scores and the selected gradients, each sized like the
    /// full image.
    func evaluate(_ state:[Float],realign:Bool,needsGradient:Bool=true,updateStep:Int=0,stage:String="qualification",alignmentCadence:NativeAlignmentCadence?=nil) throws ->([Float],[[Float]]) {
        let profileToken=Profile.begin("proxy.upload.evaluate");defer{Profile.end("proxy.upload.evaluate",profileToken)}

        // Round to 8-bit first, so the networks see exactly what a saved file holds.
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
                // Score always; gradient only when the policy selects this condition.
                let selected=NativeViewGradientPolicy.includes(condition,step:updateStep,needsGradient:needsGradient)
                let value=try self.models[model].evaluate(Self.crop(image,matrix:matrix,model:model,exact:self.exactCrop),reference:self.references[condition][model],needsGradient:selected)
                guard selected else {return (value.score,nil)}
                // Back through the 224->112 shrink (network 1 only), then the crop.
                let cropGradient=model==0 ? value.gradient:TensorMath.resizeBackward(value.gradient,width:112,height:112,originalWidth:224,originalHeight:224)
                var full=TensorMath.cropBackward(cropGradient,size:size,inverse:try TensorMath.inverse(matrix),width:self.clean.width,height:self.clean.height)
                // Condition 2 also shrank and re-enlarged the image; carry the gradient back
                // through both resizes, approximated with the bilinear-resize gradient. The
                // JPEG step is treated as passing the gradient straight through.
                if condition==2 {
                    let half=TensorMath.resizeBackward(full,width:self.clean.width,height:self.clean.height,originalWidth:self.clean.width/2,originalHeight:self.clean.height/2,antialias:true)
                    full=TensorMath.resizeBackward(half,width:self.clean.width/2,height:self.clean.height/2,originalWidth:self.clean.width,originalHeight:self.clean.height,antialias:true)
                }
                return (value.score,full)
            }
            // Keep the condition:model labels so NativeViewGradientPolicy can check order.
            for model in 0..<2 {
                scores.append(branch[model].0);scoreOrder.append("\(condition):\(model)")
                if let full=branch[model].1 {gradients.append(full);gradientOrder.append("\(condition):\(model)")}
            }
        }
        // PARALLEL_VIEW_ROUTER_END
        // Record which views got gradients; this also throws if the order is wrong.
        if stage != "qualification" {try NativeViewGradientPolicy.record(stage:stage,step:updateStep,scoreOrder:scoreOrder,gradientOrder:gradientOrder,gradients:gradients)}
        return (scores,gradients)
    }
}
