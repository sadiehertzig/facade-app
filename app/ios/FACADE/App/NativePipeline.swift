import CNative
import Foundation
import Accelerate
import CoreML
import FACADEKit

struct PipelineOutput {
    let image:RGBImage
    let energy80:RGBImage
    let stages:[String:RGBImage]
    let trace:[OptimizationRow]
    let seconds:Double
    let cadence:NativeCadenceReport
}
final class NativePipeline {
    let detector:NativeFaceDetector,parser:NativeSubjectParser,models:[NativeProxyModel]
    let codec:any FACADELocalVerificationImageCodec
    let initializer:any FACADEInitializerResidualBank
    let preparationDetector:any FACADEExactlyOneFaceDetector
    let referenceModel:FACADECoreMLEmbeddingModel
    let emap:[Float]
    init(root:URL,inswapper:String,simswap:String,detectorComputeUnits:MLComputeUnits = .cpuOnly) throws {
        let profileToken=Profile.begin("pipeline.load");defer{Profile.end("pipeline.load",profileToken)}

        if Profile.deadline != nil {try Profile.boundary("pipeline.load.before")}
        let runtime=try Profile.measure("model.legacy.loader"){try autoreleasepool {
            return try NativePreparationLoader.load(rootURL:root.appendingPathComponent("FACADEModelHandoff").resolvingSymlinksInPath())
        }}
        if Profile.deadline != nil {try Profile.boundary("pipeline.preparation.after")}
        codec=runtime.codec;initializer=runtime.initializer;preparationDetector=runtime.detector;referenceModel=runtime.arcFace
        let folder=root.appendingPathComponent("NativeModels")
        let first=try NativeProxyModel(url:folder.appendingPathComponent(inswapper))
        if Profile.deadline != nil {try Profile.boundary("pipeline.proxy.inswapper.after")}
        let second=try NativeProxyModel(url:folder.appendingPathComponent(simswap),externalNormalization:!simswap.hasSuffix(".json"))
        models=[first,second]
        if Profile.deadline != nil {try Profile.boundary("pipeline.proxy.simswap.after")}
        detector=try NativeFaceDetector(url:folder.appendingPathComponent("detector.mlpackage"),computeUnits:detectorComputeUnits)
        if Profile.deadline != nil {try Profile.boundary("pipeline.detector.after")}
        parser=try NativeSubjectParser(url:folder.appendingPathComponent("parser.mlpackage"))
        if Profile.deadline != nil {try Profile.boundary("pipeline.parser.after")}
        let data=try Data(contentsOf:folder.appendingPathComponent("emap.f32"));guard data.count==512*512*4 else {throw NativeError.invalid("Projection matrix shape changed")}
        emap=data.withUnsafeBytes {raw in (0..<(512*512)).map{Float(bitPattern:UInt32(littleEndian:raw.loadUnaligned(fromByteOffset:4*$0,as:UInt32.self)))}}
        if Profile.deadline != nil {try Profile.boundary("pipeline.load.after")}
    }
    private func phoneBoundary(_ label:String)throws {if Profile.deadline != nil {try Profile.boundary(label)}}
    func projectedReference(_ input:[Float])->[Float] {
        let a=TensorMath.normalized(input);var result=[Float](repeating:0,count:512)
        cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,1,512,512,1,a,512,emap,512,0,&result,512)
        return TensorMath.normalized(result)
    }
    func coverage(data:Data,optimizer:NativeOptimizer) async throws ->(RGBImage,RGBImage) {
        let profileToken=Profile.begin("stage.coverage");defer{Profile.end("stage.coverage",profileToken)}

        let clean=try Profile.measure("source.decode.prepare"){try codec.decodeAndPrepare(data)},empty=try RGBImage(width:256,height:256,bytes:[UInt8](repeating:0,count:3*256*256)),rgb=try empty.stored(clean.values)
        let transform=try Profile.measure("source.preparation.detector"){try preparationDetector.alignmentTransform(in:clean)}
        let headStart=Profile.begin("source.r6");let head=try await initializer.residuals(for:clean);Profile.end("source.r6",headStart)
        guard let initial=head.first else {throw NativeError.invalid("No initialization head")}
        let reference=projectedReference(try Profile.measure("source.arc.reference"){try referenceModel.embedding(for:transform.alignedImage(from:clean))})
        let simMatrix=try detector.alignments(rgb)[1],simInverse=try TensorMath.inverse(simMatrix)
        let simClean224=TensorMath.crop(clean.values,width:256,height:256,inverse:simInverse,size:224)
        let simReference=try models[1].embed(TensorMath.resize(simClean224,width:224,height:224,outWidth:112,outHeight:112))
        let limit:Float=5/255
        var d=TensorMath.clamp(TensorMath.project(TensorMath.clamp(initial.values,limit),width:256,height:256),limit)
        var best=rgb,bestScore=Float.infinity
        for step in 0...NativeSchedule.steps {
            try optimizer.tick("coverage",step)
            let needsGradient=step<NativeSchedule.steps
            let values=zip(clean.values,d).map{min(1,max(0,$0+$1))},image=try FACADEFloatImage(width:256,height:256,values:values)
            let crop=try transform.alignedImage(from:image),arc=try models[0].evaluate(crop.values,reference:reference,needsGradient:needsGradient)
            let arcGradient:[Float]
            if needsGradient {arcGradient=try transform.backProject(alignedGradient:FACADEFloatImage(width:112,height:112,values:arc.gradient),sourceWidth:256,sourceHeight:256).values}else{arcGradient=[]}
            let simCrop224=TensorMath.crop(values,width:256,height:256,inverse:simInverse,size:224)
            let sim=try models[1].evaluate(TensorMath.resize(simCrop224,width:224,height:224,outWidth:112,outHeight:112),reference:simReference,needsGradient:needsGradient)
            let simGradient:[Float]
            if needsGradient {
                let simGradient224=TensorMath.resizeBackward(sim.gradient,width:112,height:112,originalWidth:224,originalHeight:224)
                simGradient=TensorMath.cropBackward(simGradient224,size:224,inverse:simInverse,width:256,height:256)
            }else{simGradient=[]}
            let score=(arc.score+sim.score)*0.5,stored=try rgb.stored(values)
            if score<bestScore {best=stored;bestScore=score}
            optimizer.rows.append(OptimizationRow(stage:"coverage",step:step,scores:[arc.score,sim.score],mse:NativeQuality.mse(rgb,stored),eligible:true))
            if step==NativeSchedule.steps {break}
            let g=TensorMath.project(TensorMath.mean([arcGradient,simGradient]),width:256,height:256),rate:Float=(step<NativeSchedule.halfSteps ? 1:0.1)/255
            d=TensorMath.clamp(TensorMath.project(TensorMath.clamp(zip(d,g).map{$0-rate*TensorMath.sign($1)},limit),width:256,height:256),limit)
        };return (rgb,best)
    }
    func run(data:Data,clean:RGBImage,cadenceInterval:Int=1,progress:@escaping(String,Int)->Void) async throws ->PipelineOutput {
        let profileToken=Profile.begin("pipeline.run");defer{Profile.end("pipeline.run",profileToken)}

        guard clean.width%8==0,clean.height%8==0,clean.width>=128,clean.height>=128 else {throw NativeError.invalid("Unsupported image dimensions")}
        let started=Date(),cadence=try NativeCadenceRun(interval:cadenceInterval,halfSteps:NativeSchedule.halfSteps,totalSteps:NativeSchedule.steps)
        let optimizer=NativeOptimizer(progress:progress,cadence:cadence)
        try phoneBoundary("pipeline.source.before")
        _=try detector.detect(clean)
        try phoneBoundary("stage.coverage.before")
        let (lowClean,coverage)=try await self.coverage(data:data,optimizer:optimizer)
        try phoneBoundary("stage.coverage.after")
        var delta=[Float](repeating:0,count:coverage.bytes.count)
        for i in delta.indices {delta[i]=Float(Int(coverage.bytes[i])-Int(lowClean.bytes[i]))}
        var lifted=[Float](repeating:0,count:clean.bytes.count)
        guard facade_cv_lift(delta,256,256,&lifted,Int32(clean.width),Int32(clean.height))==0 else {throw NativeError.invalid("Residual resize failed")}
        let lift=try RGBImage(width:clean.width,height:clean.height,bytes:zip(clean.bytes,lifted).map{UInt8(min(255,max(0,floor(Double($0)+Double(min(5,max(-5,$1)))+0.5))))})
        try phoneBoundary("stage.native_eot8.before")
        let fixed=try NativeAlignedProxies(clean:clean,detector:detector,models:models,exactCrop:false)
        let eot=try optimizer.native(fixed,initial:lift,stage:"native_eot8")
        try phoneBoundary("stage.native_eot8.after")
        let aligned=try NativeAlignedProxies(clean:clean,detector:detector,models:models)
        try phoneBoundary("stage.native_realign8.before")
        let refreshed=try optimizer.native(aligned,initial:eot,stage:"native_realign8")
        try phoneBoundary("stage.native_realign8.after")
        try phoneBoundary("stage.energy80.before")
        let energy80=try optimizer.native(aligned,initial:refreshed,stage:"energy80")
        try phoneBoundary("stage.energy80.after")
        try phoneBoundary("stage.background.before")
        progress("background",0)
        let subject=try parser.subject(clean),alpha=try TightCleanup.alpha(subject:subject,width:clean.width,height:clean.height)
        let nearAlpha=try TightCleanup.alpha(subject:subject,width:clean.width,height:clean.height,margin:72,feather:1)
        let near=subject.indices.map{UInt8(subject[$0]==0 && nearAlpha[$0]==1 ? 1:0)}
        let tight=try RGBImage(width:clean.width,height:clean.height,bytes:TightCleanup.composite(clean:clean.bytes,protected:energy80.bytes,alpha:alpha))
        try phoneBoundary("stage.background.after")
        try phoneBoundary("stage.masked_compensation.before")
        let masked=try optimizer.masked(aligned,energy80:energy80,initial:tight,baseline:tight,alpha:alpha,subject:subject,near:near,balanced:false)
        try phoneBoundary("stage.masked_compensation.after")
        try phoneBoundary("stage.balanced_continue.before")
        let final=try optimizer.masked(aligned,energy80:energy80,initial:masked,baseline:tight,alpha:alpha,subject:subject,near:near,balanced:true)
        try phoneBoundary("stage.balanced_continue.after")
        return PipelineOutput(image:final,energy80:energy80,stages:["coverage":coverage,"native_lift":lift,"native_eot8":eot,"native_realign8":refreshed,"energy80":energy80,"tight":tight,"masked":masked,"balanced_continue":final],trace:optimizer.rows,seconds:Date().timeIntervalSince(started),cadence:try cadence.report())
    }
}
