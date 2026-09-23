import Foundation

struct OptimizationRow:Codable {
    let stage:String,step:Int,scores:[Float],mse:Double,eligible:Bool
}
enum NativeQuality {
    static func mse(_ clean:RGBImage,_ candidate:RGBImage)->Double {
        let profileToken=Profile.begin("quality.mse");defer{Profile.end("quality.mse",profileToken)}

        var sum=0.0;for i in clean.bytes.indices {let d=Double(Int(candidate.bytes[i])-Int(clean.bytes[i]));sum+=d*d};return sum/Double(clean.bytes.count)
    }
    static func flatWeights(_ image:RGBImage)->[Float] {
        let profileToken=Profile.begin("quality.flat.weights");defer{Profile.end("quality.flat.weights",profileToken)}

        let w=image.width,h=image.height,x=image.tensor();var integral=[Double](repeating:0,count:(w+1)*(h+1)),squared=integral
        for y in 0..<h {var row=0.0,row2=0.0;for xx in 0..<w {let p=y*w+xx,v=Double((x[p]+x[image.plane+p]+x[2*image.plane+p])/3);row+=v;row2+=v*v;integral[(y+1)*(w+1)+xx+1]=integral[y*(w+1)+xx+1]+row;squared[(y+1)*(w+1)+xx+1]=squared[y*(w+1)+xx+1]+row2}}
        var out=[Float](repeating:0,count:image.plane)
        for y in 0..<h {for x in 0..<w {let l=max(0,x-4),r=min(w,x+5),t=max(0,y-4),b=min(h,y+5)
            func average(_ a:[Double])->Double {(a[b*(w+1)+r]-a[t*(w+1)+r]-a[b*(w+1)+l]+a[t*(w+1)+l])/81}
            let mean=average(integral),variance=max(0,average(squared)-mean*mean);out[y*w+x]=Float(0.001/(variance+0.001))
        }};return out
    }
    static func artifact(_ clean:RGBImage,_ candidate:RGBImage,flat:[Float])->Double {
        let profileToken=Profile.begin("quality.artifact");defer{Profile.end("quality.artifact",profileToken)}

        let a=clean.tensor(),b=candidate.tensor(),n=clean.plane,w=clean.width,h=clean.height
        var y=[Float](repeating:0,count:n);var square=0.0,chroma=0.0,weighted=0.0
        for p in 0..<n {let r=b[p]-a[p],g=b[n+p]-a[n+p],blue=b[2*n+p]-a[2*n+p],mean=(r+g+blue)/3
            y[p]=0.2126*r+0.7152*g+0.0722*blue
            for d in [r,g,blue] {square+=Double(d*d);chroma+=Double((d-mean)*(d-mean));weighted+=Double(d*d*flat[p])}
        }
        var boundary=0.0,count=0
        for yy in 0..<h {for xx in stride(from:8,to:w,by:8) {let d=y[yy*w+xx]-y[yy*w+xx-1];boundary+=Double(d*d);count+=1}}
        for yy in stride(from:8,to:h,by:8) {for xx in 0..<w {let d=y[yy*w+xx]-y[(yy-1)*w+xx];boundary+=Double(d*d);count+=1}}
        return square/Double(3*n)+0.5*boundary/Double(max(count,1))+0.25*chroma/Double(3*n)+0.5*weighted/Double(3*n)
    }
}
final class MaskBudget {
    let clean:RGBImage,alpha:[Float],subject:[UInt8],near:[UInt8],flat:[Float],caps:[Double],artifactCap:Double
    let regionalEnergyEnabled:Bool
    private let regionalEnergyKernel:RegionalEnergyKernel?
    init(clean:RGBImage,baseline:RGBImage,alpha:[Float],subject:[UInt8],near:[UInt8],regionalEnergyEnabled:Bool=true) {
        self.clean=clean;self.alpha=alpha;self.subject=subject;self.near=near;flat=NativeQuality.flatWeights(clean)
        self.regionalEnergyEnabled=regionalEnergyEnabled
        let kernel=regionalEnergyEnabled ? RegionalEnergyKernel(clean:clean,subject:subject,near:near):nil
        regionalEnergyKernel=kernel
        caps=kernel?.energies(baseline) ?? Self.energies(clean,baseline,subject,near);artifactCap=NativeQuality.artifact(clean,baseline,flat:flat)
    }
    static func energies(_ clean:RGBImage,_ image:RGBImage,_ subject:[UInt8],_ near:[UInt8])->[Double] {
        let profileToken=Profile.begin("quality.regional.energies");defer{Profile.end("quality.regional.energies",profileToken)}

        var sums=[Double](repeating:0,count:4),counts=[Int](repeating:0,count:4)
        for p in 0..<clean.plane {for c in 0..<3 {let d=Double(Int(image.bytes[3*p+c])-Int(clean.bytes[3*p+c])),v=d*d;sums[0]+=v;counts[0]+=1
            let region=subject[p] != 0 ? 1:2;sums[region]+=v;counts[region]+=1;if near[p] != 0 {sums[3]+=v;counts[3]+=1}
        }};return zip(sums,counts).map{$1==0 ? 0:$0/Double($1)}
    }
    static func candidateEnergies(_ clean:RGBImage,_ image:RGBImage,_ subject:[UInt8],_ near:[UInt8])->[Double] {
        RegionalEnergyKernel(clean:clean,subject:subject,near:near)?.energies(image) ?? energies(clean,image,subject,near)
    }
    func regionalEnergies(_ image:RGBImage)->[Double] {
        regionalEnergyKernel?.energies(image) ?? Self.energies(clean,image,subject,near)
    }
    func fits(_ image:RGBImage)->Bool {zip(regionalEnergies(image),caps).allSatisfy{$0<=$1+1e-12}}
    func image(_ latent:[Float],scale:Double=1) throws ->RGBImage {
        let profileToken=Profile.begin("quality.mask.image");defer{Profile.end("quality.mask.image",profileToken)}

        var bytes=clean.bytes;for p in 0..<clean.plane {for c in 0..<3 {let v=Double(clean.bytes[3*p+c])+Double(alpha[p])*Double(latent[c*clean.plane+p])*scale;bytes[3*p+c]=UInt8(min(255,max(0,floor(v+0.5))))}}
        return try RGBImage(width:clean.width,height:clean.height,bytes:bytes)
    }
    func project(_ input:[Float]) throws ->[Float] {
        let profileToken=Profile.begin("quality.mask.project");defer{Profile.end("quality.mask.project",profileToken)}

        var x=input
        for p in 0..<clean.plane {for c in 0..<3 {let i=c*clean.plane+p,v=Float(clean.bytes[3*p+c]);x[i]=max(max(-5,-v),min(min(5,255-v),x[i]))}}
        if try fits(image(x)) {return x};var lo=0.0,hi=1.0
        for _ in 0..<20 {let mid=(lo+hi)/2;if try fits(image(x,scale:mid)){lo=mid}else{hi=mid}}
        for _ in 0..<10000 {let result=x.map{$0*Float(lo)};if try fits(image(result)){return result};lo*=0.999999}
        throw NativeError.invalid("Quality budget did not converge")
    }
    func recover(_ image:RGBImage) throws ->[Float] {
        let profileToken=Profile.begin("quality.mask.recover");defer{Profile.end("quality.mask.recover",profileToken)}

        var x=[Float](repeating:0,count:clean.bytes.count)
        for p in 0..<clean.plane where alpha[p]>0 {for c in 0..<3 {let i=3*p+c,d=Double(Int(image.bytes[i])-Int(clean.bytes[i])),v=Double(clean.bytes[i]);x[c*clean.plane+p]=Float(max(max(-5,-v),min(min(5,255-v),d/Double(alpha[p]))))}}
        if try self.image(x).bytes==image.bytes{return x}
        for p in 0..<clean.plane where alpha[p]>0 {for c in 0..<3 {let i=3*p+c,d=Double(Int(image.bytes[i])-Int(clean.bytes[i])),v=Double(clean.bytes[i]),a=Double(alpha[p]);let low=max((d-0.499999)/a,max(-5,-v)),high=min((d+0.499999)/a,min(5,255-v));x[c*clean.plane+p]=Float((low+high)/2)}}
        guard try self.image(x).bytes==image.bytes else {throw NativeError.invalid("Warm start changed stored pixels")};return x
    }
}
final class NativeOptimizer {
    var rows=[OptimizationRow]()
    let progress:(String,Int)->Void
    let cadence:NativeCadenceRun?
    init(progress:@escaping(String,Int)->Void,cadence:NativeCadenceRun?=nil){self.progress=progress;self.cadence=cadence}
    func tick(_ stage:String,_ step:Int) throws {try Task.checkCancellation();if Profile.deadline != nil {try Profile.boundary("\(stage).\(step)")};progress(stage,step)}
    func mean(_ scores:[Float])->Float {scores.reduce(0,+)/Float(scores.count)}
    func limits(_ scores:[Float])->[Float] {scores.map{1-0.95*max(0,1-$0)}}
    func rank(_ scores:[Float],_ limits:[Float])->(Float,Float) {(zip(scores,limits).map{max(0,$0-$1-1e-6)}.max() ?? 0,mean(scores))}
    func less(_ a:(Float,Float),_ b:(Float,Float))->Bool {a.0<b.0 || (a.0==b.0 && a.1<b.1)}
    func radial(_ d:[Float],clean:RGBImage,cap:Double) throws ->[Float] {
        let profileToken=Profile.begin("quality.energy.project");defer{Profile.end("quality.energy.project",profileToken)}

        guard d.count == clean.bytes.count else {throw NativeError.invalid("Non-finite optimizer state")}
        let c=clean.tensor();var bounded=d
        for i in d.indices {bounded[i]=max(max(-5/255,-c[i]),min(min(5/255,1-c[i]),d[i]))}
        // FUSED_ENERGY_ROUTER_BEGIN: one pass replaces map + stored + mse. Same
        // clamp, same floorf(v*255+0.5) quantization, same interleaved Double
        // accumulation order, so the returned scale is bit-identical.
        let cleanBytes=clean.bytes,plane=clean.plane,total=cleanBytes.count
        func mse(_ scale:Float)throws->Double {
            var sum=0.0
            try c.withUnsafeBufferPointer {cp in try bounded.withUnsafeBufferPointer {bp in
                try cleanBytes.withUnsafeBufferPointer {kp in
                    for i in 0..<total {
                        let j=(i%3)*plane+i/3
                        let v=min(1,max(0,cp[j]+scale*bp[j]))
                        guard v.isFinite else {throw NativeError.invalid("Non-finite optimizer state")}
                        let q=Int(min(255,max(0,(v*255+0.5).rounded(.down))))
                        let d=Double(q-Int(kp[i]));sum+=d*d
                    }
                }
            }}
            return sum/Double(total)
        }
        // FUSED_ENERGY_ROUTER_END
        if try mse(1)<=cap+1e-12 {return bounded};var lo:Float=0,hi:Float=1
        for _ in 0..<20 {let mid=(lo+hi)/2;if try mse(mid)<=cap+1e-12 {lo=mid}else{hi=mid}}
        return bounded.map{$0*lo}
    }
    // Native EOT, refreshed alignment and energy80 share the frozen 25+25 update schedule.
    func native(_ proxy:NativeAlignedProxies,initial:RGBImage,stage:String) throws ->RGBImage {
        let profileToken=Profile.begin("stage."+stage);defer{Profile.end("stage."+stage,profileToken)}

        let clean=proxy.clean,c=clean.tensor(),parentScores=try proxy.scores(initial),upper=limits(parentScores),parentMSE=NativeQuality.mse(clean,initial)
        let energy=stage=="energy80",realign=stage != "native_eot8",cap=energy ? 0.8*parentMSE:1.05*1.05*parentMSE
        let alignmentCadence=try realign ? cadence?.newStage(stage,width:clean.width,height:clean.height):nil
        let lim:Float=5/255
        var d=TensorMath.clamp(TensorMath.project(TensorMath.clamp(zip(initial.tensor(),c).map{$0-$1},lim),width:clean.width,height:clean.height),lim)
        if energy {d=try radial(d,clean:clean,cap:cap)}
        var best:RGBImage?=energy ? nil:initial,bestRank=energy ? (Float.infinity,Float.infinity):(Float(0),mean(parentScores))
        for step in 0...NativeSchedule.steps {
            try tick(stage,step);let x=zip(c,d).map{min(1,max(0,$0+$1))},image=try clean.stored(x)
            let (scores,gradients)=try proxy.evaluate(x,realign:realign,needsGradient:step<NativeSchedule.steps,updateStep:step,stage:stage,alignmentCadence:alignmentCadence),mse=NativeQuality.mse(clean,image)
            let fits=stage=="native_eot8" || energy || (zip(scores,upper).allSatisfy{$0<=$1+1e-6} && mse<=cap+1e-12)
            let r=energy ? rank(scores,upper):(0,mean(scores))
            if fits && (best==nil || less(r,bestRank)){best=image;bestRank=r}
            if energy && mse>cap+1e-12 {throw NativeError.invalid("Stored energy budget exceeded")}
            rows.append(OptimizationRow(stage:stage,step:step,scores:scores,mse:mse,eligible:fits))
            if step==NativeSchedule.steps {break}
            let g=TensorMath.project(try NativeViewGradientPolicy.mean(gradients),width:clean.width,height:clean.height),rate:Float=(step<NativeSchedule.halfSteps ? 1:0.1)/255
            d=TensorMath.clamp(TensorMath.project(TensorMath.clamp(zip(d,g).map{$0-rate*TensorMath.sign($1)},lim),width:clean.width,height:clean.height),lim)
            if energy {d=try radial(d,clean:clean,cap:cap)}
        };guard let result=best else {throw NativeError.invalid("No native candidate")};return result
    }
    func masked(_ proxy:NativeAlignedProxies,energy80:RGBImage,initial:RGBImage,baseline:RGBImage,alpha:[Float],subject:[UInt8],near:[UInt8],balanced:Bool) throws ->RGBImage {
        let profileToken=Profile.begin(balanced ? "stage.balanced_continue":"stage.masked_compensation");defer{Profile.end(balanced ? "stage.balanced_continue":"stage.masked_compensation",profileToken)}

        let budget=MaskBudget(clean:proxy.clean,baseline:baseline,alpha:alpha,subject:subject,near:near),clean=proxy.clean
        let target=try proxy.scores(energy80),upper=limits(target),original=try proxy.scores(initial)
        var best=initial,bestRank=rank(original,upper),d:[Float]
        if balanced {d=try budget.recover(initial)}
        else {let residual=(0..<clean.bytes.count).map {i -> Float in let p=i%clean.plane,c=i/clean.plane;return Float(Int(energy80.bytes[3*p+c])-Int(clean.bytes[3*p+c]))};d=try budget.project(TensorMath.clamp(TensorMath.project(residual,width:clean.width,height:clean.height),5))}
        let stage=balanced ? "balanced_continue":"masked_compensation"
        let alignmentCadence=try cadence?.newStage(stage,width:clean.width,height:clean.height)
        for step in 0...NativeSchedule.steps {
            try tick(stage,step);let image=try budget.image(d)
            guard budget.fits(image) else {throw NativeError.invalid("Regional image budget exceeded")}
            let (scores,gradients)=try proxy.evaluate(image.tensor(),realign:true,needsGradient:step<NativeSchedule.steps,updateStep:step,stage:stage,alignmentCadence:alignmentCadence)
            let quality=NativeQuality.artifact(clean,image,flat:budget.flat),eligible=quality<=budget.artifactCap+1e-12,r=rank(scores,upper)
            if eligible && less(r,bestRank){best=image;bestRank=r}
            rows.append(OptimizationRow(stage:stage,step:step,scores:scores,mse:NativeQuality.mse(clean,image),eligible:eligible))
            if step==NativeSchedule.steps {break};let rate:Float=step<NativeSchedule.halfSteps ? 1:0.1
            if balanced {
                let low=gradients.map {g in TensorMath.project(g.enumerated().map{$0.element*alpha[$0.offset%clean.plane]/255},width:clean.width,height:clean.height,rescale:false)}
                let g=try NativeViewGradientPolicy.mean(low)
                var proposal=TensorMath.project(g.map{-TensorMath.sign($0)},width:clean.width,height:clean.height,rescale:false)
                let peak=max(proposal.map{abs($0)}.max() ?? 0,1e-12);for i in proposal.indices {proposal[i]*=rate/peak}
                let correction=min(Float(1),rate/max(proposal.map{abs($0)}.max() ?? 0,1e-12));for i in proposal.indices {proposal[i]*=correction}
                d=try budget.project(zip(d,proposal).map{$0+$1})
            } else {
                let average=try NativeViewGradientPolicy.mean(gradients),g=TensorMath.project(average.enumerated().map{$0.element*alpha[$0.offset%clean.plane]},width:clean.width,height:clean.height)
                d=try budget.project(TensorMath.clamp(TensorMath.project(TensorMath.clamp(zip(d,g).map{$0-rate*TensorMath.sign($1)},5),width:clean.width,height:clean.height),5))
            }
        }
        guard budget.fits(best),NativeQuality.artifact(clean,best,flat:budget.flat)<=budget.artifactCap+1e-12 else {throw NativeError.invalid("Final quality check failed")}
        for p in 0..<clean.plane {for c in 0..<3 {let i=3*p+c;guard abs(Int(best.bytes[i])-Int(clean.bytes[i]))<=5,alpha[p] != 0 || best.bytes[i]==clean.bytes[i] else {throw NativeError.invalid("Stored pixel bounds failed")}}}
        return best
    }
}
