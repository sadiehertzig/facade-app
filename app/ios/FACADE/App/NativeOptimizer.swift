// NativeOptimizer.swift
//
// The optimisation loops of the FACADE engine at full photo size, plus the
// quality measures and budgets that keep the change hard to see.
//
// Where it sits: NativePipeline.run creates one NativeOptimizer and calls
// `native` three times (stages "native_eot8", "native_realign8", "energy80")
// and then `masked` twice ("masked_compensation", "balanced_continue"). Each
// stage starts from the previous stage's best image.
//
// Key ideas:
// 1. Each step scores the candidate with NativeAlignedProxies (six scores:
//    three upload conditions x two face-recognition networks), averages the
//    gradients, and moves every pixel by a fixed small amount against the sign
//    of the gradient. Lower scores mean the face matches the original less.
// 2. Every change is limited to 5/255 per channel (5 on the 0-255 scale),
//    smoothed by the 8x8 low-frequency filter, and in the later stages scaled
//    down until it fits a pixel-change budget. The step size is 1/255 in the
//    coarse phase and 0.1/255 in the fine phase.
// 3. A stage does not simply keep its last step. It keeps the best candidate
//    that met the stage's rules, and the score limits are set from the parent
//    image so a stage cannot give back much of the progress already made.
// 4. All arithmetic is written to give the same bits on every run, which is
//    why some loops look more careful about order and rounding than needed.
import Foundation

/// One trace record per stage step: the six scores, the mean squared pixel
/// change from the clean image, and whether the candidate met the stage's rules.
struct OptimizationRow:Codable {
    let stage:String,step:Int,scores:[Float],mse:Double,eligible:Bool
}
/// Image-quality measures used as budgets.
enum NativeQuality {
    /// Mean squared difference over all channel values, on the 0-255 scale.
    static func mse(_ clean:RGBImage,_ candidate:RGBImage)->Double {
        let profileToken=Profile.begin("quality.mse");defer{Profile.end("quality.mse",profileToken)}

        var sum=0.0;for i in clean.bytes.indices {let d=Double(Int(candidate.bytes[i])-Int(clean.bytes[i]));sum+=d*d};return sum/Double(clean.bytes.count)
    }
    /// A per-pixel "flatness" weight from 0 to 1: close to 1 where the image is
    /// smooth (a change there is easy to see), close to 0 where it is textured.
    /// It is 0.001 / (local variance + 0.001), with the variance of the gray
    /// value taken over a 9x9 window.
    static func flatWeights(_ image:RGBImage)->[Float] {
        let profileToken=Profile.begin("quality.flat.weights");defer{Profile.end("quality.flat.weights",profileToken)}

        // Integral images (running sums of gray and gray^2) give any window's sum
        // with four lookups. Index (y+1)*(w+1)+(x+1) leaves a zero row and column.
        let w=image.width,h=image.height,x=image.tensor();var integral=[Double](repeating:0,count:(w+1)*(h+1)),squared=integral
        for y in 0..<h {var row=0.0,row2=0.0;for xx in 0..<w {let p=y*w+xx,v=Double((x[p]+x[image.plane+p]+x[2*image.plane+p])/3);row+=v;row2+=v*v;integral[(y+1)*(w+1)+xx+1]=integral[y*(w+1)+xx+1]+row;squared[(y+1)*(w+1)+xx+1]=squared[y*(w+1)+xx+1]+row2}}
        var out=[Float](repeating:0,count:image.plane)
        // Window x-4...x+4, y-4...y+4, cut off at the image border. The average always
        // divides by 81, so windows at the border come out slightly low.
        for y in 0..<h {for x in 0..<w {let l=max(0,x-4),r=min(w,x+5),t=max(0,y-4),b=min(h,y+5)
            func average(_ a:[Double])->Double {(a[b*(w+1)+r]-a[t*(w+1)+r]-a[b*(w+1)+l]+a[t*(w+1)+l])/81}
            let mean=average(integral),variance=max(0,average(squared)-mean*mean);out[y*w+x]=Float(0.001/(variance+0.001))
        }};return out
    }
    /// A single "how visible is the change" number (lower is better). It adds:
    /// - the mean squared change,
    /// - 0.5 x the squared jump in brightness change across 8x8 block edges
    ///   (the low-frequency filter works per block, so edges could show),
    /// - 0.25 x the colour part of the change (how far R, G and B move apart),
    /// - 0.5 x the squared change weighted by flatness.
    /// Values are on the 0...1 scale, not 0-255.
    static func artifact(_ clean:RGBImage,_ candidate:RGBImage,flat:[Float])->Double {
        let profileToken=Profile.begin("quality.artifact");defer{Profile.end("quality.artifact",profileToken)}

        let a=clean.tensor(),b=candidate.tensor(),n=clean.plane,w=clean.width,h=clean.height
        var y=[Float](repeating:0,count:n);var square=0.0,chroma=0.0,weighted=0.0
        for p in 0..<n {let r=b[p]-a[p],g=b[n+p]-a[n+p],blue=b[2*n+p]-a[2*n+p],mean=(r+g+blue)/3
            // Change in brightness (Rec. 709 luma weights).
            y[p]=0.2126*r+0.7152*g+0.0722*blue
            for d in [r,g,blue] {square+=Double(d*d);chroma+=Double((d-mean)*(d-mean));weighted+=Double(d*d*flat[p])}
        }
        var boundary=0.0,count=0
        // Vertical block edges: compare columns 7|8, 15|16, ...
        for yy in 0..<h {for xx in stride(from:8,to:w,by:8) {let d=y[yy*w+xx]-y[yy*w+xx-1];boundary+=Double(d*d);count+=1}}
        // Horizontal block edges, the same way.
        for yy in stride(from:8,to:h,by:8) {for xx in 0..<w {let d=y[yy*w+xx]-y[(yy-1)*w+xx];boundary+=Double(d*d);count+=1}}
        return square/Double(3*n)+0.5*boundary/Double(max(count,1))+0.25*chroma/Double(3*n)+0.5*weighted/Double(3*n)
    }
}
/// The rules for the two masked stages. The change is written as a "latent"
/// x (channel first, 0-255 units) and the image is clean + alpha * x, where
/// alpha is the cleanup weight from TightCleanup (1 on the person, fading to 0
/// outside). A candidate fits when, in every region (whole image, subject,
/// background, near band), its mean squared change is no larger than the
/// baseline image's.
final class MaskBudget {
    let clean:RGBImage,alpha:[Float],subject:[UInt8],near:[UInt8],flat:[Float],caps:[Double],artifactCap:Double
    /// true = use the faster RegionalEnergyKernel; false = the plain loop. Both
    /// give the same numbers.
    let regionalEnergyEnabled:Bool
    private let regionalEnergyKernel:RegionalEnergyKernel?
    /// The baseline (the cleaned-up image) sets the caps: its regional energies
    /// and its artifact score.
    init(clean:RGBImage,baseline:RGBImage,alpha:[Float],subject:[UInt8],near:[UInt8],regionalEnergyEnabled:Bool=true) {
        self.clean=clean;self.alpha=alpha;self.subject=subject;self.near=near;flat=NativeQuality.flatWeights(clean)
        self.regionalEnergyEnabled=regionalEnergyEnabled
        let kernel=regionalEnergyEnabled ? RegionalEnergyKernel(clean:clean,subject:subject,near:near):nil
        regionalEnergyKernel=kernel
        caps=kernel?.energies(baseline) ?? Self.energies(clean,baseline,subject,near);artifactCap=NativeQuality.artifact(clean,baseline,flat:flat)
    }
    /// The plain-loop version of the regional energies: [total, subject,
    /// background, near], each a mean squared change. A region with no pixels gives 0.
    static func energies(_ clean:RGBImage,_ image:RGBImage,_ subject:[UInt8],_ near:[UInt8])->[Double] {
        let profileToken=Profile.begin("quality.regional.energies");defer{Profile.end("quality.regional.energies",profileToken)}

        var sums=[Double](repeating:0,count:4),counts=[Int](repeating:0,count:4)
        for p in 0..<clean.plane {for c in 0..<3 {let d=Double(Int(image.bytes[3*p+c])-Int(clean.bytes[3*p+c])),v=d*d;sums[0]+=v;counts[0]+=1
            let region=subject[p] != 0 ? 1:2;sums[region]+=v;counts[region]+=1;if near[p] != 0 {sums[3]+=v;counts[3]+=1}
        }};return zip(sums,counts).map{$1==0 ? 0:$0/Double($1)}
    }
    /// Regional energies for any image pair, kernel first and loop as fallback.
    static func candidateEnergies(_ clean:RGBImage,_ image:RGBImage,_ subject:[UInt8],_ near:[UInt8])->[Double] {
        RegionalEnergyKernel(clean:clean,subject:subject,near:near)?.energies(image) ?? energies(clean,image,subject,near)
    }
    /// Regional energies of `image` against this budget's clean image.
    func regionalEnergies(_ image:RGBImage)->[Double] {
        regionalEnergyKernel?.energies(image) ?? Self.energies(clean,image,subject,near)
    }
    /// True when no regional energy exceeds its cap (1e-12 absorbs rounding noise).
    func fits(_ image:RGBImage)->Bool {zip(regionalEnergies(image),caps).allSatisfy{$0<=$1+1e-12}}
    /// Builds the 8-bit image clean + alpha * latent * scale, rounding half up.
    func image(_ latent:[Float],scale:Double=1) throws ->RGBImage {
        let profileToken=Profile.begin("quality.mask.image");defer{Profile.end("quality.mask.image",profileToken)}

        var bytes=clean.bytes;for p in 0..<clean.plane {for c in 0..<3 {let v=Double(clean.bytes[3*p+c])+Double(alpha[p])*Double(latent[c*clean.plane+p])*scale;bytes[3*p+c]=UInt8(min(255,max(0,floor(v+0.5))))}}
        return try RGBImage(width:clean.width,height:clean.height,bytes:bytes)
    }
    /// Makes a latent obey the rules: each value within +/-5 and within what the
    /// 0-255 range allows, then, if the image still does not fit the budget,
    /// shrunk by the largest factor that fits.
    func project(_ input:[Float]) throws ->[Float] {
        let profileToken=Profile.begin("quality.mask.project");defer{Profile.end("quality.mask.project",profileToken)}

        var x=input
        for p in 0..<clean.plane {for c in 0..<3 {let i=c*clean.plane+p,v=Float(clean.bytes[3*p+c]);x[i]=max(max(-5,-v),min(min(5,255-v),x[i]))}}
        // 20 halvings find the scale to about one millionth.
        if try fits(image(x)) {return x};var lo=0.0,hi=1.0
        for _ in 0..<20 {let mid=(lo+hi)/2;if try fits(image(x,scale:mid)){lo=mid}else{hi=mid}}
        // Multiplying in Float can round slightly differently from the Double scale
        // used in the search, so nudge the scale down until the stored image fits.
        for _ in 0..<10000 {let result=x.map{$0*Float(lo)};if try fits(image(result)){return result};lo*=0.999999}
        throw NativeError.invalid("Quality budget did not converge")
    }
    /// The reverse of `image`: finds a latent that reproduces `image` exactly,
    /// so the second masked stage can continue from the first one's result.
    func recover(_ image:RGBImage) throws ->[Float] {
        let profileToken=Profile.begin("quality.mask.recover");defer{Profile.end("quality.mask.recover",profileToken)}

        // First try: latent = change / alpha. Pixels with alpha 0 cannot be changed
        // and stay 0.
        var x=[Float](repeating:0,count:clean.bytes.count)
        for p in 0..<clean.plane where alpha[p]>0 {for c in 0..<3 {let i=3*p+c,d=Double(Int(image.bytes[i])-Int(clean.bytes[i])),v=Double(clean.bytes[i]);x[c*clean.plane+p]=Float(max(max(-5,-v),min(min(5,255-v),d/Double(alpha[p]))))}}
        if try self.image(x).bytes==image.bytes{return x}
        // If rounding broke the round trip, use the middle of the range of latents
        // that round to the same byte (change +/- 0.5, divided by alpha).
        for p in 0..<clean.plane where alpha[p]>0 {for c in 0..<3 {let i=3*p+c,d=Double(Int(image.bytes[i])-Int(clean.bytes[i])),v=Double(clean.bytes[i]),a=Double(alpha[p]);let low=max((d-0.499999)/a,max(-5,-v)),high=min((d+0.499999)/a,min(5,255-v));x[c*clean.plane+p]=Float((low+high)/2)}}
        guard try self.image(x).bytes==image.bytes else {throw NativeError.invalid("Warm start changed stored pixels")};return x
    }
}
/// Runs the stages and keeps the trace of every step.
final class NativeOptimizer {
    /// One row per step of every stage, in order (checked by LivePhotoEngine).
    var rows=[OptimizationRow]()
    /// Called with (stage, step) for the progress display.
    let progress:(String,Int)->Void
    /// Face re-detection cadence for the stages that re-detect (nil = every step).
    let cadence:NativeCadenceRun?
    init(progress:@escaping(String,Int)->Void,cadence:NativeCadenceRun?=nil){self.progress=progress;self.cadence=cadence}
    /// Start of every step: stop if the user cancelled, pass a time/heat
    /// checkpoint during a live run, then report progress.
    func tick(_ stage:String,_ step:Int) throws {try Task.checkCancellation();if Profile.deadline != nil {try Profile.boundary("\(stage).\(step)")};progress(stage,step)}
    /// Plain average.
    func mean(_ scores:[Float])->Float {scores.reduce(0,+)/Float(scores.count)}
    /// Upper limits for a stage's scores, from the parent image's scores s:
    /// 1 - 0.95 * (1 - s). A candidate may give back at most 5% of each score's
    /// distance below 1. Example: parent 0.30 gives a limit of 0.335.
    func limits(_ scores:[Float])->[Float] {scores.map{1-0.95*max(0,1-$0)}}
    /// Sort key for candidates: first the largest amount by which any score is
    /// over its limit (0 when all are within), then the mean score.
    func rank(_ scores:[Float],_ limits:[Float])->(Float,Float) {(zip(scores,limits).map{max(0,$0-$1-1e-6)}.max() ?? 0,mean(scores))}
    /// Compares two ranks: lower worst-overshoot first, then lower mean.
    func less(_ a:(Float,Float),_ b:(Float,Float))->Bool {a.0<b.0 || (a.0==b.0 && a.1<b.1)}
    /// Limits a change d (0...1 units) to +/-5/255 and the valid range, then, if
    /// the stored image's mean squared change is above `cap`, scales the whole
    /// change down (keeping its direction) to the largest factor that fits.
    func radial(_ d:[Float],clean:RGBImage,cap:Double) throws ->[Float] {
        let profileToken=Profile.begin("quality.energy.project");defer{Profile.end("quality.energy.project",profileToken)}

        guard d.count == clean.bytes.count else {throw NativeError.invalid("Non-finite optimizer state")}
        let c=clean.tensor();var bounded=d
        for i in d.indices {bounded[i]=max(max(-5/255,-c[i]),min(min(5/255,1-c[i]),d[i]))}
        // FUSED_ENERGY_ROUTER_BEGIN: one pass replaces map + stored + mse. Same
        // clamp, same floorf(v*255+0.5) quantization, same interleaved Double
        // accumulation order, so the returned scale is bit-identical.
        let cleanBytes=clean.bytes,plane=clean.plane,total=cleanBytes.count
        /// Mean squared change of the stored image clean + scale * d.
        func mse(_ scale:Float)throws->Double {
            var sum=0.0
            try c.withUnsafeBufferPointer {cp in try bounded.withUnsafeBufferPointer {bp in
                try cleanBytes.withUnsafeBufferPointer {kp in
                    for i in 0..<total {
                        // i walks the packed bytes (pixel i/3, channel i%3); j is the same value's
                        // position in the channel-first tensors.
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
        // Binary search with 20 halvings; `lo` always fits.
        if try mse(1)<=cap+1e-12 {return bounded};var lo:Float=0,hi:Float=1
        for _ in 0..<20 {let mid=(lo+hi)/2;if try mse(mid)<=cap+1e-12 {lo=mid}else{hi=mid}}
        return bounded.map{$0*lo}
    }
    /// The three full-size stages. They share the step schedule in NativeSchedule
    /// (10 coarse + 6 fine steps in the app).
    ///
    /// - "native_eot8": fixed alignments from the clean photo and the faster float
    ///   crop. Every step is eligible; the best mean score wins. ("EOT" means the
    ///   change is optimised over several transformations: the upload conditions.)
    /// - "native_realign8": re-detects the face. A step is eligible only if every
    ///   score stays within the parent's limits and its mean squared change is at
    ///   most 1.05^2 = 1.1025 times the parent's.
    /// - "energy80": the change must use at most 80% of the parent's mean squared
    ///   change. Every step is forced into that cap, and the winner is picked by
    ///   `rank` against the parent's score limits.
    func native(_ proxy:NativeAlignedProxies,initial:RGBImage,stage:String) throws ->RGBImage {
        let profileToken=Profile.begin("stage."+stage);defer{Profile.end("stage."+stage,profileToken)}

        let clean=proxy.clean,c=clean.tensor(),parentScores=try proxy.scores(initial),upper=limits(parentScores),parentMSE=NativeQuality.mse(clean,initial)
        // Cap on the mean squared change, relative to the starting image. It is
        // enforced only in native_realign8 and energy80.
        let energy=stage=="energy80",realign=stage != "native_eot8",cap=energy ? 0.8*parentMSE:1.05*1.05*parentMSE
        let alignmentCadence=try realign ? cadence?.newStage(stage,width:clean.width,height:clean.height):nil
        let lim:Float=5/255
        // Starting change = initial - clean: clip, smooth, clip again (smoothing can
        // push a value slightly past the limit).
        var d=TensorMath.clamp(TensorMath.project(TensorMath.clamp(zip(initial.tensor(),c).map{$0-$1},lim),width:clean.width,height:clean.height),lim)
        if energy {d=try radial(d,clean:clean,cap:cap)}
        // Except in energy80, the starting image itself counts as a candidate.
        var best:RGBImage?=energy ? nil:initial,bestRank=energy ? (Float.infinity,Float.infinity):(Float(0),mean(parentScores))
        for step in 0...NativeSchedule.steps {
            try tick(stage,step);let x=zip(c,d).map{min(1,max(0,$0+$1))},image=try clean.stored(x)
            let (scores,gradients)=try proxy.evaluate(x,realign:realign,needsGradient:step<NativeSchedule.steps,updateStep:step,stage:stage,alignmentCadence:alignmentCadence),mse=NativeQuality.mse(clean,image)
            // Which candidates may be kept (see the list above).
            let fits=stage=="native_eot8" || energy || (zip(scores,upper).allSatisfy{$0<=$1+1e-6} && mse<=cap+1e-12)
            let r=energy ? rank(scores,upper):(0,mean(scores))
            if fits && (best==nil || less(r,bestRank)){best=image;bestRank=r}
            if energy && mse>cap+1e-12 {throw NativeError.invalid("Stored energy budget exceeded")}
            rows.append(OptimizationRow(stage:stage,step:step,scores:scores,mse:mse,eligible:fits))
            if step==NativeSchedule.steps {break}
            // Signed-gradient step: average and smooth the gradients, then move each value
            // one step against the sign, clip, smooth, clip.
            let g=TensorMath.project(try NativeViewGradientPolicy.mean(gradients),width:clean.width,height:clean.height),rate:Float=(step<NativeSchedule.halfSteps ? 1:0.1)/255
            d=TensorMath.clamp(TensorMath.project(TensorMath.clamp(zip(d,g).map{$0-rate*TensorMath.sign($1)},lim),width:clean.width,height:clean.height),lim)
            if energy {d=try radial(d,clean:clean,cap:cap)}
        };guard let result=best else {throw NativeError.invalid("No native candidate")};return result
    }
    /// The two masked stages, run after cleanup has restored the background.
    /// The change now lives in MaskBudget's latent, so it stays on the person and
    /// in the band around them, and each region keeps no more change than the
    /// cleaned-up baseline had. Candidates must also not be more visible than the
    /// baseline (artifact score).
    ///
    /// - balanced = false ("masked_compensation"): starts from energy80's change
    ///   and tries to win back the strength cleanup removed.
    /// - balanced = true ("balanced_continue"): continues from the first masked
    ///   stage with a normalised step, described below.
    /// Score limits come from energy80's scores in both stages.
    func masked(_ proxy:NativeAlignedProxies,energy80:RGBImage,initial:RGBImage,baseline:RGBImage,alpha:[Float],subject:[UInt8],near:[UInt8],balanced:Bool) throws ->RGBImage {
        let profileToken=Profile.begin(balanced ? "stage.balanced_continue":"stage.masked_compensation");defer{Profile.end(balanced ? "stage.balanced_continue":"stage.masked_compensation",profileToken)}

        let budget=MaskBudget(clean:proxy.clean,baseline:baseline,alpha:alpha,subject:subject,near:near),clean=proxy.clean
        // The starting image's own rank is the bar a step has to beat.
        let target=try proxy.scores(energy80),upper=limits(target),original=try proxy.scores(initial)
        var best=initial,bestRank=rank(original,upper),d:[Float]
        if balanced {d=try budget.recover(initial)}
        // Starting latent: energy80's change in 0-255 units, channel first
        // (i%plane is the pixel, i/plane the channel), smoothed, clipped to +/-5 and
        // fitted to the budget.
        else {let residual=(0..<clean.bytes.count).map {i -> Float in let p=i%clean.plane,c=i/clean.plane;return Float(Int(energy80.bytes[3*p+c])-Int(clean.bytes[3*p+c]))};d=try budget.project(TensorMath.clamp(TensorMath.project(residual,width:clean.width,height:clean.height),5))}
        let stage=balanced ? "balanced_continue":"masked_compensation"
        let alignmentCadence=try cadence?.newStage(stage,width:clean.width,height:clean.height)
        for step in 0...NativeSchedule.steps {
            try tick(stage,step);let image=try budget.image(d)
            guard budget.fits(image) else {throw NativeError.invalid("Regional image budget exceeded")}
            let (scores,gradients)=try proxy.evaluate(image.tensor(),realign:true,needsGradient:step<NativeSchedule.steps,updateStep:step,stage:stage,alignmentCadence:alignmentCadence)
            // Eligible only if no more visible than the baseline.
            let quality=NativeQuality.artifact(clean,image,flat:budget.flat),eligible=quality<=budget.artifactCap+1e-12,r=rank(scores,upper)
            if eligible && less(r,bestRank){best=image;bestRank=r}
            rows.append(OptimizationRow(stage:stage,step:step,scores:scores,mse:NativeQuality.mse(clean,image),eligible:eligible))
            if step==NativeSchedule.steps {break};let rate:Float=step<NativeSchedule.halfSteps ? 1:0.1
            if balanced {
                // Chain rule into latent units: image = clean + alpha * x / 255 in 0...1 units,
                // so each gradient is multiplied by alpha / 255. Smoothed without rescale.
                let low=gradients.map {g in TensorMath.project(g.enumerated().map{$0.element*alpha[$0.offset%clean.plane]/255},width:clean.width,height:clean.height,rescale:false)}
                let g=try NativeViewGradientPolicy.mean(low)
                // Proposed step: the smoothed sign, scaled so its largest value is `rate`.
                var proposal=TensorMath.project(g.map{-TensorMath.sign($0)},width:clean.width,height:clean.height,rescale:false)
                let peak=max(proposal.map{abs($0)}.max() ?? 0,1e-12);for i in proposal.indices {proposal[i]*=rate/peak}
                // Second guard on the largest value; after the scaling above it is
                // effectively 1.
                let correction=min(Float(1),rate/max(proposal.map{abs($0)}.max() ?? 0,1e-12));for i in proposal.indices {proposal[i]*=correction}
                d=try budget.project(zip(d,proposal).map{$0+$1})
            } else {
                // First masked stage: weight the gradient by alpha (only the sign is used
                // next, so the 1/255 factor does not matter), then the usual signed step.
                let average=try NativeViewGradientPolicy.mean(gradients),g=TensorMath.project(average.enumerated().map{$0.element*alpha[$0.offset%clean.plane]},width:clean.width,height:clean.height)
                d=try budget.project(TensorMath.clamp(TensorMath.project(TensorMath.clamp(zip(d,g).map{$0-rate*TensorMath.sign($1)},5),width:clean.width,height:clean.height),5))
            }
        }
        // Final checks on the chosen image: within the regional budget, no more
        // visible than the baseline, every channel within 5 of the clean photo, and
        // pixels with alpha 0 exactly unchanged.
        guard budget.fits(best),NativeQuality.artifact(clean,best,flat:budget.flat)<=budget.artifactCap+1e-12 else {throw NativeError.invalid("Final quality check failed")}
        for p in 0..<clean.plane {for c in 0..<3 {let i=3*p+c;guard abs(Int(best.bytes[i])-Int(clean.bytes[i]))<=5,alpha[p] != 0 || best.bytes[i]==clean.bytes[i] else {throw NativeError.invalid("Stored pixel bounds failed")}}}
        return best
    }
}
