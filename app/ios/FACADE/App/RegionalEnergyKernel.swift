// RegionalEnergyKernel.swift
//
// Measures how much a candidate image differs from the clean photo, split by
// region: the whole image, the subject (person), the background, and a band
// just outside the subject.
//
// Where it sits: MaskBudget (NativeOptimizer.swift) builds one of these per
// stage and calls `energies` many times to check that a candidate stays within
// the pixel-change budget set by the baseline image.
//
// Key ideas:
// 1. "Energy" here is the mean squared difference of 8-bit channel values.
// 2. The masks are fixed for the life of the kernel, so their pixel counts are
//    computed once in `init` instead of on every call.
// 3. All sums are exact 64-bit integers; they are converted to Double only at
//    the end, so the result is identical to the older Double-accumulating
//    version but faster.
import Foundation

/// Owns the masks together with their pixel counts; a kernel cannot be reused
/// with different masks.
struct RegionalEnergyKernel {
    /// The clean image and two per-pixel masks (nonzero = inside the region).
    private let clean:RGBImage,subject:[UInt8],near:[UInt8]
    /// Channel counts per region (3 channels per pixel), used as divisors.
    private let totalCount:Int,subjectCount:Int,backgroundCount:Int,nearCount:Int
    /// The four divisors in the same order as `energies` returns its values.
    var counts:[Int] {[totalCount,subjectCount,backgroundCount,nearCount]}

    /// Returns nil if the sizes do not match, so callers fall back to the slower
    /// path rather than silently computing nonsense.
    init?(clean:RGBImage,subject:[UInt8],near:[UInt8]) {
        // Same size limits as RGBImage.
        guard clean.width>0,clean.height>0,clean.width<=2048,clean.height<=2048 else {return nil}
        let n=clean.width*clean.height
        // Each mask has one byte per pixel; the image has three.
        guard clean.bytes.count==3*n,subject.count==n,near.count==n else {return nil}
        // Count mask pixels once.
        var subjectPixels=0,nearPixels=0
        for p in 0..<n {
            if subject[p] != 0 {subjectPixels+=1}
            if near[p] != 0 {nearPixels+=1}
        }
        self.clean=clean;self.subject=subject;self.near=near
        // Three channels per pixel.
        totalCount=3*n;subjectCount=3*subjectPixels
        backgroundCount=3*(n-subjectPixels);nearCount=3*nearPixels
    }

    /// Mean squared difference per region: [total, subject, background, near].
    /// Returns nil when the shapes do not match, which selects the unchanged
    /// scalar fallback in MaskBudget, including its existing error behaviour.
    func energies(_ image:RGBImage)->[Double]? {
        // Every array must describe the same width x height.
        guard image.width==clean.width,image.height==clean.height,
              clean.bytes.count==totalCount,image.bytes.count==totalCount,
              subject.count==clean.plane,near.count==clean.plane else {return nil}
        let profileToken=Profile.begin("quality.regional.energies");defer{Profile.end("quality.regional.energies",profileToken)}

        // Overflow check: the largest possible sum is 3*2048*2048*255*255 =
        // 818,203,852,800, which is below both 2^53 and Int64.max. So the Double
        // squares of the original version and every partial sum here are exact
        // integers, and this gives the same numbers.
        var total:Int64=0,foreground:Int64=0,background:Int64=0,nearby:Int64=0
        // Per pixel: squared difference summed over R, G and B.
        for p in 0..<clean.plane {
            let i=3*p
            let r=Int64(image.bytes[i])-Int64(clean.bytes[i])
            let g=Int64(image.bytes[i+1])-Int64(clean.bytes[i+1])
            let b=Int64(image.bytes[i+2])-Int64(clean.bytes[i+2])
            let value=r*r+g*g+b*b
            total+=value
            // A pixel is either subject or background, never both.
            if subject[p] != 0 {foreground+=value}else{background+=value}
            if near[p] != 0 {nearby+=value}
        }
        // Divide by channel counts; an empty region reports 0 instead of dividing by 0.
        return [
            Double(total)/Double(totalCount),
            subjectCount==0 ? 0:Double(foreground)/Double(subjectCount),
            backgroundCount==0 ? 0:Double(background)/Double(backgroundCount),
            nearCount==0 ? 0:Double(nearby)/Double(nearCount)
        ]
    }
}
