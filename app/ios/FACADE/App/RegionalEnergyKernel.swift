import Foundation

// Own the masks with their counts; a cache cannot be reused with different masks.
struct RegionalEnergyKernel {
    private let clean:RGBImage,subject:[UInt8],near:[UInt8]
    private let totalCount:Int,subjectCount:Int,backgroundCount:Int,nearCount:Int
    var counts:[Int] {[totalCount,subjectCount,backgroundCount,nearCount]}

    init?(clean:RGBImage,subject:[UInt8],near:[UInt8]) {
        guard clean.width>0,clean.height>0,clean.width<=2048,clean.height<=2048 else {return nil}
        let n=clean.width*clean.height
        guard clean.bytes.count==3*n,subject.count==n,near.count==n else {return nil}
        var subjectPixels=0,nearPixels=0
        for p in 0..<n {
            if subject[p] != 0 {subjectPixels+=1}
            if near[p] != 0 {nearPixels+=1}
        }
        self.clean=clean;self.subject=subject;self.near=near
        totalCount=3*n;subjectCount=3*subjectPixels
        backgroundCount=3*(n-subjectPixels);nearCount=3*nearPixels
    }

    // nil selects the unchanged scalar oracle, including its existing errors.
    func energies(_ image:RGBImage)->[Double]? {
        guard image.width==clean.width,image.height==clean.height,
              clean.bytes.count==totalCount,image.bytes.count==totalCount,
              subject.count==clean.plane,near.count==clean.plane else {return nil}
        let profileToken=Profile.begin("quality.regional.energies");defer{Profile.end("quality.regional.energies",profileToken)}

        // Sum <= 3*2048*2048*255*255 = 818203852800 < 2^53 and Int64.max.
        // The original Double squares and every partial sum are exact integers.
        var total:Int64=0,foreground:Int64=0,background:Int64=0,nearby:Int64=0
        for p in 0..<clean.plane {
            let i=3*p
            let r=Int64(image.bytes[i])-Int64(clean.bytes[i])
            let g=Int64(image.bytes[i+1])-Int64(clean.bytes[i+1])
            let b=Int64(image.bytes[i+2])-Int64(clean.bytes[i+2])
            let value=r*r+g*g+b*b
            total+=value
            if subject[p] != 0 {foreground+=value}else{background+=value}
            if near[p] != 0 {nearby+=value}
        }
        return [
            Double(total)/Double(totalCount),
            subjectCount==0 ? 0:Double(foreground)/Double(subjectCount),
            backgroundCount==0 ? 0:Double(background)/Double(backgroundCount),
            nearCount==0 ? 0:Double(nearby)/Double(nearCount)
        ]
    }
}
