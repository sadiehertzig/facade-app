import Foundation

struct ViewGradientRow:Codable {
    let stage:String,step:Int,scoreOrder:[String],gradientOrder:[String],meanDenominator:Int
    let gradientCount:Int,coefficients:[Int]
}
enum NativeViewGradientPolicy {
    static var arm="full_views"
    static var rows=[ViewGradientRow]()
    static func includes(_ condition:Int,step:Int,needsGradient:Bool)->Bool {
        guard needsGradient else {return false}
        return arm=="full_views" || condition==0 || condition==(step%2==0 ? 1:2)
    }
    static func record(stage:String,step:Int,scoreOrder:[String],gradientOrder:[String],gradients:[[Float]]) throws {
        let expected=(0..<3).filter{includes($0,step:step,needsGradient:step<NativeSchedule.steps)}
        let keys=expected.flatMap{c in (0..<2).map{"\(c):\($0)"}}
        guard scoreOrder==["0:0","0:1","1:0","1:1","2:0","2:1"],
              gradientOrder==keys,gradients.count==keys.count else {
            throw NativeError.invalid("View gradient policy lost scores or selected gradients")
        }
        let coefficients=keys.isEmpty ? []:(arm=="reweighted_views" ? [1,1,2,2]:Array(repeating:1,count:keys.count))
        let divisor=keys.isEmpty ? 0:(arm=="reweighted_views" ? 6:keys.count)
        rows.append(ViewGradientRow(stage:stage,step:step,scoreOrder:scoreOrder,
                                    gradientOrder:gradientOrder,meanDenominator:divisor,
                                    gradientCount:keys.count,coefficients:coefficients))
    }
    static func mean(_ gradients:[[Float]]) throws ->[Float] {
        // Preserve the original F/A reducer, including its Float order and error behavior.
        if arm != "reweighted_views" {return TensorMath.mean(gradients)}
        let token=Profile.begin("tensor.mean");defer{Profile.end("tensor.mean",token)}
        guard gradients.count==4,let first=gradients.first,!first.isEmpty,
              gradients.allSatisfy({$0.count==first.count}) else {
            throw NativeError.invalid("Reweighted gradient count or shape changed")
        }
        var out=[Float](repeating:0,count:first.count)
        // Frozen order: start at Float zero; d0, d1, 2*e0, 2*e1; one final division.
        // Finite/overflow checks are part of this R reducer's measured error handling.
        for j in 0..<4 {
            let coefficient:Float=j<2 ? 1:2
            for i in out.indices {
                let value=gradients[j][i]
                guard value.isFinite else {throw NativeError.invalid("Nonfinite reweighted gradient")}
                let term=coefficient*value
                let sum=out[i]+term
                guard term.isFinite,sum.isFinite else {throw NativeError.invalid("Reweighted Float overflow")}
                out[i]=sum
            }
        }
        for i in out.indices {out[i]/=Float(6)}
        return out
    }
}
