// NativeViewGradientPolicy.swift
//
// Decides which "views" of the photo get a gradient at each optimisation step,
// and how those gradients are averaged.
//
// Where it sits: NativeAlignedProxies (NativeVision.swift) asks `includes` to
// know whether to compute a gradient for a given upload condition, and
// NativeOptimizer calls `mean` to combine the gradients into one update.
//
// Key ideas:
// 1. A "view" is one of three upload conditions (0: untouched, 1: JPEG,
//    2: JPEG plus half-size) seen by one of two recognition models, so there
//    are six scores per step. Gradients are expensive, so not all six are
//    computed every step.
// 2. The "reweighted_views" arm (what the app uses) always computes the two
//    gradients for condition 0, and alternates conditions 1 and 2 on even and
//    odd steps. The alternating pair is counted twice in the average, so each
//    condition still has the same total weight over any two steps.
// 3. Every step records a row describing what it did; LivePhotoEngine checks
//    the row count after a run to confirm the policy was applied throughout.
import Foundation

/// One record per stage and step: which views were scored, which got a
/// gradient, and the coefficients used in the average.
struct ViewGradientRow:Codable {
    let stage:String,step:Int,scoreOrder:[String],gradientOrder:[String],meanDenominator:Int
    let gradientCount:Int,coefficients:[Int]
}
/// The policy itself. `arm` is process-wide state set by LivePhotoEngine.
enum NativeViewGradientPolicy {
    /// "full_views" computes every gradient; "reweighted_views" is the alternating
    /// scheme described in the file header.
    static var arm="full_views"
    /// Rows recorded during the current run (cleared by LivePhotoEngine).
    static var rows=[ViewGradientRow]()
    /// Should condition `condition` get a gradient at this step?
    static func includes(_ condition:Int,step:Int,needsGradient:Bool)->Bool {
        // The last step of a stage is score-only, so never a gradient there.
        guard needsGradient else {return false}
        // Condition 0 always; condition 1 on even steps, condition 2 on odd steps.
        return arm=="full_views" || condition==0 || condition==(step%2==0 ? 1:2)
    }
    /// Checks that the proxies produced exactly the expected score and gradient
    /// lists for this step, then records a row.
    static func record(stage:String,step:Int,scoreOrder:[String],gradientOrder:[String],gradients:[[Float]]) throws {
        // The conditions that should have gradients at this step.
        let expected=(0..<3).filter{includes($0,step:step,needsGradient:step<NativeSchedule.steps)}
        // Expected gradient keys, e.g. ["0:0", "0:1", "1:0", "1:1"] (condition:model).
        let keys=expected.flatMap{c in (0..<2).map{"\(c):\($0)"}}
        // All six scores must always be present and in this fixed order.
        guard scoreOrder==["0:0","0:1","1:0","1:1","2:0","2:1"],
              gradientOrder==keys,gradients.count==keys.count else {
            throw NativeError.invalid("View gradient policy lost scores or selected gradients")
        }
        // Weights 1,1 for condition 0 and 2,2 for the alternating condition.
        let coefficients=keys.isEmpty ? []:(arm=="reweighted_views" ? [1,1,2,2]:Array(repeating:1,count:keys.count))
        // The divisor 6 makes the average equal to a plain mean over six gradients.
        let divisor=keys.isEmpty ? 0:(arm=="reweighted_views" ? 6:keys.count)
        rows.append(ViewGradientRow(stage:stage,step:step,scoreOrder:scoreOrder,
                                    gradientOrder:gradientOrder,meanDenominator:divisor,
                                    gradientCount:keys.count,coefficients:coefficients))
    }
    /// Averages the gradients. Under "reweighted_views" the four gradients are
    /// combined as (d0 + d1 + 2*e0 + 2*e1) / 6, where d are the condition-0
    /// gradients and e the alternating condition's.
    static func mean(_ gradients:[[Float]]) throws ->[Float] {
        // Any other arm uses the original plain-mean reducer, including its Float
        // summation order and error behaviour.
        if arm != "reweighted_views" {return TensorMath.mean(gradients)}
        let token=Profile.begin("tensor.mean");defer{Profile.end("tensor.mean",token)}
        // Exactly four gradients of one shared length are required.
        guard gradients.count==4,let first=gradients.first,!first.isEmpty,
              gradients.allSatisfy({$0.count==first.count}) else {
            throw NativeError.invalid("Reweighted gradient count or shape changed")
        }
        var out=[Float](repeating:0,count:first.count)
        // The order is fixed so results are reproducible: start at Float zero;
        // add d0, d1, 2*e0, 2*e1; then one final division. The finite and overflow
        // checks are part of this reducer's defined error handling.
        for j in 0..<4 {
            // Coefficient 1 for the first two gradients, 2 for the last two.
            let coefficient:Float=j<2 ? 1:2
            for i in out.indices {
                let value=gradients[j][i]
                // A NaN or infinity here means a model produced garbage; stop rather than
                // let it spread into the image.
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
