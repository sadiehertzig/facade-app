// HelpResources.swift
//
// The catalogue of official help services the Get Help tab links to.
//
// Where it sits: HelpScreen.swift reads these cases to build its situation
// cards and buttons. Nothing else in the app opens external links.
//
// Key ideas:
// 1. Every destination is a fixed https address written in this file. The app
//    never appends a photo, an address, an account or a query parameter.
// 2. The wording (title, audience, detail) is user-visible and was checked
//    against each service's own eligibility guidance; keep it honest.
import Foundation

/// Fixed public destinations. No photo, address, account, or query parameter is
/// accepted by this catalogue or appended to an outgoing link.
public enum FACADEHelpResource: String, CaseIterable, Identifiable, Sendable {
    /// NCMEC's Take It Down: for images taken when the person was under 18.
    case takeItDown
    /// StopNCII.org: for adults depicted in intimate images.
    case stopNCII
    /// NCMEC CyberTipline: reporting threats and child sexual exploitation.
    case cyberTipline
    /// The FTC's page on the TAKE IT DOWN Act (a law, distinct from the service).
    case takeItDownAct
    /// The House of Representatives' official district lookup.
    case findRepresentative

    /// Stable identifier for SwiftUI lists and sheets.
    public var id: String { rawValue }

    /// Short name shown as the resource's heading.
    public var title: String {
        switch self {
        case .takeItDown: "Take It Down"
        case .stopNCII: "StopNCII.org"
        case .cyberTipline: "NCMEC CyberTipline"
        case .takeItDownAct: "FTC image-based abuse guidance"
        case .findRepresentative: "Find your representative"
        }
    }

    /// One line saying who each service is for.
    public var audience: String {
        switch self {
        case .takeItDown: "Under 18 when the intimate image was taken"
        case .stopNCII: "18 or older now and in the intimate image"
        case .cyberTipline: "Threats or suspected child sexual exploitation"
        case .takeItDownAct: "Your options under the TAKE IT DOWN Act"
        case .findRepresentative: "Official U.S. House district lookup"
        }
    }

    /// Longer explanation shown under the heading.
    public var detail: String {
        switch self {
        case .takeItDown:
            "NCMEC's free service can help with nude, partially nude, or sexually explicit images, including eligible AI images. You can use it even if you are an adult now. It creates a digital fingerprint on your device; the image stays there."
        case .stopNCII:
            "For adults depicted in intimate images, including eligible deepfakes. This free service creates a digital fingerprint on your device without uploading the image. Check its eligibility guidance."
        case .cyberTipline:
            "Report suspected online child sexual exploitation, including threats involving intimate images. You can seek help even if you do not have the image."
        case .takeItDownAct:
            "The TAKE IT DOWN Act is separate from NCMEC's Take It Down service. Covered platforms must remove qualifying intimate images and known identical copies within 48 hours of a valid request. Read the FTC's guidance on making a request and reporting a platform that does not comply."
        case .findRepresentative:
            "Use the House's official website to find your congressional district and representative. Enter your address there if needed; FACADE does not collect it."
        }
    }

    /// The exact page that gets opened. These are the only outbound links in the app.
    public var destinationURL: URL {
        switch self {
        case .takeItDown: return URL(string: "https://takeitdown.ncmec.org/")!
        case .stopNCII: return URL(string: "https://stopncii.org/")!
        case .cyberTipline: return URL(string: "https://report.cybertip.org/")!
        case .takeItDownAct: return URL(string: "https://consumer.ftc.gov/articles/image-based-abuse-what-know-and-do")!
        case .findRepresentative: return URL(string: "https://www.house.gov/representatives/find-your-representative")!
        }
    }
}
