import CNative
import Foundation
#if canImport(UIKit)
import UIKit
#endif

// Aggregated timings never touch model tensors. Spans are inclusive, not additive.
enum Profile {
 static var enabled=false
 static var destination:URL?
 static var start:UInt64=0
 static var deadline:UInt64?
 static var stopped:String?
 static let lock=NSLock()
 static var events=[[String:Any]]()
 static var observers=[NSObjectProtocol]()
 static var eventsClosed=false
 static func begin(_ name:String)->FacadeProfileToken {facade_profile_begin()}
 static func end(_ name:String,_ token:FacadeProfileToken){facade_profile_end(name,token)}
 static func measure<T>(_ name:String,_ body:() throws ->T) rethrows ->T {
  let token=begin(name);defer{end(name,token)};return try body()
 }
 static func event(_ name:String,close:Bool=false){
  guard let destination else{return}
  let now=facade_profile_now()
  let snapshot=(try? JSONSerialization.jsonObject(with:Data(String(cString:facade_profile_snapshot()).utf8))) as? [String:Any] ?? [:]
  var row:[String:Any]=["event":name,"elapsedWallSeconds":Double(now-start)/1e9,"uptimeSeconds":ProcessInfo.processInfo.systemUptime,"thermalState":ProcessInfo.processInfo.thermalState.rawValue,"utc":ISO8601DateFormatter().string(from:Date()),"residentBytes":snapshot["residentBytes"] ?? NSNull(),"physicalFootprintBytes":snapshot["physicalFootprintBytes"] ?? NSNull()]
  #if canImport(UIKit)
  // Notification callbacks can occur off-main; sample UIKit state only on main.
  if Thread.isMainThread {row["applicationState"]=UIApplication.shared.applicationState.rawValue;row["batteryState"]=UIDevice.current.batteryState.rawValue}
  #endif
  lock.lock();defer{lock.unlock()}
  guard !eventsClosed else{return}
  events.append(row)
  if close {eventsClosed=true}
  if let data=try? JSONSerialization.data(withJSONObject:events,options:[.prettyPrinted,.sortedKeys]) {try? data.write(to:destination.appendingPathComponent("lifecycle-progress.json"),options:.atomic)}
 }
 static func boundary(_ name:String)throws {
  guard destination != nil else{return}
  event(name)
  if Task.isCancelled {stopped="cancelled";event("cancelled");throw CancellationError()}
  if deadline != nil {
   let thermal=ProcessInfo.processInfo.thermalState
   if thermal == .serious || thermal == .critical {stopped="thermal-stop";event("thermal-stop");throw NativeError.invalid("Profile stopped for thermal state")}
   if facade_profile_now()>=deadline! {stopped="time-budget";event("time-budget");throw NativeError.invalid("Profile time budget reached")}
  }
 }
}

extension Profile {
    /// Output fingerprints recorded by on-device self-checks. The app's live path
    /// never records any; LivePhotoEngine checks that this stays empty.
    static var captures = [String: [String: String]]()
}
