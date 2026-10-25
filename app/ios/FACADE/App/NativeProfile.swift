// NativeProfile.swift
//
// A small timing and lifecycle recorder for the FACADE engine.
//
// Where it sits: almost every engine function begins with
// `Profile.begin("name")` and ends with `Profile.end(...)`. Those calls go to a
// C profiler (facade_profile_*) that is switched off in the shipping app, so
// they cost almost nothing. `Profile.boundary(...)` is also where the engine
// checks its time limit and the phone's thermal state between stages.
//
// Key ideas:
// 1. Nothing here ever reads or writes image pixels. It records names, times,
//    memory numbers and thermal state only.
// 2. `destination` is nil unless a run is in progress; LivePhotoEngine sets it
//    to a temporary folder in Caches and clears it at the end.
// 3. Profile spans are inclusive (an outer span includes its inner spans), so
//    they must not be added together.
import CNative
import Foundation
#if canImport(UIKit)
import UIKit
#endif

/// Process-wide profiler state. Aggregated timings never touch model tensors.
/// Spans are inclusive, not additive.
enum Profile {
 /// Whether the detailed C profiler is on. The live app keeps this false.
 static var enabled=false
 /// Folder for the lifecycle-progress JSON, or nil when no run is active.
 static var destination:URL?
 /// Nanosecond timestamp of the current run's start.
 static var start:UInt64=0
 /// Nanosecond timestamp after which `boundary` throws a time-budget error.
 static var deadline:UInt64?
 /// Why the run stopped early, if it did ("cancelled", "thermal-stop", ...).
 static var stopped:String?
 /// Protects `events` and `eventsClosed`, which can be touched from any thread.
 static let lock=NSLock()
 /// Lifecycle rows written to lifecycle-progress.json.
 static var events=[[String:Any]]()
 /// Notification observers. The live app installs none; the engine checks
 /// this is empty before starting.
 static var observers=[NSObjectProtocol]()
 /// Set when the last event has been written; later events are dropped.
 static var eventsClosed=false
 /// Start a named span in the C profiler and get a token for `end`.
 static func begin(_ name:String)->FacadeProfileToken {facade_profile_begin()}
 /// Close a span started with `begin`.
 static func end(_ name:String,_ token:FacadeProfileToken){facade_profile_end(name,token)}
 /// Run `body` inside a named span.
 static func measure<T>(_ name:String,_ body:() throws ->T) rethrows ->T {
  let token=begin(name);defer{end(name,token)};return try body()
 }
 /// Append one lifecycle row (elapsed time, uptime, thermal state, memory) and
 /// rewrite the JSON file. Does nothing when no run is active.
 static func event(_ name:String,close:Bool=false){
  guard let destination else{return}
  let now=facade_profile_now()
  // Memory numbers come from the C side as a small JSON snapshot.
  let snapshot=(try? JSONSerialization.jsonObject(with:Data(String(cString:facade_profile_snapshot()).utf8))) as? [String:Any] ?? [:]
  var row:[String:Any]=["event":name,"elapsedWallSeconds":Double(now-start)/1e9,"uptimeSeconds":ProcessInfo.processInfo.systemUptime,"thermalState":ProcessInfo.processInfo.thermalState.rawValue,"utc":ISO8601DateFormatter().string(from:Date()),"residentBytes":snapshot["residentBytes"] ?? NSNull(),"physicalFootprintBytes":snapshot["physicalFootprintBytes"] ?? NSNull()]
  #if canImport(UIKit)
  // Notification callbacks can occur off the main thread; UIKit state is only
  // safe to sample on the main thread, so it is included only then.
  if Thread.isMainThread {row["applicationState"]=UIApplication.shared.applicationState.rawValue;row["batteryState"]=UIDevice.current.batteryState.rawValue}
  #endif
  lock.lock();defer{lock.unlock()}
  guard !eventsClosed else{return}
  events.append(row)
  if close {eventsClosed=true}
  if let data=try? JSONSerialization.data(withJSONObject:events,options:[.prettyPrinted,.sortedKeys]) {try? data.write(to:destination.appendingPathComponent("lifecycle-progress.json"),options:.atomic)}
 }
 /// A checkpoint between engine stages: records an event and then throws if the
 /// run was cancelled, the phone is too hot, or the time budget has passed.
 /// Only checks when a run is active (destination set).
 static func boundary(_ name:String)throws {
  guard destination != nil else{return}
  event(name)
  // Cancellation is reported with Swift's standard CancellationError.
  if Task.isCancelled {stopped="cancelled";event("cancelled");throw CancellationError()}
  // The deadline is only set by LivePhotoEngine for live runs.
  if deadline != nil {
   let thermal=ProcessInfo.processInfo.thermalState
   if thermal == .serious || thermal == .critical {stopped="thermal-stop";event("thermal-stop");throw NativeError.invalid("Profile stopped for thermal state")}
   if facade_profile_now()>=deadline! {stopped="time-budget";event("time-budget");throw NativeError.invalid("Profile time budget reached")}
  }
 }
}

extension Profile {
    /// Output fingerprints recorded by on-device self-checks. The app's live path
    /// never records any; LivePhotoEngine checks that this stays empty before and
    /// clears it after each run.
    static var captures = [String: [String: String]]()
}
