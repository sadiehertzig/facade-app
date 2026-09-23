// swift-tools-version: 6.2
import PackageDescription
let package = Package(name: "FACADEKit", platforms: [.iOS(.v17), .macOS(.v14)],
    products: [.library(name: "FACADEKit", targets: ["FACADEKit"])], targets: [.target(name: "FACADEKit")])
