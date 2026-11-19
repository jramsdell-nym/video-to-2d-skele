import Foundation
struct CameraOrientationSnapshot: Codable, Sendable {
 let interfaceOrientation: String
 let physicalOrientation: String
 let rotationDegrees: Double
 var canvasRotationDegrees: Double = 0
 var source: String = "gravity"
 var previewRotationDegrees: Double? = nil
}
