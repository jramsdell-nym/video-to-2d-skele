import Foundation
import CoreGraphics

enum CapturePolicy {
    static let duration = 2.0
    // AVCaptureMovieFileOutput can count startup time before the first encoded frame.
    // Retain a little tail headroom, then losslessly select exactly two seconds.
    static let cameraLimit = 2.2
    static let countdown = ["one", "two", "three"]
    struct Format: Equatable {
        let index: Int
        let width: Int
        let height: Int
        let fps: Double
        var is4K: Bool { width >= 3840 && width <= 4096 && height == 2160 }
    }
    static func best(_ formats: [Format]) -> Format? {
        let usable = formats.filter { $0.fps >= 24 && $0.width > 0 && $0.height > 0 }
        let fourK = usable.filter(\.is4K)
        return (fourK.isEmpty ? usable : fourK).max {
            let lhs = $0.width * $0.height, rhs = $1.width * $1.height
            return lhs == rhs ? $0.fps < $1.fps : lhs < rhs
        }
    }
}

struct PosePoint: Codable, Equatable {
    let x: Double
    let y: Double // Upright normalized coordinates, top-left origin.
    let confidence: Float
    var visible: Bool {
        confidence >= 0.3 && x.isFinite && y.isFinite && (0...1).contains(x) && (0...1).contains(y)
    }
}

struct PoseFrame: Codable {
    let time: Double // Source presentation timestamp in the asset timeline.
    let points: [String: PosePoint]
    let people: Int
}

struct PoseResult: Codable {
    let frames: [PoseFrame]
    let duration: Double
    let width: Int
    let height: Int
    let displayWidth: Double
    let displayHeight: Double
    let nominalFPS: Float
    let measuredFPS: Double
    let selectedCaptureMode: String
    var processingSeconds: Double = 0
    var captureOrientation: CameraOrientationSnapshot? = nil
    var detectedFrames: Int { frames.filter { $0.points.values.filter(\.visible).count >= 5 }.count }
    var summary: String {
        "\(width) × \(height) · \(String(format: "%.1f", measuredFPS)) fps · \(String(format: "%.2f", duration)) s"
    }
    func frame(at time: Double) -> PoseFrame? {
        guard !frames.isEmpty, time.isFinite else { return nil }
        // Last source frame at or before playback time, never a future pose.
        var lo = 0, hi = frames.count
        while lo < hi {
            let mid = (lo + hi) / 2
            if frames[mid].time <= time + 0.00001 { lo = mid + 1 } else { hi = mid }
        }
        guard lo > 0 else { return nil }
        let frame = frames[lo - 1]
        let maxAge = max(0.05, 1.5 / max(measuredFPS, 1))
        return time - frame.time <= maxAge ? frame : nil
    }
}

enum Skeleton {
    static let bones = [
        ("nose", "neck"), ("neck", "left_shoulder"), ("neck", "right_shoulder"),
        ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
        ("neck", "root"), ("root", "left_hip"), ("root", "right_hip"),
        ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"), ("right_knee", "right_ankle")
    ]
    static func map(_ point: PosePoint, into rect: CGRect) -> CGPoint {
        CGPoint(x: rect.minX + point.x * rect.width, y: rect.minY + point.y * rect.height)
    }
    static func visibleBones(in frame: PoseFrame) -> [(PosePoint, PosePoint)] {
        bones.compactMap { a, b in
            guard let p = frame.points[a], let q = frame.points[b], p.visible, q.visible else { return nil }
            return (p, q)
        }
    }
}
