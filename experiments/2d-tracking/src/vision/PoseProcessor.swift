import AVFoundation
import Vision
import CoreImage

enum PoseProcessor {
    // Track transforms use top-left video coordinates. Core Image uses bottom-left.
    // Conjugate with Y flips so rotation AND reflection match AVPlayer exactly.
    static func uprightTransform(size: CGSize, trackTransform: CGAffineTransform) -> CGAffineTransform {
        let bounds = CGRect(origin: .zero, size: size).applying(trackTransform).standardized
        return CGAffineTransform(a: 1, b: 0, c: 0, d: -1, tx: 0, ty: size.height)
            .concatenating(trackTransform)
            .concatenating(CGAffineTransform(a: 1, b: 0, c: 0, d: -1, tx: -bounds.minX, ty: bounds.maxY))
    }
    private static let names: [(VNHumanBodyPoseObservation.JointName, String)] = [
        (.nose, "nose"), (.neck, "neck"), (.root, "root"),
        (.leftShoulder, "left_shoulder"), (.rightShoulder, "right_shoulder"),
        (.leftElbow, "left_elbow"), (.rightElbow, "right_elbow"),
        (.leftWrist, "left_wrist"), (.rightWrist, "right_wrist"),
        (.leftHip, "left_hip"), (.rightHip, "right_hip"),
        (.leftKnee, "left_knee"), (.rightKnee, "right_knee"),
        (.leftAnkle, "left_ankle"), (.rightAnkle, "right_ankle"),
        (.leftEye, "left_eye"), (.rightEye, "right_eye"),
        (.leftEar, "left_ear"), (.rightEar, "right_ear")
    ]
    typealias Estimate = (CGImage) throws -> (points: [String: PosePoint], people: Int)
    static func analyze(url: URL, captureMode: String, estimator: Estimate? = nil, progress: @escaping @Sendable (Double) -> Void) async throws -> PoseResult {
        let started = ContinuousClock.now
        let asset = AVURLAsset(url: url)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else { throw error("The recording has no video track.") }
        let duration = try await asset.load(.duration).seconds
        let size = try await track.load(.naturalSize)
        let transform = try await track.load(.preferredTransform)
        let nominalFPS = try await track.load(.nominalFrameRate)
        let display = CGRect(origin: .zero, size: size).applying(transform).standardized
        guard duration.isFinite, duration > 0, display.width > 0, display.height > 0 else { throw error("Invalid video dimensions or duration.") }

        let reader = try AVAssetReader(asset: asset)
        reader.timeRange = CMTimeRange(start: .zero, duration: CMTime(seconds: duration, preferredTimescale: 60000))
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
        ])
        output.alwaysCopiesSampleData = false
        reader.add(output)
        guard reader.startReading() else { throw reader.error ?? error("Cannot read this recording.") }
        defer { if reader.status == .reading { reader.cancelReading() } }
        let context = CIContext(options: [.cacheIntermediates: false])
        let request = VNDetectHumanBodyPoseRequest()
        var frames: [PoseFrame] = []
        // A single decoded source buffer and a <=960px image live per iteration.
        // Only compact joint numbers survive, never raw frame arrays.
        while reader.status == .reading {
            try Task.checkCancellation()
            let frame: PoseFrame? = try autoreleasepool {
                guard let sample = output.copyNextSampleBuffer() else { return nil }
                guard let buffer = CMSampleBufferGetImageBuffer(sample) else { throw error("A video frame could not be decoded.") }
                let pts = CMSampleBufferGetPresentationTimeStamp(sample).seconds
                var upright = CIImage(cvPixelBuffer: buffer).transformed(by: uprightTransform(size: size, trackTransform: transform))
                let scale = min(1, 960 / max(upright.extent.width, upright.extent.height))
                upright = upright.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
                guard let image = context.createCGImage(upright, from: upright.extent) else { throw error("Could not prepare a frame for pose detection.") }
                if let estimator {
                    let estimate = try estimator(image)
                    return PoseFrame(time: pts, points: estimate.points, people: estimate.people)
                }
                try VNImageRequestHandler(cgImage: image, orientation: .up).perform([request])
                let people = request.results ?? []
                // Ambiguous multi-person frames stay empty rather than silently switching bodies.
                var points: [String: PosePoint] = [:]
                if people.count == 1, let observation = people.first {
                    let recognized = try observation.recognizedPoints(.all)
                    for (name, key) in names {
                        guard let point = recognized[name] else { continue }
                        points[key] = PosePoint(x: point.location.x, y: 1 - point.location.y, confidence: point.confidence)
                    }
                }
                return PoseFrame(time: pts, points: points, people: people.count)
            }
            guard let frame else { break }
            frames.append(frame)
            progress(min(0.99, frame.time / duration))
        }
        guard reader.status == .completed else { throw reader.error ?? error("Video decoding did not complete.") }
        guard !frames.isEmpty else { throw error("The recording contained no readable frames.") }
        let span = (frames.last?.time ?? 0) - (frames.first?.time ?? 0)
        let fps = frames.count > 1 && span > 0 ? Double(frames.count - 1) / span : Double(nominalFPS)
        var result = PoseResult(frames: frames, duration: duration, width: Int(size.width), height: Int(size.height),
                                displayWidth: display.width, displayHeight: display.height, nominalFPS: nominalFPS,
                                measuredFPS: fps, selectedCaptureMode: captureMode)
        let elapsed = started.duration(to: .now).components
        result.processingSeconds = Double(elapsed.seconds) + Double(elapsed.attoseconds) / 1e18
        try JSONEncoder().encode(result).write(to: url.deletingPathExtension().appendingPathExtension("json"), options: .atomic)
        progress(1)
        return result
    }
    private static func error(_ message: String) -> NSError { NSError(domain: "FanLab.Pose", code: 1, userInfo: [NSLocalizedDescriptionKey: message]) }
}
