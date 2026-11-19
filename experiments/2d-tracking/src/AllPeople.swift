import Foundation
import Vision

struct CandidateFrame: Codable { let time: Double; let candidates: [[String:PosePoint]] }
struct CandidateResult: Codable {
    let revision: Int; let processingSeconds: Double; let width: Double; let height: Double
    let frames: [CandidateFrame]
}
@main struct Main {
    static func main() async throws {
        let url=URL(fileURLWithPath:CommandLine.arguments[1])
        let request=VNDetectHumanBodyPoseRequest()
        let names: [(VNHumanBodyPoseObservation.JointName,String)] = [
            (.nose,"nose"),(.neck,"neck"),(.root,"root"),
            (.leftShoulder,"left_shoulder"),(.rightShoulder,"right_shoulder"),
            (.leftElbow,"left_elbow"),(.rightElbow,"right_elbow"),
            (.leftWrist,"left_wrist"),(.rightWrist,"right_wrist"),
            (.leftHip,"left_hip"),(.rightHip,"right_hip"),
            (.leftKnee,"left_knee"),(.rightKnee,"right_knee"),
            (.leftAnkle,"left_ankle"),(.rightAnkle,"right_ankle"),
            (.leftEye,"left_eye"),(.rightEye,"right_eye"),(.leftEar,"left_ear"),(.rightEar,"right_ear")]
        var observations: [[[String:PosePoint]]] = []
        let result=try await PoseProcessor.analyze(url:url,captureMode:"offline all-person extraction; unchanged preprocessing",estimator:{image in
            try VNImageRequestHandler(cgImage:image,orientation:.up).perform([request])
            let people=request.results ?? []
            let poses=try people.map { person in
                let points=try person.recognizedPoints(.all)
                return Dictionary(uniqueKeysWithValues:names.compactMap { name,key in
                    guard let p=points[name] else { return nil as (String,PosePoint)? }
                    return (key,PosePoint(x:p.location.x,y:1-p.location.y,confidence:p.confidence))
                })
            }
            observations.append(poses)
            return (points:poses.count == 1 ? poses[0] : [:],people:poses.count)
        },progress:{_ in})
        let frames=zip(result.frames,observations).map { CandidateFrame(time:$0.0.time,candidates:$0.1) }
        let out=CandidateResult(revision:request.revision,processingSeconds:result.processingSeconds,width:result.displayWidth,height:result.displayHeight,frames:frames)
        try JSONEncoder().encode(out).write(to:url.deletingLastPathComponent().appendingPathComponent("vision-all.json"))
        print("\(url.deletingLastPathComponent().lastPathComponent): \(frames.count) frames, \(result.processingSeconds)s")
    }
}
