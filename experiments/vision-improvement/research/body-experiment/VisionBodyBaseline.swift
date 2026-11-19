import Foundation
import Vision
import CoreImage
import ImageIO
import Darwin

// Approximation of FanLab/PoseProcessor.swift on the exact reference PNGs.
// This executes macOS Vision with explicit CPU-only inference and software CI.
let names: [(VNHumanBodyPoseObservation.JointName, String)] = [
    (.leftShoulder,"Left shoulder"),(.rightShoulder,"Right shoulder"),
    (.leftElbow,"Left elbow"),(.rightElbow,"Right elbow"),
    (.leftWrist,"Left wrist"),(.rightWrist,"Right wrist"),
    (.leftHip,"Left hip"),(.rightHip,"Right hip"),
    (.leftKnee,"Left knee"),(.rightKnee,"Right knee"),
    (.leftAnkle,"Left ankle"),(.rightAnkle,"Right ankle")]

func sysctlString(_ name: String) -> String {
    var size = 0
    guard sysctlbyname(name,nil,&size,nil,0) == 0, size > 0 else { return "unknown" }
    var bytes = [CChar](repeating:0,count:size)
    guard sysctlbyname(name,&bytes,&size,nil,0) == 0 else { return "unknown" }
    return String(cString:bytes)
}

func saveJSON(_ value: Any, _ url: URL) throws {
    let data = try JSONSerialization.data(withJSONObject:value,options:[.prettyPrinted,.sortedKeys])
    try data.write(to:url,options:.atomic)
}

enum Failure: Error { case message(String) }

func main() throws {
    guard CommandLine.arguments.count == 3 else {
        throw Failure.message("Usage: VisionBodyBaseline <body-experiment-run-directory> <new-output-directory>")
    }
    let run = URL(fileURLWithPath:CommandLine.arguments[1],isDirectory:true)
    let out = URL(fileURLWithPath:CommandLine.arguments[2],isDirectory:true)
    guard !FileManager.default.fileExists(atPath:out.path) else { throw Failure.message("Output already exists") }
    guard let rows = try JSONSerialization.jsonObject(with:Data(contentsOf:run.appendingPathComponent("val-rows.json"))) as? [[String:Any]] else {
        throw Failure.message("Invalid reference rows")
    }
    let context = CIContext(options:[.useSoftwareRenderer:true,.cacheIntermediates:false])
    let request = VNDetectHumanBodyPoseRequest()
    request.usesCPUOnly = true
    guard request.usesCPUOnly else { throw Failure.message("CPU-only Vision request was not accepted") }
    let start = CFAbsoluteTimeGetCurrent()
    var predictions = [[String:Any]]()
    var allCandidates = [[String:Any]]()
    var preparationTimes = [Double](), inferenceTimes = [Double]()
    for (index,row) in rows.enumerated() {
        let result: ([String:Any],[String:Any],Double,Double) = try autoreleasepool {
            guard let imagePath=row["image"] as? String, let hash=row["videoSHA256"] as? String,
                  let frameIndex=row["frameIndex"] as? Int, let width=row["width"] as? Int, let height=row["height"] as? Int else {
                throw Failure.message("Reference row identity/dimensions missing")
            }
            let preparationStart=CFAbsoluteTimeGetCurrent()
            let sourceURL=run.appendingPathComponent("dataset").appendingPathComponent(imagePath)
            guard let source=CGImageSourceCreateWithURL(sourceURL as CFURL,nil),
                  let pixels=CGImageSourceCreateImageAtIndex(source,0,nil),pixels.width==width,pixels.height==height else {
                throw Failure.message("Cannot decode matching source PNG")
            }
            // The supplied PNGs are already upright and unmirrored. Do not apply
            // a second camera/video orientation transform or change side.
            let scale=min(1,960.0/Double(max(width,height)))
            let prepared=CIImage(cgImage:pixels).transformed(by:CGAffineTransform(scaleX:scale,y:scale))
            guard let image=context.createCGImage(prepared,from:prepared.extent) else { throw Failure.message("Cannot resize image") }
            let preparedAt=CFAbsoluteTimeGetCurrent()
            try VNImageRequestHandler(cgImage:image,orientation:.up).perform([request])
            let inferredAt=CFAbsoluteTimeGetCurrent()
            let people=request.results ?? []
            var candidates=[[String:Any]]()
            for person in people {
                let recognized=try person.recognizedPoints(.all)
                var raw=[String:Any]()
                for (joint,point) in recognized {
                    raw[joint.rawValue.rawValue] = ["x":Double(point.location.x),"y":1-Double(point.location.y),"confidence":Double(point.confidence)]
                }
                var coordinates=[[Any]](), confidences=[Double]()
                for (joint,_) in names {
                    if let point=recognized[joint] {
                        coordinates.append([Double(point.location.x)*Double(width),(1-Double(point.location.y))*Double(height)])
                        confidences.append(Double(point.confidence))
                    } else {
                        coordinates.append([NSNull(),NSNull()]);confidences.append(0)
                    }
                }
                candidates.append(["allRecognizedPoints":raw,"prediction":coordinates,"keypointConfidences":confidences])
            }
            var prediction:[String:Any]=["videoSHA256":hash,"frameIndex":frameIndex,"detected":people.count==1,"candidateCount":people.count]
            if candidates.count==1 {
                prediction["prediction"]=candidates[0]["prediction"]
                prediction["keypointConfidences"]=candidates[0]["keypointConfidences"]
                prediction["selectedCandidateIndex"]=0
            }
            let complete:[String:Any]=["videoSHA256":hash,"frameIndex":frameIndex,"candidateCount":people.count,
                                       "candidates":candidates,"resizedWidth":image.width,"resizedHeight":image.height,
                                       "preparationSeconds":preparedAt-preparationStart,"inferenceSeconds":inferredAt-preparedAt]
            return (prediction,complete,preparedAt-preparationStart,inferredAt-preparedAt)
        }
        predictions.append(result.0);allCandidates.append(result.1)
        preparationTimes.append(result.2);inferenceTimes.append(result.3)
        if (index+1)%40==0 { print("Vision CPU: \(index+1)/\(rows.count)") }
    }
    try FileManager.default.createDirectory(at:out,withIntermediateDirectories:true)
    try saveJSON(predictions,out.appendingPathComponent("predictions.json"))
    try saveJSON(allCandidates,out.appendingPathComponent("all-candidates.json"))
    let metadata:[String:Any]=[
        "method":"macOS Vision approximation of phone pipeline; not iPhone parity",
        "frames":rows.count,"hardwareModel":sysctlString("hw.model"),"cpuBrand":sysctlString("machdep.cpu.brand_string"),
        "macOS":ProcessInfo.processInfo.operatingSystemVersionString,
        "requestClass":"VNDetectHumanBodyPoseRequest","requestRevision":request.revision,
        "currentRevision":VNDetectHumanBodyPoseRequest.currentRevision,
        "supportedRevisions":VNDetectHumanBodyPoseRequest.supportedRevisions.map{$0},
        "usesCPUOnly":request.usesCPUOnly,"coreImageUseSoftwareRenderer":true,
        "input":"Exact heldout reference PNG; already upright/unmirrored; no camera orientation reapplied",
        "resize":"CIImage affine scale min(1,960/max(width,height)), then createCGImage from extent",
        "pointOrder":names.map{$0.1},"coordinates":"Original-image pixels, top-left origin; y=1-VisionY",
        "selection":"Only exactly one Vision person produces body points. All candidates retained separately. No reference selection.",
        "scorePolicy":"score_body.py uses confidence >=0.30 and finite in-frame coordinates, matching current PosePoint.visible",
        "totalSeconds":CFAbsoluteTimeGetCurrent()-start,
        "preparationSeconds":preparationTimes,"inferenceSeconds":inferenceTimes,
        "limitations":["macOS request implementation and hardware differ from iPhone.","PNG decode differs from native video YUV decode; no byte-parity claim.","usesCPUOnly is the installed SDK's deprecated but available explicit CPU request flag.","This frame baseline does not simulate the app's additional whole-clip multi-person rejection."]]
    try saveJSON(metadata,out.appendingPathComponent("runtime.json"))
    print("Saved \(predictions.count) CPU-only Vision predictions to \(out.path)")
}

do { try main() } catch { fputs("Vision baseline failed: \(error)\n",stderr);exit(1) }
