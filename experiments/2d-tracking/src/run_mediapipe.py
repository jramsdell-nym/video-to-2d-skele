"""Pinned, sequential VIDEO-mode inference. One fresh model and selector per clip."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR']=str(ROOT/'provenance/mediapipe-mpl')
import argparse,hashlib,json,platform,subprocess,sys,time
import av,mediapipe as mp,numpy as np
from common import COCO,HitterSelector,load,dump
from mediapipe_common import NAMES

def options(variant):
    return mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(ROOT/'models'/f'pose_landmarker_{variant}.task'),delegate=mp.tasks.BaseOptions.Delegate.CPU),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,num_poses=4,
        min_pose_detection_confidence=.5,min_pose_presence_confidence=.5,
        min_tracking_confidence=.5,output_segmentation_masks=False)

def landmarks(ls,world=False):
    return {n:{'x':float(p.x),'y':float(p.y),'z':float(p.z),
        'visibility':float(p.visibility or 0),'presence':float(p.presence or 0),
        **({} if world else {'confidence':float(p.visibility or 0)})} for n,p in zip(NAMES,ls)}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--variants',nargs='+',default=['full','heavy']);args=parser.parse_args()
    cfg={'runningMode':'VIDEO','numPoses':4,'minPoseDetectionConfidence':.5,'minPosePresenceConfidence':.5,'minTrackingConfidence':.5,'outputSegmentationMasks':False,'delegate':'CPU','input':'unmirrored RGB, resized to 960x540; no crop','jointConfidence':'MediaPipe visibility; presence retained separately','displayThreshold':.3,'selection':'same HitterSelector, using the same 17 COCO-name landmarks; all 33 retained after selection','reset':'fresh PoseLandmarker and HitterSelector per clip, verified with independent first-frame replay','timestamps':'round(actual source PTS * time_base * 1000); source PTS and time_base preserved separately','frames':'blocking detect_for_video, every decoded frame, no live-stream dropping','smoothing':'No added smoothing. Upstream multi-pose graph does not enable the single-pose landmark smoothing filter. VIDEO ROI tracking remains enabled.'}
    runtime={'mediapipe':mp.__version__,'python':platform.python_version(),'platform':platform.platform(),'machine':platform.machine(),'av':av.__version__,'numpy':np.__version__,'config':cfg,'models':load(ROOT/'provenance/mediapipe-models.json'),'timing':'One run; model load separately; per-frame processing includes RGB resize, Image creation, blocking inference, result conversion and selector. Decode/JSON/render excluded. No warm-up frame fed to clip tracker.','runs':[]}
    for variant in args.variants:
        for clip in load(ROOT/'manifest.json')['clips']:
            name=clip['id'];selector=HitterSelector();rows=[];previous_ms=-1;processing=[];inference=[]
            started=time.perf_counter();model=mp.tasks.vision.PoseLandmarker.create_from_options(options(variant));load_seconds=time.perf_counter()-started
            with model, av.open(str(ROOT/clip['source'])) as container:
                for i,frame in enumerate(container.decode(video=0)):
                    assert frame.rotation==0
                    source_time=float(frame.pts*frame.time_base);timestamp_ms=round(source_time*1000)
                    assert timestamp_ms>previous_ms;previous_ms=timestamp_ms
                    t=time.perf_counter();rgb=frame.reformat(width=960,height=540,format='rgb24').to_ndarray()
                    image=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
                    if i==0:first_rgb=rgb.copy()
                    ti=time.perf_counter();result=model.detect_for_video(image,timestamp_ms);inference.append(time.perf_counter()-ti)
                    candidates=[landmarks(ls) for ls in result.pose_landmarks]
                    worlds=[landmarks(ls,True) for ls in result.pose_world_landmarks]
                    k,status=selector.choose([{j:p[j] for j in COCO} for p in candidates])
                    rows.append({'frameIndex':i,'time':source_time,'pts':frame.pts,'timeBase':str(frame.time_base),'timestampMs':timestamp_ms,'points':{} if k is None else candidates[k],'worldLandmarksMeters':{} if k is None else worlds[k],'candidates':candidates,'worldCandidatesMeters':worlds,'people':len(candidates),'selectedIndex':k,'status':status,'trackId':None if k is None else 'H1'})
                    processing.append(time.perf_counter()-t)
                    if i%30==0:print(variant,name,i,'people',len(candidates),status,flush=True)
            # Independent replay detects accidental inter-clip state reuse.
            with mp.tasks.vision.PoseLandmarker.create_from_options(options(variant)) as fresh:
                replay=fresh.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=first_rgb),rows[0]['timestampMs'])
                check=[landmarks(ls) for ls in replay.pose_landmarks]
            assert len(check)==len(rows[0]['candidates'])
            delta=max([abs(p[j][axis]-q[j][axis]) for p,q in zip(check,rows[0]['candidates']) for j in NAMES for axis in ['x','y','z','visibility']] or [0])
            assert delta<1e-6,delta
            timing={'loadSeconds':load_seconds,'processingFrameSeconds':processing,'inferenceFrameSeconds':inference,'meanProcessingMs':float(np.mean(processing)*1000),'meanInferenceMs':float(np.mean(inference)*1000),'p95ProcessingMs':float(np.percentile(processing,95)*1000),'firstFrameMs':processing[0]*1000,'steadyMeanMs':float(np.mean(processing[1:])*1000),'firstFrameReplayMaxDifference':delta}
            dump(ROOT/'outputs'/name/f'mediapipe-{variant}.json',{'method':f'MediaPipe Pose Landmarker {variant.title()}','config':cfg,'model':next(m for m in runtime['models'] if m['variant']==variant),'imageSpace':'normalized original orientation; no mirroring','worldSpace':'estimated meters relative to hip midpoint; not calibrated KinaTrax 3D','frames':rows,'timing':timing})
            runtime['runs'].append({'variant':variant,'clip':name,**{k:v for k,v in timing.items() if not k.endswith('FrameSeconds')}})
            dump(ROOT/'provenance/mediapipe-runtime.json',runtime)
            print('DONE',variant,name,round(timing['meanProcessingMs'],2),'ms/frame',flush=True)
    (ROOT/'provenance/mediapipe-requirements-lock.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))

if __name__=='__main__':main()
