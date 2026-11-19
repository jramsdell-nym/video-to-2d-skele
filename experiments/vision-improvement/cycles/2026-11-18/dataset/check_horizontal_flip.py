"""Exercise installed Ultralytics RandomFlip using frozen prepared labels."""
from pathlib import Path
import copy
import hashlib
import inspect
import json
import numpy as np
import yaml
import ultralytics
from PIL import Image, ImageDraw, ImageFont
from ultralytics.data.augment import RandomFlip
from ultralytics.utils.instance import Instances

ROOT = Path(__file__).resolve().parent
CYCLE = ROOT.parent
VIDEO = '4a594302a790c92e772ce68f33ae87677153799b7959c87cda11e713950b6526'
FRAME = 372

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    imagepath = ROOT/'images'/f'{VIDEO}_{FRAME}.png'
    pixels = np.asarray(Image.open(imagepath).convert('RGB')).copy()
    inputs = {str(imagepath):sha(imagepath)}
    outputs={}; all_labels={}
    for task in ['body','bat']:
        prepared=CYCLE/'training'/task/'dataset'
        configpath=prepared/'data.yaml'; labelpath=prepared/'labels/train'/f'{VIDEO}_{FRAME}.txt'
        cfg=yaml.safe_load(configpath.read_text())
        inputs.update({str(p):sha(p) for p in (configpath,labelpath)})
        nums=np.array(labelpath.read_text().strip().split(),dtype=np.float32)
        points=nums[5:].reshape(1,*cfg['kpt_shape']); box=nums[1:5].reshape(1,4)
        labels={'img':pixels.copy(),'instances':Instances(box.copy(),keypoints=points.copy(),bbox_format='xywh',normalized=True),'cls':nums[:1].copy()}
        original=copy.deepcopy(labels)
        flip=RandomFlip(p=1.0,direction='horizontal',flip_idx=cfg['flip_idx'])
        mirrored=flip(copy.deepcopy(original))
        double=flip(copy.deepcopy(mirrored))
        expected_points=points.copy();expected_points[:,:,0]=1-expected_points[:,:,0]
        expected_points=expected_points[:,cfg['flip_idx'],:]
        expected_box=box.copy();expected_box[:,0]=1-expected_box[:,0]
        checks={
            'imagePixelsExactHorizontalReverse':np.array_equal(mirrored['img'],pixels[:,::-1,:]),
            'bboxReflectsCenterAndPreservesSize':np.allclose(mirrored['instances'].bboxes,expected_box,atol=1e-7,rtol=0),
            'keypointCoordinatesUseDeclaredMap':np.allclose(mirrored['instances'].keypoints,expected_points,atol=1e-7,rtol=0),
            'visibilityExactlyFollowsSameMap':np.array_equal(mirrored['instances'].keypoints[:,:,2],points[:,cfg['flip_idx'],2]),
            'classUnchanged':np.array_equal(mirrored['cls'],original['cls']),
            'doubleFlipImageExact':np.array_equal(double['img'],pixels),
            'doubleFlipBoxesRoundtrip':np.allclose(double['instances'].bboxes,box,atol=1e-7,rtol=0),
            'doubleFlipKeypointsRoundtrip':np.allclose(double['instances'].keypoints,points,atol=1e-7,rtol=0),
            'originalInMemoryCopyUntouched':np.array_equal(original['img'],pixels) and np.array_equal(original['instances'].keypoints,points),
        }
        assert all(checks.values()),(task,checks)
        outputs[task]={'preparedLabelPath':str(labelpath),'flipIndex':cfg['flip_idx'],'visibilityBefore':points[0,:,2].tolist(),
            'visibilityAfter':mirrored['instances'].keypoints[0,:,2].tolist(),'checks':{k:bool(v) for k,v in checks.items()},
            'doubleFlipMaxCoordinateDelta':float(np.max(np.abs(double['instances'].keypoints-points))),
            'before':{'bboxXYWH':box.tolist(),'keypoints':points.tolist()},
            'after':{'bboxXYWH':mirrored['instances'].bboxes.tolist(),'keypoints':mirrored['instances'].keypoints.tolist()}}
        all_labels[task]=(original,mirrored)
    assert any(v==1 for v in outputs['body']['visibilityBefore'])
    assert any(v==1 for v in outputs['bat']['visibilityBefore'])
    assert all(sha(p)==h for p,h in inputs.items())
    out={'scope':'Read-only verification on one real prepared training frame, with forced p=1.0 for this test only. Training remains fliplr=0.5.',
        'ultralyticsVersion':ultralytics.__version__,'implementationPath':inspect.getfile(RandomFlip),
        'implementationSHA256':sha(inspect.getfile(RandomFlip)), 'videoSHA256':VIDEO,'frameIndex':FRAME,
        'inputs':[{'path':p,'sha256':h} for p,h in inputs.items()],'sourceHashesUnchanged':True,'tasks':outputs,
        'notes':['Body17 anatomical left/right pairs swap; bat identity map [0,1] preserves tip then knob.',
                 'Zero-visibility head coordinates are reflected too, but retain visibility=0 and remain unsupervised.',
                 'Floating-point roundtrip checked at absolute tolerance 1e-7; image pixels and visibility are exact.',
                 'Horizontal reflection preserves camera occlusion; it does not turn a rear view into an open view.',
                 'This checks RandomFlip itself with prepared source labels, not the entire stochastic augmentation chain.']}
    (ROOT/'horizontal-flip-check.json').write_text(json.dumps(out,indent=2)+'\n')
    render(all_labels,pixels)
    print(json.dumps({'allChecksPassed':True,'sourceHashesUnchanged':True,'version':ultralytics.__version__,
                      'bodyVisibility':outputs['body']['visibilityBefore'],'batVisibility':outputs['bat']['visibilityBefore']}))

def render(all_labels,pixels):
    image=Image.new('RGB',(1440,940),'#141923');d=ImageDraw.Draw(image)
    font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',20)
    small=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',15)
    d.text((14,12),'ACTUAL ULTRALYTICS HORIZONTAL FLIP | same training frame; reviewed labels only',font=font,fill='white')
    d.text((14,43),'Body: green anatomical left, pink right. Amber=hidden estimate. Bat tip/knob names stay fixed.',font=font,fill='white')
    names=['N','LEye','REye','LEar','REar','LS','RS','LE','RE','LW','RW','LH','RH','LK','RK','LA','RA']
    h,w=pixels.shape[:2]
    for column in [0,1]:
        body=all_labels['body'][column];bat=all_labels['bat'][column]
        box=body['instances'].bboxes[0];cx,cy,bw,bh=box
        crop=(max(0,int((cx-bw/2)*w-100)),max(0,int((cy-bh/2)*h-100)),min(w,int((cx+bw/2)*w+100)),min(h,int((cy+bh/2)*h+80)))
        scale=min(690/(crop[2]-crop[0]),800/(crop[3]-crop[1]));size=(round((crop[2]-crop[0])*scale),round((crop[3]-crop[1])*scale))
        left=column*720+(720-size[0])//2;top=110
        canvas=Image.fromarray(body['img']).crop(crop).resize(size,Image.Resampling.LANCZOS)
        image.paste(canvas,(left,top));d.text((column*720+15,78),'Original source' if column==0 else 'Mirrored pixels + paired anatomical labels',font=font,fill='white')
        def loc(p):return left+(p[0]*w-crop[0])*scale,top+(p[1]*h-crop[1])*scale
        for j,p in enumerate(body['instances'].keypoints[0]):
            if p[2]==0:continue
            x,y=loc(p);color='#ffce65' if p[2]==1 else '#7af49e' if j%2 else '#ff80cc'
            d.ellipse((x-5,y-5,x+5,y+5),outline=color,width=2);d.text((x+5,y+3),names[j],font=small,fill=color,stroke_width=1,stroke_fill='black')
        for name,p in zip(['TIP','KNOB'],bat['instances'].keypoints[0]):
            x,y=loc(p);color='#ffce65' if p[2]==1 else '#5adfff'
            d.rectangle((x-5,y-5,x+5,y+5),outline=color,width=2);d.text((x+6,y-17),name,font=small,fill=color,stroke_width=1,stroke_fill='black')
    image.save(ROOT/'horizontal-flip-check.png')

if __name__=='__main__':main()
