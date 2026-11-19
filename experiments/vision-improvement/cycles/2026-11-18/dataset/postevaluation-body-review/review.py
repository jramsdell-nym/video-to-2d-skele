"""Bounded post-evaluation interpretation; never mutates labels or predictions."""
from pathlib import Path
import hashlib
import json
import math
from collections import Counter, defaultdict
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent
CYCLE = OUT.parent.parent
CASES = [
    ('left-wrist-follow-through', '675cb2', 115, 4),
    ('right-wrist-overhead', '345150', 98, 5),
    ('left-elbow-follow-through', 'cabf10', 119, 2),
    ('right-elbow-contact', '8a61a9', 75, 3),
    ('left-knee-rotation', '8a61a9', 90, 8),
    ('left-knee-late-follow-through', '05e7d9', 111, 8),
]
NAMES = ['LS','RS','LE','RE','LW','RW','LH','RH','LK','RK','LA','RA']
CHAINS = {4:[0,2,4], 5:[1,3,5], 2:[0,2,4], 3:[1,3,5], 8:[6,8,10]}

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    paths = {
        'rows': CYCLE/'dataset/splits/body-evaluation-rows.json',
        'candidate': CYCLE/'training/body/candidate-predictions.json',
        'previous': CYCLE/'training/body/previous-predictions.json',
        'candidateScore': CYCLE/'training/body/candidate-primary-score.json',
        'previousScore': CYCLE/'training/body/previous-primary-score.json',
        'protocol': CYCLE/'training/body/protocol.json',
        'splitReceipt': CYCLE/'split-receipt.json',
    }
    hashes = {k:sha(v) for k,v in paths.items()}
    data = {k:json.loads(v.read_text()) for k,v in paths.items()}
    lookup = {k:{(r['videoSHA256'],r['frameIndex']):r for r in data[k]}
              for k in ('rows','candidate','previous')}
    scores = {k:{(r['videoSHA256'],r['frameIndex'],r['jointIndex']):r
                 for r in data[k]['jointDetails']} for k in ('candidateScore','previousScore')}
    font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',19)
    small = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',15)
    titlefont = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',23)
    output=[]
    for name,prefix,index,joint in CASES:
        video=next(r['videoSHA256'] for r in data['rows'] if r['videoSHA256'].startswith(prefix))
        center=lookup['rows'][(video,index)]
        indices=[i for i in [index-2,index,index+2] if (video,i) in lookup['rows']]
        if len(indices)<3:
            allindices=sorted(r['frameIndex'] for r in data['rows'] if r['videoSHA256']==video)
            indices=allindices[-3:] if index>2 else allindices[:3]
        points=[]
        for i in indices:
            r=lookup['rows'][(video,i)]
            points += [(r['points'][j]['x']*r['width'],r['points'][j]['y']*r['height']) for j in CHAINS[joint]]
            for kind in ('candidate','previous'):
                points += [lookup[kind][(video,i)]['prediction'][j] for j in CHAINS[joint]]
        x0=max(0,math.floor(min(p[0] for p in points)-65)); x1=min(center['width'],math.ceil(max(p[0] for p in points)+65))
        y0=max(0,math.floor(min(p[1] for p in points)-65)); y1=min(center['height'],math.ceil(max(p[1] for p in points)+65))
        crop=(x0,y0,x1,y1); scale=min(480/(x1-x0),320/(y1-y0)); size=(round((x1-x0)*scale),round((y1-y0)*scale))
        image=Image.new('RGB',(1500,1180),'#131922');draw=ImageDraw.Draw(image)
        draw.text((10,10),f'POST-EVALUATION REVIEW | {name}',font=titlefont,fill='white')
        draw.text((10,42),f'{center["trial"]} | {center["camera"]} | target {NAMES[joint]} | green reference; orange previous; cyan new',font=font,fill='white')
        draw.text((10,70),'Original pixels (top); previous + frozen reference (middle); new + frozen reference (bottom). No relabeling.',font=font,fill='white')
        record={'case':name,'videoSHA256':video,'trial':center['trial'],'camera':center['camera'],
                'jointIndex':joint,'joint':data['candidateScore']['jointOrder'][joint],
                'centerFrameIndex':index,'neighborFrameIndices':indices,'crop':crop,'frames':[]}
        for col,i in enumerate(indices):
            r=lookup['rows'][(video,i)]; p=lookup['previous'][(video,i)]; c=lookup['candidate'][(video,i)]
            assert sha(r['imagePath'])==r['imageSHA256']
            original=Image.open(r['imagePath']).convert('RGB').crop(crop).resize(size,Image.Resampling.LANCZOS)
            left=col*500+(500-size[0])//2
            cs=scores['candidateScore'][(video,i,joint)]; ps=scores['previousScore'][(video,i,joint)]
            draw.text((col*500+10,101),f'UI frame {i+1} | t={r["frame"]["time"]:.3f}s | visibility {r["points"][joint]["visibility"]}',font=small,fill='white')
            draw.text((col*500+10,123),f'Error previous {ps["errorPixels"]:.1f}px / new {cs["errorPixels"]:.1f}px; PCK limit {cs["bodyScalePixels"]*.05:.1f}px',font=small,fill='white')
            for rownum,kind,color in [(0,None,None),(1,'previous','#ffb657'),(2,'candidate','#49d7ff')]:
                top=153+rownum*340
                image.paste(original,(left,top))
                if kind is None: continue
                pred=lookup[kind][(video,i)]
                def loc(x,y):return left+(x-x0)*scale,top+(y-y0)*scale
                refs=[loc(r['points'][j]['x']*r['width'],r['points'][j]['y']*r['height']) for j in CHAINS[joint]]
                preds=[loc(*pred['prediction'][j]) for j in CHAINS[joint]]
                draw.line(refs,fill='#7af49e',width=2);draw.line(preds,fill=color,width=2)
                for j,rr,pp in zip(CHAINS[joint],refs,preds):
                    x,y=rr;refcolor='#7af49e' if r['points'][j]['visibility']==2 else '#f1e268'
                    rad=6 if j==joint else 4
                    draw.ellipse((x-rad,y-rad,x+rad,y+rad),outline=refcolor,width=2)
                    draw.text((x+7,y+3),NAMES[j],font=small,fill=refcolor,stroke_width=1,stroke_fill='black')
                    x,y=pp;draw.line((x-rad,y-rad,x+rad,y+rad),fill=color,width=3);draw.line((x-rad,y+rad,x+rad,y-rad),fill=color,width=3)
            record['frames'].append({'frameIndex':i,'uiFrame':i+1,'imagePath':r['imagePath'],'imageSHA256':r['imageSHA256'],
                'point':r['points'][joint],'reviewMethod':r.get('reviewMethod'), 'previous':p['prediction'][joint],
                'candidate':c['prediction'][joint],'candidateConfidence':c['keypointConfidences'][joint],
                'candidateErrorPixels':cs['errorPixels'],'previousErrorPixels':ps['errorPixels'],
                'passLimitPixels':cs['bodyScalePixels']*.05})
        imagepath=OUT/(name+'.png');image.save(imagepath)
        record.update(imagePath=str(imagepath),imageSHA256=sha(imagepath));output.append(record)
    counts=defaultdict(Counter)
    for key,c in scores['candidateScore'].items():
        if c['visibility']!=2:continue
        p=scores['previousScore'][key]
        passed=lambda r:r['errorPixels'] is not None and r['errorPixels']<=r['bodyScalePixels']*.05
        outcome=('pass' if passed(p) else 'fail')+'To'+('Pass' if passed(c) else 'Fail')
        if c['jointIndex']==8:
            counts['allLeftKnee'][outcome]+=1
            counts[lookup['rows'][key[:2]]['trial']][outcome]+=1
    assert hashes=={k:sha(v) for k,v in paths.items()}
    manifest={'scope':'post-evaluation interpretation only; no training or score changes',
              'selection':'Two largest visible wrist errors (one per side), two largest visible elbow errors (one per side), and two separated largest left-knee previous-pass/new-fail error increases.',
              'inputs':{k:{'path':str(v),'sha256':hashes[k]} for k,v in paths.items()},
              'inputHashesUnchanged':True,'leftKneeTransitions':dict(counts),'cases':output}
    (OUT/'evidence.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'cases':[r['case'] for r in output],'leftKneeTransitions':dict(counts)},indent=2))

if __name__=='__main__':main()
