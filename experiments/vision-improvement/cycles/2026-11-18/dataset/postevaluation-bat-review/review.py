"""Read-only interpretation of three BAT residual sequences in frozen swing 05."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import math
from PIL import Image, ImageDraw, ImageFont

OUT=Path(__file__).resolve().parent
CYCLE=OUT.parent.parent
VIDEO='345150fc7efb286c90983ddbf72e695126c3d37ad95506fcfca0f6380894ca5e'
CASES=[('short-projected-bat',24),('fast-motion-tip',79),('late-follow-through-misses',112)]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    paths={'rows':CYCLE/'dataset/splits/bat-evaluation-rows.json','protocol':CYCLE/'training/bat/protocol.json',
           'runner':CYCLE/'training/bat/runner-source.py'}
    for model in ['current','candidate']:
        paths[model]=CYCLE/f'training/bat/{model}-predictions.json'
        paths[model+'Score']=CYCLE/f'training/bat/{model}-primary-score.json'
    hashes={k:sha(p) for k,p in paths.items()}
    rows={r['frameIndex']:r for r in json.loads(paths['rows'].read_text()) if r['videoSHA256']==VIDEO}
    predictions={m:{r['frameIndex']:r for r in json.loads(paths[m].read_text()) if r['videoSHA256']==VIDEO} for m in ['current','candidate']}
    scores={m:json.loads(paths[m+'Score'].read_text()) for m in ['current','candidate']}
    details={m:{r['frameIndex']:r for r in scores[m]['details'] if r['videoSHA256']==VIDEO} for m in ['current','candidate']}
    font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',18)
    small=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',14)
    result=[]
    for name,center in CASES:
        ids=[center-1,center,center+1];allpts=[]
        for i in ids:
            allpts += [(p['pixelX'],p['pixelY']) for p in rows[i]['points'].values()]
            for m in predictions:
                allpts += predictions[m][i].get('prediction') or []
        crop=(max(0,math.floor(min(p[0] for p in allpts)-95)),max(0,math.floor(min(p[1] for p in allpts)-90)),
              min(1920,math.ceil(max(p[0] for p in allpts)+95)),min(1080,math.ceil(max(p[1] for p in allpts)+90)))
        scale=min(485/(crop[2]-crop[0]),330/(crop[3]-crop[1]));size=(round((crop[2]-crop[0])*scale),round((crop[3]-crop[1])*scale))
        image=Image.new('RGB',(1500,1245),'#121923');d=ImageDraw.Draw(image)
        d.text((10,10),f'BAT POST-EVALUATION REVIEW | {name} | Nov 5 11:58:01 | cam5',font=font,fill='white')
        d.text((10,40),'Top: source pixels. Middle: current + reference. Bottom: new + reference. Green=reference; orange=current; cyan=new.',font=font,fill='white')
        d.text((10,68),'Circle=TIP; square=KNOB. Predictions below the frozen 0.25 box/endpoint policy are not drawn as accepted detections.',font=font,fill='white')
        record={'case':name,'videoSHA256':VIDEO,'centerFrameIndex':center,'neighborIndices':ids,'crop':crop,'frames':[]}
        for col,i in enumerate(ids):
            r=rows[i];assert sha(r['imagePath'])==r['imageSHA256']
            im=Image.open(r['imagePath']).convert('RGB').crop(crop).resize(size,Image.Resampling.LANCZOS)
            left=col*500+(500-size[0])//2
            length=details['candidate'][i].get('truthLengthPixels')
            d.text((col*500+10,102),f'UI frame {i+1} | {r["frame"]["time"]:.3f}s | reference length {length:.1f}px',font=small,fill='white')
            fs={'frameIndex':i,'uiFrame':i+1,'time':r['frame']['time'],'imagePath':r['imagePath'],'imageSHA256':r['imageSHA256'],
                'points':r['points'],'reviewMethod':r.get('reviewMethod'),'referenceLengthPixels':length,'models':{}}
            for rownum,model,color in [(0,None,None),(1,'current','#ffb657'),(2,'candidate','#49d7ff')]:
                top=145+rownum*365;image.paste(im,(left,top))
                if model is None:continue
                prediction=predictions[model][i];detail=details[model][i]
                def loc(x,y):return left+(x-crop[0])*scale,top+(y-crop[1])*scale
                def draw_bat(points,color,width):
                    if all(p is not None for p in points):d.line([loc(*p) for p in points],fill=color,width=width)
                    for j,p in enumerate(points):
                        if p is None:continue
                        x,y=loc(*p)
                        if j==0:d.ellipse((x-5,y-5,x+5,y+5),outline=color,width=width)
                        else:d.rectangle((x-5,y-5,x+5,y+5),outline=color,width=width)
                draw_bat([(p['pixelX'],p['pixelY']) for p in r['points'].values()],'#7af49e',2)
                if detail['detected'] and prediction.get('prediction'):
                    draw_bat([p if c>=.25 else None for p,c in zip(prediction['prediction'],prediction['endpointConfidences'])],color,3)
                errors=detail.get('errorsPixels') or [None,None]
                errorstr='/'.join('miss' if e is None else f'{e:.1f}' for e in errors)
                score=prediction.get('score');scorestr='none' if score is None else f'{score:.3f}'
                label=f'{model.upper()}: {"detected" if detail["detected"] else "MISS"}; box conf {scorestr}; errors tip/knob {errorstr}px'
                d.text((col*500+7,top-23),label,font=small,fill=color)
                fs['models'][model]={'acceptedByFrozenPolicy':detail['detected'],'boxConfidence':score,'rawPrediction':prediction.get('prediction'),
                    'endpointConfidences':prediction.get('endpointConfidences'),'scoreDetail':detail}
            record['frames'].append(fs)
        p=OUT/(name+'.png');image.save(p);record.update(imagePath=str(p),imageSHA256=sha(p));result.append(record)
    transitions=Counter();changed=[]
    passed=lambda d:d['detected'] and all(e is not None and e<=.1*d['truthLengthPixels'] for e in d['errorsPixels'])
    for i,c in details['candidate'].items():
        if not c['bothTruthVisible']:continue
        p=details['current'][i];cp,pp=passed(c),passed(p)
        key=('pass' if pp else 'fail')+'To'+('Pass' if cp else 'Fail');transitions[key]+=1
        if cp!=pp:changed.append({'frameIndex':i,'uiFrame':i+1,'currentPass':pp,'candidatePass':cp})
    assert hashes=={k:sha(p) for k,p in paths.items()}
    detail_path=OUT/'short-projected-detail.png'
    Image.open(rows[24]['imagePath']).crop((640,150,840,340)).resize((800,760)).save(detail_path)
    report={'scope':'Post-evaluation interpretation only. Same frozen swing 05, source labels, confidence policy and scores.',
        'videoSHA256':VIDEO,'trial':'markers_2026-11-05_11-58-01','selectionRule':'Three diagnostic situations: short projected bat; largest localized tip error; late missed-detection cluster. Selected after scoring only to explain residuals.',
        'inputs':{k:{'path':str(p),'sha256':hashes[k]} for k,p in paths.items()},'inputHashesUnchanged':True,
        'clipScores':{m:scores[m]['byVideo'][VIDEO] for m in scores},'primaryTransitions':dict(transitions),'primaryChangedFrames':changed,'cases':result,
        'detailCrop':{'imagePath':str(detail_path),'imageSHA256':sha(detail_path),'sourceImagePath':rows[24]['imagePath'],
                      'sourceImageSHA256':rows[24]['imageSHA256'],'sourcePixelCrop':[640,150,840,340],'frameIndex':24}}
    (OUT/'evidence.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'sequences':len(result),'primaryTransitions':dict(transitions),'inputHashesUnchanged':True}))

if __name__=='__main__':main()
