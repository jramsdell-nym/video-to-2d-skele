"""Render six preselected full-frame comparisons; no inference or model tuning."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIRST = ROOT.parent / "session-holdout-20261111"
FOLLOWUP = ROOT.parents[2] / "training/body-conservative-20261111"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT/"matplotlib"))
EDGES = ((0,1),(0,2),(2,4),(1,3),(3,5),(0,6),(1,7),(6,7),(6,8),(8,10),(7,9),(9,11))
TRUTH_COLOR = "#77ffb5"
MODEL_COLOR = "#ffac55"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def key(row):
    return row['videoSHA256'],row['frameIndex']


def lookup(path):
    rows=json.loads(path.read_text())
    if len(set(map(key,rows)))!=len(rows):
        raise ValueError(f"Duplicate frame identities in {path}")
    return {key(r):r for r in rows}


def finite(value):
    return isinstance(value,(float,int)) and not isinstance(value,bool) and math.isfinite(value)


def visible_model_points(pred,width,height):
    if not pred.get('detected'):
        return [None]*12
    points,confs=pred.get('prediction'),pred.get('keypointConfidences')
    if len(points)!=12 or len(confs)!=12:
        raise ValueError('Expected twelve ordered model joints')
    return [p if isinstance(p,list) and len(p)==2 and all(finite(v) for v in p)
            and finite(c) and c>=.3 and 0<=p[0]<=width and 0<=p[1]<=height else None
            for p,c in zip(points,confs)]


def draw_skeleton(ax,points,color,truth=False):
    for a,b in EDGES:
        if points[a] is not None and points[b] is not None:
            ax.plot([points[a][0],points[b][0]],[points[a][1],points[b][1]],
                    color=color,linewidth=1.0 if truth else 1.5,
                    linestyle='--' if truth else '-',alpha=.8 if truth else .95)
    xs=[p[0] for p in points if p is not None]
    ys=[p[1] for p in points if p is not None]
    ax.scatter(xs,ys,s=12 if truth else 10,facecolors='none' if truth else color,
               edgecolors=color if truth else '#171b20',linewidths=.9 if truth else .35,zorder=5)


def panel(ax,image,truth,model,title,status):
    height,width=image.height,image.width
    ax.imshow(image)
    draw_skeleton(ax,truth,TRUTH_COLOR,truth=True)
    if model is not None:
        draw_skeleton(ax,model,MODEL_COLOR)
    ax.set_xlim(0,width);ax.set_ylim(height,0)
    ax.set_xticks([]);ax.set_yticks([])
    for spine in ax.spines.values():spine.set_color('#46525f')
    ax.set_title(title,color='white',fontsize=10,pad=7,loc='left')
    ax.text(.015,.985,status,transform=ax.transAxes,ha='left',va='top',fontsize=8,
            color='white',bbox={'facecolor':'#141b22','alpha':.85,'edgecolor':'none','pad':4})


def render(run,selection_path,out,check_only=False):
    selection=json.loads(selection_path.read_text())
    if selection.get('candidatePredictionsPresentAtSelection'):
        raise ValueError('Frame selection must predate candidate prediction inspection')
    if sha(run/'val-rows.json')!=selection['validationRowsSHA256']:
        raise ValueError('Reference rows changed after frame selection')
    paths={'Generic YOLO11n':FIRST/'generic-val-predictions.json',
           'Conservative YOLO11n · 17 slots / 20 epochs':run/'candidate-val-predictions.json',
           'Mac Vision · single-person policy':FIRST/'mac-vision-cpu/predictions.json'}
    missing=[str(p) for p in paths.values() if not p.is_file()]
    if check_only:
        print(json.dumps({'ready':not missing,'selectedFrames':len(selection['frames']),'missing':missing}))
        return
    if missing:
        raise FileNotFoundError('Wait for frozen predictions: '+', '.join(missing))
    if out.exists():
        raise ValueError('Refusing to overwrite a rendered comparison')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    rows=lookup(run/'val-rows.json')
    predictions={title:lookup(path) for title,path in paths.items()}
    fig,axes=plt.subplots(len(selection['frames']),4,figsize=(20,19.8),dpi=160,facecolor='#111820')
    details=[]
    for index,selected in enumerate(selection['frames']):
        identity=key(selected);row=rows[identity]
        image_path=run/'dataset'/selected['image']
        if sha(image_path)!=selected['imageSHA256']:
            raise ValueError('Selected source PNG changed')
        with Image.open(image_path) as source:
            image=source.convert('RGB')
        truth=[(p['x']*row['width'],p['y']*row['height']) if p['visibility']==2 else None for p in row['points']]
        sources=Counter(p.get('source','unspecified') for p in row['points'])
        label=f"Video {identity[0][:8]} · frame {identity[1]+1} · {row['frame']['time']:.3f}s"
        panel(axes[index,0],image,truth,None,label,
              f"Reviewed reference · {row.get('reviewMethod','legacy review')}\n{sources.get('manual',0)} manual / {sources.get('interpolated',0)} interpolated / {sources.get('unspecified',0)} unspecified")
        frame_detail={'videoSHA256':identity[0],'frameIndex':identity[1],'models':{}}
        for column,(title,items) in enumerate(predictions.items(),1):
            if identity not in items:raise ValueError(f'Missing prediction identity {identity}')
            pred=items[identity]
            points=visible_model_points(pred,row['width'],row['height'])
            usable=sum(p is not None for p in points)
            candidates=pred.get('candidateCount',len(pred.get('candidates',[])))
            if pred.get('detected'):
                status=f"{usable}/12 usable joints · {candidates} candidate(s)"
            elif title.startswith('Mac Vision') and candidates>1:
                status=f"MISSING BY POLICY · {candidates} people"
            else:status='MISSING · no accepted person'
            panel(axes[index,column],image,truth,points,title,status)
            frame_detail['models'][title]={'usableJoints':usable,'candidateCount':candidates,'detected':bool(pred.get('detected'))}
        details.append(frame_detail)
    fig.suptitle('Body follow-up · same six frames fixed before both candidates',
                 x=.02,y=.995,ha='left',fontsize=19,color='white',fontweight='bold')
    fig.text(.02,.975,'Green dashed / open points: reviewed reference    Orange solid / filled points: model    All panels show the same full source image',
             ha='left',fontsize=11,color='#c4d2df')
    fig.text(.02,.012,'Frame selection: first, middle, last from each heldout video. Model joints require confidence ≥0.30 and in-image coordinates.\n'
             'References include interpolation and bulk review. These frames support visual inspection; they do not replace the complete 240-frame scorecard. Mac Vision is not iPhone parity.',
             ha='left',fontsize=10,color='#c4d2df')
    fig.subplots_adjust(left=.015,right=.99,top=.947,bottom=.05,wspace=.025,hspace=.22)
    out.mkdir(parents=True)
    target=out/'body-six-frame-comparison.png'
    fig.savefig(target,facecolor=fig.get_facecolor())
    plt.close(fig)
    receipt={'createdAt':datetime.now(timezone.utc).isoformat(),'selectionSHA256':sha(selection_path),
             'sourceSHA256':sha(__file__),'outputSHA256':sha(target),
             'predictionsSHA256':{title:sha(path) for title,path in paths.items()},
             'coordinateSpace':'original full-image pixels, unmirrored, no crop',
             'gate':'detected plus finite in-image points and keypoint confidence >=0.30',
             'selectedFrames':details,'inferencePerformed':False}
    (out/'render-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'image':str(target),'frames':len(details),'panels':len(details)*4}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=FOLLOWUP)
    parser.add_argument('--selection',type=Path,default=ROOT.parent/'comparison-selection.json')
    parser.add_argument('--out',type=Path,default=ROOT/'visual-review')
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    render(args.run.resolve(),args.selection.resolve(),args.out.resolve(),args.check_only)
