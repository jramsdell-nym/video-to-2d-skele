"""Read-only visual triage of training labels; no model inference or relabeling."""
from pathlib import Path
import sys
import json
import math
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from snapshot import sha, save

EDGES = [(0,1),(0,2),(2,4),(1,3),(3,5),(0,6),(1,7),(6,7),(6,8),(8,10),(7,9),(9,11)]
JOINTS = ['LS','RS','LE','RE','LW','RW','LH','RH','LK','RK','LA','RA']
CASES = [('right-wrist-jump','f8892b',115),('left-shoulder-jump','f8892b',97),
         ('hidden-elbow-jump','4a5943',16),('left-wrist-jump','2aa9de',31),
         ('bat-tip-abrupt-motion','4a5943',453),('bat-motion-oct27','8f1faf',80),
         ('bat-motion-oct30','d4c602',81),('foreshortened-bat','f8892b',1)]


def main():
    body_path,bat_path = ROOT/'splits/body-train-rows.json',ROOT/'splits/bat-train-rows.json'
    body=json.loads(body_path.read_text());bat=json.loads(bat_path.read_text())
    lookup={(r['videoSHA256'],r['frameIndex']):r for r in body}
    bats={(r['videoSHA256'],r['frameIndex']):r for r in bat}
    font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',18)
    small=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',13)
    output=[]
    for name,prefix,index in CASES:
        h=next(r['videoSHA256'] for r in body if r['videoSHA256'].startswith(prefix))
        rows=[lookup[(h,k)] for k in (max(0,index-1),index,index+1) if (h,k) in lookup]
        xmin=min(r['box'][0]*r['width'] for r in rows)-90;xmax=max(r['box'][2]*r['width'] for r in rows)+90
        ymin=min(r['box'][1]*r['height'] for r in rows)-100;ymax=max(r['box'][3]*r['height'] for r in rows)+70
        for r in rows:
            for p in bats.get((h,r['frameIndex']),{}).get('points',{}).values():
                xmin=min(xmin,p['x']*r['width']-30);xmax=max(xmax,p['x']*r['width']+30)
                ymin=min(ymin,p['y']*r['height']-30);ymax=max(ymax,p['y']*r['height']+30)
        crop=(max(0,math.floor(xmin)),max(0,math.floor(ymin)),min(rows[0]['width'],math.ceil(xmax)),min(rows[0]['height'],math.ceil(ymax)))
        scale=min(500/(crop[2]-crop[0]),450/(crop[3]-crop[1]))
        size=(round((crop[2]-crop[0])*scale),round((crop[3]-crop[1])*scale))
        image=Image.new('RGB',(1500,980),'#121923');d=ImageDraw.Draw(image)
        d.text((12,10),f"TRAINING LABEL TRIAGE | {name} | {rows[0]['trial']}",font=font,fill='white')
        d.text((12,38),'Upper row: original pixels. Lower: reviewed body and bat. Green=visible body; pink=bat; amber=hidden estimate.',font=font,fill='white')
        for col,r in enumerate(rows):
            source=Path(r['imagePath'])
            if sha(source)!=r['imageSHA256']: raise ValueError('Source image hash changed')
            original=Image.open(source).convert('RGB').crop(crop).resize(size,Image.Resampling.LANCZOS)
            left=col*500+(500-size[0])//2;top=80
            image.paste(original,(left,top));image.paste(original,(left,top+450))
            d.text((col*500+8,62),f"Frame {r['frameIndex']+1} | {r['frame']['time']:.3f}s",font=small,fill='white')
            def loc(p):return(left+(p['x']*r['width']-crop[0])*scale,top+450+(p['y']*r['height']-crop[1])*scale)
            for a,b in EDGES:
                pa,pb=r['points'][a],r['points'][b]
                if pa['visibility'] and pb['visibility']:
                    d.line([loc(pa),loc(pb)],fill='#72f0b2' if pa['visibility']==pb['visibility']==2 else '#ffbd55',width=2)
            for j,p in enumerate(r['points']):
                if p['visibility']:
                    x,y=loc(p);color='#72f0b2' if p['visibility']==2 else '#ffbd55'
                    d.ellipse((x-4,y-4,x+4,y+4),outline=color,width=2)
                    d.text((x+5,y-8),JOINTS[j],font=small,fill=color,stroke_width=1,stroke_fill='black')
            b=bats.get((h,r['frameIndex']))
            if b:
                d.line([loc(b['points']['tip']),loc(b['points']['knob'])],fill='#ff72c9',width=2)
                for n,p in b['points'].items():
                    x,y=loc(p);color='#ff72c9' if p['visibility']==2 else '#ffbd55'
                    d.ellipse((x-4,y-4,x+4,y+4),outline=color,width=2)
                    d.text((x+5,y-8),n.upper(),font=small,fill=color,stroke_width=1,stroke_fill='black')
        path=ROOT/'quality-triage'/(name+'.png');image.save(path)
        output.append({'case':name,'videoSHA256':h,'frameIndex':index,'neighborIndices':[r['frameIndex'] for r in rows], 'imagePath':str(path),'crop':crop,'imageSHA256':sha(path)})
    save(ROOT/'quality-triage/visual-selection.json',{'scope':'training only','bodyTrainSHA256':sha(body_path),'batTrainSHA256':sha(bat_path),'cases':output})
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
