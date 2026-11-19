"""Source-label QA: deterministic first/middle/last reviewed frame per video.

No inference and no coordinate editing. Visible body = green, visible bat =
pink, hidden-estimated points = amber, unlabelled points are not plotted.
"""
from collections import defaultdict
from pathlib import Path
import json
import math

from PIL import Image, ImageDraw, ImageFont
from snapshot import ROOT, sha, save

EDGES = [(0,1),(0,2),(2,4),(1,3),(3,5),(0,6),(1,7),(6,7),(6,8),(8,10),(7,9),(9,11)]
GREEN, PINK, AMBER = '#72f0b2', '#ff72c9', '#ffbd55'
BACKGROUND = '#121923'


def font(size):
    return ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', size)


def render_panel(row, bat):
    path = Path(row['imagePath'])
    if sha(path) != row['imageSHA256']:
        raise ValueError('Source image changed before QA render')
    image = Image.open(path).convert('RGB').resize((640,360), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image)
    points = [(p['x']*640, p['y']*360) if p['visibility'] else None for p in row['points']]
    for a,b in EDGES:
        if points[a] and points[b]:
            color = GREEN if row['points'][a]['visibility']==row['points'][b]['visibility']==2 else AMBER
            draw.line([points[a],points[b]],fill=color,width=2)
    for p,xy in zip(row['points'],points):
        if xy:
            x,y=xy
            color = GREEN if p['visibility']==2 else AMBER
            if p['visibility']==2: draw.ellipse((x-3,y-3,x+3,y+3),fill=color,outline='black',width=1)
            else: draw.ellipse((x-4,y-4,x+4,y+4),outline=color,width=2)
    if bat:
        endpoints = bat['points']
        locations = {key:(p['x']*640,p['y']*360) for key,p in endpoints.items()}
        color = PINK if all(p['visibility']==2 for p in endpoints.values()) else AMBER
        draw.line([locations['tip'],locations['knob']],fill=color,width=3)
        for key,p in endpoints.items():
            x,y=locations[key];color=PINK if p['visibility']==2 else AMBER
            draw.ellipse((x-4,y-4,x+4,y+4),outline=color,width=2)
            draw.text((x+6,y-8),'T' if key=='tip' else 'K',font=font(14),fill=color,stroke_width=2,stroke_fill='black')
    draw.rectangle((0,332,640,360),fill=BACKGROUND)
    draw.text((8,337),f"Frame {row['frameIndex']+1} | source {row['frame']['time']:.3f}s | " + (row.get('reviewMethod') or 'legacy review'),font=font(16),fill='white')
    if not bat:
        draw.text((370,8),'No accepted bat pair',font=font(16),fill=AMBER,stroke_width=2,stroke_fill='black')
    return image


def main():
    rows = json.loads((ROOT/'body-rows.json').read_text())
    bat_rows = json.loads((ROOT/'bat-rows.json').read_text())
    bats = {(r['videoSHA256'],r['frameIndex']):r for r in bat_rows}
    groups = defaultdict(list)
    for row in rows: groups[row['videoSHA256']].append(row)
    groups = sorted(groups.values(),key=lambda g:(g[0]['session'],g[0]['trial']))
    selected=[]
    for group in groups:
        group.sort(key=lambda r:r['frameIndex'])
        selected.append([group[0],group[len(group)//2],group[-1]])
    target=ROOT/'source-label-qa';target.mkdir(exist_ok=True)
    pages=[];manifest=[]
    for start in range(0,len(selected),6):
        chunk=selected[start:start+6]
        page=Image.new('RGB',(1920,110+400*len(chunk)),BACKGROUND)
        draw=ImageDraw.Draw(page)
        draw.text((18,14),'SOURCE LABELS: first / middle / last reviewed frame of each swing',font=font(26),fill='white')
        draw.text((18,52),'Green: visible body    Pink: visible bat    Amber rings/lines: hidden estimated    T: tip    K: knob',font=font(20),fill='white')
        draw.text((18,80),'Reviewed references include interpolation and model assistance. This is label QA, not model output.',font=font(17),fill='#c5cdd9')
        for n,triplet in enumerate(chunk):
            first=triplet[0];top=110+n*400
            title=f"{first['session']} | {first['trial']} | {first['camera']}"
            draw.text((12,top+6),title,font=font(18),fill='white')
            for j,row in enumerate(triplet):
                page.paste(render_panel(row,bats.get((row['videoSHA256'],row['frameIndex']))),(j*640,top+36))
                manifest.append({'page':start//6+1,'row':n+1,'column':j+1,'videoSHA256':row['videoSHA256'],'frameIndex':row['frameIndex'],'imageSHA256':row['imageSHA256']})
        path=target/f'labels-page-{start//6+1}.png';page.save(path)
        pages.append({'path':str(path),'sha256':sha(path),'width':page.width,'height':page.height})
    save(target/'selection.json',{'selection':'First, median-index and last reviewed source frame per video. No prediction-based selection.','panels':manifest,'pages':pages})
    print(json.dumps({'pages':pages,'videos':len(groups),'panels':len(manifest)},indent=2))


if __name__=='__main__': main()
