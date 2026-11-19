"""Build a self-contained local review page from verified exports and results.

No network, JavaScript, cloud hosting, autoplay, or external assets are used.
Media stays beside the generated HTML; it is referenced with relative paths.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import os
from pathlib import Path
from urllib.parse import quote

from render_comparisons import dump, sha256


CSS = """
:root{color-scheme:dark;--bg:#0b1219;--card:#141f2a;--ink:#f3f6f9;--muted:#a8bdcf;--line:#304253;--good:#9fe0ba;--bad:#ffb8a2;--blue:#82c9f0}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}main{max-width:1320px;margin:auto;padding:44px 28px 70px}h1,h2,h3,p{margin-top:0}h1{font-size:clamp(32px,4vw,54px);letter-spacing:-.045em;line-height:1.08;margin-bottom:14px}h2{font-size:28px;letter-spacing:-.025em;margin-bottom:12px}h3{font-size:20px;line-height:1.3;margin-bottom:8px}p{margin-bottom:14px}.eyebrow{color:var(--blue);font-weight:750;font-size:12px;letter-spacing:.15em;margin-bottom:14px}.lead{max-width:850px;font-size:19px;color:var(--muted)}.muted,.note{color:var(--muted)}.note{font-size:14px}.pill{display:inline-block;border:1px solid var(--line);border-radius:99px;padding:5px 12px;font-size:13px;color:var(--muted);margin:0 7px 7px 0}nav{display:flex;gap:8px;flex-wrap:wrap;margin:24px 0 32px}a{color:var(--blue);text-underline-offset:3px}nav a,.button{border:1px solid var(--line);border-radius:8px;padding:8px 14px;text-decoration:none;font-weight:650}a:hover{color:#d6efff;border-color:#7898b3}a:focus-visible,video:focus-visible{outline:3px solid var(--blue);outline-offset:4px}.metrics{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px;margin:20px 0}.metric,.panel,.swing{background:var(--card);border:1px solid var(--line);border-radius:14px}.metric{padding:22px}.metric-label{font-size:14px;font-weight:700;color:var(--muted)}.metric-value{font-size:clamp(24px,3.1vw,38px);font-weight:750;letter-spacing:-.045em;white-space:nowrap;margin:6px 0}.arrow{color:#6b8295;font-size:.7em;vertical-align:middle;padding:0 6px}.delta{font-size:14px;font-weight:700}.up{color:var(--good)}.down{color:var(--bad)}.flat{color:var(--muted)}.panel{padding:24px;margin:28px 0}.split-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:15px;margin:8px 0 14px}th,td{text-align:right;padding:12px;border-bottom:1px solid var(--line);white-space:nowrap}th:first-child,td:first-child{text-align:left}th{font-size:13px;color:var(--muted);font-weight:700}.details{display:grid;grid-template-columns:1fr 1fr;gap:26px}.details p{margin-bottom:8px}.legend{display:flex;gap:16px;flex-wrap:wrap;margin:16px 0;font-size:14px;color:var(--muted)}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px}.body-dot{background:#ffac55}.tip-dot{background:#ffce69}.knob-dot{background:#74cefa}.swing{margin:26px 0;overflow:hidden;scroll-margin-top:16px}.swing-head{padding:22px 24px 16px;display:flex;justify-content:space-between;gap:20px;align-items:start}.swing-head p{margin-bottom:0}.swing-number{font-size:12px;letter-spacing:.12em;color:var(--blue);font-weight:750;margin-bottom:8px}.swing video{display:block;width:100%;height:auto;aspect-ratio:3 / 2;background:#030609}.video-foot{padding:18px 24px 22px;display:flex;justify-content:space-between;align-items:center;gap:18px;flex-wrap:wrap}.clip-metrics{display:flex;gap:26px;flex-wrap:wrap;font-size:14px}.clip-metrics strong{display:block;margin-bottom:2px}.limits{padding-left:20px;color:var(--muted);font-size:14px}.limits li{margin-bottom:7px}footer{font-size:13px;color:var(--muted);margin-top:38px;border-top:1px solid var(--line);padding-top:20px}footer a{margin-right:20px}@media(max-width:720px){main{padding:26px 14px 46px}.metrics,.details{grid-template-columns:1fr}.metric{padding:18px}.metric-value{font-size:36px}.swing-head{padding:18px;display:block}.swing-head .pill{margin-top:10px}.video-foot{padding:16px 18px}.panel{padding:18px}th,td{padding:10px 8px}nav{margin-top:20px}}
"""


def escape(value):
    return html.escape(str(value), quote=True)


def fraction(value):
    if value is None:
        return None
    if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('Metric must be a finite proportion between zero and one')
    return value


def percent(value):
    value = fraction(value)
    return 'Unavailable' if value is None else f'{100*value:.1f}%'


def delta(before, after):
    before, after = fraction(before), fraction(after)
    if before is None or after is None:
        return '<span class="delta flat">Change unavailable</span>'
    difference = 100 * (after-before)
    if abs(difference) < .05:
        return '<span class="delta flat">No change at displayed precision</span>'
    color = 'up' if difference > 0 else 'down'
    return f'<span class="delta {color}">{difference:+.1f} percentage points</span>'


def metric(label, before, after, baseline):
    return f'''<article class="metric"><div class="metric-label">{escape(label)}</div>
<div class="metric-value">{percent(before)} <span class="arrow" aria-label="to">→</span> {percent(after)}</div>
{delta(before, after)}<p class="note" style="margin:7px 0 0">{escape(baseline)} → new candidate</p></article>'''


def relative_url(path, directory):
    return quote(os.path.relpath(Path(path).resolve(), directory.resolve()), safe='/')


def build(export_path, results_path, output):
    export_path, results_path, output = (Path(p).resolve() for p in (export_path, results_path, output))
    export = json.loads(export_path.read_text())
    result = json.loads(results_path.read_text())
    if output.exists():
        raise FileExistsError(f'Refusing to replace published gallery: {output}')
    clips = export['clips']
    if len(clips) != 5 or len(result['perVideo']) != 5:
        raise ValueError('This review requires the five frozen complete clips')
    per_video = {row['slug']: row for row in result['perVideo']}
    if len(per_video) != 5:
        raise ValueError('Duplicate result slugs')
    assets, cards = [], []
    for index, clip in enumerate(clips, 1):
        score = per_video[clip['slug']]
        if clip['source']['sha256'] != score['videoSHA256']:
            raise ValueError('Video/results source identity mismatch')
        validation = clip['validation']
        if not validation['allFramesDecoded'] or not validation['allTimestampsVerified']:
            raise ValueError('Video has not passed complete decoding/timestamp validation')
        if validation['frameCount'] != score['sourceFrames']:
            raise ValueError('Video/results frame count mismatch')
        video = Path(clip['output'])
        if sha256(video) != clip['outputSHA256']:
            raise ValueError('Exported video changed after validation')
        posters = validation['posters']
        poster = min(posters, key=lambda row: abs(row['frameIndex']-(validation['frameCount']-1)/2))
        if sha256(poster['path']) != poster['sha256']:
            raise ValueError('Poster changed after validation')
        video_url, poster_url = (relative_url(p, output.parent) for p in (video, poster['path']))
        duration = validation['durationSeconds']
        assets.append({'video': str(video), 'sha256': clip['outputSHA256'],
                       'poster': poster['path'], 'posterSHA256': poster['sha256']})
        cards.append(f'''<section class="swing" id="swing-{index}" aria-labelledby="swing-title-{index}">
<div class="swing-head"><div><div class="swing-number">COMPLETE SWING {index} OF 5</div>
<h3 id="swing-title-{index}">{escape(clip['title'])}</h3>
<p class="note">Same source frame in every panel · {validation['frameCount']:,} frames · {duration:.1f}s at quarter speed</p></div>
<span class="pill">Session held out</span></div>
<video controls playsinline preload="metadata" poster="{poster_url}" aria-label="Before and after 2D model comparison for swing {index}">
<source src="{video_url}" type="video/mp4">Your browser cannot play this video. <a href="{video_url}">Open the MP4</a>.</video>
<div class="video-foot"><div class="clip-metrics"><div><strong>Body: {percent(score['body']['generic'])} → {percent(score['body']['candidate'])}</strong>
{delta(score['body']['generic'],score['body']['candidate'])}</div><div><strong>Both bat endpoints: {percent(score['bat']['current'])} → {percent(score['bat']['candidate'])}</strong>
{delta(score['bat']['current'],score['bat']['candidate'])}</div></div><a class="button" href="{video_url}">Open MP4</a></div></section>''')
    dataset, splits = result['dataset'], result['split']
    allocation = []
    for task, title, current_key in [('body','Body','currentBodyFrames'),('bat','BAT','currentBatFrames')]:
        task_split = splits[task]
        cells = ''.join(f'<td>{task_split[name]["frames"]:,}<span class="note"> / {task_split[name]["videos"]} clips</span></td>' for name in ('train','val','evaluation'))
        allocation.append(f'<tr><th scope="row">{title}</th>{cells}<td>{dataset[current_key]:,}</td></tr>')
    body, bat = result['body'], result['bat']
    bat_reference_count = bat['candidate'].get('pairedReferenceFrames')
    bat_reference_note = (f' The BAT score uses {bat_reference_count:,} evaluable reference frames; all {sum(row["sourceFrames"] for row in result["perVideo"]):,} source frames are shown in the videos.'
                          if bat_reference_count is not None else '')
    cards_html = '\n'.join(cards)
    overview_metrics = ''.join([
        metric('Body joint agreement',body['generic']['bodyPCK'],body['candidate']['bodyPCK'],'Generic YOLO'),
        metric('Wrist agreement',body['generic']['wristsPCK'],body['candidate']['wristsPCK'],'Generic YOLO'),
        metric('Both bat endpoints agree',bat['current']['pairedEndpointsPCK'],bat['candidate']['pairedEndpointsPCK'],'Current BAT model')])
    body_history = ''.join(f'<tr><th scope="row">{label}</th><td>{percent(body[name]["bodyPCK"])}</td><td>{percent(body[name]["wristsPCK"])}</td></tr>'
                           for name,label in [('generic','Generic YOLO · video baseline'),('previous','Prior candidate · November 11'),('candidate','New candidate · November 18')])
    triage_path = results_path.parent/'dataset/quality-triage/LABEL_REVIEW.txt'
    triage = (f'<section class="panel"><h2>Next labeling pass</h2><p>Five training sequences are flagged for human review, with two wrist sequences highest priority. These are potential inconsistencies to inspect.</p><a class="button" href="{relative_url(triage_path,output.parent)}">Five label sequences to review next</a></section>'
              if triage_path.is_file() else '')
    handedness_path = results_path.parent/'dataset/HANDEDNESS_AUDIT.txt'
    reflection_path = results_path.parent/'dataset/horizontal-flip-check.png'
    body_review_path = results_path.parent/'dataset/postevaluation-body-review/REVIEW.txt'
    bat_review_path = results_path.parent/'dataset/postevaluation-bat-review/REVIEW.txt'
    side_scores_path = results_path.parent/'training/handedness-scores.json'
    side_scores = ''
    if side_scores_path.is_file():
        side_data = json.loads(side_scores_path.read_text())
        if side_data.get('schema') != 'catalog-handedness-slices-v1':
            raise ValueError('Unrecognized handedness score schema')
        side_rows = []
        for side, label in [('L','Left'),('R','Right'),('unknown','Unknown')]:
            models = side_data['tasks']['body']['models']
            prior, new = models['previous']['slices'][side], models['candidate']['slices'][side]
            bat_models = side_data['tasks']['bat']['models']
            old_bat, new_bat = bat_models['current']['slices'][side], bat_models['candidate']['slices'][side]
            side_rows.append(f'<tr><th scope="row">{label}</th><td>{new["videos"]} / {new["frames"]}</td>'
                             f'<td>{percent(prior["jointPCK0.05BodyScale"])} → {percent(new["jointPCK0.05BodyScale"])}</td>'
                             f'<td>{percent(prior["wrists"]["jointPCK0.05BodyScale"])} → {percent(new["wrists"]["jointPCK0.05BodyScale"])}</td>'
                             f'<td>{percent(old_bat["bothEndpointsPCK0.1BatLength"])} → {percent(new_bat["bothEndpointsPCK0.1BatLength"])}</td></tr>')
        side_scores = f'''<h3>Preliminary agreement by catalog-labeled batting side</h3><div class="split-wrap"><table>
<thead><tr><th scope="col">Side</th><th scope="col">Clips / frames</th><th scope="col">Body: prior → new</th><th scope="col">Wrists: prior → new</th><th scope="col">BAT: current → new</th></tr></thead>
<tbody>{''.join(side_rows)}</tbody></table></div><p class="note">Groups use saved catalog metadata and were not independently classified from the images for this report. Only one metadata-confirmed left-handed clip and two right-handed clips were evaluated; two clips have unknown side. Frames within a clip are correlated. These slices do not establish broad performance by batting side. “Prior” is the November 11 body candidate.</p>
<p><a href="{relative_url(side_scores_path,output.parent)}">Full batting-side score report</a></p>'''
    handedness = ''
    bat_review = (f'<p><a href="{relative_url(bat_review_path,output.parent)}">Review of remaining BAT errors</a></p>'
                  '<p class="note">Swing 5 has nine recovered frames and nine new failures, leaving its paired-endpoint score unchanged. '
                  'Before adding more labels, review the physical terminal-knob convention: the knob target should represent the bat end, '
                  'with separate visibility or uncertainty flags when gloves hide it. The residual review identifies a possible grip-versus-knob '
                  'reference issue, plus visible tip underreach during fast motion and follow-through. These findings need human adjudication; '
                  'the completed evaluation has not been relabeled.</p>'
                  if bat_review_path.is_file() else '')
    if all(path.is_file() for path in (handedness_path,reflection_path,body_review_path)):
        handedness = f'''<section class="panel"><h2>Batting side, view, and reflection</h2>
<p>Both training recipes used a 50% horizontal-reflection probability. Reflected body labels swap anatomical left/right pairs; bat tip and knob keep their identities. The installed transform passed the recorded reflection checks.</p>
<p class="note">Eleven of the 17 clips have unknown batting side, and all 17 lack a verified saved camera-view label. A mirrored rear view remains a rear view; reflection cannot reveal occluded points or replace independently captured views.</p>
{side_scores}<p><a href="{relative_url(handedness_path,output.parent)}">Handedness and camera-view audit</a> · <a href="{relative_url(body_review_path,output.parent)}">Review of remaining body errors</a></p>
{bat_review}
<details><summary><a href="{relative_url(reflection_path,output.parent)}">Checked horizontal-reflection example</a></summary>
<img src="{relative_url(reflection_path,output.parent)}" alt="Original and horizontally reflected training example with checked body and bat keypoint identities" loading="lazy" style="display:block;width:100%;height:auto;margin-top:16px;border-radius:8px"></details></section>'''
    limits = ''.join(f'<li>{escape(line)}</li>' for line in result['limitations'])
    nav = ''.join(f'<a href="#swing-{index}">Swing {index}</a>' for index in range(1,6))
    prior_body, current_body = dataset['priorBodyFrames'],dataset['currentBodyFrames']
    prior_bat, current_bat = dataset['priorBatFrames'],dataset['currentBatFrames']
    document = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fan Lab · November 18 model review</title><style>{CSS}</style></head><body><main>
<header><div class="eyebrow">FAN LAB / MODEL REVIEW / NOVEMBER 18, 2026</div>
<h1>Watch the new models.</h1><p class="lead">Five complete swings, with the same source frames shown before and after training. Review body joints and both bat endpoints side by side.</p>
<div><span class="pill">Quarter speed</span><span class="pill">2D detectors only</span><span class="pill">Five swings · three held-out sessions</span><span class="pill">No smoothing or gap filling</span></div></header>
<nav aria-label="Jump to a swing"><a href="#overview">Scores &amp; data</a>{nav}</nav>
<section id="overview" aria-labelledby="overview-heading"><h2 id="overview-heading">Measured agreement on these five swings</h2>
<p class="muted">Higher is better. These numbers measure agreement with reviewed labels, which include interpolated and model-assisted points.</p>
<div class="metrics">{overview_metrics}</div>
<p class="note">Body and wrists: within 5% of the reference body scale. Bat: both endpoints within 10% of reference bat length in the same frame. Missing predictions count as failures.{bat_reference_note}</p>
<h3>All three body models on this same holdout</h3><div class="split-wrap"><table><thead><tr><th scope="col">Body model</th><th scope="col">Body joint agreement</th><th scope="col">Wrist agreement</th></tr></thead><tbody>{body_history}</tbody></table></div>
<p class="note">The videos retain the frozen generic-versus-new comparison. The prior candidate is shown here to make progress since the last training cycle visible.</p></section>
<section class="panel" aria-labelledby="data-heading"><h2 id="data-heading">What went into this run</h2>
<p>Reviewed body frames grew from <strong>{prior_body:,} to {current_body:,}</strong> ({current_body/prior_body:.2f}×).
Eligible BAT frames grew from <strong>{prior_bat:,} to {current_bat:,}</strong> ({current_bat/prior_bat:.2f}×).</p>
<div class="split-wrap"><table><thead><tr><th scope="col">Model</th><th scope="col">Training</th><th scope="col">Validation</th><th scope="col">This review</th><th scope="col">Total frames</th></tr></thead>
<tbody>{''.join(allocation)}</tbody></table></div><p class="note">Whole recording sessions are separated between training, validation, and this review. Only the training allocation updates the model weights. Athlete independence is not established.</p>
<div class="details"><div><h3>Reading the four panels</h3><p class="note">Top: generic YOLO body model on the left; new body candidate on the right. Bottom: current BAT model on the left; new BAT candidate on the right.</p></div>
<div><h3>What to watch</h3><p class="note">Follow the wrists and elbows through the fast part of the swing, then compare the bat tip and knob. Gaps, jumps, and wrong-person selections remain visible.</p></div></div>
<div class="legend"><span><i class="dot body-dot"></i>Body joints</span><span><i class="dot tip-dot"></i>Barrel tip</span><span><i class="dot knob-dot"></i>Handle knob</span></div></section>
{cards_html}
<section class="panel" aria-labelledby="limits-heading"><h2 id="limits-heading">How to interpret the result</h2><ul class="limits">{limits}</ul></section>
{handedness}
{triage}
<footer><p>This local review contains the full frozen five-clip selection. Playback starts only when you press play.</p>
<a href="{relative_url(results_path,output.parent)}">Result data</a><a href="{relative_url(export_path,output.parent)}">Video provenance &amp; validation</a>
</footer></main></body></html>'''
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document)
    receipt = {'gallery':str(output),'gallerySHA256':sha256(output),'generatorSHA256':sha256(Path(__file__)),
               'exportManifest':str(export_path),'exportManifestSHA256':sha256(export_path),
               'results':str(results_path),'resultsSHA256':sha256(results_path),'verifiedAssets':assets,
               'externalDependencies':False,'autoplay':False,'clipCount':len(clips)}
    dump(output.with_suffix('.receipt.json'),receipt)
    return receipt


if __name__ == '__main__':
    root=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exports',type=Path,default=root/'exports/export-manifest.json')
    parser.add_argument('--results',type=Path,default=root.parent/'results.json')
    parser.add_argument('--output',type=Path,default=root/'index.html')
    args=parser.parse_args()
    print(json.dumps(build(args.exports,args.results,args.output)),flush=True)
