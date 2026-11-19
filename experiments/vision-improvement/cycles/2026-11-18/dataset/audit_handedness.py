"""Audit cached side/view metadata without mutating the frozen dataset."""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import json

ROOT=Path(__file__).resolve().parent
LIBRARY=Path('/Users/jramsdell/Documents/bitbucket/StrikeUI/react-apps-medical-wearable/bat-data-labeler.local/library')
APP=Path('/Users/jramsdell/Documents/bitbucket/react_apps/src/bat-data-labeler')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def counts(rows):
    videos={r['videoSHA256']:r for r in rows}
    count=lambda records:dict(Counter(r.get('handedness') or 'unknown' for r in records))
    return {'frames':count(rows),'videos':count(videos.values()),'totalFrames':len(rows),'totalVideos':len(videos)}

def main():
    input_paths=[ROOT/'private/catalog.json',ROOT/'private/queue.json',ROOT/'body-rows.json',ROOT/'bat-rows.json',LIBRARY/'inventory.json',LIBRARY/'discrete-objects.json']
    splits={}
    for task in ['body','bat']:
        for split in ['train','val','evaluation']:
            p=ROOT/'splits'/f'{task}-{split}-rows.json';input_paths.append(p);splits[(task,split)]=load(p)
    hashes={str(p):sha(p) for p in input_paths}
    cat=load(ROOT/'private/catalog.json');queue=load(ROOT/'private/queue.json')
    rows=load(ROOT/'body-rows.json');bat=load(ROOT/'bat-rows.json')
    catalog={r['id']:r for r in cat['videos']};local={r['id']:r for r in load(LIBRARY/'inventory.json')['videos']}
    videos={r['videoSHA256']:r for r in rows};wanted={r['trial'] for r in videos.values()}
    discrete=defaultdict(list);scanned=0
    live_discrete={o['name']:o for o in load(LIBRARY/'discrete-objects.json')}
    for p in sorted((LIBRARY/'discrete').glob('*.json')):
        data=load(p);scanned+=1
        if data.get('path') not in live_discrete:continue
        source_rows=data.get('rows',[])
        if isinstance(source_rows,dict):source_rows=source_rows.get('rows',source_rows.get('data',[]))
        for i,r in enumerate(source_rows):
            if not isinstance(r,dict):continue
            trial=r.get('trialName') or r.get('trial') or str(r.get('Sync_pitch_id','unknown'))
            if trial in wanted:
                raw=r.get('Hitter_Side');side={-1:'L',1:'R'}.get(raw)
                discrete[trial].append({'cacheFile':str(p),'cacheSHA256':sha(p),'jsonRowIndex':i,
                    'sourceDiscretePath':data.get('path'),'field':'Hitter_Side','rawValue':raw,
                    'mappedBattingSide':side,'playerId':r.get('playerId'),
                    'cachedObjectGeneration':live_discrete[data['path']].get('generation')})
    results=[]
    for video,r in sorted(videos.items(),key=lambda item:item[1]['trial']):
        row=catalog[r['assignmentId']];dpath=ROOT/'private/assignments'/f'{r["assignmentId"]}.json'
        doc=load(dpath);hashes[str(dpath)]=sha(dpath)
        stored=doc.get('row',{});clip=doc.get('clip') or {}
        sourcemeta=ROOT/'sources'/f'{video}.mp4.metadata.json';meta=load(sourcemeta);hashes[str(sourcemeta)]=sha(sourcemeta)
        hits=discrete[r['trial']];group=queue.get('reviews',{}).get('group:'+row['groupId'],{})
        membership=[split for (task,split),rr in splits.items() if task=='body' and any(x['videoSHA256']==video for x in rr)]
        assert len(membership)==1
        side_sources={
            'frozenRow':r.get('handedness'),'cloudCatalogSide':row.get('side'),
            'assignmentRowSide':stored.get('side'),'cachedOriginalInventorySide':local.get(row['id'],{}).get('side'),
            'assignmentClipSide':clip.get('side'),'assignmentClipHandedness':clip.get('handedness'),
        }
        direct=set(x['mappedBattingSide'] for x in hits if x['mappedBattingSide'])
        conflict=len(direct)>1 or any(s and s not in direct for s in side_sources.values()) if direct else False
        resolved=next(iter(direct)) if len(direct)==1 and not conflict else None
        if row.get('side'):
            assert resolved==row['side']
            assert any(h['sourceDiscretePath']==row.get('discretePath') for h in hits)
        results.append({'videoSHA256':video,'assignmentId':r['assignmentId'],'trial':r['trial'],'session':r['session'],
            'camera':r['camera'],'source':'Strike Mobile','split':membership[0],
            'bodyFrames':sum(x['videoSHA256']==video for x in rows),'batFrames':sum(x['videoSHA256']==video for x in bat),
            'sideSources':side_sources,'catalogDiscretePath':row.get('discretePath'),'cachedTrialMetadataMatches':hits,
            'authoritativeSidecarBattingSide':resolved,'newlyResolvedBeyondFrozenRows':bool(resolved and not r.get('handedness')),
            'metadataConflict':conflict,'reviewedView':group.get('view','unknown'),'viewReviewScope':'session-camera group',
            'sourceCustomMetadataHasSideOrView':any('side' in k.lower() or 'view' in k.lower() or 'hand' in k.lower() for k in meta.get('objectMetadata',{})),
            'visualAssessment':None})
    evidence_files=[APP/'scripts/prepare-gcs.py',APP/'services/cloud-library.js',APP/'services/cloud-workflow.js',
        APP/'utils/pose-data.js',APP/'utils/manual-skeleton.js',APP/'components/PoseWorkspace.jsx',APP/'components/VideoLibrary.jsx',
        Path('/Users/jramsdell/Documents/bitbucket/StrikeUI/react-apps-medical-wearable/src/strike/shared/services/swingslots/resolveSlotSide.js'),
        Path('/Users/jramsdell/.codex/worktrees/bat-data-labeler/react-apps-medical-wearable/src/bat-data-labeler/server/inventory.py')]
    code_evidence=[{'path':str(p),'sha256':sha(p)} for p in evidence_files]
    result={'scope':'Read-only sidecar audit of existing cached metadata. No frozen labels, groups, scores, or assignments changed.',
        'sideSemantics':{'field':'Hitter_Side','scope':'per-trial hitting record, joined by exact trialName; not athlete handedness preference',
            'mapping':{'-1':'L','1':'R'},'cloudField':'catalog.videos[].side',
            'evidence':'Original inventory.py lines 78–82 maps per-trial Hitter_Side and stores discretePath; lines 129–131 joins by trial. Each of the six tagged catalog rows independently matches one cached trial row and mapped Hitter_Side. Migration prepare-gcs.py copies the inventory rows.',
            'strictAuditMapping':'Only numeric -1 and 1 accepted; other values remain unknown.'},
        'summary':{'bodyAll':counts(rows),'batAll':counts(bat),
            'bySplit':{f'{t}-{s}':counts(rr) for (t,s),rr in splits.items()},
            'bySource':{'Strike Mobile':{'body':counts(rows),'bat':counts(bat)}},
            'cachedDiscreteFilesScanned':scanned,'knownTagsCorroborated':sum(bool(r['authoritativeSidecarBattingSide']) for r in results),
            'newlyResolvedTags':sum(r['newlyResolvedBeyondFrozenRows'] for r in results),'conflicts':sum(r['metadataConflict'] for r in results),
            'viewTagsByVideo':dict(Counter(r['reviewedView'] for r in results)),
            'cameraIdsByVideo':dict(Counter(r['camera'] for r in results)),
            'catalogScannedAt':cat.get('scannedAt'),
            'selectedTrialsWithMultipleCachedMetadataRows':sum(len(r['cachedTrialMetadataMatches'])>1 for r in results)},
        'labelerConvention':{'bodySides':'anatomical/person left-right, not screen-left/screen-right',
            'body17FlipIndex':[0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15],
            'body12ExportFlipIndex':[1,0,3,2,5,4,7,6,9,8,11,10], 'batOrder':['tip','knob'],'batFlipIndex':[0,1],
            'uiEvidence':'Manual guide says Use the person’s left and right. Body indices retain named COCO left/right slots.',
            'metadataCapture':'Read-only side display. Admin can set group/session-camera view unknown/open/back/side/reject; per-video keep/reject. No per-swing side editor or per-video view override found. newPose omits athlete ID and handedness, requiring a catalog join.'},
        'requiredFutureCollectionMetadata':[
            'Actual battingSide per swing: L/R/unknown, with metadata source, reviewer, timestamp, and confidence; keep athlete switch-hitter capability or throwing hand separate.',
            'Per-clip view relative to hitter at a named reference phase, with definitions for open/chest, back/rear, and side; retain physical camera orientation/calibration and session-camera ID. Camera number alone is insufficient.',
            'Source image mirrored flag and full orientation/crop/resize transform provenance; uncertainty must remain explicit.',
            'Stable pseudonymous athlete ID, physical-swing/synchronization ID across cameras, session, trial, and source video hash for grouping.',
            'Visibility versus estimated/unknown point labels, occlusion cause, phase/contact interval, blur, and review method for coverage summaries.'
        ],
        'interpretation':[
            'The training split has no source-metadata-confirmed left-handed swing, but seven training swings have unknown side; absence of confirmed lefties is not proof that all training swings are right-handed.',
            'One confirmed left-handed evaluation swing cannot establish left-handed population performance.',
            'A horizontal mirror changes image direction and swaps anatomical labels. It cannot reveal an occluded far-side limb or turn a rear view into an open/chest view.',
            'Keep mirrored copies and all camera views of the same physical swing in one split. Do not derive new benchmark membership from these sidecar tags.',
            'No per-video visual stance inference was added by this audit. The separate flip illustration is provisional evidence of preserved rear-view appearance, not an authoritative view tag.'
        ],
        'provenanceLimitations':[
            'The original scanner joins only on trialName (fallback trial, then Sync_pitch_id) and silently overwrites duplicate trial keys. The 17 selected trials contain no duplicate cached matches, but the join is not globally scoped by session/player/source hash.',
            'The original scanner only maps numeric -1 and 1. It does not normalize string values; missing/unrecognized values remain unknown.',
            'Cached metadata comes from the November 9 scan. No fresh cloud metadata fetch was performed here; unresolved does not prove that no newer authoritative metadata exists.',
            'Camera view is stored for source+session+camera, without batting side. A shared camera group may contain both batting sides and different relative views. Unknown view is eligible for labeling.',
            'No reliable per-clip mirror flag or numeric relative camera azimuth is present in the checked snapshot.'
        ],
        'videos':results,'inputHashes':[{'path':p,'sha256':h} for p,h in hashes.items()], 'codeEvidence':code_evidence,
        'flipRuntimeTest':str(ROOT/'horizontal-flip-check.json')}
    assert all(sha(p)==h for p,h in hashes.items())
    result['inputHashesUnchanged']=True
    (ROOT/'handedness-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['summary'],indent=2))

if __name__=='__main__':main()
