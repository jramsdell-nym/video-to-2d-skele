"""Interim body-only rendering of the first frozen full swing, with no inference."""
from fractions import Fraction
import json
from pathlib import Path

import av
from PIL import Image, ImageDraw, ImageFont

from render_comparisons import (annotated_panel, dump, fit_source, load_predictions,
                                sha256, verified_file, visible_points, FONT_PATH,
                                BG, TEXT, MUTED)

ROOT = Path(__file__).resolve().parent
SIZE = (1920, 700)


def main():
    output = ROOT/'body-preview'
    output.mkdir(exist_ok=True)
    target = output/'swing-01-body-generic-vs-new-quarter-speed.mp4'
    if target.exists():
        raise FileExistsError('Refusing to replace preview')
    selection_path = ROOT/'materialized/materialized-selection.json'
    selection = json.loads(selection_path.read_text())
    clip = selection['clips'][0]
    assert clip['slug'] == 'swing-01' and clip['frameCount'] == 120
    source = verified_file(ROOT, clip['source'])
    frame_path = verified_file(ROOT, clip['frames'])
    expected = json.loads(frame_path.read_text())
    pixel_audit = json.loads((ROOT/'full-pixel-alignment-audit.json').read_text())
    assert pixel_audit['everyPixelMatchesIndependentVideoExtraction'] and pixel_audit['framesCompared'] == 601
    predictions, provenance = {}, {}
    for name in ('generic','candidate'):
        path = ROOT.parent/'training/body'/f'{name}-predictions.json'
        manifest_path = path.with_name(f'{name}-prediction-manifest.json')
        manifest = json.loads(manifest_path.read_text())
        assert sha256(manifest['checkpoint']) == manifest['checkpointSHA256']
        assert sha256(path) == manifest['predictionsSHA256']
        assert sha256(manifest['rowsPath']) == manifest['rowsSHA256'] == pixel_audit['evaluationRowsSHA256']['body']
        predictions[name] = load_predictions(path)
        provenance[name] = {'predictions': str(path), 'predictionSHA256': sha256(path),
                            'manifest': str(manifest_path), 'manifestSHA256': sha256(manifest_path),
                            'weights':manifest['checkpoint'],'weightsSHA256':manifest['checkpointSHA256']}
    times, coverage = [], []
    large = ImageFont.truetype(FONT_PATH, 29)
    normal = ImageFont.truetype(FONT_PATH, 25)
    small = ImageFont.truetype(FONT_PATH, 21)
    with av.open(str(source)) as decoder:
        source_stream = decoder.streams.video[0]
        tb = source_stream.time_base
        with av.open(str(target), 'w', options={'movflags':'+faststart'}) as encoder:
            stream = encoder.add_stream('libx264',rate=source_stream.average_rate/4)
            stream.width,stream.height=SIZE
            stream.pix_fmt='yuv420p'
            stream.time_base=tb
            stream.codec_context.time_base=tb
            stream.options={'crf':'19','preset':'fast','bf':'0'}
            for index,frame in enumerate(decoder.decode(video=0)):
                assert index<len(expected)
                assert frame.pts==expected[index]['pts'] and str(frame.time_base)==expected[index]['timeBase']
                assert (frame.width,frame.height)==(clip['width'],clip['height'])
                if index==0:first_pts=frame.pts
                pts=(frame.pts-first_pts)*4
                times.append(Fraction(pts)*frame.time_base)
                image=Image.new('RGB',SIZE,BG)
                base,scale,offset=fit_source(frame.to_image())
                draw=ImageDraw.Draw(image)
                draw.text((22,7),'BODY ONLY  |  Frozen swing 1  |  Generic YOLO vs new candidate',font=large,fill=TEXT)
                shown={}
                for column,name in enumerate(('generic','candidate')):
                    prediction=predictions[name][(clip['source']['sha256'],index)]
                    if prediction.get('score',0)<.25:prediction={'detected':False}
                    points=visible_points(prediction,12,(frame.width,frame.height),.3,'keypointConfidences')
                    shown[name]=sum(p is not None for p in points)
                    x=column*960
                    image.paste(annotated_panel(base,points,scale,offset,'body'),(x,88))
                    label='BASELINE  |  Generic YOLO11n' if column==0 else 'NEW  |  Fine-tuned body candidate'
                    draw.text((x+22,50),label,font=normal,fill=TEXT)
                    draw.text((x+938,52),f'{shown[name]}/12 points',font=small,fill=MUTED,anchor='ra')
                draw.line((960,88,960,628),fill=BG,width=4)
                draw.text((22,640),f'Complete 120-frame clip  |  Frame {index+1}/120  |  Source {float(frame.pts*frame.time_base):.3f}s  |  Quarter speed  |  Orange: body joints',font=small,fill=MUTED)
                draw.text((22,671),'Interim body-only preview. BAT training is still in progress. 2D only; no smoothing or gap filling.',font=small,fill=MUTED)
                encoded=av.VideoFrame.from_image(image)
                encoded.pts,encoded.time_base=pts,frame.time_base
                for packet in stream.encode(encoded):encoder.mux(packet)
                coverage.append({'frameIndex':index,'sourcePTS':frame.pts,'sourceTimeBase':str(frame.time_base),'visiblePoints':shown})
            for packet in stream.encode():encoder.mux(packet)
    assert len(times)==120
    decoded_times,posters=[],[]
    with av.open(str(target)) as decoder:
        stream=decoder.streams.video[0]
        assert (stream.width,stream.height)==SIZE
        assert stream.codec_context.name=='h264' and stream.codec_context.format.name=='yuv420p'
        for index,frame in enumerate(decoder.decode(video=0)):
            decoded_times.append(Fraction(frame.pts)*frame.time_base)
            if index in (0,59,119):
                poster=output/f'body-preview-frame-{index:04d}.jpg'
                frame.to_image().save(poster,quality=94)
                posters.append({'frameIndex':index,'path':str(poster),'sha256':sha256(poster)})
        duration=float(stream.duration*stream.time_base)
    assert len(decoded_times)==120
    assert all(abs(a-b)<Fraction(1,100000) for a,b in zip(decoded_times,times))
    assert abs(duration-8)<.00001
    receipt={'output':str(target),'outputSHA256':sha256(target),'source':clip['source'],
             'selection':str(selection_path),'selectionSHA256':sha256(selection_path),
             'sourceFrames':str(frame_path),'sourceFramesSHA256':sha256(frame_path),
             'rendererSHA256':sha256(Path(__file__)),'drawingHelpersSHA256':sha256(ROOT/'render_comparisons.py'),
             'models':provenance,'frameCount':120,'durationSeconds':duration,'size':list(SIZE),
             'allSourceFramesIncluded':True,'allOutputFramesDecoded':True,'allOutputTimestampsVerified':True,
             'inferencePerformed':False,'smoothingOrGapFillingPerformed':False,'scope':'Interim body-only preview; BAT not shown',
             'posters':posters,'coverage':coverage}
    dump(output/'body-preview-receipt.json',receipt)
    print(json.dumps({'output':str(target),'frameCount':120,'durationSeconds':duration,'validated':True}),flush=True)


if __name__=='__main__':
    main()
