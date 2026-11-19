"""Meaningful geometry, missing-point and encoded-time checks for the renderer."""
from fractions import Fraction
import json
from pathlib import Path
import tempfile
import unittest

import av
from PIL import Image

import render_comparisons as render
from materialize_selection import materialize


class RenderingChecks(unittest.TestCase):
    def test_missing_low_confidence_and_invalid_points_stay_missing(self):
        row = {'detected': True, 'prediction': [[0, 0], [20, 30], [50, 50], [100, 20], [float('nan'), 20]],
               'keypointConfidences': [.9, .29, .3, .9, .9]}
        actual = render.visible_points(row, 5, (100, 100), .3, 'keypointConfidences')
        self.assertEqual(actual, [(0, 0), None, (50, 50), None, None])
        self.assertEqual(render.visible_points({'detected': False}, 2, (100, 100), .3, ''), [None, None])

    def test_entire_portrait_source_is_fitted_without_crop(self):
        source = Image.new('RGB', (600, 1000), 'white')
        panel, scale, offset = render.fit_source(source)
        self.assertEqual(panel.size, (960, 540))
        self.assertEqual(scale, (.54, .54))
        self.assertEqual(offset, (318, 0))
        self.assertEqual(panel.getpixel((318, 0)), (255, 255, 255))
        self.assertEqual(panel.getpixel((641, 539)), (255, 255, 255))
        marked = render.annotated_panel(panel, [(300, 500), None], scale, offset, 'bat')
        self.assertEqual(marked.getpixel((480, 270)), (255, 206, 105))

    def test_variable_source_gaps_survive_quarter_speed_encoding(self):
        with tempfile.TemporaryDirectory(prefix='renderer-check-', dir=Path(__file__).parent) as temp:
            output = Path(temp)
            target = output / 'timing.mp4'
            source_pts, tb = [20, 30, 50], Fraction(1, 600)
            expected = [Fraction((pts-source_pts[0])*4)*tb for pts in source_pts]
            with av.open(str(target), 'w', options={'movflags': '+faststart'}) as container:
                stream = container.add_stream('libx264', rate=15)
                stream.width, stream.height = render.SIZE
                stream.pix_fmt = 'yuv420p'
                stream.time_base = tb
                stream.codec_context.time_base = tb
                stream.options = {'crf': '32', 'preset': 'ultrafast', 'bf': '0'}
                for pts in source_pts:
                    render.encode_frame(container, stream, Image.new('RGB', render.SIZE, '#123456'),
                                        (pts-source_pts[0])*4, tb)
                for packet in stream.encode():
                    container.mux(packet)
            validation = render.validate_video(target, expected, [1], output)
            self.assertEqual(validation['frameCount'], 3)
            self.assertEqual(validation['size'], [1920, 1280])
            self.assertTrue(validation['allTimestampsVerified'])

    def test_materialization_preserves_frame_identity_and_hashes(self):
        with tempfile.TemporaryDirectory(prefix='source-check-', dir=Path(__file__).parent) as temp:
            directory = Path(temp).resolve()
            source = directory / 'source.mp4'
            with av.open(str(source), 'w') as container:
                stream = container.add_stream('libx264', rate=60)
                stream.width, stream.height = 64, 32
                stream.pix_fmt = 'yuv420p'
                stream.time_base = Fraction(1, 60)
                stream.codec_context.time_base = stream.time_base
                stream.options = {'bf': '0'}
                for index in range(3):
                    render.encode_frame(container, stream, Image.new('RGB', (64, 32), '#123456'), index, Fraction(1, 60))
                for packet in stream.encode():
                    container.mux(packet)
            frozen = directory / 'selection.json'
            render.dump(frozen, {'selectionPolicy': 'Synthetic test only', 'clips': [
                {'slug': 'source-check', 'title': 'Synthetic test only',
                 'source': {'path': str(source), 'sha256': render.sha256(source)},
                 'width': 64, 'height': 32, 'frameCount': 3}]})
            receipt = materialize(frozen, directory / 'materialized')
            rows_path = Path(receipt['inferenceRows'])
            rows = json.loads(rows_path.read_text())
            self.assertEqual(receipt['inferenceRowsSHA256'], render.sha256(rows_path))
            self.assertEqual([row['frameIndex'] for row in rows], [0, 1, 2])
            self.assertEqual(len({row['videoSHA256'] for row in rows}), 1)
            for index, row in enumerate(rows):
                self.assertEqual(row['imageSHA256'], render.sha256(row['imagePath']))
                self.assertAlmostEqual(row['sourceFrame']['time'], index/60)


if __name__ == '__main__':
    unittest.main()
