import tempfile
import unittest
from pathlib import Path

from batting.tracking import HitterSelector
from batting.video import model_specs, point, validate_paths


def pose(left, top, size):
    return {str(i): {'usable': True, 'normalized': xy} for i, xy in enumerate([
        [left, top], [left + size, top], [left, top + size],
        [left + size, top + size], [left + size / 2, top + size / 2]])}


class VideoContractTests(unittest.TestCase):
    def test_release_selection_preserves_mixed_defaults(self):
        previous = {'body':'older-body','bat':'older-bat'}
        retrained = {'body':'newer-body','bat':'newer-bat'}
        manifest = {'release':'r2', 'defaults':{'body':'older-body','bat':'newer-bat'},
                    'releases': {'r1':previous, 'r2':retrained}}
        self.assertEqual(model_specs(manifest, None), manifest['defaults'])
        self.assertEqual(model_specs(manifest, 'r2'), retrained)
        self.assertEqual(model_specs(manifest, 'r1'), previous)
        with self.assertRaisesRegex(ValueError, 'Available releases'):
            model_specs(manifest, '../untracked-model')

    def test_nonfinite_is_missing_and_json_safe(self):
        p = point(float('nan'), 10, .9, 100, 100, .3)
        self.assertFalse(p['usable'])
        self.assertIsNone(p['x'])
        self.assertIsNone(p['normalized'])

    def test_edge_coordinates_and_low_confidence(self):
        self.assertTrue(point(0, 0, .9, 100, 100, .3)['usable'])
        self.assertFalse(point(100, 0, .9, 100, 100, .3)['usable'])
        self.assertFalse(point(10, 10, .29, 100, 100, .3)['usable'])
        self.assertFalse(point(-1, 10, .9, 100, 100, .3)['usable'])

    def test_selector_does_not_jump_to_distant_spectator(self):
        selector = HitterSelector()
        self.assertEqual(selector.choose([pose(.1, .1, .2)]), (0, 'observed'))
        self.assertEqual(selector.choose([pose(.7, .7, .2)]), (None, 'missing'))
        self.assertEqual(selector.choose([]), (None, 'missing'))
        self.assertEqual(selector.choose([pose(.11, .1, .2)]), (0, 'observed'))

    def test_ambiguous_people_are_missing(self):
        selector = HitterSelector()
        selector.choose([pose(.1, .1, .3)])
        self.assertEqual(selector.choose([pose(.1, .1, .3), pose(.1, .1, .3)]), (None, 'ambiguous'))

    def test_initialization_uses_largest_visible_body(self):
        self.assertEqual(HitterSelector().choose([pose(.1, .1, .2), pose(.5, .5, .4)]), (1, 'observed'))

    def test_refuses_input_existing_output_and_output_collision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video, model = root / 'input.mp4', root / 'model.pt'
            video.touch()
            model.touch()
            with self.assertRaises(ValueError):
                validate_paths(video, video, None, [model])
            output = root / 'output.jsonl'
            with self.assertRaises(ValueError):
                validate_paths(video, output, output, [model])
            output.touch()
            with self.assertRaises(ValueError):
                validate_paths(video, output, None, [model])


if __name__ == '__main__':
    unittest.main()
