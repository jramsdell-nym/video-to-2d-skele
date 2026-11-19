import logging
from pathlib import Path
from types import SimpleNamespace
import unittest

from run_body import inference_warning_guard, predict_one


class FakeModel:
    def __init__(self, paths):
        self.paths = paths
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return iter(SimpleNamespace(path=path) for path in self.paths)


class BodyInferenceTests(unittest.TestCase):
    def test_scalar_source_and_exactly_one_matching_result(self):
        image = Path('/tmp/body-frame.png')
        model = FakeModel([image])
        result = predict_one(model, image)
        self.assertEqual(result.path, image)
        self.assertEqual(model.calls[0]['source'], str(image))
        self.assertEqual(model.calls[0]['batch'], 1)
        self.assertTrue(model.calls[0]['stream'])

    def test_missing_extra_and_misidentified_results_fail(self):
        image = Path('/tmp/body-frame.png')
        for paths in ([], [image, image], ['/tmp/other-frame.png']):
            with self.subTest(paths=paths), self.assertRaises(ValueError):
                predict_one(FakeModel(paths), image)

    def test_nms_timeout_warning_invalidates_inference_and_detaches_guard(self):
        logger = logging.Logger('isolated-body-test')
        warnings = []
        with self.assertRaisesRegex(RuntimeError, 'NMS warning'):
            with inference_warning_guard(logger, warnings):
                logger.warning('NMS time limit %.3fs exceeded', 0.55)
        self.assertEqual(warnings, ['NMS time limit 0.550s exceeded'])
        self.assertEqual(logger.handlers, [])

    def test_other_warning_is_preserved_without_failing(self):
        logger = logging.Logger('isolated-body-test')
        warnings = []
        with inference_warning_guard(logger, warnings):
            logger.warning('Example runtime warning')
        self.assertEqual(warnings, ['Example runtime warning'])
        self.assertEqual(logger.handlers, [])


if __name__ == '__main__':
    unittest.main()
