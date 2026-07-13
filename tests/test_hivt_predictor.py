import unittest

from prediction.hivt_predictor import HiVTPredictor


class _FakeModel:

    def __init__(self):
        self.eval_called = False
        self.device = None

    def eval(self):
        self.eval_called = True

    def to(self, device):
        self.device = device

    def __call__(self, data):
        mock_scores = 'mock_scores'
        return data, mock_scores


class _BadFakeModel:

    def __call__(self, _):
        invalid_output = 'invalid_output'
        return invalid_output


class HiVTPredictorTest(unittest.TestCase):

    def test_init_uses_loader_and_sets_eval_and_device(self):
        calls = {}

        def loader(**kwargs):
            calls.update(kwargs)
            return _FakeModel()

        predictor = HiVTPredictor('model.ckpt', parallel=False, device='cpu', model_loader=loader)

        self.assertEqual(calls['checkpoint_path'], 'model.ckpt')
        self.assertFalse(calls['parallel'])
        self.assertTrue(predictor.model.eval_called)
        self.assertEqual(predictor.model.device, 'cpu')

    def test_predict_returns_output_tuple(self):
        predictor = HiVTPredictor('model.ckpt', model_loader=lambda **_: _FakeModel())
        self.assertEqual(predictor.predict('input'), ('input', 'mock_scores'))

    def test_predict_raises_when_model_output_is_invalid(self):
        predictor = HiVTPredictor('model.ckpt', model_loader=lambda **_: _BadFakeModel())
        with self.assertRaises(ValueError):
            predictor.predict('input')

    def test_rejects_empty_checkpoint_path(self):
        with self.assertRaises(ValueError):
            HiVTPredictor('', model_loader=lambda **_: _FakeModel())


if __name__ == '__main__':
    unittest.main()
