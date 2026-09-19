"""Tiny-batch CPU checks of SegFormer-B0 and local encoder initialization."""

import unittest
from unittest.mock import Mock, patch

import torch
from transformers import SegformerConfig, SegformerModel

from losses.weighted_ce import make_weighted_cross_entropy
from models.segformer import SegFormerB0


class TestSegFormerB0(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_cpu_forward_weighted_loss_backward_offline(self):
        torch.manual_seed(19)
        with patch.object(SegformerModel, "from_pretrained", side_effect=AssertionError("Unexpected download")):
            model = SegFormerB0(use_random_init=True).to(device="cpu")
        images = torch.randn(2, 3, 64, 64, device="cpu")
        targets = torch.randint(0, 5, (2, 64, 64), device="cpu")
        targets[:, :2, :2] = 255
        logits = model(images)
        self.assertEqual(tuple(logits.shape), (2, 5, 64, 64))
        self.assertEqual(model.model.config.hidden_sizes, [32, 64, 160, 256])
        self.assertEqual(model.model.config.depths, [2, 2, 2, 2])
        loss = make_weighted_cross_entropy((1, 2, 3, 4, 5))(logits, targets)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        gradients = (model.model.segformer.encoder.patch_embeddings[0].proj.weight.grad,
                     model.model.decode_head.classifier.weight.grad)
        for gradient in gradients:
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().sum()), 0)

    def test_imagenet_encoder_injected_loader_is_local_only(self):
        encoder = SegformerModel(SegformerConfig(hidden_sizes=[32, 64, 160, 256]))
        loader = Mock(return_value=encoder)
        model = SegFormerB0(use_random_init=False, pretrained_loader=loader).to(device="cpu")
        loader.assert_called_once_with("nvidia/mit-b0", local_files_only=True)
        self.assertIs(model.model.segformer, encoder)
        self.assertEqual(model.model.decode_head.classifier.out_channels, 5)

    def test_reject_non_b0_encoder(self):
        encoder = SegformerModel(SegformerConfig(hidden_sizes=[64, 128, 320, 512]))
        with self.assertRaisesRegex(ValueError, "hidden_sizes"):
            SegFormerB0(use_random_init=False, pretrained_loader=Mock(return_value=encoder))


if __name__ == "__main__":
    unittest.main()

# Synthetic local weights exercise the real HF loader, never the Hub.
import pytest
from transformers import SegformerForImageClassification


@pytest.mark.parametrize('field,value', [
    ('hidden_act', 'relu'), ('hidden_dropout_prob', .2),
    ('attention_probs_dropout_prob', .2), ('drop_path_rate', .2),
])
def test_behavior_mismatch_is_rejected(field, value):
    encoder = SegformerModel(SegformerConfig(**{field: value})).to('cpu')
    with pytest.raises(ValueError, match=field):
        SegFormerB0(use_random_init=False, pretrained_loader=Mock(return_value=encoder))


def test_local_hf_weights_are_complete_and_resume_identical(tmp_path):
    torch.manual_seed(197)
    encoder = SegformerModel(SegformerConfig()).to('cpu')
    encoder.save_pretrained(tmp_path / 'complete')
    model = SegFormerB0(use_random_init=False, weights_path=tmp_path / 'complete').to('cpu')
    assert all(module.training for module in model.modules())
    restored = SegFormerB0().to('cpu')
    restored.load_state_dict(model.state_dict(), strict=True)
    model.eval(); restored.eval()
    inputs = torch.randn(1, 3, 35, 49, device='cpu')
    with torch.inference_mode():
        reference, actual = model(inputs), restored(inputs)
    assert reference.shape == (1, 5, 35, 49)
    # Serialized weights and non-tensor behavior match exactly. CPU float32
    # kernels may differ at rounding scale, so logits use a tight tolerance.
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
    torch.testing.assert_close(reference, actual, rtol=1e-5, atol=1e-7)
    assert torch.equal(reference.argmax(1), actual.argmax(1))

    state = encoder.state_dict()
    key = 'encoder.patch_embeddings.0.proj.weight'
    del state[key]
    encoder.save_pretrained(tmp_path / 'partial', state_dict=state)
    with pytest.raises(ValueError, match='Incomplete.*encoder weights'):
        SegFormerB0(use_random_init=False, weights_path=tmp_path / 'partial')
    torch.save(state, tmp_path / 'partial.pt')
    with pytest.raises(RuntimeError, match='Missing key'):
        SegFormerB0(use_random_init=False, weights_path=tmp_path / 'partial.pt')


def test_synthetic_imagenet_classifier_head_can_be_ignored(tmp_path):
    classifier = SegformerForImageClassification(SegformerConfig(num_labels=1000)).to('cpu')
    classifier.save_pretrained(tmp_path)
    model = SegFormerB0(use_random_init=False, weights_path=tmp_path).to('cpu')
    for key, value in classifier.segformer.state_dict().items():
        torch.testing.assert_close(model.model.segformer.state_dict()[key], value, rtol=0, atol=0)
    assert model.model.decode_head.classifier.out_channels == 5


@pytest.mark.parametrize('info', [
    {'mismatched_keys': [('encoder.weight', (3,), (5,))]},
    {'error_msgs': ['invalid checkpoint']}, {'unexpected_keys': ['encoder.typo']},
])
def test_other_hf_loading_errors_are_rejected(info):
    with pytest.raises(ValueError, match='encoder weights'):
        SegFormerB0._validate_loading_info(info)
