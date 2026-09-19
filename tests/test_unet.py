"""Tiny-batch CPU checks of our residual U-Net and shared weighted loss."""

import unittest
from unittest.mock import Mock, patch

import torch

from losses.weighted_ce import make_weighted_cross_entropy
from models.unet_vgg import VGG16_FEATURE_INDICES, VGGResUNet


class TestVGGResUNet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_cpu_forward_weighted_loss_backward(self):
        torch.manual_seed(17)
        with patch.object(VGGResUNet, "load_vgg16_convolutions",
                          side_effect=AssertionError("Random initialization must not load pretrained weights")):
            model = VGGResUNet(use_random_init=True).to(device="cpu")
        sizes = []
        handles = [pool.register_forward_hook(lambda m, x, out: sizes.append(out.shape[-2:]))
                   for pool in model.pools]
        images = torch.randn(2, 3, 64, 64, device="cpu")
        targets = torch.randint(0, 5, (2, 64, 64), device="cpu")
        targets[:, :2, :2] = 255
        logits = model(images)
        for handle in handles:
            handle.remove()
        self.assertEqual(tuple(logits.shape), (2, 5, 64, 64))
        self.assertEqual([tuple(size) for size in sizes], [(32, 32), (16, 16), (8, 8), (4, 4), (2, 2)])
        loss = make_weighted_cross_entropy((1, 2, 3, 4, 5))(logits, targets)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for gradient in (model.encoder[0].convs[0].weight.grad, model.classifier.weight.grad):
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().sum()), 0)

    def test_injected_vgg16_weight_transfer_without_network(self):
        widths = (64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512)
        state = {}
        input_width = 3
        for index, output_width in zip(VGG16_FEATURE_INDICES, widths):
            state[f"features.{index}.weight"] = torch.full((output_width, input_width, 3, 3), 0.125)
            state[f"features.{index}.bias"] = torch.full((output_width,), 0.25)
            input_width = output_width
        loader = Mock(return_value=state)
        model = VGGResUNet(use_random_init=False, pretrained_loader=loader).to(device="cpu")
        loader.assert_called_once_with()
        for stage in model.encoder:
            for conv in stage.convs:
                self.assertTrue(torch.all(conv.weight == 0.125))
                self.assertTrue(torch.all(conv.bias == 0.25))

    def test_bad_pretraining_options_fail_closed(self):
        with self.assertRaises(ValueError):
            VGGResUNet(use_random_init=True, weights_path="missing.pt")


class TestWeightedCrossEntropy(unittest.TestCase):
    def test_invalid_class_weights(self):
        for weights in ((1, 1), (0, 1, 1, 1, 1), (-1, 1, 1, 1, 1),
                        (float("nan"), 1, 1, 1, 1), (float("inf"), 1, 1, 1, 1)):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                make_weighted_cross_entropy(weights)
        with self.assertRaises(ValueError):
            make_weighted_cross_entropy(ignore_index=0)

    def test_large_finite_weights_do_not_overflow(self):
        criterion = make_weighted_cross_entropy((1e300,) * 5)
        logits = torch.zeros((2, 5, 2, 2), device="cpu")
        targets = torch.zeros((2, 2, 2), dtype=torch.long, device="cpu")
        self.assertTrue(torch.isfinite(criterion(logits, targets)))


if __name__ == "__main__":
    unittest.main()
