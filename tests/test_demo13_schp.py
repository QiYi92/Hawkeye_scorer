"""SCHP inference compatibility and actionable startup validation."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from demo13.schp_compat import InPlaceABNSync, load_schp_modules
from run_demo13 import arguments


class SCHPTests(unittest.TestCase):
    def test_normalization_matches_upstream_kernel_including_negative_gamma(self):
        for activation in ("none", "relu", "leaky_relu", "elu"):
            layer = InPlaceABNSync(2, eps=.001, activation=activation).eval()
            layer.load_state_dict({"weight": torch.tensor([-2., 3.]), "bias": torch.tensor([.2, -.3]),
                                   "running_mean": torch.tensor([.5, -.5]), "running_var": torch.tensor([2., 4.])})
            x = torch.tensor([[[[-1., 2.]], [[3., -4.]]]])
            shape = (1, 2, 1, 1)
            expected = (x - layer.running_mean.reshape(shape)) / torch.sqrt(layer.running_var.reshape(shape) + layer.eps)
            expected = expected * (layer.weight.abs().reshape(shape) + layer.eps) + layer.bias.reshape(shape)
            if activation == "relu":
                expected = expected.clamp_min(0)
            elif activation == "leaky_relu":
                expected = torch.where(expected >= 0, expected, expected * .01)
            elif activation == "elu":
                expected = torch.where(expected >= 0, expected, expected.exp() - 1)
            torch.testing.assert_close(layer(x), expected)
        with self.assertRaisesRegex(RuntimeError, "inference-only"):
            layer.train()(x)

    def test_loading_isolated_upstream_modules_does_not_import_cuda_extension(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "networks").mkdir()
            (root / "utils").mkdir()
            (root / "networks/AugmentCE2P.py").write_text("from modules import InPlaceABNSync\n")
            (root / "utils/transforms.py").write_text("marker = 13\n")
            old_path = sys.path.copy()
            previous = {k: sys.modules.get(k) for k in ("modules", "networks", "utils")}
            network, transforms = load_schp_modules(root)
            self.assertIs(network.InPlaceABNSync, InPlaceABNSync)
            self.assertEqual(transforms.marker, 13)
            self.assertEqual(sys.path, old_path)
            self.assertEqual({k: sys.modules.get(k) for k in previous}, previous)
            self.assertEqual((root / "networks/AugmentCE2P.py").read_text(), "from modules import InPlaceABNSync\n")

    def test_missing_schp_assets_report_both_entry_configuration_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "input.mp4"
            video.touch()
            error = io.StringIO()
            with contextlib.redirect_stderr(error), self.assertRaises(SystemExit) as raised:
                arguments(["--video", str(video), "--parts", "schp"])
            self.assertEqual(raised.exception.code, 2)
            for message in ("腿脚权重路径", "SCHP仓库路径", "exp-schp-201908261155-lip.pth"):
                self.assertIn(message, error.getvalue())


if __name__ == "__main__":
    unittest.main()
