"""Offline tests (no dataset or pretrained weights needed): python -m unittest discover -s tests"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from explanations import centered_prior, gradcam, patch_scores  # noqa: E402
from models import build_model  # noqa: E402
from sanity import reinitialize, stages  # noqa: E402


def toy_image(seed=0, size=160):
    rng = np.random.default_rng(seed)
    gray = (rng.random((size, size)) * 120 + 60).astype("uint8")
    gray[60:100, 50:110] = 230  # a bright "nodule" so tiles have different effects
    return gray, np.ones((size, size), bool)


class Explanations(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.models = {n: build_model(n).eval() for n in ("mobilenet", "resnet18")}

    def test_default_is_unchanged_gradcam_and_cam_variants_are_valid(self):
        gray, _ = toy_image()
        for name, net in self.models.items():
            logit, plain = gradcam(net, name, gray, "cpu")
            logit2, explicit = gradcam(net, name, gray, "cpu", "gradcam")
            _, plus = gradcam(net, name, gray, "cpu", "gradcampp")
            self.assertEqual(plain.shape, (160, 160))
            np.testing.assert_array_equal(plain, explicit)
            self.assertAlmostEqual(logit, logit2, places=6)
            self.assertTrue((plus >= 0).all() and np.isfinite(plus).all())
        with self.assertRaises(ValueError):
            gradcam(self.models["mobilenet"], "mobilenet", gray, "cpu", "bogus")

    def test_patch_scores_null_rate_and_backward_compatible_keys(self):
        gray, content = toy_image()
        for name, net in self.models.items():
            logit, heat = gradcam(net, name, gray, "cpu")
            r = patch_scores(net, gray, content, heat, centered_prior(content), logit, "cpu")
            for key in ("gap", "hot_drop", "control_drop", "center_gap", "eligible_tiles", "reason"):
                self.assertIn(key, r)
            if r["reason"] == "ok":
                self.assertTrue(0.0 <= r["null_positive_rate"] <= 1.0)
                self.assertEqual(r["eligible_tiles"], 16)

    def test_null_rate_formula(self):
        # d_j - mean(others) > 0  <=>  d_j > mean(all)
        e = np.array([3.0, 1.0, 0.0, -1.0, 2.0])
        direct = np.mean([(e[j] - np.delete(e, j).mean()) > 0 for j in range(len(e))])
        self.assertAlmostEqual(direct, float((e > e.mean() + 1e-12).mean()))


class Sanity(unittest.TestCase):
    def test_reinitialization_changes_only_requested_stages(self):
        for name in ("mobilenet", "resnet18"):
            net = build_model(name)
            groups = stages(net, name)
            before = {k: v.clone() for k, v in net.state_dict().items()}
            reinitialize(groups[0][1], 1)
            changed = [k for k, v in net.state_dict().items() if not torch.equal(v, before[k])]
            self.assertTrue(changed)
            head = "classifier" if name == "mobilenet" else "fc"
            self.assertTrue(all(k.startswith(head) for k in changed), changed)
            names = {p for g in groups for p in [g[0]]}
            self.assertGreaterEqual(len(names), 5)


class Aggregation(unittest.TestCase):
    def test_summary_files_are_parsed(self):
        from aggregate_seeds import rows_from
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "summary.json"
            f.write_text(json.dumps(dict(architecture="resnet18", phase="internal",
                                         classification=dict(auroc=.9, sensitivity=.8, specificity=.7),
                                         primary=dict(fraction_with_positive_gap=.66, mean_gap=.2))))
            row, = rows_from(f)
            self.assertEqual(row["model"], "resnet18")
            self.assertIsNone(row["null_positive_rate"])


if __name__ == "__main__":
    unittest.main()
