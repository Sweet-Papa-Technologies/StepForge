"""Regenerate tests/fixtures/neural/tiny_model/ — a randomly initialised,
few-thousand-parameter StepForge Flow export. It charts nonsense; it exists so
the runtime tests can exercise the real ONNX path on machines without PyTorch.

    PYTHONPATH=tools python3 tools/beatforge/tests/fixtures/neural/make_tiny_model.py

Re-run after any change to the feature layout, the codec version or the model
input/output signature (the runtime refuses a model whose meta disagrees).
"""
from pathlib import Path

import torch

from beatforge.adapters.stepmania.neural.train import export_onnx as E
from beatforge.adapters.stepmania.neural.train.choreo_net import ChoreoConfig, ChoreoNet
from beatforge.adapters.stepmania.neural.train.rhythm_net import RhythmConfig, RhythmNet

OUT = Path(__file__).parent / "tiny_model"


def main():
    torch.manual_seed(20261113)
    a = RhythmNet(RhythmConfig(d_model=16, n_layers=1, n_heads=2, dropout=0.0))
    b = ChoreoNet(ChoreoConfig(d_model=16, n_layers=1, n_heads=2, d_audio=8, dropout=0.0))
    for model in (a, b):
        for p in model.parameters():
            torch.nn.init.normal_(p, std=0.25)
    # Rescale the random onset head so its probabilities straddle the decoder's
    # threshold range on ordinary input; otherwise every test chart would sit at
    # the "almost no notes" floor and exercise nothing.
    import numpy as np
    from beatforge.adapters.stepmania.neural import conditioning as cond
    from beatforge.adapters.stepmania.neural import grid_features as GF
    rng = np.random.default_rng(1)
    y = np.zeros(int(30 * GF.SR), np.float32)
    for beat in range(140):
        k = int((0.2 + beat * 0.2) * GF.SR)
        y[k:k + 60] += rng.standard_normal(60).astype(np.float32)
    g = GF.extract_grid_features(y, GF.TimingMap.constant(150.0, 0.2))
    x = torch.from_numpy(g.X)[None]
    c = torch.from_numpy(cond.cond19(cond.tier_only_vector("hard")))[None]
    with torch.no_grad():
        z = a(x, c)[0]
        scale = 2.5 / float(z.std())
        a.onset.weight.mul_(scale)
        a.onset.bias.copy_((a.onset.bias - z.mean()) * scale - 0.3)
    meta = E.export_models(a, b, OUT, metrics={"note": "random init test fixture"},
                           extra={"fixture": True})
    print(OUT, meta["stage_a"]["params"], meta["stage_b"]["params"])


if __name__ == "__main__":
    main()
