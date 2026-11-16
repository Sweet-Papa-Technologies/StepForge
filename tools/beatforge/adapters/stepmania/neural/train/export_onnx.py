"""export_onnx.py — export RhythmNet and ChoreoNet to ONNX and prove the
exported graphs compute what PyTorch computes (spec §7, REQ-SFN-06).

    python -m beatforge.adapters.stepmania.neural.train.export_onnx \
        --stage-a build/neural_runs/stage_a/best.pt \
        --stage-b build/neural_runs/stage_b/best.pt \
        --out models/stepforge_flow [--int8]

Writes `stage_a_rhythm.onnx` (dynamic batch, ticks), `stage_b_choreo.onnx`
(dynamic batch, events) and `model_meta.json`. The meta file is written LAST
and only after both graphs match PyTorch to 1e-4 at sequence lengths other
than the one they were traced at — its existence is what tells the runtime a
model is installed, so it must never describe a graph that was not verified.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .. import codec as C
from .. import conditioning as cond
from .. import event_features as EF
from .. import grid_features as GF
from .choreo_net import INPUT_NAMES as B_INPUTS
from .choreo_net import ChoreoConfig, ChoreoNet
from .rhythm_net import OUTPUT_NAMES as A_OUTPUTS
from .rhythm_net import RhythmConfig, RhythmNet

OPSET = 17
STAGE_A_FILE = "stage_a_rhythm.onnx"
STAGE_B_FILE = "stage_b_choreo.onnx"
META_FILE = "model_meta.json"
PARITY_ATOL = 1e-4


def _export(model, args, path, input_names, output_names, dynamic_axes) -> None:
    model.eval()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(model, args, str(path), input_names=list(input_names),
                          output_names=list(output_names), dynamic_axes=dynamic_axes,
                          opset_version=OPSET, do_constant_folding=True, dynamo=False)


def stage_a_inputs(batch: int, ticks: int, cfg: RhythmConfig, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(batch, ticks, cfg.d_in, generator=g),
            torch.rand(batch, cfg.cond_dim, generator=g))


def stage_b_inputs(batch: int, events: int, cfg: ChoreoConfig, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(batch, events, cfg.ctx_k, cfg.d_in, generator=g),
            torch.randn(batch, events, cfg.n_scalars, generator=g),
            torch.randint(0, C.N_TOKENS + 1, (batch, events), generator=g),
            torch.rand(batch, cfg.cond_dim, generator=g))


def export_stage_a(model: RhythmNet, path) -> None:
    axes = {"x": {0: "batch", 1: "ticks"}, "cond": {0: "batch"}}
    axes.update({n: {0: "batch", 1: "ticks"} for n in A_OUTPUTS})
    _export(model, stage_a_inputs(1, 2 * C.G, model.cfg), path, ("x", "cond"),
            A_OUTPUTS, axes)


def export_stage_b(model: ChoreoNet, path) -> None:
    axes = {"ctx": {0: "batch", 1: "events"}, "scalars": {0: "batch", 1: "events"},
            "tokens": {0: "batch", 1: "events"}, "cond": {0: "batch"},
            "logits": {0: "batch", 1: "events"}}
    _export(model, stage_b_inputs(1, 9, model.cfg), path, B_INPUTS, ("logits",), axes)


def _session(path):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def parity(model, onnx_path, input_names, cases) -> float:
    """Largest |pytorch - onnxruntime| over `cases` (tuples of input tensors),
    across every output. Raises if any output misses PARITY_ATOL."""
    model.eval()
    sess = _session(onnx_path)
    worst = 0.0
    for case in cases:
        with torch.no_grad():
            ref = model(*case)
        ref = ref if isinstance(ref, tuple) else (ref,)
        got = sess.run(None, {n: t.numpy() for n, t in zip(input_names, case)})
        for r, o in zip(ref, got):
            r = r.numpy()
            if r.shape != o.shape:
                raise AssertionError(f"shape mismatch {r.shape} vs {o.shape}")
            worst = max(worst, float(np.abs(r - o).max()))
            if not np.allclose(r, o, atol=PARITY_ATOL):
                raise AssertionError(
                    f"ONNX parity failed for {Path(onnx_path).name}: "
                    f"max |diff| {np.abs(r - o).max():.2e} > {PARITY_ATOL}")
    return worst


def verify_stage_a(model: RhythmNet, path) -> float:
    cases = [stage_a_inputs(b, t, model.cfg, seed=i)
             for i, (b, t) in enumerate(((1, C.G), (2, 5 * C.G), (1, 13 * C.G)))]
    return parity(model, path, ("x", "cond"), cases)


def verify_stage_b(model: ChoreoNet, path) -> float:
    cases = [stage_b_inputs(b, n, model.cfg, seed=i)
             for i, (b, n) in enumerate(((1, 1), (2, 37), (1, 211)))]
    return parity(model, path, B_INPUTS, cases)


def quantize_int8(src, dst) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8,
                     op_types_to_quantize=["MatMul", "Gemm"])


def feature_spec() -> dict:
    """Everything the runtime must agree on with training, recorded so a
    mismatched build fails loudly instead of quietly charting from features the
    model has never seen."""
    return {"sr": GF.SR, "n_fft": GF.N_FFT, "hop": GF.HOP, "n_mel": GF.N_MEL,
            "fmin": GF.FMIN, "fmax": GF.FMAX, "d_in": GF.D_IN,
            "flux_lead_s": GF.FLUX_LEAD_S, "ticks_per_measure": C.G,
            "ctx_k": EF.CTX_K, "n_scalars": EF.N_SCALARS,
            "cond_dim": cond.COND_DIM, "hold_dur_buckets": list(C.HOLD_DUR_BUCKETS)}


def export_models(model_a: RhythmNet, model_b: ChoreoNet, out_dir, *,
                  metrics: dict | None = None, trained_with_stems: bool = False,
                  int8: bool = False, extra: dict | None = None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / META_FILE
    if meta_path.exists():
        meta_path.unlink()                 # never leave a meta describing old graphs
    model_a, model_b = model_a.cpu().eval(), model_b.cpu().eval()
    export_stage_a(model_a, out_dir / STAGE_A_FILE)
    export_stage_b(model_b, out_dir / STAGE_B_FILE)
    diff_a = verify_stage_a(model_a, out_dir / STAGE_A_FILE)
    diff_b = verify_stage_b(model_b, out_dir / STAGE_B_FILE)
    files = {"stage_a": STAGE_A_FILE, "stage_b": STAGE_B_FILE}
    quantization = "fp32"
    if int8:
        for key, name in list(files.items()):
            q = name.replace(".onnx", ".int8.onnx")
            quantize_int8(out_dir / name, out_dir / q)
            files[key + "_int8"] = q
        quantization = "fp32+int8"
    meta = {
        "name": "stepforge_flow", "codec_version": C.CODEC_VERSION,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "opset": OPSET, "files": files, "quantization": quantization,
        "trained_with_stems": bool(trained_with_stems),
        "features": feature_spec(),
        "stage_a": {"config": model_a.cfg.to_dict(), "params": model_a.n_params,
                    "outputs": list(A_OUTPUTS), "parity_max_abs_diff": diff_a},
        "stage_b": {"config": model_b.cfg.to_dict(), "params": model_b.n_params,
                    "inputs": list(B_INPUTS), "parity_max_abs_diff": diff_b},
        "metrics": metrics or {},
    }
    meta.update(extra or {})
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta


def load_checkpoint(path, stage: str):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if stage == "a":
        model = RhythmNet(RhythmConfig(**state["config"]))
    else:
        model = ChoreoNet(ChoreoConfig(**state["config"]))
    model.load_state_dict(state["model"])
    return model.eval(), state


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage-a", required=True, help="RhythmNet checkpoint (.pt)")
    ap.add_argument("--stage-b", required=True, help="ChoreoNet checkpoint (.pt)")
    ap.add_argument("--out", default="models/stepforge_flow")
    ap.add_argument("--int8", action="store_true",
                    help="also write dynamic-quantized int8 variants")
    ap.add_argument("--trained-with-stems", action="store_true")
    a = ap.parse_args(argv)
    model_a, sa = load_checkpoint(a.stage_a, "a")
    model_b, sb = load_checkpoint(a.stage_b, "b")
    metrics = {"stage_a": sa.get("val", {}), "stage_b": sb.get("val", {}),
               "stage_a_epoch": sa.get("epoch"), "stage_b_epoch": sb.get("epoch")}
    meta = export_models(model_a, model_b, a.out, metrics=metrics,
                         trained_with_stems=a.trained_with_stems, int8=a.int8)
    print(f"[export] wrote {a.out}: stage A {meta['stage_a']['params'] / 1e6:.2f}M "
          f"(parity {meta['stage_a']['parity_max_abs_diff']:.1e}), stage B "
          f"{meta['stage_b']['params'] / 1e6:.2f}M "
          f"(parity {meta['stage_b']['parity_max_abs_diff']:.1e}), "
          f"{meta['quantization']}")


if __name__ == "__main__":
    main()
