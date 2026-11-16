"""StepForge Flow — the custom beat-grid & biomechanical chart model
(docs/stepforge_neural_architecture_spec.md).

Runtime modules here (`codec`, `conditioning`, `grid_features`, `infer`,
`style_rag`, `decisions_hook`) import only numpy / scipy / onnxruntime / simfile
and existing BeatForge modules. Everything that needs PyTorch lives under
`neural.train` and is never imported from the runtime path (REQ-SFN-01).
"""

MODEL_DIR_NAME = "stepforge_flow"
