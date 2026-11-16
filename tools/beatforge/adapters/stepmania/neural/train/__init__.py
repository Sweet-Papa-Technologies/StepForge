"""Offline training for StepForge Flow.

`corpus` needs only numpy/scipy/soundfile/simfile and runs anywhere the runtime
does. Every other module here imports PyTorch and must never be imported from
the inference path (REQ-SFN-01); see requirements-train.txt.
"""
