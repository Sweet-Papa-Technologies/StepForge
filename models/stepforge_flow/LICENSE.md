# StepForge Flow model weights — license

The files in this folder that make up the model — `stage_a_rhythm.onnx`,
`stage_b_choreo.onnx`, their `.int8.onnx` variants, `phrase_index.npz` and
`model_meta.json` — are licensed under the
**Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International**
license (CC BY-NC-SA 4.0): https://creativecommons.org/licenses/by-nc-sa/4.0/

The model was trained on community step charts and music whose licenses vary
by song (see `MODEL_CARD.md`). Some of them are non-commercial or share-alike,
so the weights carry the most restrictive combination of those terms.

**Attribution.** The model learned from the work of the musicians and step
artists listed in `CREDITS.csv`. When you redistribute the weights, keep
`CREDITS.csv` with them.

**Charts you make with StepForge are yours.** This license applies to the model
files, not to the charts the app writes for your songs.

**Removal requests.** If you made a song or chart listed in `CREDITS.csv` and
want it left out of future models, open an issue on
https://github.com/Sweet-Papa-Technologies/StepForge or contact us (see the
README); it will be removed and the model retrained.

The source code of StepForge is licensed separately under the MIT license
(`LICENSE` at the repository root).
