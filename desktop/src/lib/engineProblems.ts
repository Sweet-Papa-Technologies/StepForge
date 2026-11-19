import type { EngineStatus, NeuralStatus } from "./types";

/** What to do about a broken engine. In a release build the engine is inside
 *  the app, so "pip install" is never the fix. */
export function engineAdvice(s: EngineStatus | null): string {
  return s?.bundled === false
    ? "Install the engine's packages: python3 -m pip install -r requirements.txt"
    : "Reinstalling StepForge usually fixes this. If it does not, use Copy diagnostics and send us the result.";
}

/** One line per distinct problem with the engine, each with its real cause.
 *  Empty when the engine can run. */
export function engineProblems(s: EngineStatus | null): string[] {
  if (!s) return ["StepForge could not check its charting engine."];
  const out: string[] = [];
  if (!s.engineRoot) {
    out.push(
      s.bundled === false
        ? "The engine folder was not found. Set it in Settings."
        : "The charting engine's files were not found.",
    );
  }
  if (!s.pythonOk) {
    out.push(`${s.pythonError || "Python could not be started"} (${s.python})`);
  } else if (!s.beatforgeOk && s.engineRoot) {
    out.push(s.importError || "The charting engine did not load, and gave no reason.");
  }
  return out;
}

export interface NeuralProblem {
  title: string;
  /** The real reason, as the engine reported it. */
  detail: string | null;
}

/** Why StepForge Flow cannot run, or null when it can (or is still being
 *  checked). Each cause has its own wording: an old engine, an engine that did
 *  not answer, missing model files, onnxruntime not loading, an unusable model. */
export function neuralProblem(st: NeuralStatus | null): NeuralProblem | null {
  if (!st) return null;
  if (st.installed && st.onnxruntime) return null;
  if (st.legacy)
    return {
      title: "This engine is older than the StepForge Flow model",
      detail: "It does not have the neural chart engine. Update or reinstall StepForge.",
    };
  if (st.probe_failed)
    return { title: "The charting engine did not answer", detail: st.error };
  if (!st.weights_present)
    return { title: "The StepForge Flow model files are missing", detail: st.error };
  if (!st.onnxruntime)
    return {
      title: "onnxruntime, which runs the model, did not load",
      detail: st.onnxruntime_error ?? st.error,
    };
  return { title: "The StepForge Flow model could not be used", detail: st.error };
}
