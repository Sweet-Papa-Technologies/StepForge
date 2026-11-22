import { invoke } from "@tauri-apps/api/core";

/** What the feedback dialog asks `feedback.rs` to write up. */
export interface FeedbackRequest {
  kind: "general" | "chart";
  /** Chart reports: which chart. The song folder's name is used to find its
   *  report.json and is never written into the issue. */
  base?: string;
  target?: string;
  difficulty?: string;
  /** Written into the issue only when `includeTitle` is true. */
  songTitle?: string;
  includeTitle?: boolean;
  /** "StepForge Flow v1.6", when known. */
  model?: string | null;
  /** The Copy diagnostics text (`api.engineDiagnostics()`). */
  diagnostics: string;
}

/** The issue exactly as it will be put into GitHub's form. */
export interface FeedbackIssue {
  title: string;
  body: string;
  /** Opens GitHub's new-issue form with `title` and `body` filled in. */
  url: string;
  /** The diagnostics were cut to keep the link short (the body says where). */
  truncated: boolean;
}

/** Build the issue text and link. Sends nothing and opens nothing. */
export const feedbackIssue = (request: FeedbackRequest) =>
  invoke<FeedbackIssue>("feedback_issue", { request });

/** Where to paste the text by hand if the filled-in link cannot be opened. */
export const NEW_ISSUE_PAGE = "https://github.com/Sweet-Papa-Technologies/StepForge/issues/new";
