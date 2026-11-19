import { invoke } from "@tauri-apps/api/core";

/** A newer release than the running app, as `update.rs` reports it. */
export interface UpdateNotice {
  /** "0.1.0-beta.7" */
  version: string;
  tag: string;
  name: string;
  /** Release notes as written (Markdown source). Show as plain text only. */
  notes: string;
  /** The release page on GitHub. */
  url: string;
  prerelease: boolean;
  /** The version that is running. */
  current: string;
}

/**
 * The only request the update check makes: the public list of StepForge's
 * releases. Nothing about the user is in it — no account, no identifier, not
 * even the version they have; `per_page` just keeps the answer small. The
 * comparison with the running version happens on this computer, in `update.rs`.
 *
 * (`/releases/latest` would do, except that it skips pre-releases, and every
 * beta is one.) A test in `update.rs` holds this file to that URL.
 */
const RELEASES_URL = "https://api.github.com/repos/Sweet-Papa-Technologies/StepForge/releases?per_page=10";

/** Long enough for a slow connection; the check is not worth waiting on. */
const TIMEOUT_MS = 8000;

/**
 * The release list as text, or null when it could not be had — offline, timed
 * out, rate-limited, anything. Never throws: a failed check is not an error
 * the user needs to hear about.
 *
 * Made from the web view rather than from Rust because the web view already
 * speaks HTTPS on every platform; doing it in Rust would add an HTTP client
 * and a TLS stack to the app for one GET. GitHub's API allows cross-origin
 * reads, and `Accept` is a header any page may set, so no preflight is sent.
 */
async function fetchReleaseList(): Promise<string | null> {
  const abort = new AbortController();
  const timer = window.setTimeout(() => abort.abort(), TIMEOUT_MS);
  try {
    const res = await fetch(RELEASES_URL, {
      method: "GET",
      headers: { Accept: "application/vnd.github+json" },
      credentials: "omit",
      referrerPolicy: "no-referrer",
      cache: "no-store",
      signal: abort.signal,
    });
    return res.ok ? await res.text() : null;
  } catch {
    return null;
  } finally {
    window.clearTimeout(timer);
  }
}

/**
 * Ask whether a release newer than this app (and newer than `dismissed`) is
 * out. Null for "no", and for every way of not finding out.
 */
export async function checkForUpdate(dismissed: string): Promise<UpdateNotice | null> {
  const body = await fetchReleaseList();
  if (body === null) return null;
  try {
    return await invoke<UpdateNotice | null>("update_from_releases", { body, dismissed });
  } catch {
    return null;
  }
}
