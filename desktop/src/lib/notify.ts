import {
  isPermissionGranted,
  requestPermission,
  sendNotification,
} from "@tauri-apps/plugin-notification";

let granted: boolean | null = null;

/** Ask once, lazily; a denied permission simply disables notifications. */
export async function ensureNotificationPermission(): Promise<boolean> {
  if (granted !== null) return granted;
  try {
    granted = await isPermissionGranted();
    if (!granted) granted = (await requestPermission()) === "granted";
  } catch {
    granted = false;
  }
  return granted;
}

/** Best-effort desktop notification — never throws into caller flow. */
export async function notify(title: string, body: string): Promise<void> {
  try {
    if (!(await ensureNotificationPermission())) return;
    sendNotification({ title, body });
  } catch {
    /* notifications are a nicety, not a requirement */
  }
}
