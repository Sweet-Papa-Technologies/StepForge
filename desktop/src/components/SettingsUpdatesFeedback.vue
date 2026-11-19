<script setup lang="ts">
/** Settings → "Updates & feedback": the switch for the update check, with
 *  what it does in plain words, and the ways to tell us something.
 *
 *  The wording below describes the request in lib/updateCheck.ts. If that
 *  request changes, this text has to change with it. */
import { useSettings } from "../stores/settings";
import CopyDiagnostics from "./CopyDiagnostics.vue";
import SendFeedback from "./SendFeedback.vue";

const settings = useSettings();
</script>

<template>
  <section class="mb-10 space-y-4">
    <h3 class="font-label-mono text-label-mono text-on-surface-variant uppercase tracking-wider">
      Updates &amp; feedback
    </h3>

    <label class="flex items-start gap-3 cursor-pointer">
      <input
        v-model="settings.value.checkForUpdates"
        type="checkbox"
        class="mt-1 w-4 h-4 accent-[#bc13fe] cursor-pointer shrink-0"
      />
      <span class="min-w-0">
        <span class="block font-body-md text-body-md text-on-surface">Check for updates when StepForge starts</span>
        <span class="block text-[11px] text-on-surface-variant mt-1">
          Once per start, this asks GitHub for the list of StepForge releases: their version numbers and release
          notes. Nothing about you or your songs is sent, not even which version you have; the comparison happens on
          this computer. As with any web request, GitHub sees your IP address and your system’s standard browser
          identification.
        </span>
        <span class="block text-[11px] text-on-surface-variant mt-1">
          If there is a newer version you get a notice with a link to its download page. StepForge never downloads
          or installs anything by itself.
        </span>
      </span>
    </label>

    <div>
      <div class="flex flex-wrap items-start gap-2">
        <SendFeedback />
        <CopyDiagnostics />
      </div>
      <p class="text-[11px] text-on-surface-variant mt-2">
        Send feedback opens a GitHub issue form in your browser, filled in with your app version, system and
        diagnostics. You see the text first, and nothing is sent unless you submit it on GitHub. To report a chart,
        use “Report this chart” on its Results screen.
      </p>
    </div>
  </section>
</template>
