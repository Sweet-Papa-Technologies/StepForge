<script setup lang="ts">
import { computed, onMounted, ref, watch } from "vue";
import { useRouter } from "vue-router";
import { api } from "../lib/api";
import { useEngine } from "../stores/engine";
import type { SongCost } from "../lib/types";

const engine = useEngine();
const router = useRouter();
const songs = ref<SongCost[]>([]);
const expanded = ref<string | null>(null);
const data = ref<{ rollup: any; markdown: string | null; hasLedger: boolean } | null>(null);
const jobId = ref<string | null>(null);
const job = computed(() => engine.job(jobId.value));

const totals = computed(() => data.value?.rollup?.totals ?? null);
const byStage = computed<Record<string, any>>(() => data.value?.rollup?.by_stage ?? data.value?.rollup?.stages ?? {});
const topCalls = computed<any[]>(() => data.value?.rollup?.top_calls ?? data.value?.rollup?.priciest ?? []);
/** Only meaningful when hybrid routing ran; a ledger from before it existed (or
 *  from a single-model run) carries no tier and this stays empty. */
const byTier = computed<Record<string, any>>(() => {
  const t = data.value?.rollup?.by_tier ?? {};
  const keys = Object.keys(t).filter((k) => k !== "unrouted");
  return keys.length ? Object.fromEntries(keys.map((k) => [k, t[k]])) : {};
});

function usd(v: any): string {
  return typeof v === "number" ? `$${v.toFixed(4)}` : "—";
}

async function load() {
  [data.value, songs.value] = await Promise.all([api.costReport(), api.costBySong()]);
}
async function refresh() {
  const j = await engine.startCostRefresh();
  jobId.value = j.id;
}
watch(
  () => job.value?.running,
  async (r, was) => {
    if (was && !r) await load();
  },
);
onMounted(load);
</script>

<template>
  <div class="h-full flex flex-col">
    <header class="h-16 border-b border-outline-variant flex items-center justify-between px-margin-desktop shrink-0 bg-surface">
      <div>
        <h2 class="font-headline-md text-headline-md text-on-surface">Cost Ledger</h2>
      </div>
      <button
        class="px-4 py-2 rounded border border-outline text-on-surface hover:bg-surface-variant transition-colors font-label-mono text-label-mono flex items-center gap-2 disabled:opacity-50"
        :disabled="job?.running"
        @click="refresh"
      >
        <span class="material-symbols-outlined text-sm" :class="{ spinner: job?.running }">
          {{ job?.running ? "progress_activity" : "refresh" }}
        </span>
        Recompute report
      </button>
    </header>

    <div class="flex-1 overflow-y-auto p-margin-desktop">
      <div v-if="!totals && !data?.hasLedger && !songs.length" class="text-on-surface-variant font-body-md py-16 text-center">
        <span class="material-symbols-outlined text-4xl text-on-surface-variant/50 mb-2 block">payments</span>
        <p>No cost ledger yet. Run a chart generation, then recompute the report.</p>
      </div>

      <template v-else>
        <!-- Totals -->
        <div class="grid grid-cols-2 lg:grid-cols-4 gap-gutter mb-8">
          <div class="bg-surface-container border border-outline-variant rounded-xl p-4">
            <p class="font-label-mono text-[10px] uppercase text-on-surface-variant tracking-widest mb-2">Total spend</p>
            <p class="font-headline-md text-headline-md text-primary-fixed-dim">{{ usd(totals?.usd) }}</p>
          </div>
          <div class="bg-surface-container border border-outline-variant rounded-xl p-4">
            <p class="font-label-mono text-[10px] uppercase text-on-surface-variant tracking-widest mb-2">Per song</p>
            <p class="font-headline-md text-headline-md text-on-surface">{{ usd(totals?.usd_per_song) }}</p>
          </div>
          <div class="bg-surface-container border border-outline-variant rounded-xl p-4">
            <p class="font-label-mono text-[10px] uppercase text-on-surface-variant tracking-widest mb-2">Per chart</p>
            <p class="font-headline-md text-headline-md text-on-surface">{{ usd(totals?.usd_per_chart) }}</p>
          </div>
          <div class="bg-surface-container border border-outline-variant rounded-xl p-4">
            <p class="font-label-mono text-[10px] uppercase text-on-surface-variant tracking-widest mb-2">Model calls</p>
            <p class="font-headline-md text-headline-md text-on-surface">{{ totals?.model_calls ?? "—" }}</p>
          </div>
        </div>

        <!-- Per song: what each song cost and what that bought -->
        <div v-if="songs.length" class="mb-8">
          <h3 class="font-label-mono text-label-mono text-on-surface-variant uppercase tracking-wider mb-3">
            Per song
          </h3>
          <div class="bg-surface-container border border-outline-variant rounded-xl overflow-hidden">
            <div
              v-for="s in songs"
              :key="s.song"
              class="border-b border-outline-variant/50 last:border-b-0"
            >
              <button
                class="w-full flex items-center gap-3 px-4 py-3 hover:bg-surface-variant/40 transition-colors text-left"
                @click="expanded === s.song ? (expanded = null) : (expanded = s.song)"
              >
                <span class="material-symbols-outlined text-[18px] text-on-surface-variant shrink-0">
                  {{ expanded === s.song ? "expand_less" : "expand_more" }}
                </span>
                <span class="flex-1 min-w-0">
                  <span class="block truncate text-on-surface font-body-md">{{ s.title ?? s.song }}</span>
                  <span class="block font-label-mono text-[10px] text-on-surface-variant">
                    {{ s.charts.length }} chart{{ s.charts.length === 1 ? "" : "s" }} ·
                    {{ s.totalNotes }} notes ·
                    {{ s.modelCalls }} model call{{ s.modelCalls === 1 ? "" : "s" }}
                    <span v-if="s.cacheHitAnalysis"> · analysis cached</span>
                  </span>
                </span>
                <span v-if="s.bestScore != null" class="font-label-mono text-[12px] shrink-0"
                      :class="s.bestScore >= 7 ? 'text-green-400' : 'text-tertiary-fixed-dim'">
                  {{ s.bestScore.toFixed(1) }}/10
                </span>
                <span class="font-label-mono text-[13px] text-primary-fixed-dim w-20 text-right shrink-0">
                  {{ usd(s.usd) }}
                </span>
              </button>

              <div v-if="expanded === s.song" class="px-4 pb-3 bg-surface-container-low">
                <div class="grid grid-cols-3 gap-3 py-2 font-label-mono text-[11px] text-on-surface-variant">
                  <span>LLM: <span class="text-secondary">{{ usd(s.llmUsd) }}</span></span>
                  <span>Compute: <span class="text-secondary">{{ usd(s.computeUsd) }}</span></span>
                  <span v-if="s.charts.length">Per chart: <span class="text-secondary">{{ usd(s.usd / s.charts.length) }}</span></span>
                </div>
                <table v-if="s.charts.length" class="w-full font-label-mono text-[11px]">
                  <thead class="text-on-surface-variant">
                    <tr class="text-left">
                      <th class="py-1 font-normal">Difficulty</th>
                      <th class="py-1 font-normal">Meter</th>
                      <th class="py-1 font-normal">Notes</th>
                      <th class="py-1 font-normal">Score</th>
                      <th class="py-1 font-normal">Gates</th>
                    </tr>
                  </thead>
                  <tbody class="text-on-surface">
                    <tr v-for="c in s.charts" :key="c.difficulty" class="border-t border-outline-variant/30">
                      <td class="py-1 capitalize">{{ c.difficulty }}</td>
                      <td class="py-1">{{ c.meter ?? "—" }}</td>
                      <td class="py-1">{{ c.notes }}</td>
                      <td class="py-1" :class="c.score != null && c.score >= 7 ? 'text-green-400' : ''">
                        {{ c.score != null ? c.score.toFixed(1) : "—" }}
                      </td>
                      <td class="py-1">
                        <span v-if="c.gatesPass === null" class="text-on-surface-variant">—</span>
                        <span v-else :class="c.gatesPass ? 'text-green-400' : 'text-error'">
                          {{ c.gatesPass ? "pass" : "fail" }}
                        </span>
                      </td>
                    </tr>
                  </tbody>
                </table>
                <p v-else class="py-2 text-[11px] text-on-surface-variant">
                  No emitted charts — this spend didn't produce output.
                </p>
                <button
                  class="mt-2 px-3 py-1 rounded border border-outline-variant text-on-surface hover:bg-surface-variant font-label-mono text-[11px]"
                  @click.stop="router.push({ name: 'results', params: { base: s.base } })"
                >
                  Open results
                </button>
              </div>
            </div>
          </div>
        </div>

        <!-- By stage -->
        <div v-if="Object.keys(byStage).length" class="mb-8">
          <h3 class="font-label-mono text-label-mono text-on-surface-variant uppercase tracking-wider mb-3">Cost by stage</h3>
          <div class="bg-surface-container border border-outline-variant rounded-xl overflow-hidden">
            <div
              v-for="(v, k) in byStage"
              :key="k"
              class="flex justify-between items-center px-4 py-3 border-b border-outline-variant/50 last:border-b-0 font-label-mono text-[12px]"
            >
              <span class="text-on-surface capitalize">{{ k }}</span>
              <span class="text-secondary-fixed">{{ usd((v as any).usd ?? v) }}</span>
            </div>
          </div>
        </div>

        <!-- By hybrid tier: which model actually spent the money. Hidden when
             routing is off, so a single-model run isn't given a meaningless row. -->
        <div v-if="Object.keys(byTier).length" class="mb-8">
          <h3 class="font-label-mono text-label-mono text-on-surface-variant uppercase tracking-wider mb-3">
            Cost by model tier
          </h3>
          <div class="bg-surface-container border border-outline-variant rounded-xl overflow-hidden">
            <div
              v-for="(v, k) in byTier"
              :key="k"
              class="flex justify-between items-center px-4 py-3 border-b border-outline-variant/50 last:border-b-0 font-label-mono text-[12px]"
            >
              <span class="text-on-surface">
                {{ k === "lite" ? "Drafting (lite)" : k === "heavy" ? "Authoring & critic (heavy)" : k }}
                <span class="text-on-surface-variant">· {{ (v as any).calls }} call(s)</span>
              </span>
              <span class="text-secondary-fixed">{{ usd((v as any).usd) }}</span>
            </div>
          </div>
        </div>

        <!-- Top calls -->
        <div v-if="topCalls.length" class="mb-8">
          <h3 class="font-label-mono text-label-mono text-on-surface-variant uppercase tracking-wider mb-3">Priciest calls</h3>
          <div class="bg-surface-container border border-outline-variant rounded-xl overflow-hidden">
            <div
              v-for="(c, i) in topCalls"
              :key="i"
              class="flex justify-between items-center px-4 py-3 border-b border-outline-variant/50 last:border-b-0 font-label-mono text-[12px]"
            >
              <span class="text-on-surface truncate">{{ c.model ?? c.stage ?? c.name ?? "call" }}</span>
              <span class="text-tertiary-fixed-dim">{{ usd(c.usd ?? c.cost) }}</span>
            </div>
          </div>
        </div>

        <!-- Raw markdown fallback -->
        <details v-if="data?.markdown" class="border border-outline-variant rounded-xl bg-surface-container-low overflow-hidden">
          <summary class="font-label-mono text-label-mono uppercase tracking-wider p-3 cursor-pointer text-on-surface-variant">Full report</summary>
          <pre class="p-4 text-[12px] text-on-surface-variant whitespace-pre-wrap font-label-mono border-t border-outline-variant">{{ data.markdown }}</pre>
        </details>
      </template>
    </div>
  </div>
</template>
