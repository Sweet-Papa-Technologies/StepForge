import { createRouter, createWebHashHistory, type RouteRecordRaw } from "vue-router";

const routes: RouteRecordRaw[] = [
  { path: "/", redirect: "/library" },
  { path: "/library", name: "library", component: () => import("../views/Library.vue") },
  { path: "/analyze/:base", name: "analyze", component: () => import("../views/Analyze.vue"), props: true },
  { path: "/generate/:base?", name: "generate", component: () => import("../views/Generate.vue"), props: true },
  { path: "/results/:base?", name: "results", component: () => import("../views/Results.vue"), props: true },
  { path: "/cost", name: "cost", component: () => import("../views/Cost.vue") },
  { path: "/settings", name: "settings", component: () => import("../views/Settings.vue") },
];

export const router = createRouter({
  history: createWebHashHistory(),
  routes,
});
