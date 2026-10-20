/// <reference types="vitest/config" />
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig, loadEnv } from "vite";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const api = `http://127.0.0.1:${env.SYNTHETICS_PORT || process.env.SYNTHETICS_PORT || 8610}`;
  return {
    plugins: [react(), tailwindcss()],
    server: { proxy: { "/api": { target: api, changeOrigin: false } } },
    preview: { proxy: { "/api": { target: api } } },
    build: { outDir: "dist", sourcemap: true },
    test: { environment: "jsdom", include: ["src/**/*.test.{ts,tsx}"], globals: false },
  };
});
