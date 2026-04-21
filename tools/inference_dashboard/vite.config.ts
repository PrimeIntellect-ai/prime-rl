import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  root: "frontend",
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
  server: {
    port: 5174,
  },
});
