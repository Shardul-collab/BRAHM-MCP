import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/v2': {
        target: 'http://localhost:8010',
        changeOrigin: true,
        ws: true,
        // Explicit generous timeouts - /v2/plan calls NIM (gpt-oss-120b) synchronously
        // and can legitimately take 30-60+s for a longer prompt with the full tool-schema
        // system prompt. Without this, Vite's proxy was returning a bare 502 to the
        // browser before the coordinator finished, even though the backend succeeded
        // every time when hit directly (confirmed via curl, 56s, 200 OK).
        // Raised 120s -> 180s alongside planning.py's httpx client timeout
        // (170s) -- must stay above it or the proxy cuts off before the
        // coordinator's own timeout would. See planning.py's generate_plan
        // comment for why NIM needs this much headroom right now.
        timeout: 180000,
        proxyTimeout: 180000,
      },
    },
  },
})
