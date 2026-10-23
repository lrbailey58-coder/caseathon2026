import path from 'node:path'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(import.meta.dirname, './src'),
    },
  },
  server: {
    cors: true,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000', // 127.0.0.1, not localhost: on Windows, Node may resolve localhost to IPv6 (::1) while uvicorn listens on IPv4
        changeOrigin: true,
      },
    },
  },
})
