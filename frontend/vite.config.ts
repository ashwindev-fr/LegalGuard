import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/answer': 'http://127.0.0.1:8000',
      '/retrieve': 'http://127.0.0.1:8000',
      '/health': 'http://127.0.0.1:8000',
      '/graph': 'http://127.0.0.1:8000',
      '/ingest': 'http://127.0.0.1:8000',
    },
  },
});
