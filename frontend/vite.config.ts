import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  // Support for dev & prod
  const backendPort = process.env.BACKEND_PORT || '8080'
  const isDev = mode === 'development'

  return {
    plugins: [react()],
    server: {
      port: parseInt(process.env.FRONTEND_PORT || '3001', 10),
      host: process.env.FRONTEND_HOST === 'true' ? true : (process.env.FRONTEND_HOST || '0.0.0.0'),
      proxy: {
        '/api': {
          target: `http://localhost:${backendPort}`,
          changeOrigin: true,
          secure: false,
        },
        '/ws': {
          target: `http://localhost:${backendPort}`,
          ws: true,
          changeOrigin: true,
          secure: false,
        },
      },
    },
    define: {
      // Provide env so frontend can read at build time
      __BACKEND_PORT__: JSON.stringify(backendPort),
    },
  }
})
