import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // 🔹 后端 A（例如负责上传）
      "/api_a": {
        target: "http://127.0.0.1:8001", // backend_a 本地端口
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api_a/, ""),
      },

      // 🔹 后端 B（例如负责模型推理、分析等）
      "/api_b": {
        target: "http://127.0.0.1:8002", // backend_b 本地端口
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api_b/, ""),
      },

      // 🔹 静态图片或文件（如果前端需要直接访问 /pictures）
      "/pictures": {
        target: "http://127.0.0.1:8001", // 让它指向 backend_a 或你存放图片的后端
        changeOrigin: true,
      },
    },
  },
});
