# ---------- 构建阶段 ----------
FROM node:20.19 AS build

WORKDIR /app

# 先拷贝 package.json 和 package-lock（如果有），利用缓存加速安装
COPY package*.json ./

# 安装依赖（第一次会比较慢）
RUN npm install

# 拷贝源代码
COPY . .

# 可选：如果你需要在构建时注入 Vite 环境变量（VITE_ 前缀）
# ARG VITE_API_URL
# ENV VITE_API_URL=${VITE_API_URL}
# RUN echo "VITE_API_URL=${VITE_API_URL}" > .env.production

# 运行构建命令，产出 dist/ 目录
RUN npm run build

# ---------- 运行阶段（更小、更安全） ----------
FROM nginx:alpine

# 将构建阶段产物复制到 nginx 的静态目录
COPY --from=build /app/dist /usr/share/nginx/html

# 暴露端口（可选）
EXPOSE 80

# 前台运行 nginx（容器不会退出）
CMD ["nginx", "-g", "daemon off;"]
