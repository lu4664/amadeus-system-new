# 使用 Node.js 18 作为基础镜像
FROM node:18-alpine

# 安装必要的工具 - 使用国内镜像加速
RUN apk update && \
    apk add --no-cache python3 make g++ git

# 全局安装 pnpm 和 pm2 - 使用国内镜像
RUN npm config set registry https://registry.npmmirror.com && \
    npm install -g pnpm pm2

# 设置工作目录
WORKDIR /app

# 复制依赖配置文件
COPY package.json pnpm-lock.yaml* ./
COPY service/package.json service/pnpm-lock.yaml* ./service/

# --- 修改开始 ---
# 使用 pnpm 安装根目录依赖，保持与 pnpm-lock.yaml 一致
RUN pnpm install --registry=https://registry.npmmirror.com && \
    cd service && pnpm install --registry=https://registry.npmmirror.com && \
    cd ..
# --- 修改结束 ---

# 复制源代码
COPY . .

# 构建后端
RUN cd service && \
    pnpm build && \
    rm -rf src

# 创建启动脚本，使用环境变量
RUN echo '#!/bin/sh' > /app/start.sh && \
    echo '' >> /app/start.sh && \
    echo '# 生成前端环境变量文件' >> /app/start.sh && \
    echo 'echo "VITE_APP_API_BASE_URL=$VITE_APP_API_BASE_URL" > /app/.env' >> /app/start.sh && \
    echo 'echo "VITE_APP_DEFAULT_USERNAME=$VITE_APP_DEFAULT_USERNAME" >> /app/.env' >> /app/start.sh && \
    echo '' >> /app/start.sh && \
    echo '# 生成后端环境变量文件' >> /app/start.sh && \
    echo 'echo "PORT=3002" > /app/service/.env' >> /app/start.sh && \
    echo 'echo "WEBRTC_API_URL=$WEBRTC_API_URL" >> /app/service/.env' >> /app/start.sh && \
    echo '' >> /app/start.sh && \
    echo '# 重新构建前端' >> /app/start.sh && \
    echo 'cd /app && pnpm build' >> /app/start.sh && \
    echo '' >> /app/start.sh && \
    echo '# 启动服务' >> /app/start.sh && \
    echo 'cd /app/service && pnpm prod &' >> /app/start.sh && \
    echo 'cd /app && pnpm preview --host 0.0.0.0' >> /app/start.sh && \
    chmod +x /app/start.sh

# 暴露端口
EXPOSE 3002

# 启动服务
CMD ["/bin/sh", "/app/start.sh"]