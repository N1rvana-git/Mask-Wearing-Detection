# 基于Python 3.9的官方镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONPATH=/app
ENV UVICORN_APP=backend.app:app
ENV UVICORN_HOST=0.0.0.0
ENV UVICORN_PORT=5000
ENV APP_ENV=production

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# 复制requirements文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# 创建必要的目录
RUN mkdir -p models/weights logs

# 设置权限
RUN chmod +x deployment/scripts/start.sh

# 暴露端口
EXPOSE 5000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/api/health || exit 1

# 启动命令
CMD ["sh", "-c", "uvicorn ${UVICORN_APP:-backend.app:app} --host ${UVICORN_HOST:-0.0.0.0} --port ${UVICORN_PORT:-5000}"]