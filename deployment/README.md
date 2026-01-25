# 口罩检测系统 - Docker 部署指南

本指南将帮助您使用 Docker 和 Docker Compose 部署口罩检测系统。

## 📋 前置要求

1. **Docker Desktop**: 确保已安装并运行 Docker Desktop
   - Windows: [下载 Docker Desktop for Windows](https://www.docker.com/products/docker-desktop)
   - 最低版本要求: Docker 20.10+, Docker Compose 2.0+

2. **系统要求**:
   - Windows 10/11 (推荐)
   - 至少 4GB RAM
   - 至少 10GB 可用磁盘空间

## 🚀 快速开始

### 方法一: 使用自动化脚本 (推荐)

1. **启动 Docker Desktop**:
   ```powershell
   .\deployment\scripts\start-docker.ps1
   ```

2. **构建应用镜像**:
   ```powershell
   .\deployment\scripts\deploy.ps1 -Build
   ```

3. **启动所有服务**:
   ```powershell
   .\deployment\scripts\deploy.ps1 -Start
   ```

4. **访问应用**:
   - 前端页面: http://localhost
   - API 接口: http://localhost/api
   - 健康检查: http://localhost/api/health

### 方法二: 手动部署

1. **确保 Docker 运行**:
   ```powershell
   docker --version
   docker-compose --version
   ```

2. **构建镜像**:
   ```powershell
   docker build -t mask-detection-app:latest .
   ```

3. **启动服务**:
   ```powershell
   docker-compose up -d
   ```

4. **检查服务状态**:
   ```powershell
   docker-compose ps
   ```

## 🏗️ 架构说明

系统采用微服务架构，包含以下组件:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     Nginx       │    │  FastAPI API    │    │     Redis       │
│  (反向代理)      │────│   (后端服务)     │────│   (缓存服务)     │
│   Port: 80      │    │   Port: 5000    │    │   Port: 6379    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 服务详情

- **Nginx**: 反向代理服务器，处理静态文件和API路由
- **FastAPI 服务**: 主要的后端服务，提供口罩检测功能
- **Redis**: 缓存服务，提升性能 (可选)

## 📁 目录结构

```
deployment/
├── docker/
│   ├── Dockerfile              # 专用 Dockerfile
│   └── docker-compose.yml      # 专用 compose 配置
├── nginx/
│   ├── nginx.conf              # Nginx 主配置
│   ├── default.conf            # 站点配置
│   └── ssl/                    # SSL 证书目录
├── scripts/
│   ├── start-docker.ps1        # Docker 启动脚本
│   └── deploy.ps1              # 部署管理脚本
└── README.md                   # 本文档
```

## 🛠️ 管理命令

### 使用部署脚本

```powershell
# 构建镜像
.\deployment\scripts\deploy.ps1 -Build

# 启动服务
.\deployment\scripts\deploy.ps1 -Start

# 停止服务
.\deployment\scripts\deploy.ps1 -Stop

# 重启服务
.\deployment\scripts\deploy.ps1 -Restart

# 查看日志
.\deployment\scripts\deploy.ps1 -Logs

# 清理资源
.\deployment\scripts\deploy.ps1 -Clean
```

### 使用 Docker Compose

```powershell
# 启动服务 (后台运行)
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down

# 重启特定服务
docker-compose restart backend

# 查看资源使用情况
docker-compose top
```

## 🔧 配置说明

### 环境变量

在 `docker-compose.yml` 中可以配置以下环境变量:

```yaml
environment:
   - APP_ENV=production           # FastAPI 运行环境标识
   - PYTHONPATH=/app              # Python 路径
   - UVICORN_HOST=0.0.0.0         # 绑定主机
   - UVICORN_PORT=5000            # 服务端口
```

### 端口映射

- `80:80` - Nginx HTTP 端口
- `443:443` - Nginx HTTPS 端口 (如果配置了 SSL)
- `5000:5000` - FastAPI 端口 (开发时使用)
- `6379:6379` - Redis 端口 (如果启用)

### 数据卷

- `./backend/uploads:/app/backend/uploads` - 上传文件存储
- `./logs:/app/logs` - 应用日志
- `./models/weights:/app/models/weights` - 模型权重文件
- `./frontend:/usr/share/nginx/html` - 前端静态资源（Nginx 根目录）
- `./deployment/nginx/ssl:/etc/nginx/ssl` - SSL 证书目录

## 🐛 故障排除

### 常见问题

1. **Docker Desktop 未启动**:
   ```
   error during connect: Get "http://%2F%2F.%2Fpipe%2FdockerDesktopLinuxEngine/...
   ```
   **解决**: 运行 `.\deployment\scripts\start-docker.ps1`

2. **端口被占用**:
   ```
   Error starting userland proxy: listen tcp 0.0.0.0:80: bind: address already in use
   ```
   **解决**: 停止占用端口的服务或修改 docker-compose.yml 中的端口映射

3. **内存不足**:
   ```
   docker: Error response from daemon: could not select device driver
   ```
   **解决**: 增加 Docker Desktop 的内存限制

4. **镜像构建失败**:
   ```
   ERROR: failed to solve: process "/bin/sh -c pip install..." did not complete successfully
   ```
   **解决**: 检查网络连接，或使用国内镜像源

### 日志查看

```powershell
# 查看所有服务日志
docker-compose logs

# 查看特定服务日志
docker-compose logs backend
docker-compose logs nginx

# 实时查看日志
docker-compose logs -f --tail=100
```

### 健康检查

```powershell
# 检查服务健康状态
docker-compose ps

# 手动健康检查
curl http://localhost/api/health
```

## 🔒 安全配置

### SSL/HTTPS 配置

1. 将 SSL 证书放置在 `deployment/nginx/ssl/` 目录
2. 修改 `deployment/nginx/default.conf` 启用 HTTPS
3. 重启 Nginx 服务

### 防火墙配置

确保以下端口在防火墙中开放:
- 80 (HTTP)
- 443 (HTTPS, 如果使用)

## 📊 监控和维护

### 性能监控

```powershell
# 查看容器资源使用情况
docker stats

# 查看系统资源
docker system df
```

### 定期维护

```powershell
# 清理未使用的镜像和容器
docker system prune -f

# 更新镜像
docker-compose pull
docker-compose up -d
```

## 🆘 获取帮助

如果遇到问题，请:

1. 查看日志: `docker-compose logs`
2. 检查服务状态: `docker-compose ps`
3. 验证配置文件语法: `docker-compose config`
4. 重启服务: `docker-compose restart`

---

**注意**: 首次部署可能需要较长时间来下载依赖和构建镜像，请耐心等待。