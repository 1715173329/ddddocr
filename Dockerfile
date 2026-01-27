# 基础镜像使用 Python 3.10 (slim 版本可以减小镜像体积)
FROM python:3.10-slim

# 镜像作者信息
LABEL maintainer="sml2h3"
LABEL description="DdddOcr - 通用验证码识别API服务"

# 设置工作目录
WORKDIR /app

# 安装系统依赖 (apt-get 非交互式安装并在安装后清理缓存以减小镜像大小)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 复制项目依赖文件
COPY requirements.txt .

# 安装 Python 依赖
# --no-cache-dir: 不缓存下载的包，减小镜像大小
# -r requirements.txt: 从文件安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件到工作目录
COPY . .

# 设置 Python 路径
ENV PYTHONPATH=/app

# 设置 DdddOcr API 服务的默认环境变量
# 这些环境变量可以在 docker run 或 docker-compose 中覆盖

# API 服务器配置
ENV DDDDOCR_HOST=0.0.0.0       # 监听所有网络接口
ENV DDDDOCR_PORT=8000          # 服务运行端口
ENV DDDDOCR_WORKERS=1          # API 服务工作进程数

# OCR 引擎配置
ENV DDDDOCR_OCR=true           # 是否启用 OCR 功能
ENV DDDDOCR_DET=false          # 是否启用目标检测功能
ENV DDDDOCR_OLD=false          # 是否使用旧版 OCR 模型
ENV DDDDOCR_BETA=false         # 是否使用 Beta 版 OCR 模型
ENV DDDDOCR_USE_GPU=false      # 是否使用 GPU 加速
ENV DDDDOCR_DEVICE_ID=0        # GPU 设备 ID
ENV DDDDOCR_SHOW_AD=true       # 是否显示广告

# 自定义模型配置（需要挂载卷才能访问）
ENV DDDDOCR_IMPORT_ONNX_PATH="" # 自定义模型路径
ENV DDDDOCR_CHARSETS_PATH=""    # 自定义字符集路径

# 暴露端口（与 DDDDOCR_PORT 环境变量保持一致）
EXPOSE 8000

# 容器启动时执行的命令，使用 python -m ddddocr api 启动 API 服务
# 参数从环境变量读取
CMD python -m ddddocr api \
    --host=${DDDDOCR_HOST} \
    --port=${DDDDOCR_PORT} \
    --workers=${DDDDOCR_WORKERS} \
    --ocr=${DDDDOCR_OCR} \
    --det=${DDDDOCR_DET} \
    --old=${DDDDOCR_OLD} \
    --beta=${DDDDOCR_BETA} \
    --use-gpu=${DDDDOCR_USE_GPU} \
    --device-id=${DDDDOCR_DEVICE_ID} \
    --show-ad=${DDDDOCR_SHOW_AD} \
    --import-onnx-path=${DDDDOCR_IMPORT_ONNX_PATH} \
    --charsets-path=${DDDDOCR_CHARSETS_PATH}

# 健康检查，确保容器正常运行
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -f http://localhost:${DDDDOCR_PORT}/health || exit 1 
