# -------------------------------------------------
# 基础镜像：Ubuntu + CUDA + Python 3.8
# 说明：该镜像支持 PyTorch GPU（cuDNN8, CUDA 11.2）
# -------------------------------------------------
FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

# -------------------------------------------------
# 环境变量设置
# -------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# -------------------------------------------------
# 安装 Python3.8 + pip
# -------------------------------------------------
RUN apt-get update && \
    apt-get install -y python3.8 python3-pip python3-dev libgl1 libglib2.0-0 && \
    ln -sf /usr/bin/python3.8 /usr/bin/python && \
    pip install --upgrade pip && \
    rm -rf /var/lib/apt/lists/*

# -------------------------------------------------
# 设置工作目录
# -------------------------------------------------
WORKDIR /app

# -------------------------------------------------
# 复制依赖文件并安装 Python 包
# -------------------------------------------------
COPY requirements.txt /app/

# ✅ 安装 requirements.txt（基础依赖）
RUN pip install --no-cache-dir -r requirements.txt

# ✅ 重点：安装 GPU 版 PyTorch
# 注意：+cu118 表示 CUDA 11.8，可兼容 11.2 运行时
RUN pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    -f https://pytorch.tuna.tsinghua.edu.cn/whl/torch_stable.html

# -------------------------------------------------
# 复制后端代码和模型目录
# -------------------------------------------------
COPY . /app/

# -------------------------------------------------
# 暴露端口（FastAPI）
# -------------------------------------------------
EXPOSE 8000

# -------------------------------------------------
# 启动 FastAPI 应用
# -------------------------------------------------
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
