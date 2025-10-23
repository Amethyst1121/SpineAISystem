# 基础镜像：Ubuntu + CUDA + Python 3.8（支持 TensorFlow-GPU 2.10）
FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

# 防止 Python 写缓存 + 立即输出日志
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 安装 Python 和 pip
RUN apt-get update && apt-get install -y python3.8 python3-pip python3-dev && \
    ln -s /usr/bin/python3.8 /usr/bin/python && \
    pip install --upgrade pip

# 设置容器工作目录
WORKDIR /app

# 复制 requirements 并安装依赖
COPY requirements.txt /app/
RUN pip install -r requirements.txt

# 复制后端代码和模型目录
COPY . /app/

# 暴露端口
EXPOSE 8000

# 启动 FastAPI 应用
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
