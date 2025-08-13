#!/bin/bash

# 检查是否提供了模型目录
if [ -z "$1" ]; then
    echo "请提供本地模型目录路径"
    echo "用法: $0 <本地模型目录>"
    exit 1
fi

# 检查模型目录是否存在
if [ ! -d "$1" ]; then
    echo "错误: 模型目录 $1 不存在"
    exit 1
fi

# 设置环境变量 - 替换为你的令牌
export VALID_TOKENS="your_secure_token1,your_secure_token2"

# 创建临时目录用于构建
mkdir -p build
cp -r $1 build/local_model
cp Dockerfile build/
cp requirements.txt build/
cp main.py build/
cp token_auth.py build/

# 构建Docker镜像
cd build
docker build -t huggingface-model-api .

# 运行容器
docker run -d -p 8000:8000 \
    -e "VALID_TOKENS=$VALID_TOKENS" \
    --name hf-model-service \
    huggingface-model-api

echo "服务已启动，访问 http://localhost:8000/docs 查看API文档"
