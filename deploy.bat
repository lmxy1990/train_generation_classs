@echo off
setlocal enabledelayedexpansion

:: 检查是否提供了模型目录
if "%1"=="" (
    echo 请提供本地模型目录路径
    echo 用法: %0 ^<本地模型目录^>
    exit /b 1
)

:: 检查模型目录是否存在
if not exist "%1\" (
    echo 错误: 模型目录 %1 不存在
    exit /b 1
)

:: 设置环境变量 - 替换为你的令牌
set "VALID_TOKENS=your_secure_token1,your_secure_token2"

:: 创建临时目录用于构建
mkdir build 2>nul
rmdir /s /q build\local_model 2>nul
xcopy /s /e /i "%1" build\local_model >nul
copy Dockerfile build\ >nul
copy requirements.txt build\ >nul
copy main.py build\ >nul
copy token_auth.py build\ >nul

:: 构建Docker镜像
cd build
docker build -t huggingface-model-api .

:: 运行容器
docker run -d -p 8000:8000 ^
    -e "VALID_TOKENS=!VALID_TOKENS!" ^
    --name hf-model-service ^
    huggingface-model-api

echo 服务已启动，访问 http://localhost:8000/docs 查看API文档

:: 清理临时文件（可选，注释掉可保留构建文件用于调试）
cd ..
rmdir /s /q build 2>nul

endlocal
