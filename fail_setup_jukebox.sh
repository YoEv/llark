#!/bin/bash

# Jukebox 安装脚本
# 基于 docker/jukebox-embed.dockerfile 修改

set -e

echo "开始安装 Jukebox..."

# 创建工作目录
WORK_DIR="/home/hice1/xli3252/Desktop/diversity-eval/external/llark"
cd "$WORK_DIR"

# 设置缓存目录路径（使用scratch目录，与现有链接保持一致）
PROJECT_ROOT="/home/hice1/xli3252/Desktop/diversity-eval"
CACHE_DIR="$PROJECT_ROOT/scratch/.cache"

# 确保scratch缓存目录存在
echo "设置缓存目录..."
mkdir -p "$CACHE_DIR/jukebox/models/5b"

# 确保软链接存在（如果还没有创建的话）
if [ ! -L ~/.cache ]; then
    echo "创建缓存目录软链接..."
    mkdir -p "$PROJECT_ROOT/scratch/.cache"
    ln -sf "$PROJECT_ROOT/scratch/.cache" ~/.cache
fi

# 下载Jukebox模型文件
echo "下载 Jukebox 模型文件..."
if [ ! -f "$CACHE_DIR/jukebox/models/5b/vqvae.pth.tar" ]; then
    echo "下载 vqvae.pth.tar..."
    wget -O "$CACHE_DIR/jukebox/models/5b/vqvae.pth.tar" https://openaipublic.azureedge.net/jukebox/models/5b/vqvae.pth.tar
else
    echo "vqvae.pth.tar 已存在，跳过下载"
fi

if [ ! -f "$CACHE_DIR/jukebox/models/5b/prior_level_2.pth.tar" ]; then
    echo "下载 prior_level_2.pth.tar..."
    wget -O "$CACHE_DIR/jukebox/models/5b/prior_level_2.pth.tar" https://openaipublic.azureedge.net/jukebox/models/5b/prior_level_2.pth.tar
else
    echo "prior_level_2.pth.tar 已存在，跳过下载"
fi

# 下载Jukebox源码
COMMIT_ID="08efbbc1d4ed1a3cef96e08a931944c8b4d63bb3"
if [ ! -d "jukebox_src" ]; then
    echo "下载 Jukebox 源码..."
    wget https://github.com/openai/jukebox/archive/${COMMIT_ID}.zip
    unzip ${COMMIT_ID}.zip
    rm ${COMMIT_ID}.zip
    mv jukebox-${COMMIT_ID} jukebox_src
else
    echo "Jukebox 源码已存在，跳过下载"
fi

# 应用补丁
echo "应用补丁..."
if [ -f "jukebox/make_models.py.patch" ]; then
    cp jukebox/make_models.py.patch .
    cd jukebox_src
    patch -p0 < ../make_models.py.patch || echo "补丁可能已经应用过了"
    cd ..
fi

# 安装Jukebox
echo "安装 Jukebox 库..."
cd jukebox_src
pip install -e .
cd ..

# 测试安装
echo "测试 Jukebox 安装..."
python -c "import jukebox; print('Jukebox 安装成功！')"

echo "Jukebox 安装完成！"