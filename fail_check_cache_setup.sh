#!/bin/bash

echo "检查缓存设置..."

PROJECT_ROOT="/home/hice1/xli3252/Desktop/diversity-eval"

echo "1. 检查 scratch 目录："
ls -la "$PROJECT_ROOT/scratch/" 2>/dev/null || echo "scratch 目录不存在"

echo "2. 检查 ~/.cache 软链接："
if [ -L ~/.cache ]; then
    echo "~/.cache 是软链接，指向: $(readlink ~/.cache)"
else
    echo "~/.cache 不是软链接"
fi

echo "3. 检查缓存目录空间："
df -h "$PROJECT_ROOT/scratch" 2>/dev/null || echo "无法检查 scratch 目录空间"

echo "4. 检查 Jukebox 缓存目录："
if [ -d ~/.cache/jukebox ]; then
    echo "Jukebox 缓存目录存在"
    ls -la ~/.cache/jukebox/
else
    echo "Jukebox 缓存目录不存在"
fi