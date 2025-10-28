#!/bin/bash

# 官方 Jukebox 运行脚本
# 自动激活环境并运行LLark的main.py

set -e

# 检查参数
if [ $# -lt 2 ]; then
    echo "使用方法: $0 <input_dir> <output_dir> [其他参数]"
    echo ""
    echo "示例:"
    echo "  $0 /path/to/audio /path/to/output"
    echo "  $0 /path/to/audio /path/to/output --pool-frames-per-second 10"
    echo ""
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"
shift 2  # 移除前两个参数，剩下的作为额外参数

# 检查输入目录
if [ ! -d "$INPUT_DIR" ]; then
    echo "❌ 错误: 输入目录不存在: $INPUT_DIR"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 环境配置
ENV_NAME="jukebox"
WORK_DIR="/home/hice1/xli3252/Desktop/diversity-eval/external/llark"

cd "$WORK_DIR"

# 检查环境是否存在
if ! conda env list | grep -q "^$ENV_NAME "; then
    echo "❌ 错误: Jukebox 环境不存在"
    echo "请先运行: ./setup_official_jukebox.sh"
    exit 1
fi

# 检查LLark的main.py是否存在
if [ ! -f "jukebox/main.py" ]; then
    echo "❌ 错误: 找不到 jukebox/main.py"
    echo "请确保LLark的Jukebox模块存在"
    exit 1
fi

echo "🎵 使用官方 Jukebox 进行音频编码..."
echo "📁 输入目录: $INPUT_DIR"
echo "📁 输出目录: $OUTPUT_DIR"
echo "🔄 激活环境: $ENV_NAME"

# 激活环境并运行
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

echo "🚀 开始处理..."

# 运行LLark的main.py
python jukebox/main.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    "$@"

echo "✅ 处理完成！"
echo "📁 结果保存在: $OUTPUT_DIR"

# 退出环境
conda deactivate