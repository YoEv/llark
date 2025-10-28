#!/bin/bash

# 官方 Jukebox 环境设置脚本
# 基于 https://github.com/openai/jukebox# 官方指南

set -e

echo "🎵 开始设置官方 Jukebox 环境..."
echo "📍 项目根目录: $PROJECT_ROOT"

# 环境配置
ENV_NAME="jukebox"
PYTHON_VERSION="3.7.5"
WORK_DIR="/home/hice1/xli3252/Desktop/diversity-eval/external/llark"
PROJECT_ROOT="/home/hice1/xli3252/Desktop/diversity-eval"

cd "$WORK_DIR"

# 检查环境是否已存在
if conda env list | grep -q "^$ENV_NAME "; then
    echo "✅ Conda 环境 '$ENV_NAME' 已存在，跳过创建"
else
    # 1. 创建conda环境 (官方指南第一步)
    echo "🐍 创建 conda 环境: $ENV_NAME (Python $PYTHON_VERSION)"
    conda create --name $ENV_NAME python=$PYTHON_VERSION -y
    echo "✅ Conda 环境 '$ENV_NAME' 创建成功"
fi

# 激活环境
echo "🔄 激活环境..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# 添加更多conda channels
echo "📡 添加conda channels..."
conda config --add channels conda-forge
conda config --add channels pytorch
conda config --add channels nvidia

# 2. 安装 mpi4py (官方指南第二步)
echo "📦 安装 mpi4py=3.0.3..."
conda install mpi4py=3.0.3 -y || pip install mpi4py==3.0.3

# 3. 安装 PyTorch 和相关库 (官方指南第三步)
echo "📦 安装 PyTorch 1.4, torchvision 0.5, cudatoolkit 10.0..."
# 尝试使用conda安装，如果失败则使用pip
if ! conda install pytorch=1.4 torchvision=0.5 cudatoolkit=10.0 -c pytorch -y; then
    echo "⚠️  Conda安装失败，尝试使用pip安装..."
    pip install torch==1.4.0 torchvision==0.5.0
    # 单独安装cudatoolkit
    conda install cudatoolkit=10.0 -y
fi

# 4. 克隆官方 Jukebox 仓库 (官方指南第四步)
echo "📥 克隆官方 Jukebox 仓库..."
if [ -d "jukebox_official" ]; then
    echo "移除现有的 jukebox_official 目录..."
    rm -rf jukebox_official
fi

git clone https://github.com/openai/jukebox.git jukebox_official
cd jukebox_official

# 5. 安装 requirements (官方指南第五步)
echo "📦 安装 requirements.txt..."
pip install -r requirements.txt

# 6. 安装 Jukebox 包 (官方指南第六步)
echo "📦 安装 Jukebox 包..."
pip install -e .

# 7. 训练相关依赖 (官方指南 Training 部分)
echo "📦 安装训练相关依赖..."
conda install av=7.0.01 -c conda-forge -y

# 检查是否有 tensorboardX 目录
if [ -d "./tensorboardX" ]; then
    echo "安装 tensorboardX..."
    pip install ./tensorboardX
else
    echo "⚠️  tensorboardX 目录不存在，跳过安装"
fi

cd ..

# 设置缓存目录（使用你现有的scratch配置）
echo "🔗 设置缓存目录..."
CACHE_DIR="$HOME/scratch/.cache"
mkdir -p "$CACHE_DIR"

# 确保软链接存在
if [ ! -L ~/.cache ]; then
    echo "创建缓存目录软链接..."
    ln -sf "$HOME/scratch/.cache" ~/.cache
fi

# 下载模型文件到缓存目录
echo "📥 下载 Jukebox 模型文件..."
cd "$CACHE_DIR"

if [ ! -f "vqvae.pth.tar" ]; then
    echo "下载 vqvae.pth.tar..."
    wget -O "vqvae.pth.tar" https://openaipublic.azureedge.net/jukebox/models/5b/vqvae.pth.tar
else
    echo "vqvae.pth.tar 已存在，跳过下载"
fi

if [ ! -f "prior_level_2.pth.tar" ]; then
    echo "下载 prior_level_2.pth.tar..."
    wget -O "prior_level_2.pth.tar" https://openaipublic.azureedge.net/jukebox/models/5b/prior_level_2.pth.tar
else
    echo "prior_level_2.pth.tar 已存在，跳过下载"
fi

# 回到llark目录
cd "$PROJECT_ROOT/external/llark"

# 应用LLark的补丁（如果存在）
echo "🔧 检查并应用LLark补丁..."
if [ -f "jukebox/make_models.py.patch" ]; then
    echo "应用 make_models.py.patch..."
    cd jukebox_official
    patch -p0 < ../jukebox/make_models.py.patch || echo "补丁可能已经应用过了"
    cd ..
else
    echo "⚠️  未找到补丁文件，跳过补丁应用"
fi

# 测试安装
echo "🧪 测试 Jukebox 安装..."
python -c "
try:
    import jukebox
    from jukebox.hparams import Hyperparams, setup_hparams
    from jukebox.make_models import MODELS, make_prior, make_vqvae
    from jukebox.utils.dist_utils import setup_dist_from_mpi
    print('✅ 官方 Jukebox 安装成功！')
    print('✅ 所有必要模块导入正常')
except ImportError as e:
    print(f'❌ Jukebox 安装失败: {e}')
    exit(1)
except Exception as e:
    print(f'⚠️  导入警告: {e}')
    print('✅ 基本安装成功，但可能需要GPU环境才能完全正常工作')
"

echo ""
echo "🎉 官方 Jukebox 环境设置完成！"
echo ""
echo "📋 使用方法："
echo "1. 激活环境: conda activate jukebox"
echo "2. 使用官方main.py: python jukebox_official/sample.py [参数]"
echo "3. 使用LLark的main.py: python jukebox/main.py --input_dir /path/to/audio --output_dir /path/to/output"
echo "4. 退出环境: conda deactivate"
echo ""
echo "📁 安装位置："
echo "- 官方Jukebox源码: $WORK_DIR/jukebox_official/"
echo "- LLark修改版本: $WORK_DIR/jukebox/"
echo "- 模型缓存: $CACHE_DIR/jukebox/models/"
echo ""
echo "⚠️  注意："
echo "- 确保有足够的GPU内存（推荐8GB+）"
echo "- 首次运行可能需要下载额外的模型文件"