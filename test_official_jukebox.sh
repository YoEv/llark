#!/bin/bash

# 官方 Jukebox 测试脚本

set -e

echo "🧪 测试官方 Jukebox 安装..."

ENV_NAME="jukebox"
WORK_DIR="/home/hice1/xli3252/Desktop/diversity-eval/external/llark"

cd "$WORK_DIR"

# 检查环境是否存在
if ! conda env list | grep -q "^$ENV_NAME "; then
    echo "❌ 错误: Jukebox 环境不存在"
    echo "请先运行: ./setup_official_jukebox.sh"
    exit 1
fi

# 激活环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

echo "🔍 检查Python版本..."
python --version

echo "🔍 检查PyTorch版本..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

echo "🔍 检查CUDA可用性..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

echo "🔍 检查Jukebox模块..."
python -c "
try:
    import jukebox
    print('✅ jukebox 模块导入成功')
    
    from jukebox.hparams import Hyperparams, setup_hparams
    print('✅ hparams 模块导入成功')
    
    from jukebox.make_models import MODELS, make_prior, make_vqvae
    print('✅ make_models 模块导入成功')
    
    from jukebox.utils.dist_utils import setup_dist_from_mpi
    print('✅ dist_utils 模块导入成功')
    
    print('✅ 所有核心模块导入正常')
except ImportError as e:
    print(f'❌ 模块导入失败: {e}')
    exit(1)
"

echo "🔍 检查模型文件..."
CACHE_DIR="/home/hice1/xli3252/Desktop/diversity-eval/scratch/.cache"
if [ -f "$CACHE_DIR/jukebox/models/5b/vqvae.pth.tar" ]; then
    echo "✅ vqvae.pth.tar 存在"
else
    echo "⚠️  vqvae.pth.tar 不存在"
fi

if [ -f "$CACHE_DIR/jukebox/models/5b/prior_level_2.pth.tar" ]; then
    echo "✅ prior_level_2.pth.tar 存在"
else
    echo "⚠️  prior_level_2.pth.tar 不存在"
fi

echo "🔍 检查LLark的main.py..."
if [ -f "jukebox/main.py" ]; then
    echo "✅ LLark的main.py 存在"
else
    echo "❌ LLark的main.py 不存在"
fi

conda deactivate

echo ""
echo "🎉 测试完成！"
echo ""
echo "如果所有检查都通过，你可以使用以下命令运行："
echo "  ./run_official_jukebox.sh /path/to/audio /path/to/output"