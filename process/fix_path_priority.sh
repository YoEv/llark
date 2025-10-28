#!/bin/bash
# 修复 llama 命令 PATH 优先级问题

echo "🔧 修复 llama 命令 PATH 优先级..."

# 显示当前 PATH
echo "当前 PATH:"
echo $PATH | tr ':' '\n' | nl

# 临时移除 /snap/bin 从 PATH
export PATH=$(echo $PATH | tr ':' '\n' | grep -v '/snap/bin' | tr '\n' ':' | sed 's/:$//')

echo ""
echo "修改后的 PATH (移除 /snap/bin):"
echo $PATH | tr ':' '\n' | nl

# 测试 llama 命令
echo ""
echo "测试 llama 命令:"
which llama 2>/dev/null && echo "找到 llama: $(which llama)" || echo "未找到 llama 命令"

# 尝试查找 llama-stack 相关命令
echo ""
echo "查找 llama-stack 相关命令:"
find /usr/local/bin /usr/bin ~/.local/bin -name "*llama*" 2>/dev/null | head -10

# 测试 Python 模块方式调用
echo ""
echo "测试 Python 模块调用:"
python -c "import llama_stack; print('✅ llama_stack 模块可用')" 2>/dev/null || echo "❌ llama_stack 模块不可用"

# 尝试不同的 llama-stack 命令
echo ""
echo "尝试不同的命令格式:"
for cmd in "llama-stack" "llamastack" "python -m llama_stack"; do
    echo "测试: $cmd"
    if eval "$cmd --help" >/dev/null 2>&1; then
        echo "  ✅ $cmd 可用"
    else
        echo "  ❌ $cmd 不可用"
    fi
done

echo ""
echo "✅ PATH 修复完成！"
echo "💡 要永久修复，可以编辑 ~/.bashrc 或 ~/.profile"
echo ""
echo "🚀 现在可以尝试下载模型:"
echo "   python -m llama_stack model download"
echo "   或者使用我们找到的可用命令"