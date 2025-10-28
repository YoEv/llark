#!/usr/bin/env python3
"""
修复 llama 命令冲突问题
"""

import subprocess
import sys
import os
from pathlib import Path

def fix_llama_command():
    """修复 llama 命令冲突问题"""
    
    print("🔧 修复 llama 命令冲突问题...")
    print("=" * 50)
    
    # 1. 检查 llama-stack 的正确命令
    print("\n1. 查找 llama-stack 的正确命令:")
    
    # 尝试不同的可能命令名称
    possible_commands = [
        'llama-stack',
        'llamastack', 
        'llama_stack',
        'python -m llama_stack',
        'python -m llama_stack.cli'
    ]
    
    working_command = None
    
    for cmd in possible_commands:
        try:
            print(f"   测试命令: {cmd}")
            if cmd.startswith('python'):
                result = subprocess.run(cmd.split(), capture_output=True, text=True, timeout=5)
            else:
                result = subprocess.run([cmd, '--help'], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0 and ('llama' in result.stdout.lower() or 'model' in result.stdout.lower()):
                working_command = cmd
                print(f"   ✅ 找到工作命令: {cmd}")
                break
            else:
                print(f"   ❌ 命令不工作")
                
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError) as e:
            print(f"   ❌ 命令失败: {e}")
    
    if not working_command:
        print("\n2. 尝试直接使用 Python 模块:")
        try:
            # 尝试导入 llama_stack 模块
            result = subprocess.run([sys.executable, '-c', 'import llama_stack; print("模块导入成功")'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("   ✅ llama_stack 模块可以导入")
                
                # 尝试查找模块的命令行接口
                result = subprocess.run([sys.executable, '-c', 
                    'import llama_stack; import pkgutil; print([name for _, name, _ in pkgutil.iter_modules(llama_stack.__path__)])'], 
                    capture_output=True, text=True)
                print(f"   模块内容: {result.stdout.strip()}")
                
            else:
                print(f"   ❌ 模块导入失败: {result.stderr}")
                
        except Exception as e:
            print(f"   ❌ 检查失败: {e}")
    
    # 3. 创建解决方案
    print(f"\n3. 创建解决方案:")
    
    if working_command:
        print(f"   使用工作命令: {working_command}")
        create_wrapper_script(working_command)
    else:
        print("   未找到工作的 llama-stack 命令")
        print("   建议使用最小化模型设置方案")
        
    # 4. 提供替代方案
    print(f"\n4. 替代方案 - 最小化模型设置:")
    print("   由于 llama 命令冲突，我们可以:")
    print("   a) 使用 setup_minimal_model.py 创建测试用的模型结构")
    print("   b) 绕过复杂的模型下载过程")
    print("   c) 直接测试 LLark 的模型加载功能")
    
    print("\n" + "=" * 50)
    print("🎯 修复分析完成!")

def create_wrapper_script(working_command):
    """创建包装脚本来使用正确的 llama 命令"""
    
    wrapper_path = Path("/home/evev/diversity-eval/external/llark/llama_stack_wrapper.py")
    
    wrapper_content = f'''#!/usr/bin/env python3
"""
LLama Stack 包装脚本
绕过系统中冲突的 llama 命令
"""

import subprocess
import sys

def main():
    """运行正确的 llama-stack 命令"""
    
    # 使用工作的命令
    cmd = "{working_command}"
    
    # 传递所有参数
    if len(sys.argv) > 1:
        if cmd.startswith('python'):
            full_cmd = cmd.split() + sys.argv[1:]
        else:
            full_cmd = [cmd] + sys.argv[1:]
    else:
        if cmd.startswith('python'):
            full_cmd = cmd.split() + ['--help']
        else:
            full_cmd = [cmd, '--help']
    
    try:
        result = subprocess.run(full_cmd)
        sys.exit(result.returncode)
    except Exception as e:
        print(f"错误: {{e}}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
    
    with open(wrapper_path, 'w') as f:
        f.write(wrapper_content)
    
    # 添加执行权限
    wrapper_path.chmod(0o755)
    
    print(f"   ✅ 创建包装脚本: {wrapper_path}")
    print(f"   使用方法: python {wrapper_path} model list")

if __name__ == "__main__":
    fix_llama_command()