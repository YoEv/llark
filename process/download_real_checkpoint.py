#!/usr/bin/env python3
"""
下载真正的 Llama-2-7b-chat-hf checkpoint
"""

import os
import sys
from pathlib import Path

def download_llama_checkpoint():
    """下载 Llama-2-7b-chat-hf checkpoint"""
    
    try:
        from huggingface_hub import snapshot_download, login
        import torch
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("请安装: pip install huggingface_hub torch")
        return False
    
    # 设置下载路径
    base_path = Path("/home/evev/diversity-eval/external/llark/checkpoints/meta-llama")
    model_path = base_path / "Llama-2-7b-chat-hf"
    checkpoint_path = model_path / "checkpoint-100000"
    
    print(f"🎯 目标路径: {model_path}")
    
    # 创建目录
    model_path.mkdir(parents=True, exist_ok=True)
    
    # 检查是否需要 Hugging Face 登录
    print("🔐 检查 Hugging Face 访问权限...")
    
    try:
        # 尝试下载模型
        print("🚀 开始下载 Llama-2-7b-chat-hf...")
        print("⚠️  这可能需要很长时间，模型大小约 13GB")
        
        downloaded_path = snapshot_download(
            repo_id="meta-llama/Llama-2-7b-chat-hf",
            local_dir=str(model_path),
            local_dir_use_symlinks=False,
            resume_download=True
        )
        
        print(f"✅ 模型下载完成: {downloaded_path}")
        
        # 如果需要，创建 checkpoint-100000 目录的符号链接
        if not checkpoint_path.exists():
            print(f"🔗 创建 checkpoint 链接: {checkpoint_path}")
            checkpoint_path.symlink_to(model_path, target_is_directory=True)
        
        # 验证关键文件
        required_files = [
            "config.json",
            "tokenizer_config.json", 
            "pytorch_model.bin.index.json",
            "tokenizer.model"
        ]
        
        print("\n📋 验证下载的文件:")
        for file_name in required_files:
            file_path = model_path / file_name
            if file_path.exists():
                size = file_path.stat().st_size / (1024*1024)  # MB
                print(f"  ✅ {file_name} ({size:.1f} MB)")
            else:
                print(f"  ❌ {file_name} (缺失)")
        
        print(f"\n🎉 Llama-2-7b-chat-hf 下载完成!")
        print(f"📁 模型路径: {model_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        
        if "gated repo" in str(e).lower() or "access" in str(e).lower():
            print("\n🔐 这是一个受限模型，需要:")
            print("1. 在 https://huggingface.co/meta-llama/Llama-2-7b-chat-hf 申请访问权限")
            print("2. 使用 huggingface-cli login 登录")
            print("3. 或设置 HF_TOKEN 环境变量")
            
        return False

def main():
    """主函数"""
    print("🦙 Llama-2-7b-chat-hf 下载器")
    print("=" * 50)
    
    if download_llama_checkpoint():
        print("\n✅ 下载成功! 现在可以使用真正的 checkpoint 了")
    else:
        print("\n❌ 下载失败，继续使用最小化模型进行测试")
        print("💡 或者手动设置 Hugging Face 访问权限后重试")

if __name__ == "__main__":
    main()