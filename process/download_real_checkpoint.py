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
    
    # 设置下载路径 - 使用scratch目录避免磁盘配额限制
    scratch_path = Path.home() / "scratch" / "llark_models"
    model_path = scratch_path / "meta-llama" / "Llama-2-7b-chat-hf"
    
    # 在项目目录创建符号链接
    current_dir = Path(__file__).parent.parent  # external/llark/
    project_base_path = current_dir / "checkpoints" / "meta-llama"
    project_model_path = project_base_path / "Llama-2-7b-chat-hf"
    checkpoint_path = project_base_path / "checkpoint-100000"
    
    print(f"🎯 目标路径: {model_path}")
    
    model_path.mkdir(parents=True, exist_ok=True)
    
    # 检查 Hugging Face token
    print("🔐 检查 Hugging Face 访问权限...")
    
    # 尝试从环境变量获取 token
    hf_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
    
    if hf_token:
        print("✅ 找到 HF_TOKEN 环境变量")
        try:
            login(token=hf_token)
            print("✅ Hugging Face 登录成功")
        except Exception as e:
            print(f"⚠️  登录警告: {e}")
    else:
        print("⚠️  未找到 HF_TOKEN 环境变量")
        print("💡 如果下载失败，请设置: export HF_TOKEN=your_token_here")
    
    # 检查是否已经存在完整的模型文件
    required_files = [
        "config.json",
        "tokenizer_config.json", 
        "pytorch_model.bin.index.json",
        "tokenizer.model",
        "pytorch_model-00001-of-00002.bin",
        "pytorch_model-00002-of-00002.bin"
    ]
    
    all_files_exist = True
    missing_files = []
    
    for file_name in required_files:
        file_path = model_path / file_name
        if not file_path.exists():
            all_files_exist = False
            missing_files.append(file_name)
        elif file_name.endswith('.bin') and file_path.stat().st_size < 1024*1024:  # 检查.bin文件大小至少1MB
            all_files_exist = False
            missing_files.append(f"{file_name} (文件不完整)")
    
    if all_files_exist:
        print("✅ 发现已存在的完整模型文件，跳过下载")
        print(f"📁 模型路径: {model_path}")
        return True
    else:
        print(f"⚠️  检测到缺失或不完整的文件: {', '.join(missing_files)}")
        print("🔄 将重新下载模型...")
    
    try:
        # 尝试下载模型
        print("🚀 开始下载 Llama-2-7b-chat-hf...")
        print("⚠️  这可能需要很长时间，模型大小约 13GB")
        
        downloaded_path = snapshot_download(
            repo_id="meta-llama/Llama-2-7b-chat-hf",
            local_dir=str(model_path),
            local_dir_use_symlinks=False,
            resume_download=True,
            token=hf_token  # 传递 token
        )
        
        print(f"✅ 模型下载完成: {downloaded_path}")
        
        # 创建项目目录到scratch目录的符号链接
        project_model_path.parent.mkdir(parents=True, exist_ok=True)
        if project_model_path.exists() and project_model_path.is_symlink():
            project_model_path.unlink()
        elif project_model_path.exists():
            import shutil
            shutil.rmtree(project_model_path)
            
        print(f"🔗 创建项目链接: {project_model_path} -> {model_path}")
        project_model_path.symlink_to(model_path, target_is_directory=True)
        
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
        
        if "gated repo" in str(e).lower() or "access" in str(e).lower() or "401" in str(e):
            print("\n🔐 这是一个受限模型，需要:")
            print("1. 在 https://huggingface.co/meta-llama/Llama-2-7b-chat-hf 申请访问权限")
            print("2. 获取你的 Hugging Face token:")
            print("   - 访问 https://huggingface.co/settings/tokens")
            print("   - 创建一个新的 token (需要 'Read' 权限)")
            print("3. 设置环境变量:")
            print("   export HF_TOKEN=your_token_here")
            print("4. 或者使用 huggingface-cli login 命令登录")
            print("\n💡 你提到已经申请了访问权限，现在只需要设置 token:")
            print("   export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
            
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