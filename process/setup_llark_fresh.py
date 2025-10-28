#!/usr/bin/env python3
"""
Fresh setup script for LLark - complete reinstallation
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def run_command(cmd, description="", check=True):
    """Run a shell command with error handling"""
    print(f"🔧 {description}")
    print(f"   Command: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=check, 
                              capture_output=True, text=True)
        if result.stdout:
            print(f"   Output: {result.stdout.strip()}")
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Error: {e}")
        if e.stdout:
            print(f"   Stdout: {e.stdout}")
        if e.stderr:
            print(f"   Stderr: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    print(f"   Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major != 3 or version.minor < 8:
        print("   ❌ Python 3.8+ required")
        return False
    print("   ✅ Python version compatible")
    return True

def check_cuda():
    """Check CUDA availability"""
    print("🔥 Checking CUDA availability...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"   ✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            print("   ⚠️  CUDA not available, will use CPU")
            return False
    except ImportError:
        print("   ⚠️  PyTorch not installed yet")
        return False

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    
    llark_root = Path("/home/evev/diversity-eval/external/llark")
    requirements_file = llark_root / "requirements.txt"
    
    if not requirements_file.exists():
        print(f"   ❌ Requirements file not found: {requirements_file}")
        return False
    
    # Install requirements
    success = run_command(
        f"pip install -r {requirements_file}",
        "Installing requirements.txt"
    )
    
    if not success:
        print("   ⚠️  Some packages failed to install, trying individual installation...")
        
        # Try installing key packages individually
        key_packages = [
            "torch==1.13.0",
            "transformers==4.29.2",
            "datasets==2.10.1",
            "librosa",
            "numpy",
            "pandas",
            "scipy"
        ]
        
        for package in key_packages:
            run_command(f"pip install {package}", f"Installing {package}", check=False)
    
    return True

def setup_huggingface_auth():
    """Setup Hugging Face authentication"""
    print("🤗 Setting up Hugging Face authentication...")
    
    # Check if already logged in
    result = subprocess.run("huggingface-cli whoami", shell=True, 
                          capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"   ✅ Already logged in as: {result.stdout.strip()}")
        return True
    
    print("   ⚠️  Not logged in to Hugging Face")
    print("   Please run: huggingface-cli login")
    print("   Or set HF_TOKEN environment variable")
    
    # Check for HF_TOKEN
    if os.getenv("HF_TOKEN"):
        print("   ✅ HF_TOKEN environment variable found")
        return True
    
    return False

def setup_llark_model():
    """Setup LLark model structure"""
    print("🦙 Setting up LLark model...")
    
    llark_root = Path("/home/evev/diversity-eval/external/llark")
    checkpoint_dir = llark_root / "checkpoints" / "meta-llama" / "Llama-2-7b-chat-hf"
    
    # Create directory structure
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        from transformers import AutoTokenizer, AutoConfig
        
        model_name = "meta-llama/Llama-2-7b-chat-hf"
        
        # Download tokenizer
        print(f"   📥 Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False,
            trust_remote_code=True
        )
        tokenizer.save_pretrained(str(checkpoint_dir))
        
        # Download config
        print(f"   📥 Downloading config...")
        config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        config.save_pretrained(str(checkpoint_dir))
        
        print(f"   ✅ Model setup completed at: {checkpoint_dir}")
        return True
        
    except Exception as e:
        print(f"   ❌ Error setting up model: {e}")
        return False

def setup_llark_paths():
    """Setup LLark Python paths"""
    print("🛤️  Setting up Python paths...")
    
    llark_root = Path("/home/evev/diversity-eval/external/llark")
    
    # Add to Python path
    if str(llark_root) not in sys.path:
        sys.path.insert(0, str(llark_root))
        print(f"   ✅ Added to Python path: {llark_root}")
    
    # Check if m2t module can be imported
    try:
        sys.path.insert(0, str(llark_root))
        import m2t
        print("   ✅ m2t module can be imported")
        return True
    except ImportError as e:
        print(f"   ⚠️  m2t import issue: {e}")
        return False

def test_llark_import():
    """Test if LLark components can be imported"""
    print("🧪 Testing LLark imports...")
    
    llark_root = Path("/home/evev/diversity-eval/external/llark")
    sys.path.insert(0, str(llark_root))
    
    test_imports = [
        ("m2t.models.utils", "load_pretrained_model"),
        ("m2t.infer", "infer_with_prompt"),
        ("m2t.arguments", "ModelArguments"),
    ]
    
    success_count = 0
    for module_name, function_name in test_imports:
        try:
            module = __import__(module_name, fromlist=[function_name])
            getattr(module, function_name)
            print(f"   ✅ {module_name}.{function_name}")
            success_count += 1
        except Exception as e:
            print(f"   ❌ {module_name}.{function_name}: {e}")
    
    return success_count == len(test_imports)

def main():
    """Main setup function"""
    print("🚀 Starting fresh LLark setup...")
    print("=" * 50)
    
    # Step 1: Check Python version
    if not check_python_version():
        return False
    
    # Step 2: Check CUDA
    check_cuda()
    
    # Step 3: Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        return False
    
    # Step 4: Setup HF auth
    if not setup_huggingface_auth():
        print("⚠️  Hugging Face authentication not set up")
        print("   You may need to run: huggingface-cli login")
    
    # Step 5: Setup model
    if not setup_llark_model():
        print("❌ Failed to setup LLark model")
        return False
    
    # Step 6: Setup paths
    if not setup_llark_paths():
        print("⚠️  Python path setup issues")
    
    # Step 7: Test imports
    if not test_llark_import():
        print("⚠️  Some imports failed")
    
    print("\n" + "=" * 50)
    print("🎉 LLark setup completed!")
    print("\n📋 Next steps:")
    print("1. Test with: python external/llark/scripts/inference/infer_from_encodings_optimized.py --help")
    print("2. Make sure you have audio encodings in the expected directory")
    print("3. Run inference with your audio files")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)