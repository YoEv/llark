#!/usr/bin/env python3
"""
Setup LLark model using official Llama CLI
"""

import os
import sys
import subprocess
from pathlib import Path

def setup_with_llama_cli():
    """Setup LLark model using Llama CLI"""
    print("🚀 Setting up LLark model using Llama CLI...")
    
    # Set up paths
    llark_root = Path("/home/evev/diversity-eval/external/llark")
    checkpoint_dir = llark_root / "checkpoints" / "meta-llama" / "Llama-2-7b-chat-hf" / "checkpoint-100000"
    
    print(f"📁 Creating directory structure at: {checkpoint_dir}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Check if llama-stack is installed
        result = subprocess.run(['llama', '--version'], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Llama CLI not found. Installing...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'llama-stack'], check=True)
            print("✅ Llama CLI installed successfully")
        else:
            print("✅ Llama CLI found")
        
        # List available models
        print("📋 Listing available models...")
        result = subprocess.run(['llama', 'model', 'list'], capture_output=True, text=True)
        if result.returncode == 0:
            print("Available models:")
            print(result.stdout)
        
        # Prompt user for model selection
        print("\n🔍 To download Llama-2-7b-chat-hf, you can run:")
        print("llama model download --source meta --model-id Llama2-7b-chat-hf")
        print("\n💡 If you have a custom URL from the official website, use:")
        print("llama model download --source meta --model-id Llama2-7b-chat-hf --custom-url YOUR_URL")
        
        # Ask user if they want to proceed with download
        response = input("\n❓ Do you want to proceed with the download now? (y/n): ")
        if response.lower() == 'y':
            custom_url = input("🔗 Enter your custom URL (or press Enter to skip): ").strip()
            
            if custom_url:
                cmd = ['llama', 'model', 'download', '--source', 'meta', '--model-id', 'Llama2-7b-chat-hf', '--custom-url', custom_url]
            else:
                cmd = ['llama', 'model', 'download', '--source', 'meta', '--model-id', 'Llama2-7b-chat-hf']
            
            print(f"🚀 Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, cwd=str(llark_root))
            
            if result.returncode == 0:
                print("✅ Model downloaded successfully!")
                
                # Try to find the downloaded model and create symlinks
                print("🔗 Setting up model structure for LLark...")
                setup_model_structure(llark_root, checkpoint_dir)
            else:
                print("❌ Model download failed")
                return False
        else:
            print("ℹ️  You can download the model later using the commands above")
            
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def setup_model_structure(llark_root, checkpoint_dir):
    """Setup the model structure that LLark expects"""
    print("📁 Setting up model structure...")
    
    # Look for downloaded models in common locations
    possible_locations = [
        llark_root / "models",
        Path.home() / ".llama" / "checkpoints",
        Path("/tmp/llama_download"),
    ]
    
    for location in possible_locations:
        if location.exists():
            print(f"🔍 Checking {location}...")
            for model_dir in location.rglob("*Llama*2*7b*"):
                if model_dir.is_dir():
                    print(f"📦 Found model at: {model_dir}")
                    
                    # Create symlinks or copy files to expected location
                    try:
                        if not checkpoint_dir.exists():
                            checkpoint_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Copy or link model files
                        for file in model_dir.rglob("*"):
                            if file.is_file():
                                target = checkpoint_dir / file.name
                                if not target.exists():
                                    try:
                                        target.symlink_to(file)
                                        print(f"🔗 Linked: {file.name}")
                                    except:
                                        # Fallback to copy if symlink fails
                                        import shutil
                                        shutil.copy2(file, target)
                                        print(f"📋 Copied: {file.name}")
                        
                        print(f"✅ Model structure setup complete at: {checkpoint_dir}")
                        return True
                        
                    except Exception as e:
                        print(f"❌ Error setting up structure: {e}")
    
    print("⚠️  Could not find downloaded model. You may need to manually set up the structure.")
    return False

if __name__ == "__main__":
    success = setup_with_llama_cli()
    sys.exit(0 if success else 1)