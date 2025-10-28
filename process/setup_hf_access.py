#!/usr/bin/env python3
"""
Setup Hugging Face access for LLark model download.

This script helps you:
1. Install huggingface-cli if needed
2. Login to Hugging Face
3. Accept the Llama-2 license
4. Download the required model files
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd, check=True):
    """Run a shell command and return the result."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    return result

def check_huggingface_cli():
    """Check if huggingface-cli is installed."""
    result = run_command("which huggingface-cli", check=False)
    return result.returncode == 0

def install_huggingface_hub():
    """Install huggingface_hub package."""
    print("Installing huggingface_hub...")
    result = run_command("pip install huggingface_hub", check=False)
    return result.returncode == 0

def login_to_huggingface():
    """Guide user through Hugging Face login."""
    print("\n" + "="*60)
    print("HUGGING FACE LOGIN REQUIRED")
    print("="*60)
    print("You need to login to Hugging Face to access Llama-2 models.")
    print("\nSteps:")
    print("1. Go to https://huggingface.co/settings/tokens")
    print("2. Create a new token (read access is sufficient)")
    print("3. Copy the token")
    print("4. Run the login command below")
    print("\nAfter getting your token, run:")
    print("huggingface-cli login")
    print("\nOr set the environment variable:")
    print("export HUGGING_FACE_HUB_TOKEN=your_token_here")
    print("="*60)
    
    # Try to run login interactively
    try:
        subprocess.run(["huggingface-cli", "login"], check=True)
        return True
    except subprocess.CalledProcessError:
        print("Login failed. Please try manually.")
        return False
    except KeyboardInterrupt:
        print("\nLogin cancelled.")
        return False

def check_llama_access():
    """Check if user has access to Llama-2 model."""
    print("\nChecking access to meta-llama/Llama-2-7b-chat-hf...")
    
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        
        # Try to get model info
        model_info = api.model_info("meta-llama/Llama-2-7b-chat-hf")
        print("✓ Access confirmed!")
        return True
        
    except Exception as e:
        print(f"✗ Access denied: {e}")
        print("\nYou need to:")
        print("1. Accept the license at: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf")
        print("2. Make sure you're logged in with a valid token")
        return False

def download_model_files(checkpoint_dir):
    """Download only the necessary files (tokenizer and config)."""
    print(f"\nDownloading model files to: {checkpoint_dir}")
    
    # Create directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    try:
        from transformers import AutoTokenizer, AutoConfig
        
        # Download tokenizer
        print("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf",
            use_auth_token=True
        )
        tokenizer.save_pretrained(checkpoint_dir)
        print(f"✓ Tokenizer saved to {checkpoint_dir}")
        
        # Download config
        print("Downloading config...")
        config = AutoConfig.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf",
            use_auth_token=True
        )
        config.save_pretrained(checkpoint_dir)
        print(f"✓ Config saved to {checkpoint_dir}")
        
        print("\n" + "="*60)
        print("SUCCESS!")
        print("="*60)
        print(f"Model files downloaded to: {checkpoint_dir}")
        print("\nNote: This only includes tokenizer and config files.")
        print("For actual inference, you would need trained LLark weights.")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"Error downloading files: {e}")
        return False

def main():
    print("LLark Hugging Face Setup")
    print("=" * 40)
    
    # Check if huggingface-cli is available
    if not check_huggingface_cli():
        print("huggingface-cli not found. Installing huggingface_hub...")
        if not install_huggingface_hub():
            print("Failed to install huggingface_hub")
            sys.exit(1)
    
    print("✓ huggingface-cli is available")
    
    # Check if already logged in
    result = run_command("huggingface-cli whoami", check=False)
    if result.returncode == 0:
        print(f"✓ Already logged in as: {result.stdout.strip()}")
    else:
        print("Not logged in to Hugging Face")
        if not login_to_huggingface():
            print("Please login manually and run this script again")
            sys.exit(1)
    
    # Check access to Llama-2
    if not check_llama_access():
        print("\nPlease visit: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf")
        print("Accept the license and try again.")
        sys.exit(1)
    
    # Download model files
    checkpoint_dir = "checkpoints/meta-llama/Llama-2-7b-chat-hf/checkpoint-100000"
    if download_model_files(checkpoint_dir):
        print("\nSetup completed successfully!")
        print(f"You can now try running LLark inference with model path: {checkpoint_dir}")
    else:
        print("Setup failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()