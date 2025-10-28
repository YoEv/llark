#!/usr/bin/env python3
"""
Setup script for LLark model - downloads base Llama-2 model and sets up directory structure.

This script helps set up the model structure needed for LLark inference.
Since LLark doesn't provide pre-trained models, you need either:
1. A base Llama-2 model for testing (this script)
2. A trained LLark checkpoint

Usage:
    python setup_llark_model.py --checkpoint-dir checkpoints/meta-llama/Llama-2-7b-chat-hf/checkpoint-100000
"""

import argparse
import os
import sys
from pathlib import Path

def setup_base_model(checkpoint_dir):
    """
    Set up base Llama-2 model structure for testing.
    
    Note: This creates a basic structure but you'll need actual trained LLark weights
    for real inference.
    """
    print(f"Setting up model directory: {checkpoint_dir}")
    
    # Create directory structure
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Check if we can download from Hugging Face
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        print("Downloading base Llama-2-7b-chat-hf model...")
        print("Note: This requires Hugging Face access to meta-llama models")
        print("You may need to:")
        print("1. Accept the license at https://huggingface.co/meta-llama/Llama-2-7b-chat-hf")
        print("2. Login with: huggingface-cli login")
        
        # Download tokenizer
        print("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        tokenizer.save_pretrained(checkpoint_dir)
        print(f"✓ Tokenizer saved to {checkpoint_dir}")
        
        # Download model config (not the full model weights)
        print("Downloading model config...")
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        config.save_pretrained(checkpoint_dir)
        print(f"✓ Config saved to {checkpoint_dir}")
        
        # Create a placeholder for model weights
        placeholder_file = os.path.join(checkpoint_dir, "pytorch_model.bin")
        if not os.path.exists(placeholder_file):
            print(f"Creating placeholder model file: {placeholder_file}")
            print("WARNING: This is just a placeholder - you need actual LLark weights!")
            
            # Create minimal placeholder
            import torch
            placeholder_weights = {"placeholder": torch.tensor([1.0])}
            torch.save(placeholder_weights, placeholder_file)
        
        print("\n" + "="*60)
        print("IMPORTANT NOTES:")
        print("="*60)
        print("1. This setup only provides the tokenizer and config files.")
        print("2. You need actual LLark model weights for inference.")
        print("3. LLark paper states: 'Note that this paper is not accompanied with any trained models.'")
        print("4. You would need to either:")
        print("   - Train your own LLark model using the provided training scripts")
        print("   - Obtain trained weights from another source")
        print("5. The current placeholder model will not work for actual inference.")
        print("="*60)
        
        return True
        
    except ImportError:
        print("Error: transformers library not found")
        return False
    except Exception as e:
        print(f"Error downloading model: {e}")
        print("\nThis might be because:")
        print("1. You don't have access to meta-llama models on Hugging Face")
        print("2. You need to login: huggingface-cli login")
        print("3. You need to accept the license at https://huggingface.co/meta-llama/Llama-2-7b-chat-hf")
        return False

def check_existing_model(checkpoint_dir):
    """Check if model files already exist."""
    if not os.path.exists(checkpoint_dir):
        return False
    
    required_files = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    existing_files = os.listdir(checkpoint_dir)
    
    print(f"Checking existing model at: {checkpoint_dir}")
    print(f"Found files: {existing_files}")
    
    for file in required_files:
        if file in existing_files:
            print(f"✓ Found {file}")
        else:
            print(f"✗ Missing {file}")
            return False
    
    # Check for model weights
    weight_files = [f for f in existing_files if f.endswith(('.bin', '.safetensors'))]
    if weight_files:
        print(f"✓ Found model weights: {weight_files}")
        return True
    else:
        print("⚠ No model weight files found")
        return False

def main():
    parser = argparse.ArgumentParser(description="Setup LLark model directory")
    parser.add_argument(
        "--checkpoint-dir", 
        default="checkpoints/meta-llama/Llama-2-7b-chat-hf/checkpoint-100000",
        help="Directory to set up the model checkpoint"
    )
    parser.add_argument(
        "--force", 
        action="store_true",
        help="Force re-download even if files exist"
    )
    
    args = parser.parse_args()
    
    # Convert to absolute path
    checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    
    print("LLark Model Setup")
    print("="*50)
    print(f"Target directory: {checkpoint_dir}")
    
    # Check if model already exists
    if not args.force and check_existing_model(checkpoint_dir):
        print("\n✓ Model files already exist!")
        print("Use --force to re-download")
        return
    
    # Set up the model
    if setup_base_model(checkpoint_dir):
        print(f"\n✓ Model setup completed at: {checkpoint_dir}")
        print("\nNext steps:")
        print("1. Obtain actual LLark model weights")
        print("2. Replace the placeholder pytorch_model.bin with real weights")
        print("3. Test with: python scripts/inference/infer_from_encodings.py")
    else:
        print("\n✗ Model setup failed")
        sys.exit(1)

if __name__ == "__main__":
    main()