#!/usr/bin/env python3
"""
Complete setup script for LLark model with Hugging Face access
"""

import os
import sys
import json
from pathlib import Path

def setup_llark_model():
    """Setup LLark model with proper directory structure and files"""
    print("🚀 Setting up LLark model with Hugging Face access...")
    
    # Set up paths
    llark_root = Path("/home/evev/diversity-eval/external/llark")
    checkpoint_dir = llark_root / "checkpoints" / "meta-llama" / "Llama-2-7b-chat-hf" / "checkpoint-100000"
    
    print(f"📁 Creating directory structure at: {checkpoint_dir}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        from transformers import AutoTokenizer, AutoConfig
        print("✅ transformers library imported successfully")
        
        model_name = "meta-llama/Llama-2-7b-chat-hf"
        
        # Download and save tokenizer
        print(f"📥 Downloading tokenizer for {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False,
            trust_remote_code=True,
            use_auth_token=True
        )
        
        # Save tokenizer to checkpoint directory
        tokenizer_path = checkpoint_dir / "tokenizer"
        tokenizer.save_pretrained(str(tokenizer_path))
        print(f"✅ Tokenizer saved to: {tokenizer_path}")
        
        # Download and save config
        print(f"📥 Downloading config for {model_name}...")
        config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_auth_token=True
        )
        
        # Save config to checkpoint directory
        config_path = checkpoint_dir / "config.json"
        config.save_pretrained(str(checkpoint_dir))
        print(f"✅ Config saved to: {config_path}")
        
        # Create a placeholder pytorch_model.bin (empty file for now)
        model_file = checkpoint_dir / "pytorch_model.bin"
        if not model_file.exists():
            print("📝 Creating placeholder model file...")
            # Create an empty file as placeholder
            model_file.touch()
            print(f"⚠️  Created placeholder: {model_file}")
            print("   Note: This is just a placeholder. You'll need actual LLark weights for inference.")
        
        # Create trainer_state.json
        trainer_state = {
            "best_metric": None,
            "best_model_checkpoint": str(checkpoint_dir),
            "epoch": 100000,
            "global_step": 100000,
            "is_hyper_param_search": False,
            "is_local_process_zero": True,
            "is_world_process_zero": True,
            "log_history": [],
            "max_steps": 100000,
            "num_train_epochs": 1,
            "total_flos": 0,
            "trial_name": None,
            "trial_params": None
        }
        
        trainer_state_path = checkpoint_dir / "trainer_state.json"
        with open(trainer_state_path, 'w') as f:
            json.dump(trainer_state, f, indent=2)
        print(f"✅ Created trainer_state.json: {trainer_state_path}")
        
        # Create training_args.bin placeholder
        training_args_path = checkpoint_dir / "training_args.bin"
        training_args_path.touch()
        print(f"✅ Created training_args.bin: {training_args_path}")
        
        # Print summary
        print("\n🎉 LLark model setup completed!")
        print(f"📂 Model directory: {checkpoint_dir}")
        print("📋 Created files:")
        for file_path in checkpoint_dir.rglob("*"):
            if file_path.is_file():
                print(f"   - {file_path.relative_to(checkpoint_dir)}")
        
        print("\n⚠️  Important Notes:")
        print("1. The pytorch_model.bin is currently a placeholder")
        print("2. For actual inference, you need trained LLark weights")
        print("3. You can now test the model loading with the fixed code")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: Failed to setup LLark model")
        print(f"Error details: {str(e)}")
        
        if "401" in str(e) or "authentication" in str(e).lower():
            print("\n🔑 Authentication issue detected. Please:")
            print("1. Make sure you're logged in: hf auth login")
            print("2. Accept the Llama-2 license at: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf")
            print("3. Check your token permissions")
        
        return False

if __name__ == "__main__":
    success = setup_llark_model()
    sys.exit(0 if success else 1)