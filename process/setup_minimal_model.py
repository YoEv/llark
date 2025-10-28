#!/usr/bin/env python3
"""
最小化模型设置脚本
创建LLark所需的基本文件结构，用于测试代码修复
"""

import os
import json
import torch
from pathlib import Path

def create_minimal_model_structure():
    """创建最小化的模型文件结构"""
    
    # 目标路径
    model_path = Path("/home/evev/diversity-eval/external/llark/checkpoints/meta-llama/Llama-2-7b-chat-hf/checkpoint-100000")
    
    print(f"🎯 创建模型目录: {model_path}")
    model_path.mkdir(parents=True, exist_ok=True)
    
    # 1. 创建 config.json
    config = {
        "architectures": ["LlamaForCausalLM"],
        "bos_token_id": 1,
        "eos_token_id": 2,
        "hidden_act": "silu",
        "hidden_size": 4096,
        "initializer_range": 0.02,
        "intermediate_size": 11008,
        "max_position_embeddings": 4096,
        "model_type": "llama",
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "num_key_value_heads": 32,
        "pretraining_tp": 1,
        "rms_norm_eps": 1e-06,
        "rope_scaling": None,
        "rope_theta": 10000.0,
        "tie_word_embeddings": False,
        "torch_dtype": "float16",
        "transformers_version": "4.31.0",
        "use_cache": True,
        "vocab_size": 32000
    }
    
    config_path = model_path / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✅ 创建 config.json")
    
    # 2. 创建 tokenizer_config.json
    tokenizer_config = {
        "add_bos_token": True,
        "add_eos_token": False,
        "bos_token": "<s>",
        "eos_token": "</s>",
        "model_max_length": 4096,
        "pad_token": None,
        "sp_model_kwargs": {},
        "tokenizer_class": "LlamaTokenizer",
        "unk_token": "<unk>",
        "use_default_system_prompt": False
    }
    
    tokenizer_config_path = model_path / "tokenizer_config.json"
    with open(tokenizer_config_path, 'w') as f:
        json.dump(tokenizer_config, f, indent=2)
    print(f"✅ 创建 tokenizer_config.json")
    
    # 3. 创建 trainer_state.json
    trainer_state = {
        "best_metric": None,
        "best_model_checkpoint": None,
        "epoch": 1.0,
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
    
    trainer_state_path = model_path / "trainer_state.json"
    with open(trainer_state_path, 'w') as f:
        json.dump(trainer_state, f, indent=2)
    print(f"✅ 创建 trainer_state.json")
    
    # 4. 创建 training_args.bin (空文件)
    training_args_path = model_path / "training_args.bin"
    with open(training_args_path, 'wb') as f:
        f.write(b'')
    print(f"✅ 创建 training_args.bin")
    
    # 5. 创建 pytorch_model.bin 占位符
    pytorch_model_path = model_path / "pytorch_model.bin"
    
    # 创建一个最小的占位符模型状态字典
    placeholder_state_dict = {
        'model.embed_tokens.weight': torch.randn(32000, 4096, dtype=torch.float16),
        'model.norm.weight': torch.ones(4096, dtype=torch.float16),
        'lm_head.weight': torch.randn(32000, 4096, dtype=torch.float16)
    }
    
    # 添加一些层的占位符权重
    for i in range(2):  # 只创建前2层作为示例
        placeholder_state_dict.update({
            f'model.layers.{i}.self_attn.q_proj.weight': torch.randn(4096, 4096, dtype=torch.float16),
            f'model.layers.{i}.self_attn.k_proj.weight': torch.randn(4096, 4096, dtype=torch.float16),
            f'model.layers.{i}.self_attn.v_proj.weight': torch.randn(4096, 4096, dtype=torch.float16),
            f'model.layers.{i}.self_attn.o_proj.weight': torch.randn(4096, 4096, dtype=torch.float16),
            f'model.layers.{i}.mlp.gate_proj.weight': torch.randn(11008, 4096, dtype=torch.float16),
            f'model.layers.{i}.mlp.up_proj.weight': torch.randn(11008, 4096, dtype=torch.float16),
            f'model.layers.{i}.mlp.down_proj.weight': torch.randn(4096, 11008, dtype=torch.float16),
            f'model.layers.{i}.input_layernorm.weight': torch.ones(4096, dtype=torch.float16),
            f'model.layers.{i}.post_attention_layernorm.weight': torch.ones(4096, dtype=torch.float16),
        })
    
    torch.save(placeholder_state_dict, pytorch_model_path)
    print(f"✅ 创建 pytorch_model.bin 占位符 ({pytorch_model_path.stat().st_size / 1024 / 1024:.1f} MB)")
    
    # 6. 创建 special_tokens_map.json
    special_tokens = {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>"
    }
    
    special_tokens_path = model_path / "special_tokens_map.json"
    with open(special_tokens_path, 'w') as f:
        json.dump(special_tokens, f, indent=2)
    print(f"✅ 创建 special_tokens_map.json")
    
    print(f"\n🎉 最小化模型结构创建完成!")
    print(f"📁 模型路径: {model_path}")
    print(f"📋 创建的文件:")
    for file_path in model_path.iterdir():
        if file_path.is_file():
            size = file_path.stat().st_size
            if size > 1024 * 1024:
                size_str = f"{size / 1024 / 1024:.1f} MB"
            elif size > 1024:
                size_str = f"{size / 1024:.1f} KB"
            else:
                size_str = f"{size} B"
            print(f"   - {file_path.name} ({size_str})")
    
    print(f"\n⚠️  注意: pytorch_model.bin 是占位符文件，包含随机权重")
    print(f"   实际推理需要使用LLark训练后的权重")

if __name__ == "__main__":
    try:
        create_minimal_model_structure()
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()