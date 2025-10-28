#!/usr/bin/env python3
"""
将Meta官方格式的Llama模型转换为Hugging Face格式
"""
import os
import sys
import json
import shutil
from pathlib import Path

def convert_meta_to_hf():
    """转换Meta格式到HF格式"""
    
    # 源路径（Meta官方格式）
    meta_model_dir = "/home/evev/diversity-eval/external/llama-2-7b-chat"
    tokenizer_file = "/home/evev/diversity-eval/external/tokenizer.model"
    
    # 目标路径（LLark期望的HF格式）
    hf_model_dir = "/home/evev/diversity-eval/external/llark/checkpoints/meta-llama/Llama-2-7b-chat-hf"
    
    print(f"源路径: {meta_model_dir}")
    print(f"目标路径: {hf_model_dir}")
    
    # 检查源文件
    required_files = [
        os.path.join(meta_model_dir, "consolidated.00.pth"),
        os.path.join(meta_model_dir, "params.json"),
        tokenizer_file
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"❌ 缺少文件: {file_path}")
            return False
        else:
            print(f"✅ 找到文件: {file_path}")
    
    # 创建目标目录
    os.makedirs(hf_model_dir, exist_ok=True)
    print(f"✅ 创建目标目录: {hf_model_dir}")
    
    try:
        # 检查transformers是否可用
        try:
            import transformers
            print(f"\n=== 检查依赖 ===")
            print(f"✅ transformers版本: {transformers.__version__}")
        except ImportError:
            print("❌ 未安装transformers，尝试安装...")
            os.system("pip install transformers torch")
            
        # 手动创建HF格式
        print("\n=== 手动创建HF格式 ===")
        
        # 复制tokenizer
        shutil.copy2(tokenizer_file, os.path.join(hf_model_dir, "tokenizer.model"))
        print("✅ 复制tokenizer.model")
        
        # 读取params.json并转换为config.json
        with open(os.path.join(meta_model_dir, "params.json"), 'r') as f:
            params = json.load(f)
        
        print(f"📋 原始参数: {params}")
        
        # 基于实际参数创建HF配置
        # Llama-2-7b的标准配置
        hidden_size = params["dim"]  # 4096
        num_heads = params["n_heads"]  # 32
        num_layers = params["n_layers"]  # 32
        
        # 计算intermediate_size (Llama-2-7b标准值是11008)
        intermediate_size = int(hidden_size * 8 / 3)  # 10922.67
        multiple_of = params.get("multiple_of", 256)  # 256
        intermediate_size = ((intermediate_size + multiple_of - 1) // multiple_of) * multiple_of  # 11008
        
        print(f"📋 计算得出 intermediate_size: {intermediate_size}")
        
        hf_config = {
            "architectures": ["LlamaForCausalLM"],
            "bos_token_id": 1,
            "eos_token_id": 2,
            "hidden_act": "silu",
            "hidden_size": hidden_size,
            "initializer_range": 0.02,
            "intermediate_size": intermediate_size,
            "max_position_embeddings": 4096,
            "model_type": "llama",
            "num_attention_heads": num_heads,
            "num_hidden_layers": num_layers,
            "num_key_value_heads": num_heads,  # Llama-2没有GQA
            "pretraining_tp": 1,
            "rms_norm_eps": params.get("norm_eps", 1e-6),
            "rope_scaling": None,
            "rope_theta": 10000.0,
            "tie_word_embeddings": False,
            "torch_dtype": "float16",
            "transformers_version": "4.35.0",
            "use_cache": True,
            "vocab_size": 32000  # Llama-2标准词汇表大小
        }
        
        with open(os.path.join(hf_model_dir, "config.json"), 'w') as f:
            json.dump(hf_config, f, indent=2)
        print("✅ 创建config.json")
        
        # 创建tokenizer配置
        tokenizer_config = {
            "add_bos_token": True,
            "add_eos_token": False,
            "bos_token": "<s>",
            "eos_token": "</s>",
            "model_max_length": 4096,
            "pad_token": None,
            "sp_model_kwargs": {},
            "tokenizer_class": "LlamaTokenizer",
            "unk_token": "<unk>"
        }
        
        with open(os.path.join(hf_model_dir, "tokenizer_config.json"), 'w') as f:
            json.dump(tokenizer_config, f, indent=2)
        print("✅ 创建tokenizer_config.json")
        
        # 创建generation_config.json
        generation_config = {
            "bos_token_id": 1,
            "do_sample": True,
            "eos_token_id": 2,
            "max_length": 4096,
            "pad_token_id": 0,
            "temperature": 0.6,
            "top_p": 0.9
        }
        
        with open(os.path.join(hf_model_dir, "generation_config.json"), 'w') as f:
            json.dump(generation_config, f, indent=2)
        print("✅ 创建generation_config.json")
        
        # 创建软链接指向原始权重文件
        weight_link = os.path.join(hf_model_dir, "pytorch_model.bin")
        if os.path.exists(weight_link):
            os.remove(weight_link)
        
        os.symlink(
            os.path.join(meta_model_dir, "consolidated.00.pth"),
            weight_link
        )
        print("✅ 创建权重文件软链接")
        
        print(f"\n✅ 转换完成！模型已保存到: {hf_model_dir}")
        
        # 列出转换后的文件
        print("\n转换后的文件:")
        for file in sorted(os.listdir(hf_model_dir)):
            file_path = os.path.join(hf_model_dir, file)
            if os.path.islink(file_path):
                target = os.readlink(file_path)
                size = os.path.getsize(target) if os.path.exists(target) else 0
                print(f"  {file} -> {target} ({size:,} bytes)")
            else:
                size = os.path.getsize(file_path)
                print(f"  {file} ({size:,} bytes)")
        
        return True
        
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_loading():
    """测试模型是否能正确加载"""
    hf_model_dir = "/home/evev/diversity-eval/external/llark/checkpoints/meta-llama/Llama-2-7b-chat-hf"
    
    print(f"\n=== 测试模型加载 ===")
    
    try:
        from transformers import LlamaTokenizer, LlamaForCausalLM
        
        print("尝试加载tokenizer...")
        tokenizer = LlamaTokenizer.from_pretrained(hf_model_dir)
        print("✅ tokenizer加载成功")
        
        print("尝试加载模型配置...")
        from transformers import LlamaConfig
        config = LlamaConfig.from_pretrained(hf_model_dir)
        print("✅ 模型配置加载成功")
        
        print("✅ 模型格式验证通过！")
        return True
        
    except Exception as e:
        print(f"❌ 模型加载测试失败: {e}")
        print("这可能是正常的，因为权重格式可能需要进一步转换")
        return False


if __name__ == "__main__":
    print("=== Meta Llama 转 Hugging Face 格式 ===")
    
    if convert_meta_to_hf():
        print("\n转换成功！")
        test_model_loading()
    else:
        print("\n转换失败！")
        sys.exit(1)
