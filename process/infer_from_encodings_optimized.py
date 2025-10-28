#!/usr/bin/env python3

import os
import sys
import torch
import gc
import glob
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
import transformers
from tqdm import tqdm

# Add LLark root to path
llark_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, llark_root)

from m2t.models.utils import load_pretrained_model
from m2t.data_modules import make_mm_config
from m2t.infer import infer_with_prompt
from m2t.tokenizer import get_prompt_end_token_sequence
from m2t.conversation_utils import extract_response_tokens
from m2t.utils import get_autocast_type
from m2t.arguments import DataArguments, ModelArguments, TrainingArguments

# Dynamic memory management
def clear_gpu_memory():
    """强力清理GPU内存，包括处理内存碎片化"""
    if torch.cuda.is_available():
        # 多次清理以确保彻底释放
        for _ in range(3):
            torch.cuda.empty_cache()
            gc.collect()
        
        # 设置内存分配策略以减少碎片化
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
        
        # 强制同步GPU操作
        torch.cuda.synchronize()
        
        print(f"[INFO] GPU内存清理完成")

def force_clear_gpu_memory():
    """在OOM后的强力内存清理"""
    if torch.cuda.is_available():
        print("[INFO] 执行强力GPU内存清理...")
        
        # 多轮清理
        for i in range(5):
            torch.cuda.empty_cache()
            gc.collect()
            if i < 4:  # 最后一轮不等待
                torch.cuda.synchronize()
        
        # 重置内存分配器
        try:
            torch.cuda.reset_peak_memory_stats()
            print("[INFO] GPU内存统计已重置")
        except:
            pass
            
        # 检查清理效果
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            cached = torch.cuda.memory_reserved(0) / (1024**3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            free = total - cached
            print(f"[INFO] 清理后GPU内存: 总计{total:.1f}GB, 已缓存{cached:.1f}GB, 可用{free:.1f}GB")

def collect_encoding_files(enc_dir: str):
    patterns = ["**/*.npy", "*.npy", "**/*.pt", "*.pt"]
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(enc_dir, pat), recursive=True))
    files = sorted(set(files))
    if len(files) == 0:
        print(f"[WARN] No encodings found under {enc_dir}. Tried patterns: {patterns}")
    else:
        print(f"[INFO] Found {len(files)} encoding files. Sample: {files[:3]}")
    return files

def load_audio_encoding(path: str):
    if path.endswith(".npy"):
        return np.load(path)
    elif path.endswith(".pt"):
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy()
        elif isinstance(obj, np.ndarray):
            return obj
        else:
            try:
                return np.array(obj)
            except Exception as e:
                raise ValueError(f"Unsupported encoding content in {path}: {type(obj)}") from e
    else:
        raise ValueError(f"Unsupported encoding extension: {path}")

def main(
    data_args: DataArguments,
    model_args: ModelArguments,
    training_args: TrainingArguments,
    audio_encodings_dir: str,
    outfile: str,
    ckpt_num: int,
    prompt: str,
    max_samples: int = None,
    max_new_tokens: int = 512,
):
    print("loading model and data...")

    # 清理GPU内存
    clear_gpu_memory()
    
    # 检查GPU可用性和内存
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        allocated_memory = torch.cuda.memory_allocated(0) / (1024**3)
        cached_memory = torch.cuda.memory_reserved(0) / (1024**3)
        free_memory = gpu_memory - cached_memory
        
        print(f"[INFO] GPU memory status:")
        print(f"  Total: {gpu_memory:.1f} GB")
        print(f"  Allocated: {allocated_memory:.1f} GB") 
        print(f"  Cached: {cached_memory:.1f} GB")
        print(f"  Free: {free_memory:.1f} GB")
        
        if free_memory >= 8:  # 如果可用内存>=8GB，尝试GPU加载
            print("[INFO] Sufficient free GPU memory, loading model to GPU...")
            
            # 尝试GPU加载，如果失败则强力清理后重试
            gpu_load_success = False
            for attempt in range(2):  # 最多尝试2次
                try:
                    # 不使用low_memory模式，直接加载到GPU
                    model, tokenizer = load_pretrained_model(
                        model_args.model_name_or_path,
                        ckpt_num=ckpt_num,
                        low_memory=False,  # 关键：不使用low_memory避免meta tensor
                    )
                    print("[INFO] Model loaded successfully to GPU")
                    gpu_load_success = True
                    break
                    
                except torch.cuda.OutOfMemoryError as e:
                    print(f"[WARNING] GPU OOM during loading (attempt {attempt+1}/2): {e}")
                    if attempt == 0:  # 第一次失败，强力清理后重试
                        print("[INFO] 执行强力内存清理后重试...")
                        force_clear_gpu_memory()
                        # 等待一下让GPU完全释放
                        import time
                        time.sleep(2)
                    else:  # 第二次失败，回退到CPU
                        print("[INFO] GPU加载失败，回退到CPU模式...")
                        force_clear_gpu_memory()
                        break
            
            # 如果GPU加载失败，使用CPU模式
            if not gpu_load_success:
                model, tokenizer = load_pretrained_model(
                    model_args.model_name_or_path,
                    ckpt_num=ckpt_num,
                    low_memory=False,  # 即使CPU模式也不用low_memory
                )
                
        else:
            print(f"[INFO] Limited free GPU memory ({free_memory:.1f}GB < 8GB), using CPU mode...")
            model, tokenizer = load_pretrained_model(
                model_args.model_name_or_path,
                ckpt_num=ckpt_num,
                low_memory=False,  # 不使用low_memory避免meta tensor
            )
    else:
        print("[INFO] No GPU available, loading to CPU...")
        model, tokenizer = load_pretrained_model(
            model_args.model_name_or_path,
            ckpt_num=ckpt_num,
            low_memory=False,  # 不使用low_memory避免meta tensor
        )
    
    # 强制绑定权重以消除警告并优化内存
    if hasattr(model, "tie_weights"):
        try:
            model.tie_weights()
            print("[INFO] Model weights tied successfully")
        except Exception as e:
            print(f"[WARN] Failed to tie weights: {e}")
    
    # 详细检查模型设备分配情况
    device = next(model.parameters()).device
    print(f"[INFO] Model main device: {device}")
    
    # 检查模型各部分的设备分配
    gpu_params = 0
    cpu_params = 0
    total_params = 0
    
    for name, param in model.named_parameters():
        total_params += 1
        if param.is_cuda:
            gpu_params += 1
        else:
            cpu_params += 1
    
    print(f"[INFO] Parameter distribution: {gpu_params} on GPU, {cpu_params} on CPU (total: {total_params})")
    
    # 检查mm_projector的设备
    if hasattr(model.get_model(), 'mm_projector'):
        proj_device = next(model.get_model().mm_projector.parameters()).device
        print(f"[INFO] MM projector device: {proj_device}")
    
    # 修复tokenizer缺少pad_token的问题
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"[INFO] Set pad_token to eos_token: {tokenizer.pad_token}")

    data_args.mm_use_audio_start_end = True

    # 保留梯度检查点；低内存模式下不要手动model.cuda(); recommute gradient, instead of storing gradient to save memory.
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # 不要在这里调用 model.cuda()，low_memory模式下由device_map控制
    end_seq = get_prompt_end_token_sequence(tokenizer, model_args.model_name_or_path)

    if outfile:
        out_dir = os.path.dirname(outfile)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

    multimodal_cfg = make_mm_config(data_args)
    audio_encodings = collect_encoding_files(audio_encodings_dir)

    if max_samples:
        audio_encodings = audio_encodings[:max_samples]

    print(f"Found {len(audio_encodings)} audio encoding files")

    outputs = []
    dynamic_max_new_tokens = max_new_tokens

    with torch.autocast(device_type="cuda", dtype=get_autocast_type(training_args)):
        with torch.inference_mode():
            for i, encoding_fp in tqdm(enumerate(audio_encodings), total=len(audio_encodings)):
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()

                    audio_encoding = load_audio_encoding(encoding_fp)

                    outputs_i = infer_with_prompt(
                        prompt,
                        model=model,
                        audio_encoding=audio_encoding,
                        multimodal_cfg=multimodal_cfg,
                        end_seq=end_seq,
                        tokenizer=tokenizer,
                        audio_first=True,
                        max_new_tokens=dynamic_max_new_tokens,
                        use_cache=False, # do not cache KV
                        temperature=0.0,
                        num_beams=1,
                        top_p=0.9,
                    )
                    # 使用传入的 prompt
                    prompt_text = prompt

                    print("[PROMPT]")
                    print(prompt_text)

                    print("[MODEL COMPLETION]")
                    model_completion_ids = extract_response_tokens(outputs_i[0], end_seq)
                    model_completion_text = tokenizer.decode(model_completion_ids)
                    print(model_completion_text)

                    output_dict = {
                        "example_id": os.path.basename(encoding_fp).replace(".npy", ""),
                        "prompt_text": prompt_text,
                        "model_completion_text": model_completion_text,
                    }

                    outputs.append(output_dict)

                    print("%" * 40)

                    # Clean up memory after each sample
                    del audio_encoding, outputs_i
                    clear_gpu_memory()

                except torch.cuda.OutOfMemoryError as e:
                    print(f"GPU OOM error for {encoding_fp}: {e}")
                    new_tokens = max(dynamic_max_new_tokens // 2, 32)
                    if new_tokens < dynamic_max_new_tokens:
                        dynamic_max_new_tokens = new_tokens
                        print(f"[INFO] Reducing max_new_tokens to {dynamic_max_new_tokens} and retrying once...")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            gc.collect()
                        try:
                            outputs_i = infer_with_prompt(
                                prompt,
                                model=model,
                                audio_encoding=audio_encoding,
                                multimodal_cfg=multimodal_cfg,
                                end_seq=end_seq,
                                tokenizer=tokenizer,
                                audio_first=True,
                                max_new_tokens=dynamic_max_new_tokens,
                                use_cache=False,
                                temperature=0.0,
                                num_beams=1,
                                top_p=0.9,
                            )
                            prompt_text = prompt

                            print("[PROMPT]")
                            print(prompt_text)

                            print("[MODEL COMPLETION]")
                            model_completion_ids = extract_response_tokens(outputs_i[0], end_seq)
                            model_completion_text = tokenizer.decode(model_completion_ids)
                            print(model_completion_text)

                            output_dict = {
                                "example_id": os.path.basename(encoding_fp).replace(".npy", ""),
                                "prompt_text": prompt_text,
                                "model_completion_text": model_completion_text,
                            }

                            outputs.append(output_dict)

                            print("%" * 40)

                            # Clean up memory after each sample
                            del audio_encoding, outputs_i
                            clear_gpu_memory()

                        except torch.cuda.OutOfMemoryError as e2:
                            print(f"[WARN] OOM again at {dynamic_max_new_tokens}; skipping sample.")
                    else:
                        print("[WARN] Cannot reduce max_new_tokens further; skipping sample.")
                    continue
                except Exception as e:
                    print(f"Error processing {encoding_fp}: {e}")
                    continue

    df = pd.DataFrame(outputs)
    df.to_csv(outfile, index=False)
    print(f"Results saved to {outfile}")

def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]

if __name__ == "__main__":
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    parser.add_argument(
        "--ckpt-num",
        type=int,
        help="Step number of the trained checkpoint.",
        required=True,
    )
    parser.add_argument("--max_samples", default=None, type=int, help="max eval samples to use.")
    parser.add_argument(
        "--audio-encodings-dir",
        required=True,
        help="Path to a directory containing the audio encodings .npy files.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Prompt to use. If set, this will override the prompt in all examples. "
        "Do not add conversation headers (e.g. 'ASSISTANT:') or other formatting"
        "to the prompt; these are added automatically under the hood.",
    )
    parser.add_argument(
        "--outfile",
        default="infer_results.csv",
        help="path to csv file to write results.",
    )
    parser.add_argument(
        "--max-new-tokens",
        default=256,
        type=int,
        help="Maximum new tokens to generate (lower uses less GPU memory).",
    )
    (
        model_args,
        data_args,
        training_args,
        other_args,
    ) = parser.parse_args_into_dataclasses()

    main(
        data_args=data_args,
        model_args=model_args,
        training_args=training_args,
        **vars(other_args),
    )
