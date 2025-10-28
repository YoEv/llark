#!/usr/bin/env python3
"""
创建测试用的音频编码文件
由于完整的Jukebox编码需要大量依赖和GPU资源，这个脚本创建模拟的编码文件用于测试LLark推理流程
"""

import os
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm

def create_mock_jukebox_encoding(audio_file, output_file):
    """
    创建模拟的Jukebox编码
    实际的Jukebox编码维度是 [time_steps, 4800]
    """
    try:
        # 加载音频文件
        audio, sr = librosa.load(audio_file, sr=44100)
        
        # 计算音频长度对应的时间步数
        # Jukebox的时间步数大约是 8192 对应 23.8秒的音频
        audio_duration = len(audio) / sr
        time_steps = int(8192 * audio_duration / 23.8)
        time_steps = max(1, min(time_steps, 8192))  # 限制在合理范围内
        
        # 创建模拟编码 [time_steps, 4800]
        # 使用音频的一些基本特征来生成"编码"
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        
        # 将特征重复和组合以达到4800维
        features = np.concatenate([
            mfcc.flatten(),
            spectral_centroid.flatten(),
            spectral_rolloff.flatten()
        ])
        
        # 扩展到4800维
        if len(features) < 4800:
            features = np.tile(features, (4800 // len(features) + 1))[:4800]
        else:
            features = features[:4800]
        
        # 创建时间序列编码
        encoding = np.tile(features, (time_steps, 1))
        
        # 添加一些随机变化使其更真实
        noise = np.random.normal(0, 0.1, encoding.shape)
        encoding = encoding + noise
        
        # 保存编码
        np.save(output_file, encoding.astype(np.float32))
        
        print(f"✅ 创建编码: {os.path.basename(audio_file)} -> {os.path.basename(output_file)} (shape: {encoding.shape})")
        return True
        
    except Exception as e:
        print(f"❌ 编码失败: {audio_file} - {e}")
        return False

def main():
    print("🎵 创建测试用音频编码文件")
    print("=" * 60)
    
    # 输入和输出目录
    input_dir = Path("/home/evev/diversity-eval/data/input/GTZAN_Dataset/Solo/genres_original/pop")
    output_dir = Path("/home/evev/diversity-eval/data/output/llark_encodings")
    
    print(f"📂 输入目录: {input_dir}")
    print(f"📂 输出目录: {output_dir}")
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找所有音频文件
    audio_files = list(input_dir.glob("*.wav"))
    
    if not audio_files:
        print("❌ 未找到音频文件")
        return False
    
    print(f"🎵 找到 {len(audio_files)} 个音频文件")
    
    # 只处理前10个文件作为测试
    audio_files = audio_files[:10]
    print(f"🎯 处理前 {len(audio_files)} 个文件作为测试")
    
    # 处理每个音频文件
    success_count = 0
    for audio_file in tqdm(audio_files, desc="编码音频文件"):
        # 生成输出文件名
        output_file = output_dir / (audio_file.stem + ".npy")
        
        # 如果文件已存在，跳过
        if output_file.exists():
            print(f"⏭️  跳过已存在的文件: {output_file.name}")
            success_count += 1
            continue
        
        # 创建编码
        if create_mock_jukebox_encoding(audio_file, output_file):
            success_count += 1
    
    print("\n" + "=" * 60)
    print(f"🎉 编码完成!")
    print(f"✅ 成功: {success_count}/{len(audio_files)} 个文件")
    print(f"📁 编码文件保存在: {output_dir}")
    
    # 显示一些示例文件
    encoding_files = list(output_dir.glob("*.npy"))
    if encoding_files:
        print(f"\n📋 示例编码文件:")
        for i, f in enumerate(encoding_files[:5]):
            try:
                encoding = np.load(f)
                print(f"   {f.name}: {encoding.shape}")
            except:
                print(f"   {f.name}: 读取失败")
        if len(encoding_files) > 5:
            print(f"   ... 还有 {len(encoding_files) - 5} 个文件")
    
    return success_count > 0

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n✅ 脚本执行成功!")
        else:
            print("\n❌ 脚本执行失败!")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")