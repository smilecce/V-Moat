import os
import pandas as pd
import random
from glob import glob

def create_voxceleb_pairs_csv(voxceleb_root, output_csv):
    """
    创建VoxCeleb音频对CSV文件
    前50对：同一个人的enroll和trial (id10001-id10050)
    后50对：不同人的随机配对 (id10051-id10100)
    """
    pairs = []
    
    # 前50对：同一个人的音频对 (id10001-id10050)
    print("正在处理前50对（同一人音频对）...")
    for i in range(1, 51):
        speaker_id = f"id100{i:02d}"  # id10001, id10002, ..., id10050
        speaker_path = os.path.join(voxceleb_root, speaker_id)
        
        if not os.path.exists(speaker_path):
            print(f"警告: {speaker_path} 不存在，跳过")
            continue
            
        # 获取该说话人的所有音频文件
        audio_files = []
        for root, dirs, files in os.walk(speaker_path):
            for file in files:
                if file.endswith('.wav') or file.endswith('.flac') or file.endswith('.m4a'):
                    audio_files.append(os.path.join(root, file))
        
        if len(audio_files) >= 2:
            # 随机选择两个不同的音频文件作为enroll和trial
            selected_files = random.sample(audio_files, 2)
            pairs.append({
                'enroll': selected_files[0],
                'trial': selected_files[1],
                'groundtruth': 1  # 同一个人，标签为1
            })
            print(f"✓ {speaker_id}: 找到 {len(audio_files)} 个音频文件")
        else:
            print(f"警告: {speaker_id} 音频文件不足2个（只有{len(audio_files)}个），跳过")
    
    # 后50对：不同人的随机配对 (id10051-id10100)
    print("\n正在处理后50对（不同人音频对）...")
    available_speakers = []
    for i in range(51, 101):
        speaker_id = f"id100{i:02d}"  # id10051, id10052, ..., id10100
        speaker_path = os.path.join(voxceleb_root, speaker_id)
        
        if os.path.exists(speaker_path):
            # 获取该说话人的音频文件
            audio_files = []
            for root, dirs, files in os.walk(speaker_path):
                for file in files:
                    if file.endswith('.wav') or file.endswith('.flac') or file.endswith('.m4a'):
                        audio_files.append(os.path.join(root, file))
            
            if len(audio_files) > 0:
                available_speakers.append((speaker_id, audio_files))
                print(f"✓ {speaker_id}: 找到 {len(audio_files)} 个音频文件")
            else:
                print(f"警告: {speaker_id} 没有找到音频文件")
    
    print(f"可用于配对的说话人数量: {len(available_speakers)}")
    
    # 随机配对不同的说话人
    for pair_idx in range(50):
        if len(available_speakers) >= 2:
            # 随机选择两个不同的说话人
            speaker1, speaker2 = random.sample(available_speakers, 2)
            
            # 为每个说话人随机选择一个音频文件
            enroll_file = random.choice(speaker1[1])
            trial_file = random.choice(speaker2[1])
            
            pairs.append({
                'enroll': enroll_file,
                'trial': trial_file,
                'groundtruth': 0  # 不同人，标签为0
            })
            print(f"配对 {pair_idx+1}: {speaker1[0]} vs {speaker2[0]}")
        else:
            print(f"警告: 可用说话人不足，無法创建第 {pair_idx+1} 对")
            break
    
    # 创建DataFrame并保存
    df = pd.DataFrame(pairs)
    df.to_csv(output_csv, index=False)
    print(f"\n=== 结果摘要 ===")
    print(f"已创建CSV文件: {output_csv}")
    print(f"总计 {len(pairs)} 对音频")
    print(f"同一人对数: {sum(df['groundtruth'])}")
    print(f"不同人对数: {len(df) - sum(df['groundtruth'])}")
    
    return df

if __name__ == "__main__":
    # 设置VoxCeleb数据集路径 - 请修改为您的实际路径
    voxceleb_root = "/mnt/data/old_home/strength/vmoat/vmoat_code25/vmoat_code2025/voxceleb1_subset_100"  # 修改这里！
    output_csv = "voxceleb_pairs_100.csv"
    
    # 设置随机种子以确保可重复性
    random.seed(42)
    
    # 检查路径是否存在
    if not os.path.exists(voxceleb_root):
        print(f"错误: VoxCeleb路径不存在: {voxceleb_root}")
        print("请修改 voxceleb_root 变量为正确的路径")
        exit(1)
    
    create_voxceleb_pairs_csv(voxceleb_root, output_csv)