# 导入必要的库
import os
import cv2
import h5py
import numpy as np
from tqdm import tqdm
import json

# 设置输出视频路径和参数
output_dir = './output_videos'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 读取GazeCapture测试集数据
with open('./gazecapture_split.json', 'r') as f:
    gc_splits = json.load(f)
test_prefixes = gc_splits['test']

# 设置视频参数
fps = 30
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
frame_size = (128, 128) # 根据实际图像大小调整

# 生成10段视频样本
for i in range(10):
    # 创建视频写入器
    video_path = os.path.join(output_dir, f'gazecapture_test_sample_{i+1}.mp4')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, frame_size)

    # 读取HDF文件并写入视频
    with h5py.File('/mnt/data/xhy/outputs_sted/GazeCapture.h5', 'r') as hdf:
        # 每段取10个样本
        start_idx = i * 10
        end_idx = start_idx + 10
        current_prefixes = test_prefixes[start_idx:end_idx]

        for prefix in tqdm(current_prefixes, desc=f'生成样本 {i+1}/10'):
            if prefix in hdf:
                group = hdf[prefix]
                frames = group['pixels'][:]

                # 写入每一帧
                for frame in frames:
                    # 转换为BGR格式
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    video_writer.write(frame_bgr)

        # 释放资源
        video_writer.release()
        print(f'视频样本 {i+1} 已保存至: {video_path}')

        # 显示当前视频样本的统计信息
        video_cap = cv2.VideoCapture(video_path)
        total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        print(f'\n视频样本 {i+1} 统计信息:')
        print(f'总帧数: {total_frames}')
        print(f'视频时长: {duration:.2f} 秒')
        print(f'分辨率: {frame_size}')
        print(f'帧率: {fps} fps')
        print('-' * 50)

        video_cap.release()

print('所有10段视频样本生成完成!')
