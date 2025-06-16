#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import torch
import argparse
import os
from pathlib import Path
from models import STED
from utils import load_model
import dlib  # 用于人脸检测和关键点定位
from collections import OrderedDict
import time

class GazeRedirectionSystem:
    def __init__(self, model_path, device='cuda', face_detector_path=None):
        """初始化视线重定向系统

        Args:
            model_path: STED模型权重路径
            device: 运行设备
            face_detector_path: dlib人脸检测器路径
        """
        self.device = device
        self.image_size = (128, 128)  # 根据ST-ED模型的normalized_camera设置

        # 初始化人脸检测器
        self.detector = dlib.get_frontal_face_detector()
        if face_detector_path:
            self.predictor = dlib.shape_predictor(face_detector_path)
        else:
            # 默认使用较小的5点模型，如果需要可替换为68点模型
            self.predictor = None
            print("警告: 未提供人脸关键点检测器，将仅使用人脸检测")

        # 加载STED模型
        self.model = STED().to(device)
        load_model(self.model, model_path)
        self.model.eval()
        print(f"模型已加载: {model_path}")

    def detect_face(self, frame):
        """检测面部并返回面部框"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        if len(faces) == 0:
            return None

        # 使用最大的人脸
        face = max(faces, key=lambda rect: rect.width() * rect.height())

        # 扩大人脸框以包含更多面部区域
        x, y = face.left(), face.top()
        w, h = face.width(), face.height()

        # 稍微扩大框以包含整个面部
        x = max(0, x - int(w * 0.1))
        y = max(0, y - int(h * 0.1))
        w = min(frame.shape[1] - x, w + int(w * 0.2))
        h = min(frame.shape[0] - y, h + int(h * 0.2))

        return (x, y, w, h)

    def preprocess_face(self, frame, face_rect):
        """预处理人脸图像为模型输入"""
        x, y, w, h = face_rect
        face_img = frame[y:y+h, x:x+w]

        # 调整为模型输入尺寸
        face_img = cv2.resize(face_img, self.image_size)

        # 转换为YCrCb并对Y通道进行直方图均衡化
        ycrcb = cv2.cvtColor(face_img, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        face_img = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

        # 转换为Tensor并归一化到[-1, 1]范围
        face_img = np.transpose(face_img, [2, 0, 1])  # 转置为[C,H,W]
        face_img = 2.0 * face_img / 255.0 - 1.0  # 归一化

        # 转换为Tensor
        face_tensor = torch.from_numpy(face_img.astype(np.float32)).unsqueeze(0).to(self.device)
        return face_tensor

    def postprocess_image(self, tensor_img):
        """将模型输出转换回OpenCV图像"""
        img = tensor_img.squeeze().permute(1, 2, 0).detach().cpu().numpy()
        img = (img + 1.0) * 127.5
        img = np.clip(img, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def redirect_gaze(self, frame, target_gaze, target_head=None):
        """重定向视线

        Args:
            frame: 输入帧
            target_gaze: 目标视线方向 [pitch, yaw]，单位为弧度
            target_head: 目标头部姿态 [pitch, yaw]，单位为弧度
                         如果为None，则保持原始头部姿态

        Returns:
            原始帧和重定向后的帧
        """
        # 检测人脸
        face_rect = self.detect_face(frame)
        if face_rect is None:
            return frame, frame

        # 预处理人脸
        face_tensor = self.preprocess_face(frame, face_rect)

        # 准备输入数据
        data = {
            'image_a': face_tensor,
        }

        # 获取当前视线和头部姿态估计
        with torch.no_grad():
            pseudo_labels, _ = self.model.encoder(data['image_a'])
            current_gaze = pseudo_labels[-1]
            current_head = pseudo_labels[-2]

        # 如果未提供目标头部姿态，则保持原始姿态
        if target_head is None:
            target_head = current_head
        else:
            target_head = torch.tensor([target_head], dtype=torch.float).to(self.device)

        # 构造重定向数据
        target_gaze = torch.tensor([target_gaze], dtype=torch.float).to(self.device)
        redirect_data = {
            'image_a': face_tensor,
            'gaze_b_r': target_gaze,
            'head_b_r': target_head
        }

        # 执行重定向
        with torch.no_grad():
            output_dict = self.model.redirect(redirect_data)

        # 处理重定向后的图像
        redirected_face = self.postprocess_image(output_dict['image_b_hat_r'])

        # 将重定向后的面部放回原始图像
        x, y, w, h = face_rect
        result_frame = frame.copy()
        face_resized = cv2.resize(redirected_face, (w, h))
        result_frame[y:y+h, x:x+w] = face_resized

        return frame, result_frame, current_gaze.cpu().numpy()

def realtime_redirection(camera_id=0, model_path='logs/11-11/checkpoints/model_best.pt', face_detector_path=None):
    """实时视线重定向

    Args:
        camera_id: 摄像头ID
        model_path: STED模型权重路径
        face_detector_path: dlib人脸关键点检测器路径
    """
    # 初始化重定向系统
    system = GazeRedirectionSystem(model_path, face_detector_path=face_detector_path)

    # 打开摄像头
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"错误：无法打开摄像头 {camera_id}")
        return

    # 获取摄像头参数
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 设置目标视线为正向(向前看)
    target_gaze = [0.0, 0.0]  # 正向视线，pitch=0, yaw=0

    # 用于显示帧率
    prev_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头帧")
            break

        # 处理当前帧
        original, redirected, current_gaze = system.redirect_gaze(frame, target_gaze)

        # 计算帧率
        frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - prev_time
        if elapsed_time > 1.0:  # 每秒更新一次帧率
            fps_real = frame_count / elapsed_time
            prev_time = current_time
            frame_count = 0
        else:
            fps_real = fps

        # 创建左右对比视图
        comparison = np.hstack([original, redirected])

        # 添加标签和信息
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, "Original", (10, 30), font, 1, (0, 255, 0), 2)
        cv2.putText(comparison, "Redirected", (width + 10, 30), font, 1, (0, 255, 0), 2)

        # 显示当前视线角度和目标视线角度
        if len(current_gaze) > 0:
            gaze_text = f"Current: [{current_gaze[0][0]:.2f}, {current_gaze[0][1]:.2f}] -> Target: [0.00, 0.00]"
            cv2.putText(comparison, gaze_text, (10, 70), font, 0.7, (0, 255, 0), 2)

        # 显示帧率
        fps_text = f"FPS: {fps_real:.1f}"
        cv2.putText(comparison, fps_text, (10, height - 20), font, 0.7, (0, 255, 0), 2)

        # 显示结果
        cv2.imshow('Gaze Redirection (Press q to quit)', comparison)

        # 按q键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

def display_comparison(video_path, target_gaze, target_head=None, output_path=None, model_path='logs/11-11/checkpoints/model_best.pt', face_detector_path=None):
    """显示视线重定向前后的对比

    Args:
        video_path: 输入视频路径
        target_gaze: 目标视线方向 [pitch, yaw]，单位为弧度
        target_head: 目标头部姿态 [pitch, yaw]，单位为弧度
        output_path: 输出视频路径
        model_path: STED模型权重路径
        face_detector_path: dlib人脸关键点检测器路径
    """
    # 初始化重定向系统
    system = GazeRedirectionSystem(model_path, face_detector_path=face_detector_path)

    # 打开视频
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 创建输出视频
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width*2, height))

    # 用于计算平均处理时间
    processing_times = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 测量处理时间
        start_time = time.time()

        # 处理当前帧
        original, redirected, current_gaze = system.redirect_gaze(frame, target_gaze, target_head)

        # 计算处理时间
        processing_time = time.time() - start_time
        processing_times.append(processing_time)

        # 创建左右对比视图
        comparison = np.hstack([original, redirected])

        # 添加标签
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, "Original", (10, 30), font, 1, (0, 255, 0), 2)
        cv2.putText(comparison, "Redirected", (width + 10, 30), font, 1, (0, 255, 0), 2)

        # 显示当前视线和目标视线
        if len(current_gaze) > 0:
            gaze_text = f"Current: [{current_gaze[0][0]:.2f}, {current_gaze[0][1]:.2f}] -> Target: [{target_gaze[0]:.2f}, {target_gaze[1]:.2f}]"
            cv2.putText(comparison, gaze_text, (10, 70), font, 0.7, (0, 255, 0), 2)

        # 显示平均处理时间
        avg_time = sum(processing_times) / len(processing_times)
        time_text = f"Proc time: {avg_time*1000:.1f}ms ({1.0/avg_time:.1f} FPS)"
        cv2.putText(comparison, time_text, (10, height - 20), font, 0.7, (0, 255, 0), 2)

        # 显示结果
        cv2.imshow('Original vs Redirected', comparison)

        # 保存视频
        if output_path:
            out.write(comparison)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='STED视线重定向对比系统')
    parser.add_argument('--video', type=str, help='输入视频路径，不指定则使用摄像头输入')
    parser.add_argument('--camera', type=int, default=0, help='摄像头ID，默认为0')
    parser.add_argument('--target_gaze', type=float, nargs=2, default=[0.0, 0.0],
                        help='目标视线方向 [pitch, yaw]，单位为弧度，默认为[0.0, 0.0]（向前看）')
    parser.add_argument('--target_head', type=float, nargs=2, default=None,
                        help='目标头部姿态 [pitch, yaw]，单位为弧度')
    parser.add_argument('--output', type=str, help='输出视频路径')
    parser.add_argument('--model', type=str, default='logs/11-11/checkpoints/model_best.pt',
                        help='STED模型路径')
    parser.add_argument('--face_detector', type=str, default=None,
                        help='dlib人脸关键点检测器路径')

    args = parser.parse_args()

    if args.video:
        # 处理视频文件
        display_comparison(
            args.video,
            args.target_gaze,
            args.target_head,
            args.output,
            args.model,
            args.face_detector
        )
    else:
        # 实时摄像头处理
        realtime_redirection(
            args.camera,
            args.model,
            args.face_detector
        )

if __name__ == "__main__":
    main()
