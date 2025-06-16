# STED视线重定向对比系统

本工具基于STED-gaze视线估计模型，实现了实时视频中的视线重定向效果对比展示。系统可以读取视频输入，同时显示原始视频和视线重定向后的视频，以便于直观比较效果。

## 功能特点

- 实时人脸检测与跟踪
- 基于深度学习的视线重定向
- 可自定义目标视线方向和头部姿态
- 左右对比显示原始视频和重定向效果
- 支持视频录制保存

## 系统要求

- Python 3.6+
- PyTorch 1.7.0+
- CUDA支持(推荐)
- dlib
- OpenCV

## 安装方法

1. 确保安装了所有依赖项：

```bash
pip install -r requirements.txt
```

2. 对于dlib的安装，可能需要额外的步骤：

```bash
# 在Ubuntu上
apt-get install cmake
pip install dlib

# 在Windows上，确保已安装Visual Studio和CMake
pip install dlib
```

3. 下载预训练模型权重(如果尚未包含在项目中)。

## 使用方法

### 基本命令

```bash
python gaze_redirection_comparison.py --video <视频路径> --target_gaze <pitch> <yaw>
```

### 参数说明

- `--video`: 输入视频路径(必填)
- `--target_gaze`: 目标视线方向，以[pitch, yaw]形式指定，单位为弧度(默认[0, 0])
- `--target_head`: 目标头部姿态，以[pitch, yaw]形式指定，单位为弧度(可选)
- `--output`: 输出视频路径(可选)
- `--model`: STED模型权重路径(默认为'logs/11-11/checkpoints/model_best.pt')
- `--face_detector`: dlib人脸关键点检测器路径(可选，如shape_predictor_68_face_landmarks.dat)

### 示例

1. 使用默认设置进行视线重定向：

```bash
python gaze_redirection_comparison.py --video sample.mp4
```

2. 指定目标视线方向并保存输出：

```bash
python gaze_redirection_comparison.py --video sample.mp4 --target_gaze 0.2 -0.3 --output result.avi
```

3. 同时指定头部姿态：

```bash
python gaze_redirection_comparison.py --video sample.mp4 --target_gaze 0.2 -0.3 --target_head 0.1 0.1 --output result.avi
```

## 交互控制

在视频播放过程中：
- 按`q`键退出程序

## 注意事项

1. 人脸检测质量直接影响重定向效果
2. 对于高分辨率视频，处理可能较慢
3. 确保模型路径正确，否则将无法进行视线重定向
4. 视线角度单位为弧度，范围通常在[-1,1]之间

## 故障排除

1. 如果出现"未检测到人脸"问题，请确保视频中人脸清晰可见
2. 如果重定向效果不理想，尝试调整目标视线参数
3. 如果运行速度过慢，可考虑降低输入视频分辨率

## 引用与致谢

本工具基于STED-gaze视线估计模型开发，如果您在科研工作中使用了本工具，请引用相关论文。
