import torch
import numpy as np
import cv2
import os
import time
from torch.nn.functional import interpolate
import torch.nn.functional as F
import torch.cuda
from collections import deque
import shutil

# 基本参数
startFrame = 0
TimeWindow = 10  # 时间窗口大小
batch_size = 256  # 减小批量大小以更好地管理GPU内存

DTYPE = torch.float32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 优化CUDA设置
torch.backends.cudnn.benchmark = True
if hasattr(torch.backends.cudnn, 'allow_tf32'):
    torch.backends.cudnn.allow_tf32 = True
if hasattr(torch.backends.cuda, 'matmul') and hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
    torch.backends.cuda.matmul.allow_tf32 = True

# 颜色常量移到GPU
LOWER_COLOR = torch.tensor([30, 50, 90], dtype=DTYPE, device=device)
UPPER_COLOR = torch.tensor([200, 200, 200], dtype=DTYPE, device=device)


def create_gaussian_kernel(kernel_size: int, sigma: float, device: torch.device) -> torch.Tensor:
    """创建高斯核"""
    x = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=DTYPE, device=device)
    x = x.view(-1, 1)
    y = x.t()
    gaussian = torch.exp(-(x * x + y * y) / (2 * sigma * sigma))
    return gaussian / gaussian.sum()


def getColor_batch(batch_img: torch.Tensor, lower_color: torch.Tensor, upper_color: torch.Tensor) -> torch.Tensor:
    """颜色检测函数"""
    mask = ~torch.logical_and(
        (batch_img >= lower_color).all(dim=3),
        (batch_img <= upper_color).all(dim=3)
    )
    return mask * 255


def denoise_motion_map_gpu(motion_map: torch.Tensor, kernel_size: int = 7, sigma: float = 1.5) -> torch.Tensor:
    """GPU加速的降噪操作"""
    # 创建高斯核
    kernel = create_gaussian_kernel(kernel_size, sigma, motion_map.device)
    kernel = kernel.view(1, 1, kernel_size, kernel_size)

    # 应用卷积
    motion_map = motion_map.unsqueeze(0).unsqueeze(0)
    motion_map = F.conv2d(motion_map, kernel, padding=kernel_size // 2)

    return motion_map.squeeze()


class VideoProcessor:
    def __init__(self, input_path: str, save_path: str):
        self.input_path = input_path
        self.save_path = save_path
        self.lower_color = LOWER_COLOR
        self.upper_color = UPPER_COLOR

        # 记录所有处理过的帧数的全局计数器
        self.global_frame_idx = 0

        self.setup_paths()
        self.setup_video_capture()
        self.setup_cuda_resources()

        # 创建CUDA流
        self.stream = torch.cuda.Stream()

    def setup_paths(self):
        video_name = os.path.basename(self.input_path)
        video_number = video_name[-7:-4]
        self.folder_path = os.path.join(self.save_path, 'v' + video_number)
        os.makedirs(self.folder_path, exist_ok=True)
        self.output_video_path = os.path.join(
            self.folder_path, f'output_{video_number}.mp4')

    def setup_video_capture(self):
        self.cap = cv2.VideoCapture(self.input_path, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {self.input_path}")

        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))

    def setup_cuda_resources(self):
        # 创建CPU端的pinned内存
        self.frame_buffer = torch.zeros(
            (batch_size, self.height, self.width, 3),
            dtype=torch.uint8, device='cpu', pin_memory=True)

        # 记录所有帧的信息，将用于时空处理
        self.frame_deque = deque(maxlen=TimeWindow)
        self.target_deque = deque(maxlen=TimeWindow)
        self.color_deque = deque(maxlen=TimeWindow)

        # 预分配GPU内存
        self.gpu_buffer = torch.zeros(
            (batch_size, self.height, self.width, 3),
            dtype=torch.uint8, device=device)

        # 清理GPU内存
        torch.cuda.empty_cache()

    def getting_frame_record_batch(self, frRec_batch: torch.Tensor, tarRec_batch: torch.Tensor
                                   ) -> tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]:
        """处理帧批次，用于做帧间差分等统计"""
        batch_size_local = frRec_batch.shape[0]
        frames = frRec_batch.view(batch_size_local, self.height, self.width)

        # 计算指数权重
        weights = torch.exp(torch.linspace(0, 1, batch_size_local - 1, device=device)) - 1
        weights = weights / weights.max()

        # 计算帧差
        frame_diffs = torch.abs(frames[1:] - frames[:-1]) * weights.view(-1, 1, 1)
        frameDiffComm = frame_diffs.sum(dim=0)

        # 统计阈值处理
        mean_diff = frameDiffComm.mean()
        std_diff = frameDiffComm.std()
        threshold = mean_diff + 3.0 * std_diff
        frameDiffComm = torch.where(frameDiffComm > threshold, frameDiffComm,
                                    torch.zeros_like(frameDiffComm))

        maxMovement = frameDiffComm.max().item()
        targets = tarRec_batch.view(batch_size_local, self.height, self.width)

        return (maxMovement,
                frames.view(-1, self.height * self.width),
                targets.view(-1, self.height * self.width),
                frameDiffComm.view(-1))

    def center_of_gravity_batch(self, cfrVectRec_batch: torch.Tensor) -> torch.Tensor:
        """计算重心"""
        sh = cfrVectRec_batch.shape
        F_comp = torch.abs(torch.fft.fft(cfrVectRec_batch, dim=0))
        av = torch.arange(1, sh[0] + 1, dtype=DTYPE, device=device)
        A = av.repeat(sh[1], 1).T

        FA = F_comp * A
        sF = F_comp.sum(dim=0)
        sFA = FA.sum(dim=0)

        return torch.where(sF != 0, sFA / sF, torch.zeros_like(sF))

    def process_batch(self, frames):
        """处理一批帧。修改逻辑：
           - 每一帧都更新滑动窗口（窗口始终保存最近 TimeWindow 帧，包含所有帧的信息）。
           - 如果当前帧为偶数帧且窗口已满，则调用时空处理；否则直接返回原始帧。
        """
        with torch.no_grad():
            # 使用异步数据传输，将帧数据从 CPU 的 pinned 内存传输到 GPU
            with torch.cuda.stream(self.stream):
                self.frame_buffer[:len(frames)].copy_(torch.from_numpy(frames))
                self.gpu_buffer[:len(frames)].copy_(self.frame_buffer[:len(frames)], non_blocking=True)

                frames_tensor = self.gpu_buffer[:len(frames)]
                target_array = getColor_batch(frames_tensor, self.lower_color, self.upper_color)

                results = []
                for frame, target in zip(frames_tensor, target_array):
                    self.global_frame_idx += 1

                    # 将当前帧转换为灰度，并将相关信息添加到滑动窗口中
                    gray = frame.to(DTYPE).mean(dim=2)
                    self.frame_deque.append(gray.reshape(-1))
                    self.target_deque.append(target.reshape(-1))
                    self.color_deque.append(frame)

                    # 保持滑动窗口为最近 TimeWindow 帧
                    if len(self.frame_deque) > TimeWindow:
                        self.frame_deque.popleft()
                        self.target_deque.popleft()
                        self.color_deque.popleft()

                    # 当窗口已满且当前帧为偶数帧时，执行时空处理；其他情况返回原始帧
                    if (self.global_frame_idx % 2 == 0) and (len(self.frame_deque) == TimeWindow):
                        result = self.process_frame_window()
                    else:
                        result = frame.cpu().numpy()

                    results.append(result)

                torch.cuda.current_stream().synchronize()
                return results

    def process_frame_window(self):
        """处理时间窗口内的帧(差分、重心、降噪等)"""
        frames_tensor = torch.stack(list(self.frame_deque))
        targets_tensor = torch.stack(list(self.target_deque))

        maxMovement, cfrVectRec, tfrVectRec, frameDiffComm = self.getting_frame_record_batch(
            frames_tensor, targets_tensor)

        cG = self.center_of_gravity_batch(cfrVectRec)
        I = torch.clamp(cG.view(self.height, self.width) - 1, 0, 1)

        I = denoise_motion_map_gpu(I)

        # 使用最后一帧彩色图像作为背景进行叠加
        last_color_frame = self.color_deque[-1]

        return self.create_visualization(I, last_color_frame)

    def create_visualization(self, motion_map, color_frame):
        """将运动信息(黄色)叠加到原彩色帧上"""
        motion_intensity = (motion_map * 255).to(torch.uint8)
        colored = torch.zeros(self.height, self.width, 3, device=device, dtype=torch.float32)

        # 将运动区域染成黄色 (红色 + 绿色)
        threshold_1 = 200
        threshold_2 = 150
        colored[..., 2] = torch.where(motion_intensity > threshold_1,
                                      torch.tensor(255.0, device=device),
                                      motion_intensity.float())
        colored[..., 1] = torch.where(motion_intensity > threshold_2,
                                      torch.tensor(255.0, device=device),
                                      motion_intensity.float())

        original_rgb = color_frame.to(torch.float32)
        alpha = motion_intensity.float() / 255.0
        alpha = torch.clamp(alpha * 1.2, 0, 0.8)

        output = original_rgb * (1 - alpha.unsqueeze(-1)) + colored * alpha.unsqueeze(-1)
        return output.cpu().numpy().astype(np.uint8)

    def process_video(self):
        """处理整个视频并输出处理后的视频文件"""
        frame_buffer = []
        processed_frames = 0
        total_time = 0

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, startFrame)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_video_path, fourcc,
                              self.fps, (self.width, self.height))

        try:
            while processed_frames < self.frame_count:
                ret, frame = self.cap.read()
                if not ret:
                    break

                frame_buffer.append(frame)

                # 当累计到 batch_size 帧或最后一批时，进行处理
                if len(frame_buffer) == batch_size or processed_frames + len(frame_buffer) == self.frame_count:
                    start_time = time.time()
                    frame_array = np.stack(frame_buffer)
                    results = self.process_batch(frame_array)

                    for result in results:
                        out.write(result)

                    end_time = time.time()
                    total_time += end_time - start_time
                    processed_frames += len(frame_buffer)
                    frame_buffer = []

                    if processed_frames % (batch_size * 10) == 0:
                        torch.cuda.empty_cache()

                    if processed_frames % 100 == 0:
                        print(f"进度: {processed_frames}/{self.frame_count} 帧")
                        current_fps = len(results) / (end_time - start_time) if (end_time - start_time) > 0 else 0
                        print(f"当前FPS: {current_fps:.2f}")

        finally:
            self.cap.release()
            out.release()
            torch.cuda.empty_cache()

        return {
            "total_frames": processed_frames,
            "total_time": total_time,
            "avg_fps": processed_frames / total_time if total_time > 0 else 0
        }


def process_video(input_path: str, save_path: str):
    processor = VideoProcessor(input_path, save_path)
    stats = processor.process_video()

    print("\n处理总结:")
    print(f"总处理帧数: {stats['total_frames']}")
    print(f"总耗时: {stats['total_time']:.2f} 秒")
    print(f"平均FPS: {stats['avg_fps']:.2f}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    input_path = r'input_1-2.mp4'
    save_path = r'milu-re'
    process_video(input_path, save_path)