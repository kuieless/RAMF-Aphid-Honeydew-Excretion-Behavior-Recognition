import torch
import numpy as np
import cv2
import os
import time
import subprocess
from torch.nn.functional import interpolate
import torch.nn.functional as F
import torch.cuda
from collections import deque
import shutil
import gc

# 基本参数
startFrame = 0
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

def get_gpu_power_usage():
    """获取GPU功耗信息"""
    try:
        # 执行nvidia-smi命令获取功耗信息
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=power.draw,power.limit', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )
        output = result.stdout.strip()
        if output:
            power_draw, power_limit = map(float, output.split(','))
            return power_draw, power_limit
        return 0, 0
    except (subprocess.SubprocessError, ValueError, FileNotFoundError):
        return 0, 0  # 出错时返回0

# 显存监控函数
def print_gpu_memory_stats():
    """打印GPU内存状态"""
    if torch.cuda.is_available():
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = t - r  # 空闲内存
        # 转换为MiB
        t_mb = t / 1024 / 1024
        r_mb = r / 1024 / 1024
        a_mb = a / 1024 / 1024
        f_mb = f / 1024 / 1024
        print(f"GPU内存: 总计={t_mb:.0f}MiB, 已分配={a_mb:.0f}MiB, 已缓存={r_mb:.0f}MiB, 空闲={f_mb:.0f}MiB")
        return a_mb, r_mb, t_mb
    return 0, 0, 0

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
    def __init__(self, input_path: str, save_path: str, time_window: int):
        self.input_path = input_path
        self.save_path = save_path
        self.TimeWindow = time_window  # 可配置的时间窗口大小
        self.lower_color = LOWER_COLOR
        self.upper_color = UPPER_COLOR

        # 记录所有处理过的帧数的全局计数器
        self.global_frame_idx = 0

        # 添加帧保存间隔参数
        self.frame_save_interval = 10
        
        # GPU统计信息
        self.memory_allocated_samples = []
        self.memory_reserved_samples = []
        self.total_memory = 0
        
        # 内存峰值跟踪
        self.max_memory_allocated = 0
        self.max_memory_reserved = 0
        
        # 添加功耗统计
        self.power_samples = []
        self.max_power_draw = 0
        self.power_limit = 0
        
        # 打印初始GPU内存状态
        print("初始GPU内存状态:")
        allocated, reserved, total = self.get_memory_stats()
        self.total_memory = total
        self.initial_memory_allocated = allocated  # 保存初始内存分配，用于计算增量
        
        # 获取初始功耗信息
        power_draw, power_limit = get_gpu_power_usage()
        self.power_limit = power_limit
        print(f"初始GPU功耗: {power_draw:.1f}W / {power_limit:.1f}W")
        
        self.setup_paths()
        self.setup_video_capture()
        self.setup_cuda_resources()

        # 创建CUDA流
        self.stream = torch.cuda.Stream()
        
        # 设置后检查内存状态
        print("设置完成后GPU内存状态:")
        self.get_memory_stats()

    def get_memory_stats(self):
        """获取GPU内存状态并更新统计信息"""
        if torch.cuda.is_available():
            # 获取当前GPU内存状态
            total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)  # 转为MiB
            reserved = torch.cuda.memory_reserved(0) / (1024 * 1024)
            allocated = torch.cuda.memory_allocated(0) / (1024 * 1024)
            
            # 获取功耗信息
            power_draw, power_limit = get_gpu_power_usage()
            self.power_samples.append(power_draw)
            self.max_power_draw = max(self.max_power_draw, power_draw)
            self.power_limit = power_limit
            
            # 输出内存状态和功耗
            print(f"GPU内存状态: 已分配={allocated:.1f}MiB, 已缓存={reserved:.1f}MiB, 总计={total:.1f}MiB")
            print(f"GPU功耗: {power_draw:.1f}W / {power_limit:.1f}W")
            
            # 更新峰值记录
            self.max_memory_allocated = max(self.max_memory_allocated, allocated)
            self.max_memory_reserved = max(self.max_memory_reserved, reserved)
            
            # 记录采样数据
            self.memory_allocated_samples.append(allocated)
            self.memory_reserved_samples.append(reserved)
            
            return allocated, reserved, total
        return 0, 0, 0

    def setup_paths(self):
        video_name = os.path.basename(self.input_path)
        video_number = video_name[-7:-4]
        self.folder_path = os.path.join(self.save_path, 'v' + video_number, f'tw{self.TimeWindow}')
        os.makedirs(self.folder_path, exist_ok=True)

        # 创建帧保存路径
        self.frames_path = os.path.join(self.folder_path, 'frames')
        os.makedirs(self.frames_path, exist_ok=True)

        self.output_video_path = os.path.join(
            self.folder_path, f'output_{video_number}_tw{self.TimeWindow}.mp4')

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
        self.frame_deque = deque(maxlen=self.TimeWindow)
        self.target_deque = deque(maxlen=self.TimeWindow)
        self.color_deque = deque(maxlen=self.TimeWindow)

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
        # 处理批次前记录GPU内存状态
        print(f"\n处理批次前 - 帧数: {len(frames)}")
        _, _, _ = self.get_memory_stats()
        
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
                    if len(self.frame_deque) > self.TimeWindow:
                        self.frame_deque.popleft()
                        self.target_deque.popleft()
                        self.color_deque.popleft()

                    # 当窗口已满且当前帧为偶数帧时，执行时空处理；其他情况返回原始帧
                    if (self.global_frame_idx % 2 == 0) and (len(self.frame_deque) == self.TimeWindow):
                        result = self.process_frame_window()
                    else:
                        result = frame.cpu().numpy()

                    results.append(result)

                torch.cuda.current_stream().synchronize()
                
                # 处理批次后记录GPU内存状态
                print(f"处理批次后 - 帧数: {len(frames)}")
                _, _, _ = self.get_memory_stats()
                
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

    def save_frame(self, frame_idx, frame):
        """保存指定帧为图片"""
        if frame_idx % self.frame_save_interval == 0:
            frame_filename = os.path.join(self.frames_path, f'frame_{frame_idx:04d}.jpg')
            cv2.imwrite(frame_filename, frame)

    def get_avg_memory_stats(self):
        """计算平均内存统计数据"""
        if not self.memory_allocated_samples:
            return {
                "avg_allocated": 0,
                "max_allocated": 0,
                "avg_reserved": 0,
                "max_reserved": 0,
                "total_memory": self.total_memory,
                "avg_power": 0,
                "max_power": 0,
                "power_limit": self.power_limit
            }
            
        memory_allocated_arr = np.array(self.memory_allocated_samples)
        memory_reserved_arr = np.array(self.memory_reserved_samples)
        power_arr = np.array(self.power_samples) if self.power_samples else np.array([0])
        
        # 计算平均值，去掉最低的10%的采样点
        if len(memory_allocated_arr) > 10:
            allocated_threshold = np.percentile(memory_allocated_arr, 10)
            reserved_threshold = np.percentile(memory_reserved_arr, 10)
            power_threshold = np.percentile(power_arr, 10) if len(power_arr) > 10 else 0
            
            valid_allocated = memory_allocated_arr[memory_allocated_arr > allocated_threshold]
            valid_reserved = memory_reserved_arr[memory_reserved_arr > reserved_threshold]
            valid_power = power_arr[power_arr > power_threshold] if len(power_arr) > 10 else power_arr
            
            avg_allocated = valid_allocated.mean() if len(valid_allocated) > 0 else memory_allocated_arr.mean()
            avg_reserved = valid_reserved.mean() if len(valid_reserved) > 0 else memory_reserved_arr.mean()
            avg_power = valid_power.mean() if len(valid_power) > 0 else power_arr.mean()
        else:
            avg_allocated = memory_allocated_arr.mean()
            avg_reserved = memory_reserved_arr.mean()
            avg_power = power_arr.mean()
        
        return {
            "avg_allocated": avg_allocated,
            "max_allocated": self.max_memory_allocated,
            "avg_reserved": avg_reserved,
            "max_reserved": self.max_memory_reserved,
            "total_memory": self.total_memory,
            "avg_power": avg_power,
            "max_power": self.max_power_draw,
            "power_limit": self.power_limit
        }

    def process_video(self):
        """处理整个视频并输出处理后的视频文件"""
        frame_buffer = []
        processed_frames = 0
        output_frame_idx = 0  # 用于跟踪输出帧的索引
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
                        output_frame_idx += 1

                        # 每隔指定帧数保存一帧图片
                        self.save_frame(output_frame_idx, result)

                    end_time = time.time()
                    total_time += end_time - start_time
                    processed_frames += len(frame_buffer)
                    frame_buffer = []

                    if processed_frames % (batch_size * 10) == 0:
                        print("执行显存缓存清理...")
                        torch.cuda.empty_cache()
                        gc.collect()
                        print("缓存清理后内存状态:")
                        self.get_memory_stats()

                    if processed_frames % 100 == 0:
                        print(f"时间窗口 {self.TimeWindow} - 进度: {processed_frames}/{self.frame_count} 帧")
                        current_fps = len(results) / (end_time - start_time) if (end_time - start_time) > 0 else 0
                        print(f"当前FPS: {current_fps:.2f}")
                        # 显示内存状态
                        self.get_memory_stats()

        finally:
            self.cap.release()
            out.release()
            print("处理完成，最终内存状态:")
            self.get_memory_stats()
            torch.cuda.empty_cache()

        # 获取GPU统计信息
        memory_stats = self.get_avg_memory_stats()

        return {
            "time_window": self.TimeWindow,
            "total_frames": processed_frames,
            "total_time": total_time,
            "avg_fps": processed_frames / total_time if total_time > 0 else 0,
            "avg_allocated": memory_stats["avg_allocated"],
            "max_allocated": memory_stats["max_allocated"],
            "avg_reserved": memory_stats["avg_reserved"],
            "max_reserved": memory_stats["max_reserved"],
            "total_memory": memory_stats["total_memory"],
            "avg_power": memory_stats["avg_power"],
            "max_power": memory_stats["max_power"],
            "power_limit": memory_stats["power_limit"]
        }

def batch_process_video(input_path: str, save_path: str, time_windows: list):
    """批量处理不同时间窗口的视频"""
    results = []

    # 获取CUDA设备信息
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        memory_info = torch.cuda.get_device_properties(current_device).total_memory / (1024**3)  # GB
        
        print("\nCUDA设备信息:")
        print(f"可用设备数: {device_count}")
        print(f"当前设备ID: {current_device}")
        print(f"当前设备名称: {device_name}")
        print(f"当前设备总内存: {memory_info:.2f} GB")
        print()

    for tw in time_windows:
        print(f"\n开始处理时间窗口 {tw}")
        processor = VideoProcessor(input_path, save_path, tw)
        stats = processor.process_video()
        results.append(stats)

        # 强制清理GPU内存
        print(f"时间窗口 {tw} 处理完成，清理GPU内存...")
        torch.cuda.empty_cache()
        gc.collect()

        print(f"\n时间窗口 {tw} 处理总结:")
        print(f"总处理帧数: {stats['total_frames']}")
        print(f"总耗时: {stats['total_time']:.2f} 秒")
        print(f"平均FPS: {stats['avg_fps']:.2f}")
        print(f"平均显存分配: {stats['avg_allocated']:.0f}MiB / {stats['total_memory']:.0f}MiB")
        print(f"最大显存分配: {stats['max_allocated']:.0f}MiB / {stats['total_memory']:.0f}MiB")
        print(f"平均显存缓存: {stats['avg_reserved']:.0f}MiB / {stats['total_memory']:.0f}MiB")
        print(f"最大显存缓存: {stats['max_reserved']:.0f}MiB / {stats['total_memory']:.0f}MiB")
        print(f"平均功耗: {stats['avg_power']:.1f}W / {stats['power_limit']:.1f}W")
        print(f"最大功耗: {stats['max_power']:.1f}W / {stats['power_limit']:.1f}W")

    # 打印所有结果的汇总
    print("\n\n所有时间窗口处理结果汇总:")
    print("时间窗口大小 | 平均FPS | 显存占比(分配/总计) | 显存占比(缓存/总计) | 功耗占比")
    print("------------|---------|-------------------|-------------------|----------")
    for result in results:
        allocated_ratio = f"{result['avg_allocated']:.0f}/{result['total_memory']:.0f}MiB"
        reserved_ratio = f"{result['avg_reserved']:.0f}/{result['total_memory']:.0f}MiB"
        power_ratio = f"{result['avg_power']:.1f}W/{result['power_limit']:.1f}W"
        print(f"{result['time_window']:12d} | {result['avg_fps']:7.2f} | {allocated_ratio:19} | {reserved_ratio:19} | {power_ratio}")

    # 创建CSV文件保存结果
    csv_path = os.path.join(save_path, 'processing_results.csv')
    with open(csv_path, 'w') as f:
        f.write("时间窗口大小,平均FPS,总处理帧数,总耗时(秒),平均显存分配(MiB),最大显存分配(MiB),平均显存缓存(MiB),最大显存缓存(MiB),总显存(MiB),平均功耗(W),最大功耗(W),功耗上限(W)\n")
        for result in results:
            f.write(
                f"{result['time_window']},{result['avg_fps']:.2f},{result['total_frames']},{result['total_time']:.2f},"
                f"{result['avg_allocated']:.0f},{result['max_allocated']:.0f},{result['avg_reserved']:.0f},"
                f"{result['max_reserved']:.0f},{result['total_memory']:.0f},{result['avg_power']:.1f},{result['max_power']:.1f},{result['power_limit']:.1f}\n")

    print(f"\n结果已保存到: {csv_path}")

if __name__ == "__main__":
    input_path = r'/root/1280.mp4'
    save_path = r'milu-re'

    # 要处理的时间窗口列表
    time_windows = [5, 7, 9, 10, 11, 13, 15, 17]

    batch_process_video(input_path, save_path, time_windows)