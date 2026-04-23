import cv2
import threading
import subprocess
import numpy as np
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class Camera:
    """基于 rpicam-vid 的内存视频流封装。"""

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        shutter_us: int = 30000,
        gain: float = 3.0,
        awb: str = "auto",
    ):
        self.width = width
        self.height = height
        self.fps = fps

        self.shutter_us = shutter_us
        self.gain = gain
        self.awb = awb

        self._latest_frame = None
        self._lock = threading.Lock()
        self._restart_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None
        self.process = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        print("[INFO] 正在建立 libcamera 内存数据流管道")
        
        awb_mode = str(self.awb).strip().lower()
        awb_extra_args = []
        if awb_mode == "manual":
            # rpicam-vid 不支持 manual，使用 custom + 固定增益实现等价锁定效果。
            awb_mode = "custom"
            awb_extra_args = ["--awbgains", "1,1"]

        # 使用 rpicam-vid 持续输出 MJPEG 到 stdout。
        cmd = [
            "rpicam-vid",
            "-t", "0",
            "--width", str(self.width),
            "--height", str(self.height),
            "--framerate", str(self.fps),
            "--codec", "mjpeg",
            "--shutter", str(self.shutter_us),
            "--gain", str(self.gain),
            "--awb", awb_mode,
            "--vflip",
            "--hflip",
            "-o", "-",
            "--nopreview"
        ]
        cmd.extend(awb_extra_args)

        try:
            # 建立采集子进程。
            self.process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                bufsize=10**8
            )
        except FileNotFoundError:
            raise RuntimeError("系统未安装 rpicam-vid，请检查硬件系统")

        self._stop_event.clear()
        
        # 启动后台读取线程。
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        
        # 等待相机输出首帧。
        time.sleep(0.5)
        if self.process and self.process.poll() is not None:
            error_message = ""
            if self.process.stderr is not None:
                try:
                    error_message = self.process.stderr.read().decode("utf-8", errors="ignore").strip()
                except Exception:
                    error_message = ""
            print(f"[ERROR] rpicam-vid 启动失败，退出码={self.process.returncode}")
            if error_message:
                print(f"[ERROR] rpicam-vid stderr: {error_message}")

        if self._latest_frame is None:
            print("[WARN] 管道已建立，当前未读取到图像")
        else:
            print("[INFO] 视频流通道已建立")

    def stop(self) -> None:
        self._stop_event.set()
        if self.process:
            self.process.terminate()
            self.process.wait()
            self.process = None
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        print("[INFO] 摄像头管道已关闭")

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """返回内存中最新一帧的副本。"""
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def _capture_loop(self) -> None:
        """后台读取循环：从字节流中提取完整 JPEG。"""
        if self.process is None or self.process.stdout is None:
            return

        stdout = self.process.stdout
        bytes_stream = b''
        while not self._stop_event.is_set():
            # 每次读取固定字节数，拼接到缓存。
            chunk = stdout.read(4096)
            if not chunk:
                time.sleep(0.01)
                continue
                
            bytes_stream += chunk
            
            # 定位 JPEG 起止标记。
            a = bytes_stream.find(b'\xff\xd8') # JPEG start
            b = bytes_stream.find(b'\xff\xd9') # JPEG end
            
            if a != -1 and b != -1:
                # 截取单帧 JPEG 数据。
                jpg_data = bytes_stream[a:b+2]
                # 移除已处理字节。
                bytes_stream = bytes_stream[b+2:]
                
                # 解码为 BGR 图像。
                frame = cv2.imdecode(np.frombuffer(jpg_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    with self._lock:
                        self._latest_frame = frame

    def set_exposure(self, shutter_us: int, gain: float = 1.0, awb: str = "auto") -> None:
        """动态修改曝光参数并通过重启底层管道生效。"""
        with self._restart_lock:
            print(
                "[INFO] 相机硬件参数热重载: "
                f"shutter_us={shutter_us}, gain={gain}, awb={awb}"
            )

            # 停止旧采集循环并回收子进程，确保新参数由硬件重新加载。
            self._stop_event.set()
            if self.process:
                self.process.terminate()
                self.process.wait()
                self.process = None

            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=1.0)
            self._thread = None

            # 清空旧帧，避免主流程读取到重启前残影。
            with self._lock:
                self._latest_frame = None

            self.shutter_us = shutter_us
            self.gain = gain
            self.awb = awb

            self.start()