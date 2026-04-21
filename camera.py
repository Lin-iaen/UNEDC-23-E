import cv2
import threading
import subprocess
import numpy as np
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class Camera:
    """封装基于 libcamera-vid 的内存管道流，绕过树莓派 5 的 V4L2 硬件限制"""

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps

        self._latest_frame = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None
        self.process = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        print("[INFO] 正在建立 libcamera 内存数据流管道...")
        
        # 核心：使用 rpicam-vid 开启无限流 (-t 0)
        # --codec mjpeg: 输出 MJPEG 视频流
        # -o - : 极其关键，不保存文件，直接输出到标准输出 (stdout)
        cmd = [
            "rpicam-vid",
            "-t", "0",
            "--width", str(self.width),
            "--height", str(self.height),
            "--framerate", str(self.fps),
            "--codec", "mjpeg",
            "-o", "-",
            "--nopreview"
        ]

        try:
            # 建立子进程管道
            self.process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.DEVNULL, # 屏蔽底层烦人的警告信息
                bufsize=10**8
            )
        except FileNotFoundError:
            raise RuntimeError("系统未安装 rpicam-vid，请检查硬件系统！")

        self._stop_event.clear()
        
        # 启动消费者后台线程，死死盯住这根管道
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        
        # 稍微等个0.5秒，让底层硬件启动并吐出第一帧
        time.sleep(0.5)
        if self._latest_frame is None:
            print("[WARN] 管道已建立，但暂时没读到图像，等待流中...")
        else:
            print("[INFO] 高速视频流通道已建立！")

    def stop(self) -> None:
        self._stop_event.set()
        if self.process:
            self.process.terminate()
            self.process.wait()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        print("[INFO] 摄像头管道已安全关闭")

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """瞬间返回内存中的最新一帧副本"""
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def _capture_loop(self) -> None:
        """后台字节流猎手：不断从管道中寻找完整的 JPEG 照片"""
        if self.process is None or self.process.stdout is None:
            return

        stdout = self.process.stdout
        bytes_stream = b''
        while not self._stop_event.is_set():
            # 每次从管道里吸取 4096 字节
            chunk = stdout.read(4096)
            if not chunk:
                time.sleep(0.01)
                continue
                
            bytes_stream += chunk
            
            # 在字节流中寻找 JPEG 图像的“开头”和“结尾”标志
            a = bytes_stream.find(b'\xff\xd8') # JPEG start
            b = bytes_stream.find(b'\xff\xd9') # JPEG end
            
            if a != -1 and b != -1:
                # 成功剥离出一张完整的图片字节码
                jpg_data = bytes_stream[a:b+2]
                # 把处理过的数据扔掉，保持内存干净
                bytes_stream = bytes_stream[b+2:]
                
                # 让 OpenCV 将字节码解码成真实的图像矩阵
                frame = cv2.imdecode(np.frombuffer(jpg_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    # 修正相机画面上下颠倒：按垂直方向翻转。
                    frame = cv2.flip(frame, 0)
                    with self._lock:
                        self._latest_frame = frame