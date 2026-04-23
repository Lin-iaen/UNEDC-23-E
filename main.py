"""IBVS 主控制模块。

职责：
1. 管理系统状态机。
2. 协调相机、视觉、串口和 Web 控制。
3. 维护运行期共享数据。
"""

from __future__ import annotations

import logging
import math
import queue
import threading
import time
from enum import Enum, auto
from typing import List, Optional, Tuple

import cv2
import numpy as np

from camera import Camera
from tracker import process_init_mode, process_tracking_mode
from trajectory import TrajectoryController
from uart_comm import UartController
from web_stream import WebStreamServer

# 以下为主控制器实现。

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

class State(Enum):
    """系统主状态枚举。"""
    # auto() 自动分配值，状态按声明顺序递增。

    INIT = auto()
    ALIGN = auto()
    HOMING = auto()
    TRACKING = auto()
    IDLE = auto()
    FINISH = auto()


class FrameHub:
    """线程安全共享缓冲：保存最新处理后帧给 Web 层读取。"""

    def __init__(self):
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()

    def set(self, frame: Optional[np.ndarray]) -> None:
        if frame is None:
            return
        with self._lock:
            self._frame = frame

    def get(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()


def _corners_to_numpy(corners: List[Tuple[float, float]]) -> np.ndarray:
    """将角点列表转换为 OpenCV 透视变换所需的 float32 数组。"""
    return np.array(corners, dtype=np.float32)


def _max_corner_jitter(history: List[List[Tuple[float, float]]]) -> float:
    """计算角点历史序列的最大抖动（像素）。"""
    if not history:
        return float("inf")

    arr = np.array(history, dtype=np.float32)  # shape=(N,4,2)
    mean = arr.mean(axis=0, keepdims=True)
    diff = arr - mean
    dist = np.linalg.norm(diff, axis=2)  # shape=(N,4)
    return float(dist.max())


class IBVSController:
    """IBVS 主控制器：封装状态机、资源与状态处理逻辑。"""

    def __init__(
        self,
        canvas_width: int = 640,
        canvas_height: int = 480,
        step_size: float = 8.0,
    ) -> None:
        # 初始化画布和轨迹参数。
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.step_size = step_size

        # 状态机运行态。
        self.state = State.INIT
        self.perspective_matrix: Optional[np.ndarray] = None  # 透视矩阵，未完成标定时为空。
        self.trajectory: Optional[TrajectoryController] = None
        self.current_target: Optional[Tuple[float, float]] = None
        self.cmd_queue: queue.Queue[str] = queue.Queue()

        # INIT 阶段的稳定检测与人工确认控制。
        self.user_confirmed = False
        self.confirm_thread_started = False
        self.stable_frames_required = 12
        self.stable_jitter_threshold_px = 6.0
        self.stable_corner_history: List[List[Tuple[float, float]]] = []

        # 基础资源：相机、共享帧、串口、Web 服务。
        self.camera = Camera(width=self.canvas_width, height=self.canvas_height, fps=30)
        self.hub = FrameHub()
        self.uart = UartController()
        self.web_server = WebStreamServer(
            frame_provider=self.hub.get,
            jpeg_quality=85,
            on_command=self.cmd_queue.put,
        )
        self.web_thread: Optional[threading.Thread] = None

    def _process_command(self, cmd: str) -> None:
        """处理来自 Web 控制面板的命令。"""
        logger.info("[CMD] 收到命令: %s", cmd)

        if cmd == "home":
            self.trajectory = None
            self.current_target = None
            self.state = State.HOMING
            logger.info("状态迁移: * -> HOMING（命令触发）")
            return

        if cmd == "track" and self.state == State.IDLE:
            self.trajectory = TrajectoryController(
                width=self.canvas_width,
                height=self.canvas_height,
                step_size=self.step_size,
                loop=False,
            )
            logger.info(
                "[CMD] 轨迹控制器已创建，总目标点=%d",
                self.trajectory.total_points,
            )
            self.current_target = self.trajectory.get_next_target()
            self.state = State.TRACKING
            logger.info("状态迁移: IDLE -> TRACKING（命令触发）")
            return

        logger.info("[CMD] 忽略命令 cmd=%s, 当前状态=%s", cmd, self.state.name)

    def _wait_for_enter(self) -> None:
        """等待终端回车确认，用于 INIT->HOMING 人工放行。"""
        input("按回车键确认标定完成，系统将进入 HOMING 阶段...")
        self.user_confirmed = True

    def _restore_init_camera_profile(self, reason: str) -> None:
        """回退 INIT 前恢复建图曝光配置，确保画面重新可见。"""
        logger.warning("[CAMERA] 回退 INIT 前执行复明: %s", reason)
        self.camera.set_exposure(shutter_us=30000, gain=3.0, awb="auto")

    def _handle_init(self, frame: np.ndarray) -> None:
        # INIT: 提取角点，满足稳定条件后计算透视矩阵。

        # 返回标注画面和角点列表。
        annotated_frame, corners = process_init_mode(frame)

        if len(corners) == 4:
            self.stable_corner_history.append(corners)
            if len(self.stable_corner_history) > self.stable_frames_required:
                self.stable_corner_history.pop(0)

            jitter = _max_corner_jitter(self.stable_corner_history)

            if (
                len(self.stable_corner_history) >= self.stable_frames_required
                and jitter <= self.stable_jitter_threshold_px
            ):
                src = _corners_to_numpy(self.stable_corner_history[-1])
                dst = np.array(
                    [
                        (0.0, 0.0),
                        (640.0, 0.0),
                        (640.0, 480.0),
                        (0.0, 480.0),
                    ],
                    dtype=np.float32,
                )

                self.perspective_matrix = cv2.getPerspectiveTransform(src, dst)
                # 可在此输出矩阵用于标定排查。

                cv2.putText(
                    annotated_frame,
                    "Press ENTER in terminal",
                    (30, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.1,
                    (0, 255, 255),
                    3,
                )

                if not self.confirm_thread_started:
                    threading.Thread(
                        target=self._wait_for_enter,
                        daemon=True,
                        name="init-confirm-thread",
                    ).start()
                    self.confirm_thread_started = True
                    logger.info("[INIT] 已启动终端回车确认线程，等待用户确认")

                if self.user_confirmed:
                    self.camera.set_exposure(
                        shutter_us=30000,
                        gain=1.5,
                        awb="manual",
                    )
                    time.sleep(0.5)
                    self.state = State.HOMING
                    logger.info("状态迁移: INIT -> HOMING（用户已确认）")
        else:
            if self.stable_corner_history:
                logger.debug("[INIT] 角点中断，重置稳定计数")
            self.stable_corner_history.clear()

        self.hub.set(annotated_frame)

    def _handle_align(self) -> None:
        logger.info("跳过 ALIGN 自检，进入 HOMING")
        self.state = State.HOMING
        logger.info("状态迁移: ALIGN -> HOMING")

    def _handle_homing(self, frame: np.ndarray) -> None:
        # HOMING: 将激光拉回画布中心。

        if self.perspective_matrix is None:
            self._restore_init_camera_profile("[HOMING] perspective_matrix 丢失")
            logger.error("[HOMING] perspective_matrix 为空，回退到 INIT")
            self.state = State.INIT
            return

        homing_target = (self.canvas_width / 2.0, self.canvas_height / 2.0)
        annotated_frame, mapped_xy = process_tracking_mode(
            frame,
            self.perspective_matrix,
            current_target=homing_target,
        )
        self.hub.set(annotated_frame)

        if mapped_xy is None:
            # 未检测到激光点时保持等待。
            return

        laser_x, laser_y = mapped_xy
        dx = homing_target[0] - laser_x
        dy = homing_target[1] - laser_y
        self.uart.send_error(dx, dy)

        error_dist = math.hypot(dx, dy)
        # 可在此输出归中误差用于调参。

        if error_dist < 8.0:
            logger.info("已成功复位到中心")
            logger.info("复位完成，进入待机")
            self.state = State.IDLE
            logger.info("状态迁移: HOMING -> IDLE")

    def _handle_idle(self, frame: np.ndarray) -> None:
        """待机状态：仅显示中心诱饵，不发送 UART 控制。"""
        if self.perspective_matrix is None:
            self._restore_init_camera_profile("[IDLE] perspective_matrix 丢失")
            logger.error("[IDLE] perspective_matrix 为空，回退到 INIT")
            self.state = State.INIT
            return

        homing_target = (self.canvas_width / 2.0, self.canvas_height / 2.0)
        annotated_frame, _ = process_tracking_mode(
            frame,
            self.perspective_matrix,
            current_target=homing_target,
        )
        self.hub.set(annotated_frame)

    def _handle_tracking(self, frame: np.ndarray) -> None:
        # TRACKING: 计算误差并发送闭环控制量。

        if self.trajectory is None:
            self._restore_init_camera_profile("[TRACKING] trajectory 丢失")
            logger.error("[TRACKING] trajectory 丢失，回退到 INIT")
            self.state = State.INIT
            return

        if self.perspective_matrix is None:
            self._restore_init_camera_profile("[TRACKING] perspective_matrix 丢失")
            logger.error("[TRACKING] perspective_matrix 为空，回退到 INIT")
            self.state = State.INIT
            return

        if self.current_target is None:
            logger.warning("[TRACKING] 当前目标为空，尝试取下一个目标点")
            self.current_target = self.trajectory.get_next_target()

        if self.current_target is None:
            logger.info("[TRACKING] 轨迹执行完成，进入 FINISH")
            self.state = State.FINISH
            logger.info("状态迁移: TRACKING -> FINISH")
            return

        # 可在此输出当前目标点和轨迹索引。

        annotated_frame, mapped_xy = process_tracking_mode(
            frame,
            self.perspective_matrix,
            current_target=self.current_target,
        )
        self.hub.set(annotated_frame)

        if mapped_xy is None:
            logger.warning("未检测到激光点，等待中...")
            return

        laser_x, laser_y = mapped_xy
        dx = self.current_target[0] - laser_x
        dy = self.current_target[1] - laser_y
        self.uart.send_error(dx, dy)

        tolerance = 15.0
        reached = self.trajectory.check_and_fast_forward(laser_x, laser_y, tolerance, lookahead_window=15)

        if reached:
            # 到达或快进后更新下一个目标点。
            self.current_target = self.trajectory.get_next_target()
            if self.current_target is None:
                logger.info("[TRACKING] 无后续目标点，进入 FINISH")
                self.state = State.FINISH
                logger.info("状态迁移: TRACKING -> FINISH")

    def _handle_finish(self) -> None:
        logger.info("[FINISH] 任务结束，进行收尾处理")

        # TODO: 发送停机或复位指令。
        # TODO: 保存运行统计信息。
        logger.info("任务完成，1秒后系统回到 IDLE，等待下一次命令")
        time.sleep(1)

        self.current_target = None
        self.trajectory = None
        self.state = State.IDLE
        logger.info("状态迁移: FINISH -> IDLE（等待命令）")

    def _start_runtime_resources(self) -> None:
        logger.info("启动摄像头")
        self.camera.start()

        self.web_thread = threading.Thread(
            target=self.web_server.run,
            kwargs={"host": "0.0.0.0", "port": 5000, "threaded": True},
            daemon=True,
            name="web-stream-thread",
        )
        self.web_thread.start()
        # Web 服务后台线程已启动。

    def run(self) -> None:
        """主控制循环：获取最新帧并按状态分发处理。"""
        logger.info("系统启动，进入 IBVS 主循环")
        self._start_runtime_resources()

        try:
            while True:
                # 主循环：优先处理命令，再处理图像。

                while not self.cmd_queue.empty():
                    try:
                        cmd = self.cmd_queue.get_nowait()
                    except queue.Empty:
                        break
                    self._process_command(cmd)

                # 获取最新图像帧。
                frame = self.camera.get_latest_frame()
                if frame is None:
                    if self.state == State.INIT:
                        # INIT 等待图像。
                        pass
                    elif self.state == State.HOMING:
                        # HOMING 等待图像。
                        pass
                    elif self.state == State.TRACKING:
                        # TRACKING 等待图像。
                        pass
                    elif self.state == State.IDLE:
                        # IDLE 等待图像。
                        pass
                    time.sleep(0.02)
                    continue

                if self.state == State.INIT:
                    self._handle_init(frame)
                elif self.state == State.ALIGN:
                    self._handle_align()
                elif self.state == State.HOMING:
                    self._handle_homing(frame)
                elif self.state == State.TRACKING:
                    self._handle_tracking(frame)
                elif self.state == State.IDLE:
                    self._handle_idle(frame)
                elif self.state == State.FINISH:
                    self._handle_finish()
                else:
                    logger.error("检测到未知状态=%s，强制回到 INIT", self.state)
                    self._restore_init_camera_profile("未知状态保护回退")
                    self.state = State.INIT

                # 降低空转占用，后续可按帧率同步。
                time.sleep(0.01)
        finally:
            logger.info("正在关闭 Camera 资源")
            self.camera.stop()


def main() -> None:
    controller = IBVSController()
    controller.run()


if __name__ == "__main__":
    main()