"""UART 通信控制桩模块。

当前阶段仅提供控制接口与日志留痕，后续再替换为真实串口收发实现。
"""

from __future__ import annotations

import logging
import time


logger = logging.getLogger(__name__)


class UartController:
    """串口控制器（占位实现）。"""

    def __init__(self) -> None:
        self._last_log_time = 0.0

    def send_error(self, dx: float, dy: float) -> None:
        """发送闭环微调误差。

        参数：
        - dx: X 方向误差
        - dy: Y 方向误差
        """
        # TODO: 在此处接入真实串口发送逻辑（当前仅保留接口）。

        now = time.time()
        if now - self._last_log_time < 0.2:
            return

        self._last_log_time = now
        logger.info(f"UART 发送微调指令: dx={dx:.2f}, dy={dy:.2f}")
