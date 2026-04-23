"""UART 通信模块。"""

from __future__ import annotations

import logging
import struct
import time
import serial


logger = logging.getLogger(__name__)


class UartController:
    """物理串口控制器。"""

    def __init__(self, port: str = "/dev/ttyAMA0", baudrate: int = 115200) -> None:
        self._last_log_time = 0.0
        self.port = port
        self.baudrate = baudrate
        self.serial: serial.Serial | None = None

        self._connect()

    def _connect(self) -> None:
        """尝试打开物理串口。"""
        try:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=0.1,
            )
            logger.info(f"串口 {self.port} 打开成功，波特率: {self.baudrate}")
        except serial.SerialException as e:
            logger.error(f"无法打开串口 {self.port}: {e}")
            self.serial = None

    def send_error(self, dx: float, dy: float) -> None:
        """打包并发送闭环微调误差。"""
        # 日志节流，减少高频输出。
        now = time.time()
        should_log = (now - self._last_log_time) >= 0.2
        if should_log:
            self._last_log_time = now

        if self.serial is None or not self.serial.is_open:
            if should_log:
                logger.warning("串口未开启，跳过发送")
            return

        # 1) 误差缩放为 int16，保留 1 位小数精度。
        idx = int(round(dx * 10))
        idy = int(round(dy * 10))

        # 限制为 int16 范围。
        idx = max(-32768, min(32767, idx))
        idy = max(-32768, min(32767, idy))

        # 2) 按协议打包（大端序）。
        try:
            # 打包帧头与误差数据。
            header_data = struct.pack('>BBhh', 0xAA, 0x55, idx, idy)
            
            # 3) 累加和校验。
            checksum = sum(header_data) & 0xFF
            
            # 4) 组装完整数据帧。
            frame = header_data + struct.pack('>BB', checksum, 0x0A)
            
            # 5) 发送字节流。
            self.serial.write(frame)

            if should_log:
                # 输出十六进制帧用于调试。
                hex_str = " ".join([f"{b:02X}" for b in frame])
                logger.info(f"UART 发送: dx={dx:.1f}, dy={dy:.1f} | Frame: [{hex_str}]")
            
            time.sleep(0.005) 
            if self.serial.in_waiting >= 8:
                recv = self.serial.read(8)
                recv_hex = " ".join([f"{b:02X}" for b in recv])
                if should_log:
                    logger.info(f"==> 环回收到: [{recv_hex}]")
                
        except Exception as e:
            logger.error(f"串口发送异常: {e}")