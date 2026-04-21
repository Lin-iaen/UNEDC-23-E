"""轨迹规划模块（Trajectory Planner）。

本模块负责：
1. 将矩形边界离散为细粒度目标点序列（插补轨迹）；
2. 通过统一步长降低“狗骨头效应”带来的拐角误差；
3. 向主控制循环提供稳定、可迭代的目标点输出接口。
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


logger = logging.getLogger(__name__)


Point = Tuple[float, float]


def _interpolate_segment(start: Point, end: Point, step_size: float) -> List[Point]:
	"""对线段进行等距插补，返回包含起点和终点的点序列。

	说明：
	- 使用欧氏距离计算该线段需要的采样数；
	- 采用线性插值生成中间点，保证相邻点间距接近 step_size；
	- 若线段长度不足 step_size，也至少返回 [start, end]。
	"""
	x0, y0 = start
	x1, y1 = end
	dx = x1 - x0
	dy = y1 - y0
	length = math.hypot(dx, dy)

	if length == 0:
		return [start]

	# 至少切分为 1 段，避免除零；向上取整保证步长不超过 step_size。
	steps = max(1, math.ceil(length / step_size))
	points: List[Point] = []

	for i in range(steps + 1):
		t = i / steps
		points.append((x0 + dx * t, y0 + dy * t))

	return points


def generate_rect_path(width: float, height: float, step_size: float) -> List[Point]:
	"""生成矩形闭环轨迹点列。

	参数：
	- width: 矩形宽度（像素）
	- height: 矩形高度（像素）
	- step_size: 插补步长（像素），例如 2.0

	返回：
	- 按顺时针方向排列的闭环点序列，起点为 (0, 0)

	轨迹顺序：
	(0,0) -> (width,0) -> (width,height) -> (0,height) -> (0,0)
	"""
	if width <= 0:
		raise ValueError("width 必须大于 0")
	if height <= 0:
		raise ValueError("height 必须大于 0")
	if step_size <= 0:
		raise ValueError("step_size 必须大于 0")

	corners: List[Point] = [
		(0.0, 0.0),
		(float(width), 0.0),
		(float(width), float(height)),
		(0.0, float(height)),
		(0.0, 0.0),
	]

	path: List[Point] = []
	for idx in range(len(corners) - 1):
		segment = _interpolate_segment(corners[idx], corners[idx + 1], step_size)

		if idx > 0 and segment:
			# 后续线段首点与前一线段终点重复，跳过以避免重复目标点。
			segment = segment[1:]

		path.extend(segment)

	logger.debug(
		"轨迹生成完成: width=%.2f, height=%.2f, step_size=%.2f, points=%d",
		width,
		height,
		step_size,
		len(path),
	)
	return path


@dataclass
class TrajectoryController:
	"""轨迹控制器：维护离散目标点列表并按序输出。"""

	width: float
	height: float
	step_size: float
	loop: bool = False
	_path: List[Point] = field(init=False, repr=False)
	_index: int = field(default=0, init=False, repr=False)

	def __post_init__(self) -> None:
		self._path = generate_rect_path(
			width=self.width,
			height=self.height,
			step_size=self.step_size,
		)
		logger.info("TrajectoryController 初始化完成，目标点总数=%d", len(self._path))

	@property
	def path(self) -> List[Point]:
		"""返回轨迹点副本，避免外部意外篡改内部状态。"""
		return list(self._path)

	@property
	def total_points(self) -> int:
		return len(self._path)

	@property
	def current_index(self) -> int:
		return self._index

	def reset(self) -> None:
		self._index = 0
		logger.info("TrajectoryController 已重置到起点")

	def get_next_target(self) -> Optional[Point]:
		"""获取下一个目标点。

		返回：
		- 非循环模式 loop=False：轨迹结束后返回 None；
		- 循环模式 loop=True：轨迹结束后从头开始。
		"""
		if not self._path:
			logger.warning("轨迹为空，无法提供目标点")
			return None

		if self._index >= len(self._path):
			if not self.loop:
				logger.debug("轨迹已完成，loop=False，返回 None")
				return None

			logger.debug("轨迹已完成，loop=True，回到起点继续")
			self._index = 0

		target = self._path[self._index]
		logger.debug(
			"输出目标点 index=%d/%d, target=(%.3f, %.3f)",
			self._index,
			len(self._path) - 1,
			target[0],
			target[1],
		)
		self._index += 1
		return target

	def check_and_fast_forward(self, laser_x: float, laser_y: float, tolerance: float, lookahead_window: int = 15) -> bool:
		"""
		前瞻窗口机制：检查激光是否已经走在了轨迹前面。
		如果在前方 lookahead_window 个点内找到了满足容差的点，直接快进到该点。
		"""
		if not self._path:
			return False

		# 当前正在追踪的点的索引其实是 self._index - 1
		# 因为 get_next_target 调用后 self._index 已经自增了
		current_idx = max(0, self._index - 1)
		
		# 防跌落保护
		if current_idx >= len(self._path):
			return False

		# 视野范围：从当前点，一直看到前方 lookahead_window 个点
		end_idx = min(current_idx + lookahead_window + 1, len(self._path))
		
		for i in range(current_idx, end_idx):
			tx, ty = self._path[i]
			dist = math.hypot(tx - laser_x, ty - laser_y)
			
			if dist <= tolerance:
				# 如果吃到了后面的点，触发快进机制
				if i > current_idx:
					logger.debug(f"快进跳跃到点 {i}")
					# 快进后，下一个该取的点应该是 i + 1
					self._index = i + 1  
				return True  # 满足容差，任务可以继续推演
				
		return False
