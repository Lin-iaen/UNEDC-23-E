import logging
import time
from typing import Callable, Optional

import cv2
import numpy as np
from flask import Flask, Response, jsonify


logger = logging.getLogger(__name__)


class WebStreamServer:
	"""仅负责 Flask 服务与 MJPEG 推流，不包含硬件采集逻辑。"""

	def __init__(
		self,
		frame_provider: Callable[[], Optional[np.ndarray]],
		jpeg_quality: int = 85,
		on_command: Optional[Callable[[str], None]] = None,
	):
		self.frame_provider = frame_provider
		self.jpeg_quality = jpeg_quality
		self.on_command = on_command
		self.app = Flask(__name__)
		self._register_routes()

	def _register_routes(self) -> None:
		@self.app.route("/")
		def index() -> str:
			return (
				"<!doctype html>"
				"<html><head><meta charset='utf-8'><title>Vision Control Panel</title>"
				"<style>"
				"body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f3f5f7; }"
				"h2 { margin: 0 0 12px 0; }"
				".layout { display: grid; grid-template-columns: 2fr 1fr; gap: 20px; align-items: start; }"
				".stream { background: #fff; padding: 12px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.08); }"
				".stream img { width: 100%; height: auto; border-radius: 8px; }"
				".panel { background: #fff; padding: 16px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.08); }"
				".panel h3 { margin-top: 0; }"
				".btn { width: 100%; border: none; color: #fff; font-size: 20px; font-weight: bold; padding: 18px 12px; border-radius: 10px; cursor: pointer; margin-bottom: 12px; }"
				".btn-home { background: #d63031; }"
				".btn-track { background: #2e7d32; }"
				".status { font-size: 14px; color: #333; min-height: 20px; }"
				"@media (max-width: 900px) { .layout { grid-template-columns: 1fr; } }"
				"</style></head>"
				"<body>"
				"<h2>Vision Control Panel</h2>"
				"<div class='layout'>"
				"<div class='stream'><img src='/video_feed' alt='video stream'></div>"
				"<div class='panel'>"
				"<h3>控制面板</h3>"
				"<button class='btn btn-home' onclick=\"sendCmd('home')\">复位中心 (Homing)</button>"
				"<button class='btn btn-track' onclick=\"sendCmd('track')\">开始循迹 (Track)</button>"
				"<div id='status' class='status'>等待指令...</div>"
				"</div>"
				"</div>"
				"<script>"
				"function sendCmd(cmd) {"
				"  fetch('/api/command/' + cmd)"
				"    .then(r => r.json())"
				"    .then(d => { document.getElementById('status').textContent = '已发送: ' + d.command; })"
				"    .catch(() => { document.getElementById('status').textContent = '发送失败'; });"
				"}"
				"</script>"
				"</body></html>"
			)

		@self.app.route("/video_feed")
		def video_feed() -> Response:
			return Response(
				self._mjpeg_generator(),
				mimetype="multipart/x-mixed-replace; boundary=frame",
			)

		@self.app.route("/api/command/<cmd>")
		def command(cmd: str):
			if self.on_command:
				self.on_command(cmd)
			return jsonify({"status": "success", "command": cmd})

	def _mjpeg_generator(self):
		while True:
			try:
				frame = self.frame_provider()
				if frame is None:
					time.sleep(0.02)
					continue

				ok, jpeg = cv2.imencode(
					".jpg",
					frame,
					[cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
				)
				if not ok:
					logger.warning("Failed to encode JPEG frame")
					continue

				yield (
					b"--frame\r\n"
					b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
				)
			except Exception:
				logger.exception("Streaming loop error")
				time.sleep(0.02)

	def run(self, host: str = "0.0.0.0", port: int = 5000, threaded: bool = True) -> None:
		self.app.run(host=host, port=port, threaded=threaded)
