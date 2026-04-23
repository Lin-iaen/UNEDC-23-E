import cv2
import numpy as np
import threading
import subprocess
import time
from flask import Flask, Response, request, render_template_string, jsonify

# ================= 配置区 =================
cam_params = {
    "shutter": 25000,   # 快门 (微秒)，30fps物理极限约33000
    "gain": 1.0,        # 增益
    "awb": "auto",      # 白平衡
    "brightness": 0.0,
    "contrast": 1.0
}
# ==========================================

class CameraStreamer:
    def __init__(self):
        self.process = None
        self.frame = None
        self.lock = threading.Lock()
        self.start_pipeline()
        threading.Thread(target=self._read_loop, daemon=True).start()

    def start_pipeline(self):
        if self.process:
            self.process.terminate()
            self.process.wait()

        # ====== 硬件级降维打击 ======
        # 加入了 --vflip 和 --hflip，直接命令硬件翻转，解放 CPU！
        cmd = [
            "rpicam-vid", "-t", "0",
            "--width", "640", "--height", "480", "--framerate", "30",
            "--codec", "mjpeg", "-o", "-", "--nopreview",
            "--shutter", str(cam_params["shutter"]),
            "--gain", str(cam_params["gain"]),
            "--awb", cam_params["awb"],
            "--brightness", str(cam_params["brightness"]),
            "--contrast", str(cam_params["contrast"]),
            "--vflip", "--hflip" 
        ]
        print(f"\n[启动相机] {' '.join(cmd)}\n")
        self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def _read_loop(self):
        bytes_stream = b''
        while True:
            if not self.process or not self.process.stdout:
                time.sleep(0.05)
                continue
            
            chunk = self.process.stdout.read(4096)
            if not chunk: continue
                
            bytes_stream += chunk
            a = bytes_stream.find(b'\xff\xd8')
            b = bytes_stream.find(b'\xff\xd9')
            
            if a != -1 and b != -1:
                jpg = bytes_stream[a:b+2]
                bytes_stream = bytes_stream[b+2:]
                img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    # CPU 翻转代码 (cv2.flip) 已被彻底删除！
                    with self.lock:
                        self.frame = img

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

# ================= Web 服务区 =================
app = Flask(__name__)
streamer = None

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>CSI Camera 物理级调参</title>
    <style>
        body { font-family: sans-serif; background: #222; color: #eee; margin: 0; padding: 20px; display: flex; gap: 20px;}
        .video-box { flex: 2; }
        .video-box img { width: 100%; border: 2px solid #555; border-radius: 8px; }
        .control-box { flex: 1; background: #333; padding: 20px; border-radius: 8px; }
        .control-group { margin-bottom: 20px; }
        label { display: flex; justify-content: space-between; margin-bottom: 5px; font-weight: bold;}
        input[type=range] { width: 100%; }
        select { width: 100%; padding: 5px; background: #444; color: white; border: 1px solid #666;}
        button { width: 100%; padding: 10px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; margin-bottom: 10px;}
        button.dark { background: #dc3545; }
    </style>
</head>
<body>
    <div class="video-box">
        <h2>实时监控 (零 CPU 算力翻转版)</h2>
        <img src="/video_feed" alt="Camera Stream">
    </div>
    <div class="control-box">
        <h2>物理参数控制台</h2>
        
        <div class="control-group">
            <label>快门 Shutter (us) <span id="shutter_val">25000</span></label>
            <input type="range" id="shutter" min="500" max="33000" step="500" value="25000" oninput="updateVal('shutter')">
        </div>

        <div class="control-group">
            <label>增益 Gain (ISO) <span id="gain_val">1.0</span></label>
            <input type="range" id="gain" min="1.0" max="16.0" step="0.5" value="1.0" oninput="updateVal('gain')">
        </div>

        <div class="control-group">
            <label>对比度 Contrast <span id="contrast_val">1.0</span></label>
            <input type="range" id="contrast" min="0.0" max="2.0" step="0.1" value="1.0" oninput="updateVal('contrast')">
        </div>

        <div class="control-group">
            <label>白平衡 AWB</label>
            <select id="awb" onchange="sendParams()">
                <option value="auto">自动 (Auto)</option>
                <option value="custom">锁死 (Manual)</option>
            </select>
        </div>

        <button onclick="sendParams()">写寄存器！(重启底层)</button>
        <hr style="border-color: #555; margin: 20px 0;">
        <button class="dark" onclick="presetLaser()">追光模式 (Shutter=2000, Gain=1)</button>
        <button onclick="presetNormal()">建图模式 (Shutter=25000, Auto)</button>
    </div>

    <script>
        function updateVal(id) { document.getElementById(id + '_val').innerText = document.getElementById(id).value; }
        function sendParams() {
            let data = {
                shutter: parseInt(document.getElementById('shutter').value),
                gain: parseFloat(document.getElementById('gain').value),
                contrast: parseFloat(document.getElementById('contrast').value),
                awb: document.getElementById('awb').value
            };
            fetch('/update', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(data) });
        }
        function presetLaser() {
            document.getElementById('shutter').value = 2000; document.getElementById('gain').value = 1.0; document.getElementById('awb').value = 'custom';
            updateVal('shutter'); updateVal('gain'); sendParams();
        }
        function presetNormal() {
            document.getElementById('shutter').value = 25000; document.getElementById('gain').value = 1.0; document.getElementById('awb').value = 'auto';
            updateVal('shutter'); updateVal('gain'); sendParams();
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index(): return render_template_string(HTML_TEMPLATE)

@app.route('/update', methods=['POST'])
def update():
    global cam_params
    cam_params.update(request.json)
    streamer.start_pipeline()
    return jsonify({"status": "success"})

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            frame = streamer.get_frame()
            if frame is None:
                time.sleep(0.05)
                continue
            ret, jpeg = cv2.imencode('.jpg', frame)
            if ret:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    subprocess.run(["killall", "-9", "rpicam-vid"], stderr=subprocess.DEVNULL)
    streamer = CameraStreamer()
    print("\n[INFO] 物理实验室已启动！请访问: http://<树莓派IP>:5000\n")
    app.run(host='0.0.0.0', port=5000, threaded=True)