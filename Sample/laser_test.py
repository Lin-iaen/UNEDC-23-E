import cv2
import numpy as np
import subprocess

def capture_extreme_dark(filename="dark_scene.jpg"):
    print("[INFO] 正在以暗场模式启动 CSI 摄像头")
    
    # 参数说明：短快门、低增益和固定白平衡用于抑制环境光。
    
    cmd = [
        "rpicam-jpeg",
        "--nopreview",
        "-t", "1000",
        "--shutter", "500",
        "--gain", "1",
        "--awbgains", "1,1",
        "-o", filename
    ]
    
    subprocess.run(cmd)
    print(f"[INFO] 暗场图像已保存为 {filename}")
    return filename

def analyze_light_intensity(filename):
    print("[INFO] OpenCV 正在分析光强")
    
    img = cv2.imread(filename)
    if img is None:
        print("[ERROR] 图像读取失败，请检查摄像头是否正常工作")
        return

    # 缩小分辨率以加快处理。
    img = cv2.resize(img, (640, 480))

    # OpenCV 通道顺序为 B, G, R。
    b, g, r = cv2.split(img)
    
    # 输出每个通道的最大亮度。
    max_r = np.max(r)
    max_g = np.max(g)
    max_b = np.max(b)
    
    print("-" * 40)
    print(f"🔴 画面中最亮的红光强度: {max_r} / 255")
    print(f"🟢 画面中最亮的绿光强度: {max_g} / 255")
    print(f"🔵 画面中最亮的蓝光强度: {max_b} / 255")
    print("-" * 40)

    # 二值化阈值测试：定位高亮区域。
    _, red_mask = cv2.threshold(r, 200, 255, cv2.THRESH_BINARY)
    _, green_mask = cv2.threshold(g, 200, 255, cv2.THRESH_BINARY)

    # 保存掩膜结果。
    cv2.imwrite("red_laser_mask.jpg", red_mask)
    cv2.imwrite("green_laser_mask.jpg", green_mask)
    
    print("[INFO] 测试完成")
    print("请在目录中检查以下文件")
    print("1. dark_scene.jpg")
    print("2. red_laser_mask.jpg")
    print("3. green_laser_mask.jpg")

if __name__ == "__main__":
    img_file = capture_extreme_dark()
    analyze_light_intensity(img_file)