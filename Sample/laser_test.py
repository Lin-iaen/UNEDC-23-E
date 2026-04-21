import cv2
import numpy as np
import subprocess

def capture_extreme_dark(filename="dark_scene.jpg"):
    print("[INFO] 正在以【极限暗场模式】唤醒 CSI 摄像头...")
    
    # 核心物理降维参数解析：
    # --shutter 500 : 快门速度设为 500 微秒 (0.5毫秒)，极其短暂，扼杀所有环境光
    # --gain 1      : 将传感器的模拟增益（感光度 ISO）降到最低的 1.0，杜绝噪点
    # --awbgains 1,1: 锁定红蓝通道的白平衡增益比例为 1:1，防止系统在黑夜里瞎调颜色
    # --nopreview   : 关闭无头系统不支持的预览
    # -t 1000       : 预热 1 秒，让硬件初始化完毕
    
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
    print(f"[INFO] 咔嚓！暗场原图已保存为 {filename}")
    return filename

def analyze_light_intensity(filename):
    print("[INFO] OpenCV 正在对画面进行光子强度扫描...")
    
    img = cv2.imread(filename)
    if img is None:
        print("[ERROR] 图像读取失败，请检查摄像头是否正常工作！")
        return

    # 为了加快处理速度并过滤细微噪点，缩小分辨率
    img = cv2.resize(img, (640, 480))

    # 在 OpenCV 中，彩色图像的通道顺序是 B, G, R
    b, g, r = cv2.split(img)
    
    # 打印每个通道的全局最高亮度 (0-255)
    max_r = np.max(r)
    max_g = np.max(g)
    max_b = np.max(b)
    
    print("-" * 40)
    print(f"🔴 画面中最亮的红光强度: {max_r} / 255")
    print(f"🟢 画面中最亮的绿光强度: {max_g} / 255")
    print(f"🔵 画面中最亮的蓝光强度: {max_b} / 255")
    print("-" * 40)

    # 粗暴且极其有效的二值化测试：寻找亮度超过 200 的发光体
    # 因为我们扼杀了环境光，能超过 200 的大概率只有激光或直射光源
    _, red_mask = cv2.threshold(r, 200, 255, cv2.THRESH_BINARY)
    _, green_mask = cv2.threshold(g, 200, 255, cv2.THRESH_BINARY)

    # 保存掩膜结果
    cv2.imwrite("red_laser_mask.jpg", red_mask)
    cv2.imwrite("green_laser_mask.jpg", green_mask)
    
    print("[INFO] 物理降维测试完毕！")
    print("👉 请在 VS Code 左侧目录双击打开以下三张图片进行验收：")
    print("   1. dark_scene.jpg      (查看是否做到了全屏漆黑，只有光源)")
    print("   2. red_laser_mask.jpg  (寻找红光掩膜)")
    print("   3. green_laser_mask.jpg(寻找绿光掩膜)")

if __name__ == "__main__":
    img_file = capture_extreme_dark()
    analyze_light_intensity(img_file)