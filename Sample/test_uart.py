import serial
import time

print("--- 物理串口环回测试 (Loopback Test) ---")
print("提示：请确保 main.py 已关闭")

try:
    # 打开串口。
    ser = serial.Serial('/dev/ttyAMA0', 115200, timeout=1)
    print("串口打开成功")
    
    # 清空接收缓冲区。
    ser.reset_input_buffer()
    
    # 构造测试帧。
    test_frame = bytes([0xAA, 0x55, 0xFF, 0x01, 0x00, 0x10, 0x65, 0x0A])
    
    print(f"发送数据 -> : {' '.join([f'{b:02X}' for b in test_frame])}")
    ser.write(test_frame)
    
    # 等待数据回环。
    time.sleep(0.1)
    
    # 检查回环数据。
    if ser.in_waiting >= len(test_frame):
        recv_data = ser.read(ser.in_waiting)
        print(f"接收数据 <- : {' '.join([f'{b:02X}' for b in recv_data])}")
        
        if recv_data == test_frame:
            print("环回测试通过，硬件链路正常")
        else:
            print("收到数据，但内容不一致")
    else:
        print("发送后未收到数据，请检查")
        print("1. 引脚 8 和引脚 10 是否短接")
        print("2. raspi-config 中串口设置是否正确")
        
except serial.SerialException as e:
    print(f"串口打开失败: {e}")
except KeyboardInterrupt:
    print("\n测试手动中止")
finally:
    if 'ser' in locals() and ser.is_open:
        ser.close()