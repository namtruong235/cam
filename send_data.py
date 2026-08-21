import json
import math

import numpy as np


import serial
import time

# Mở cổng COMx (thay COMx bằng cổng thực tế, ví dụ: COM4, COM5...)
com_port = 'COM8'  # Thay COM4 bằng cổng COM của bạn
baud_rate = 9600    # Tốc độ baud

data = {
    "x": 60,
    "y": 120,
    "a": 90,
}

# Khởi tạo kết nối với cổng COM
ser = serial.Serial(com_port, baud_rate, timeout=1)

# Kiểm tra nếu cổng COM được mở thành công
if ser.is_open:
    print(f"Đã kết nối với {com_port} thành công.")

# Gửi dữ liệu đến cổng COM
while True:
    # json_data = json.dumps(data)
    json_data = str(60) + ";" + str(120) + ";" + str(90) + "#"

    # Gửi chuỗi JSON qua cổng Serial
    ser.write(json_data.encode())  # Chuyển đổi dữ liệu thành dạng byte và gửi đi
    print(f"Đã gửi: {json_data}")

    # Chờ 1 giây trước khi gửi tiếp
    time.sleep(0.05)

# Đóng cổng COM sau khi sử dụng
ser.close()

# def calculate_angle(vector1, vector2):
#     dot_product = np.dot(vector1, vector2)
#
#     magnitude_vector1 = np.linalg.norm(vector1)
#     magnitude_vector2 = np.linalg.norm(vector2)
#
#     cos_theta = dot_product / (magnitude_vector1 * magnitude_vector2)
#
#     angle_radians = np.arccos(cos_theta)
#
#     angle_degrees = np.degrees(angle_radians)
#
#     return angle_degrees
#
#
# print(math.copysign(int(calculate_angle((10, 0), (7, -7))), -7))
