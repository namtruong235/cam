# 3-Robot ArUco Tracking & Waypoint Control

Hệ thống theo dõi **3 robot bằng 2 camera**, sử dụng **ArUco + Homography**, hiển thị vị trí/góc/quỹ đạo trên dashboard OpenCV và truyền dữ liệu tới **ESP32 Gateway** qua Serial.

## 1. Robot hiện tại

- Robot 1: ArUco ID **8**
- Robot 2: ArUco ID **7**
- Robot 3: ArUco ID **3**
- ArUco dictionary: `DICT_4X4_100`

> Nếu repository dùng bản `easy-change-ID`, nên gom ID về một khu vực cấu hình duy nhất để người dùng chỉ sửa một số khi đổi marker.

---

## 2. Kiến trúc hệ thống

```text
Camera 1 ─┐
          ├──> Python + OpenCV
Camera 2 ─┘        │
                   │ ArUco + Homography
                   │ X, Y, Angle
                   ▼
                USB Serial
                   │
                   ▼
             ESP32 Gateway
                   │
                ESP-NOW
          ┌────────┼────────┐
          ▼        ▼        ▼
       ESP32 R8 ESP32 R7 ESP32 R3
          │        │        │
         UART     UART     UART
          │        │        │
          ▼        ▼        ▼
        Mega     Mega     Mega
```

---

## 3. Chức năng chính

Chương trình Python hỗ trợ:

- Đọc đồng thời 2 camera USB.
- Nhận dạng ArUco.
- Tính tọa độ thực `X, Y` bằng Homography.
- Tính góc robot trong hệ tọa độ thực.
- Hợp nhất dữ liệu Camera 1 / Camera 2 trong vùng overlap.
- Giữ camera đang active để hạn chế nhảy tọa độ.
- Lọc vị trí bằng EMA.
- Lọc góc và xử lý đúng trường hợp `359° -> 1°`.
- Vẽ toàn bộ quỹ đạo robot.
- Tính khoảng cách, tốc độ, thời gian chạy.
- START / STOP / CLEAR PATH riêng cho từng robot.
- Chấm nhiều điểm đích trực tiếp trên map.
- Gửi danh sách waypoint trong **một packet**.
- `CLEAR TARGET` độc lập với `CLEAR PATH`.
- Camera preview chỉ mở khi người dùng bấm nút.
- Luồng Serial chạy độc lập với FPS camera, mặc định khoảng `30 Hz`.

---

## 4. Yêu cầu phần mềm

Khuyến nghị:

- Python 3.10+
- PyCharm hoặc VS Code
- Arduino IDE 2.x

Cài thư viện Python:

```bash
pip install numpy pyserial opencv-contrib-python
```

Có thể tạo `requirements.txt`:

```text
numpy
pyserial
opencv-contrib-python
```

Sau đó:

```bash
pip install -r requirements.txt
```

> Phải dùng `opencv-contrib-python` vì chương trình sử dụng `cv2.aruco`.

---

## 5. Cấu hình Serial

Trong code hiện tại:

```python
com_port = 'COM19'
baud_rate = 115200
```

Nếu ESP32 Gateway ở COM khác, ví dụ COM8:

```python
com_port = 'COM8'
```

Baud rate trên Python và ESP32 Gateway phải giống nhau.

---

## 6. Cấu hình Camera

Trong code:

```python
cam_stream = CameraStream(0).start()
cam_stream2 = CameraStream(2).start()
```

Tức là:

```text
Camera 1 = index 0
Camera 2 = index 2
```

Nếu máy tính nhận camera khác index, hãy sửa số tương ứng.

Camera đang dùng:

```text
Resolution: 640 x 480
FOURCC: MJPG
Buffer: 1 frame
```

---

## 7. Homography Camera 1

Camera 1 sử dụng 4 điểm pixel cố định:

```python
img_points = np.array([
    (228, 288),
    (560, 290),
    (564, 121),
    (226, 124)
], dtype=np.float32)
```

Tương ứng tọa độ thực:

```python
real_points = np.array([
    (0, 0),
    (240, 0),
    (240, 120),
    (0, 120)
], dtype=np.float32)
```

Homography được tính bằng:

```python
matrix_cam1_to_real = cv2.getPerspectiveTransform(
    img_points,
    real_points
)
```

Nếu thay đổi vị trí camera, độ phân giải hoặc vùng làm việc thì nên hiệu chuẩn lại.

---

## 8. Homography Camera 2

File Homography Camera 2:

```text
calib_values2/homography_cam2_to_real.pkl
```

Khi chạy, chương trình tự nạp file nếu tồn tại.

Để hiệu chuẩn Camera 2, nhấn:

```text
E
```

Chương trình sử dụng các marker xuất hiện đồng thời trên cả Camera 1 và Camera 2. Các robot trong `ROBOT_IDS` không được dùng làm điểm calibration.

Cần tối thiểu:

```text
4 marker chung
```

Các kiểm tra hiện có:

```python
RANSAC_REPROJ_THRESHOLD_CM = 2.0
MIN_INLIER_RATIO = 0.75
MAX_HOMOGRAPHY_RMSE_CM = 2.0
MIN_SINGULAR_GAP = 1.2
```

Pipeline hiệu chuẩn gồm:

```text
common markers
    ↓
RANSAC
    ↓
inlier filtering
    ↓
reprojection error
    ↓
SVD / rank check
    ↓
Homography Camera 2
```

---

## 9. Quy ước góc robot

```text
0°   = +X
90°  = +Y
180° = -X
270° = -Y
```

Cạnh trên của ArUco được coi là hướng đầu robot.

---

## 10. Chuyển Camera 1 ↔ Camera 2

Mỗi robot có candidate riêng từ hai camera.

Khi cả hai camera cùng nhìn thấy robot, chương trình **giữ nguyên camera đang active**, không đổi qua lại từng frame.

Chỉ chuyển camera khi nguồn hiện tại mất robot đủ:

```python
ROBOT_SWITCH_CONFIRM_FRAMES = 4
```

Ngưỡng chặn bước nhảy:

```python
ROBOT_MAX_JUMP_CM = 25.0
```

Lọc vị trí:

```python
ROBOT_POSITION_EMA_ALPHA = 0.45
```

---

## 11. Quỹ đạo robot

Ngưỡng thêm điểm mới:

```python
ROBOT_PATH_MIN_STEP_CM = 0.8
```

Ngưỡng coi là bước nhảy lỗi:

```python
ROBOT_PATH_BREAK_STEP_CM = 18.0
```

Code dùng **path cache**. Đường cũ chỉ vẽ một lần, sau đó mỗi lần robot di chuyển chỉ vẽ thêm đoạn mới. Vì vậy quỹ đạo có thể dài mà không phải vẽ lại toàn bộ lịch sử mỗi frame.

---

## 12. CLEAR PATH và CLEAR TARGET

Hai chức năng này hoàn toàn khác nhau.

### CLEAR PATH

Các nút:

```text
R8 CLR PATH
R7 CLR PATH
R3 CLR PATH
```

Chức năng:

- xóa quỹ đạo robot trên map;
- reset DIST / SPEED;
- không gửi `WPCLR#`;
- không STOP robot;
- không thay đổi trạng thái START/STOP;
- không xóa danh sách target.

Phím:

```text
C
```

xóa path của cả 3 robot.

### CLEAR TARGET

Nút:

```text
CLEAR TARGET
```

Chức năng:

- xóa waypoint đang chấm của robot được chọn;
- gửi `WPCLR#` tới đúng robot;
- không xóa đường robot đã chạy.

Ví dụ Robot 8:

```text
8;WPCLR#
```

---

## 13. Chọn nhiều điểm đích

Chọn robot:

```text
R8 TARGET
R7 TARGET
R3 TARGET
```

Sau đó **click chuột trái** trên map để thêm waypoint.

Ví dụ:

```text
P1 = (100.0, 80.0)
P2 = (120.0, 60.0)
P3 = (140.0, 80.0)
```

Điểm được lưu theo thứ tự:

```text
P1 -> P2 -> P3
```

Nút:

```text
UNDO
```

xóa waypoint vừa thêm gần nhất.

---

## 14. Gửi waypoint bằng một packet

Nhấn:

```text
SEND 1 PACKET
```

Payload:

```text
WPLIST;X1;Y1;X2;Y2;...;XN;YN#
```

Ví dụ:

```text
WPLIST;100.0;80.0;120.0;60.0;140.0;80.0#
```

Python thêm ID robot ở đầu.

Robot 8:

```text
8;WPLIST;100.0;80.0;120.0;60.0;140.0;80.0#
```

Robot 7:

```text
7;WPLIST;200.0;80.0;220.0;60.0;240.0;80.0#
```

Robot 3:

```text
3;WPLIST;300.0;80.0;320.0;60.0;340.0;80.0#
```

Packet **không có trường số lượng waypoint**. Arduino Mega có thể đọc liên tục theo từng cặp:

```text
X1,Y1
X2,Y2
X3,Y3
...
```

cho tới hết packet.

---

## 15. Giới hạn waypoint packet

Code hiện tại:

```python
ESPNOW_SAFE_WAYPOINT_BYTES = 240
```

Python kiểm tra kích thước trước khi gửi.

Nếu payload vượt giới hạn cấu hình thì packet không được gửi.

Nếu cần đường rất dài, có thể cân nhắc:

- giảm số chữ số của tọa độ;
- dùng binary thay text;
- hoặc chia thành nhiều packet.

---

## 16. Packet vị trí robot

Format:

```text
ID;X;Y;ANGLE#
```

Ví dụ:

```text
8;352.4;64.7;43.2#
7;420.8;80.1;271.6#
3;500.2;40.0;90.0#
```

Python có thể nối liên tiếp:

```text
8;352.4;64.7;43.2#7;420.8;80.1;271.6#3;500.2;40.0;90.0#
```

Gateway cần tách theo dấu:

```text
#
```

---

## 17. START / STOP

START:

```text
8;START#
7;START#
3;START#
```

STOP:

```text
8;STOP#
7;STOP#
3;STOP#
```

Từng robot hoạt động độc lập.

---

## 18. Tần số truyền Serial

```python
TX_TARGET_HZ = 30.0
```

Serial chạy ở thread riêng so với vòng xử lý camera.

Do đó:

```text
Vision FPS != Serial TX rate
```

Nếu vision chậm một nhịp, luồng Serial vẫn có thể gửi packet tọa độ mới nhất.

---

## 19. Phím và chuột

Phím:

```text
Q           Thoát
E           Calibration Camera 2
C           Clear path cả 3 robot
+           Zoom in
-           Zoom out
0           Reset zoom
F           Fit map
Enter       Send waypoint
Backspace   Undo waypoint
X           Clear target
```

Chuột:

```text
Left click      Add waypoint
Mouse wheel     Zoom
Right drag      Pan map
```

---

## 20. Camera Preview

Dashboard có:

```text
CAM 1 VIEW
CAM 2 VIEW
```

Preview chỉ được render khi bật, giúp giảm tải giao diện khi không cần xem camera.

---

## 21. Chạy chương trình

### Bước 1 - Kết nối

Kết nối vào PC:

```text
Camera 1
Camera 2
ESP32 Gateway
```

### Bước 2 - Kiểm tra COM

Mở Device Manager và tìm cổng của ESP32 Gateway.

Sửa:

```python
com_port = 'COM19'
```

nếu cần.

### Bước 3 - Cài thư viện

```bash
pip install numpy pyserial opencv-contrib-python
```

### Bước 4 - Chạy Python

Ví dụ:

```bash
python realtime_aruco_detect.py
```

hoặc chạy trong PyCharm.

---

## 22. Quy trình test khuyến nghị

### Test Camera

Kiểm tra cả Camera 1 và Camera 2 có hoạt động.

### Test ArUco

Đặt lần lượt:

```text
ID8
ID7
ID3
```

và kiểm tra dashboard nhận đúng.

### Test tọa độ

Di chuyển robot và kiểm tra:

```text
X
Y
Angle
```

### Test overlap

Cho robot đi:

```text
Camera 1
   ↓
Overlap
   ↓
Camera 2
```

Kiểm tra tọa độ không nhảy bất thường.

### Test Serial

Nhấn START từng robot và kiểm tra Gateway nhận:

```text
ID;X;Y;ANGLE#
```

### Test Waypoint

Chọn robot, click vài điểm rồi:

```text
SEND 1 PACKET
```

Kiểm tra Receiver / Mega nhận `WPLIST`.

### Test CLEAR

Kiểm tra riêng:

```text
CLR PATH
```

và:

```text
CLEAR TARGET
```

Hai nút không được ảnh hưởng lẫn nhau.

---

## 23. Cấu trúc repository GitHub khuyến nghị

```text
robot-tracking-project/
│
├── python/
│   └── realtime_aruco_detect.py
│
├── esp32_gateway/
│   └── ESP32_gateway.ino
│
├── esp32_receiver/
│   └── ESP32_receiver.ino
│
├── arduino_mega/
│   └── ArduinoMega.ino
│
├── calib_values2/
│   └── homography_cam2_to_real.pkl
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 24. `.gitignore` khuyến nghị

```gitignore
__pycache__/
*.pyc
.idea/
.vscode/
venv/
.venv/

# Nếu muốn người dùng tự calibration:
calib_values2/*.pkl
```

---

## 25. Troubleshooting

### Không có `cv2.aruco`

Cài:

```bash
pip uninstall opencv-python opencv-contrib-python -y
pip install opencv-contrib-python
```

### Không kết nối được ESP32

Kiểm tra:

- COM có đúng không;
- ESP32 đã kết nối USB chưa;
- Arduino Serial Monitor có đang giữ COM không;
- baud rate có phải `115200` không.

### Không mở được camera

Thử các index:

```python
CameraStream(0)
CameraStream(1)
CameraStream(2)
CameraStream(3)
```

### Camera 2 cho tọa độ sai

Nhấn:

```text
E
```

để calibration lại và đảm bảo có ít nhất 4 marker chung phân bố đủ rộng.

### Waypoint không gửi

Kiểm tra số waypoint và kích thước payload.

Giới hạn hiện tại:

```python
ESPNOW_SAFE_WAYPOINT_BYTES = 240
```

---

## 26. Tóm tắt giao thức

| Chức năng | Ví dụ |
|---|---|
| START Robot 8 | `8;START#` |
| STOP Robot 8 | `8;STOP#` |
| Pose Robot 8 | `8;352.4;64.7;43.2#` |
| Waypoint Robot 8 | `8;WPLIST;100.0;80.0;120.0;60.0#` |
| Clear Target Robot 8 | `8;WPCLR#` |

Robot 7 và Robot 3 sử dụng cùng format, chỉ thay ID đầu packet.

---

## 27. Công nghệ

- Python
- OpenCV
- OpenCV ArUco
- NumPy
- PySerial
- Homography
- RANSAC
- SVD
- ESP32
- ESP-NOW
- UART
- Arduino Mega

---

## 28. License

Nếu muốn người khác được phép sử dụng, chỉnh sửa và phân phối code, có thể thêm:

```text
MIT License
```

Nếu đồ án chưa hoàn thành hoặc chưa muốn công khai, có thể để repository ở chế độ **Private** trước.

---

## 29. Author

Điền thông tin nhóm tại đây:

```text
Author:
University:
Project:
Contact:
```
