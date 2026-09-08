# 4-Robot ArUco Tracking & Serial Control

Hệ thống này dùng 2 camera USB để theo dõi 4 robot gắn marker ArUco, tính vị trí `X, Y` theo cm và góc hướng theo radian, hiển thị trên dashboard OpenCV, rồi gửi dữ liệu tới ESP32 Gateway qua Serial. ESP32 Gateway route packet tới từng robot bằng ESP-NOW.

## 1. Trạng thái hiện tại

Code hiện tại đang theo dõi 4 robot:

| Robot | ArUco ID | Ghi chú |
|---|---:|---|
| Robot 1 | `8` | Có START / STOP / CLEAR PATH / TARGET |
| Robot 2 | `7` | Có START / STOP / CLEAR PATH / TARGET |
| Robot 3 | `3` | Có START / STOP / CLEAR PATH / TARGET |
| Robot 4 | `29` | Có START / STOP / CLEAR PATH / TARGET |

Thông số chính trong `realtime_aruco_detect.py`:

| Thông số | Giá trị hiện tại |
|---|---|
| ArUco dictionary | `DICT_4X4_100` |
| Camera 1 | index `0` |
| Camera 2 | index `2` |
| Camera resolution | `640 x 480` |
| Serial port | `COM19` |
| Serial baud rate | `115200` |
| Serial transmit rate | `30 Hz` |
| Map width | `540 cm` |
| Map height | `120 cm` |
| Max waypoint mỗi robot | `50` |
| Giới hạn payload waypoint | `240 bytes` |

## 2. Kiến trúc

```text
Camera 1 ─┐
          ├──> Python + OpenCV
Camera 2 ─┘        │
                   │ Detect ArUco
                   │ Homography
                   │ X, Y, Angle(rad)
                   ▼
              USB Serial
                   │
                   ▼
             ESP32 Gateway
                   │
                ESP-NOW
          ┌────────┼────────┬────────┐
          ▼        ▼        ▼        ▼
       Robot 8  Robot 7  Robot 3  Robot 29
```

Gateway hiện chỉ làm nhiệm vụ tách `ID` đầu packet và gửi phần payload còn lại tới MAC của robot tương ứng.

Receiver ESP32 hiện đã nhận được `START#`, `STOP#`, `X;Y;ANGLE_RAD#` và in dữ liệu ra Serial. Phần thuật toán điều khiển motor cần được thêm tại comment trong file receiver.

## 3. Cấu trúc thư mục

```text
aruco_python/
├── realtime_aruco_detect.py              # Chương trình chính
├── get_image.py                          # Chụp ảnh calib Camera 1
├── get_image2.py                         # Chụp ảnh calib Camera 2
├── calib_and_save_values.py              # Calib Camera 1 bằng bàn cờ
├── calib_and_save_values2.py             # Calib Camera 2, có báo cáo lỗi
├── test.py                               # A/B test Camera 2 RAW vs UNDISTORT
├── send_data.py                          # Test gửi Serial giả lập
├── calib_cam1/                           # Ảnh bàn cờ Camera 1
├── calib_cam2/                           # Ảnh bàn cờ Camera 2
├── calib_values/                         # Kết quả calib Camera 1
├── calib_values2_new/                    # Kết quả calib Camera 2 đang dùng
└── Ardunoide/
    ├── ESP32gui/ESP32/ESP32.ino          # ESP32 Gateway
    └── ESP32nhan/ESP8266/ESP8266.ino     # ESP32 Receiver
```

Tên thư mục `ESP8266` trong đường dẫn receiver chỉ là tên thư mục cũ. File bên trong đang dùng thư viện ESP32: `WiFi.h`, `esp_now.h`, `esp_wifi.h`.

## 4. Cài đặt Python

Khuyến nghị dùng Python 3.10 trở lên.

```bash
pip install numpy pyserial opencv-contrib-python
```

Bắt buộc dùng `opencv-contrib-python` vì code dùng module:

```python
cv2.aruco
```

Nếu bị lỗi không có `cv2.aruco`, cài lại:

```bash
pip uninstall opencv-python opencv-contrib-python -y
pip install opencv-contrib-python
```

## 5. Chạy chương trình chính

Kết nối đủ phần cứng:

```text
Camera 1
Camera 2
ESP32 Gateway qua USB
```

Chạy từ đúng thư mục `aruco_python` để các đường dẫn calibration tương đối hoạt động đúng:

```bash
cd aruco_python
python realtime_aruco_detect.py
```

Nếu ESP32 Gateway không nằm ở `COM19`, sửa trong `realtime_aruco_detect.py`:

```python
com_port = 'COM19'
baud_rate = 115200
```

Nếu camera không đúng index, sửa:

```python
cam_stream = CameraStream(0).start()
cam_stream2 = CameraStream(2).start()
```

## 6. Quy ước tọa độ và góc

Tọa độ robot được gửi theo đơn vị cm:

```text
X = vị trí theo trục Ox
Y = vị trí theo trục Oy
```

Góc robot được tính từ cạnh trên của marker ArUco, coi đó là hướng đầu robot.

Góc hiện tại dùng radian trong khoảng `[-pi, pi]`:

```text
0 rad       = +X
pi / 2 rad  = +Y
pi rad      = -X
-pi / 2 rad = -Y
```

Góc được gửi với 5 số sau dấu phẩy.

Ví dụ:

```text
1.57080 rad = 90 độ
3.14159 rad = 180 độ
-1.57080 rad = -90 độ
```

## 7. Homography và calibration

Camera 1 đang dùng 4 điểm pixel cố định trong code để tạo Homography:

```python
img_points = np.array([
    (228, 288),
    (560, 290),
    (564, 121),
    (226, 124)
], dtype=np.float32)
```

Các điểm này ánh xạ sang vùng thật:

```python
real_points = np.array([
    (0, 0),
    (240, 0),
    (240, 120),
    (0, 120)
], dtype=np.float32)
```

Camera 2 dùng calibration và Homography đã lưu ở:

```text
calib_values2_new/k_matrix.pkl
calib_values2_new/dist_coef.pkl
calib_values2_new/p_matrix.pkl
calib_values2_new/homography_cam2_to_real_fullcorners.pkl
```

Khi chạy chương trình chính, nếu file Homography Camera 2 tồn tại thì code tự nạp. Nếu cần tạo lại Homography Camera 2, mở chương trình chính và nhấn:

```text
E
```

Điều kiện để calibration Camera 2 trong lúc chạy:

| Điều kiện | Giá trị |
|---|---:|
| Số frame thu | `60` |
| Mẫu hợp lệ tối thiểu mỗi marker | `40` |
| Số marker chung tối thiểu | `4` |
| Marker robot bị loại khỏi calibration | `8`, `7`, `3`, `29` |

## 8. Chụp ảnh và calib camera

Chụp ảnh bàn cờ Camera 1:

```bash
python get_image.py
```

Chụp ảnh bàn cờ Camera 2:

```bash
python get_image2.py
```

Trong cửa sổ camera:

```text
S   Chụp ảnh
ESC Thoát
```

Calib Camera 1:

```bash
python calib_and_save_values.py
```

Calib Camera 2:

```bash
python calib_and_save_values2.py
```

Camera 2 sẽ lưu thêm các file kiểm tra:

```text
calibration_report.txt
per_view_reprojection_errors.csv
rejected_images.csv
corner_coverage.png
undistort_examples/
```

## 9. Dashboard

Dashboard OpenCV hiển thị:

```text
Vị trí X, Y của 4 robot
Góc A theo radian
Camera đang được dùng cho từng robot
Quỹ đạo robot
Waypoint đã chấm
Khoảng cách đã chạy
Tốc độ
Thời gian chạy
Trạng thái ESP32 / Serial
Event log
```

Các thao tác chính:

| Phím / chuột | Chức năng |
|---|---|
| `Q` | Thoát |
| `E` | Calib Homography Camera 2 |
| `C` | Xóa path của cả 4 robot |
| `+` / `-` | Zoom map |
| `0` | Reset zoom |
| `F` | Fit map |
| `Enter` | Gửi waypoint của robot đang chọn |
| `Backspace` | Xóa waypoint mới nhất |
| `X` | Clear target của robot đang chọn |
| Click trái trên map | Thêm waypoint |
| Lăn chuột | Zoom map |
| Kéo chuột phải | Pan map |

Các nút trên dashboard:

```text
R8 START / STOP / CLR PATH
R7 START / STOP / CLR PATH
R3 START / STOP / CLR PATH
R29 START / STOP / CLR PATH

R8 TARGET / R7 TARGET / R3 TARGET / R29 TARGET
SEND 1 PACKET
UNDO
CLEAR TARGET
CAM 1 VIEW
CAM 2 VIEW
```

## 10. START, STOP và CLEAR

START gửi lệnh cho đúng robot và bật truyền tọa độ robot đó:

```text
8;START#
7;START#
3;START#
29;START#
```

STOP gửi lệnh cho đúng robot và dừng truyền tọa độ robot đó:

```text
8;STOP#
7;STOP#
3;STOP#
29;STOP#
```

`CLEAR PATH` chỉ xóa quỹ đạo trên dashboard và reset distance/speed. Nó không gửi `STOP#`, không gửi `WPCLR#`, không xóa waypoint.

`CLEAR TARGET` xóa waypoint của robot đang chọn và gửi:

```text
ID;WPCLR#
```

Ví dụ:

```text
8;WPCLR#
```

## 11. Giao thức Serial Python -> Gateway

Mỗi packet kết thúc bằng dấu `#`.

Gateway tách packet theo `#`, đọc ID trước dấu `;` đầu tiên, rồi gửi phần payload còn lại qua ESP-NOW.

### Pose packet

Format:

```text
ID;X;Y;ANGLE_RAD#
```

Trong đó:

| Trường | Ý nghĩa | Format |
|---|---|---|
| `ID` | ID robot | số nguyên |
| `X` | tọa độ X cm | 1 số sau dấu phẩy |
| `Y` | tọa độ Y cm | 1 số sau dấu phẩy |
| `ANGLE_RAD` | góc radian `[-pi, pi]` | 5 số sau dấu phẩy |

Ví dụ:

```text
8;352.4;64.7;0.75400#
7;420.8;80.1;-1.54300#
3;500.2;40.0;1.57080#
29;510.0;60.0;3.14159#
```

Một lần gửi có thể nối nhiều packet liên tiếp:

```text
8;352.4;64.7;0.75400#7;420.8;80.1;-1.54300#3;500.2;40.0;1.57080#29;510.0;60.0;3.14159#
```

### Waypoint packet

Format:

```text
ID;WPLIST;X1;Y1;X2;Y2;...;XN;YN#
```

Ví dụ:

```text
8;WPLIST;100.0;80.0;120.0;60.0;140.0;80.0#
```

Packet waypoint không có trường số lượng điểm. Bên nhận cần đọc lần lượt theo cặp `X, Y` cho tới hết packet.

Python kiểm tra kích thước payload trước khi gửi:

```python
ESPNOW_SAFE_WAYPOINT_BYTES = 240
```

## 12. ESP32 Gateway

File:

```text
Ardunoide/ESP32gui/ESP32/ESP32.ino
```

Nhiệm vụ:

```text
Đọc Serial từ Python
Tách packet bằng dấu #
Đọc ID robot
Chọn MAC tương ứng
Bỏ ID khỏi packet
Gửi payload qua ESP-NOW
```

Gateway đang cấu hình ESP-NOW channel:

```cpp
#define ESPNOW_CHANNEL 1
```

Baud rate:

```cpp
Serial.begin(115200);
```

Khi đổi board robot, cần cập nhật MAC tại:

```cpp
robot8Mac
robot7Mac
robot3Mac
robot29Mac
```

## 13. ESP32 Receiver

File:

```text
Ardunoide/ESP32nhan/ESP8266/ESP8266.ino
```

Receiver hiện xử lý:

```text
START#
STOP#
X;Y;ANGLE_RAD#
```

Khi nhận pose hợp lệ, receiver lưu:

```cpp
robotX
robotY
robotAngle
```

Sau đó in ra Serial:

```text
X = 352.4 | Y = 64.7 | ANGLE = 0.75400 rad
```

Chỗ cần thêm thuật toán điều khiển motor nằm trong `loop()`:

```cpp
// DAT THUAT TOAN DIEU KHIEN MOTOR CUA ROBOT TAI DAY.
// x, y, angle DEU LA FLOAT.
```

Lưu ý hiện tại: receiver chưa có logic xử lý `WPLIST` và `WPCLR`. Gateway đã route được các packet này, nhưng code receiver cần được bổ sung nếu robot phải chạy theo waypoint.

## 14. Luồng hợp nhất 2 camera

Mỗi robot có candidate riêng từ Camera 1 và Camera 2. Khi cả hai camera đều nhìn thấy cùng robot, code giữ nguyên camera đang active để tránh vị trí bị nhảy qua lại.

Các tham số liên quan:

```python
ROBOT_SWITCH_CONFIRM_FRAMES = 4
ROBOT_MAX_JUMP_CM = 25.0
ROBOT_POSITION_EMA_ALPHA = 0.45
ROBOT_PATH_MIN_STEP_CM = 0.8
ROBOT_PATH_BREAK_STEP_CM = 18.0
ROBOT_LOST_HIDE_FRAMES = 12
ROBOT_LOST_RESET_FRAMES = 20
```

Ý nghĩa:

| Tham số | Chức năng |
|---|---|
| `ROBOT_SWITCH_CONFIRM_FRAMES` | Camera active mất robot đủ số frame này mới chuyển sang camera kia |
| `ROBOT_MAX_JUMP_CM` | Bỏ qua phép đo nhảy quá xa |
| `ROBOT_POSITION_EMA_ALPHA` | Hệ số lọc mượt vị trí |
| `ROBOT_PATH_MIN_STEP_CM` | Robot đi quá ngưỡng này mới thêm điểm vào path |
| `ROBOT_PATH_BREAK_STEP_CM` | Nếu nhảy quá xa thì ngắt path, không nối đường |

## 15. Quy trình test đề xuất

Test camera:

```text
Mở chương trình chính
Bật CAM 1 VIEW và CAM 2 VIEW
Kiểm tra cả hai camera có hình
```

Test ArUco:

```text
Đặt marker ID 8, 7, 3, 29 vào vùng nhìn
Kiểm tra dashboard hiện đúng robot
```

Test tọa độ và góc:

```text
Di chuyển robot trên map
Kiểm tra X, Y thay đổi theo cm
Xoay robot
Kiểm tra A thay đổi theo radian [-pi, pi]
```

Test Serial:

```text
Kết nối ESP32 Gateway
Nhấn START từng robot
Kiểm tra gateway/receiver nhận packet đúng ID
```

Test overlap:

```text
Cho robot đi từ vùng Camera 1 sang vùng Camera 2
Kiểm tra tọa độ không bị nhảy lớn ở vùng giao
```

Test waypoint:

```text
Chọn R8 TARGET hoặc robot khác
Click nhiều điểm trên map
Nhấn SEND 1 PACKET
Kiểm tra gateway route WPLIST đúng robot
```

## 16. Lỗi thường gặp

Không mở được camera:

```text
Thử đổi index CameraStream(0), CameraStream(1), CameraStream(2), CameraStream(3)
Kiểm tra camera có đang bị phần mềm khác giữ không
```

Không kết nối được ESP32:

```text
Kiểm tra COM port trong Device Manager
Đóng Arduino Serial Monitor nếu đang mở cùng COM
Đảm bảo baud rate Python và Gateway đều là 115200
```

Camera 2 cho tọa độ sai:

```text
Kiểm tra các file trong calib_values2_new/
Nhấn E để calib lại Homography Camera 2
Đảm bảo có ít nhất 4 marker chung giữa 2 camera, không dùng ID robot
```

Góc sai hoặc ngược hướng:

```text
Kiểm tra marker ArUco có được dán đúng chiều trên robot không
Code coi cạnh trên của marker là đầu robot
```

Waypoint không chạy trên robot:

```text
Gateway đã gửi được WPLIST/WPCLR
Receiver hiện chưa xử lý WPLIST/WPCLR
Cần thêm parser waypoint và thuật toán điều khiển motor trong ESP32 receiver hoặc Arduino Mega
```

## 17. Các file nên chạy

| Mục đích | Lệnh |
|---|---|
| Chạy hệ thống chính | `python realtime_aruco_detect.py` |
| Chụp ảnh calib Cam 1 | `python get_image.py` |
| Chụp ảnh calib Cam 2 | `python get_image2.py` |
| Calib Cam 1 | `python calib_and_save_values.py` |
| Calib Cam 2 | `python calib_and_save_values2.py` |
| Test gửi Serial giả lập | `python send_data.py` |
| Test Cam2 raw vs undistort | `python test.py` |

## 18. Ghi chú phát triển

Một số điểm nên cải thiện nếu tiếp tục phát triển:

```text
Tách cấu hình robot/camera/serial ra file config riêng
Gom logic xử lý 4 robot thành class để giảm lặp code
Thêm parser WPLIST/WPCLR cho ESP32 receiver
Thêm thuật toán điều khiển motor
Thêm requirements.txt
Đổi tên thư mục Ardunoide và ESP8266 cho đúng nội dung ESP32
```

