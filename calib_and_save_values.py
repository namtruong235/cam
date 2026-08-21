import cv2
import numpy as np
import glob
import pickle
import os

# ================= CÀI ĐẶT THÔNG SỐ BÀN CỜ =================
CHECKERBOARD = (8, 6)  # 9x7 ô vuông -> 8x6 điểm giao cắt bên trong
SQUARE_SIZE_CM = 2.88   # Kích thước thật của 1 ô vuông (cm)
IMAGE_FOLDER = 'calib_cam1'      # Thư mục chứa ảnh calib của Cam 1
RESULT_FOLDER = 'calib_values'   # Thư mục lưu kết quả calib Cam 1

# ================= CÀI ĐẶT HIỂN THỊ ========================
SHOW_PREVIEW = True            # Hiện từng ảnh calib trong lúc chạy
PREVIEW_DELAY_MS = 700         # Thời gian hiện mỗi ảnh, không lưu ảnh preview
WINDOW_NAME = 'Calibration Preview - Cam1'
# ===========================================================


def draw_chessboard_overlay(image, corners, pattern_size):
    """Vẽ các điểm góc và các đường nối dạng lưới giống minh họa."""
    vis = image.copy()
    pts = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    cols, rows = pattern_size  # ví dụ (8,6)

    # Màu cho từng hàng (BGR)
    row_colors = [
        (0, 0, 255),      # đỏ
        (0, 128, 255),    # cam
        (0, 255, 255),    # vàng
        (0, 200, 0),      # xanh lá
        (255, 255, 0),    # cyan
        (255, 0, 0),      # xanh dương
        (255, 0, 255),    # tím
        (180, 180, 180),  # xám
    ]

    # Vẽ các đường ngang theo từng hàng
    for r in range(rows):
        color = row_colors[r % len(row_colors)]
        for c in range(cols - 1):
            p1 = tuple(np.round(pts[r * cols + c]).astype(int))
            p2 = tuple(np.round(pts[r * cols + c + 1]).astype(int))
            cv2.line(vis, p1, p2, color, 2, cv2.LINE_AA)

    # Vẽ các đường dọc mảnh hơn để nhìn rõ cấu trúc lưới
    for c in range(cols):
        for r in range(rows - 1):
            p1 = tuple(np.round(pts[r * cols + c]).astype(int))
            p2 = tuple(np.round(pts[(r + 1) * cols + c]).astype(int))
            cv2.line(vis, p1, p2, (80, 220, 220), 1, cv2.LINE_AA)

    # Vẽ điểm góc
    for idx, p in enumerate(pts):
        center = tuple(np.round(p).astype(int))
        cv2.circle(vis, center, 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(vis, center, 2, (0, 0, 0), -1, cv2.LINE_AA)

    return vis


def show_preview(window_name, image, delay_ms=700):
    cv2.imshow(window_name, image)
    key = cv2.waitKey(delay_ms) & 0xFF
    return key


def run_calibration_cam1():
    os.makedirs(RESULT_FOLDER, exist_ok=True)
    # Tiêu chuẩn tinh chỉnh điểm góc sub-pixel
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Tạo tọa độ 3D lý tưởng trên mặt phẳng bàn cờ
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_CM

    objpoints = []
    imgpoints = []

    images = sorted(glob.glob(f'{IMAGE_FOLDER}/*.png'))
    if len(images) == 0:
        print(f"❌ LỖI: Không tìm thấy ảnh nào trong thư mục '{IMAGE_FOLDER}'")
        return

    print(f"Đang quét {len(images)} bức ảnh Cam 1...\n")
    if SHOW_PREVIEW:
        print("Mỗi ảnh sẽ hiện lên cùng lưới góc đã phát hiện.")
        print("Nhấn ESC khi cửa sổ đang mở nếu muốn bỏ qua phần preview các ảnh tiếp theo.\n")

    success_count = 0
    img_shape = None
    skip_preview = False

    for idx, fname in enumerate(images, start=1):
        img = cv2.imread(fname)
        if img is None:
            print(f"❌ {fname}: Không đọc được ảnh")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_shape is None:
            img_shape = gray.shape[::-1]

        ret, corners = cv2.findChessboardCorners(
            gray,
            CHECKERBOARD,
            cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_FAST_CHECK
            + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )

        display = img.copy()

        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            success_count += 1
            print(f"✔️ [{idx}/{len(images)}] {fname}: BẮT ĐƯỢC GÓC")

            display = draw_chessboard_overlay(display, corners2, CHECKERBOARD)
            cv2.putText(display, 'DETECTED', (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 180, 0), 2, cv2.LINE_AA)
        else:
            print(f"❌ [{idx}/{len(images)}] {fname}: Không thấy bàn cờ")
            cv2.putText(display, 'NOT DETECTED', (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 0, 255), 2, cv2.LINE_AA)

        basename = os.path.basename(fname)
        cv2.putText(display, basename, (20, display.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(display, basename, (20, display.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 1, cv2.LINE_AA)

        if SHOW_PREVIEW and not skip_preview:
            key = show_preview(WINDOW_NAME, display, PREVIEW_DELAY_MS)
            if key == 27:  # ESC
                skip_preview = True
                print("→ Đã tắt preview cho các ảnh tiếp theo.")

    if SHOW_PREVIEW:
        cv2.destroyAllWindows()

    print("\n=========================================")
    print(f"Bắt góc thành công: {success_count}/{len(images)} ảnh.")

    if success_count < 3:
        print("❌ Quá ít ảnh bắt góc thành công để calib ổn định. Hãy chụp thêm ảnh.")
        return

    print("Đang tính toán ma trận camera Cam 1...\n")

    ret, k_matrix, dist_coef, r_matrix, t_vector = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape, None, None
    )

    p_matrix, _ = cv2.getOptimalNewCameraMatrix(k_matrix, dist_coef, img_shape, 1, img_shape)

    pickle.dump(p_matrix, open(f'{RESULT_FOLDER}/p_matrix.pkl', 'wb'))
    pickle.dump(r_matrix, open(f'{RESULT_FOLDER}/r_matrix.pkl', 'wb'))
    pickle.dump(t_vector, open(f'{RESULT_FOLDER}/t_vector.pkl', 'wb'))
    pickle.dump(k_matrix, open(f'{RESULT_FOLDER}/k_matrix.pkl', 'wb'))
    pickle.dump(dist_coef, open(f'{RESULT_FOLDER}/dist_coef.pkl', 'wb'))

    print(f"✅ HOÀN TẤT! Đã lưu 5 file vào thư mục '{RESULT_FOLDER}'")
    print(f"RMS reprojection error: {ret:.6f}")
    print("=========================================")


if __name__ == '__main__':
    run_calibration_cam1()