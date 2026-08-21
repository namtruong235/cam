import cv2
import numpy as np
import glob
import pickle
import os

# ================= CÀI ĐẶT THÔNG SỐ BÀN CỜ =================
CHECKERBOARD = (8, 6)  # 9x7 ô vuông -> 8x6 điểm giao cắt
SQUARE_SIZE_CM = 2.88  # ĐIỀN LẠI SỐ ĐO THƯỚC KẺ GIỐNG HỆT BÊN CAM 1
IMAGE_FOLDER = 'calib_cam2'  # Lấy ảnh từ Cam 2
RESULT_FOLDER = 'calib_values2'  # XUẤT FILE VÀO ĐÚNG THƯ MỤC CỦA CAM 2


# ===========================================================

def run_calibration_cam2():
    if not os.path.exists(RESULT_FOLDER):
        os.makedirs(RESULT_FOLDER)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp = objp * SQUARE_SIZE_CM

    objpoints = []
    imgpoints = []

    images = glob.glob(f'{IMAGE_FOLDER}/*.png')
    if len(images) == 0:
        print(f"❌ LỖI: Không tìm thấy ảnh nào trong thư mục '{IMAGE_FOLDER}'")
        return

    print(f"Đang quét {len(images)} bức ảnh của Cam 2... Vui lòng đợi!\n")

    success_count = 0
    img_shape = None

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_shape is None:
            img_shape = gray.shape[::-1]

        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                                 cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            success_count += 1
            print(f"✔️ {fname}: BẮT ĐƯỢC GÓC")
        else:
            print(f"❌ {fname}: Không thấy bàn cờ")

    print(f"\n=========================================")
    print(f"Bắt góc thành công: {success_count}/{len(images)} ảnh.")
    print("Đang tính toán Siêu Ma Trận Cam 2...")

    ret, k_matrix, dist_coef, r_matrix, t_vector = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape, None, None)

    p_matrix, _ = cv2.getOptimalNewCameraMatrix(k_matrix, dist_coef, img_shape, 1, img_shape)

    # Đổ dữ liệu ra thư mục calib_values2
    pickle.dump(p_matrix, open(f'{RESULT_FOLDER}/p_matrix.pkl', "wb"))
    pickle.dump(r_matrix, open(f'{RESULT_FOLDER}/r_matrix.pkl', "wb"))
    pickle.dump(t_vector, open(f'{RESULT_FOLDER}/t_vector.pkl', "wb"))
    pickle.dump(k_matrix, open(f'{RESULT_FOLDER}/k_matrix.pkl', "wb"))
    pickle.dump(dist_coef, open(f'{RESULT_FOLDER}/dist_coef.pkl', "wb"))

    print(f"✅ HOÀN TẤT! Đã lưu 5 file .pkl vào thư mục '{RESULT_FOLDER}'")
    print(f"=========================================")


if __name__ == "__main__":
    run_calibration_cam2()