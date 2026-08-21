import cv2
import os

# ================= CÀI ĐẶT =================
CAM_ID = 2                # ID của Cam 2
FOLDER_NAME = 'calib_cam2' # Thư mục chứa ảnh Cam 2
# ===========================================



# Mở camera
cap = cv2.VideoCapture(CAM_ID)

# ÉP CỨNG ĐỘ PHÂN GIẢI Ở 640x480 (BẮT BUỘC)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

num = 0
print("=========================================")
print(f"BẮT ĐẦU CHỤP ẢNH CALIB CHO CAMERA 2")
print(f"Lưu tại thư mục: {FOLDER_NAME}/")
print("-> Bấm phím 'S' để chụp.")
print("-> Bấm 'ESC' để thoát.")
print("=========================================")

while cap.isOpened():
    success, img = cap.read()
    if not success: continue

    cv2.imshow('Chup Anh Calib - Cam 2', img)
    k = cv2.waitKey(1)

    if k == 27: # Bấm ESC để thoát
        break
    elif k == ord('s') or k == ord('S'):
        img_path = f"{FOLDER_NAME}/img{num}.png"
        cv2.imwrite(img_path, img)
        print(f"📸 Đã lưu ảnh: {img_path}")
        num += 1

cap.release()
cv2.destroyAllWindows()