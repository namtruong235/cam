import cv2
import os

# ================= CÀI ĐẶT =================
CAM_ID = 0
FOLDER_NAME = 'calib_cam1'
# ===========================================
# Mở camera
cap = cv2.VideoCapture(CAM_ID)

# ÉP CỨNG ĐỘ PHÂN GIẢI (BẮT BUỘC)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

num = 0
print("=========================================")
print(f"BẮT ĐẦU CHỤP ẢNH CALIB CHO CAMERA 1")
print(f"Lưu tại thư mục: {FOLDER_NAME}/")
print("-> Bấm phím 'S' để chụp.")
print("-> Bấm 'ESC' để thoát.")
print("=========================================")

while cap.isOpened():
    success, img = cap.read()
    if not success: continue

    cv2.imshow('Chup Anh Calib - Cam 1', img)
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