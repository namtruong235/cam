import pickle
import cv2
import numpy as np

with open('calib_values/p_matrix.pkl', 'rb') as f:
    p_matrix = pickle.load(f)

with open('calib_values/r_matrix.pkl', 'rb') as f:
    r_matrix = pickle.load(f)

with open('calib_values/t_vector.pkl', 'rb') as f:
    t_vector = pickle.load(f)

with open('calib_values/k_matrix.pkl', 'rb') as f:
    k_matrix = pickle.load(f)

with open('calib_values/dist_coef.pkl', 'rb') as f:
    dist_coef = pickle.load(f)


print(p_matrix)
print(r_matrix)
print(t_vector)
print(k_matrix)
print(dist_coef)


# with open('calib_values2/p_matrix.pkl', 'rb') as f:
#     p_matrix = pickle.load(f)
#
# with open('calib_values2/r_matrix.pkl', 'rb') as f:
#     r_matrix = pickle.load(f)
#
# with open('calib_values2/t_vector.pkl', 'rb') as f:
#     t_vector = pickle.load(f)
#
# with open('calib_values2/k_matrix.pkl', 'rb') as f:
#     k_matrix = pickle.load(f)
#
# with open('calib_values2/dist_coef.pkl', 'rb') as f:
#     dist_coef = pickle.load(f)
#
# print(p_matrix)
# print(r_matrix)
# print(t_vector)
# print(k_matrix)
# print(dist_coef)

img = cv2.imread('image_test/img8.png')


# cv2.imshow('result', preprocess(img)[0])
img = cv2.resize(img, None, fx=0.6, fy=0.6)
cv2.imshow('original', img)
imgGray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
cv2.imshow('result2', imgGray)
imgGauss = cv2.GaussianBlur(imgGray, (9, 9), 0)
cv2.imshow('result3', imgGauss)

epsilon = 1e-6
contrast_enhanced = imgGray / (imgGauss + epsilon)

# Chuẩn hóa giá trị pixel về 0-255 và chuyển về kiểu uint8
imgContrast = np.clip(contrast_enhanced * 255, 0, 255).astype(np.uint8)
cv2.imshow('result4', imgContrast)

_, imageBinary = cv2.threshold(imgContrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow('result5', imageBinary)
#
# cv2.imwrite('rgb-sample' + '.jpg', img)
# cv2.imwrite('gray-sample' + '.jpg', imgGray)
# cv2.imwrite('gauss-sample' + '.jpg', imgGauss)
# cv2.imwrite('contrast-sample' + '.jpg', imgContrast)
# cv2.imwrite('binary-sample' + '.jpg', imageBinary)

# cv2.imshow('preprocess', detect_aruco.preprocess(img)[0])

# imgContrast = cv2.equalizeHist(imgGray)
# cv2.imshow('result4', imgContrast)
#
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# clahe_image = clahe.apply(imgGray)
#
# cv2.imshow('result5', clahe_image)

# cv2.imwrite('contrast-histogram-sample' + '.jpg', imgContrast)
# cv2.imwrite('contrast-clahe-sample' + '.jpg', clahe_image)

# cv2.waitKey(0)
cv2.waitKey(0)