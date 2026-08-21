import cv2
import numpy as np
import glob
import torch


def cameraCalibration(folder_name: str):
    # Defining the dimensions of checkerboard
    CHECKERBOARD = (8, 6)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Creating vector to store vectors of 3D points for each checkerboard image
    obj_points = []
    # Creating vector to store vectors of 2D points for each checkerboard image
    img_points = []

    # Defining the world coordinates for 3D points
    # obj_p = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    # obj_p[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    # prev_img_shape = None

    obj_p = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    obj_p[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    size_of_chessboard_squares_cm = 2.88
    obj_p = obj_p * size_of_chessboard_squares_cm

    # Extracting path of individual image stored in a given directory
    images = glob.glob(f'{folder_name}/*.png')
    for fname in images:
        img = cv2.imread(fname)
        # cv2.imwrite(fname, cv2.resize(img, (300, 400)))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        # If desired number of corners are found in the image then ret = true
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                                 cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        """
        If desired number of corner are detected,
        we refine the pixel coordinates and display 
        them on the images of checker board
        """
        if ret == True:
            obj_points.append(obj_p)
            # refining pixel coordinates for given 2d points.
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            img_points.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)

            cv2.imshow('img', img)

            k = cv2.waitKey(50)

    ret, k_matrix, dist_coef, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

    r_matrix = cv2.Rodrigues(rvecs[0])[0]
    t_vector = tvecs[0]
    # t_vector = tvecs[0] / 10
    rt_matrix = np.column_stack((r_matrix, t_vector))
    p_matrix = np.matmul(k_matrix, rt_matrix)  # A[R|t]

    return p_matrix, r_matrix, t_vector, k_matrix, dist_coef


def convert2Dto3DPoint2(point: np.ndarray, r_matrix: np.ndarray, t_vector: np.ndarray, k_matrix: np.ndarray):
    camera_matrix_inv = np.linalg.inv(k_matrix)
    vec_1 = np.array([point[0]*t_vector[-1], point[1] * t_vector[-1], t_vector[-1]]).reshape((3, 1))
    camera_point = np.dot(camera_matrix_inv, vec_1)
    t_vector = t_vector.reshape((3, 1))
    vec_2 = camera_point - t_vector
    rotation_matrix_inv = np.linalg.inv(r_matrix)
    world_point = (np.dot(rotation_matrix_inv, vec_2) * 78).reshape(-1)
    world_point[-1] = 0.0
    return world_point


def convert2Dto3DPoint(point: np.ndarray,
                       r_matrix: np.ndarray,
                       t_vector: np.ndarray,
                       k_matrix: np.ndarray,
                       depth: float):
    camera_matrix_inv = np.linalg.inv(k_matrix)
    normalized_point = np.array([point[0] * depth, point[1] * depth, depth]).reshape((3, 1))
    camera_point = np.dot(camera_matrix_inv, normalized_point)
    camera_point -= t_vector.reshape((3, 1))
    rotation_matrix_inv = np.linalg.inv(r_matrix)
    world_point = np.dot(rotation_matrix_inv, camera_point).reshape(-1)
    return world_point


def convert_2d_to_3d(K, R, t, point_2d, depth):
    """
    Chuyển đổi tọa độ từ không gian 2D sang không gian 3D với thông tin về độ sâu.

    Args:
    - K: Ma trận nội suy 3x3 của camera (intrinsic matrix).
    - R: Ma trận quay 3x3 của camera (rotation matrix).
    - t: Vector tịnh tiến 3x1 của camera (translation vector).
    - point_2d: Tọa độ điểm trong không gian 2D dưới dạng numpy array (u, v).
    - depth: Giá trị độ sâu của điểm trong không gian 3D.

    Returns:
    - point_3d: Tọa độ điểm trong không gian 3D (X, Y, Z).
    """
    # Chuyển đổi điểm 2D thành vector đồng nhất
    point_2d_homogeneous = np.array([point_2d[0], point_2d[1], 1]).reshape(3, 1)

    # Tính toán nghịch đảo của ma trận nội suy
    K_inv = np.linalg.inv(K)

    # Tính toán điểm trong hệ tọa độ camera
    point_camera = depth * np.dot(K_inv, point_2d_homogeneous)

    # Chuyển đổi từ hệ tọa độ camera sang hệ tọa độ thế giới
    point_3d = np.dot(R.T, (point_camera - t.reshape(3, 1)))

    # Trả về kết quả (X, Y, Z) dưới dạng tuple
    return (point_3d[0, 0], point_3d[1, 0], point_3d[2, 0])


def convert3DTo2DPoint(point3d: np.ndarray, p_matrix: np.ndarray):
    point = np.append(point3d.copy() / 78, 1.0).reshape((4, 1))

    point_not_norm = np.matmul(p_matrix, point)
    result = np.array(
        [np.floor(point_not_norm[0] / point_not_norm[-1]), np.floor(point_not_norm[1] / point_not_norm[-1])])

    return result.astype(np.int16).reshape(-1)


def getBoundingBox(model, image):
    result = model.predict(image, imgsz=320, conf=0.1, verbose=False)
    for r in result:
        boxes = r.boxes
        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    return (x1, y1), (x2, y2)


def calculateAlpha(x_c, y_c, x_a, y_a, x_p, y_p):
    n = x_a.shape[0]
    gamma = torch.sqrt(torch.pow((x_p - x_a), 2) + torch.pow((y_p - y_a), 2))

    alpha = torch.sqrt(torch.pow(x_c - x_a, 2) + torch.pow(y_c - y_a, 2)) / (n * gamma)
    alpha = torch.sum(alpha)

    return alpha


def calculateCostFunction(x_c, y_c, x_a, y_a, x_p, y_p):
    n = x_a.shape[0]
    gamma = torch.sqrt(torch.pow((x_p - x_a), 2) + torch.pow((y_p - y_a), 2))
    alpha = torch.sqrt(torch.pow(x_c - x_a, 2) + torch.pow(y_c - y_a, 2)) / (n * gamma)
    alpha = torch.sum(alpha)

    cost = torch.pow((alpha - torch.sqrt(torch.pow(x_c - x_a, 2)) / gamma), 2)
    cost = (1 / n) * torch.sum(cost)

    return cost
