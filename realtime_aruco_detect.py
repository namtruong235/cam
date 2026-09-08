# FOUR ROBOT VERSION
# Robot IDs: 8, 7, 3, 29
# Serial packet: ID;X;Y;ANGLE_RAD#
# Angle is radians in [-pi, pi].
# Example: 8;352.4;64.7;0.75400#29;510.0;60.0;3.14159#

import threading
import math
import time
import pickle
import os
import cv2
import numpy as np
import serial
from concurrent.futures import ThreadPoolExecutor
import builtins

# CONSOLE LOG
# Mặc định tắt toàn bộ console log để tránh I/O không cần thiết.
# Khi cần debug có thể đổi thành True.
ENABLE_CONSOLE_LOG = False

def _debug_print(*args, **kwargs):
    if ENABLE_CONSOLE_LOG:
        builtins.print(*args, **kwargs)


def _calib_print(*args, **kwargs):
    """Luôn in riêng kết quả calibration H2 ra Run/Console.

    Không phụ thuộc ENABLE_CONSOLE_LOG để tránh phải bật toàn bộ debug log.
    """
    builtins.print(*args, **kwargs, flush=True)


# ArUco dictionary used by the project.
arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
def _make_aruco_params():
    params = cv2.aruco.DetectorParameters()
    params.adaptiveThreshWinSizeMin = 5
    params.adaptiveThreshWinSizeMax = 23
    params.adaptiveThreshWinSizeStep = 10
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return params

# Mỗi worker dùng một bộ DetectorParameters riêng để tránh dùng chung
# object khi detect hai camera song song.
arucoParams = _make_aruco_params()
arucoParamsCam2 = _make_aruco_params()

# Camera 2 calibration mới (RMS ~0.226 px).
# p_matrix.pkl của pipeline calibration hiện tại thực chất là new camera matrix 3x3.
CAM2_K_FILE = 'calib_values2_new/k_matrix.pkl'
CAM2_D_FILE = 'calib_values2_new/dist_coef.pkl'
CAM2_NEW_K_FILE = 'calib_values2_new/p_matrix.pkl'

# H2 này thuộc HỆ PIXEL CAM2 ĐÃ UNDISTORT.
# Bản full-corner dùng file riêng để không nạp nhầm H2 center/pose cũ.
HOMOGRAPHY_CAM2_FILE = 'calib_values2_new/homography_cam2_to_real_fullcorners.pkl'
RANSAC_REPROJ_THRESHOLD_CM = 2.0
MIN_INLIER_RATIO = 0.75
MAX_HOMOGRAPHY_RMSE_CM = 2.0
MIN_SINGULAR_GAP = 1.2
# RANSAC Cam2-undistorted pixel -> Cam1 raw pixel bằng toàn bộ corner ArUco.
INTERCAM_RANSAC_THRESHOLD_PX = 3.0
MAX_INTERCAM_RMSE_PX = 3.0

# Homography Camera 2: thu nhiều frame rồi lấy median riêng cho từng corner.
H2_CALIB_COLLECT_FRAMES = 60
H2_CALIB_MIN_VALID_SAMPLES = 40
H2_CALIB_FRAME_INTERVAL_SEC = 1.0 / 30.0

# ================= HỢP NHẤT ROBOT TRONG VÙNG CHỒNG LẤN =================
# Giữ nguyên camera đang theo dõi khi cả hai camera cùng nhìn thấy robot.
# Chỉ chuyển nguồn sau khi camera hiện tại mất robot liên tiếp vài nhịp.
ROBOT_SWITCH_CONFIRM_FRAMES = 4
# Quét ArUco mỗi frame; chỉ sửa tốc độ nhận robot, không đổi thuật toán handoff.
ARUCO_DETECT_INTERVAL = 1
# Bỏ phép đo gây nhảy quá xa so với vị trí đã lọc ở nhịp trước.
ROBOT_MAX_JUMP_CM = 25.0
# Lọc mượt theo EMA: lớn hơn -> bám nhanh hơn, nhỏ hơn -> mượt hơn.
ROBOT_POSITION_EMA_ALPHA = 0.45
# Ngưỡng tạo một điểm mới trên quỹ đạo và ngưỡng ngắt đoạn khi có bước nhảy lỗi.
ROBOT_PATH_MIN_STEP_CM = 0.8
ROBOT_PATH_BREAK_STEP_CM = 18.0
# Thời gian mất robot phải dài hơn thời gian chuyển camera, tránh ẩn robot ở vùng nối.
ROBOT_LOST_HIDE_FRAMES = 12
ROBOT_LOST_RESET_FRAMES = 20


def transform_point_with_homography(point, matrix):
    """Biến đổi một điểm 2D bằng H đã tính sẵn. Trả về None nếu H không hợp lệ."""
    if matrix is None:
        return None

    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        return None

    try:
        pt = np.array([[[float(point[0]), float(point[1])]]], dtype=np.float64)
        result = cv2.perspectiveTransform(pt, matrix)
        x, y = result[0, 0]
        if not np.isfinite(x) or not np.isfinite(y):
            return None
        return float(x), float(y)
    except cv2.error:
        return None


def get_marker_center(marker_corner):
    """Tâm ArUco = giao điểm hai đường chéo, giữ độ chính xác sub-pixel."""
    pts = np.asarray(marker_corner, dtype=np.float64).reshape(4, 2)
    tl, tr, br, bl = pts

    r = br - tl
    s = bl - tr
    denominator = r[0] * s[1] - r[1] * s[0]

    if abs(denominator) < 1e-9:
        center = pts.mean(axis=0)
    else:
        q_minus_p = tr - tl
        t = (q_minus_p[0] * s[1] - q_minus_p[1] * s[0]) / denominator
        center = tl + t * r

    if not np.all(np.isfinite(center)):
        return None

    return float(center[0]), float(center[1])


def load_cam2_calibration():
    """Load K2, D2 và newK2 dùng cho hệ pixel Camera 2 đã khử méo."""
    required = (CAM2_K_FILE, CAM2_D_FILE, CAM2_NEW_K_FILE)
    missing = [path for path in required if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            'Thiếu calibration Camera 2: ' + ', '.join(missing)
        )

    with open(CAM2_K_FILE, 'rb') as f:
        K2 = np.asarray(pickle.load(f), dtype=np.float64)
    with open(CAM2_D_FILE, 'rb') as f:
        D2 = np.asarray(pickle.load(f), dtype=np.float64)
    with open(CAM2_NEW_K_FILE, 'rb') as f:
        newK2 = np.asarray(pickle.load(f), dtype=np.float64)

    if K2.shape != (3, 3) or newK2.shape != (3, 3):
        raise ValueError('K2/newK2 phải là ma trận 3x3.')
    if D2.size < 4:
        raise ValueError('D2 không hợp lệ.')
    if not (np.all(np.isfinite(K2)) and np.all(np.isfinite(D2)) and np.all(np.isfinite(newK2))):
        raise ValueError('Calibration Camera 2 chứa NaN/Inf.')

    return K2, D2, newK2


def undistort_marker_corners(marker_corner, K, D, new_K):
    """Đổi 4 corner RAW sang hệ pixel đã undistort, giữ sub-pixel."""
    try:
        pts = np.asarray(marker_corner, dtype=np.float64).reshape(4, 1, 2)
        corrected = cv2.undistortPoints(pts, K, D, P=new_K)
        corrected = corrected.reshape(1, 4, 2)
        if not np.all(np.isfinite(corrected)):
            return None
        return corrected
    except (cv2.error, ValueError):
        return None


def get_marker_world_angle(marker_corner, H):
    """
    Tính góc quay của marker trong hệ tọa độ thực, trả về radian [-pi, pi].

    Quy ước:
        0       = +X
        pi / 2  = +Y
        pi      = -X
        -pi / 2 = -Y

    Coi cạnh trên của ArUco là hướng đầu robot.
    """

    if H is None:
        return None

    try:
        # 4 góc ArUco theo thứ tự:
        # topLeft, topRight, bottomRight, bottomLeft
        pts_pixel = np.asarray(
            marker_corner,
            dtype=np.float32
        ).reshape(4, 1, 2)

        # Biến đổi toàn bộ 4 góc sang hệ tọa độ thực
        pts_world = cv2.perspectiveTransform(
            pts_pixel,
            np.asarray(H, dtype=np.float64)
        ).reshape(4, 2)

        topLeft, topRight, bottomRight, bottomLeft = pts_world

        # Trung điểm cạnh trên
        top_mid = (topLeft + topRight) / 2.0

        # Trung điểm cạnh dưới
        bottom_mid = (bottomLeft + bottomRight) / 2.0

        # Vector hướng robot:
        # từ cạnh dưới -> cạnh trên
        dx = float(top_mid[0] - bottom_mid[0])
        dy = float(top_mid[1] - bottom_mid[1])

        angle = math.atan2(dy, dx)

        # math.atan2 already returns radians in [-pi, pi].
        return angle

    except (cv2.error, ValueError):
        return None


def _normalize_angle_rad(angle):
    """Normalize an angle to [-pi, pi)."""
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def smooth_angle_rad(old_angle, new_angle, alpha=0.35):
    """
    EMA dành riêng cho góc radian.
    Xử lý đúng trường hợp pi -> -pi.
    """

    if new_angle is None:
        return old_angle

    if old_angle is None:
        return new_angle

    # Chênh lệch góc ngắn nhất trong [-pi, pi).
    delta = (
        new_angle
        - old_angle
        + math.pi
    ) % (2.0 * math.pi) - math.pi

    result = (
        old_angle
        + alpha * delta
    )

    return _normalize_angle_rad(result)


def _normalize_points_for_dlt(points):
    """Chuẩn hóa Hartley để việc kiểm tra SVD ít phụ thuộc đơn vị pixel/cm."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    center = points.mean(axis=0)
    shifted = points - center
    mean_distance = np.mean(np.linalg.norm(shifted, axis=1))
    if mean_distance < 1e-12:
        raise ValueError('Các điểm bị trùng hoặc tập trung tại một vị trí.')

    scale = np.sqrt(2.0) / mean_distance
    T = np.array([
        [scale, 0.0, -scale * center[0]],
        [0.0, scale, -scale * center[1]],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    homogeneous = np.column_stack((points, np.ones(len(points))))
    normalized = (T @ homogeneous.T).T[:, :2]
    return normalized, T


def _build_dlt_matrix(src_points, dst_points):
    src_norm, _ = _normalize_points_for_dlt(src_points)
    dst_norm, _ = _normalize_points_for_dlt(dst_points)
    rows = []
    for (u, v), (x, y) in zip(src_norm, dst_norm):
        rows.append([-u, -v, -1.0, 0.0, 0.0, 0.0, x * u, x * v, x])
        rows.append([0.0, 0.0, 0.0, -u, -v, -1.0, y * u, y * v, y])
    return np.asarray(rows, dtype=np.float64)


def _check_point_distribution(points, min_span_x, min_span_y, min_area_ratio=0.02):
    """Phát hiện điểm trùng, gần thẳng hàng hoặc tập trung trong vùng quá nhỏ."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(points) < 4:
        return False, 'Cần ít nhất 4 điểm.'

    unique_count = len(np.unique(np.round(points, decimals=4), axis=0))
    if unique_count < 4:
        return False, 'Có điểm bị trùng; cần ít nhất 4 vị trí khác nhau.'

    span_x = float(np.ptp(points[:, 0]))
    span_y = float(np.ptp(points[:, 1]))
    if span_x < min_span_x or span_y < min_span_y:
        return False, f'Điểm phân bố quá hẹp (span={span_x:.1f} x {span_y:.1f}).'

    hull = cv2.convexHull(points.astype(np.float32))
    hull_area = float(cv2.contourArea(hull))
    bbox_area = max(span_x * span_y, 1e-12)
    area_ratio = hull_area / bbox_area
    if area_ratio < min_area_ratio:
        return False, f'Các điểm gần thẳng hàng (tỷ lệ diện tích={area_ratio:.4f}).'

    return True, ''


def _reprojection_errors(matrix, src_points, dst_points):
    src = np.asarray(src_points, dtype=np.float64).reshape(-1, 1, 2)
    dst = np.asarray(dst_points, dtype=np.float64).reshape(-1, 2)
    projected = cv2.perspectiveTransform(src, matrix).reshape(-1, 2)
    return np.linalg.norm(projected - dst, axis=1)


def estimate_homography_checked(src_points, dst_points):
    """
    Ước lượng H bằng RANSAC, chỉ tinh chỉnh bằng inlier và kiểm tra suy biến.
    H ở đây ánh xạ pixel Camera 2 -> tọa độ thực (cm).
    """
    src = np.asarray(src_points, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dst_points, dtype=np.float64).reshape(-1, 2)

    if len(src) != len(dst):
        return None, None, {}, 'Hai tập điểm không cùng số lượng.'
    if len(src) < 4:
        return None, None, {}, 'Cần ít nhất 4 cặp điểm.'

    ok, reason = _check_point_distribution(src, 20.0, 20.0)
    if not ok:
        return None, None, {}, 'Điểm Camera 2 không hợp lệ: ' + reason
    ok, reason = _check_point_distribution(dst, 5.0, 5.0)
    if not ok:
        return None, None, {}, 'Điểm tọa độ thực không hợp lệ: ' + reason

    H, mask = cv2.findHomography(
        src, dst, cv2.RANSAC,
        RANSAC_REPROJ_THRESHOLD_CM,
        maxIters=5000, confidence=0.995
    )
    if H is None or mask is None:
        return None, None, {}, 'OpenCV không tìm được Homography.'

    inlier_mask = mask.ravel().astype(bool)
    inlier_count = int(inlier_mask.sum())
    required_inliers = max(4, int(np.ceil(MIN_INLIER_RATIO * len(src))))
    if inlier_count < required_inliers:
        return None, mask, {
            'inliers': inlier_count,
            'total': len(src),
            'inlier_ratio': inlier_count / len(src)
        }, f'Quá ít inlier: {inlier_count}/{len(src)}, cần ít nhất {required_inliers}.'

    src_inliers = src[inlier_mask]
    dst_inliers = dst[inlier_mask]

    ok, reason = _check_point_distribution(src_inliers, 20.0, 20.0)
    if not ok:
        return None, mask, {}, 'Tập inlier Camera 2 bị suy biến: ' + reason
    ok, reason = _check_point_distribution(dst_inliers, 5.0, 5.0)
    if not ok:
        return None, mask, {}, 'Tập inlier tọa độ thực bị suy biến: ' + reason

    # Tính lại H chỉ từ các điểm RANSAC xác nhận là inlier.
    H_refined, _ = cv2.findHomography(src_inliers, dst_inliers, 0)
    if H_refined is None or not np.all(np.isfinite(H_refined)):
        return None, mask, {}, 'Không thể tinh chỉnh Homography từ tập inlier.'

    if abs(H_refined[2, 2]) > 1e-12:
        H_refined = H_refined / H_refined[2, 2]
    else:
        H_refined = H_refined / max(np.linalg.norm(H_refined), 1e-12)

    if np.linalg.matrix_rank(H_refined) < 3:
        return None, mask, {}, 'Ma trận Homography bị suy biến (rank < 3).'

    errors = _reprojection_errors(H_refined, src_inliers, dst_inliers)
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    max_error = float(np.max(errors))
    if rmse > MAX_HOMOGRAPHY_RMSE_CM:
        return None, mask, {
            'inliers': inlier_count,
            'total': len(src),
            'rmse_cm': rmse,
            'max_error_cm': max_error
        }, f'Sai số chiếu lại quá lớn: RMSE={rmse:.2f} cm.'

    # Kiểm tra SVD của hệ DLT đã chuẩn hóa.
    A = _build_dlt_matrix(src_inliers, dst_inliers)
    _, singular_values, _ = np.linalg.svd(A, full_matrices=True)
    rank_A = int(np.linalg.matrix_rank(A))

    if len(src_inliers) == 4:
        # A có kích thước 8x9: cần rank đúng bằng 8 để null-space chỉ có 1 chiều.
        if rank_A < 8:
            return None, mask, {}, f'Dữ liệu suy biến: rank(A)={rank_A}, cần 8.'
        singular_gap = float('inf')
    else:
        if len(singular_values) < 2:
            return None, mask, {}, 'Không đủ giá trị kỳ dị để đánh giá.'
        singular_gap = float(singular_values[-2] / max(singular_values[-1], 1e-12))
        if singular_gap < MIN_SINGULAR_GAP:
            return None, mask, {
                'singular_gap': singular_gap,
                'rank_A': rank_A
            }, (f'Homography không ổn định: hai giá trị kỳ dị cuối quá gần nhau '
                f'(gap={singular_gap:.2f}).')

    metrics = {
        'inliers': inlier_count,
        'total': len(src),
        'inlier_ratio': inlier_count / len(src),
        'rmse_cm': rmse,
        'max_error_cm': max_error,
        'rank_A': rank_A,
        'singular_gap': singular_gap,
        'src_inliers': src_inliers.astype(np.float32),
        'dst_inliers': dst_inliers.astype(np.float32)
    }
    return H_refined.astype(np.float64), mask, metrics, ''


def estimate_intercamera_homography_checked(
    src_points_cam2,
    dst_points_cam1,
    matrix_cam1_to_real
):
    """
    Ước lượng phép ghép Camera 2 -> Camera 1 bằng các CORNER ArUco chung.

    src_points_cam2 : corner Camera 2 SAU UNDISTORT (pixel hệ newK2)
    dst_points_cam1 : corner tương ứng Camera 1 RAW (pixel)

    Sau khi có H21 (Cam2 -> Cam1), dựng:
        H2 = H1 @ H21
    để Camera 2 đi vào đúng cùng hệ tọa độ thực với Camera 1.
    """
    src = np.asarray(src_points_cam2, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dst_points_cam1, dtype=np.float64).reshape(-1, 2)
    H1 = np.asarray(matrix_cam1_to_real, dtype=np.float64).reshape(3, 3)

    if len(src) != len(dst):
        return None, None, {}, 'Hai tập corner không cùng số lượng.'
    if len(src) < 4:
        return None, None, {}, 'Cần ít nhất 4 cặp corner.'
    if not np.all(np.isfinite(src)) or not np.all(np.isfinite(dst)):
        return None, None, {}, 'Tập corner chứa NaN/Inf.'
    if not np.all(np.isfinite(H1)) or np.linalg.matrix_rank(H1) < 3:
        return None, None, {}, 'H1 không hợp lệ.'

    # Vì đây đều là pixel, yêu cầu cả hai tập phủ đủ một vùng 2D.
    ok, reason = _check_point_distribution(src, 20.0, 20.0)
    if not ok:
        return None, None, {}, 'Corner Camera 2 không hợp lệ: ' + reason
    ok, reason = _check_point_distribution(dst, 20.0, 20.0)
    if not ok:
        return None, None, {}, 'Corner Camera 1 không hợp lệ: ' + reason

    # H21 chỉ mô tả phép ghép giữa hai ảnh trên cùng mặt phẳng sàn.
    H21, mask = cv2.findHomography(
        src,
        dst,
        cv2.RANSAC,
        INTERCAM_RANSAC_THRESHOLD_PX,
        maxIters=5000,
        confidence=0.995
    )
    if H21 is None or mask is None:
        return None, None, {}, 'OpenCV không tìm được H Cam2->Cam1.'

    inlier_mask = mask.ravel().astype(bool)
    inlier_count = int(inlier_mask.sum())
    required_inliers = max(4, int(np.ceil(MIN_INLIER_RATIO * len(src))))
    if inlier_count < required_inliers:
        return None, mask, {
            'inliers': inlier_count,
            'total': len(src),
            'inlier_ratio': inlier_count / len(src)
        }, f'Quá ít corner inlier: {inlier_count}/{len(src)}, cần ít nhất {required_inliers}.'

    src_inliers = src[inlier_mask]
    dst_inliers = dst[inlier_mask]

    ok, reason = _check_point_distribution(src_inliers, 20.0, 20.0)
    if not ok:
        return None, mask, {}, 'Corner inlier Camera 2 bị suy biến: ' + reason
    ok, reason = _check_point_distribution(dst_inliers, 20.0, 20.0)
    if not ok:
        return None, mask, {}, 'Corner inlier Camera 1 bị suy biến: ' + reason

    # Tinh chỉnh lại bằng toàn bộ corner được RANSAC xác nhận.
    H21_refined, _ = cv2.findHomography(src_inliers, dst_inliers, 0)
    if H21_refined is None or not np.all(np.isfinite(H21_refined)):
        return None, mask, {}, 'Không thể tinh chỉnh H Cam2->Cam1.'

    if abs(H21_refined[2, 2]) > 1e-12:
        H21_refined = H21_refined / H21_refined[2, 2]
    else:
        H21_refined = H21_refined / max(np.linalg.norm(H21_refined), 1e-12)

    if np.linalg.matrix_rank(H21_refined) < 3:
        return None, mask, {}, 'H Cam2->Cam1 bị suy biến.'

    pixel_errors = _reprojection_errors(H21_refined, src_inliers, dst_inliers)
    pixel_rmse = float(np.sqrt(np.mean(pixel_errors ** 2)))
    pixel_max = float(np.max(pixel_errors))
    if pixel_rmse > MAX_INTERCAM_RMSE_PX:
        return None, mask, {
            'inliers': inlier_count,
            'total': len(src),
            'pixel_rmse_px': pixel_rmse,
            'pixel_max_px': pixel_max
        }, f'Sai số ghép Cam2->Cam1 quá lớn: RMSE={pixel_rmse:.2f} px.'

    # Kiểm tra điều kiện hình học của H21 bằng DLT chuẩn hóa.
    A = _build_dlt_matrix(src_inliers, dst_inliers)
    _, singular_values, _ = np.linalg.svd(A, full_matrices=True)
    rank_A = int(np.linalg.matrix_rank(A))
    if len(src_inliers) == 4:
        if rank_A < 8:
            return None, mask, {}, f'Dữ liệu suy biến: rank(A)={rank_A}, cần 8.'
        singular_gap = float('inf')
    else:
        if len(singular_values) < 2:
            return None, mask, {}, 'Không đủ giá trị kỳ dị để đánh giá H21.'
        singular_gap = float(singular_values[-2] / max(singular_values[-1], 1e-12))
        if singular_gap < MIN_SINGULAR_GAP:
            return None, mask, {
                'singular_gap': singular_gap,
                'rank_A': rank_A
            }, f'H Cam2->Cam1 không ổn định (singular gap={singular_gap:.2f}).'

    # Ghép với H1 đã được kiểm chứng tốt trên sân.
    H2 = H1 @ H21_refined
    if not np.all(np.isfinite(H2)):
        return None, mask, {}, 'H2 sau khi ghép H1 @ H21 chứa NaN/Inf.'
    if abs(H2[2, 2]) > 1e-12:
        H2 = H2 / H2[2, 2]
    else:
        H2 = H2 / max(np.linalg.norm(H2), 1e-12)
    if np.linalg.matrix_rank(H2) < 3:
        return None, mask, {}, 'H2 sau khi ghép bị suy biến.'

    # Đánh giá trong hệ tọa độ thật: cùng một physical corner nhìn từ hai camera
    # phải cho cùng X,Y sau khi qua H1 và H2.
    cam1_world = cv2.perspectiveTransform(
        dst_inliers.reshape(-1, 1, 2), H1
    ).reshape(-1, 2)
    cam2_world = cv2.perspectiveTransform(
        src_inliers.reshape(-1, 1, 2), H2
    ).reshape(-1, 2)
    world_errors = np.linalg.norm(cam2_world - cam1_world, axis=1)
    world_rmse = float(np.sqrt(np.mean(world_errors ** 2)))
    world_max = float(np.max(world_errors))

    if world_rmse > MAX_HOMOGRAPHY_RMSE_CM:
        return None, mask, {
            'inliers': inlier_count,
            'total': len(src),
            'pixel_rmse_px': pixel_rmse,
            'pixel_max_px': pixel_max,
            'world_rmse_cm': world_rmse,
            'world_max_cm': world_max
        }, f'Sai số đồng nhất tọa độ quá lớn: RMSE={world_rmse:.2f} cm.'

    metrics = {
        'inliers': inlier_count,
        'total': len(src),
        'inlier_ratio': inlier_count / len(src),
        'pixel_rmse_px': pixel_rmse,
        'pixel_max_px': pixel_max,
        'world_rmse_cm': world_rmse,
        'world_max_cm': world_max,
        'rank_A': rank_A,
        'singular_gap': singular_gap,
        'H_cam2_to_cam1': H21_refined.astype(np.float64),
        'src_inliers': src_inliers.astype(np.float32),
        'dst_inliers': dst_inliers.astype(np.float32)
    }
    return H2.astype(np.float64), mask, metrics, ''


# com_port = 'COM8'
# baud_rate = 9600
# Robot IDs
robot_id = 8
robot2_id = 7
robot3_id = 3
robot4_id = 29
ROBOT_IDS = {robot_id, robot2_id, robot3_id, robot4_id}

# ser = serial.Serial(com_port, baud_rate, timeout=1)
# ESP32 + DIEU KHIEN GUI

com_port = 'COM19'
baud_rate = 115200

ser = None

# Khóa Serial
serial_lock = threading.Lock()

# Công tắc START / STOP riêng cho từng robot.
# Ban đầu tất cả đều STOP.
send_robot8_enabled = threading.Event()
send_robot7_enabled = threading.Event()
send_robot3_enabled = threading.Event()
send_robot29_enabled = threading.Event()

def _send_event_for_robot(target_robot_id):
    if target_robot_id == robot_id:
        return send_robot8_enabled
    if target_robot_id == robot2_id:
        return send_robot7_enabled
    if target_robot_id == robot3_id:
        return send_robot3_enabled
    if target_robot_id == robot4_id:
        return send_robot29_enabled
    raise ValueError(f'Robot ID không hỗ trợ: {target_robot_id}')


# Timer RUN METRICS riêng cho từng robot.
# accumulated: tổng thời gian đã chạy trước đó
# started_at : mốc thời gian khi robot được START gần nhất, None nếu đang STOP
robot_run_timers = {
    robot_id: {
        'accumulated': 0.0,
        'started_at': None,
    },
    robot2_id: {
        'accumulated': 0.0,
        'started_at': None,
    },
    robot3_id: {
        'accumulated': 0.0,
        'started_at': None,
    },
    robot4_id: {
        'accumulated': 0.0,
        'started_at': None,
    },
}


def get_robot_run_seconds(target_robot_id):
    timer_info = robot_run_timers[target_robot_id]
    elapsed = timer_info['accumulated']

    if timer_info['started_at'] is not None:
        elapsed += max(
            0.0,
            time.perf_counter() - timer_info['started_at']
        )

    return elapsed


def reset_robot_run_timer(target_robot_id):
    timer_info = robot_run_timers[target_robot_id]
    timer_info['accumulated'] = 0.0

    if _send_event_for_robot(target_robot_id).is_set():
        timer_info['started_at'] = time.perf_counter()
    else:
        timer_info['started_at'] = None

# Tọa độ cho GUI - lưu riêng cho 4 robot
latest_x = None
latest_y = None
latest_x2 = None
latest_y2 = None
latest_x3 = None
latest_y3 = None
latest_x4 = None
latest_y4 = None
latest_xy_lock = threading.Lock()


# Serial gửi fixed-rate, độc lập với FPS camera.
TX_TARGET_HZ = 30.0
TX_PERIOD_SEC = 1.0 / TX_TARGET_HZ

latest_packet_lock = threading.Lock()
latest_robot_packets = {
    robot_id: None,
    robot2_id: None,
    robot3_id: None,
    robot4_id: None,
}


def _set_latest_robot_packet(target_robot_id, packet):
    with latest_packet_lock:
        latest_robot_packets[target_robot_id] = packet


def connect_serial():
    global ser

    with serial_lock:
        if ser is not None and ser.is_open:
            return True

        try:
            ser = serial.Serial(
                com_port,
                baud_rate,
                timeout=0.1,
                write_timeout=0.1
            )

            _debug_print(f">>> DA KET NOI ESP32 TAI {com_port}")

            time.sleep(2)

            return True

        except serial.SerialException:
            ser = None

            _debug_print(f">>> CHUA KET NOI ESP32 TAI {com_port}")

            return False


connect_serial()


def handle_camera():
    global latest_x, latest_y, latest_x2, latest_y2, latest_x3, latest_y3, latest_x4, latest_y4

    # ================== TRẠM BƠM ẢNH ĐA LUỒNG ==================
    class CameraStream:
        def __init__(self, src=0):
            self.stream = cv2.VideoCapture(src)
            self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            # Chỉ giữ frame mới nhất để giảm độ trễ và backlog USB.
            self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            (self.grabbed, self.frame) = self.stream.read()
            self.stopped = False

        def start(self):
            # Khởi động một luồng chạy ngầm song song với chương trình chính
            threading.Thread(target=self.update, args=(), daemon=True).start()
            return self

        def update(self):
            # Liên tục hút ảnh từ USB vào RAM. Một frame lỗi tạm thời không
            # được phép làm thread camera dừng vĩnh viễn.
            while not self.stopped:
                grabbed, frame = self.stream.read()

                if grabbed and frame is not None and frame.size > 0:
                    self.grabbed = True
                    self.frame = frame
                else:
                    self.grabbed = False
                    time.sleep(0.01)

        def read(self):
            if not self.grabbed or self.frame is None or self.frame.size == 0:
                return False, None
            return True, self.frame.copy()

        def stop(self):
            self.stopped = True
            if self.stream.isOpened():
                self.stream.release()

    # Camera 2: load calibration trước khi tracking.
    K2, D2, newK2 = load_cam2_calibration()

    cam_stream = CameraStream(0).start()
    cam_stream2 = CameraStream(2).start()
    time.sleep(1)

    # ================= KHỞI TẠO HỆ TRỤC TỌA ĐỘ GỐC =================
    CAM1_WIDTH = 240  #
    MAP_WIDTH = 540  # Tổng chiều rộng sân hiển thị 0-540 cm
    MAP_HEIGHT = 120  # Trục Oy dài 120 cm

    # H Camera 2 UNDISTORTED pixel -> hệ tọa độ thực (cm).
    # H2 = H1 @ H(Cam2-undist -> Cam1-raw), fit từ median 4 corner của các ArUco chung.
    # Nếu chưa có file này, bấm E để thu 60 frame và tạo H2 full-corner.
    matrix_cam2_to_real = None
    # TỌA ĐỘ PIXEL TRÊN MÀN HÌNH (Lấy chuẩn từ ảnh)
    img_points = np.array([
        (228, 288),  # Dưới-Trái -> GỐC TỌA ĐỘ O (0,0)
        (560, 290),  # Dưới-Phải -> TRỤC Ox
        (564, 121),  # Trên-Phải -> GÓC CHÉO XA
        (226, 124)  # Trên-Trái -> TRỤC Oy
    ], dtype=np.float32)

    # TỌA ĐỘ THỰC TẾ TRÊN RADAR (Khớp 100% với mảng trên)
    real_points = np.array([
        (0, 0),
        (CAM1_WIDTH, 0),
        (CAM1_WIDTH, MAP_HEIGHT),
        (0, MAP_HEIGHT)
    ], dtype=np.float32)

    # Camera 1 dùng 4 điểm cố định nên chỉ tính H đúng một lần.
    matrix_cam1_to_real = cv2.getPerspectiveTransform(img_points, real_points)

    # Tự động nạp Homography Camera 2 đã hiệu chỉnh ở lần chạy trước.
    if os.path.exists(HOMOGRAPHY_CAM2_FILE):
        try:
            with open(HOMOGRAPHY_CAM2_FILE, 'rb') as f:
                loaded_H = pickle.load(f)
            loaded_H = np.asarray(loaded_H, dtype=np.float64)
            if loaded_H.shape == (3, 3) and np.all(np.isfinite(loaded_H)) and np.linalg.matrix_rank(loaded_H) == 3:
                matrix_cam2_to_real = loaded_H
                _debug_print(f">>> Đã nạp Homography Camera 2 từ '{HOMOGRAPHY_CAM2_FILE}'")
            else:
                _debug_print('>>> Bỏ qua file Homography cũ vì ma trận không hợp lệ.')
        except (OSError, pickle.PickleError, ValueError, TypeError) as exc:
            _debug_print(f'>>> Không thể nạp Homography cũ: {exc}')


    positions_map = {}
    positions_map2 = {}
    real_map1 = {}
    real_angle_map = {}
    # --- LƯU LỊCH SỬ VÀ TRẠNG THÁI HỢP NHẤT ROBOT ---
    # robot_path có thể chứa None để đánh dấu ngắt đoạn, tránh vẽ đường nối qua một bước nhảy lỗi.
    robot_path = []
    # Tổng quãng đường robot đã đi theo quỹ đạo hợp lệ trên hệ tọa độ thực (cm).
    # Chỉ cộng các đoạn được chấp nhận vào robot_path; không cộng qua bước nhảy lỗi / đoạn bị ngắt.
    robot_total_distance_cm = 0.0
    # Metrics phụ trợ cho dashboard: vận tốc ước lượng từ các bước quỹ đạo hợp lệ.
    robot_speed_cm_s = 0.0
    robot_last_motion_time = None
    session_started_at = time.perf_counter()
    robot_active_camera = None  # 'cam1' hoặc 'cam2'
    robot_active_missing_frames = 0
    robot_filtered_position = None
    robot_filtered_angle = None
    robot_source_label = 'NONE'
    robot_lost_frames = 0
    # Bù dịch riêng cho từng camera tại thời điểm bàn giao.
    # Offset của camera mới được neo vào vị trí cuối của camera cũ để quỹ đạo không bị ngắt/nhảy.
    robot_camera_offsets = {
        'cam1': (0.0, 0.0),
        'cam2': (0.0, 0.0),
    }

    # ================= ROBOT 2 (ID 7) =================
    # Robot 2 tận dụng real_map1/real_angle_map đã hợp nhất Cam1/Cam2 cho các ID thường,
    # sau đó có bộ lọc, quỹ đạo, góc và packet điều khiển riêng.
    robot2_filtered_position = None
    robot2_filtered_angle = None
    robot2_lost_frames = 0
    robot2_source_label = 'NONE'

    # Robot ID 7 dùng cùng cơ chế khóa camera / handoff như Robot ID 8.
    # Khi cả hai camera cùng nhìn thấy, nguồn hiện tại được GIỮ NGUYÊN.
    # Chỉ đổi camera khi nguồn hiện tại mất robot đủ số frame xác nhận.
    robot2_active_camera = None
    robot2_active_missing_frames = 0
    robot2_camera_offsets = {
        'cam1': (0.0, 0.0),
        'cam2': (0.0, 0.0),
    }

    robot2_path = []
    robot2_total_distance_cm = 0.0
    robot2_speed_cm_s = 0.0
    robot2_last_motion_time = None

    # ================= ROBOT 3 (ID 3) =================
    robot3_filtered_position = None
    robot3_filtered_angle = None
    robot3_lost_frames = 0
    robot3_source_label = 'NONE'

    robot3_active_camera = None
    robot3_active_missing_frames = 0
    robot3_camera_offsets = {
        'cam1': (0.0, 0.0),
        'cam2': (0.0, 0.0),
    }

    robot3_path = []
    robot3_total_distance_cm = 0.0
    robot3_speed_cm_s = 0.0
    robot3_last_motion_time = None

    # ================= ROBOT 4 (ID 29) =================
    robot4_filtered_position = None
    robot4_filtered_angle = None
    robot4_lost_frames = 0
    robot4_source_label = 'NONE'

    robot4_active_camera = None
    robot4_active_missing_frames = 0
    robot4_camera_offsets = {
        'cam1': (0.0, 0.0),
        'cam2': (0.0, 0.0),
    }

    robot4_path = []
    robot4_total_distance_cm = 0.0
    robot4_speed_cm_s = 0.0
    robot4_last_motion_time = None

    start = time.time()
    num_frames = 0
    fps = 0

    frame_counter = 0
    last_corners = ()  # Lưu khung viền của nhịp trước
    last_ids = None  # Lưu ID của nhịp trước
    last_corners2 = ()
    last_ids2 = None

    # ================= DASHBOARD : GỌN + RÕ + NHẸ =================
    # OpenCV dùng BGR, các màu dưới đây đã khai báo đúng BGR để tránh giao diện ngả nâu.
    MAP_SCALE = 3
    MAP_MIN_X = -180
    MAP_MAX_X = MAP_WIDTH
    MAP_LEFT_MARGIN = 50
    MAP_RIGHT_MARGIN = 50
    MAP_OFFSET_X = MAP_LEFT_MARGIN + int(-MAP_MIN_X * MAP_SCALE)
    MAP_OFFSET_Y = 550
    MAP_CANVAS_WIDTH = (
        MAP_LEFT_MARGIN
        + int((MAP_MAX_X - MAP_MIN_X) * MAP_SCALE)
        + MAP_RIGHT_MARGIN
    )
    MAP_CANVAS_HEIGHT = 720

    DASHBOARD_WINDOW = '4 Robot Tracking Dashboard PRO'
    HEADER_H = 82
    UI_PAD = 14
    MAP_VIEW_WIDTH = 1080
    MAP_VIEW_HEIGHT = MAP_CANVAS_HEIGHT
    SIDE_PANEL_WIDTH = 390
    SLIDER_H = 42
    DASHBOARD_WIDTH = UI_PAD + MAP_VIEW_WIDTH + UI_PAD + SIDE_PANEL_WIDTH + UI_PAD
    DASHBOARD_HEIGHT = HEADER_H + UI_PAD + MAP_VIEW_HEIGHT + SLIDER_H + UI_PAD
    BACKGROUND_MAX_SCROLL_X = max(0, MAP_CANVAS_WIDTH - MAP_VIEW_WIDTH)

    MAP_X = UI_PAD
    MAP_Y = HEADER_H + UI_PAD
    SIDE_X = MAP_X + MAP_VIEW_WIDTH + UI_PAD
    SIDE_Y = MAP_Y

    # Bảng màu BGR hiện đại, tương phản cao nhưng vẫn dịu mắt.
    UI_BG = (38, 24, 14)
    HEADER_BG = (26, 17, 9)
    PANEL_BG = (47, 31, 19)
    CARD_BG = (62, 43, 28)
    CARD_BG_ALT = (70, 49, 32)
    CARD_EDGE = (95, 72, 52)
    TEXT_MAIN = (250, 248, 244)
    TEXT_MUTED = (184, 170, 153)
    TEXT_SOFT = (142, 130, 116)
    BLUE = (238, 126, 49)
    CYAN = (221, 181, 71)
    GREEN = (82, 181, 34)
    RED = (49, 57, 232)
    ORANGE = (44, 158, 239)
    YELLOW = (71, 204, 242)
    SLATE = (112, 95, 81)

    MAP_BG = (248, 247, 245)
    GRID_MINOR = (237, 233, 229)
    GRID_MAJOR = (220, 214, 207)
    AXIS_COLOR = (74, 64, 55)
    MAP_TEXT = (112, 103, 94)

    # Nút điều khiển riêng cho 4 robot.
    R8_START_BUTTON = (SIDE_X + 18, SIDE_Y + 258, SIDE_X + 128, SIDE_Y + 290)
    R8_STOP_BUTTON  = (SIDE_X + 136, SIDE_Y + 258, SIDE_X + 246, SIDE_Y + 290)
    R8_CLEAR_BUTTON = (SIDE_X + 254, SIDE_Y + 258, SIDE_X + 372, SIDE_Y + 290)

    R7_START_BUTTON = (SIDE_X + 18, SIDE_Y + 296, SIDE_X + 128, SIDE_Y + 328)
    R7_STOP_BUTTON  = (SIDE_X + 136, SIDE_Y + 296, SIDE_X + 246, SIDE_Y + 328)
    R7_CLEAR_BUTTON = (SIDE_X + 254, SIDE_Y + 296, SIDE_X + 372, SIDE_Y + 328)

    R3_START_BUTTON = (SIDE_X + 18, SIDE_Y + 334, SIDE_X + 128, SIDE_Y + 366)
    R3_STOP_BUTTON  = (SIDE_X + 136, SIDE_Y + 334, SIDE_X + 246, SIDE_Y + 366)
    R3_CLEAR_BUTTON = (SIDE_X + 254, SIDE_Y + 334, SIDE_X + 372, SIDE_Y + 366)

    R29_START_BUTTON = (SIDE_X + 18, SIDE_Y + 372, SIDE_X + 128, SIDE_Y + 404)
    R29_STOP_BUTTON  = (SIDE_X + 136, SIDE_Y + 372, SIDE_X + 246, SIDE_Y + 404)
    R29_CLEAR_BUTTON = (SIDE_X + 254, SIDE_Y + 372, SIDE_X + 372, SIDE_Y + 404)

    # WAYPOINT CONTROL
    TARGET_R8_BUTTON  = (SIDE_X + 18,  SIDE_Y + 658, SIDE_X + 100, SIDE_Y + 686)
    TARGET_R7_BUTTON  = (SIDE_X + 106, SIDE_Y + 658, SIDE_X + 188, SIDE_Y + 686)
    TARGET_R3_BUTTON  = (SIDE_X + 194, SIDE_Y + 658, SIDE_X + 276, SIDE_Y + 686)
    TARGET_R29_BUTTON = (SIDE_X + 282, SIDE_Y + 658, SIDE_X + 372, SIDE_Y + 686)

    TARGET_SEND_BUTTON  = (SIDE_X + 18, SIDE_Y + 692, SIDE_X + 128, SIDE_Y + 720)
    TARGET_UNDO_BUTTON  = (SIDE_X + 136, SIDE_Y + 692, SIDE_X + 246, SIDE_Y + 720)
    TARGET_CLEAR_BUTTON = (SIDE_X + 254, SIDE_Y + 692, SIDE_X + 372, SIDE_Y + 720)

    CAM1_VIEW_BUTTON = (SIDE_X + 18, SIDE_Y + 726, SIDE_X + 190, SIDE_Y + 754)
    CAM2_VIEW_BUTTON = (SIDE_X + 200, SIDE_Y + 726, SIDE_X + 372, SIDE_Y + 754)
    CAM1_WINDOW = 'Camera 1 Preview'
    CAM2_WINDOW = 'Camera 2 Preview'

    # Hiệu chuẩn Camera 2 vẫn dùng phím E.
    QUIT_BUTTON = (DASHBOARD_WIDTH - 132, 18, DASHBOARD_WIDTH - 18, 62)

    SLIDER_LEFT = MAP_X + 20
    SLIDER_RIGHT = MAP_X + MAP_VIEW_WIDTH - 20
    SLIDER_Y = MAP_Y + MAP_VIEW_HEIGHT + 18

    # MAP ZOOM / PAN
    #
    # zoom = 1.0  -> đúng kích thước cũ
    # zoom < 1.0  -> thu nhỏ / nhìn được vùng rộng hơn
    # zoom > 1.0  -> phóng to
    #
    # MAP_MIN_ZOOM được chọn để khi thu nhỏ tối đa có thể nhìn gần như
    # toàn bộ chiều ngang canvas bản đồ.
    MAP_MIN_ZOOM = max(0.45, MAP_VIEW_WIDTH / float(MAP_CANVAS_WIDTH))
    MAP_MAX_ZOOM = 2.50
    MAP_ZOOM_STEP = 1.15

    initial_left_world_x = -50
    initial_scroll_x = int(MAP_OFFSET_X + initial_left_world_x * MAP_SCALE)
    initial_scroll_x = max(0, min(initial_scroll_x, BACKGROUND_MAX_SCROLL_X))

    initial_center_x = initial_scroll_x + MAP_VIEW_WIDTH / 2.0
    initial_center_y = MAP_VIEW_HEIGHT / 2.0

    ui_state = {
        'zoom': 1.0,
        'center_x': float(initial_center_x),
        'center_y': float(initial_center_y),
        'drag_slider': False,
        'drag_map': False,
        'last_mouse_x': 0,
        'last_mouse_y': 0,
        # None / 'cam1' / 'cam2'
        'camera_view': None,

        # Khi đang thu 60 frame calibration H2, bỏ qua toàn bộ mouse action
        # để tránh click/stale callback vô tình kích hoạt CLEAR/START/STOP.
        'calibrating_h2': False,

        # Robot đang được chọn để chấm waypoint.
        'target_robot': robot_id,

        'action': None,
    }

    # Mỗi robot có một danh sách waypoint riêng.
    # Mỗi phần tử: (x_cm, y_cm)
    target_waypoints = {
        robot_id: [],
        robot2_id: [],
        robot3_id: [],
        robot4_id: [],
    }

    MAX_WAYPOINTS_PER_ROBOT = 50

    def _inside(rect, px, py):
        x1, y1, x2, y2 = rect
        return x1 <= px <= x2 and y1 <= py <= y2

    def _map_rect_contains(px, py):
        return (
            MAP_X <= px < MAP_X + MAP_VIEW_WIDTH
            and MAP_Y <= py < MAP_Y + MAP_VIEW_HEIGHT
        )

    def _view_source_size(zoom_value=None):
        if zoom_value is None:
            zoom_value = float(ui_state['zoom'])

        zoom_value = max(MAP_MIN_ZOOM, min(MAP_MAX_ZOOM, float(zoom_value)))

        src_w = MAP_VIEW_WIDTH / zoom_value
        src_h = MAP_VIEW_HEIGHT / zoom_value

        return src_w, src_h

    def _clamp_map_center():
        zoom_value = float(ui_state['zoom'])
        src_w, src_h = _view_source_size(zoom_value)

        # Theo chiều X không cho viewport đi ra ngoài canvas.
        if src_w <= MAP_CANVAS_WIDTH:
            min_cx = src_w / 2.0
            max_cx = MAP_CANVAS_WIDTH - src_w / 2.0
            ui_state['center_x'] = max(
                min_cx,
                min(max_cx, float(ui_state['center_x']))
            )
        else:
            ui_state['center_x'] = MAP_CANVAS_WIDTH / 2.0

        # Theo chiều Y:
        # - zoom <= 1: giữ giữa canvas để thu nhỏ cân đối.
        # - zoom > 1: cho phép pan dọc.
        if src_h <= MAP_CANVAS_HEIGHT:
            min_cy = src_h / 2.0
            max_cy = MAP_CANVAS_HEIGHT - src_h / 2.0
            ui_state['center_y'] = max(
                min_cy,
                min(max_cy, float(ui_state['center_y']))
            )
        else:
            ui_state['center_y'] = MAP_CANVAS_HEIGHT / 2.0

    def _slider_geometry():
        track_w = SLIDER_RIGHT - SLIDER_LEFT

        src_w, _ = _view_source_size()

        if MAP_CANVAS_WIDTH <= 0:
            return track_w, track_w, 0

        visible_ratio = min(1.0, src_w / float(MAP_CANVAS_WIDTH))
        knob_w = max(80, int(track_w * visible_ratio))
        knob_w = min(track_w, knob_w)

        travel = max(1, track_w - knob_w)

        return track_w, knob_w, travel

    def _set_scroll_from_mouse(mouse_x):
        _, knob_w, travel = _slider_geometry()

        src_w, _ = _view_source_size()

        if src_w >= MAP_CANVAS_WIDTH:
            ui_state['center_x'] = MAP_CANVAS_WIDTH / 2.0
            return

        ratio = (
            mouse_x - SLIDER_LEFT - knob_w / 2.0
        ) / float(travel)

        ratio = max(0.0, min(1.0, ratio))

        max_left = MAP_CANVAS_WIDTH - src_w
        left = ratio * max_left

        ui_state['center_x'] = left + src_w / 2.0
        _clamp_map_center()

    def _apply_zoom(new_zoom, anchor_mouse_x=None, anchor_mouse_y=None):
        old_zoom = float(ui_state['zoom'])
        new_zoom = max(
            MAP_MIN_ZOOM,
            min(MAP_MAX_ZOOM, float(new_zoom))
        )

        if abs(new_zoom - old_zoom) < 1e-9:
            return

        old_src_w, old_src_h = _view_source_size(old_zoom)

        old_left = float(ui_state['center_x']) - old_src_w / 2.0
        old_top = float(ui_state['center_y']) - old_src_h / 2.0

        # Nếu con trỏ đang trên map: giữ nguyên điểm thế giới nằm dưới con trỏ.
        if (
            anchor_mouse_x is not None
            and anchor_mouse_y is not None
            and _map_rect_contains(anchor_mouse_x, anchor_mouse_y)
        ):
            local_x = anchor_mouse_x - MAP_X
            local_y = anchor_mouse_y - MAP_Y

            base_x_under_mouse = old_left + local_x / old_zoom
            base_y_under_mouse = old_top + local_y / old_zoom

            new_src_w, new_src_h = _view_source_size(new_zoom)

            new_left = base_x_under_mouse - local_x / new_zoom
            new_top = base_y_under_mouse - local_y / new_zoom

            ui_state['center_x'] = new_left + new_src_w / 2.0
            ui_state['center_y'] = new_top + new_src_h / 2.0

        ui_state['zoom'] = new_zoom
        _clamp_map_center()

    def _reset_map_view():
        ui_state['zoom'] = 1.0
        ui_state['center_x'] = float(initial_center_x)
        ui_state['center_y'] = float(initial_center_y)
        _clamp_map_center()

    def _fit_map_width():
        ui_state['zoom'] = MAP_MIN_ZOOM
        ui_state['center_x'] = MAP_CANVAS_WIDTH / 2.0
        ui_state['center_y'] = MAP_CANVAS_HEIGHT / 2.0
        _clamp_map_center()


    def _screen_to_world(mouse_x, mouse_y):
        """Đổi một điểm click trên map dashboard thành X,Y thật theo cm."""
        if not _map_rect_contains(mouse_x, mouse_y):
            return None

        zoom_value = float(ui_state['zoom'])
        src_w, src_h = _view_source_size(zoom_value)

        source_left = float(ui_state['center_x']) - src_w / 2.0
        source_top = float(ui_state['center_y']) - src_h / 2.0

        local_x = float(mouse_x - MAP_X)
        local_y = float(mouse_y - MAP_Y)

        # Pixel trong canvas gốc trước khi zoom.
        base_x = source_left + local_x / zoom_value
        base_y = source_top + local_y / zoom_value

        world_x = (base_x - MAP_OFFSET_X) / float(MAP_SCALE)
        world_y = (MAP_OFFSET_Y - base_y) / float(MAP_SCALE)

        # Chỉ nhận điểm nằm trong vùng canvas hợp lệ.
        min_y_world = -(MAP_CANVAS_HEIGHT - MAP_OFFSET_Y) / float(MAP_SCALE)
        max_y_world = MAP_OFFSET_Y / float(MAP_SCALE)

        if not (MAP_MIN_X <= world_x <= MAP_MAX_X):
            return None

        if not (min_y_world <= world_y <= max_y_world):
            return None

        return (
            round(float(world_x), 1),
            round(float(world_y), 1)
        )


    def _world_to_map_pixel(world_x, world_y):
        px = MAP_OFFSET_X + int(round(float(world_x) * MAP_SCALE))
        py = MAP_OFFSET_Y - int(round(float(world_y) * MAP_SCALE))
        return px, py

    def _mouse_callback(event, mouse_x, mouse_y, flags, _param):
        # Calibration H2 chạy blocking khoảng 2 giây. Trong thời gian này
        # không nhận thao tác chuột để action không bị treo sang frame kế tiếp.
        if ui_state.get('calibrating_h2', False):
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            if _inside(QUIT_BUTTON, mouse_x, mouse_y):
                ui_state['action'] = 'quit'
                return
            if _inside(R8_START_BUTTON, mouse_x, mouse_y):
                ui_state['action'] = 'start_r8'
                return
            if _inside(R8_STOP_BUTTON, mouse_x, mouse_y):
                ui_state['action'] = 'stop_r8'
                return
            if _inside(R8_CLEAR_BUTTON, mouse_x, mouse_y):
                ui_state['action'] = 'clear_r8'
                return
            if _inside(R7_START_BUTTON, mouse_x, mouse_y):
                ui_state['action'] = 'start_r7'
                return
            if _inside(R7_STOP_BUTTON, mouse_x, mouse_y):
                ui_state['action'] = 'stop_r7'
                return
            if _inside(R7_CLEAR_BUTTON, mouse_x, mouse_y):
                ui_state['action'] = 'clear_r7'
                return
            if _inside(R3_START_BUTTON, mouse_x, mouse_y):
                ui_state['action'] = 'start_r3'
                return
            if _inside(R3_STOP_BUTTON, mouse_x, mouse_y):
                ui_state['action'] = 'stop_r3'
                return
            if _inside(R3_CLEAR_BUTTON, mouse_x, mouse_y):
                ui_state['action'] = 'clear_r3'
                return
            if _inside(R29_START_BUTTON, mouse_x, mouse_y):
                ui_state['action'] = 'start_r29'
                return
            if _inside(R29_STOP_BUTTON, mouse_x, mouse_y):
                ui_state['action'] = 'stop_r29'
                return
            if _inside(R29_CLEAR_BUTTON, mouse_x, mouse_y):
                ui_state['action'] = 'clear_r29'
                return

            # CHỌN ROBOT ĐỂ CHẤM WAYPOINT
            if _inside(TARGET_R8_BUTTON, mouse_x, mouse_y):
                ui_state['target_robot'] = robot_id
                return

            if _inside(TARGET_R7_BUTTON, mouse_x, mouse_y):
                ui_state['target_robot'] = robot2_id
                return

            if _inside(TARGET_R3_BUTTON, mouse_x, mouse_y):
                ui_state['target_robot'] = robot3_id
                return

            if _inside(TARGET_R29_BUTTON, mouse_x, mouse_y):
                ui_state['target_robot'] = robot4_id
                return

            if _inside(TARGET_SEND_BUTTON, mouse_x, mouse_y):
                ui_state['action'] = 'send_target_path'
                return

            if _inside(TARGET_UNDO_BUTTON, mouse_x, mouse_y):
                ui_state['action'] = 'undo_target_point'
                return

            if _inside(TARGET_CLEAR_BUTTON, mouse_x, mouse_y):
                ui_state['action'] = 'clear_target_path'
                return

            # Toggle preview. Chỉ mở tối đa một camera cùng lúc.
            if _inside(CAM1_VIEW_BUTTON, mouse_x, mouse_y):
                ui_state['camera_view'] = (
                    None if ui_state['camera_view'] == 'cam1' else 'cam1'
                )
                return

            if _inside(CAM2_VIEW_BUTTON, mouse_x, mouse_y):
                ui_state['camera_view'] = (
                    None if ui_state['camera_view'] == 'cam2' else 'cam2'
                )
                return

            # Click trái trực tiếp trên bản đồ -> thêm điểm đích.
            if _map_rect_contains(mouse_x, mouse_y):
                target_robot_id = ui_state['target_robot']
                points = target_waypoints[target_robot_id]

                if len(points) < MAX_WAYPOINTS_PER_ROBOT:
                    world_point = _screen_to_world(mouse_x, mouse_y)

                    if world_point is not None:
                        points.append(world_point)
                        ui_state['action'] = 'target_point_added'

                return

            if abs(mouse_y - SLIDER_Y) <= 15 and SLIDER_LEFT <= mouse_x <= SLIDER_RIGHT:
                ui_state['drag_slider'] = True
                _set_scroll_from_mouse(mouse_x)

        elif event == cv2.EVENT_MOUSEMOVE and ui_state['drag_slider']:
            _set_scroll_from_mouse(mouse_x)

        elif event == cv2.EVENT_LBUTTONUP:
            ui_state['drag_slider'] = False

        # Chuột phải kéo trực tiếp bản đồ theo cả X và Y.
        elif event == cv2.EVENT_RBUTTONDOWN and _map_rect_contains(mouse_x, mouse_y):
            ui_state['drag_map'] = True
            ui_state['last_mouse_x'] = mouse_x
            ui_state['last_mouse_y'] = mouse_y

        elif event == cv2.EVENT_MOUSEMOVE and ui_state['drag_map']:
            dx = mouse_x - ui_state['last_mouse_x']
            dy = mouse_y - ui_state['last_mouse_y']

            zoom_value = max(MAP_MIN_ZOOM, float(ui_state['zoom']))

            # Kéo ảnh sang phải -> viewport dịch sang trái.
            ui_state['center_x'] -= dx / zoom_value
            ui_state['center_y'] -= dy / zoom_value

            ui_state['last_mouse_x'] = mouse_x
            ui_state['last_mouse_y'] = mouse_y

            _clamp_map_center()

        elif event == cv2.EVENT_RBUTTONUP:
            ui_state['drag_map'] = False

        # Lăn chuột:
        #   lên   -> zoom in
        #   xuống -> zoom out
        elif event == cv2.EVENT_MOUSEWHEEL and _map_rect_contains(mouse_x, mouse_y):
            if flags > 0:
                _apply_zoom(
                    float(ui_state['zoom']) * MAP_ZOOM_STEP,
                    mouse_x,
                    mouse_y
                )
            elif flags < 0:
                _apply_zoom(
                    float(ui_state['zoom']) / MAP_ZOOM_STEP,
                    mouse_x,
                    mouse_y
                )

    # WINDOW_AUTOSIZE giúp OpenCV không tự kéo giãn ảnh dashboard làm chữ bị nhòe.
    cv2.namedWindow(DASHBOARD_WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(DASHBOARD_WINDOW, _mouse_callback)

    def _fit_preview(frame, width=166, height=124):
        if frame is None or frame.size == 0:
            return np.full((height, width, 3), PANEL_BG, dtype=np.uint8)
        h, w = frame.shape[:2]
        scale = min(width / float(w), height / float(h))
        nw = max(1, int(w * scale))
        nh = max(1, int(h * scale))
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.full((height, width, 3), PANEL_BG, dtype=np.uint8)
        x0 = (width - nw) // 2
        y0 = (height - nh) // 2
        canvas[y0:y0 + nh, x0:x0 + nw] = resized
        return canvas

    def _draw_round_rect(img, rect, color, radius=10, border=None):
        x1, y1, x2, y2 = rect
        radius = max(1, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1, cv2.LINE_AA)
        if border is not None:
            cv2.rectangle(img, (x1, y1), (x2, y2), border, 1, cv2.LINE_AA)


    def _draw_status_dot(img, center, color, radius=5):
        cv2.circle(img, center, radius + 2, (245, 245, 245), -1, cv2.LINE_AA)
        cv2.circle(img, center, radius, color, -1, cv2.LINE_AA)

    def _put_fit_text(img, text_value, origin, max_width, font, scale, color, thickness=1):
        text_scale = float(scale)
        for _ in range(14):
            (tw, _), _ = cv2.getTextSize(text_value, font, text_scale, thickness)
            if tw <= max_width or text_scale <= 0.28:
                break
            text_scale -= 0.025
        cv2.putText(img, text_value, origin, font, text_scale, color, thickness, cv2.LINE_AA)

    def _format_elapsed(seconds):
        seconds = max(0, int(seconds))
        mm, ss = divmod(seconds, 60)
        hh, mm = divmod(mm, 60)
        return f'{hh:02d}:{mm:02d}:{ss:02d}'

    event_log = []

    def _add_event(message, color=TEXT_MUTED):
        stamp = time.strftime('%H:%M:%S')
        event_log.append((stamp, str(message), color))
        if len(event_log) > 20:
            del event_log[:-20]

    _add_event('Dashboard ready', CYAN)
    last_logged_connected = None
    last_logged_send8_enabled = None
    last_logged_send7_enabled = None
    last_logged_send3_enabled = None
    last_logged_send29_enabled = None
    last_logged_source = None
    last_logged_robot_visible = None

    # Tao luoi map tinh mot lan. Vong lap chi copy lai, khong ve lai hang tram line moi frame.
    base_map = np.full((MAP_CANVAS_HEIGHT, MAP_CANVAS_WIDTH, 3), MAP_BG, dtype=np.uint8)
    first_grid_x = int(math.ceil(MAP_MIN_X / 5.0) * 5)
    for gx_cm in range(first_grid_x, MAP_MAX_X + 1, 5):
        px = MAP_OFFSET_X + int(gx_cm * MAP_SCALE)
        color = GRID_MAJOR if gx_cm % 20 == 0 else GRID_MINOR
        cv2.line(base_map, (px, 0), (px, MAP_CANVAS_HEIGHT - 1), color, 1, cv2.LINE_AA)

    max_y_pos = int(MAP_OFFSET_Y // MAP_SCALE)
    max_y_neg = int((MAP_CANVAS_HEIGHT - 1 - MAP_OFFSET_Y) // MAP_SCALE)
    first_grid_y = -int(math.floor(max_y_neg / 5.0) * 5)
    for gy_cm in range(first_grid_y, max_y_pos + 1, 5):
        py = MAP_OFFSET_Y - int(gy_cm * MAP_SCALE)
        color = GRID_MAJOR if gy_cm % 20 == 0 else GRID_MINOR
        cv2.line(base_map, (0, py), (MAP_CANVAS_WIDTH - 1, py), color, 1, cv2.LINE_AA)

    cv2.line(base_map, (0, MAP_OFFSET_Y), (MAP_CANVAS_WIDTH - 1, MAP_OFFSET_Y), AXIS_COLOR, 2, cv2.LINE_AA)
    cv2.line(base_map, (MAP_OFFSET_X, 0), (MAP_OFFSET_X, MAP_CANVAS_HEIGHT - 1), AXIS_COLOR, 2, cv2.LINE_AA)

    first_label_x = int(math.ceil(MAP_MIN_X / 20.0) * 20)
    for gx_cm in range(first_label_x, MAP_MAX_X + 1, 20):
        if gx_cm == 0:
            continue
        px = MAP_OFFSET_X + int(gx_cm * MAP_SCALE)
        cv2.putText(base_map, str(gx_cm), (px - 12, MAP_OFFSET_Y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, MAP_TEXT, 1, cv2.LINE_AA)

    label_y_min = -int(math.floor(max_y_neg / 20.0) * 20)
    for gy_cm in range(label_y_min, max_y_pos + 1, 20):
        if gy_cm == 0:
            continue
        py = MAP_OFFSET_Y - int(gy_cm * MAP_SCALE)
        cv2.putText(base_map, str(gy_cm), (MAP_OFFSET_X + 8, py - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, MAP_TEXT, 1, cv2.LINE_AA)

    cv2.putText(base_map, 'O', (MAP_OFFSET_X - 22, MAP_OFFSET_Y + 24),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, AXIS_COLOR, 1, cv2.LINE_AA)


    # PATH CACHE - GIỮ TOÀN BỘ ĐƯỜNG ĐI
    # Không giới hạn 140/500 điểm nữa.
    # Mỗi đoạn quỹ đạo chỉ được vẽ MỘT LẦN vào map_with_paths.
    # Vì vậy robot có thể vẽ chữ/quỹ đạo rất dài mà FPS không giảm dần
    # theo chiều dài đường đi.
    PATH_COLORS = {
        robot_id: (60, 130, 220),
        robot2_id: (220, 150, 35),
        robot3_id: (160, 70, 210),
        robot4_id: (80, 180, 180),
    }

    map_with_paths = base_map.copy()

    def _append_cached_path_segment(target_robot_id, previous_point, current_point):
        if previous_point is None or current_point is None:
            return

        cv2.line(
            map_with_paths,
            (int(previous_point[2]), int(previous_point[3])),
            (int(current_point[2]), int(current_point[3])),
            PATH_COLORS[target_robot_id],
            2,
            cv2.LINE_AA
        )

    def _draw_complete_path(canvas, path_values, color):
        # Chỉ gọi khi CLEAR để dựng lại các đường còn lại.
        # CLEAR là thao tác hiếm nên không ảnh hưởng FPS tracking.
        for idx_path in range(1, len(path_values)):
            prev_path = path_values[idx_path - 1]
            curr_path = path_values[idx_path]

            if prev_path is None or curr_path is None:
                continue

            cv2.line(
                canvas,
                (int(prev_path[2]), int(prev_path[3])),
                (int(curr_path[2]), int(curr_path[3])),
                color,
                2,
                cv2.LINE_AA
            )

    def _rebuild_cached_paths():
        rebuilt = base_map.copy()
        _draw_complete_path(rebuilt, robot_path, PATH_COLORS[robot_id])
        _draw_complete_path(rebuilt, robot2_path, PATH_COLORS[robot2_id])
        _draw_complete_path(rebuilt, robot3_path, PATH_COLORS[robot3_id])
        _draw_complete_path(rebuilt, robot4_path, PATH_COLORS[robot4_id])
        return rebuilt

    # UI / PERFORMANCE
    # Dashboard vẫn khá mượt; tracking ArUco không bị giới hạn bởi con số này.
    UI_RENDER_INTERVAL = 1.0 / 18.0

    # Chỉ có tác dụng khi người dùng mở CAM1 hoặc CAM2.
    PREVIEW_INTERVAL = 1.0 / 8.0

    last_ui_render = 0.0
    last_preview_update = 0.0
    camera_preview_cache = np.full((96, 338, 3), PANEL_BG, dtype=np.uint8)
    fps_display = 0.0

    # Detect ArUco 2 camera song song để tận dụng CPU đa nhân.
    aruco_detect_pool = ThreadPoolExecutor(
        max_workers=2,
        thread_name_prefix='aruco'
    )

    while True:
        success, img = cam_stream.read()
        success2, img2 = cam_stream2.read()
        # success2 = True
        # img2 = np.zeros((10, 10, 3), dtype=np.uint8)
        if (
            not success
            or not success2
            or img is None
            or img2 is None
            or img.size == 0
            or img2.size == 0
        ):
            continue

        # img2 = cv2.resize(img2, None, fx=0.5625, fy=0.5625)
        # ── UNDISTORT (sửa méo lens trước khi làm bất cứ thứ gì) ──
        # img2 = cv2.resize(img2, None, fx=1.5825, fy=1.5825)

        # ================= MAP CANVAS =================
        SCALE = MAP_SCALE
        _OFFSET_X = MAP_OFFSET_X
        _OFFSET_Y = MAP_OFFSET_Y
        BG_W = MAP_CANVAS_WIDTH
        BG_H = MAP_CANVAS_HEIGHT

        background = map_with_paths.copy()

        # Mỗi camera tạo một ứng viên độc lập. Chỉ sau đó mới chọn một nguồn duy nhất.
        robot_candidate_cam1 = None
        robot_candidate_cam2 = None
        robot_angle_cam1 = None
        robot_angle_cam2 = None

        # Robot ID 7 cũng phải có candidate riêng cho từng camera.
        # Không lấy trực tiếp từ real_map1 trong overlap vì nguồn Cam1/Cam2
        # có thể thay đổi theo từng frame và làm tọa độ nhảy qua lại.
        robot2_candidate_cam1 = None
        robot2_candidate_cam2 = None
        robot2_angle_cam1 = None
        robot2_angle_cam2 = None

        # Robot ID 3 cũng phải reset candidate theo từng frame.
        # Nếu không, dữ liệu cũ có thể bị giữ lại và gây nhảy / nhầm nguồn.
        robot3_candidate_cam1 = None
        robot3_candidate_cam2 = None
        robot3_angle_cam1 = None
        robot3_angle_cam2 = None

        robot4_candidate_cam1 = None
        robot4_candidate_cam2 = None
        robot4_angle_cam1 = None
        robot4_angle_cam2 = None

        angles_cam1 = {}
        angles_cam2 = {}
        if success and success2:
            num_frames += 1
            frame_counter += 1
            if frame_counter % ARUCO_DETECT_INTERVAL == 0:
                imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                imgGray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)  # Xám hóa ảnh cam 2

                # Quét ArUco hai camera SONG SONG.
                future_cam1 = aruco_detect_pool.submit(
                    cv2.aruco.detectMarkers,
                    imgGray,
                    arucoDict,
                    parameters=arucoParams
                )
                future_cam2 = aruco_detect_pool.submit(
                    cv2.aruco.detectMarkers,
                    imgGray2,
                    arucoDict,
                    parameters=arucoParamsCam2
                )

                corners, ids, rejected = future_cam1.result()
                corners2, ids2, rejected2 = future_cam2.result()

                last_corners = corners
                last_ids = ids
                last_corners2 = corners2
                last_ids2 = ids2
            else:
                # Khung hình lẻ: Cả 2 cam lấy ngay kết quả cũ ra dùng tiếp
                corners = last_corners
                ids = last_ids
                corners2 = last_corners2
                ids2 = last_ids2

            # XỬ LÝ ẢNH CAM 1
            if len(corners) > 0:
                ids = ids.flatten()
                for idx in ids:
                    if idx not in positions_map:
                        positions_map.update({idx: []})

                for (markerCorner, markerID) in zip(corners, ids):

                    c_pts = markerCorner.reshape((4, 2))
                    center = get_marker_center(markerCorner)
                    if center is None:
                        continue
                    cX, cY = center
                    center_draw = (int(round(cX)), int(round(cY)))

                    topLeft, topRight, bottomRight, bottomLeft = [
                        tuple(np.rint(p).astype(int)) for p in c_pts
                    ]

                    if ui_state['camera_view'] == 'cam1':
                        cv2.line(img, topLeft, topRight, (0, 255, 0), 2)
                        cv2.line(img, topRight, bottomRight, (0, 255, 0), 2)
                        cv2.line(img, bottomRight, bottomLeft, (0, 255, 0), 2)
                        cv2.line(img, bottomLeft, topLeft, (0, 255, 0), 2)
                    marker_angle_cam1 = get_marker_world_angle(
                        markerCorner,
                        matrix_cam1_to_real
                    )

                    marker_id_int = int(markerID)

                    if marker_angle_cam1 is not None:
                        angles_cam1[marker_id_int] = marker_angle_cam1

                    if marker_id_int in ROBOT_IDS:
                        # Robot luôn lấy tâm của frame hiện tại, không dùng lịch sử cũ.
                        positions_map[marker_id_int] = [(cX, cY)]

                        robot_now_cam1 = transform_point_with_homography(
                            (cX, cY),
                            matrix_cam1_to_real
                        )

                        if marker_id_int == robot_id:
                            robot_candidate_cam1 = robot_now_cam1
                            robot_angle_cam1 = marker_angle_cam1
                        elif marker_id_int == robot2_id:
                            robot2_candidate_cam1 = robot_now_cam1
                            robot2_angle_cam1 = marker_angle_cam1
                        elif marker_id_int == robot3_id:
                            robot3_candidate_cam1 = robot_now_cam1
                            robot3_angle_cam1 = marker_angle_cam1
                        elif marker_id_int == robot4_id:
                            robot4_candidate_cam1 = robot_now_cam1
                            robot4_angle_cam1 = marker_angle_cam1

                        if marker_angle_cam1 is not None:
                            cv2.putText(
                                img,
                                f"R{marker_id_int} A={marker_angle_cam1:.5f} rad",
                                (center_draw[0] + 10, center_draw[1] + 25),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.55,
                                (255, 0, 255),
                                2,
                                cv2.LINE_AA
                            )
                    else:
                        positions = positions_map[marker_id_int]
                        if len(positions) > 0:
                            if math.dist((cX, cY), (positions[-1][0], positions[-1][1])) > 2:
                                positions_map[marker_id_int].append((cX, cY))
                        else:
                            positions_map[marker_id_int].append((cX, cY))

                    if ui_state['camera_view'] == 'cam1':
                        cv2.circle(img, topLeft, 2, (0, 0, 255), -1)
                        cv2.circle(img, center_draw, 2, (0, 255, 255), -1)

            # 1. CẬP NHẬT TỌA ĐỘ CAM 1
            current_ids_cam1 = set(ids.tolist()) if ids is not None else set()
            for key in current_ids_cam1:
                if key not in positions_map or len(positions_map[key]) == 0:
                    continue

                last_position = positions_map[key][-1]
                real_point = transform_point_with_homography(last_position, matrix_cam1_to_real)
                if real_point is None:
                    continue

                if key == robot_id:
                    robot_candidate_cam1 = real_point
                elif key == robot2_id:
                    robot2_candidate_cam1 = real_point
                elif key == robot3_id:
                    robot3_candidate_cam1 = real_point
                elif key == robot4_id:
                    robot4_candidate_cam1 = real_point
                else:
                    real_map1[key] = real_point

            # 2. CAMERA 2: detect trên RAW, nhưng toàn bộ geometry dùng corner UNDISTORT.
            if len(corners2) > 0:
                ids2 = ids2.flatten()
                for idx in ids2:
                    if idx not in positions_map2:
                        positions_map2.update({idx: []})

                for (markerCorner, markerID) in zip(corners2, ids2):
                    markerCornerUnd = undistort_marker_corners(
                        markerCorner, K2, D2, newK2
                    )
                    if markerCornerUnd is None:
                        continue

                    # Tâm dùng cho Homography được tính sau distortion correction.
                    center2 = get_marker_center(markerCornerUnd)
                    if center2 is None:
                        continue
                    cX, cY = center2

                    # Preview vẫn là ảnh RAW, nên chỉ dùng corner/center RAW để vẽ.
                    c_pts2_raw = np.asarray(markerCorner, dtype=np.float64).reshape(4, 2)
                    raw_center2 = get_marker_center(markerCorner)
                    if raw_center2 is None:
                        raw_center2 = center2
                    center_draw2 = (int(round(raw_center2[0])), int(round(raw_center2[1])))

                    topLeft2, topRight2, bottomRight2, bottomLeft2 = [
                        tuple(np.rint(p).astype(int)) for p in c_pts2_raw
                    ]

                    if ui_state['camera_view'] == 'cam2':
                        cv2.line(img2, topLeft2, topRight2, (0, 255, 0), 2)
                        cv2.line(img2, topRight2, bottomRight2, (0, 255, 0), 2)
                        cv2.line(img2, bottomRight2, bottomLeft2, (0, 255, 0), 2)
                        cv2.line(img2, bottomLeft2, topLeft2, (0, 255, 0), 2)

                    marker_angle_cam2 = None
                    if matrix_cam2_to_real is not None:
                        marker_angle_cam2 = get_marker_world_angle(
                            markerCornerUnd,
                            matrix_cam2_to_real
                        )

                        if marker_angle_cam2 is not None:
                            angles_cam2[int(markerID)] = marker_angle_cam2

                    marker_id_int = int(markerID)

                    if marker_id_int in ROBOT_IDS:
                        positions_map2[marker_id_int] = [(cX, cY)]

                        robot_now_cam2 = None
                        if matrix_cam2_to_real is not None:
                            robot_now_cam2 = transform_point_with_homography(
                                (cX, cY),
                                matrix_cam2_to_real
                            )

                        if marker_id_int == robot_id:
                            robot_candidate_cam2 = robot_now_cam2
                            robot_angle_cam2 = marker_angle_cam2
                        elif marker_id_int == robot2_id:
                            robot2_candidate_cam2 = robot_now_cam2
                            robot2_angle_cam2 = marker_angle_cam2
                        elif marker_id_int == robot3_id:
                            robot3_candidate_cam2 = robot_now_cam2
                            robot3_angle_cam2 = marker_angle_cam2
                        elif marker_id_int == robot4_id:
                            robot4_candidate_cam2 = robot_now_cam2
                            robot4_angle_cam2 = marker_angle_cam2

                        if marker_angle_cam2 is not None:
                            cv2.putText(
                                img2,
                                f"R{marker_id_int} A={marker_angle_cam2:.5f} rad",
                                (center_draw2[0] + 10, center_draw2[1] + 25),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.55,
                                (255, 0, 255),
                                2,
                                cv2.LINE_AA
                            )
                    else:
                        positions2 = positions_map2[marker_id_int]
                        if len(positions2) > 0:
                            if math.dist((cX, cY), (positions2[-1][0], positions2[-1][1])) > 2:
                                positions_map2[marker_id_int].append((cX, cY))
                        else:
                            positions_map2[marker_id_int].append((cX, cY))

                    if ui_state['camera_view'] == 'cam2':
                        cv2.circle(img2, topLeft2, 2, (0, 0, 255), -1)
                        cv2.circle(img2, center_draw2, 2, (0, 255, 255), -1)

            current_ids_cam2 = set(ids2.tolist()) if ids2 is not None else set()
            if matrix_cam2_to_real is not None:
                for key in current_ids_cam2:
                    if key not in positions_map2 or len(positions_map2[key]) == 0:
                        continue

                    px2, py2 = positions_map2[key][-1]
                    real_pt = transform_point_with_homography(
                        (px2, py2), matrix_cam2_to_real
                    )
                    if real_pt is None:
                        continue

                    if key == robot_id:
                        robot_candidate_cam2 = real_pt
                    elif key == robot2_id:
                        robot2_candidate_cam2 = real_pt
                    elif key == robot3_id:
                        robot3_candidate_cam2 = real_pt
                    elif key == robot4_id:
                        robot4_candidate_cam2 = real_pt
                    elif key not in current_ids_cam1:
                        # Chỉ các marker thường mới dùng quy tắc ưu tiên Cam1 trực tiếp.
                        real_map1[key] = real_pt
            for marker_id in current_ids_cam1:

                marker_id = int(marker_id)

                if marker_id in ROBOT_IDS:
                    continue

                if marker_id in angles_cam1:
                    real_angle_map[marker_id] = angles_cam1[marker_id]

            for marker_id in current_ids_cam2:

                marker_id = int(marker_id)

                if marker_id in ROBOT_IDS:
                    continue

                # Cam1 được ưu tiên trong vùng overlap
                if (
                        marker_id not in current_ids_cam1
                        and marker_id in angles_cam2
                ):
                    real_angle_map[marker_id] = angles_cam2[marker_id]
            # 2B. HỢP NHẤT ROBOT TRONG VÙNG CHỒNG LẤN
            candidates = {
                'cam1': robot_candidate_cam1,
                'cam2': robot_candidate_cam2,
            }

            # THEO DÕI MẤT / XUẤT HIỆN LẠI robot
            robot_seen_now = (
                    robot_candidate_cam1 is not None
                    or robot_candidate_cam2 is not None
            )
            robot_reacquired = False

            if not robot_seen_now:
                robot_lost_frames += 1

                # Chỉ ẩn sau khoảng chờ đủ dài để camera 2 có thời gian tiếp quản.
                if robot_lost_frames >= ROBOT_LOST_HIDE_FRAMES:
                    real_map1.pop(robot_id, None)
                    robot_source_label = 'NONE'

                # Chỉ reset khi chắc chắn cả hai camera đã mất robot đủ lâu.
                # Không thêm None vào robot_path ở đây vì khoảng mất có thể xảy ra
                # đúng lúc robot đi qua ranh giới hai camera.
                if robot_lost_frames >= ROBOT_LOST_RESET_FRAMES:
                    robot_active_camera = None
                    robot_active_missing_frames = 0
                    robot_filtered_position = None
                    robot_filtered_angle = None
                    robot_camera_offsets['cam1'] = (0.0, 0.0)
                    robot_camera_offsets['cam2'] = (0.0, 0.0)

            else:
                robot_was_hidden = (
                        robot_lost_frames >= ROBOT_LOST_HIDE_FRAMES
                )
                robot_lost_frames = 0

                if robot_was_hidden:
                    # Robot vừa xuất hiện lại: nhận ngay vị trí mới, không để
                    # ROBOT_MAX_JUMP_CM so với vị trí cũ rồi khóa robot.
                    if robot_candidate_cam1 is not None:
                        robot_active_camera = 'cam1'
                        first_position = robot_candidate_cam1
                    else:
                        robot_active_camera = 'cam2'
                        first_position = robot_candidate_cam2

                    # Sau một lần mất thật sự, bắt đầu lại bằng tọa độ Homography gốc.
                    robot_camera_offsets[robot_active_camera] = (0.0, 0.0)
                    robot_filtered_position = (
                        float(first_position[0]),
                        float(first_position[1]),
                    )
                    robot_active_missing_frames = 0
                    robot_source_label = robot_active_camera.upper()
                    robot_reacquired = True

                    # Chỉ ngắt quỹ đạo nếu robot được đặt lại ở vị trí thực sự xa.
                    last_path_point = next(
                        (point for point in reversed(robot_path) if point is not None),
                        None
                    )
                    if last_path_point is not None:
                        distance_from_old_path = math.dist(
                            first_position,
                            last_path_point[0:2]
                        )
                        if distance_from_old_path > ROBOT_PATH_BREAK_STEP_CM:
                            if robot_path[-1] is not None:
                                robot_path.append(None)

            # CHỌN CAMERA VÀ BÀN GIAO LIỀN MẠCH
            if robot_active_camera is None:
                if robot_candidate_cam1 is not None:
                    robot_active_camera = 'cam1'
                elif robot_candidate_cam2 is not None:
                    robot_active_camera = 'cam2'

                # Lần bắt đầu đầu tiên không cần bù nối camera.
                if robot_active_camera is not None and robot_filtered_position is None:
                    robot_camera_offsets[robot_active_camera] = (0.0, 0.0)

            def corrected_candidate(source_name):
                raw_position = candidates.get(source_name)
                if raw_position is None:
                    return None
                offset_x, offset_y = robot_camera_offsets[source_name]
                return (
                    float(raw_position[0]) + float(offset_x),
                    float(raw_position[1]) + float(offset_y),
                )

            selected_source = robot_active_camera
            selected_position = (
                corrected_candidate(selected_source)
                if selected_source is not None
                else None
            )

            if selected_source is not None and selected_position is not None:
                # Camera hiện tại vẫn thấy robot: giữ khóa nguồn.
                robot_active_missing_frames = 0

            elif selected_source is not None:
                # Camera hiện tại mất robot. Đợi đủ vài frame rồi mới chuyển.
                robot_active_missing_frames += 1
                other_source = 'cam2' if selected_source == 'cam1' else 'cam1'
                other_raw_position = candidates.get(other_source)

                if (
                        other_raw_position is not None
                        and robot_active_missing_frames >= ROBOT_SWITCH_CONFIRM_FRAMES
                ):
                    # Neo camera mới vào đúng vị trí cuối của camera cũ.
                    # Nhờ vậy lần chuyển đầu tiên có khoảng cách gần bằng 0,
                    # không bị ROBOT_MAX_JUMP_CM từ chối và không làm đứt đường.
                    if robot_filtered_position is not None:
                        robot_camera_offsets[other_source] = (
                            float(robot_filtered_position[0]) - float(other_raw_position[0]),
                            float(robot_filtered_position[1]) - float(other_raw_position[1]),
                        )
                    else:
                        robot_camera_offsets[other_source] = (0.0, 0.0)

                    robot_active_camera = other_source
                    selected_source = other_source
                    selected_position = corrected_candidate(other_source)
                    robot_active_missing_frames = 0
                else:
                    selected_position = None
            else:
                selected_position = None

            # Chặn một phép đo nhảy bất thường khi đang theo cùng một camera.
            # Frame vừa xuất hiện lại đã được khởi tạo trực tiếp nên không kiểm tra bước nhảy.
            if (
                    selected_position is not None
                    and robot_filtered_position is not None
                    and not robot_reacquired
            ):
                selected_jump = math.dist(
                    selected_position,
                    robot_filtered_position
                )

                if selected_jump > ROBOT_MAX_JUMP_CM:
                    # Không đổi camera ngay chỉ vì một frame bị jump.
                    # Coi frame này như một phép đo không hợp lệ và yêu cầu
                    # nguồn kia phải ổn định đủ ROBOT_SWITCH_CONFIRM_FRAMES.
                    selected_position = None
                    robot_active_missing_frames += 1

                    other_source = 'cam2' if selected_source == 'cam1' else 'cam1'
                    other_raw_position = candidates.get(other_source)

                    if (
                            other_raw_position is not None
                            and robot_active_missing_frames >= ROBOT_SWITCH_CONFIRM_FRAMES
                    ):
                        # Handoff có neo offset vào vị trí lọc cuối cùng,
                        # tránh nhảy giữa H1 và H2 trong vùng overlap.
                        robot_camera_offsets[other_source] = (
                            float(robot_filtered_position[0]) - float(other_raw_position[0]),
                            float(robot_filtered_position[1]) - float(other_raw_position[1]),
                        )
                        robot_active_camera = other_source
                        selected_source = other_source
                        selected_position = corrected_candidate(other_source)
                        robot_active_missing_frames = 0

            # 6. CHỌN GÓC THEO CAMERA ĐANG ĐƯỢC SỬ DỤNG

            selected_angle = None
            if selected_position is not None:
                if selected_source == 'cam1':
                    selected_angle = robot_angle_cam1
                elif selected_source == 'cam2':
                    selected_angle = robot_angle_cam2

            # 7. LỌC MƯỢT GÓC ROBOT

            if selected_angle is not None:
                robot_filtered_angle = smooth_angle_rad(
                    robot_filtered_angle,
                    selected_angle,
                    alpha=0.35
                )
            if robot_filtered_angle is not None:
                real_angle_map[robot_id] = robot_filtered_angle
            # TIẾP TỤC XỬ LÝ VỊ TRÍ NHƯ CODE CŨ

            if selected_position is not None:

                if robot_filtered_position is None:

                    robot_filtered_position = (
                        float(selected_position[0]),
                        float(selected_position[1]),
                    )

                else:

                    alpha = ROBOT_POSITION_EMA_ALPHA

                    robot_filtered_position = (
                        (1.0 - alpha) * robot_filtered_position[0]
                        + alpha * selected_position[0],

                        (1.0 - alpha) * robot_filtered_position[1]
                        + alpha * selected_position[1],
                    )

                robot_source_label = selected_source.upper()

            # Giữ dấu robot trong một khoảng ngắn khi nhận dạng chớp tắt,
            # nhưng xóa khỏi radar khi đã mất đủ lâu.
            keep_robot_visible = (
                    robot_seen_now
                    or robot_lost_frames < ROBOT_LOST_HIDE_FRAMES
            )

            if robot_filtered_position is not None and keep_robot_visible:
                fused_robot = (
                    round(float(robot_filtered_position[0]), 1),
                    round(float(robot_filtered_position[1]), 1),
                )
                real_map1[robot_id] = fused_robot
            else:
                real_map1.pop(robot_id, None)

            # 2C. HỢP NHẤT ROBOT 2 (ID 7) TRONG VÙNG CHỒNG LẤN
            # QUAN TRỌNG:
            # Không còn lấy robot2_raw_position = real_map1[robot2_id].
            # Robot 2 có 2 candidate độc lập và một active_camera riêng.
            # Khi Cam1 + Cam2 cùng thấy ID7, camera đang active được khóa lại,
            # nên tọa độ không nhảy qua lại giữa H1 và H2 theo từng frame.
            robot2_candidates = {
                'cam1': robot2_candidate_cam1,
                'cam2': robot2_candidate_cam2,
            }
            robot2_angles = {
                'cam1': robot2_angle_cam1,
                'cam2': robot2_angle_cam2,
            }

            robot2_seen_now = (
                robot2_candidate_cam1 is not None
                or robot2_candidate_cam2 is not None
            )
            robot2_reacquired = False

            # MẤT / XUẤT HIỆN LẠI ROBOT 2
            if not robot2_seen_now:
                robot2_lost_frames += 1

                if robot2_lost_frames >= ROBOT_LOST_HIDE_FRAMES:
                    real_map1.pop(robot2_id, None)
                    real_angle_map.pop(robot2_id, None)
                    robot2_source_label = 'NONE'

                if robot2_lost_frames >= ROBOT_LOST_RESET_FRAMES:
                    robot2_active_camera = None
                    robot2_active_missing_frames = 0
                    robot2_filtered_position = None
                    robot2_filtered_angle = None
                    robot2_camera_offsets['cam1'] = (0.0, 0.0)
                    robot2_camera_offsets['cam2'] = (0.0, 0.0)

            else:
                robot2_was_hidden = (
                    robot2_lost_frames >= ROBOT_LOST_HIDE_FRAMES
                )
                robot2_lost_frames = 0

                if robot2_was_hidden:
                    # Ưu tiên Cam1 lúc bắt lại nếu Cam1 đang thấy; nếu không thì Cam2.
                    if robot2_candidate_cam1 is not None:
                        robot2_active_camera = 'cam1'
                        robot2_first_position = robot2_candidate_cam1
                    else:
                        robot2_active_camera = 'cam2'
                        robot2_first_position = robot2_candidate_cam2

                    robot2_camera_offsets[robot2_active_camera] = (0.0, 0.0)
                    robot2_filtered_position = (
                        float(robot2_first_position[0]),
                        float(robot2_first_position[1]),
                    )
                    robot2_active_missing_frames = 0
                    robot2_source_label = robot2_active_camera.upper()
                    robot2_reacquired = True

            # KHÓA CAMERA VÀ HANDOFF ROBOT 2
            if robot2_active_camera is None:
                if robot2_candidate_cam1 is not None:
                    robot2_active_camera = 'cam1'
                elif robot2_candidate_cam2 is not None:
                    robot2_active_camera = 'cam2'

                if (
                    robot2_active_camera is not None
                    and robot2_filtered_position is None
                ):
                    robot2_camera_offsets[robot2_active_camera] = (0.0, 0.0)

            def corrected_candidate_robot2(source_name):
                raw_position = robot2_candidates.get(source_name)
                if raw_position is None:
                    return None

                offset_x, offset_y = robot2_camera_offsets[source_name]
                return (
                    float(raw_position[0]) + float(offset_x),
                    float(raw_position[1]) + float(offset_y),
                )

            robot2_selected_source = robot2_active_camera
            robot2_selected_position = (
                corrected_candidate_robot2(robot2_selected_source)
                if robot2_selected_source is not None
                else None
            )

            if (
                robot2_selected_source is not None
                and robot2_selected_position is not None
            ):
                # Camera hiện tại vẫn thấy Robot 2 -> tuyệt đối không đổi nguồn.
                robot2_active_missing_frames = 0

            elif robot2_selected_source is not None:
                # Chỉ khi camera active mất ID7 mới bắt đầu đếm để chuyển nguồn.
                robot2_active_missing_frames += 1
                robot2_other_source = (
                    'cam2' if robot2_selected_source == 'cam1' else 'cam1'
                )
                robot2_other_raw = robot2_candidates.get(robot2_other_source)

                if (
                    robot2_other_raw is not None
                    and robot2_active_missing_frames >= ROBOT_SWITCH_CONFIRM_FRAMES
                ):
                    # Neo camera mới vào vị trí lọc cuối cùng để tránh bước nhảy
                    # do H1 và H2 còn chênh nhau vài cm ở vùng overlap.
                    if robot2_filtered_position is not None:
                        robot2_camera_offsets[robot2_other_source] = (
                            float(robot2_filtered_position[0]) - float(robot2_other_raw[0]),
                            float(robot2_filtered_position[1]) - float(robot2_other_raw[1]),
                        )
                    else:
                        robot2_camera_offsets[robot2_other_source] = (0.0, 0.0)

                    robot2_active_camera = robot2_other_source
                    robot2_selected_source = robot2_other_source
                    robot2_selected_position = corrected_candidate_robot2(
                        robot2_other_source
                    )
                    robot2_active_missing_frames = 0
                else:
                    # Trong thời gian xác nhận handoff: GIỮ vị trí lọc cũ,
                    # không lấy xen kẽ dữ liệu từ camera còn lại.
                    robot2_selected_position = None
            else:
                robot2_selected_position = None

            # CHẶN JUMP BẤT THƯỜNG
            if (
                robot2_selected_position is not None
                and robot2_filtered_position is not None
                and not robot2_reacquired
            ):
                robot2_selected_jump = math.dist(
                    robot2_selected_position,
                    robot2_filtered_position
                )

                if robot2_selected_jump > ROBOT_MAX_JUMP_CM:
                    # Không switch ngay vì một frame jump.
                    robot2_selected_position = None
                    robot2_active_missing_frames += 1

                    robot2_other_source = (
                        'cam2' if robot2_selected_source == 'cam1' else 'cam1'
                    )
                    robot2_other_raw = robot2_candidates.get(robot2_other_source)

                    if (
                        robot2_other_raw is not None
                        and robot2_active_missing_frames >= ROBOT_SWITCH_CONFIRM_FRAMES
                    ):
                        robot2_camera_offsets[robot2_other_source] = (
                            float(robot2_filtered_position[0]) - float(robot2_other_raw[0]),
                            float(robot2_filtered_position[1]) - float(robot2_other_raw[1]),
                        )
                        robot2_active_camera = robot2_other_source
                        robot2_selected_source = robot2_other_source
                        robot2_selected_position = corrected_candidate_robot2(
                            robot2_other_source
                        )
                        robot2_active_missing_frames = 0

            # GÓC: PHẢI ĐI CÙNG CAMERA ĐANG CUNG CẤP TỌA ĐỘ
            robot2_selected_angle = None
            if robot2_selected_position is not None:
                robot2_selected_angle = robot2_angles.get(
                    robot2_selected_source
                )

            if robot2_selected_angle is not None:
                robot2_filtered_angle = smooth_angle_rad(
                    robot2_filtered_angle,
                    robot2_selected_angle,
                    alpha=0.35
                )

            # EMA VỊ TRÍ ROBOT 2
            if robot2_selected_position is not None:
                if robot2_filtered_position is None:
                    robot2_filtered_position = (
                        float(robot2_selected_position[0]),
                        float(robot2_selected_position[1]),
                    )
                else:
                    alpha2 = ROBOT_POSITION_EMA_ALPHA
                    robot2_filtered_position = (
                        (1.0 - alpha2) * robot2_filtered_position[0]
                        + alpha2 * robot2_selected_position[0],
                        (1.0 - alpha2) * robot2_filtered_position[1]
                        + alpha2 * robot2_selected_position[1],
                    )

                robot2_source_label = robot2_selected_source.upper()

            keep_robot2_visible = (
                robot2_seen_now
                or robot2_lost_frames < ROBOT_LOST_HIDE_FRAMES
            )

            if robot2_filtered_position is not None and keep_robot2_visible:
                real_map1[robot2_id] = (
                    round(float(robot2_filtered_position[0]), 1),
                    round(float(robot2_filtered_position[1]), 1),
                )
                if robot2_filtered_angle is not None:
                    real_angle_map[robot2_id] = robot2_filtered_angle
            else:
                real_map1.pop(robot2_id, None)
                real_angle_map.pop(robot2_id, None)

            # 2C. HỢP NHẤT ROBOT 3 (ID 3) TRONG VÙNG CHỒNG LẤN
            # QUAN TRỌNG:
            # Không còn lấy robot3_raw_position = real_map1[robot3_id].
            # Robot 3 có 2 candidate độc lập và một active_camera riêng.
            # Khi Cam1 + Cam2 cùng thấy ID29, camera đang active được khóa lại,
            # nên tọa độ không nhảy qua lại giữa H1 và H2 theo từng frame.
            robot3_candidates = {
                'cam1': robot3_candidate_cam1,
                'cam2': robot3_candidate_cam2,
            }
            robot3_angles = {
                'cam1': robot3_angle_cam1,
                'cam2': robot3_angle_cam2,
            }

            robot3_seen_now = (
                robot3_candidate_cam1 is not None
                or robot3_candidate_cam2 is not None
            )
            robot3_reacquired = False

            # MẤT / XUẤT HIỆN LẠI ROBOT 3
            if not robot3_seen_now:
                robot3_lost_frames += 1

                if robot3_lost_frames >= ROBOT_LOST_HIDE_FRAMES:
                    real_map1.pop(robot3_id, None)
                    real_angle_map.pop(robot3_id, None)
                    robot3_source_label = 'NONE'

                if robot3_lost_frames >= ROBOT_LOST_RESET_FRAMES:
                    robot3_active_camera = None
                    robot3_active_missing_frames = 0
                    robot3_filtered_position = None
                    robot3_filtered_angle = None
                    robot3_camera_offsets['cam1'] = (0.0, 0.0)
                    robot3_camera_offsets['cam2'] = (0.0, 0.0)

            else:
                robot3_was_hidden = (
                    robot3_lost_frames >= ROBOT_LOST_HIDE_FRAMES
                )
                robot3_lost_frames = 0

                if robot3_was_hidden:
                    # Ưu tiên Cam1 lúc bắt lại nếu Cam1 đang thấy; nếu không thì Cam2.
                    if robot3_candidate_cam1 is not None:
                        robot3_active_camera = 'cam1'
                        robot3_first_position = robot3_candidate_cam1
                    else:
                        robot3_active_camera = 'cam2'
                        robot3_first_position = robot3_candidate_cam2

                    robot3_camera_offsets[robot3_active_camera] = (0.0, 0.0)
                    robot3_filtered_position = (
                        float(robot3_first_position[0]),
                        float(robot3_first_position[1]),
                    )
                    robot3_active_missing_frames = 0
                    robot3_source_label = robot3_active_camera.upper()
                    robot3_reacquired = True

            # KHÓA CAMERA VÀ HANDOFF ROBOT 3
            if robot3_active_camera is None:
                if robot3_candidate_cam1 is not None:
                    robot3_active_camera = 'cam1'
                elif robot3_candidate_cam2 is not None:
                    robot3_active_camera = 'cam2'

                if (
                    robot3_active_camera is not None
                    and robot3_filtered_position is None
                ):
                    robot3_camera_offsets[robot3_active_camera] = (0.0, 0.0)

            def corrected_candidate_robot3(source_name):
                raw_position = robot3_candidates.get(source_name)
                if raw_position is None:
                    return None

                offset_x, offset_y = robot3_camera_offsets[source_name]
                return (
                    float(raw_position[0]) + float(offset_x),
                    float(raw_position[1]) + float(offset_y),
                )

            robot3_selected_source = robot3_active_camera
            robot3_selected_position = (
                corrected_candidate_robot3(robot3_selected_source)
                if robot3_selected_source is not None
                else None
            )

            if (
                robot3_selected_source is not None
                and robot3_selected_position is not None
            ):
                # Camera hiện tại vẫn thấy Robot 3 -> tuyệt đối không đổi nguồn.
                robot3_active_missing_frames = 0

            elif robot3_selected_source is not None:
                # Chỉ khi camera active mất ID29 mới bắt đầu đếm để chuyển nguồn.
                robot3_active_missing_frames += 1
                robot3_other_source = (
                    'cam2' if robot3_selected_source == 'cam1' else 'cam1'
                )
                robot3_other_raw = robot3_candidates.get(robot3_other_source)

                if (
                    robot3_other_raw is not None
                    and robot3_active_missing_frames >= ROBOT_SWITCH_CONFIRM_FRAMES
                ):
                    # Neo camera mới vào vị trí lọc cuối cùng để tránh bước nhảy
                    # do H1 và H2 còn chênh nhau vài cm ở vùng overlap.
                    if robot3_filtered_position is not None:
                        robot3_camera_offsets[robot3_other_source] = (
                            float(robot3_filtered_position[0]) - float(robot3_other_raw[0]),
                            float(robot3_filtered_position[1]) - float(robot3_other_raw[1]),
                        )
                    else:
                        robot3_camera_offsets[robot3_other_source] = (0.0, 0.0)

                    robot3_active_camera = robot3_other_source
                    robot3_selected_source = robot3_other_source
                    robot3_selected_position = corrected_candidate_robot3(
                        robot3_other_source
                    )
                    robot3_active_missing_frames = 0
                else:
                    # Trong thời gian xác nhận handoff: GIỮ vị trí lọc cũ,
                    # không lấy xen kẽ dữ liệu từ camera còn lại.
                    robot3_selected_position = None
            else:
                robot3_selected_position = None

            # CHẶN JUMP BẤT THƯỜNG
            if (
                robot3_selected_position is not None
                and robot3_filtered_position is not None
                and not robot3_reacquired
            ):
                robot3_selected_jump = math.dist(
                    robot3_selected_position,
                    robot3_filtered_position
                )

                if robot3_selected_jump > ROBOT_MAX_JUMP_CM:
                    # Không switch ngay vì một frame jump.
                    robot3_selected_position = None
                    robot3_active_missing_frames += 1

                    robot3_other_source = (
                        'cam2' if robot3_selected_source == 'cam1' else 'cam1'
                    )
                    robot3_other_raw = robot3_candidates.get(robot3_other_source)

                    if (
                        robot3_other_raw is not None
                        and robot3_active_missing_frames >= ROBOT_SWITCH_CONFIRM_FRAMES
                    ):
                        robot3_camera_offsets[robot3_other_source] = (
                            float(robot3_filtered_position[0]) - float(robot3_other_raw[0]),
                            float(robot3_filtered_position[1]) - float(robot3_other_raw[1]),
                        )
                        robot3_active_camera = robot3_other_source
                        robot3_selected_source = robot3_other_source
                        robot3_selected_position = corrected_candidate_robot3(
                            robot3_other_source
                        )
                        robot3_active_missing_frames = 0

            # GÓC: PHẢI ĐI CÙNG CAMERA ĐANG CUNG CẤP TỌA ĐỘ
            robot3_selected_angle = None
            if robot3_selected_position is not None:
                robot3_selected_angle = robot3_angles.get(
                    robot3_selected_source
                )

            if robot3_selected_angle is not None:
                robot3_filtered_angle = smooth_angle_rad(
                    robot3_filtered_angle,
                    robot3_selected_angle,
                    alpha=0.35
                )

            # EMA VỊ TRÍ ROBOT 3
            if robot3_selected_position is not None:
                if robot3_filtered_position is None:
                    robot3_filtered_position = (
                        float(robot3_selected_position[0]),
                        float(robot3_selected_position[1]),
                    )
                else:
                    alpha3 = ROBOT_POSITION_EMA_ALPHA
                    robot3_filtered_position = (
                        (1.0 - alpha3) * robot3_filtered_position[0]
                        + alpha3 * robot3_selected_position[0],
                        (1.0 - alpha3) * robot3_filtered_position[1]
                        + alpha3 * robot3_selected_position[1],
                    )

                robot3_source_label = robot3_selected_source.upper()

            keep_robot3_visible = (
                robot3_seen_now
                or robot3_lost_frames < ROBOT_LOST_HIDE_FRAMES
            )

            if robot3_filtered_position is not None and keep_robot3_visible:
                real_map1[robot3_id] = (
                    round(float(robot3_filtered_position[0]), 1),
                    round(float(robot3_filtered_position[1]), 1),
                )
                if robot3_filtered_angle is not None:
                    real_angle_map[robot3_id] = robot3_filtered_angle
            else:
                real_map1.pop(robot3_id, None)
                real_angle_map.pop(robot3_id, None)

            # 2D. HỢP NHẤT ROBOT 4 (ID 29) TRONG VÙNG CHỒNG LẤN
            # QUAN TRỌNG:
            # Không còn lấy robot4_raw_position = real_map1[robot4_id].
            # Robot 4 có 2 candidate độc lập và một active_camera riêng.
            # Khi Cam1 + Cam2 cùng thấy ID29, camera đang active được khóa lại,
            # nên tọa độ không nhảy qua lại giữa H1 và H2 theo từng frame.
            robot4_candidates = {
                'cam1': robot4_candidate_cam1,
                'cam2': robot4_candidate_cam2,
            }
            robot4_angles = {
                'cam1': robot4_angle_cam1,
                'cam2': robot4_angle_cam2,
            }

            robot4_seen_now = (
                robot4_candidate_cam1 is not None
                or robot4_candidate_cam2 is not None
            )
            robot4_reacquired = False

            # MẤT / XUẤT HIỆN LẠI ROBOT 4
            if not robot4_seen_now:
                robot4_lost_frames += 1

                if robot4_lost_frames >= ROBOT_LOST_HIDE_FRAMES:
                    real_map1.pop(robot4_id, None)
                    real_angle_map.pop(robot4_id, None)
                    robot4_source_label = 'NONE'

                if robot4_lost_frames >= ROBOT_LOST_RESET_FRAMES:
                    robot4_active_camera = None
                    robot4_active_missing_frames = 0
                    robot4_filtered_position = None
                    robot4_filtered_angle = None
                    robot4_camera_offsets['cam1'] = (0.0, 0.0)
                    robot4_camera_offsets['cam2'] = (0.0, 0.0)

            else:
                robot4_was_hidden = (
                    robot4_lost_frames >= ROBOT_LOST_HIDE_FRAMES
                )
                robot4_lost_frames = 0

                if robot4_was_hidden:
                    # Ưu tiên Cam1 lúc bắt lại nếu Cam1 đang thấy; nếu không thì Cam2.
                    if robot4_candidate_cam1 is not None:
                        robot4_active_camera = 'cam1'
                        robot4_first_position = robot4_candidate_cam1
                    else:
                        robot4_active_camera = 'cam2'
                        robot4_first_position = robot4_candidate_cam2

                    robot4_camera_offsets[robot4_active_camera] = (0.0, 0.0)
                    robot4_filtered_position = (
                        float(robot4_first_position[0]),
                        float(robot4_first_position[1]),
                    )
                    robot4_active_missing_frames = 0
                    robot4_source_label = robot4_active_camera.upper()
                    robot4_reacquired = True

            # KHÓA CAMERA VÀ HANDOFF ROBOT 4
            if robot4_active_camera is None:
                if robot4_candidate_cam1 is not None:
                    robot4_active_camera = 'cam1'
                elif robot4_candidate_cam2 is not None:
                    robot4_active_camera = 'cam2'

                if (
                    robot4_active_camera is not None
                    and robot4_filtered_position is None
                ):
                    robot4_camera_offsets[robot4_active_camera] = (0.0, 0.0)

            def corrected_candidate_robot4(source_name):
                raw_position = robot4_candidates.get(source_name)
                if raw_position is None:
                    return None

                offset_x, offset_y = robot4_camera_offsets[source_name]
                return (
                    float(raw_position[0]) + float(offset_x),
                    float(raw_position[1]) + float(offset_y),
                )

            robot4_selected_source = robot4_active_camera
            robot4_selected_position = (
                corrected_candidate_robot4(robot4_selected_source)
                if robot4_selected_source is not None
                else None
            )

            if (
                robot4_selected_source is not None
                and robot4_selected_position is not None
            ):
                # Camera hiện tại vẫn thấy Robot 4 -> tuyệt đối không đổi nguồn.
                robot4_active_missing_frames = 0

            elif robot4_selected_source is not None:
                # Chỉ khi camera active mất ID29 mới bắt đầu đếm để chuyển nguồn.
                robot4_active_missing_frames += 1
                robot4_other_source = (
                    'cam2' if robot4_selected_source == 'cam1' else 'cam1'
                )
                robot4_other_raw = robot4_candidates.get(robot4_other_source)

                if (
                    robot4_other_raw is not None
                    and robot4_active_missing_frames >= ROBOT_SWITCH_CONFIRM_FRAMES
                ):
                    # Neo camera mới vào vị trí lọc cuối cùng để tránh bước nhảy
                    # do H1 và H2 còn chênh nhau vài cm ở vùng overlap.
                    if robot4_filtered_position is not None:
                        robot4_camera_offsets[robot4_other_source] = (
                            float(robot4_filtered_position[0]) - float(robot4_other_raw[0]),
                            float(robot4_filtered_position[1]) - float(robot4_other_raw[1]),
                        )
                    else:
                        robot4_camera_offsets[robot4_other_source] = (0.0, 0.0)

                    robot4_active_camera = robot4_other_source
                    robot4_selected_source = robot4_other_source
                    robot4_selected_position = corrected_candidate_robot4(
                        robot4_other_source
                    )
                    robot4_active_missing_frames = 0
                else:
                    # Trong thời gian xác nhận handoff: GIỮ vị trí lọc cũ,
                    # không lấy xen kẽ dữ liệu từ camera còn lại.
                    robot4_selected_position = None
            else:
                robot4_selected_position = None

            # CHẶN JUMP BẤT THƯỜNG
            if (
                robot4_selected_position is not None
                and robot4_filtered_position is not None
                and not robot4_reacquired
            ):
                robot4_selected_jump = math.dist(
                    robot4_selected_position,
                    robot4_filtered_position
                )

                if robot4_selected_jump > ROBOT_MAX_JUMP_CM:
                    # Không switch ngay vì một frame jump.
                    robot4_selected_position = None
                    robot4_active_missing_frames += 1

                    robot4_other_source = (
                        'cam2' if robot4_selected_source == 'cam1' else 'cam1'
                    )
                    robot4_other_raw = robot4_candidates.get(robot4_other_source)

                    if (
                        robot4_other_raw is not None
                        and robot4_active_missing_frames >= ROBOT_SWITCH_CONFIRM_FRAMES
                    ):
                        robot4_camera_offsets[robot4_other_source] = (
                            float(robot4_filtered_position[0]) - float(robot4_other_raw[0]),
                            float(robot4_filtered_position[1]) - float(robot4_other_raw[1]),
                        )
                        robot4_active_camera = robot4_other_source
                        robot4_selected_source = robot4_other_source
                        robot4_selected_position = corrected_candidate_robot4(
                            robot4_other_source
                        )
                        robot4_active_missing_frames = 0

            # GÓC: PHẢI ĐI CÙNG CAMERA ĐANG CUNG CẤP TỌA ĐỘ
            robot4_selected_angle = None
            if robot4_selected_position is not None:
                robot4_selected_angle = robot4_angles.get(
                    robot4_selected_source
                )

            if robot4_selected_angle is not None:
                robot4_filtered_angle = smooth_angle_rad(
                    robot4_filtered_angle,
                    robot4_selected_angle,
                    alpha=0.35
                )

            # EMA VỊ TRÍ ROBOT 4
            if robot4_selected_position is not None:
                if robot4_filtered_position is None:
                    robot4_filtered_position = (
                        float(robot4_selected_position[0]),
                        float(robot4_selected_position[1]),
                    )
                else:
                    alpha4 = ROBOT_POSITION_EMA_ALPHA
                    robot4_filtered_position = (
                        (1.0 - alpha4) * robot4_filtered_position[0]
                        + alpha4 * robot4_selected_position[0],
                        (1.0 - alpha4) * robot4_filtered_position[1]
                        + alpha4 * robot4_selected_position[1],
                    )

                robot4_source_label = robot4_selected_source.upper()

            keep_robot4_visible = (
                robot4_seen_now
                or robot4_lost_frames < ROBOT_LOST_HIDE_FRAMES
            )

            if robot4_filtered_position is not None and keep_robot4_visible:
                real_map1[robot4_id] = (
                    round(float(robot4_filtered_position[0]), 1),
                    round(float(robot4_filtered_position[1]), 1),
                )
                if robot4_filtered_angle is not None:
                    real_angle_map[robot4_id] = robot4_filtered_angle
            else:
                real_map1.pop(robot4_id, None)
                real_angle_map.pop(robot4_id, None)

            # 3. GỬI 4 ROBOT QUA SERIAL -> ESP32
            # Giao thức Python -> ESP32 gateway:
            #   ID;X.X;Y.Y;ANGLE_RAD#
            # X, Y gửi 1 chữ số thập phân; ANGLE_RAD gửi 5 chữ số thập phân.
            # Ví dụ một nhịp có đủ 4 robot:
            #   8;352.4;64.7;0.75400#7;420.8;80.1;-1.54300#3;500.2;40.0;1.57080#29;510.0;60.0;3.14159#
            # ESP32 gateway chỉ cần tách từng packet theo dấu '#', đọc ID đầu tiên
            # rồi route packet đến đúng robot.
            # ---------------- Robot 1 ----------------
            if robot_filtered_position is not None and keep_robot_visible:
                x_send = round(float(robot_filtered_position[0]), 1)
                y_send = round(float(robot_filtered_position[1]), 1)
                angle_send = (
                    float(robot_filtered_angle)
                    if robot_filtered_angle is not None else 0.0
                )

                with latest_xy_lock:
                    latest_x = x_send
                    latest_y = y_send

                _set_latest_robot_packet(
                    robot_id,
                    f"{robot_id};{x_send:.1f};{y_send:.1f};{angle_send:.5f}#"
                )
            else:
                with latest_xy_lock:
                    latest_x = None
                    latest_y = None
                _set_latest_robot_packet(robot_id, None)

            # ---------------- Robot 2 ----------------
            if robot2_filtered_position is not None and keep_robot2_visible:
                x2_send = round(float(robot2_filtered_position[0]), 1)
                y2_send = round(float(robot2_filtered_position[1]), 1)
                angle2_send = (
                    float(robot2_filtered_angle)
                    if robot2_filtered_angle is not None else 0.0
                )

                with latest_xy_lock:
                    latest_x2 = x2_send
                    latest_y2 = y2_send

                _set_latest_robot_packet(
                    robot2_id,
                    f"{robot2_id};{x2_send:.1f};{y2_send:.1f};{angle2_send:.5f}#"
                )
            else:
                with latest_xy_lock:
                    latest_x2 = None
                    latest_y2 = None
                _set_latest_robot_packet(robot2_id, None)

            # ---------------- Robot 3 ----------------
            if robot3_filtered_position is not None and keep_robot3_visible:
                x3_send = round(float(robot3_filtered_position[0]), 1)
                y3_send = round(float(robot3_filtered_position[1]), 1)
                angle3_send = (
                    float(robot3_filtered_angle)
                    if robot3_filtered_angle is not None else 0.0
                )

                with latest_xy_lock:
                    latest_x3 = x3_send
                    latest_y3 = y3_send

                _set_latest_robot_packet(
                    robot3_id,
                    f"{robot3_id};{x3_send:.1f};{y3_send:.1f};{angle3_send:.5f}#"
                )
            else:
                with latest_xy_lock:
                    latest_x3 = None
                    latest_y3 = None
                _set_latest_robot_packet(robot3_id, None)

            # ---------------- Robot 4 ----------------
            if robot4_filtered_position is not None and keep_robot4_visible:
                x4_send = round(float(robot4_filtered_position[0]), 1)
                y4_send = round(float(robot4_filtered_position[1]), 1)
                angle4_send = (
                    float(robot4_filtered_angle)
                    if robot4_filtered_angle is not None else 0.0
                )

                with latest_xy_lock:
                    latest_x4 = x4_send
                    latest_y4 = y4_send

                _set_latest_robot_packet(
                    robot4_id,
                    f"{robot4_id};{x4_send:.1f};{y4_send:.1f};{angle4_send:.5f}#"
                )
            else:
                with latest_xy_lock:
                    latest_x4 = None
                    latest_y4 = None
                _set_latest_robot_packet(robot4_id, None)

            # Không gửi Serial ở luồng camera nữa.
            # send_data() sẽ tự đọc các packet mới nhất và phát ở TX_TARGET_HZ.

            # 4. HỆ THỐNG RADAR TỔNG (Độc lập 100%)
            # đúng cùng tỷ lệ với lưới tọa độ đã vẽ phía trên.
            SCALE = MAP_SCALE
            OFFSET_X = _OFFSET_X
            OFFSET_Y = _OFFSET_Y

            current_seen_ids = []
            if ids is not None:
                current_seen_ids.extend(ids.flatten().tolist())
            if ids2 is not None:
                current_seen_ids.extend(ids2.flatten().tolist())

            # DỌN DẸP RADAR (Chỉ giữ lại Robot và các mã đang được 2 Cam nhìn thấy)
            keys_to_delete = []

            for check_key in list(real_map1.keys()):

                if (
                        check_key not in ROBOT_IDS
                        and check_key not in current_seen_ids
                ):
                    keys_to_delete.append(check_key)

            for k in keys_to_delete:
                # Xóa tọa độ
                real_map1.pop(k, None)

                # Xóa luôn góc
                real_angle_map.pop(k, None)

            # QUÉT VÀ VẼ MỌI VẬT THỂ LÊN RADAR
            for key in real_map1:
                real_x = real_map1[key][0]
                real_y = real_map1[key][1]

                draw_x = int(real_x * SCALE + OFFSET_X)
                draw_y = int(OFFSET_Y - real_y * SCALE)

                if key == robot_id:
                    last_path_point = next((p for p in reversed(robot_path) if p is not None), None)
                    if last_path_point is None:
                        robot_path.append((real_x, real_y, draw_x, draw_y))
                        robot_last_motion_time = time.perf_counter()
                    else:
                        path_step = math.dist((real_x, real_y), last_path_point[0:2])
                        if path_step >= ROBOT_PATH_BREAK_STEP_CM:
                            # Không nối/cộng một bước nhảy bất thường.
                            robot_path.append(None)
                            robot_path.append((real_x, real_y, draw_x, draw_y))
                            robot_speed_cm_s = 0.0
                            robot_last_motion_time = time.perf_counter()
                        elif path_step >= ROBOT_PATH_MIN_STEP_CM:
                            # Cộng đúng chiều dài đoạn quỹ đạo hợp lệ và ước lượng vận tốc.
                            motion_now = time.perf_counter()
                            if robot_last_motion_time is not None:
                                motion_dt = motion_now - robot_last_motion_time
                                if motion_dt > 1e-3:
                                    instant_speed = path_step / motion_dt
                                    robot_speed_cm_s = (
                                        instant_speed if robot_speed_cm_s <= 0.0
                                        else 0.70 * robot_speed_cm_s + 0.30 * instant_speed
                                    )
                            robot_last_motion_time = motion_now
                            robot_total_distance_cm += path_step

                            new_path_point = (real_x, real_y, draw_x, draw_y)

                            if robot_path and robot_path[-1] is not None:
                                _append_cached_path_segment(
                                    robot_id,
                                    robot_path[-1],
                                    new_path_point
                                )

                            robot_path.append(new_path_point)

                    # Không cắt ngắn quỹ đạo R8.

                    cv2.circle(background, (draw_x, draw_y), 7, (48, 79, 220), -1, cv2.LINE_AA)
                    robot1_angle_text = (
                        f" A={real_angle_map[robot_id]:.5f} rad"
                        if robot_id in real_angle_map else ''
                    )
                    cv2.putText(background,
                                f"Robot [{key}]: ({int(real_x)}, {int(real_y)}){robot1_angle_text}",
                                (draw_x + 10, draw_y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (48, 79, 180), 1,
                                cv2.LINE_AA)

                elif key == robot2_id:
                    # ================= Robot 2 - quỹ đạo và metrics riêng =================
                    last_path_point2 = next((p for p in reversed(robot2_path) if p is not None), None)
                    if last_path_point2 is None:
                        robot2_path.append((real_x, real_y, draw_x, draw_y))
                        robot2_last_motion_time = time.perf_counter()
                    else:
                        path_step2 = math.dist((real_x, real_y), last_path_point2[0:2])
                        if path_step2 >= ROBOT_PATH_BREAK_STEP_CM:
                            robot2_path.append(None)
                            robot2_path.append((real_x, real_y, draw_x, draw_y))
                            robot2_speed_cm_s = 0.0
                            robot2_last_motion_time = time.perf_counter()
                        elif path_step2 >= ROBOT_PATH_MIN_STEP_CM:
                            motion_now2 = time.perf_counter()
                            if robot2_last_motion_time is not None:
                                motion_dt2 = motion_now2 - robot2_last_motion_time
                                if motion_dt2 > 1e-3:
                                    instant_speed2 = path_step2 / motion_dt2
                                    robot2_speed_cm_s = (
                                        instant_speed2 if robot2_speed_cm_s <= 0.0
                                        else 0.70 * robot2_speed_cm_s + 0.30 * instant_speed2
                                    )
                            robot2_last_motion_time = motion_now2
                            robot2_total_distance_cm += path_step2

                            new_path_point2 = (real_x, real_y, draw_x, draw_y)

                            if robot2_path and robot2_path[-1] is not None:
                                _append_cached_path_segment(
                                    robot2_id,
                                    robot2_path[-1],
                                    new_path_point2
                                )

                            robot2_path.append(new_path_point2)

                    # Không cắt ngắn quỹ đạo R7.

                    # Robot 2 dùng màu xanh lam/cyan để phân biệt Robot 1.
                    robot2_color = (220, 150, 35)
                    cv2.circle(background, (draw_x, draw_y), 7, robot2_color, -1, cv2.LINE_AA)
                    robot2_angle_text = (
                        f" A={real_angle_map[robot2_id]:.5f} rad"
                        if robot2_id in real_angle_map else ''
                    )
                    cv2.putText(background,
                                f"Robot [{key}]: ({int(real_x)}, {int(real_y)}){robot2_angle_text}",
                                (draw_x + 10, draw_y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.46, robot2_color, 1,
                                cv2.LINE_AA)
                elif key == robot3_id:
                    # ================= Robot 3 - quỹ đạo và metrics riêng =================
                    last_path_point3 = next((p for p in reversed(robot3_path) if p is not None), None)
                    if last_path_point3 is None:
                        robot3_path.append((real_x, real_y, draw_x, draw_y))
                        robot3_last_motion_time = time.perf_counter()
                    else:
                        path_step3 = math.dist((real_x, real_y), last_path_point3[0:2])
                        if path_step3 >= ROBOT_PATH_BREAK_STEP_CM:
                            robot3_path.append(None)
                            robot3_path.append((real_x, real_y, draw_x, draw_y))
                            robot3_speed_cm_s = 0.0
                            robot3_last_motion_time = time.perf_counter()
                        elif path_step3 >= ROBOT_PATH_MIN_STEP_CM:
                            motion_now3 = time.perf_counter()
                            if robot3_last_motion_time is not None:
                                motion_dt3 = motion_now3 - robot3_last_motion_time
                                if motion_dt3 > 1e-3:
                                    instant_speed3 = path_step3 / motion_dt3
                                    robot3_speed_cm_s = (
                                        instant_speed3 if robot3_speed_cm_s <= 0.0
                                        else 0.70 * robot3_speed_cm_s + 0.30 * instant_speed3
                                    )
                            robot3_last_motion_time = motion_now3
                            robot3_total_distance_cm += path_step3

                            new_path_point3 = (real_x, real_y, draw_x, draw_y)

                            if robot3_path and robot3_path[-1] is not None:
                                _append_cached_path_segment(
                                    robot3_id,
                                    robot3_path[-1],
                                    new_path_point3
                                )

                            robot3_path.append(new_path_point3)

                    # Không cắt ngắn quỹ đạo R3.

                    robot3_color = (160, 70, 210)
                    cv2.circle(background, (draw_x, draw_y), 7, robot3_color, -1, cv2.LINE_AA)
                    robot3_angle_text = (
                        f" A={real_angle_map[robot3_id]:.5f} rad"
                        if robot3_id in real_angle_map else ''
                    )
                    cv2.putText(
                        background,
                        f"Robot [{key}]: ({int(real_x)}, {int(real_y)}){robot3_angle_text}",
                        (draw_x + 10, draw_y - 12),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.46,
                        robot3_color,
                        1,
                        cv2.LINE_AA
                    )

                elif key == robot4_id:
                    # ================= Robot 4 - quỹ đạo và metrics riêng =================
                    last_path_point4 = next((p for p in reversed(robot4_path) if p is not None), None)
                    if last_path_point4 is None:
                        robot4_path.append((real_x, real_y, draw_x, draw_y))
                        robot4_last_motion_time = time.perf_counter()
                    else:
                        path_step4 = math.dist((real_x, real_y), last_path_point4[0:2])
                        if path_step4 >= ROBOT_PATH_BREAK_STEP_CM:
                            robot4_path.append(None)
                            robot4_path.append((real_x, real_y, draw_x, draw_y))
                            robot4_speed_cm_s = 0.0
                            robot4_last_motion_time = time.perf_counter()
                        elif path_step4 >= ROBOT_PATH_MIN_STEP_CM:
                            motion_now4 = time.perf_counter()
                            if robot4_last_motion_time is not None:
                                motion_dt4 = motion_now4 - robot4_last_motion_time
                                if motion_dt4 > 1e-3:
                                    instant_speed4 = path_step4 / motion_dt4
                                    robot4_speed_cm_s = (
                                        instant_speed4 if robot4_speed_cm_s <= 0.0
                                        else 0.70 * robot4_speed_cm_s + 0.30 * instant_speed4
                                    )
                            robot4_last_motion_time = motion_now4
                            robot4_total_distance_cm += path_step4

                            new_path_point4 = (real_x, real_y, draw_x, draw_y)

                            if robot4_path and robot4_path[-1] is not None:
                                _append_cached_path_segment(
                                    robot4_id,
                                    robot4_path[-1],
                                    new_path_point4
                                )

                            robot4_path.append(new_path_point4)

                    # Không cắt ngắn quỹ đạo R29.

                    robot4_color = (80, 180, 180)
                    cv2.circle(background, (draw_x, draw_y), 7, robot4_color, -1, cv2.LINE_AA)
                    robot4_angle_text = (
                        f" A={real_angle_map[robot4_id]:.5f} rad"
                        if robot4_id in real_angle_map else ''
                    )
                    cv2.putText(
                        background,
                        f"Robot [{key}]: ({int(real_x)}, {int(real_y)}){robot4_angle_text}",
                        (draw_x + 10, draw_y - 12),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.46,
                        robot4_color,
                        1,
                        cv2.LINE_AA
                    )

                else:
                    # Các mã còn lại (bao gồm cả 0, 1, 2, 3) đều vẽ màu cam đậm như vật thể bình thường
                    cv2.circle(
                        background,
                        (draw_x, draw_y),
                        5,
                        (45, 158, 230),
                        -1,
                        cv2.LINE_AA
                    )

                    marker_angle = real_angle_map.get(key)

                    if marker_angle is not None:

                        marker_text = (
                            f"[{key}] "
                            f"({int(real_x)}, {int(real_y)}) "
                            f"A={marker_angle:.5f} rad"
                        )

                    else:

                        marker_text = (
                            f"[{key}] "
                            f"({int(real_x)}, {int(real_y)})"
                        )

                    cv2.putText(
                        background,
                        marker_text,
                        (draw_x + 8, draw_y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.38,
                        (50, 120, 170),
                        1,
                        cv2.LINE_AA
                    )

            # Quỹ đạo đầy đủ đã được cache trong map_with_paths.
            # Không vẽ lại toàn bộ lịch sử ở mỗi frame.

        # VẼ WAYPOINT CỦA 4 ROBOT
        waypoint_colors = {
            robot_id: (60, 130, 220),
            robot2_id: (220, 150, 35),
            robot3_id: (160, 70, 210),
            robot4_id: (80, 180, 180),
        }

        for target_rid, points in target_waypoints.items():
            if not points:
                continue

            color = waypoint_colors[target_rid]

            for point_idx, (tx_cm, ty_cm) in enumerate(points):
                px, py = _world_to_map_pixel(tx_cm, ty_cm)

                # Đường nối giữa các điểm đích theo đúng thứ tự robot phải đi.
                if point_idx > 0:
                    prev_x, prev_y = points[point_idx - 1]
                    ppx, ppy = _world_to_map_pixel(prev_x, prev_y)

                    cv2.line(
                        background,
                        (ppx, ppy),
                        (px, py),
                        color,
                        1,
                        cv2.LINE_AA
                    )

                # Điểm đích.
                cv2.circle(
                    background,
                    (px, py),
                    6,
                    color,
                    -1,
                    cv2.LINE_AA
                )

                cv2.circle(
                    background,
                    (px, py),
                    9,
                    color,
                    1,
                    cv2.LINE_AA
                )

                # Số thứ tự waypoint.
                cv2.putText(
                    background,
                    str(point_idx + 1),
                    (px + 9, py - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.36,
                    color,
                    1,
                    cv2.LINE_AA
                )

        elapsed = time.time() - start
        if elapsed > 1.0:
            fps_now = num_frames / elapsed
            fps_display = fps_now if fps_display <= 0 else (0.78 * fps_display + 0.22 * fps_now)
            fps = fps_display
            num_frames = 0
            start = time.time()

        # ================= RENDER DASHBOARD  =================
        now_ui = time.perf_counter()
        if now_ui - last_ui_render >= UI_RENDER_INTERVAL:
            last_ui_render = now_ui

            # CẮT + SCALE BẢN ĐỒ THEO ZOOM
            _clamp_map_center()

            zoom_value = float(ui_state['zoom'])
            src_w_float, src_h_float = _view_source_size(zoom_value)

            src_w = max(1, int(round(src_w_float)))
            src_h = max(1, int(round(src_h_float)))

            center_x = float(ui_state['center_x'])
            center_y = float(ui_state['center_y'])

            src_x1 = int(round(center_x - src_w / 2.0))
            src_y1 = int(round(center_y - src_h / 2.0))
            src_x2 = src_x1 + src_w
            src_y2 = src_y1 + src_h

            # Tạo source canvas để zoom-out vẫn hoạt động kể cả khi
            # viewport cao hơn canvas gốc.
            map_source = np.full(
                (src_h, src_w, 3),
                MAP_BG,
                dtype=np.uint8
            )

            copy_x1 = max(0, src_x1)
            copy_y1 = max(0, src_y1)
            copy_x2 = min(background.shape[1], src_x2)
            copy_y2 = min(background.shape[0], src_y2)

            if copy_x2 > copy_x1 and copy_y2 > copy_y1:
                dst_x1 = copy_x1 - src_x1
                dst_y1 = copy_y1 - src_y1

                dst_x2 = dst_x1 + (copy_x2 - copy_x1)
                dst_y2 = dst_y1 + (copy_y2 - copy_y1)

                map_source[
                    dst_y1:dst_y2,
                    dst_x1:dst_x2
                ] = background[
                    copy_y1:copy_y2,
                    copy_x1:copy_x2
                ]

            interpolation = (
                cv2.INTER_AREA
                if zoom_value < 1.0
                else cv2.INTER_LINEAR
            )

            map_view = cv2.resize(
                map_source,
                (MAP_VIEW_WIDTH, MAP_VIEW_HEIGHT),
                interpolation=interpolation
            )

            dashboard = np.full(
                (DASHBOARD_HEIGHT, DASHBOARD_WIDTH, 3),
                UI_BG,
                dtype=np.uint8
            )

            current_clock = time.strftime('%H:%M:%S')
            current_date = time.strftime('%d/%m/%Y')

            # ================= HEADER =================
            cv2.rectangle(dashboard, (0, 0), (DASHBOARD_WIDTH, HEADER_H), HEADER_BG, -1)
            cv2.rectangle(dashboard, (0, HEADER_H - 4), (DASHBOARD_WIDTH, HEADER_H), BLUE, -1)
            cv2.putText(dashboard, '4 ROBOT TRACKING', (24, 37),
                        cv2.FONT_HERSHEY_DUPLEX, 0.93, TEXT_MAIN, 1, cv2.LINE_AA)
            cv2.putText(dashboard, 'SMART CONTROL DASHBOARD', (24, 67),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, TEXT_MUTED, 1, cv2.LINE_AA)
            cv2.putText(dashboard, 'C  CLEAR PATH      E  CALIB CAM2 UNDIST      Q  QUIT',
                        (365, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                        TEXT_MUTED, 1, cv2.LINE_AA)
            cv2.putText(dashboard, f'{current_date}    {current_clock}',
                        (365, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        TEXT_SOFT, 1, cv2.LINE_AA)

            _draw_round_rect(dashboard, QUIT_BUTTON, RED, 12, border=CARD_EDGE)
            qx1, qy1, qx2, qy2 = QUIT_BUTTON
            cv2.putText(dashboard, 'QUIT  Q', (qx1 + 19, qy1 + 30),
                        cv2.FONT_HERSHEY_DUPLEX, 0.48, TEXT_MAIN, 1, cv2.LINE_AA)

            # ================= MAP =================
            _draw_round_rect(dashboard,
                             (MAP_X - 4, MAP_Y - 4,
                              MAP_X + MAP_VIEW_WIDTH + 4, MAP_Y + MAP_VIEW_HEIGHT + 4),
                             (245, 245, 245), 14, border=CARD_EDGE)
            dashboard[MAP_Y:MAP_Y + MAP_VIEW_HEIGHT,
                      MAP_X:MAP_X + MAP_VIEW_WIDTH] = map_view
            cv2.rectangle(dashboard, (MAP_X, MAP_Y),
                          (MAP_X + MAP_VIEW_WIDTH - 1, MAP_Y + MAP_VIEW_HEIGHT - 1),
                          CARD_EDGE, 1, cv2.LINE_AA)

            with latest_xy_lock:
                ui_x = latest_x
                ui_y = latest_y
                ui_x2 = latest_x2
                ui_y2 = latest_y2
                ui_x3 = latest_x3
                ui_y3 = latest_y3
                ui_x4 = latest_x4
                ui_y4 = latest_y4

            robot_visible = (
                ui_x is not None and ui_y is not None
                and robot_source_label != 'NONE'
            )
            robot2_visible = (
                ui_x2 is not None and ui_y2 is not None
                and robot2_source_label != 'NONE'
            )
            robot3_visible = (
                ui_x3 is not None and ui_y3 is not None
                and robot3_source_label != 'NONE'
            )
            robot4_visible = (
                ui_x4 is not None and ui_y4 is not None
                and robot4_source_label != 'NONE'
            )
            robot_state_text = 'VISIBLE' if robot_visible else 'SEARCHING'
            robot_state_color = GREEN if robot_visible else ORANGE
            robot2_state_text = 'VISIBLE' if robot2_visible else 'SEARCHING'
            robot2_state_color = GREEN if robot2_visible else ORANGE
            robot3_state_text = 'VISIBLE' if robot3_visible else 'SEARCHING'
            robot3_state_color = GREEN if robot3_visible else ORANGE
            robot4_state_text = 'VISIBLE' if robot4_visible else 'SEARCHING'
            robot4_state_color = GREEN if robot4_visible else ORANGE
            # Overlay trạng thái 4 robot trên map.
            map_status_rect = (MAP_X + 14, MAP_Y + 14, MAP_X + 285, MAP_Y + 135)
            _draw_round_rect(dashboard, map_status_rect, HEADER_BG, 11, border=CARD_EDGE)

            _draw_status_dot(dashboard, (MAP_X + 33, MAP_Y + 33), robot_state_color)
            cv2.putText(
                dashboard,
                f'R{robot_id} {robot_state_text} {robot_source_label}',
                (MAP_X + 49, MAP_Y + 38),
                cv2.FONT_HERSHEY_DUPLEX, 0.36, TEXT_MAIN, 1, cv2.LINE_AA
            )

            _draw_status_dot(dashboard, (MAP_X + 33, MAP_Y + 60), robot2_state_color)
            cv2.putText(
                dashboard,
                f'R{robot2_id} {robot2_state_text} {robot2_source_label}',
                (MAP_X + 49, MAP_Y + 65),
                cv2.FONT_HERSHEY_DUPLEX, 0.36, TEXT_MAIN, 1, cv2.LINE_AA
            )

            _draw_status_dot(dashboard, (MAP_X + 33, MAP_Y + 87), robot3_state_color)
            cv2.putText(
                dashboard,
                f'R{robot3_id} {robot3_state_text} {robot3_source_label}',
                (MAP_X + 49, MAP_Y + 92),
                cv2.FONT_HERSHEY_DUPLEX, 0.36, TEXT_MAIN, 1, cv2.LINE_AA
            )

            _draw_status_dot(dashboard, (MAP_X + 33, MAP_Y + 114), robot4_state_color)
            cv2.putText(
                dashboard,
                f'R{robot4_id} {robot4_state_text} {robot4_source_label}',
                (MAP_X + 49, MAP_Y + 119),
                cv2.FONT_HERSHEY_DUPLEX, 0.36, TEXT_MAIN, 1, cv2.LINE_AA
            )

            # Chú giải nhỏ trên map.
            legend_rect = (MAP_X + MAP_VIEW_WIDTH - 250, MAP_Y + 14,
                           MAP_X + MAP_VIEW_WIDTH - 14, MAP_Y + 82)
            _draw_round_rect(dashboard, legend_rect, HEADER_BG, 11, border=CARD_EDGE)
            cv2.circle(dashboard, (legend_rect[0] + 20, legend_rect[1] + 22), 5,
                       (48, 79, 220), -1, cv2.LINE_AA)
            cv2.putText(dashboard, f'R{robot_id}', (legend_rect[0] + 34, legend_rect[1] + 27),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.37, TEXT_MUTED, 1, cv2.LINE_AA)
            cv2.line(dashboard, (legend_rect[0] + 92, legend_rect[1] + 22),
                     (legend_rect[0] + 116, legend_rect[1] + 22),
                     (60, 130, 220), 2, cv2.LINE_AA)
            cv2.putText(dashboard, 'Path', (legend_rect[0] + 124, legend_rect[1] + 27),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.37, TEXT_MUTED, 1, cv2.LINE_AA)
            cv2.circle(dashboard, (legend_rect[0] + 20, legend_rect[1] + 50), 4,
                       (45, 158, 230), -1, cv2.LINE_AA)
            cv2.putText(dashboard, 'Marker', (legend_rect[0] + 34, legend_rect[1] + 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.37, TEXT_MUTED, 1, cv2.LINE_AA)

            # ================= SIDE PANEL =================
            side_bottom = max(
                MAP_Y + MAP_VIEW_HEIGHT + SLIDER_H,
                CAM1_VIEW_BUTTON[3] + UI_PAD
            )
            _draw_round_rect(dashboard,
                             (SIDE_X, SIDE_Y, SIDE_X + SIDE_PANEL_WIDTH, side_bottom),
                             PANEL_BG, 18, border=CARD_EDGE)

            connected = ser is not None and getattr(ser, 'is_open', False)
            r8_send_on = send_robot8_enabled.is_set()
            r7_send_on = send_robot7_enabled.is_set()
            r3_send_on = send_robot3_enabled.is_set()
            r29_send_on = send_robot29_enabled.is_set()
            conn_text = f'CONNECTED  {com_port}' if connected else 'DISCONNECTED'
            conn_color = GREEN if connected else RED

            # Event log theo thay đổi trạng thái.
            if last_logged_connected is None or connected != last_logged_connected:
                _add_event('ESP32 connected' if connected else 'ESP32 disconnected',
                           GREEN if connected else RED)
                last_logged_connected = connected
            send8_state_now = send_robot8_enabled.is_set()
            if last_logged_send8_enabled is None or send8_state_now != last_logged_send8_enabled:
                _add_event(
                    f'R{robot_id} send ' + ('started' if send8_state_now else 'stopped'),
                    GREEN if send8_state_now else TEXT_MUTED
                )
                last_logged_send8_enabled = send8_state_now

            send7_state_now = send_robot7_enabled.is_set()
            if last_logged_send7_enabled is None or send7_state_now != last_logged_send7_enabled:
                _add_event(
                    f'R{robot2_id} send ' + ('started' if send7_state_now else 'stopped'),
                    GREEN if send7_state_now else TEXT_MUTED
                )
                last_logged_send7_enabled = send7_state_now

            send3_state_now = send_robot3_enabled.is_set()
            if last_logged_send3_enabled is None or send3_state_now != last_logged_send3_enabled:
                _add_event(
                    f'R{robot3_id} send ' + ('started' if send3_state_now else 'stopped'),
                    GREEN if send3_state_now else TEXT_MUTED
                )
                last_logged_send3_enabled = send3_state_now

            send29_state_now = send_robot29_enabled.is_set()
            if last_logged_send29_enabled is None or send29_state_now != last_logged_send29_enabled:
                _add_event(
                    f'R{robot4_id} send ' + ('started' if send29_state_now else 'stopped'),
                    GREEN if send29_state_now else TEXT_MUTED
                )
                last_logged_send29_enabled = send29_state_now
            if last_logged_source is None or robot_source_label != last_logged_source:
                if robot_source_label != 'NONE':
                    _add_event(f'Source -> {robot_source_label}', CYAN)
                last_logged_source = robot_source_label
            if last_logged_robot_visible is None or robot_visible != last_logged_robot_visible:
                _add_event('Robot detected' if robot_visible else 'Searching robot',
                           GREEN if robot_visible else ORANGE)
                last_logged_robot_visible = robot_visible

            # SIDE PANEL - 4 ROBOT

            # ---------- SYSTEM STATUS ----------
            status_rect = (SIDE_X + 14, SIDE_Y + 14, SIDE_X + 376, SIDE_Y + 112)
            _draw_round_rect(dashboard, status_rect, CARD_BG, 14, border=CARD_EDGE)

            cv2.putText(dashboard, 'SYSTEM STATUS', (SIDE_X + 28, SIDE_Y + 36),
                        cv2.FONT_HERSHEY_DUPLEX, 0.52, TEXT_MAIN, 1, cv2.LINE_AA)
            _draw_status_dot(dashboard, (SIDE_X + 345, SIDE_Y + 31),
                             GREEN if connected else RED)

            cv2.putText(dashboard, 'ESP32', (SIDE_X + 28, SIDE_Y + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, TEXT_SOFT, 1, cv2.LINE_AA)
            _put_fit_text(dashboard, conn_text, (SIDE_X + 100, SIDE_Y + 60), 240,
                          cv2.FONT_HERSHEY_SIMPLEX, 0.39, conn_color, 1)

            cv2.putText(dashboard, 'SEND', (SIDE_X + 28, SIDE_Y + 82),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, TEXT_SOFT, 1, cv2.LINE_AA)
            send_status = (
                f'R8:{"ON" if r8_send_on else "OFF"}   '
                f'R7:{"ON" if r7_send_on else "OFF"}   '
                f'R3:{"ON" if r3_send_on else "OFF"}   '
                f'R29:{"ON" if r29_send_on else "OFF"}'
            )
            _put_fit_text(dashboard, send_status, (SIDE_X + 82, SIDE_Y + 82), 270,
                          cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                          GREEN if (r8_send_on or r7_send_on or r3_send_on or r29_send_on) else TEXT_MUTED, 1)

            cv2.putText(dashboard, 'FPS', (SIDE_X + 28, SIDE_Y + 104),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, TEXT_SOFT, 1, cv2.LINE_AA)
            cv2.putText(dashboard, f'{fps:4.1f}', (SIDE_X + 100, SIDE_Y + 104),
                        cv2.FONT_HERSHEY_DUPLEX, 0.40, CYAN, 1, cv2.LINE_AA)

            # ---------- ROBOTS POSITION ----------
            pos_rect = (SIDE_X + 14, SIDE_Y + 120, SIDE_X + 376, SIDE_Y + 250)
            _draw_round_rect(dashboard, pos_rect, CARD_BG_ALT, 14, border=CARD_EDGE)
            cv2.putText(dashboard, 'ROBOTS POSITION', (SIDE_X + 28, SIDE_Y + 142),
                        cv2.FONT_HERSHEY_DUPLEX, 0.42, TEXT_MAIN, 1, cv2.LINE_AA)

            r1_x = '--' if ui_x is None else f'{ui_x:.1f}'
            r1_y = '--' if ui_y is None else f'{ui_y:.1f}'
            r1_a = '--' if robot_filtered_angle is None else f'{robot_filtered_angle:.5f}'
            r2_x = '--' if ui_x2 is None else f'{ui_x2:.1f}'
            r2_y = '--' if ui_y2 is None else f'{ui_y2:.1f}'
            r2_a = '--' if robot2_filtered_angle is None else f'{robot2_filtered_angle:.5f}'
            r3_x = '--' if ui_x3 is None else f'{ui_x3:.1f}'
            r3_y = '--' if ui_y3 is None else f'{ui_y3:.1f}'
            r3_a = '--' if robot3_filtered_angle is None else f'{robot3_filtered_angle:.5f}'
            r4_x = '--' if ui_x4 is None else f'{ui_x4:.1f}'
            r4_y = '--' if ui_y4 is None else f'{ui_y4:.1f}'
            r4_a = '--' if robot4_filtered_angle is None else f'{robot4_filtered_angle:.5f}'

            robot_rows = [
                (robot_id, r1_x, r1_y, r1_a, GREEN, SIDE_Y + 168),
                (robot2_id, r2_x, r2_y, r2_a, CYAN, SIDE_Y + 192),
                (robot3_id, r3_x, r3_y, r3_a, (200, 120, 230), SIDE_Y + 216),
                (robot4_id, r4_x, r4_y, r4_a, (80, 180, 180), SIDE_Y + 240),
            ]
            for rid, xx, yy, aa, color, row_y in robot_rows:
                cv2.putText(dashboard, f'R{rid}', (SIDE_X + 28, row_y),
                            cv2.FONT_HERSHEY_DUPLEX, 0.40, color, 1, cv2.LINE_AA)
                _put_fit_text(
                    dashboard,
                    f'X {xx}   Y {yy}   A {aa} rad',
                    (SIDE_X + 70, row_y),
                    282,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.37,
                    TEXT_MAIN,
                    1
                )

            # ---------- CONTROL BUTTONS ----------
            def _draw_button(rect, label, fill, active=False):
                base_color = fill if active else tuple(max(0, int(c * 0.78)) for c in fill)
                _draw_round_rect(dashboard, rect, base_color, 9, border=CARD_EDGE)
                x1, y1, x2, y2 = rect
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.38, 1)
                tx = x1 + (x2 - x1 - tw) // 2
                ty = y1 + (y2 - y1 + th) // 2
                cv2.putText(dashboard, label, (tx, ty),
                            cv2.FONT_HERSHEY_DUPLEX, 0.38, TEXT_MAIN, 1, cv2.LINE_AA)

            _draw_button(R8_START_BUTTON, f'R{robot_id} START', GREEN, send_robot8_enabled.is_set())
            _draw_button(R8_STOP_BUTTON, f'R{robot_id} STOP', RED, not send_robot8_enabled.is_set())
            _draw_button(R8_CLEAR_BUTTON, f'R{robot_id} CLR PATH', SLATE)

            _draw_button(R7_START_BUTTON, f'R{robot2_id} START', GREEN, send_robot7_enabled.is_set())
            _draw_button(R7_STOP_BUTTON, f'R{robot2_id} STOP', RED, not send_robot7_enabled.is_set())
            _draw_button(R7_CLEAR_BUTTON, f'R{robot2_id} CLR PATH', SLATE)

            _draw_button(R3_START_BUTTON, f'R{robot3_id} START', GREEN, send_robot3_enabled.is_set())
            _draw_button(R3_STOP_BUTTON, f'R{robot3_id} STOP', RED, not send_robot3_enabled.is_set())
            _draw_button(R3_CLEAR_BUTTON, f'R{robot3_id} CLR PATH', SLATE)

            _draw_button(R29_START_BUTTON, f'R{robot4_id} START', GREEN, send_robot29_enabled.is_set())
            _draw_button(R29_STOP_BUTTON, f'R{robot4_id} STOP', RED, not send_robot29_enabled.is_set())
            _draw_button(R29_CLEAR_BUTTON, f'R{robot4_id} CLR PATH', SLATE)

            # ---------- RUN METRICS ----------
            metrics_rect = (SIDE_X + 14, SIDE_Y + 414, SIDE_X + 376, SIDE_Y + 542)
            _draw_round_rect(dashboard, metrics_rect, CARD_BG, 14, border=CARD_EDGE)
            cv2.putText(dashboard, 'RUN METRICS', (SIDE_X + 28, SIDE_Y + 437),
                        cv2.FONT_HERSHEY_DUPLEX, 0.42, TEXT_MAIN, 1, cv2.LINE_AA)

            cv2.putText(dashboard, 'ROBOT', (SIDE_X + 28, SIDE_Y + 458),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.27, TEXT_SOFT, 1, cv2.LINE_AA)
            cv2.putText(dashboard, 'DIST', (SIDE_X + 90, SIDE_Y + 458),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.27, TEXT_SOFT, 1, cv2.LINE_AA)
            cv2.putText(dashboard, 'SPEED', (SIDE_X + 180, SIDE_Y + 458),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.27, TEXT_SOFT, 1, cv2.LINE_AA)
            cv2.putText(dashboard, 'TIME', (SIDE_X + 292, SIDE_Y + 458),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.27, TEXT_SOFT, 1, cv2.LINE_AA)

            def _format_distance(dist_cm):
                return f'{dist_cm / 100.0:.2f} m' if dist_cm >= 100.0 else f'{dist_cm:.1f} cm'

            def _metric_speed(speed_value, last_motion):
                if last_motion is None or (now_ui - last_motion) > 0.85:
                    return 0.0
                return speed_value

            metrics_rows = [
                (robot_id, GREEN, robot_total_distance_cm,
                 _metric_speed(robot_speed_cm_s, robot_last_motion_time),
                 get_robot_run_seconds(robot_id), SIDE_Y + 482),
                (robot2_id, CYAN, robot2_total_distance_cm,
                 _metric_speed(robot2_speed_cm_s, robot2_last_motion_time),
                 get_robot_run_seconds(robot2_id), SIDE_Y + 501),
                (robot3_id, (200, 120, 230), robot3_total_distance_cm,
                 _metric_speed(robot3_speed_cm_s, robot3_last_motion_time),
                 get_robot_run_seconds(robot3_id), SIDE_Y + 520),
                (robot4_id, (80, 180, 180), robot4_total_distance_cm,
                 _metric_speed(robot4_speed_cm_s, robot4_last_motion_time),
                 get_robot_run_seconds(robot4_id), SIDE_Y + 539),
            ]

            for rid, color, dist_value, speed_value, run_sec, row_y in metrics_rows:
                cv2.putText(dashboard, f'R{rid}', (SIDE_X + 30, row_y),
                            cv2.FONT_HERSHEY_DUPLEX, 0.35, color, 1, cv2.LINE_AA)
                _put_fit_text(dashboard, _format_distance(dist_value),
                              (SIDE_X + 80, row_y), 83,
                              cv2.FONT_HERSHEY_DUPLEX, 0.31, TEXT_MAIN, 1)
                _put_fit_text(dashboard, f'{speed_value:.1f} cm/s',
                              (SIDE_X + 170, row_y), 105,
                              cv2.FONT_HERSHEY_DUPLEX, 0.31, TEXT_MAIN, 1)
                _put_fit_text(dashboard, _format_elapsed(run_sec),
                              (SIDE_X + 286, row_y), 72,
                              cv2.FONT_HERSHEY_DUPLEX, 0.31, TEXT_MAIN, 1)

            # ---------- EVENT LOG ----------
            log_rect = (SIDE_X + 14, SIDE_Y + 550, SIDE_X + 376, SIDE_Y + 618)
            _draw_round_rect(dashboard, log_rect, CARD_BG, 14, border=CARD_EDGE)
            cv2.putText(dashboard, 'EVENT LOG', (SIDE_X + 28, SIDE_Y + 572),
                        cv2.FONT_HERSHEY_DUPLEX, 0.40, TEXT_MAIN, 1, cv2.LINE_AA)

            recent_events = event_log[-2:]
            for row_idx, (stamp, message, event_color) in enumerate(recent_events):
                yy = SIDE_Y + 594 + row_idx * 17
                cv2.putText(dashboard, stamp, (SIDE_X + 28, yy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.30, TEXT_SOFT, 1, cv2.LINE_AA)
                _put_fit_text(dashboard, message, (SIDE_X + 95, yy), 250,
                              cv2.FONT_HERSHEY_SIMPLEX, 0.32, event_color, 1)

            # ---------- TARGET WAYPOINTS ----------
            target_card = (
                SIDE_X + 14,
                SIDE_Y + 626,
                SIDE_X + 376,
                SIDE_Y + 758
            )
            _draw_round_rect(
                dashboard,
                target_card,
                CARD_BG,
                14,
                border=CARD_EDGE
            )

            selected_target_robot = ui_state['target_robot']
            selected_points = target_waypoints[selected_target_robot]

            cv2.putText(
                dashboard,
                f'TARGET POINTS - R{selected_target_robot}: {len(selected_points)}',
                (SIDE_X + 28, SIDE_Y + 648),
                cv2.FONT_HERSHEY_DUPLEX,
                0.40,
                TEXT_MAIN,
                1,
                cv2.LINE_AA
            )

            _draw_button(
                TARGET_R8_BUTTON,
                'R8 TARGET',
                GREEN,
                selected_target_robot == robot_id
            )

            _draw_button(
                TARGET_R7_BUTTON,
                'R7 TARGET',
                CYAN,
                selected_target_robot == robot2_id
            )

            _draw_button(
                TARGET_R3_BUTTON,
                'R3 TARGET',
                (200, 120, 230),
                selected_target_robot == robot3_id
            )

            _draw_button(
                TARGET_R29_BUTTON,
                'R29 TARGET',
                (80, 180, 180),
                selected_target_robot == robot4_id
            )

            _draw_button(
                TARGET_SEND_BUTTON,
                'SEND 1 PACKET',
                GREEN,
                False
            )

            _draw_button(
                TARGET_UNDO_BUTTON,
                'UNDO',
                BLUE,
                False
            )

            _draw_button(
                TARGET_CLEAR_BUTTON,
                'CLEAR TARGET',
                RED,
                False
            )

            # ---------- CAMERA VIEW BUTTONS ----------
            cam1_active = ui_state['camera_view'] == 'cam1'
            cam2_active = ui_state['camera_view'] == 'cam2'

            _draw_button(
                CAM1_VIEW_BUTTON,
                'CAM 1 HIDE' if cam1_active else 'CAM 1 VIEW',
                BLUE,
                cam1_active
            )

            _draw_button(
                CAM2_VIEW_BUTTON,
                'CAM 2 HIDE' if cam2_active else 'CAM 2 VIEW',
                CYAN,
                cam2_active
            )

            # Preview camera mở ở WINDOW riêng, không chiếm diện tích dashboard.
            selected_camera = ui_state['camera_view']

            if selected_camera is not None:
                if now_ui - last_preview_update >= PREVIEW_INTERVAL:
                    last_preview_update = now_ui

                    selected_frame = (
                        img if selected_camera == 'cam1' else img2
                    )

                    camera_preview_cache = _fit_preview(
                        selected_frame,
                        width=640,
                        height=360
                    )

                    if selected_camera == 'cam1':
                        cv2.imshow(CAM1_WINDOW, camera_preview_cache)

                        try:
                            cv2.destroyWindow(CAM2_WINDOW)
                        except cv2.error:
                            pass
                    else:
                        cv2.imshow(CAM2_WINDOW, camera_preview_cache)

                        try:
                            cv2.destroyWindow(CAM1_WINDOW)
                        except cv2.error:
                            pass
            else:
                try:
                    cv2.destroyWindow(CAM1_WINDOW)
                except cv2.error:
                    pass

                try:
                    cv2.destroyWindow(CAM2_WINDOW)
                except cv2.error:
                    pass

            # ================= SLIDER / FOOTER =================
            track_w, knob_w, travel = _slider_geometry()
            cv2.line(
                dashboard,
                (SLIDER_LEFT, SLIDER_Y),
                (SLIDER_RIGHT, SLIDER_Y),
                CARD_EDGE,
                4,
                cv2.LINE_AA
            )

            src_w_float, _ = _view_source_size()
            current_left = (
                float(ui_state['center_x']) - src_w_float / 2.0
            )
            max_left = max(
                0.0,
                MAP_CANVAS_WIDTH - src_w_float
            )

            scroll_ratio = (
                0.0
                if max_left <= 1e-9
                else current_left / max_left
            )
            scroll_ratio = max(0.0, min(1.0, scroll_ratio))

            knob_left = int(
                SLIDER_LEFT + scroll_ratio * travel
            )

            _draw_round_rect(
                dashboard,
                (
                    knob_left,
                    SLIDER_Y - 8,
                    knob_left + knob_w,
                    SLIDER_Y + 8
                ),
                BLUE,
                8
            )

            left_world_x = (
                current_left - MAP_OFFSET_X
            ) / float(MAP_SCALE)

            right_world_x = (
                current_left + src_w_float - MAP_OFFSET_X
            ) / float(MAP_SCALE)

            footer_y = SLIDER_Y + 28

            cv2.putText(
                dashboard,
                (
                    f'VIEW X {left_world_x:.0f}..{right_world_x:.0f} cm'
                    f'   ZOOM {float(ui_state["zoom"]) * 100.0:.0f}%'
                ),
                (MAP_X + 6, footer_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                TEXT_MUTED,
                1,
                cv2.LINE_AA
            )

            cv2.putText(
                dashboard,
                'Left click: add target   |   Wheel: zoom   |   Right-drag: pan   |   F: fit map',
                (MAP_X + 390, footer_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                TEXT_SOFT,
                1,
                cv2.LINE_AA
            )

            cv2.imshow(DASHBOARD_WINDOW, dashboard)

        # XỬ LÝ PHÍM BẤM BÀN PHÍM
        key_press = cv2.waitKey(1) & 0xFF
        action = ui_state.get('action')
        ui_state['action'] = None

        # Phím E là lệnh calibration ưu tiên tuyệt đối. Nếu cùng frame có
        # mouse action cũ (đặc biệt CLEAR R8), bỏ action đó để E không bị nuốt.
        if key_press in (ord('e'), ord('E')):
            action = None
            ui_state['action'] = None

        # PHÍM ZOOM MAP
        if key_press in (ord('+'), ord('=')):
            _apply_zoom(float(ui_state['zoom']) * MAP_ZOOM_STEP)

        elif key_press in (ord('-'), ord('_')):
            _apply_zoom(float(ui_state['zoom']) / MAP_ZOOM_STEP)

        elif key_press == ord('0'):
            _reset_map_view()

        elif key_press in (ord('f'), ord('F')):
            _fit_map_width()

        # Waypoint shortcuts.
        elif key_press == ord('8'):
            ui_state['target_robot'] = robot_id

        elif key_press == ord('9'):
            ui_state['target_robot'] = robot2_id

        elif key_press == ord('3'):
            ui_state['target_robot'] = robot3_id

        elif key_press == ord('4'):
            ui_state['target_robot'] = robot4_id

        elif key_press in (10, 13):
            ui_state['action'] = 'send_target_path'
            action = 'send_target_path'

        elif key_press in (8, 127):
            ui_state['action'] = 'undo_target_point'
            action = 'undo_target_point'

        elif key_press in (ord('x'), ord('X')):
            ui_state['action'] = 'clear_target_path'
            action = 'clear_target_path'

        if action == 'quit':
            break

        # WAYPOINT ACTIONS
        if action == 'target_point_added':
            selected_rid = ui_state['target_robot']
            point_count = len(target_waypoints[selected_rid])

            if point_count > 0:
                px_target, py_target = target_waypoints[selected_rid][-1]
                _add_event(
                    f'R{selected_rid} P{point_count}: {px_target:.1f},{py_target:.1f}',
                    CYAN
                )

        elif action == 'undo_target_point':
            selected_rid = ui_state['target_robot']

            if target_waypoints[selected_rid]:
                target_waypoints[selected_rid].pop()
                _add_event(f'R{selected_rid} target undo', TEXT_MUTED)

        elif action == 'clear_target_path':
            # CLEAR TARGET:
            # - Xóa các điểm đích đã chấm của robot đang chọn.
            # - Gửi WPCLR# tới đúng robot để Arduino Mega xóa target cũ.
            # - KHÔNG xóa đường robot đã chạy trên map.
            selected_rid = ui_state['target_robot']
            target_waypoints[selected_rid].clear()

            send_waypoint_clear(selected_rid)
            _add_event(f'R{selected_rid} target points cleared', RED)

        elif action == 'send_target_path':
            selected_rid = ui_state['target_robot']
            points_to_send = list(target_waypoints[selected_rid])

            if points_to_send:
                if send_waypoint_path(selected_rid, points_to_send):
                    _add_event(
                        f'R{selected_rid} sent 1 packet: {len(points_to_send)} pts',
                        GREEN
                    )
            else:
                _add_event(
                    f'R{selected_rid} target path empty',
                    ORANGE
                )

        # START / STOP RIÊNG TỪNG ROBOT
        if action == 'start_r8':
            if start_robot_output(robot_id):
                _add_event(f'R{robot_id} START sent', GREEN)
        elif action == 'stop_r8':
            stop_robot_output(robot_id)
            _add_event(f'R{robot_id} STOP sent', TEXT_MUTED)
        elif action == 'start_r7':
            if start_robot_output(robot2_id):
                _add_event(f'R{robot2_id} START sent', GREEN)
        elif action == 'stop_r7':
            stop_robot_output(robot2_id)
            _add_event(f'R{robot2_id} STOP sent', TEXT_MUTED)
        elif action == 'start_r3':
            if start_robot_output(robot3_id):
                _add_event(f'R{robot3_id} START sent', GREEN)
        elif action == 'stop_r3':
            stop_robot_output(robot3_id)
            _add_event(f'R{robot3_id} STOP sent', TEXT_MUTED)
        elif action == 'start_r29':
            if start_robot_output(robot4_id):
                _add_event(f'R{robot4_id} START sent', GREEN)
        elif action == 'stop_r29':
            stop_robot_output(robot4_id)
            _add_event(f'R{robot4_id} STOP sent', TEXT_MUTED)

        # CLEAR PATH CHỈ XÓA QUỸ ĐẠO / METRICS.
        # KHÁC HOÀN TOÀN VỚI CLEAR TARGET.
        #
        # CLEAR PATH TUYỆT ĐỐI KHÔNG:
        #   - gửi WPCLR#
        #   - clear send_robot*_enabled
        #   - gửi STOP# tới ESP32
        #
        # Như vậy robot đang START vẫn tiếp tục truyền sau khi bấm CLEAR.

        if action == 'clear_r8':
            # Ghi nhớ trạng thái truyền hiện tại để đảm bảo CLEAR không làm thay đổi.
            r8_was_sending = send_robot8_enabled.is_set()

            robot_path.clear()
            robot_total_distance_cm = 0.0
            robot_speed_cm_s = 0.0
            robot_last_motion_time = None

            map_with_paths = _rebuild_cached_paths()

            # CLEAR đường không reset trạng thái START/STOP.
            # Timer RUN vẫn tiếp tục nếu robot đang chạy.
            if r8_was_sending:
                send_robot8_enabled.set()

            _add_event(
                f'R{robot_id} path cleared - send unchanged',
                CYAN
            )
            _debug_print(
                f">>> DA XOA QUY DAO ROBOT ID {robot_id} "
                f"| SEND={'ON' if send_robot8_enabled.is_set() else 'OFF'}"
            )

        elif action == 'clear_r7':
            r7_was_sending = send_robot7_enabled.is_set()

            robot2_path.clear()
            robot2_total_distance_cm = 0.0
            robot2_speed_cm_s = 0.0
            robot2_last_motion_time = None

            map_with_paths = _rebuild_cached_paths()

            if r7_was_sending:
                send_robot7_enabled.set()

            _add_event(
                f'R{robot2_id} path cleared - send unchanged',
                CYAN
            )
            _debug_print(
                f">>> DA XOA QUY DAO ROBOT ID {robot2_id} "
                f"| SEND={'ON' if send_robot7_enabled.is_set() else 'OFF'}"
            )

        elif action == 'clear_r3':
            r3_was_sending = send_robot3_enabled.is_set()

            robot3_path.clear()
            robot3_total_distance_cm = 0.0
            robot3_speed_cm_s = 0.0
            robot3_last_motion_time = None

            map_with_paths = _rebuild_cached_paths()

            if r3_was_sending:
                send_robot3_enabled.set()

            _add_event(
                f'R{robot3_id} path cleared - send unchanged',
                (200, 120, 230)
            )
            _debug_print(
                f">>> DA XOA QUY DAO ROBOT ID {robot3_id} "
                f"| SEND={'ON' if send_robot3_enabled.is_set() else 'OFF'}"
            )


        elif action == 'clear_r29':
            r29_was_sending = send_robot29_enabled.is_set()

            robot4_path.clear()
            robot4_total_distance_cm = 0.0
            robot4_speed_cm_s = 0.0
            robot4_last_motion_time = None

            map_with_paths = _rebuild_cached_paths()

            if r29_was_sending:
                send_robot29_enabled.set()

            _add_event(
                f'R{robot4_id} path cleared - send unchanged',
                (80, 180, 180)
            )
            _debug_print(
                f">>> DA XOA QUY DAO ROBOT ID {robot4_id} "
                f"| SEND={'ON' if send_robot29_enabled.is_set() else 'OFF'}"
            )

        # Phím C: xóa cả bốn quỹ đạo nhưng vẫn giữ nguyên trạng thái truyền.
        elif key_press in (ord('c'), ord('C')):
            r8_was_sending = send_robot8_enabled.is_set()
            r7_was_sending = send_robot7_enabled.is_set()
            r3_was_sending = send_robot3_enabled.is_set()
            r29_was_sending = send_robot29_enabled.is_set()

            robot_path.clear()
            robot2_path.clear()
            robot3_path.clear()
            robot4_path.clear()

            map_with_paths = base_map.copy()

            robot_total_distance_cm = 0.0
            robot2_total_distance_cm = 0.0
            robot3_total_distance_cm = 0.0
            robot4_total_distance_cm = 0.0

            robot_speed_cm_s = 0.0
            robot2_speed_cm_s = 0.0
            robot3_speed_cm_s = 0.0
            robot4_speed_cm_s = 0.0

            robot_last_motion_time = None
            robot2_last_motion_time = None
            robot3_last_motion_time = None
            robot4_last_motion_time = None

            # CLEAR không làm thay đổi START/STOP.
            if r8_was_sending:
                send_robot8_enabled.set()
            if r7_was_sending:
                send_robot7_enabled.set()
            if r3_was_sending:
                send_robot3_enabled.set()
            if r29_was_sending:
                send_robot29_enabled.set()

            _add_event('All 4 paths cleared - send unchanged', CYAN)
            _debug_print(
                ">>> DA XOA QUY DAO CUA CA 4 ROBOT "
                f"| R8 SEND={'ON' if send_robot8_enabled.is_set() else 'OFF'} "
                f"| R7 SEND={'ON' if send_robot7_enabled.is_set() else 'OFF'} "
                f"| R3 SEND={'ON' if send_robot3_enabled.is_set() else 'OFF'} "
                f"| R29 SEND={'ON' if send_robot29_enabled.is_set() else 'OFF'}"
            )

        # E: CALIB CAM2 (UNDISTORTED, 60-FRAME MEDIAN, FULL ARUCO CORNERS)
        elif key_press in (ord('e'), ord('E')):
            # Khóa mouse action trong suốt thời gian lấy 60 frame.
            ui_state['calibrating_h2'] = True
            ui_state['action'] = None
            samples = {}

            # Thu 60 cặp frame nhưng không render/progress để không tăng tải GUI.
            # Mỗi sample giữ nguyên cả 4 corner của cùng một marker ở cả hai camera.
            for _ in range(H2_CALIB_COLLECT_FRAMES):
                ok_cal1, frame_cal1 = cam_stream.read()
                ok_cal2, frame_cal2 = cam_stream2.read()

                if (
                    not ok_cal1 or not ok_cal2
                    or frame_cal1 is None or frame_cal2 is None
                    or frame_cal1.size == 0 or frame_cal2.size == 0
                ):
                    time.sleep(H2_CALIB_FRAME_INTERVAL_SEC)
                    continue

                gray_cal1 = cv2.cvtColor(frame_cal1, cv2.COLOR_BGR2GRAY)
                gray_cal2 = cv2.cvtColor(frame_cal2, cv2.COLOR_BGR2GRAY)

                future_cal1 = aruco_detect_pool.submit(
                    cv2.aruco.detectMarkers,
                    gray_cal1,
                    arucoDict,
                    parameters=arucoParams
                )
                future_cal2 = aruco_detect_pool.submit(
                    cv2.aruco.detectMarkers,
                    gray_cal2,
                    arucoDict,
                    parameters=arucoParamsCam2
                )

                corners_cal1, ids_cal1, _ = future_cal1.result()
                corners_cal2, ids_cal2, _ = future_cal2.result()

                if ids_cal1 is None or ids_cal2 is None:
                    time.sleep(H2_CALIB_FRAME_INTERVAL_SEC)
                    continue

                ids1_flat = ids_cal1.flatten().tolist()
                ids2_flat = ids_cal2.flatten().tolist()
                common_ids = set(ids1_flat) & set(ids2_flat)

                for marker_id in common_ids:
                    # 4 robot không dùng làm mốc H2 trong vận hành.
                    if marker_id in ROBOT_IDS:
                        continue

                    idx1 = ids1_flat.index(marker_id)
                    idx2 = ids2_flat.index(marker_id)

                    corner1 = np.asarray(
                        corners_cal1[idx1], dtype=np.float64
                    ).reshape(4, 2)
                    corner2_und = undistort_marker_corners(
                        corners_cal2[idx2], K2, D2, newK2
                    )
                    if corner2_und is None:
                        continue
                    corner2_und = np.asarray(corner2_und, dtype=np.float64).reshape(4, 2)

                    if (
                        not np.all(np.isfinite(corner1))
                        or not np.all(np.isfinite(corner2_und))
                    ):
                        continue

                    marker_samples = samples.setdefault(
                        int(marker_id),
                        {'cam1_corners': [], 'cam2_corners': []}
                    )
                    # Hai mảng được append cùng lúc nên luôn giữ correspondence theo frame.
                    marker_samples['cam1_corners'].append(corner1)
                    marker_samples['cam2_corners'].append(corner2_und)

                time.sleep(H2_CALIB_FRAME_INTERVAL_SEC)

            valid_ids = sorted(
                marker_id
                for marker_id, data in samples.items()
                if min(
                    len(data['cam1_corners']),
                    len(data['cam2_corners'])
                ) >= H2_CALIB_MIN_VALID_SAMPLES
            )

            # Vẫn yêu cầu tối thiểu 4 MARKER chung để vùng ghép đủ rộng.
            # Mỗi marker đóng góp 4 corner => 4/5/6 marker = 16/20/24 correspondence.
            if len(valid_ids) < 4:
                _add_event(
                    f'Calib failed: {len(valid_ids)} valid markers',
                    RED
                )
                _calib_print(
                    "\n========== CALIB H2 FULL-CORNER =========="
                )
                _calib_print(
                    f"FAILED: chỉ có {len(valid_ids)} marker đủ "
                    f">= {H2_CALIB_MIN_VALID_SAMPLES}/"
                    f"{H2_CALIB_COLLECT_FRAMES} mẫu."
                )
                _calib_print("===========================================\n")
            else:
                pts_cam2_und = []
                pts_cam1_raw = []

                for marker_id in valid_ids:
                    cam1_samples = np.asarray(
                        samples[marker_id]['cam1_corners'], dtype=np.float64
                    )
                    cam2_samples = np.asarray(
                        samples[marker_id]['cam2_corners'], dtype=np.float64
                    )

                    # Median THEO TỪNG CORNER qua 60 frame, không dùng center nữa.
                    # Shape sau median: (4, 2), giữ đúng thứ tự corner ArUco.
                    corners1_median = np.median(cam1_samples, axis=0)
                    corners2_median = np.median(cam2_samples, axis=0)

                    pts_cam1_raw.extend(corners1_median.tolist())
                    pts_cam2_und.extend(corners2_median.tolist())

                pts_src = np.asarray(pts_cam2_und, dtype=np.float64)
                pts_dst = np.asarray(pts_cam1_raw, dtype=np.float64)

                H_candidate, mask, metrics, reason = estimate_intercamera_homography_checked(
                    pts_src,
                    pts_dst,
                    matrix_cam1_to_real
                )

                if H_candidate is None:
                    # Hiện chẩn đoán trực tiếp trên 2 dòng Event Log, không chỉ console.
                    if metrics:
                        inliers = metrics.get('inliers', 0)
                        total = metrics.get('total', len(pts_src))
                        _add_event(
                            f'H2 FAIL {inliers}/{total}C inlier',
                            RED
                        )

                        px_rmse = metrics.get('pixel_rmse_px')
                        cm_rmse = metrics.get('world_rmse_cm')
                        if px_rmse is not None or cm_rmse is not None:
                            px_text = '--' if px_rmse is None else f'{px_rmse:.2f}px'
                            cm_text = '--' if cm_rmse is None else f'{cm_rmse:.2f}cm'
                            _add_event(
                                f'RMSE px={px_text} world={cm_text}',
                                RED
                            )
                        else:
                            _add_event('Calib Cam2 rejected', RED)
                    else:
                        _add_event('Calib Cam2 rejected', RED)

                    _calib_print("\n========== CALIB H2 FULL-CORNER ==========")
                    _calib_print(f"STATUS      : FAILED")
                    _calib_print(f"MARKERS     : {len(valid_ids)} -> {valid_ids}")
                    _calib_print(f"REASON      : {reason}")
                    if metrics:
                        _calib_print(
                            f"INLIERS     : {metrics.get('inliers', 0)}/"
                            f"{metrics.get('total', len(pts_src))} corners"
                        )
                        px_rmse = metrics.get('pixel_rmse_px')
                        cm_rmse = metrics.get('world_rmse_cm')
                        if px_rmse is not None:
                            _calib_print(f"PIXEL RMSE  : {px_rmse:.4f} px")
                        if cm_rmse is not None:
                            _calib_print(f"WORLD RMSE  : {cm_rmse:.4f} cm")
                    _calib_print("===========================================\n")
                else:
                    matrix_cam2_to_real = H_candidate

                    os.makedirs(os.path.dirname(HOMOGRAPHY_CAM2_FILE), exist_ok=True)
                    with open(HOMOGRAPHY_CAM2_FILE, 'wb') as f:
                        pickle.dump(matrix_cam2_to_real, f)

                    _calib_print("\n========== CALIB H2 FULL-CORNER ==========")
                    _calib_print(f"STATUS      : OK")
                    _calib_print(f"MARKERS     : {len(valid_ids)} -> {valid_ids}")
                    _calib_print(f"CORNERS     : {metrics['total']}")
                    _calib_print(
                        f"INLIERS     : {metrics['inliers']}/{metrics['total']} "
                        f"({100.0 * metrics['inliers'] / max(metrics['total'], 1):.1f}%)"
                    )
                    _calib_print(f"PIXEL RMSE  : {metrics['pixel_rmse_px']:.4f} px")
                    _calib_print(f"WORLD RMSE  : {metrics['world_rmse_cm']:.4f} cm")
                    # Event Log chỉ có 2 dòng hiển thị, nên dành trọn 2 dòng
                    # cho kết quả calibration: inlier và cả pixel/world RMSE.
                    _add_event(
                        f"H2 OK {len(valid_ids)}M {metrics['inliers']}/{metrics['total']}C",
                        GREEN
                    )
                    _add_event(
                        f"RMSE {metrics['pixel_rmse_px']:.2f}px | {metrics['world_rmse_cm']:.2f}cm",
                        GREEN
                    )
                    _calib_print(f"SAVED       : {HOMOGRAPHY_CAM2_FILE}")
                    _calib_print("===========================================\n")

            # Xóa mọi mouse action phát sinh trong lúc calibration và mở lại UI.
            ui_state['action'] = None
            ui_state['calibrating_h2'] = False
        # Q hoặc đóng cửa sổ để thoát.
        elif key_press in (ord('q'), ord('Q')):
            break

        if cv2.getWindowProperty(DASHBOARD_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break

    stop_all_robot_outputs()
    aruco_detect_pool.shutdown(wait=False)
    cam_stream.stop()
    cam_stream2.stop()

    try:
        cv2.destroyWindow(CAM1_WINDOW)
    except cv2.error:
        pass

    try:
        cv2.destroyWindow(CAM2_WINDOW)
    except cv2.error:
        pass

    cv2.destroyAllWindows()


# LUỒNG TRUYỀN CỐ ĐỊNH - ĐỘC LẬP FPS CAMERA
def send_data():
    global ser

    last_reconnect_try = 0.0
    next_send_time = time.perf_counter()

    while True:
        now_perf = time.perf_counter()

        if now_perf < next_send_time:
            time.sleep(
                min(
                    0.002,
                    next_send_time - now_perf
                )
            )
            continue

        # Tránh cộng dồn backlog nếu Windows/Python bị trễ một nhịp.
        next_send_time += TX_PERIOD_SEC
        if now_perf - next_send_time > TX_PERIOD_SEC:
            next_send_time = now_perf + TX_PERIOD_SEC

        # Snapshot cực nhanh, sau đó nhả lock ngay.
        with latest_packet_lock:
            packet8 = latest_robot_packets.get(robot_id)
            packet7 = latest_robot_packets.get(robot2_id)
            packet3 = latest_robot_packets.get(robot3_id)
            packet29 = latest_robot_packets.get(robot4_id)

        packets = []

        if (
            send_robot8_enabled.is_set()
            and packet8 is not None
        ):
            packets.append(packet8)

        if (
            send_robot7_enabled.is_set()
            and packet7 is not None
        ):
            packets.append(packet7)

        if (
            send_robot3_enabled.is_set()
            and packet3 is not None
        ):
            packets.append(packet3)

        if (
            send_robot29_enabled.is_set()
            and packet29 is not None
        ):
            packets.append(packet29)

        if packets:
            if ser is None or not ser.is_open:
                now = time.time()

                if now - last_reconnect_try >= 2.0:
                    last_reconnect_try = now
                    connect_serial()

                if ser is None or not ser.is_open:
                    continue

            try:
                payload = ''.join(packets).encode('ascii')

                with serial_lock:
                    ser.write(payload)

            except serial.SerialException as exc:
                _debug_print(f">>> MAT KET NOI ESP32: {exc}")

                with serial_lock:
                    try:
                        if ser is not None:
                            ser.close()
                    except Exception:
                        pass

                    ser = None


def _send_immediate_packet(packet_text):
    """Gửi một packet điều khiển ngay qua Serial, dùng chung serial_lock."""
    global ser

    if ser is None or not ser.is_open:
        connect_serial()

    if ser is None or not ser.is_open:
        return False

    try:
        with serial_lock:
            ser.write(packet_text.encode('ascii'))
        return True
    except serial.SerialException:
        return False


# ESP-NOW SINGLE-PACKET WAYPOINT
# Giữ dưới 240 byte để an toàn với ESP-NOW v1.0 (giới hạn 250 byte).
# Gateway sẽ bỏ "ID;" trước khi phát ESP-NOW, nên phép kiểm tra bên dưới
# kiểm tra đúng phần payload mà ESP32 receiver thực sự nhận.
ESPNOW_SAFE_WAYPOINT_BYTES = 240


def send_waypoint_clear(target_robot_id):
    """
    Xóa danh sách điểm đích cũ trên robot.

    Python -> Gateway:
        8;WPCLR#

    Gateway -> ESP32 receiver:
        WPCLR#
    """
    return _send_immediate_packet(
        f"{target_robot_id};WPCLR#"
    )


def _build_waypoint_payload(points):
    """
    Tạo MỘT payload duy nhất, KHÔNG gửi số lượng điểm:

        WPLIST;X1;Y1;X2;Y2;...;XN;YN#

    Ví dụ 3 điểm:
        WPLIST;100.0;80.0;120.0;60.0;140.0;80.0#

    Arduino Mega tự suy ra số điểm bằng cách đọc từng cặp X,Y
    cho tới khi hết packet.
    """
    fields = ["WPLIST"]

    for x_cm, y_cm in points:
        fields.append(f"{float(x_cm):.1f}")
        fields.append(f"{float(y_cm):.1f}")

    return ";".join(fields) + "#"


def send_waypoint_path(target_robot_id, points):
    """
    Gửi TOÀN BỘ danh sách điểm bằng đúng MỘT packet dài.

    Python -> Gateway:
        8;WPLIST;X1;Y1;X2;Y2;...#

    Gateway bỏ ID đầu và route theo MAC:
        WPLIST;X1;Y1;X2;Y2;...#

    Nếu payload vượt giới hạn an toàn 240 byte thì không gửi.
    """
    if not points:
        return False

    payload = _build_waypoint_payload(points)
    payload_size = len(payload.encode("ascii"))

    if payload_size > ESPNOW_SAFE_WAYPOINT_BYTES:
        _debug_print(
            f">>> WAYPOINT PACKET TOO LONG R{target_robot_id}: "
            f"{payload_size}/{ESPNOW_SAFE_WAYPOINT_BYTES} bytes"
        )
        return False

    full_packet = f"{target_robot_id};{payload}"

    return _send_immediate_packet(full_packet)


def start_robot_output(target_robot_id):
    """Bật truyền riêng cho một robot và gửi ID;START# tới gateway."""
    global ser

    event = _send_event_for_robot(target_robot_id)

    if ser is None or not ser.is_open:
        connect_serial()

    if ser is None or not ser.is_open:
        _debug_print(f">>> KHONG THE START R{target_robot_id}: CHUA KET NOI ESP32")
        return False

    command = f"{target_robot_id};START#".encode('ascii')

    try:
        with serial_lock:
            ser.write(command)
        event.set()

        timer_info = robot_run_timers[target_robot_id]
        if timer_info['started_at'] is None:
            timer_info['started_at'] = time.perf_counter()

        _debug_print(f">>> START SEND ROBOT ID {target_robot_id}")
        return True
    except serial.SerialException as exc:
        _debug_print(f">>> LOI GUI R{target_robot_id};START#: {exc}")
        return False


def stop_robot_output(target_robot_id):
    """Dừng truyền riêng cho một robot và gửi ID;STOP# tới gateway."""
    global ser

    event = _send_event_for_robot(target_robot_id)

    timer_info = robot_run_timers[target_robot_id]
    if timer_info['started_at'] is not None:
        timer_info['accumulated'] += max(
            0.0,
            time.perf_counter() - timer_info['started_at']
        )
        timer_info['started_at'] = None

    event.clear()

    # Không xóa packet tọa độ mới nhất.
    # Event OFF đảm bảo robot này không được send_data() truyền nữa.

    if ser is not None and ser.is_open:
        try:
            command = f"{target_robot_id};STOP#".encode('ascii')
            with serial_lock:
                ser.write(command)
        except serial.SerialException as exc:
            _debug_print(f">>> LOI GUI R{target_robot_id};STOP#: {exc}")
            return False

    _debug_print(f">>> STOP SEND ROBOT ID {target_robot_id}")
    return True


def stop_all_robot_outputs():
    """Dừng cả bốn robot khi thoát chương trình."""
    stop_robot_output(robot_id)
    stop_robot_output(robot2_id)
    stop_robot_output(robot3_id)
    stop_robot_output(robot4_id)
    return True

def main():
    # OpenCV/vision chạy ở main thread; Serial fixed-rate chạy nền độc lập.
    thread_send = threading.Thread(target=send_data, daemon=True)
    thread_send.start()
    handle_camera()


# Gọi hàm main kèm bộ lọc chặn chữ đỏ khi bấm Ctrl+C
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        _debug_print("\n>>> ĐÃ TẮT TOÀN BỘ CÁC LUỒNG VÀ CHƯƠNG TRÌNH AN TOÀN!")
