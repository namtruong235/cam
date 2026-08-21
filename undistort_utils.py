from pathlib import Path
import cv2
import numpy as np

def save_calibration(
    path: Path,
    camera_matrix,
    distortion,
    new_camera_matrix,
    roi,
    image_size,
    rms,
    alpha,
    flags,
    per_view_errors,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        camera_matrix=np.asarray(camera_matrix, dtype=np.float64),
        distortion=np.asarray(distortion, dtype=np.float64),
        new_camera_matrix=np.asarray(new_camera_matrix, dtype=np.float64),
        roi=np.asarray(roi, dtype=np.int32),
        image_size=np.asarray(image_size, dtype=np.int32),
        rms=np.asarray([rms], dtype=np.float64),
        alpha=np.asarray([alpha], dtype=np.float64),
        flags=np.asarray([flags], dtype=np.int32),
        per_view_errors=np.asarray(per_view_errors, dtype=np.float64),
    )

def load_calibration(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy calibration: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {
            "camera_matrix": data["camera_matrix"].astype(np.float64),
            "distortion": data["distortion"].astype(np.float64),
            "new_camera_matrix": data["new_camera_matrix"].astype(np.float64),
            "roi": tuple(int(v) for v in data["roi"].reshape(-1)),
            "image_size": tuple(int(v) for v in data["image_size"].reshape(-1)),
            "rms": float(data["rms"].reshape(-1)[0]),
            "alpha": float(data["alpha"].reshape(-1)[0]),
            "flags": int(data["flags"].reshape(-1)[0]),
            "per_view_errors": (
                data["per_view_errors"].astype(np.float64)
                if "per_view_errors" in data.files else np.empty(0)
            ),
        }

def build_maps(camera_matrix, distortion, image_size, alpha):
    width, height = image_size
    new_k, roi = cv2.getOptimalNewCameraMatrix(
        np.asarray(camera_matrix, dtype=np.float64),
        np.asarray(distortion, dtype=np.float64),
        (width, height),
        float(alpha),
        (width, height),
    )
    map_x, map_y = cv2.initUndistortRectifyMap(
        np.asarray(camera_matrix, dtype=np.float64),
        np.asarray(distortion, dtype=np.float64),
        None,
        new_k,
        (width, height),
        cv2.CV_32FC1,
    )
    return map_x, map_y, new_k, tuple(int(v) for v in roi)

def undistort(frame, map_x, map_y):
    return cv2.remap(
        frame, map_x, map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )

def add_label(frame, text):
    result = frame.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 38), (0, 0, 0), -1)
    cv2.putText(
        result, text, (10, 27),
        cv2.FONT_HERSHEY_SIMPLEX, 0.68,
        (255, 255, 255), 2, cv2.LINE_AA,
    )
    return result
