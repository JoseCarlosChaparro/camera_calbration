from ultralytics import YOLO
import cv2
import numpy as np

def scale_fisheye_calibration(K, D, original_size, new_size):
    """
    Ajusta la matriz intrínseca K para una nueva resolución.
    
    Params:
        K: matriz 3x3
        D: distorsión 4x1
        original_size: (w_original, h_original)
        new_size:      (w_new, h_new)
    """
    w0, h0 = original_size
    w1, h1 = new_size

    scale_x = w1 / w0
    scale_y = h1 / h0

    scaled_K = K.copy()
    scaled_K[0, 0] *= scale_x   # fx
    scaled_K[1, 1] *= scale_y   # fy
    scaled_K[0, 2] *= scale_x   # cx
    scaled_K[1, 2] *= scale_y   # cy

    # La distorsión NO se escala
    scaled_D = D.copy()

    return scaled_K, scaled_D

from ultralytics import YOLO
import cv2
import numpy as np

# Load calibration
data = np.load('fisheye_calibration.npz')
K = data['K']
D = data['D']
orig_w, orig_h = data['img_shape']

# Choose ANY resolution for tracking
new_w = 1280
new_h = 720

scaled_K, scaled_D = scale_fisheye_calibration(
    K, D,
    original_size=(orig_w, orig_h),
    new_size=(new_w, new_h)
)

# Open cam at desired resolution
cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, new_w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, new_h)

# Ensure one frame is grabbed
ret, frame = cap.read()
h, w = frame.shape[:2]

# Build maps for the new resolution
map1, map2 = cv2.fisheye.initUndistortRectifyMap(
    scaled_K, scaled_D, np.eye(3), scaled_K, (w, h), cv2.CV_16SC2
)

model = YOLO('best.pt')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    corrected = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)

    results = model.track(
        corrected,
        conf=0.3,
        tracker='bytetrack.yaml',
        persist=True
    )

    annotated = results[0].plot()
    cv2.imshow('tracking', annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
