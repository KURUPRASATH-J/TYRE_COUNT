import cv2
import numpy as np

# ================= SETTINGS =================
CAM1_VIDEO = "left.mp4"
CAM2_VIDEO = "right.mp4"
CALIB_FILE = "stereo_calibration.npz"
# ===========================================

# ---------- Load calibration ----------
calib = np.load(CALIB_FILE)
K1, D1 = calib["K1"], calib["D1"]
K2, D2 = calib["K2"], calib["D2"]
R, T = calib["R"], calib["T"]

# ---------- Videos ----------
cap1 = cv2.VideoCapture(CAM1_VIDEO)
cap2 = cv2.VideoCapture(CAM2_VIDEO)

w = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

# ---------- Stereo Rectification ----------
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    K1, D1,
    K2, D2,
    (w, h),
    R, T,
    flags=cv2.CALIB_ZERO_DISPARITY,
    alpha=0.7
)

map1x, map1y = cv2.initUndistortRectifyMap(
    K1, D1, R1, P1, (w, h), cv2.CV_32FC1
)
map2x, map2y = cv2.initUndistortRectifyMap(
    K2, D2, R2, P2, (w, h), cv2.CV_32FC1
)

points = []  # stores 4 clicks

def mouse_callback(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))

cv2.namedWindow("Stereo Click Height")
cv2.setMouseCallback("Stereo Click Height", mouse_callback)

print("""
CLICK ORDER:
1) TOP - Cam1
2) BOTTOM - Cam1
3) TOP - Cam2
4) BOTTOM - Cam2

Keys:
'n'  -> next frame
'ESC'-> exit
""")

while True:
    ret1, raw1 = cap1.read()
    ret2, raw2 = cap2.read()
    if not ret1 or not ret2:
        break

    points.clear()

    while True:
        disp1 = raw1.copy()
        disp2 = raw2.copy()

        # draw cam1 points
        if len(points) >= 1:
            cv2.circle(disp1, points[0], 6, (0,255,0), -1)
        if len(points) >= 2:
            cv2.circle(disp1, points[1], 6, (0,0,255), -1)
            cv2.line(disp1, points[0], points[1], (255,255,0), 2)

        # draw cam2 points
        if len(points) >= 3:
            cv2.circle(disp2, points[2], 6, (0,255,0), -1)
        if len(points) == 4:
            cv2.circle(disp2, points[3], 6, (0,0,255), -1)
            cv2.line(disp2, points[2], points[3], (255,255,0), 2)

            # ---- Rectify frames ----
            r1 = cv2.remap(raw1, map1x, map1y, cv2.INTER_LINEAR)
            r2 = cv2.remap(raw2, map2x, map2y, cv2.INTER_LINEAR)

            # ---- Triangulate ----
            pts1 = np.array([points[0], points[1]], dtype=np.float32).T
            pts2 = np.array([points[2], points[3]], dtype=np.float32).T

            pts4d = cv2.triangulatePoints(P1, P2, pts1, pts2)
            pts3d = pts4d[:3] / pts4d[3]

            # ✅ TRUE HEIGHT (vertical axis only)
            height_mm = abs(pts3d[1,0] - pts3d[1,1])

            cv2.putText(
                disp1,
                f"Height: {height_mm:.2f} mm",
                (30,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,255),
                2
            )

        combined = np.hstack((disp1, disp2))
        cv2.imshow("Stereo Click Height", combined)

        key = cv2.waitKey(1)
        if key == ord('n'):
            break
        if key == 27:
            cap1.release()
            cap2.release()
            cv2.destroyAllWindows()
            exit()

cap1.release()
cap2.release()
cv2.destroyAllWindows()
