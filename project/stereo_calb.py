import cv2
import numpy as np
import glob

# ================= PARAMETERS =================
CHECKERBOARD = (6, 8)       # inner corners (cols, rows)
SQUARE_SIZE = 22.0          # mm

LEFT_VIDEO = "left.mp4"
RIGHT_VIDEO = "right.mp4"

FRAME_SKIP = 10             # take every 10th frame
# ============================================

# ---------- Prepare object points ----------
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []      # 3D points
imgpoints_l = []    # 2D points left cam
imgpoints_r = []    # 2D points right cam

# ---------- Open videos ----------
capL = cv2.VideoCapture(LEFT_VIDEO)
capR = cv2.VideoCapture(RIGHT_VIDEO)

frame_id = 0
print("[INFO] Detecting checkerboard corners...")

while True:
    retL, frameL = capL.read()
    retR, frameR = capR.read()
    if not retL or not retR:
        break

    if frame_id % FRAME_SKIP != 0:
        frame_id += 1
        continue

    grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

    retL_c, cornersL = cv2.findChessboardCorners(grayL, CHECKERBOARD, None)
    retR_c, cornersR = cv2.findChessboardCorners(grayR, CHECKERBOARD, None)

    if retL_c and retR_c:
        cornersL = cv2.cornerSubPix(
            grayL, cornersL, (11,11), (-1,-1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        cornersR = cv2.cornerSubPix(
            grayR, cornersR, (11,11), (-1,-1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )

        objpoints.append(objp)
        imgpoints_l.append(cornersL)
        imgpoints_r.append(cornersR)

        print(f"[OK] Pair {len(objpoints)}")

    frame_id += 1

capL.release()
capR.release()

print(f"[INFO] Valid stereo pairs: {len(objpoints)}")

# ---------- Image size ----------
h, w = grayL.shape[:2]

# ---------- Calibrate individual cameras ----------
print("[INFO] Calibrating LEFT camera...")
_, K1, D1, _, _ = cv2.calibrateCamera(
    objpoints, imgpoints_l, (w,h), None, None
)

print("[INFO] Calibrating RIGHT camera...")
_, K2, D2, _, _ = cv2.calibrateCamera(
    objpoints, imgpoints_r, (w,h), None, None
)

# ---------- Stereo calibration ----------
print("[INFO] Stereo calibration...")
_, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
    objpoints,
    imgpoints_l,
    imgpoints_r,
    K1, D1,
    K2, D2,
    (w,h),
    flags=cv2.CALIB_FIX_INTRINSIC
)

baseline = np.linalg.norm(T)

print("\n==============================")
print("Stereo calibration complete")
print(f"Baseline (mm): {baseline:.2f}")
print("Rotation R:\n", R)
print("Translation T:\n", T)
print("==============================")

# ---------- Save ----------
np.savez(
    "stereo_calibration.npz",
    K1=K1, D1=D1,
    K2=K2, D2=D2,
    R=R, T=T
)

print("[SAVED] stereo_calibration.npz")
