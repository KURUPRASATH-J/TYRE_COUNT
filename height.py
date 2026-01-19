import cv2
import numpy as np
from ultralytics import YOLO
from statistics import median

# ================= SETTINGS =================
CAM1_VIDEO = "leftdetect.mp4"
CAM2_VIDEO = "rightdetct.mp4"
CALIB_FILE = "stereo_calibration.npz"
MODEL_PATH = "best.pt"

# Detection confidence thresholds (lower = more detections)
CONF_THRESH_CAM1 = 0.05     # Same threshold for both cameras
CONF_THRESH_CAM2 = 0.05     # Same threshold for both cameras

FRAME_STEP = 1
MAX_FRAMES = 100

TOP_CROP = 0.11
BOTTOM_CROP = 0.27

# Quality check thresholds - use same minimum for both
MIN_CONFIDENCE_CAM1 = 0.05  # Same quality bar for both cameras
MIN_CONFIDENCE_CAM2 = 0.05  # Same quality bar for both cameras
MIN_BOX_AREA = 2000         # Minimum detection box area
MAX_HEIGHT_VARIATION = 30   # Max mm deviation from median (for outlier detection)
MIN_BOX_WIDTH = 30          # Minimum box width in pixels
MIN_BOX_HEIGHT = 50         # Minimum box height in pixels
# ==========================================

# ---------- Load calibration ----------
calib = np.load(CALIB_FILE)
K1, D1 = calib["K1"], calib["D1"]
K2, D2 = calib["K2"], calib["D2"]
R, T = calib["R"], calib["T"]

P1 = K1 @ np.hstack((np.eye(3), np.zeros((3,1))))
P2 = K2 @ np.hstack((R, T))

# ---------- YOLO ----------
model = YOLO(MODEL_PATH)

# ---------- Videos ----------
cap1 = cv2.VideoCapture(CAM1_VIDEO)
cap2 = cv2.VideoCapture(CAM2_VIDEO)

w = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_video_frames = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap1.get(cv2.CAP_PROP_FPS)
fps = fps if fps > 0 else 20

out = cv2.VideoWriter(
    "height_verification.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w * 2, h)
)

heights = []
height_data = []  # Store detailed info for each measurement
frame_id = 0
frames_processed = 0
cam1_det_count = 0
cam2_det_count = 0
both_det_count = 0
quality_rejected = 0

print("[INFO] Generating verification video with quality checks...")
print(f"Total frames in video: {total_video_frames}")
print(f"Cam1 detection threshold: {CONF_THRESH_CAM1} (min quality: {MIN_CONFIDENCE_CAM1})")
print(f"Cam2 detection threshold: {CONF_THRESH_CAM2} (min quality: {MIN_CONFIDENCE_CAM2})")
print(f"Quality checks enabled\n")

while True:
    ret1, f1 = cap1.read()
    ret2, f2 = cap2.read()
    if not ret1 or not ret2:
        break

    frames_processed += 1
    
    # Store original frames BEFORE undistortion
    orig_f1 = f1.copy()
    orig_f2 = f2.copy()

    # Undistort for detection and measurement
    f1_undist = cv2.undistort(f1, K1, D1)
    f2_undist = cv2.undistort(f2, K2, D2)

    # Use separate confidence thresholds for each camera
    r1 = model(f1_undist, conf=CONF_THRESH_CAM1, verbose=False)[0]
    r2 = model(f2_undist, conf=CONF_THRESH_CAM2, verbose=False)[0]

    # Count detections
    has_det1 = len(r1.boxes) > 0
    has_det2 = len(r2.boxes) > 0
    
    if has_det1:
        cam1_det_count += 1
    if has_det2:
        cam2_det_count += 1
    if has_det1 and has_det2:
        both_det_count += 1

    # Add status text to original frames
    status1 = f"Cam1: {'OK' if has_det1 else 'NO DETECT'}"
    status2 = f"Cam2: {'OK' if has_det2 else 'NO DETECT'}"
    
    color1 = (0, 255, 0) if has_det1 else (0, 0, 255)
    color2 = (0, 255, 0) if has_det2 else (0, 0, 255)
    
    cv2.putText(orig_f1, status1, (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color1, 2)
    cv2.putText(orig_f2, status2, (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color2, 2)
    
    cv2.putText(orig_f1, f"Frame: {frame_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.putText(orig_f2, f"Frame: {frame_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    if not has_det1 or not has_det2:
        out.write(np.hstack((orig_f1, orig_f2)))
        frame_id += 1
        continue

    # Get detections
    b1 = r1.boxes.xyxy.cpu().numpy()[0]
    b2 = r2.boxes.xyxy.cpu().numpy()[0]
    
    # Get confidence scores
    conf1 = float(r1.boxes.conf.cpu().numpy()[0])
    conf2 = float(r2.boxes.conf.cpu().numpy()[0])

    x1,y1,x2,y2 = b1
    x1p,y1p,x2p,y2p = b2
    
    # Quality checks
    quality_pass = True
    reject_reason = ""
    
    # Check 1: Camera-specific confidence thresholds
    if conf1 < MIN_CONFIDENCE_CAM1:
        quality_pass = False
        reject_reason = f"Cam1 conf too low: {conf1:.2f} < {MIN_CONFIDENCE_CAM1}"
    
    if conf2 < MIN_CONFIDENCE_CAM2:
        quality_pass = False
        reject_reason = f"Cam2 conf too low: {conf2:.2f} < {MIN_CONFIDENCE_CAM2}"
    
    # Check 2: Box sizes
    width1, height1 = x2-x1, y2-y1
    width2, height2 = x2p-x1p, y2p-y1p
    area1 = width1 * height1
    area2 = width2 * height2
    
    if area1 < MIN_BOX_AREA or area2 < MIN_BOX_AREA:
        quality_pass = False
        reject_reason = f"Box too small: {int(area1)}, {int(area2)}"
    
    if width1 < MIN_BOX_WIDTH or height1 < MIN_BOX_HEIGHT:
        quality_pass = False
        reject_reason = f"Cam1 box dim: {int(width1)}x{int(height1)}"
    
    # Check 3: Box aspect ratio (should be reasonable for vertical object)
    aspect1 = height1 / width1 if width1 > 0 else 0
    aspect2 = height2 / width2 if width2 > 0 else 0
    
    if aspect1 < 0.5 or aspect1 > 10:  # Unreasonable aspect ratio
        quality_pass = False
        reject_reason = f"Cam1 bad aspect: {aspect1:.2f}"
    
    # If quality checks fail, draw warning and skip
    if not quality_pass:
        quality_rejected += 1
        
        # Draw boxes in RED to show rejected detection
        cv2.rectangle(orig_f1, (int(x1),int(y1)), (int(x2),int(y2)), (0,0,255), 2)
        cv2.rectangle(orig_f2, (int(x1p),int(y1p)), (int(x2p),int(y2p)), (0,0,255), 2)
        
        # Show rejection reason
        cv2.putText(orig_f1, f"REJECTED: {reject_reason}", (30,60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        cv2.putText(orig_f1, f"Conf1: {conf1:.2f} Conf2: {conf2:.2f}", (30,85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        out.write(np.hstack((orig_f1, orig_f2)))
        frame_id += 1
        continue

    # Calculate height
    h1 = y2 - y1
    h2 = y2p - y1p

    top1 = ((x1+x2)/2, y1 + TOP_CROP*h1)
    bot1 = ((x1+x2)/2, y2 - BOTTOM_CROP*h1)

    top2 = ((x1p+x2p)/2, y1p + TOP_CROP*h2)
    bot2 = ((x1p+x2p)/2, y2p - BOTTOM_CROP*h2)

    pts1 = np.array([top1, bot1], dtype=np.float32).T
    pts2 = np.array([top2, bot2], dtype=np.float32).T

    pts4d = cv2.triangulatePoints(P1, P2, pts1, pts2)
    pts3d = pts4d[:3] / pts4d[3]

    height_mm = np.linalg.norm(pts3d[:,0] - pts3d[:,1])
    
    # Check 4: Outlier detection (after we have some measurements)
    if len(heights) >= 5:
        current_median = median(heights)
        if abs(height_mm - current_median) > MAX_HEIGHT_VARIATION:
            quality_rejected += 1
            
            cv2.rectangle(orig_f1, (int(x1),int(y1)), (int(x2),int(y2)), (0,165,255), 2)
            cv2.rectangle(orig_f2, (int(x1p),int(y1p)), (int(x2p),int(y2p)), (0,165,255), 2)
            
            cv2.putText(orig_f1, f"OUTLIER: {height_mm:.1f}mm vs median {current_median:.1f}mm", 
                       (30,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 2)
            
            out.write(np.hstack((orig_f1, orig_f2)))
            frame_id += 1
            continue
    
    heights.append(height_mm)
    height_data.append({
        'frame': frame_id,
        'height': height_mm,
        'conf1': conf1,
        'conf2': conf2,
        'area1': area1,
        'area2': area2
    })

    # ---------- DRAW ON ORIGINAL FRAMES (GREEN = GOOD) ----------
    cx1 = int((x1 + x2) / 2)
    cx2 = int((x1p + x2p) / 2)

    cv2.rectangle(orig_f1, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 3)
    cv2.rectangle(orig_f2, (int(x1p),int(y1p)), (int(x2p),int(y2p)), (0,255,0), 3)

    cv2.line(orig_f1, (cx1, int(top1[1])), (cx1, int(bot1[1])), (255,255,0), 3)
    cv2.line(orig_f2, (cx2, int(top2[1])), (cx2, int(bot2[1])), (255,255,0), 3)
    
    # Draw measurement points
    cv2.circle(orig_f1, (cx1, int(top1[1])), 5, (0,255,0), -1)
    cv2.circle(orig_f1, (cx1, int(bot1[1])), 5, (0,0,255), -1)
    cv2.circle(orig_f2, (cx2, int(top2[1])), 5, (0,255,0), -1)
    cv2.circle(orig_f2, (cx2, int(bot2[1])), 5, (0,0,255), -1)

    # Show measurement info
    cv2.putText(orig_f1, f"Height: {height_mm:.1f} mm", (30,60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
    cv2.putText(orig_f1, f"Conf: {conf1:.2f} | Box: {int(width1)}x{int(height1)}", (30,95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(orig_f2, f"Conf: {conf2:.2f} | Box: {int(width2)}x{int(height2)}", (30,95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    
    # Measurement count with quality indicator
    cv2.putText(orig_f1, f"Valid: {len(heights)}/{MAX_FRAMES} | Rejected: {quality_rejected}", 
               (30,125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)

    out.write(np.hstack((orig_f1, orig_f2)))
    
    print(f"✓ Frame {frame_id}: H={height_mm:.1f}mm, C1={conf1:.2f}, C2={conf2:.2f}, Box1={int(width1)}x{int(height1)}")

    if len(heights) >= MAX_FRAMES:
        print(f"\n✓ Reached {MAX_FRAMES} valid measurements!")
        break

    frame_id += 1

cap1.release()
cap2.release()
out.release()

print("\n" + "="*70)
print("DETECTION QUALITY REPORT:")
print(f"  Frames processed: {frames_processed}")
print(f"  Both cameras detected: {both_det_count} frames")
print(f"  Quality rejected: {quality_rejected} frames")
print(f"  Valid measurements: {len(heights)} frames")
print(f"  Success rate: {len(heights)/both_det_count*100:.1f}% of detections passed quality checks")
print("="*70)

if heights:
    print(f"\n✓ FINAL HEIGHT = {median(heights):.2f} mm")
    print(f"  Mean: {np.mean(heights):.2f} mm")
    print(f"  Min-Max: {min(heights):.2f} - {max(heights):.2f} mm")
    print(f"  Std dev: {np.std(heights):.2f} mm")
    print(f"  Range: {max(heights) - min(heights):.2f} mm")
    
    # Show confidence statistics
    conf1_values = [d['conf1'] for d in height_data]
    conf2_values = [d['conf2'] for d in height_data]
    print(f"\n  Cam1 confidence: {np.mean(conf1_values):.3f} ± {np.std(conf1_values):.3f}")
    print(f"  Cam2 confidence: {np.mean(conf2_values):.3f} ± {np.std(conf2_values):.3f}")
    
    # Flag any concerning measurements
    low_conf1 = [d for d in height_data if d['conf1'] < 0.20]
    if low_conf1:
        print(f"\n  ⚠ {len(low_conf1)} measurements with Cam1 confidence < 0.20:")
        for d in low_conf1[:5]:  # Show first 5
            print(f"    Frame {d['frame']}: conf={d['conf1']:.2f}, height={d['height']:.1f}mm")
    else:
        print(f"\n  ✓ All Cam1 detections have good confidence (>0.20)")
    
    print(f"\n✓ Video saved: height_verification.mp4")
    print("  - GREEN boxes = accepted measurement")
    print("  - RED boxes = rejected (quality issues)")
    print("  - ORANGE boxes = outliers")
else:
    print("❌ No valid height measurements")
    
print("="*70)