#!/usr/bin/env python3
"""
side.py — complete, ready-to-run keypoint distance calculator for Ultralytics pose models.

Usage:
    python side.py --model side_view_model.pt --image cow.jpg

Features:
- Loads Ultralytics YOLO Pose model
- Extracts 12 named keypoints
- Computes pixel distances
- Computes cm/in distances if 3" sticker detected
- Saves annotated output as kp_overlay.jpg
"""

import argparse
import math
import cv2
import numpy as np
from ultralytics import YOLO


# -------------------- KEYPOINT DEFINITIONS --------------------

KP_NAMES = [
    "wither",
    "pinbone",
    "shoulderbone",
    "chest_top",
    "elbow",
    "body_girth_top",
    "rear_elbow",
    "spine_between_hips",
    "hoof",
    "belly_deepest_point",
    "hock",
    "hip_bone"
]
N_KP = len(KP_NAMES)

# 3-inch reference sticker size
STICKER_CM = 7.620 * 2.54 

# HSV detection range for green/yellow sticker (tune if needed)
HSV_LOWER = np.array([25, 60, 60])
HSV_UPPER = np.array([80, 255, 255])


# -------------------- STICKER SCALE DETECTION --------------------

def detect_sticker_scale(img):
    """Detect 3x3 inch sticker and compute cm-per-pixel scale."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    c = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    w, h = rect[1]
    sticker_px = max(w, h)

    if sticker_px < 1:
        return None

    scale = STICKER_CM / sticker_px
    return scale


# -------------------- ULTRALYTICS KEYPOINT EXTRACTION --------------------

def extract_kps(result):
    """
    Extract (x,y,conf) for 12 keypoints from YOLO Pose result.
    result.keypoints.xy: shape (num_instances, N_KP, 2)
    result.keypoints.conf: shape (num_instances, N_KP)
    """
    if result.keypoints is None:
        raise RuntimeError("Model returned no keypoints. Ensure your model is a YOLO Pose model.")

    xy = result.keypoints.xy[0]     # first detected animal
    conf = result.keypoints.conf[0] # confidence per keypoint

    if xy.shape[0] != N_KP:
        raise RuntimeError(f"Expected {N_KP} keypoints but model returned {xy.shape[0]}.")

    kps = np.hstack([xy, conf.reshape(-1, 1)])  # (12, 3)
    return kps


# -------------------- KEYPOINT DISTANCE CLASS --------------------

class KeypointDistance:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.kps = None
        self.scale = None  # cm per pixel

    def predict(self, img):
        results = self.model.predict(img, imgsz=640)
        r = results[0]
        self.kps = extract_kps(r)
        self.scale = detect_sticker_scale(img)
        return self.kps

    def get(self, name):
        idx = KP_NAMES.index(name)
        return self.kps[idx]

    def dist_between(self, name_a, name_b):
        xa, ya, _ = self.get(name_a)
        xb, yb, _ = self.get(name_b)

        px = math.dist((xa, ya), (xb, yb))
        cm = px * self.scale if self.scale else None
        inch = (cm / 2.54) if cm else None

        return {"pixels": px, "cm": cm, "inch": inch}


# -------------------- MAIN --------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print("Error: Could not load image.")
        return

    app = KeypointDistance(args.model)
    kps = app.predict(img)

    print("\nDetected Keypoints:")
    for name, (x, y, c) in zip(KP_NAMES, kps):
        print(f"{name:20s}  x={x:.1f}  y={y:.1f}  conf={c:.3f}")

    print("\nScale from sticker:", app.scale, "cm/px")

    # Example: measure wither to hip_bone
    d = app.dist_between("wither", "hip_bone")
    print("\nExample distance wither → hip_bone")
    print("Pixels:", d["pixels"])
    print("CM:", d["cm"])
    print("Inch:", d["inch"])

    # Draw overlay
    overlay = img.copy()
    for name, (x, y, c) in zip(KP_NAMES, kps):
        cv2.circle(overlay, (int(x), int(y)), 5, (0, 255, 255), -1)
        cv2.putText(overlay, name, (int(x)+5, int(y)-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)

    # Save final image
    # -----------------------------------------------
    # Draw a measurement line between two keypoints
    # -----------------------------------------------
    ptA_name = "wither"
    ptB_name = "pinbone"

    (xA, yA, _) = app.get(ptA_name)
    (xB, yB, _) = app.get(ptB_name)

    ptA = (int(xA), int(yA))
    ptB = (int(xB), int(yB))

    # draw line
    cv2.line(overlay, ptA, ptB, (0, 255, 0), 2)

    # calculate distance
    dist = app.dist_between(ptA_name, ptB_name)

    # choose label text
    if dist["cm"] is not None:
        label = f"{dist['cm']:.2f} cm"
    else:
        label = f"{dist['pixels']:.1f} px"

    # place label at midpoint
    mid = ( (ptA[0] + ptB[0]) // 2, (ptA[1] + ptB[1]) // 2 )
    cv2.putText(overlay, label, (mid[0] + 10, mid[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    print("\nDistance:", dist)

    # Save final annotated image
    cv2.imwrite("kp_overlay.jpg", overlay)
    print("\nSaved: kp_overlay.jpg with measurement line.")



if __name__ == "__main__":
    main()
