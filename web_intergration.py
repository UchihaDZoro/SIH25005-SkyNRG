#!/usr/bin/env python3
"""
udder_metrics_json.py

No GUI, no args. Reads keypoints from kp.json (preferred) or attempts to auto-detect
using 'udder_view_model.pt' (ultralytics) if present and an image file exists.
Reads an image (tries input.jpg / cow_udder.jpg / cow.jpg / udder.jpg).
Detects a sticker to derive cm/px (default sticker side = 21 cm).
Computes udder traits from keypoints pt_1..pt_8 and prints a single JSON object to stdout.

Output schema (example):
{
  "traits": [
    {
      "trait": "Teat pair 1 distance",
      "features": ["pt_1","pt_2"],
      "value_px": 45.12,
      "value_cm": 7.35,
      "score": null
    },
    ...
  ],
  "meta": {
    "image_used": "cow_udder.jpg",
    "kp_source": "kp.json",
    "scale_cm_per_px": 0.1632
  }
}
"""
import json
import os
import math
import cv2
import numpy as np

# ---------- CONFIG ----------
DISPLAY_SIZE = 640
KP_NAMES = ["pt_1","pt_2","pt_3","pt_4","pt_5","pt_6","pt_7","pt_8"]
STICKER_CM_DEFAULT = 21.0
HSV_LOWER = np.array([25,60,60])
HSV_UPPER = np.array([90,255,255])
MODEL_FILE = "udder_view_model.pt"
KP_FILE = "kp.json"

# ---------- helpers ----------
def dist_px(a,b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def letterbox_to_640(img):
    h,w = img.shape[:2]
    scale = DISPLAY_SIZE / max(h,w)
    nh, nw = int(h*scale), int(w*scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((DISPLAY_SIZE, DISPLAY_SIZE, 3), 114, dtype=np.uint8)
    top = (DISPLAY_SIZE - nh)//2; left = (DISPLAY_SIZE - nw)//2
    canvas[top:top+nh, left:left+nw] = resized
    return canvas, scale, left, top

def detect_sticker_cm_per_px(img_bgr, sticker_cm=STICKER_CM_DEFAULT, min_area=100):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
    k = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area:
        return None, None
    rect = cv2.minAreaRect(c)
    w,h = rect[1]
    px = max(w,h)
    if px < 1:
        return None, None
    box = cv2.boxPoints(rect).astype(int)
    return float(sticker_cm) / float(px), box

def spatial_order_points(pts):
    # deterministic: left->right then top->bottom
    return sorted(pts, key=lambda p: (p[0], p[1]))

def cm_or_deg_to_score(value, ranges):
    if value is None: return None
    for score, lo, hi in ranges:
        lo_ok = (lo is None) or (value >= lo)
        hi_ok = (hi is None) or (value <= hi)
        if lo_ok and hi_ok:
            return int(score)
    # clamp to ends
    if value is not None:
        if ranges[0][2] is not None and value < ranges[0][2]:
            return int(ranges[0][0])
        if ranges[-1][1] is not None and value > ranges[-1][1]:
            return int(ranges[-1][0])
    return None

# ----- scoring references (udders) based on the table you provided (Gir udder section) -----
UDDER_SCORES = {
  # rear udder height (vs vulva) mapping (cm)
  "rear_udder_height": [
    (1, None, 7.99), (2, 8.0, 9.99), (3, 10.0, 11.99),
    (4, 12.0, 13.99), (5, 14.0, 15.99), (6, 16.0, 17.99),
    (7, 18.0, 19.99), (8, 20.0, 21.99), (9, 22.0, None)
  ],
  # udder depth vs hock — we can't compute relative to hock with only teats so we skip scoring
}

# ---------- acquire data ----------
# find image
image_candidates = ["input.jpg","cow_udder.jpg","cow.jpg","udder.jpg"]
image_path = None
for fn in image_candidates:
    if os.path.exists(fn):
        image_path = fn
        break

# load image if available
img = None
if image_path:
    img_full = cv2.imread(image_path)
    if img_full is None:
        image_path = None
    else:
        img, _, _, _ = letterbox_to_640(img_full)

# load kp.json if present
kp_map = {}
if os.path.exists(KP_FILE):
    try:
        j = json.load(open(KP_FILE, "r"))
        # accept either {"pt_1":[x,y],...} or {"pt_1":{"x":..,"y":..}}
        for k,v in j.items():
            if isinstance(v, (list,tuple)) and len(v) >= 2:
                kp_map[k] = (float(v[0]), float(v[1]))
            elif isinstance(v, dict) and "x" in v and "y" in v:
                kp_map[k] = (float(v["x"]), float(v["y"]))
    except Exception:
        kp_map = {}

# attempt auto-detect with model if kp.json missing and model exists and image present
if not kp_map and image_path and os.path.exists(MODEL_FILE):
    try:
        from ultralytics import YOLO
        model = YOLO(MODEL_FILE)
        results = model.predict(img, imgsz=DISPLAY_SIZE)
        if len(results) > 0:
            r = results[0]
            det = getattr(r, "keypoints", None)
            if det is not None and getattr(det, "xy", None) is not None:
                # flatten detections then spatially order to fill pt_1..pt_8
                flat = []
                for i in range(len(det.xy)):
                    for p in det.xy[i]:
                        flat.append((float(p[0]), float(p[1])))
                # If there are >=8 points assign first 8 in detection order
                if len(flat) >= 8:
                    for i in range(8):
                        kp_map[KP_NAMES[i]] = flat[i]
                else:
                    # spatially order available points and assign sequentially
                    s = spatial_order_points(flat)
                    for i in range(min(len(s),8)):
                        kp_map[KP_NAMES[i]] = s[i]
    except Exception:
        # model not usable — leave kp_map empty
        pass

# ---------- compute scale ----------
scale_cm_per_px = None
sticker_box = None
if img is not None:
    sc_box = detect_sticker_cm_per_px(img, STICKER_CM_DEFAULT)
    if sc_box is not None:
        scale_cm_per_px, sticker_box = sc_box

# ---------- compute udder measurements ----------
# Convert kp_map into ordered list of points (None where missing)
kp = {k: kp_map.get(k) for k in KP_NAMES}

def safe_pair(a,b):
    pa = kp.get(a); pb = kp.get(b)
    if pa is None or pb is None:
        return (None, None)
    p = dist_px(pa,pb)
    c = p * scale_cm_per_px if (scale_cm_per_px is not None) else None
    return (p, c)

def midpoint(a,b):
    pa = kp.get(a); pb = kp.get(b)
    if pa is None or pb is None:
        return None
    return ((pa[0]+pb[0])/2.0, (pa[1]+pb[1])/2.0)

# Pair distances: assume pairs are sequential (pt_1<->pt_2, pt_3<->pt_4, ...)
pair_distances = {}
for i in range(0,8,2):
    a = KP_NAMES[i]; b = KP_NAMES[i+1]
    pair_distances[f"pair_{i//2+1}"] = {
        "features": [a,b],
        "value_px": None,
        "value_cm": None
    }
    px, cm = safe_pair(a,b)
    if px is not None:
        pair_distances[f"pair_{i//2+1}"]["value_px"] = round(float(px),2)
    if cm is not None:
        pair_distances[f"pair_{i//2+1}"]["value_cm"] = round(float(cm),2)

# midpoints of the 4 pairs (where available)
pair_midpoints = []
for i in range(0,8,2):
    m = midpoint(KP_NAMES[i], KP_NAMES[i+1])
    pair_midpoints.append(m)

# Udder width: distance between midpoint of left-most pair and right-most pair
# Strategy: find left-most and right-most midpoints among available pair_midpoints
valid_midpoints = [m for m in pair_midpoints if m is not None]
udder_width_px = None; udder_width_cm = None
if len(valid_midpoints) >= 2:
    # choose extreme left and extreme right by x
    left_m = min(valid_midpoints, key=lambda p: p[0])
    right_m = max(valid_midpoints, key=lambda p: p[0])
    udder_width_px = dist_px(left_m, right_m)
    udder_width_cm = udder_width_px * scale_cm_per_px if scale_cm_per_px is not None else None

# Udder depth (front-back): compute distance between centroid of front-most pairs and rear-most pairs
udder_depth_px = None; udder_depth_cm = None
if len(valid_midpoints) >= 2:
    # front-most = smallest y (top), rear-most = largest y (bottom)
    front = min(valid_midpoints, key=lambda p: p[1])
    rear = max(valid_midpoints, key=lambda p: p[1])
    udder_depth_px = dist_px(front, rear)
    udder_depth_cm = udder_depth_px * scale_cm_per_px if scale_cm_per_px is not None else None

# Rear udder height approximation (relative to image bottom):
# We approximate rear teats as the two points with highest y (closest to bottom).
rear_teats = []
all_pts = [kp.get(k) for k in KP_NAMES if kp.get(k) is not None]
if len(all_pts) >= 2:
    sorted_by_y = sorted(all_pts, key=lambda p: p[1], reverse=True)
    # pick 2 largest y
    rear_teats = sorted_by_y[:2]
rear_udder_height_px = None; rear_udder_height_cm = None
if rear_teats:
    avg_y = sum(p[1] for p in rear_teats) / len(rear_teats)
    # distance from average rear teat y to image bottom (bottom y = DISPLAY_SIZE - 1)
    if img is not None:
        bottom_y = DISPLAY_SIZE - 1
        rear_udder_height_px = max(0.0, bottom_y - avg_y)
        rear_udder_height_cm = rear_udder_height_px * scale_cm_per_px if scale_cm_per_px is not None else None

# central ligament & fore udder attachment: not computable reliably from teat-only points
central_ligament_score = None
fore_udder_attachment_score = None

# ---------- scoring ----------
scores = {}
# rear udder height scoring using UDDER_SCORES mapping (Gir)
rv_val = rear_udder_height_cm
scores["rear_udder_height"] = cm_or_deg_to_score(rv_val, UDDER_SCORES["rear_udder_height"]) if rv_val is not None else None
# other udder-specific subjective features left None (can't compute reliably)
scores["udder_depth"] = None
scores["fore_udder_attachment"] = None
scores["central_ligament"] = None

# ---------- build JSON output ----------
out = {"traits": []}

# add pair distances
for i in range(1,5):
    key = f"pair_{i}"
    rec = {
        "trait": f"Teat pair {i} distance",
        "features": pair_distances[key]["features"],
        "value_px": pair_distances[key]["value_px"],
        "value_cm": pair_distances[key]["value_cm"],
        "score": None
    }
    out["traits"].append(rec)

# udder width
out["traits"].append({
    "trait": "Udder width (midpoint left ↔ midpoint right)",
    "features": ["pair_midpoints"],
    "value_px": None if udder_width_px is None else round(float(udder_width_px),2),
    "value_cm": None if udder_width_cm is None else round(float(udder_width_cm),2),
    "score": None
})

# udder depth
out["traits"].append({
    "trait": "Udder depth (front ↔ rear midpoints)",
    "features": ["pair_midpoints"],
    "value_px": None if udder_depth_px is None else round(float(udder_depth_px),2),
    "value_cm": None if udder_depth_cm is None else round(float(udder_depth_cm),2),
    "score": scores.get("udder_depth")
})

# rear udder height (approx)
out["traits"].append({
    "trait": "Rear udder height (approx, distance of rear teats from image bottom)",
    "features": ["rear teats (2 lowest points)"],
    "value_px": None if rear_udder_height_px is None else round(float(rear_udder_height_px),2),
    "value_cm": None if rear_udder_height_cm is None else round(float(rear_udder_height_cm),2),
    "score": scores.get("rear_udder_height")
})

# central ligament / fore-attachment placeholders (cannot compute reliably from teats alone)
out["traits"].append({
    "trait": "Fore udder attachment (not computable from teats only)",
    "features": [],
    "value_px": None,
    "value_cm": None,
    "score": None
})
out["traits"].append({
    "trait": "Central ligament (not computable from teats only)",
    "features": [],
    "value_px": None,
    "value_cm": None,
    "score": None
})

# per-teat raw coordinates (helpful to debug or for downstream tools)
pts = []
for k in KP_NAMES:
    v = kp.get(k)
    pts.append({"name": k, "x": None if v is None else round(float(v[0]),2), "y": None if v is None else round(float(v[1]),2)})

out["meta"] = {
    "image_used": image_path,
    "kp_source": ("kp.json" if os.path.exists(KP_FILE) else ("model:"+MODEL_FILE if os.path.exists(MODEL_FILE) else None)),
    "scale_cm_per_px": None if scale_cm_per_px is None else float(scale_cm_per_px),
    "teat_points": pts
}

# print a single JSON object to stdout
print(json.dumps(out, indent=2))
