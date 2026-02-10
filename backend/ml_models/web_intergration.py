#!/usr/bin/env python3
"""
side_metrics_json.py

No GUI, no args. Attempts to read keypoints from kp.json OR auto-detect
using 'side_view_model.pt' (ultralytics YOLO) if available. Reads image
from a default filename list. Outputs a single JSON object with traits,
features used, measured values (cm if scale found else px), and scores.

Files tried (in order):
 - kp.json   (preferred; format: {"wither":[x,y], "pinbone":[x,y], ...})
 - side_view_model.pt (optional ultralytics YOLO model)
 - image file: input.jpg / cow.jpg / cow_side.jpg / cow_udder.jpg

Output (stdout): JSON with list of trait dicts:
  { "trait": "...", "features": [...], "value_cm": 12.3 or null, "value_px": 123.4 or null, "score": 7 or null }
"""
import json, os, math, sys
import cv2
import numpy as np

# ---------- CONFIG ----------
DISPLAY_SIZE = 640
KP_NAMES = [
    "wither", "pinbone", "shoulderbone", "chest_top", "elbow",
    "body_girth_top", "rear_elbow", "spine_between_hips", "hoof",
    "belly_deepest_point", "hock", "hip_bone", "hoof_tip", "hairline_hoof"
]

# sticker detection (green-ish square) for scale
HSV_LOWER = np.array([25, 60, 60])
HSV_UPPER = np.array([90, 255, 255])
STICKER_CM_DEFAULT = 17.0  # adjust to the real sticker size if used

# BREED scoring table (Gir) - used to map values -> 1..9
BREED_SCORES = {
    "Gir": {
        "stature":[(1,None,110),(2,111,113),(3,114,116),(4,117,118),(5,119,121),(6,122,123),(7,124,125),(8,126,127),(9,128,None)],
        "heart_girth":[(1,None,145),(2,146,149),(3,150,153),(4,154,157),(5,158,162),(6,163,165),(7,166,168),(8,169,171),(9,172,None)],
        "body_length":[(1,None,115),(2,116,118),(3,119,121),(4,122,123),(5,124,126),(6,127,128),(7,129,131),(8,132,134),(9,135,None)],
        "body_depth":[(1,None,58),(2,59,59),(3,60,61),(4,62,62),(5,63,64),(6,65,65),(7,66,67),(8,68,69),(9,70,None)],
        "rump_angle_drop":[(1,12.01,None),(2,11.0,12.0),(3,10.0,11.0),(4,9.0,10.0),(5,8.0,9.0),(6,7.0,8.0),(7,6.0,7.0),(8,5.0,6.0),(9,None,4.99)],
        "rear_legs_set":[(1,170.0,None),(2,165.0,169.9),(3,160.0,164.9),(4,156.0,159.9),(5,150.0,155.9),(6,146.0,149.9),(7,141.0,145.9),(8,135.0,140.9),(9,None,134.9)],
        "foot_angle":[(1,None,42.0),(2,43.0,44.0),(3,45.0,46.0),(4,47.0,48.0),(5,49.0,50.0),(6,51.0,52.0),(7,53.0,55.0),(8,56.0,59.0),(9,60.0,None)]
    }
}

# ---------- helpers ----------
def cm_or_deg_to_score(value, ranges):
    if value is None or ranges is None:
        return None
    for score, low, high in ranges:
        lo_ok = (low is None) or (value >= low)
        hi_ok = (high is None) or (value <= high)
        if lo_ok and hi_ok:
            return score
    # clamp
    if value is not None:
        if ranges[0][2] is not None and value < ranges[0][2]:
            return ranges[0][0]
        if ranges[-1][1] is not None and value > ranges[-1][1]:
            return ranges[-1][0]
    return None

def angle_with_vertical(a,b):
    v = (b[0]-a[0], b[1]-a[1])
    n = math.hypot(v[0], v[1])
    if n==0: return None
    cosv = max(-1.0, min(1.0, (-v[1]) / n))
    return math.degrees(math.acos(cosv))

def angle_with_horizontal(a,b):
    v = (b[0]-a[0], b[1]-a[1])
    n = math.hypot(v[0], v[1])
    if n==0: return None
    cosv = max(-1.0, min(1.0, (v[0]) / n))
    return math.degrees(math.acos(cosv))

def angle_between_lines(p1,p2,q1,q2):
    vx,vy = (p2[0]-p1[0], p2[1]-p1[1])
    wx,wy = (q2[0]-q1[0], q2[1]-q1[1])
    nv = math.hypot(vx,vy); nw = math.hypot(wx,wy)
    if nv==0 or nw==0: return None
    dot = vx*wx + vy*wy
    cosv = max(-1.0, min(1.0, dot/(nv*nw)))
    return math.degrees(math.acos(cosv))

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

def detect_sticker_scale(img_bgr, sticker_cm=STICKER_CM_DEFAULT, min_area=100):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
    k = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area:
        return None
    rect = cv2.minAreaRect(c)
    w,h = rect[1]
    sticker_px = max(w,h)
    if sticker_px < 1:
        return None
    return sticker_cm / sticker_px

# ---------- data acquisition ----------
# Preferred keypoints file
kp_file = "kp.json"
kp_map = {}

if os.path.exists(kp_file):
    try:
        d = json.load(open(kp_file, "r"))
        # ensure numeric pairs
        for k,v in d.items():
            if isinstance(v, (list,tuple)) and len(v)>=2:
                kp_map[k] = (float(v[0]), float(v[1]))
    except Exception:
        kp_map = {}

# find image file
image_candidates = ["input.jpg","cow.jpg","cow_side.jpg","cow_udder.jpg","cow_top.jpg"]
img_path = None
for fn in image_candidates:
    if os.path.exists(fn):
        img_path = fn; break

# attempt to load image if present (needed for sticker detection or model input)
img = None
if img_path:
    img = cv2.imread(img_path)
    if img is None:
        img_path = None

# if no kp.json, try to auto-detect with ultralytics model if present
model_path = "side_view_model_v2.pt"
if not kp_map and os.path.exists(model_path) and img is not None:
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        img640, scale_img, left_pad, top_pad = letterbox_to_640(img)
        results = model.predict(img640, imgsz=DISPLAY_SIZE)
        if len(results)>0:
            r = results[0]
            # try canonical keypoints -> first instance
            if getattr(r, "keypoints", None) is not None:
                try:
                    xy = r.keypoints.xy[0]
                    for i, name in enumerate(KP_NAMES):
                        if i < xy.shape[0]:
                            x = float(xy[i,0]); y = float(xy[i,1])
                            # if letterbox produced padding, coordinates are already in 640 frame
                            kp_map[name] = (x, y)
                except Exception:
                    # flatten across instances
                    merged = []
                    for ii in range(len(r.keypoints.xy)):
                        for p in r.keypoints.xy[ii]:
                            merged.append((float(p[0]), float(p[1])))
                    for i,name in enumerate(KP_NAMES):
                        if i < len(merged):
                            kp_map[name] = merged[i]
    except Exception:
        # model not usable; ignore
        pass

# If we still have no image path, try to return empty JSON
# ---------- compute metrics ----------
# helper to get (px, cm) pairs for a pair of kp names
def pair_measure(a,b, scale_cm_per_px):
    if a not in kp_map or b not in kp_map:
        return (None, None)
    pa = kp_map[a]; pb = kp_map[b]
    px = dist_px(pa, pb)
    cm = px * scale_cm_per_px if (scale_cm_per_px is not None) else None
    return (px, cm)

# compute scale (try sticker detection on loaded image)
scale_cm_per_px = None
if img is not None:
    # work on letterboxed 640
    img640, scale_img, left, top = letterbox_to_640(img)
    scale = detect_sticker_scale(img640, STICKER_CM_DEFAULT)
    if scale is not None:
        scale_cm_per_px = float(scale)

# Build metrics according to your definitions:
# Body length: shoulderbone <-> pinbone
# Stature: wither <-> hoof
# Heart girth: circumference = 2 * distance(chest_top, elbow)
# Body depth: body_girth_top <-> belly_deepest_point
# Angularity: angle at belly_deepest_point between body_girth_top and rear_elbow
# Rump vertical: vertical distance between spine_between_hips and hip_bone (cm if possible)
# Foot angle: angle between hairline_hoof -> hoof_tip and vertical
# Rear legs set: hock -> hoof angle vs horizontal
metrics = {}

def safe_get(k): return kp_map.get(k)

# body_length
metrics["body_length"] = pair_measure("shoulderbone", "pinbone", scale_cm_per_px)
# stature
metrics["stature"] = pair_measure("wither", "hoof", scale_cm_per_px)
# heart_girth: 2 * distance(chest_top, elbow)
hg_px, hg_cm_single = pair_measure("chest_top", "elbow", scale_cm_per_px)
if hg_px is None:
    metrics["heart_girth"] = (None, None)
else:
    circ_px = hg_px * 2.0
    circ_cm = hg_cm_single * 2.0 if hg_cm_single is not None else None
    metrics["heart_girth"] = (circ_px, circ_cm)
# body_depth
metrics["body_depth"] = pair_measure("body_girth_top", "belly_deepest_point", scale_cm_per_px)
# angularity (angle at belly deepest)
if safe_get("body_girth_top") and safe_get("belly_deepest_point") and safe_get("rear_elbow"):
    a = safe_get("body_girth_top"); b = safe_get("belly_deepest_point"); c = safe_get("rear_elbow")
    # compute angle at b between a-b and c-b
    ba = (a[0]-b[0], a[1]-b[1]); bc = (c[0]-b[0], c[1]-b[1])
    na = math.hypot(ba[0], ba[1]); nb = math.hypot(bc[0], bc[1])
    if na==0 or nb==0:
        metrics["angularity"] = None
    else:
        cosv = max(-1.0, min(1.0, (ba[0]*bc[0] + ba[1]*bc[1])/(na*nb)))
        metrics["angularity"] = math.degrees(math.acos(cosv))
else:
    metrics["angularity"] = None

# rump vertical (vertical px distance)
if safe_get("spine_between_hips") and safe_get("hip_bone"):
    drop_px = abs(safe_get("spine_between_hips")[1] - safe_get("hip_bone")[1])
    drop_cm = drop_px * scale_cm_per_px if scale_cm_per_px is not None else None
    metrics["rump_vertical"] = (drop_px, drop_cm)
else:
    metrics["rump_vertical"] = (None, None)

# foot angle
if safe_get("hairline_hoof") and safe_get("hoof_tip"):
    metrics["foot_angle"] = angle_with_vertical(safe_get("hairline_hoof"), safe_get("hoof_tip"))
else:
    metrics["foot_angle"] = None

# rear legs set
if safe_get("hock") and safe_get("hoof"):
    metrics["rear_legs_set"] = angle_with_horizontal(safe_get("hock"), safe_get("hoof"))
else:
    metrics["rear_legs_set"] = None

# rear legs rear-view / rump width not part of this side-dashboard, ignore

# ---------- scoring ----------
defs = BREED_SCORES.get("Gir", {})

def pair_px_or_cm_to_cm(pair, is_circ=False):
    if not pair: return None
    px, cm = pair
    if cm is not None:
        return cm
    if px is not None and scale_cm_per_px is not None:
        return px * scale_cm_per_px
    return None

scores = {}
# body_length
bl_cm = pair_px_or_cm_to_cm(metrics["body_length"])
scores["body_length"] = cm_or_deg_to_score(bl_cm, defs.get("body_length"))
# stature
st_cm = pair_px_or_cm_to_cm(metrics["stature"])
scores["stature"] = cm_or_deg_to_score(st_cm, defs.get("stature"))
# heart_girth
hg_cm = pair_px_or_cm_to_cm(metrics["heart_girth"])
scores["heart_girth"] = cm_or_deg_to_score(hg_cm, defs.get("heart_girth"))
# body_depth
bd_cm = pair_px_or_cm_to_cm(metrics["body_depth"])
scores["body_depth"] = cm_or_deg_to_score(bd_cm, defs.get("body_depth"))
# rump_vertical uses rump_angle_drop ranges
rv_cm = None
rv_pair = metrics.get("rump_vertical")
if rv_pair and rv_pair[1] is not None:
    rv_cm = rv_pair[1]
elif rv_pair and rv_pair[0] is not None and scale_cm_per_px is not None:
    rv_cm = rv_pair[0] * scale_cm_per_px
scores["rump_vertical"] = cm_or_deg_to_score(rv_cm, defs.get("rump_angle_drop"))
# foot angle
scores["foot_angle"] = cm_or_deg_to_score(metrics.get("foot_angle"), defs.get("foot_angle"))
# rear legs set
scores["rear_legs_set"] = cm_or_deg_to_score(metrics.get("rear_legs_set"), defs.get("rear_legs_set"))

# ---------- build JSON output ----------
out = {"traits": []}

def add_trait(label, features, pair_or_value, score, is_angle=False, is_circ=False):
    # pair_or_value for length-like traits is (px, cm) or None
    entry = {"trait": label, "features": features, "score": (int(score) if score is not None else None)}
    if isinstance(pair_or_value, tuple):
        px, cm = pair_or_value
        entry["value_px"] = None if px is None else round(float(px), 2)
        # for heart girth we already stored doubled px/cm
        entry["value_cm"] = None if cm is None else round(float(cm), 2)
    else:
        # single numeric value (angle)
        entry["value_px"] = None
        entry["value_cm"] = None
        entry["value"] = None if pair_or_value is None else (round(float(pair_or_value),2))
    out["traits"].append(entry)

# Add rows in requested order
add_trait("Body Length", ["shoulderbone","pinbone"], metrics.get("body_length"), scores.get("body_length"))
add_trait("Stature", ["wither","hoof"], metrics.get("stature"), scores.get("stature"))
add_trait("Heart Girth (circumference)", ["chest_top","elbow"], metrics.get("heart_girth"), scores.get("heart_girth"))
add_trait("Body Depth", ["body_girth_top","belly_deepest_point"], metrics.get("body_depth"), scores.get("body_depth"))

# angularity is an angle (degrees)
out["traits"].append({
    "trait":"Angularity (angle at belly_deepest_point between body_girth_top & rear_elbow)",
    "features":["body_girth_top","belly_deepest_point","rear_elbow"],
    "value_deg": None if metrics.get("angularity") is None else round(float(metrics.get("angularity")),2),
    "score": None
})

# rump vertical
rv_pair = metrics.get("rump_vertical")
out["traits"].append({
    "trait":"Rump vertical (spine_between_hips ↔ hip_bone)",
    "features":["spine_between_hips","hip_bone"],
    "value_px": None if rv_pair[0] is None else round(float(rv_pair[0]),2),
    "value_cm": None if rv_pair[1] is None else round(float(rv_pair[1]),2),
    "score": (int(scores.get("rump_vertical")) if scores.get("rump_vertical") is not None else None)
})

# foot angle
out["traits"].append({
    "trait":"Foot angle (hairline_hoof → hoof_tip vs vertical)",
    "features":["hairline_hoof","hoof_tip"],
    "value_deg": None if metrics.get("foot_angle") is None else round(float(metrics.get("foot_angle")),2),
    "score": (int(scores.get("foot_angle")) if scores.get("foot_angle") is not None else None)
})

# rear legs set
out["traits"].append({
    "trait":"Rear legs set (hock → hoof vs horizontal)",
    "features":["hock","hoof"],
    "value_deg": None if metrics.get("rear_legs_set") is None else round(float(metrics.get("rear_legs_set")),2),
    "score": (int(scores.get("rear_legs_set")) if scores.get("rear_legs_set") is not None else None)
})

# meta
out["meta"] = {
    "image_used": img_path,
    "kp_source": "kp.json" if os.path.exists(kp_file) else ("model:"+model_path if os.path.exists(model_path) else None),
    "scale_cm_per_px": None if scale_cm_per_px is None else float(scale_cm_per_px)
}

# print JSON to stdout only
print(json.dumps(out, indent=2))
