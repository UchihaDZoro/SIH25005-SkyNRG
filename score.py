#!/usr/bin/env python3
"""
dashboard_side_metrics.py — corrected stature/body_length + scoring behavior.

Usage:
    python dashboard_side_metrics.py --image cow.jpg [--model side_view_model.pt] [--sticker_cm 21]

Notes:
- Stature = wither -> hoof (height at withers)
- Body length = shoulderbone -> pinbone
- Heart girth = 2 * distance(chest_top, elbow)
"""
import argparse
import math
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2, numpy as np

# ----------------- CONFIG -----------------
KP_NAMES = [
    "wither", "pinbone", "shoulderbone", "chest_top", "elbow",
    "body_girth_top", "rear_elbow", "spine_between_hips", "hoof",
    "belly_deepest_point", "hock", "hip_bone", "hoof_tip", "hairline_hoof"
]

STICKER_CM_DEFAULT = 17.0
HSV_LOWER = np.array([25, 60, 60])
HSV_UPPER = np.array([90, 255, 255])
DISPLAY_SIZE = 640

PALETTES = {
    "Teal":   {"line": (0,160,150), "text": (20,20,20)},
    "Blue":   {"line": (10,120,200), "text": (20,20,20)},
    "Orange": {"line": (230,120,20), "text": (20,20,20)},
    "Gray":   {"line": (80,80,80), "text": (10,10,10)},
}

TEXT_COLORS = {
    "White":   (255,255,255),
    "Black":   (0,0,0),
    "Yellow":  (0,255,255),
    "Cyan":    (255,255,0),
    "Magenta": (255,0,255),
    "Lime":    (0,255,0),
}

METRIC_CHECKS = [
    ("Body length (shoulderbone ↔ pinbone)", "body_length"),
    ("Stature (height at withers: wither ↔ hoof)", "stature"),
    ("Heart girth (chest_top ↔ elbow) — circumference = Ramanujan Formula", "heart_girth"),
    ("Body depth (body_girth_top ↔ belly_deepest_point)", "body_depth"),
    ("Angularity (belly angle)", "angularity"),
    ("Rump vertical (spine_between_hips ↔ hip_bone)", "rump_vertical"),
    ("Foot angle (hairline_hoof → hoof_tip)", "foot_angle"),
    ("Rear legs set (hock → hoof vs horizontal/vertical)", "rear_legs_set"),
]

# --- Breed scoring (unchanged structure) ---
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

# ----------------- helpers -----------------
def cm_or_deg_to_score(value, ranges):
    if value is None or ranges is None: return None
    for score, low, high in ranges:
        lo_ok = (low is None) or (value >= low)
        hi_ok = (high is None) or (value <= high)
        if lo_ok and hi_ok:
            return score
    # clamp to ends if outside
    if value is not None:
        if ranges[0][2] is not None and value < ranges[0][2]:
            return ranges[0][0]
        if ranges[-1][1] is not None and value > ranges[-1][1]:
            return ranges[-1][0]
    return None

def letterbox_to_640(img):
    h,w = img.shape[:2]
    scale = DISPLAY_SIZE / max(h,w)
    nh, nw = int(h*scale), int(w*scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((DISPLAY_SIZE, DISPLAY_SIZE, 3), 114, dtype=np.uint8)
    top = (DISPLAY_SIZE - nh)//2; left = (DISPLAY_SIZE - nw)//2
    canvas[top:top+nh, left:left+nw] = resized
    return canvas, scale, left, top

def detect_sticker_scale(img_bgr, sticker_cm, min_area=100):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
    k = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None, mask, None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area: return None, mask, None
    rect = cv2.minAreaRect(c); box = cv2.boxPoints(rect).astype(int)
    w,h = rect[1]; px = max(w,h)
    if px < 1: return None, mask, box
    return (sticker_cm/px), mask, box

def dist_pixels(a,b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def angle_at_point(a,b,c):
    ba = (a[0]-b[0], a[1]-b[1]); bc = (c[0]-b[0], c[1]-b[1])
    na = math.hypot(ba[0], ba[1]); nb = math.hypot(bc[0], bc[1])
    if na==0 or nb==0: return None
    cosv = max(-1.0, min(1.0, (ba[0]*bc[0]+ba[1]*bc[1])/(na*nb)))
    return math.degrees(math.acos(cosv))

def angle_with_vertical(a,b):
    v = (b[0]-a[0], b[1]-a[1]); n = math.hypot(v[0], v[1])
    if n==0: return None
    cosv = max(-1.0, min(1.0, (-v[1]) / n))
    return math.degrees(math.acos(cosv))

def angle_with_horizontal(a,b):
    v = (b[0]-a[0], b[1]-a[1]); n = math.hypot(v[0], v[1])
    if n==0: return None
    cosv = max(-1.0, min(1.0, (v[0]) / n))
    return math.degrees(math.acos(cosv))

def pick_value_for_ranges(candidates, ranges):
    for cand in candidates:
        sc = cm_or_deg_to_score(cand, ranges)
        if sc is not None:
            return cand, sc
    if candidates:
        return candidates[0], cm_or_deg_to_score(candidates[0], ranges)
    return None, None

# ----------------- Dashboard class -----------------
class SideDashboard:
    def __init__(self, root, image_path, model_path=None, sticker_cm=STICKER_CM_DEFAULT):
        self.root = root
        self.root.title("Side Metrics Dashboard")
        self.image_path = image_path; self.model_path = model_path
        self.kp_map = {n: None for n in KP_NAMES}

        img = cv2.imread(image_path)
        if img is None: raise RuntimeError("Cannot read image")
        self.orig_full = img.copy()
        self.orig, self.scale_img, self.left_pad, self.top_pad = letterbox_to_640(img)
        self.display = self.orig.copy()

        self.scale_cm_per_px, self.sticker_mask, self.sticker_box = detect_sticker_scale(self.orig, sticker_cm)

        self.model = None
        if model_path:
            try:
                from ultralytics import YOLO
                self.model = YOLO(model_path)
            except Exception as e:
                messagebox.showwarning("Model", f"Could not load model: {e}"); self.model = None

        left = ttk.Frame(root, padding=8); mid = ttk.Frame(root, padding=8); right = ttk.Frame(root, padding=8)
        left.grid(row=0,column=0,sticky="nsw"); mid.grid(row=0,column=1,sticky="nsew"); right.grid(row=0,column=2,sticky="ns")
        root.grid_columnconfigure(1, weight=1); root.grid_rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(mid, width=DISPLAY_SIZE, height=DISPLAY_SIZE, bg="black")
        self.canvas.pack()
        self._render_canvas()

        ttk.Label(left, text="Breed:", font=("Segoe UI",10,"bold")).pack(anchor="w")
        self.breed_var = tk.StringVar(value="Gir"); ttk.Combobox(left, textvariable=self.breed_var, values=list(BREED_SCORES.keys()), state="readonly").pack(anchor="w", pady=(0,6))

        ttk.Label(left, text="Metrics to show:", font=("Segoe UI",10,"bold")).pack(anchor="w")
        self.metric_vars = {}
        for label,key in METRIC_CHECKS:
            v=tk.BooleanVar(value=False); ttk.Checkbutton(left, text=label, variable=v).pack(anchor="w", pady=2); self.metric_vars[key]=v

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=6)
        ttk.Label(left, text="Line color theme:", font=("Segoe UI",10,"bold")).pack(anchor="w")
        self.color_var = tk.StringVar(value="Teal")
        for name in PALETTES.keys(): ttk.Radiobutton(left, text=name, variable=self.color_var, value=name).pack(anchor="w")

        ttk.Label(left, text="Text color:", font=("Segoe UI",10,"bold")).pack(anchor="w", pady=(6,0))
        self.text_color_var = tk.StringVar(value="White")
        ttk.Combobox(left, textvariable=self.text_color_var, values=list(TEXT_COLORS.keys()), state="readonly").pack(anchor="w", pady=(0,6))

        btn_frame = ttk.Frame(left); btn_frame.pack(fill="x", pady=6)
        ttk.Button(btn_frame, text="Auto-detect keypoints (model)", command=self.auto_detect).pack(fill="x", pady=2)
        ttk.Button(btn_frame, text="Add point (click)", command=self.enable_add_mode).pack(fill="x", pady=2)
        ttk.Button(btn_frame, text="Clear keypoints", command=self.clear_keypoints).pack(fill="x", pady=2)
        ttk.Button(btn_frame, text="Refresh overlay", command=self.refresh_overlay).pack(fill="x", pady=2)
        ttk.Button(btn_frame, text="Save overlay", command=self.save_overlay).pack(fill="x", pady=2)
        ttk.Button(btn_frame, text="Quit", command=root.quit).pack(fill="x", pady=8)

        self.info_var = tk.StringVar(value=self._info_text()); ttk.Label(left, textvariable=self.info_var, wraplength=260).pack(anchor="w", pady=4)

        ttk.Label(right, text="ATC Scores", font=("Segoe UI",11,"bold")).pack(anchor="w")
        self.score_text = tk.Text(right, width=52, height=32, font=("Courier New",10)); self.score_text.pack(fill="both", expand=True)

        self.add_mode=False
        self.canvas.bind("<Button-1>", self.on_click)
        self.refresh_overlay()

    def _info_text(self):
        sc = f"{self.scale_cm_per_px:.6f} cm/px" if self.scale_cm_per_px else "scale unknown (pixels)"
        return f"Image: {self.image_path}\nSize: {DISPLAY_SIZE}×{DISPLAY_SIZE}\nScale: {sc}\nBreed: {self.breed_var.get()}"

    def _render_canvas(self):
        vis=self.display.copy()
        if self.sticker_box is not None: cv2.drawContours(vis,[self.sticker_box],-1,(0,0,255),2)
        for name,v in self.kp_map.items():
            if v is not None:
                x,y,c = v; cv2.circle(vis,(int(x),int(y)),5,(0,255,255),-1)
        vis_rgb=cv2.cvtColor(vis,cv2.COLOR_BGR2RGB); pil=Image.fromarray(vis_rgb); self.tkimg=ImageTk.PhotoImage(pil)
        self.canvas.create_image(0,0,anchor="nw",image=self.tkimg)

    def auto_detect(self):
        if self.model is None:
            messagebox.showwarning("Model","No model loaded.")
            return
        results = self.model.predict(self.orig, imgsz=640)
        if len(results)==0:
            messagebox.showerror("Model","No detections"); return
        r = results[0]
        if getattr(r,"keypoints",None) is None:
            messagebox.showerror("Model","Model returned no keypoints"); return
        try:
            xy = r.keypoints.xy[0]
            if xy.shape[0] >= len(KP_NAMES):
                for i,name in enumerate(KP_NAMES):
                    self.kp_map[name] = (float(xy[i,0]), float(xy[i,1]), float(r.keypoints.conf[0][i]) if getattr(r.keypoints,'conf',None) is not None else 1.0)
                self.refresh_overlay(); self.info_var.set("Auto-detect (single instance)"); return
        except Exception:
            pass
        det = r.keypoints
        merged=[]
        for i in range(len(det.xy)):
            for p in det.xy[i]:
                merged.append((float(p[0]), float(p[1])))
        for idx,name in enumerate(KP_NAMES):
            if idx < len(merged): self.kp_map[name]=(merged[idx][0], merged[idx][1], 1.0)
            else: self.kp_map[name]=None
        self.refresh_overlay(); self.info_var.set(f"Auto-detect (flattened) assigned {sum(1 for v in self.kp_map.values() if v is not None)}/{len(KP_NAMES)}")

    def enable_add_mode(self):
        self.add_mode=True; messagebox.showinfo("Add point","Click image to add point then assign a name.")

    def on_click(self, event):
        if not self.add_mode: return
        x,y = event.x,event.y
        name = self._ask_assign_name()
        if name is None: self.add_mode=False; return
        self.kp_map[name] = (float(x), float(y), 1.0)
        self.add_mode=False; self.refresh_overlay()

    def _ask_assign_name(self):
        top=tk.Toplevel(self.root); top.title("Assign Keypoint")
        ttk.Label(top,text="Choose keypoint name:").pack(padx=8,pady=6)
        var=tk.StringVar(value=KP_NAMES[0])
        cb=ttk.Combobox(top,textvariable=var,values=KP_NAMES,state="readonly"); cb.pack(padx=8,pady=6)
        res={"name":None}
        def ok(): res["name"]=var.get(); top.destroy()
        def cancel(): top.destroy()
        frm=ttk.Frame(top); frm.pack(pady=6)
        ttk.Button(frm,text="OK",command=ok).pack(side="left",padx=6)
        ttk.Button(frm,text="Cancel",command=cancel).pack(side="left",padx=6)
        top.grab_set(); self.root.wait_window(top)
        return res["name"]

    def clear_keypoints(self):
        self.kp_map = {n: None for n in KP_NAMES}; self.refresh_overlay()

    # compute raw measurements (px and cm where possible)
    def compute_metrics(self):
        km = {k:(None if v is None else (v[0], v[1])) for k,v in self.kp_map.items()}
        s = self.scale_cm_per_px
        def to_cm(px): return px*s if (px is not None and s is not None) else None
        out={}
        def pair(a,b):
            if km[a] is None or km[b] is None: return (None,None)
            p = dist_pixels(km[a], km[b]); return (p, to_cm(p))
        # BODY LENGTH = shoulderbone <-> pinbone
        out["body_length"]=pair("shoulderbone","pinbone")
        # STATURE = wither <-> hoof (height at withers)
        out["stature"]=pair("wither","hoof")
        # HEART GIRTH: circumference approximated as 2 * distance(chest_top, elbow)
        hg_pair = pair("chest_top", "elbow")  # (px, cm_single)
        if hg_pair[0] is None:
            out["heart_girth"] = (None, None)
        else:
            px_single = hg_pair[0]
            cm_single = hg_pair[1]
            circ_px = px_single
            circ_cm = (cm_single * 2) if cm_single is not None else None
            out["heart_girth"] = (circ_px, circ_cm)

        out["body_depth"]=pair("body_girth_top","belly_deepest_point")
        out["angularity"] = angle_at_point(km["body_girth_top"], km["belly_deepest_point"], km["rear_elbow"]) if km["body_girth_top"] and km["belly_deepest_point"] and km["rear_elbow"] else None
        if km["spine_between_hips"] and km["hip_bone"]:
            drop_px = abs(km["spine_between_hips"][1] - km["hip_bone"][1]); out["rump_vertical"]=(drop_px, to_cm(drop_px))
        else: out["rump_vertical"]=(None,None)
        if km["hairline_hoof"] and km["hoof_tip"]:
            raw_fa = angle_with_vertical(km["hairline_hoof"], km["hoof_tip"])
            if raw_fa is None: out["foot_angle"]=None
            else:
                cand = [raw_fa, 180 - raw_fa, min(raw_fa, 180 - raw_fa)]
                out["foot_angle_candidates"] = cand
                out["foot_angle"] = cand[0]
        else: out["foot_angle"]=None; out["foot_angle_candidates"]=None
        if km["hock"] and km["hoof"]:
            raw_rl = angle_with_horizontal(km["hock"], km["hoof"])
            if raw_rl is None: out["rear_legs_set"]=None
            else:
                cand = [raw_rl, 180 - raw_rl]
                out["rear_legs_set_candidates"] = cand
                out["rear_legs_set"] = cand[0]
        else: out["rear_legs_set"]=None; out["rear_legs_set_candidates"]=None
        return out

    def pick_best_candidate_and_score(self, candidates, ranges):
        for c in candidates:
            sc = cm_or_deg_to_score(c, ranges)
            if sc is not None:
                return c, sc
        if candidates:
            return candidates[0], cm_or_deg_to_score(candidates[0], ranges)
        return None, None

    # compute ATC scores
    def compute_atc_scores(self, metrics):
        defs = BREED_SCORES.get(self.breed_var.get(), {})
        scores = {}

        def pair_to_cm(pair):
            if not pair:
                return None
            px, cm = pair
            if cm is not None:
                return cm
            if px is not None and self.scale_cm_per_px is not None:
                return px * self.scale_cm_per_px
            return None

        bl_cm = pair_to_cm(metrics.get("body_length"))
        st_cm = pair_to_cm(metrics.get("stature"))
        hg_cm = pair_to_cm(metrics.get("heart_girth"))
        bd_cm = pair_to_cm(metrics.get("body_depth"))

        scores["body_length"] = cm_or_deg_to_score(bl_cm, defs.get("body_length")) if bl_cm is not None else None
        scores["stature"] = cm_or_deg_to_score(st_cm, defs.get("stature")) if st_cm is not None else None
        scores["heart_girth"] = cm_or_deg_to_score(hg_cm, defs.get("heart_girth")) if hg_cm is not None else None
        scores["body_depth"] = cm_or_deg_to_score(bd_cm, defs.get("body_depth")) if bd_cm is not None else None

        rv = metrics.get("rump_vertical")
        rv_cm = None
        if rv and rv[1] is not None:
            rv_cm = rv[1]
        elif rv and rv[0] is not None and self.scale_cm_per_px is not None:
            rv_cm = rv[0] * self.scale_cm_per_px
        scores["rump_vertical"] = cm_or_deg_to_score(rv_cm, defs.get("rump_angle_drop")) if rv_cm is not None else None

        if metrics.get("foot_angle_candidates"):
            cand, sc = self.pick_best_candidate_and_score(metrics.get("foot_angle_candidates"), defs.get("foot_angle"))
            scores["foot_angle"] = sc
        else:
            scores["foot_angle"] = None

        if metrics.get("rear_legs_set_candidates"):
            cand, sc = self.pick_best_candidate_and_score(metrics.get("rear_legs_set_candidates"), defs.get("rear_legs_set"))
            scores["rear_legs_set"] = sc
        else:
            scores["rear_legs_set"] = None

        return scores

    def refresh_overlay(self):
        vis = self.orig.copy()
        pal = PALETTES.get(self.color_var.get(), PALETTES["Teal"])
        color_line = pal["line"]; color_text = TEXT_COLORS.get(self.text_color_var.get(), (20,20,20))
        for name, v in self.kp_map.items():
            if v is not None:
                x,y,c = v; cv2.circle(vis,(int(x),int(y)),5,(0,255,255),-1)
                cv2.putText(vis, name, (int(x)+6,int(y)-6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_text, 1)

        metrics = self.compute_metrics(); scores = self.compute_atc_scores(metrics)

        def draw_line(a,b,label):
            if self.kp_map[a] is None or self.kp_map[b] is None: return
            pa=(int(self.kp_map[a][0]), int(self.kp_map[a][1])); pb=(int(self.kp_map[b][0]), int(self.kp_map[b][1]))
            cv2.line(vis, pa, pb, color_line, 2)
            mid = ((pa[0]+pb[0])//2, (pa[1]+pb[1])//2)
            cv2.putText(vis, label, (mid[0]+8, mid[1]+4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_text, 2)

        if self.metric_vars["body_length"].get():
            val = metrics.get("body_length")
            if val and val[0] is not None:
                lab = f"{val[1]:.1f} cm" if val[1] is not None else f"{val[0]:.1f} px"
                draw_line("shoulderbone","pinbone","Body length: "+lab)

        if self.metric_vars["stature"].get():
            val = metrics.get("stature")
            if val and val[0] is not None:
                lab = f"{val[1]:.1f} cm" if val[1] is not None else f"{val[0]:.1f} px"
                draw_line("wither","hoof","Stature: "+lab)

        if self.metric_vars["heart_girth"].get():
            val = metrics.get("heart_girth")
            if val and val[0] is not None:
                lab = f"{val[1]:.1f} cm" if val[1] is not None else f"{val[0]:.1f} px"
                draw_line("chest_top","elbow","Heart girth: "+lab)

        if self.metric_vars["body_depth"].get():
            val = metrics.get("body_depth")
            if val and val[0] is not None:
                lab = f"{val[1]:.1f} cm" if val[1] is not None else f"{val[0]:.1f} px"
                draw_line("body_girth_top","belly_deepest_point","Body depth: "+lab)

        if self.metric_vars["angularity"].get():
            ang = metrics.get("angularity")
            if ang is not None:
                bp = self.kp_map.get("belly_deepest_point")
                if bp: cv2.putText(vis, f"Angularity: {ang:.1f}°", (int(bp[0])+8, int(bp[1])+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_text, 2)

        if self.metric_vars["rump_vertical"].get():
            rv = metrics.get("rump_vertical")
            if rv and rv[0] is not None:
                x = int(self.kp_map["spine_between_hips"][0]); y1=int(self.kp_map["spine_between_hips"][1]); y2=int(self.kp_map["hip_bone"][1])
                cv2.line(vis,(x,y1),(x,y2),(200,30,30),2)
                lab = f"{rv[1]:.1f} cm" if rv[1] is not None else f"{rv[0]:.1f} px"
                cv2.putText(vis, "Rump drop: "+lab,(x+8,(y1+y2)//2),cv2.FONT_HERSHEY_SIMPLEX,0.6,color_text,2)

        if self.metric_vars["foot_angle"].get():
            fa = metrics.get("foot_angle")
            if fa is not None:
                ht = self.kp_map.get("hoof_tip")
                if ht:
                    txt = f"Foot angle candidates: {metrics.get('foot_angle_candidates')}"
                    cv2.putText(vis, txt, (int(ht[0])+8, int(ht[1])+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_text, 1)

        if self.metric_vars["rear_legs_set"].get():
            rs = metrics.get("rear_legs_set")
            if rs is not None:
                hk = self.kp_map.get("hock")
                if hk: cv2.putText(vis, f"Rear legs candidates: {metrics.get('rear_legs_set_candidates')}", (int(hk[0])+8, int(hk[1])+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_text,1)

        if self.sticker_box is not None:
            cv2.drawContours(vis,[self.sticker_box],-1,(0,0,255),2)
            if self.scale_cm_per_px: cv2.putText(vis, f"scale: {self.scale_cm_per_px:.4f} cm/px", (8,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255),2)

        self.display = vis
        vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB); pil=Image.fromarray(vis_rgb); self.tkimg=ImageTk.PhotoImage(pil)
        self.canvas.create_image(0,0,anchor="nw",image=self.tkimg)
        self.info_var.set(self._info_text())

        # ---- update scores pane ----
        self.score_text.delete("1.0", tk.END)
        rows=[]
        def add_row(label, value, score, unit=""):
            v_str="--" if value is None else f"{value:.1f}{unit}"
            s_str="--" if score is None else str(score)
            rows.append((label, v_str, s_str))

        # prepare values (prefer cm directly, else convert px->cm if scale exists)
        def try_get_cm(pair, multiply_for_circumference=False):
            if not pair: return None
            px, cm = pair
            if cm is not None:
                return cm * (2 if multiply_for_circumference else 1)
            if px is not None and self.scale_cm_per_px is not None:
                return px * self.scale_cm_per_px * (2 if multiply_for_circumference else 1)
            return None

        bl_cm = try_get_cm(metrics.get("body_length"), multiply_for_circumference=False)
        st_cm = try_get_cm(metrics.get("stature"), multiply_for_circumference=False)
        hg_cm = try_get_cm(metrics.get("heart_girth"), multiply_for_circumference=False)  # heart_girth already stored as circumference cm if sticker available
        bd_cm = try_get_cm(metrics.get("body_depth"), multiply_for_circumference=False)
        rv_cm = try_get_cm(metrics.get("rump_vertical"), multiply_for_circumference=False) if metrics.get("rump_vertical") else None

        fa_val = None
        if metrics.get("foot_angle_candidates"):
            fa_val, _ = self.pick_best_candidate_and_score(metrics.get("foot_angle_candidates"), BREED_SCORES.get(self.breed_var.get(),{}).get("foot_angle"))
            if fa_val is None: fa_val = metrics["foot_angle_candidates"][0]
        rl_val = None
        if metrics.get("rear_legs_set_candidates"):
            rl_val, _ = self.pick_best_candidate_and_score(metrics.get("rear_legs_set_candidates"), BREED_SCORES.get(self.breed_var.get(),{}).get("rear_legs_set"))
            if rl_val is None: rl_val = metrics["rear_legs_set_candidates"][0]

        scores = self.compute_atc_scores(metrics)
        add_row("Body Length", bl_cm, scores.get("body_length"), " cm")
        add_row("Stature", st_cm, scores.get("stature"), " cm")
        add_row("Heart Girth", hg_cm, scores.get("heart_girth"), " cm")
        add_row("Body Depth", bd_cm, scores.get("body_depth"), " cm")
        add_row("Rump Drop", rv_cm, scores.get("rump_vertical"), " cm")
        add_row("Foot Angle", fa_val, scores.get("foot_angle"), "°")
        add_row("Rear Legs Set", rl_val, scores.get("rear_legs_set"), "°")

        col1 = max(len(r[0]) for r in rows + [("Trait","", "")])
        col2 = max(len(r[1]) for r in rows + [("", "Value","")])
        col3 = max(len(r[2]) for r in rows + [("", "", "Score")])
        header = f"{'Trait'.ljust(col1)}  {'Value'.ljust(col2)}  {'Score'.ljust(col3)}\n"
        sep = "-"*(len(header)-1) + "\n"
        self.score_text.insert(tk.END, header); self.score_text.insert(tk.END, sep)
        for label, v_str, s_str in rows:
            line = f"{label.ljust(col1)}  {v_str.ljust(col2)}  {s_str.ljust(col3)}\n"
            self.score_text.insert(tk.END, line)

    def save_overlay(self):
        out = "side_metrics_overlay.png"; cv2.imwrite(out, self.display); messagebox.showinfo("Saved", f"Saved {out}")

# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", required=False)
    parser.add_argument("--sticker_cm", type=float, default=STICKER_CM_DEFAULT, help="physical sticker side in cm")
    args = parser.parse_args()
    root = tk.Tk()
    app = SideDashboard(root, args.image, model_path=args.model, sticker_cm=args.sticker_cm)
    root.mainloop()

if __name__ == "__main__":
    main()
