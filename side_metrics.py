#!/usr/bin/env python3
"""
dashboard_side_metrics.py

Interactive dashboard to compute & view side-view metrics for cattle.

Usage:
    python dashboard_side_metrics.py --image cow_side.jpg [--model side_view_model_v2.pt]

Features:
- Resize image to 640x640 for consistent display/model input
- Auto-detect keypoints with Ultralytics model (optional)
- Manual add-point mode: click image, assign a keypoint name
- Choose which metrics to display (checkboxes)
- Pick a clean color theme for overlays
- Save overlay image
"""

import argparse
import math
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np

# ----------------- CONFIG -----------------
KP_NAMES = [
    "wither", "pinbone", "shoulderbone", "chest_top", "elbow",
    "body_girth_top", "rear_elbow", "spine_between_hips", "hoof",
    "belly_deepest_point", "hock", "hip_bone", "hoof_tip", "hairline_hoof"
]
STICKER_CM = 23.0  # sticker is 21x21 cm
HSV_LOWER = np.array([25, 60, 60])
HSV_UPPER = np.array([90, 255, 255])
DISPLAY_SIZE = 640

# clean color palette (RGB)
PALETTES = {
    "Teal": {"line": (0,160,150), "text": (20,20,20)},
    "Blue": {"line": (10,120,200), "text": (20,20,20)},
    "Orange": {"line": (230,120,20), "text": (20,20,20)},
    "Gray": {"line": (80,80,80), "text": (10,10,10)}
}

METRIC_CHECKS = [
    ("Body length (shoulderbone ↔ pinbone)", "body_length"),
    ("Stature (spine_between_hips ↔ hoof)", "stature"),
    ("Heart girth (chest_top ↔ elbow)", "heart_girth"),
    ("Body depth (body_girth_top ↔ belly_deepest_point)", "body_depth"),
    ("Angularity (rump angle at belly)", "angularity"),
    ("Rump vertical (spine_between_hips ↔ hip_bone)", "rump_vertical"),
    ("Foot angle (hairline_hoof → hoof_tip)", "foot_angle"),
    ("Rear legs set (hock → hoof vs horizontal)", "rear_legs_set")
]

# ----------------- UTILITIES -----------------
def letterbox_to_640(img):
    # preserve aspect, pad to 640x640 (YOLO style)
    h, w = img.shape[:2]
    scale = DISPLAY_SIZE / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((DISPLAY_SIZE, DISPLAY_SIZE, 3), 114, dtype=np.uint8)
    top = (DISPLAY_SIZE - nh) // 2
    left = (DISPLAY_SIZE - nw) // 2
    canvas[top:top+nh, left:left+nw] = resized
    return canvas, scale, left, top

def detect_sticker_scale(img_bgr, sticker_cm=STICKER_CM, min_area=200):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, mask, None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area:
        return None, mask, None
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect).astype(int)
    w, h = rect[1]
    sticker_px = max(w, h)
    if sticker_px < 1:
        return None, mask, box
    scale = sticker_cm / sticker_px
    return scale, mask, box

def dist_pixels(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def angle_at_point(a,b,c):
    ba = (a[0]-b[0], a[1]-b[1])
    bc = (c[0]-b[0], c[1]-b[1])
    na = math.hypot(ba[0], ba[1]); nb = math.hypot(bc[0], bc[1])
    if na==0 or nb==0:
        return None
    cosv = max(-1.0, min(1.0, (ba[0]*bc[0]+ba[1]*bc[1])/(na*nb)))
    return math.degrees(math.acos(cosv))

def angle_with_vertical(a,b):
    v = (b[0]-a[0], b[1]-a[1])
    n = math.hypot(v[0], v[1]); 
    if n==0: return None
    cosv = max(-1.0, min(1.0, (v[1]*-1.0)/n))  # vertical up = (0,-1) => dot = -v.y
    return math.degrees(math.acos(cosv))

def angle_with_horizontal(a,b):
    v = (b[0]-a[0], b[1]-a[1])
    n = math.hypot(v[0], v[1]); 
    if n==0: return None
    cosv = max(-1.0, min(1.0, (v[0])/n))
    return math.degrees(math.acos(cosv))

# ----------------- DASHBOARD CLASS -----------------
class SideDashboard:
    def __init__(self, root, image_path, model_path=None):
        self.root = root
        self.root.title("Side Metrics Dashboard")
        self.image_path = image_path
        self.model_path = model_path
        self.model = None
        self.kp_map = {name: None for name in KP_NAMES}  # name -> (x,y,conf)

        # load image, letterbox to 640
        img = cv2.imread(self.image_path)
        if img is None:
            raise RuntimeError(f"Cannot read image {self.image_path}")
        self.orig_full = img.copy()
        self.orig, self.scale_img, self.left_pad, self.top_pad = letterbox_to_640(img)
        self.display = self.orig.copy()

        # sticker scale detection
        self.scale_cm_per_px, self.sticker_mask, self.sticker_box = detect_sticker_scale(self.orig, STICKER_CM)

        # attempt load model lazily
        if self.model_path is not None:
            try:
                from ultralytics import YOLO
                self.model = YOLO(self.model_path)
            except Exception as e:
                messagebox.showwarning("Model", f"Could not load ultralytics model: {e}")
                self.model = None

        # UI layout
        left = ttk.Frame(root, padding=8)
        left.grid(row=0, column=0, sticky="ns")
        right = ttk.Frame(root, padding=8)
        right.grid(row=0, column=1)

        # Canvas
        self.canvas = tk.Canvas(right, width=DISPLAY_SIZE, height=DISPLAY_SIZE, bg="black")
        self.canvas.pack()
        self._render_canvas()

        # Controls
        ttk.Label(left, text="Metrics to show:", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.metric_vars = {}
        for label, key in METRIC_CHECKS:
            v = tk.BooleanVar(value=False)
            cb = ttk.Checkbutton(left, text=label, variable=v)
            cb.pack(anchor="w", pady=2)
            self.metric_vars[key] = v

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=6)
        ttk.Label(left, text="Color theme:", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.color_var = tk.StringVar(value="Teal")
        for name in PALETTES.keys():
            rb = ttk.Radiobutton(left, text=name, variable=self.color_var, value=name)
            rb.pack(anchor="w")

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=6)
        btn_frame = ttk.Frame(left)
        btn_frame.pack(fill="x", pady=6)
        ttk.Button(btn_frame, text="Auto-detect keypoints (model)", command=self.auto_detect).pack(fill="x", pady=2)
        ttk.Button(btn_frame, text="Add point (click)", command=self.enable_add_mode).pack(fill="x", pady=2)
        ttk.Button(btn_frame, text="Clear keypoints", command=self.clear_keypoints).pack(fill="x", pady=2)
        ttk.Button(btn_frame, text="Refresh overlay", command=self.refresh_overlay).pack(fill="x", pady=2)
        ttk.Button(btn_frame, text="Save overlay", command=self.save_overlay).pack(fill="x", pady=2)
        ttk.Button(btn_frame, text="Quit", command=root.quit).pack(fill="x", pady=8)

        # info box
        self.info_var = tk.StringVar(value=self._info_text())
        ttk.Label(left, textvariable=self.info_var, wraplength=260).pack(anchor="w", pady=4)

        # state
        self.add_mode = False
        self.canvas.bind("<Button-1>", self.on_click)
        self._render_canvas()  # initial

    def _info_text(self):
        sc = f"{self.scale_cm_per_px:.6f} cm/px" if self.scale_cm_per_px else "scale unknown (pixels)"
        return f"Image: {self.image_path}\nSize: {DISPLAY_SIZE}×{DISPLAY_SIZE}\nScale: {sc}"

    def _render_canvas(self):
        vis = self.display.copy()
        # draw sticker box if available
        if self.sticker_box is not None:
            cv2.drawContours(vis, [self.sticker_box], -1, (0,0,255), 2)
        # draw keypoints
        for name, v in self.kp_map.items():
            if v is not None:
                x,y,c = v
                cv2.circle(vis, (int(x), int(y)), 5, (0,255,255), -1)
                cv2.putText(vis, name, (int(x)+6, int(y)-6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,200,0), 1)
        # convert and show
        vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(vis_rgb)
        self.tkimg = ImageTk.PhotoImage(pil)
        self.canvas.create_image(0,0,anchor="nw",image=self.tkimg)

    def auto_detect(self):
        if self.model is None:
            messagebox.showwarning("Model", "No model loaded. Provide --model or install ultralytics.")
            return
        results = self.model.predict(self.orig, imgsz=640)
        if len(results)==0:
            messagebox.showerror("Model", "No detections")
            return
        r = results[0]
        if getattr(r, "keypoints", None) is None:
            messagebox.showerror("Model", "Model returned no keypoints")
            return
        xy = r.keypoints.xy[0]
        conf = r.keypoints.conf[0]
        for i, name in enumerate(KP_NAMES):
            if i < xy.shape[0]:
                self.kp_map[name] = (float(xy[i,0]), float(xy[i,1]), float(conf[i]) if conf is not None else 1.0)
        self.refresh_overlay()
        self.info_var.set("Auto-detect finished. " + self._info_text())

    def enable_add_mode(self):
        self.add_mode = True
        messagebox.showinfo("Add point", "Click on the image to add a keypoint, then assign a name in the popup.")

    def on_click(self, event):
        if not self.add_mode:
            return
        x, y = event.x, event.y
        name = self._ask_assign_name()
        if name is None:
            self.add_mode = False
            return
        self.kp_map[name] = (float(x), float(y), 1.0)
        self.add_mode = False
        self.refresh_overlay()

    def _ask_assign_name(self):
        top = tk.Toplevel(self.root)
        top.title("Assign Keypoint")
        ttk.Label(top, text="Choose keypoint name:").pack(padx=8, pady=6)
        var = tk.StringVar(value=KP_NAMES[0])
        cb = ttk.Combobox(top, textvariable=var, values=KP_NAMES, state="readonly")
        cb.pack(padx=8, pady=6)
        res = {"name": None}
        def ok():
            res["name"] = var.get(); top.destroy()
        def cancel():
            top.destroy()
        frm = ttk.Frame(top); frm.pack(pady=6)
        ttk.Button(frm, text="OK", command=ok).pack(side="left", padx=6)
        ttk.Button(frm, text="Cancel", command=cancel).pack(side="left", padx=6)
        top.grab_set(); self.root.wait_window(top)
        return res["name"]

    def clear_keypoints(self):
        self.kp_map = {name: None for name in KP_NAMES}
        self.refresh_overlay()

    def compute_metrics(self):
        # gather xy map
        km = {k: (None if v is None else (v[0], v[1])) for k,v in self.kp_map.items()}
        s = self.scale_cm_per_px
        def cm(px): return px*s if s else None

        out = {}
        def present(a,b):
            if km[a] is None or km[b] is None: return None, None
            px = dist_pixels(km[a], km[b]); return px, cm(px)

        out["body_length"] = present("shoulderbone","pinbone")
        out["stature"] = present("spine_between_hips","hoof")
        out["heart_girth"] = present("chest_top","elbow")
        out["body_depth"] = present("body_girth_top","belly_deepest_point")
        # angularity at belly_deepest_point between body_girth_top and rear_elbow
        if km["body_girth_top"] and km["belly_deepest_point"] and km["rear_elbow"]:
            out["angularity"] = angle_at_point(km["body_girth_top"], km["belly_deepest_point"], km["rear_elbow"])
        else:
            out["angularity"] = None
        # rump vertical (vertical distance)
        if km["spine_between_hips"] and km["hip_bone"]:
            out["rump_vertical"] = (abs(km["spine_between_hips"][1]-km["hip_bone"][1]), cm(abs(km["spine_between_hips"][1]-km["hip_bone"][1])))
        else:
            out["rump_vertical"] = (None, None)
        # foot angle: hairline_hoof -> hoof_tip vs vertical
        if km["hairline_hoof"] and km["hoof_tip"]:
            out["foot_angle"] = angle_with_vertical(km["hairline_hoof"], km["hoof_tip"])
        else:
            out["foot_angle"] = None
        # rear legs set: hock -> hoof vs horizontal
        if km["hock"] and km["hoof"]:
            out["rear_legs_set"] = angle_with_horizontal(km["hock"], km["hoof"])
        else:
            out["rear_legs_set"] = None

        return out

    def refresh_overlay(self):
        # recompute display image (draw selected metrics)
        vis = self.orig.copy()
        palette = PALETTES.get(self.color_var.get(), PALETTES["Teal"])
        color_line = palette["line"]
        color_text = palette["text"]
        # draw keypoints
        for name, v in self.kp_map.items():
            if v is not None:
                x,y,_ = v
                cv2.circle(vis, (int(x), int(y)), 5, (0,255,255), -1)
                cv2.putText(vis, name, (int(x)+6, int(y)-6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_text, 1)
        metrics = self.compute_metrics()

        # draw metric visuals if checked
        def draw_line(a,b,label):
            if self.kp_map[a] is None or self.kp_map[b] is None: return
            pa = (int(self.kp_map[a][0]), int(self.kp_map[a][1])); pb = (int(self.kp_map[b][0]), int(self.kp_map[b][1]))
            cv2.line(vis, pa, pb, color_line, 2)
            mid = ((pa[0]+pb[0])//2, (pa[1]+pb[1])//2)
            cv2.putText(vis, label, (mid[0]+8, mid[1]+4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_text, 2)

        if self.metric_vars["body_length"].get():
            val = metrics.get("body_length")
            if val[0] is not None:
                lab = f"{val[1]:.1f} cm" if val[1] is not None else f"{val[0]:.1f} px"
                draw_line("shoulderbone","pinbone","Body length: "+lab)
        if self.metric_vars["stature"].get():
            val = metrics.get("stature")
            if val[0] is not None:
                lab = f"{val[1]:.1f} cm" if val[1] is not None else f"{val[0]:.1f} px"
                draw_line("spine_between_hips","hoof","Stature: "+lab)
        if self.metric_vars["heart_girth"].get():
            val = metrics.get("heart_girth")
            if val[0] is not None:
                lab = f"{val[1]:.1f} cm" if val[1] is not None else f"{val[0]:.1f} px"
                draw_line("chest_top","elbow","Heart width: "+lab)
        if self.metric_vars["body_depth"].get():
            val = metrics.get("body_depth")
            if val[0] is not None:
                lab = f"{val[1]:.1f} cm" if val[1] is not None else f"{val[0]:.1f} px"
                draw_line("body_girth_top","belly_deepest_point","Body depth: "+lab)
        if self.metric_vars["angularity"].get():
            ang = metrics.get("angularity")
            if ang is not None:
                bp = self.kp_map.get("belly_deepest_point")
                if bp: cv2.putText(vis, f"Angularity: {ang:.1f} deg", (int(bp[0])+8, int(bp[1])+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_text, 2)
        if self.metric_vars["rump_vertical"].get():
            rv = metrics.get("rump_vertical")
            if rv[0] is not None:
                x = int(self.kp_map["spine_between_hips"][0])
                y1 = int(self.kp_map["spine_between_hips"][1]); y2 = int(self.kp_map["hip_bone"][1])
                cv2.line(vis, (x,y1),(x,y2), (200,30,30), 2)
                lab = f"{rv[1]:.1f} cm" if rv[1] is not None else f"{rv[0]:.1f} px"
                cv2.putText(vis, "Rump vert: "+lab, (x+8, (y1+y2)//2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_text, 2)
        if self.metric_vars["foot_angle"].get():
            fa = metrics.get("foot_angle")
            if fa is not None:
                ht = self.kp_map.get("hoof_tip")
                if ht: cv2.putText(vis, f"Foot angle: {fa:.1f} deg", (int(ht[0])+8,int(ht[1])+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_text, 2)
        if self.metric_vars["rear_legs_set"].get():
            rs = metrics.get("rear_legs_set")
            if rs is not None:
                hk = self.kp_map.get("hock")
                if hk: cv2.putText(vis, f"Rear legs set: {rs:.1f} deg", (int(hk[0])+8,int(hk[1])+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_text, 2)

        # draw sticker box if available
        if self.sticker_box is not None:
            cv2.drawContours(vis, [self.sticker_box], -1, (0,0,255), 2)
            if self.scale_cm_per_px:
                cv2.putText(vis, f"scale: {self.scale_cm_per_px:.4f} cm/px", (8,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        # show
        self.display = vis
        vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(vis_rgb)
        self.tkimg = ImageTk.PhotoImage(pil)
        self.canvas.create_image(0,0,anchor="nw",image=self.tkimg)
        self.info_var.set(self._info_text())

    def save_overlay(self):
        outname = "side_metrics_overlay.png"
        cv2.imwrite(outname, self.display)
        messagebox.showinfo("Saved", f"Overlay saved to {outname}")

# ----------------- CLI / RUN -----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", required=False)
    args = parser.parse_args()

    root = tk.Tk()
    app = SideDashboard(root, args.image, model_path=args.model)
    root.mainloop()

if __name__ == "__main__":
    main()
