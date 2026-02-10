#!/usr/bin/env python3
"""
dashboard_rear_full.py

Rear-view metrics dashboard with "Rear legs (rear view)" angle.

Usage:
    python dashboard_rear_full.py --image cow_rear.jpg [--model rear_view_model.pt] [--sticker_cm 21]

Dependencies:
    pip install opencv-python pillow numpy
    pip install ultralytics    # optional for auto-detect

Features:
- Letterbox image to 640x640
- Optional Ultralytics model for auto keypoint detection
- Manual add-point mode (click + assign)
- Metrics: left/right rear foot angle, rump width, PLUS angle between hoof1-hock1 & hoof2-hock2
- Clean color themes, selectable metrics, overlay preview, save overlay
"""

import argparse
import math
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np

# ---------------- CONFIG ----------------
KP_NAMES = [
    "pin_bone_1","pin_bone_2",
    "hip_bone_1","hip_bone_2",
    "hock_1","hock_2",
    "hoof_1","hoof_2"
]

DISPLAY_SIZE = 640
DEFAULT_STICKER_CM = 21.0  # optional sticker for scale detection

HSV_LOWER = np.array([25, 60, 60])
HSV_UPPER = np.array([90, 255, 255])

PALETTES = {
    "Teal": {"line": (0,160,150), "text": (20,20,20)},
    "Blue": {"line": (10,120,200), "text": (20,20,20)},
    "Orange": {"line": (230,120,20), "text": (20,20,20)},
    "Gray": {"line": (80,80,80), "text": (10,10,10)}
}

METRIC_CHECKS = [
    ("Left rear foot angle (hock_1 → hoof_1)", "left_angle"),
    ("Right rear foot angle (hock_2 → hoof_2)", "right_angle"),
    ("Rump width (pin_bone_1 ↔ pin_bone_2)", "rump_width"),
    ("Rear legs (rear view) angle (between hoof1-hock1 & hoof2-hock2)", "rear_view_angle")
]

# ---------------- utilities ----------------

def letterbox(img, new_size=DISPLAY_SIZE, color=(114,114,114)):
    h, w = img.shape[:2]
    scale = new_size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_size, new_size, 3), color, dtype=np.uint8)
    top = (new_size - nh) // 2
    left = (new_size - nw) // 2
    canvas[top:top+nh, left:left+nw] = resized
    return canvas, scale, left, top

def detect_sticker_scale(img_bgr, sticker_cm=DEFAULT_STICKER_CM, min_area=200):
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

def dist_px(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def angle_with_vertical(a, b):
    # vector a->b, compare with vertical-up (0,-1)
    v = (b[0]-a[0], b[1]-a[1])
    n = math.hypot(v[0], v[1])
    if n == 0:
        return None
    cosv = max(-1.0, min(1.0, (-v[1]) / n))
    return math.degrees(math.acos(cosv))

def angle_between_lines(p1, p2, q1, q2):
    """
    Angle between line p1->p2 and q1->q2 in degrees [0..180].
    """
    if p1 is None or p2 is None or q1 is None or q2 is None:
        return None
    vx, vy = (p2[0]-p1[0], p2[1]-p1[1])
    wx, wy = (q2[0]-q1[0], q2[1]-q1[1])
    nv = math.hypot(vx, vy); nw = math.hypot(wx, wy)
    if nv == 0 or nw == 0:
        return None
    dot = vx*wx + vy*wy
    cosv = max(-1.0, min(1.0, dot / (nv * nw)))
    return math.degrees(math.acos(cosv))

# ---------------- measurement functions ----------------

def compute_rear_metrics(kp_map, scale_cm_per_px=None):
    """
    kp_map: name -> (x,y) or None
    returns dict with left_angle_deg, right_angle_deg, rump_width_px, rump_width_cm, rear_view_angle_deg
    """
    out = {
        "left_angle_deg": None,
        "right_angle_deg": None,
        "rump_width_px": None,
        "rump_width_cm": None,
        "rear_view_angle_deg": None
    }

    # left angle: hock_1 -> hoof_1 (angle vs vertical)
    h1 = kp_map.get("hock_1"); f1 = kp_map.get("hoof_1")
    if h1 is not None and f1 is not None:
        out["left_angle_deg"] = angle_with_vertical(h1, f1)

    # right angle: hock_2 -> hoof_2
    h2 = kp_map.get("hock_2"); f2 = kp_map.get("hoof_2")
    if h2 is not None and f2 is not None:
        out["right_angle_deg"] = angle_with_vertical(h2, f2)

    # rump width
    p1 = kp_map.get("pin_bone_1"); p2 = kp_map.get("pin_bone_2")
    if p1 is not None and p2 is not None:
        px = dist_px(p1, p2)
        out["rump_width_px"] = px
        if scale_cm_per_px is not None:
            out["rump_width_cm"] = px * scale_cm_per_px

    # rear-view angle between hoof1->hock1 and hoof2->hock2
    # note lines defined from hoof -> hock so direction consistent with user's description
    if f1 is not None and h1 is not None and f2 is not None and h2 is not None:
        out["rear_view_angle_deg"] = angle_between_lines(f1, h1, f2, h2)

    return out

# ---------------- drawing ----------------

def draw_rear_overlay(img, kp_map, metrics, palette):
    vis = img.copy()
    line_col = palette["line"]
    text_col = palette["text"]

    # draw keypoints
    for name, v in kp_map.items():
        if v is not None:
            cv2.circle(vis, (int(v[0]), int(v[1])), 5, (0,255,255), -1)
            cv2.putText(vis, name, (int(v[0])+6, int(v[1])-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_col, 1)

    # rump width
    if metrics.get("rump_width_px") is not None:
        p1 = kp_map.get("pin_bone_1"); p2 = kp_map.get("pin_bone_2")
        if p1 and p2:
            a = (int(p1[0]), int(p1[1])); b = (int(p2[0]), int(p2[1]))
            cv2.line(vis, a, b, line_col, 2)
            mid = ((a[0]+b[0])//2, (a[1]+b[1])//2)
            lab = f"Rump: {metrics['rump_width_px']:.1f}px"
            if metrics.get("rump_width_cm") is not None:
                lab += f" / {metrics['rump_width_cm']:.2f}cm"
            cv2.putText(vis, lab, (mid[0]+8, mid[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_col, 2)

    # left/right angle vectors
    if metrics.get("left_angle_deg") is not None:
        h1 = kp_map.get("hock_1"); f1 = kp_map.get("hoof_1")
        if h1 and f1:
            cv2.arrowedLine(vis, (int(f1[0]), int(f1[1])), (int(h1[0]), int(h1[1])), line_col, 2, tipLength=0.15)
            lab = f"L angle: {metrics['left_angle_deg']:.1f}°"
            cv2.putText(vis, lab, (int(h1[0]) - 60, int(h1[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_col, 2)

    if metrics.get("right_angle_deg") is not None:
        h2 = kp_map.get("hock_2"); f2 = kp_map.get("hoof_2")
        if h2 and f2:
            cv2.arrowedLine(vis, (int(f2[0]), int(f2[1])), (int(h2[0]), int(h2[1])), line_col, 2, tipLength=0.15)
            lab = f"R angle: {metrics['right_angle_deg']:.1f}°"
            cv2.putText(vis, lab, (int(h2[0]) + 10, int(h2[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_col, 2)

    # rear_view angle annotation
    if metrics.get("rear_view_angle_deg") is not None:
        # place text between the two hocks if available, else center
        h1 = kp_map.get("hock_1"); h2 = kp_map.get("hock_2")
        if h1 and h2:
            label_x = int((h1[0] + h2[0]) / 2)
            label_y = int((h1[1] + h2[1]) / 2) - 10
        else:
            label_x, label_y = 20, 20
        cv2.putText(vis, f"Rear legs (rear view): {metrics['rear_view_angle_deg']:.1f}°",
                    (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_col, 2)

    return vis

# ---------------- Dashboard class ----------------

class RearDashboard:
    def __init__(self, root, image_path, model_path=None, sticker_cm=None):
        self.root = root
        self.root.title("Rear Metrics Dashboard")
        self.image_path = image_path
        self.model_path = model_path
        self.sticker_cm = sticker_cm

        # load and letterbox
        img = cv2.imread(self.image_path)
        if img is None:
            raise RuntimeError(f"Cannot read image {self.image_path}")
        self.orig_full = img.copy()
        self.orig, self.scale_img, self.left_pad, self.top_pad = letterbox(img, DISPLAY_SIZE)

        # sticker scale if requested
        self.scale_cm_per_px = None
        self.sticker_box = None
        if self.sticker_cm is not None:
            self.scale_cm_per_px, _, self.sticker_box = detect_sticker_scale(self.orig, sticker_cm=self.sticker_cm)

        # kp map
        self.kp_map = {k: None for k in KP_NAMES}

        # try load model lazily
        self.model = None
        if self.model_path:
            try:
                from ultralytics import YOLO
                self.model = YOLO(self.model_path)
            except Exception as e:
                messagebox.showwarning("Model", f"Could not load model: {e}")
                self.model = None

        # UI
        left = ttk.Frame(root, padding=8); left.grid(row=0, column=0, sticky="ns")
        right = ttk.Frame(root, padding=8); right.grid(row=0, column=1)

        # canvas
        self.canvas = tk.Canvas(right, width=DISPLAY_SIZE, height=DISPLAY_SIZE, bg="black")
        self.canvas.pack()
        self.display = self.orig.copy()
        self._update_canvas_image()

        # controls
        ttk.Label(left, text="Metrics to show:", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.metric_vars = {}
        for label, key in METRIC_CHECKS:
            v = tk.BooleanVar(value=False)
            ttk.Checkbutton(left, text=label, variable=v).pack(anchor="w", pady=2)
            self.metric_vars[key] = v

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=6)
        ttk.Label(left, text="Color theme:", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.color_var = tk.StringVar(value="Teal")
        for name in PALETTES.keys():
            ttk.Radiobutton(left, text=name, variable=self.color_var, value=name).pack(anchor="w")

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=6)
        frm = ttk.Frame(left); frm.pack(fill="x", pady=6)
        ttk.Button(frm, text="Auto-detect keypoints", command=self.auto_detect).pack(fill="x", pady=2)
        ttk.Button(frm, text="Add point (click)", command=self.enable_add_mode).pack(fill="x", pady=2)
        ttk.Button(frm, text="Clear keypoints", command=self.clear_keypoints).pack(fill="x", pady=2)
        ttk.Button(frm, text="Refresh overlay", command=self.refresh_overlay).pack(fill="x", pady=2)
        ttk.Button(frm, text="Save overlay", command=self.save_overlay).pack(fill="x", pady=2)
        ttk.Button(frm, text="Quit", command=root.quit).pack(fill="x", pady=6)

        self.info_var = tk.StringVar(value=self._info_text())
        ttk.Label(left, textvariable=self.info_var, wraplength=260).pack(anchor="w", pady=6)

        # state
        self.add_mode = False
        self.canvas.bind("<Button-1>", self.on_click)

    def _info_text(self):
        sc = f"{self.scale_cm_per_px:.6f} cm/px" if self.scale_cm_per_px else "scale unknown (pixels)"
        return f"Image: {self.image_path}\nSize: {DISPLAY_SIZE}×{DISPLAY_SIZE}\nScale: {sc}"

    def _update_canvas_image(self):
        vis_rgb = cv2.cvtColor(self.display, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(vis_rgb)
        self.tkimg = ImageTk.PhotoImage(pil)
        self.canvas.create_image(0,0,anchor="nw",image=self.tkimg)

    def auto_detect(self):
        if self.model is None:
            messagebox.showwarning("Model", "No model loaded or ultralytics unavailable.")
            return
        results = self.model.predict(self.orig, imgsz=640)
        if len(results) == 0:
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
        messagebox.showinfo("Add point", "Click on the image to add a point, then assign a keypoint name.")

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
        top = tk.Toplevel(self.root); top.title("Assign Keypoint")
        ttk.Label(top, text="Choose keypoint name:").pack(padx=8, pady=6)
        var = tk.StringVar(value=KP_NAMES[0])
        cb = ttk.Combobox(top, textvariable=var, values=KP_NAMES, state="readonly")
        cb.pack(padx=8, pady=6)
        res = {"name": None}
        def ok(): res["name"] = var.get(); top.destroy()
        def cancel(): top.destroy()
        frm = ttk.Frame(top); frm.pack(pady=6)
        ttk.Button(frm, text="OK", command=ok).pack(side="left", padx=6)
        ttk.Button(frm, text="Cancel", command=cancel).pack(side="left", padx=6)
        top.grab_set(); self.root.wait_window(top)
        return res["name"]

    def clear_keypoints(self):
        self.kp_map = {k: None for k in KP_NAMES}
        self.refresh_overlay()

    def compute_metrics(self):
        # convert to simple xy map
        km = {k: (None if v is None else (v[0], v[1])) for k,v in self.kp_map.items()}
        return compute_rear_metrics(km, scale_cm_per_px=self.scale_cm_per_px)

    def refresh_overlay(self):
        pal = PALETTES.get(self.color_var.get(), PALETTES["Teal"])
        metrics = self.compute_metrics()
        vis = self.orig.copy()

        # draw sticker box if available
        if self.sticker_box is not None:
            cv2.drawContours(vis, [self.sticker_box], -1, (0,0,255), 2)
            if self.scale_cm_per_px:
                cv2.putText(vis, f"scale: {self.scale_cm_per_px:.4f} cm/px", (8,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        # always draw keypoints
        for name, v in self.kp_map.items():
            if v is not None:
                cv2.circle(vis, (int(v[0]), int(v[1])), 5, (0,255,255), -1)
                cv2.putText(vis, name, (int(v[0])+6, int(v[1])-6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, pal["text"], 1)

        # draw selected metrics
        if self.metric_vars["rump_width"].get() and metrics.get("rump_width_px") is not None:
            p1 = self.kp_map.get("pin_bone_1"); p2 = self.kp_map.get("pin_bone_2")
            if p1 and p2:
                a = (int(p1[0]), int(p1[1])); b = (int(p2[0]), int(p2[1]))
                cv2.line(vis, a, b, pal["line"], 2)
                mid = ((a[0]+b[0])//2, (a[1]+b[1])//2)
                lab = f"Rump: {metrics['rump_width_px']:.1f}px"
                if metrics.get("rump_width_cm") is not None:
                    lab += f" / {metrics['rump_width_cm']:.2f}cm"
                cv2.putText(vis, lab, (mid[0]+8, mid[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, pal["text"], 2)

        if self.metric_vars["left_angle"].get() and metrics.get("left_angle_deg") is not None:
            h1 = self.kp_map.get("hock_1"); f1 = self.kp_map.get("hoof_1")
            if h1 and f1:
                cv2.arrowedLine(vis, (int(f1[0]), int(f1[1])), (int(h1[0]), int(h1[1])), pal["line"], 2, tipLength=0.15)
                cv2.putText(vis, f"L angle: {metrics['left_angle_deg']:.1f}°", (int(h1[0])-60, int(h1[1])-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, pal["text"], 2)

        if self.metric_vars["right_angle"].get() and metrics.get("right_angle_deg") is not None:
            h2 = self.kp_map.get("hock_2"); f2 = self.kp_map.get("hoof_2")
            if h2 and f2:
                cv2.arrowedLine(vis, (int(f2[0]), int(f2[1])), (int(h2[0]), int(h2[1])), pal["line"], 2, tipLength=0.15)
                cv2.putText(vis, f"R angle: {metrics['right_angle_deg']:.1f}°", (int(h2[0])+10, int(h2[1])-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, pal["text"], 2)

        # rear-view combined angle: hoof1->hock1 vs hoof2->hock2
        if self.metric_vars["rear_view_angle"].get():
            f1 = self.kp_map.get("hoof_1"); h1 = self.kp_map.get("hock_1")
            f2 = self.kp_map.get("hoof_2"); h2 = self.kp_map.get("hock_2")
            if f1 and h1 and f2 and h2:
                # draw both lines (hoof -> hock)
                cv2.line(vis, (int(f1[0]), int(f1[1])), (int(h1[0]), int(h1[1])), pal["line"], 2)
                cv2.line(vis, (int(f2[0]), int(f2[1])), (int(h2[0]), int(h2[1])), pal["line"], 2)
                # compute angle
                ang = angle_between_lines((f1[0], f1[1]), (h1[0], h1[1]), (f2[0], f2[1]), (h2[0], h2[1]))
                if ang is not None:
                    label_x = int((h1[0] + h2[0]) / 2)
                    label_y = int((h1[1] + h2[1]) / 2) - 10
                    cv2.putText(vis, f"Rear legs (rear view): {ang:.1f}°", (label_x, label_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, pal["text"], 2)

        self.display = vis
        self._update_canvas_image()
        self.info_var.set(self._info_text())

    def save_overlay(self):
        out = "rear_metrics_overlay.png"
        cv2.imwrite(out, self.display)
        messagebox.showinfo("Saved", f"Overlay saved to {out}")

# ---------------- CLI / RUN ----------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", required=False)
    parser.add_argument("--sticker_cm", type=float, required=False, help="size of square sticker in cm for scale detection")
    args = parser.parse_args()

    root = tk.Tk()
    app = RearDashboard(root, args.image, model_path=args.model, sticker_cm=args.sticker_cm)
    root.mainloop()

if __name__ == "__main__":
    main()
