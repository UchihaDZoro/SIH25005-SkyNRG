#!/usr/bin/env python3
"""
Dashboard to select two features and calculate distance between them.

Usage:
    python dashboard.py --image image.jpg [--model side_view_model.pt] [--sticker_cm 5.0]

Dependencies:
    pip install opencv-python pillow
    pip install ultralytics    # optional, only if you want automatic keypoint detection
"""

import argparse
import math
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np

# ---- default keypoint names (your 12 names) ----
KP_NAMES = [
    "wither", "pinbone", "shoulderbone", "chest_top", "elbow",
    "body_girth_top", "rear_elbow", "spine_between_hips", "hoof",
    "belly_deepest_point", "hock", "hip_bone", "hoof_tip", "hairline_hoof"
]

# ---- sticker detection HSV (tune if needed) ----
HSV_LOWER = np.array([40, 70, 70], dtype=np.uint8)
HSV_UPPER = np.array([90, 255, 255], dtype=np.uint8)


def detect_sticker_scale(img_bgr, sticker_cm=5.0):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, mask, None
    c = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    w, h = rect[1]
    sticker_px = max(w, h)
    if sticker_px < 1:
        return None, mask, None
    scale = float(sticker_cm) / float(sticker_px)
    box = cv2.boxPoints(rect).astype(int)
    return scale, mask, box


# ---- Optional ultralytics predictor (loaded on demand) ----
def try_load_ultralytics(model_path):
    try:
        from ultralytics import YOLO
    except Exception:
        return None
    try:
        model = YOLO(model_path)
    except Exception:
        return None
    return model


def extract_kps_from_ultralytics_result(result):
    """Return kps array (N,3) from first detection. Raise if missing."""
    if getattr(result, "keypoints", None) is None:
        return None
    # take first instance
    xy = result.keypoints.xy[0]    # (N,2)
    conf = result.keypoints.conf[0]  # (N,)
    kps = np.hstack([xy, conf.reshape(-1, 1)])
    return kps


class Dashboard:
    def __init__(self, root, image_path, sticker_cm=5.0, model_path=None):
        self.root = root
        self.root.title("Measurement Dashboard")
        self.image_path = image_path
        self.sticker_cm = sticker_cm
        self.model_path = model_path

        # load image
        self.img_bgr = cv2.imread(self.image_path)
        if self.img_bgr is None:
            raise RuntimeError(f"Cannot read image {self.image_path}")

        self.orig = self.img_bgr.copy()
        self.display = self.img_bgr.copy()
        self.scale, self.mask, self.sticker_box = detect_sticker_scale(self.orig, sticker_cm=self.sticker_cm)

        # data structures: map name -> (x,y,conf) if available
        self.kps = {name: None for name in KP_NAMES}

        # optional model
        self.model = None
        if self.model_path:
            self.model = try_load_ultralytics(self.model_path)
            if self.model is None:
                messagebox.showwarning("Model", "Ultralytics not available or model failed to load. Auto-detect disabled.")
                self.model = None

        # GUI elements
        self.left_frame = ttk.Frame(root, padding=6)
        self.left_frame.grid(row=0, column=0, sticky="nsw")
        self.canvas_frame = ttk.Frame(root, padding=6)
        self.canvas_frame.grid(row=0, column=1)

        # Canvas image
        self.canvas = tk.Canvas(self.canvas_frame, width=self.orig.shape[1], height=self.orig.shape[0])
        self.canvas.pack()
        self._render_image()

        # Controls
        ttk.Label(self.left_frame, text="Feature A:").grid(row=0, column=0, sticky="w")
        self.var_a = tk.StringVar(value=KP_NAMES[0])
        self.cb_a = ttk.Combobox(self.left_frame, textvariable=self.var_a, values=KP_NAMES, state="readonly")
        self.cb_a.grid(row=1, column=0, sticky="we", pady=2)

        ttk.Label(self.left_frame, text="Feature B:").grid(row=2, column=0, sticky="w")
        self.var_b = tk.StringVar(value=KP_NAMES[1])
        self.cb_b = ttk.Combobox(self.left_frame, textvariable=self.var_b, values=KP_NAMES, state="readonly")
        self.cb_b.grid(row=3, column=0, sticky="we", pady=2)

        self.btn_detect = ttk.Button(self.left_frame, text="Auto-detect keypoints (model)", command=self.auto_detect)
        self.btn_detect.grid(row=4, column=0, pady=6, sticky="we")
        if not self.model:
            self.btn_detect.state(["disabled"])

        self.btn_addmode = ttk.Button(self.left_frame, text="Add point (click image)", command=self.enable_add_mode)
        self.btn_addmode.grid(row=5, column=0, pady=6, sticky="we")

        self.btn_calc = ttk.Button(self.left_frame, text="Calculate distance", command=self.calculate_distance)
        self.btn_calc.grid(row=6, column=0, pady=6, sticky="we")

        self.btn_clear = ttk.Button(self.left_frame, text="Clear points", command=self.clear_points)
        self.btn_clear.grid(row=7, column=0, pady=6, sticky="we")

        self.btn_quit = ttk.Button(self.left_frame, text="Quit", command=self.root.quit)
        self.btn_quit.grid(row=8, column=0, pady=6, sticky="we")

        # info label
        self.info_var = tk.StringVar(value=self._scale_text())
        self.lbl_info = ttk.Label(self.left_frame, textvariable=self.info_var, wraplength=200)
        self.lbl_info.grid(row=9, column=0, pady=10, sticky="w")

        # state
        self.add_mode = False
        self.canvas.bind("<Button-1>", self.on_canvas_click)

    def _scale_text(self):
        if self.scale:
            return f"Detected scale: {self.scale:.4f} cm/px (sticker = {self.sticker_cm} cm)"
        return "Sticker not found — scale unknown."

    def _render_image(self):
        # prepares an RGB image for Tkinter
        vis = self.display.copy()
        # draw sticker box if present
        if getattr(self, "sticker_box", None) is not None:
            cv2.drawContours(vis, [self.sticker_box], -1, (255, 0, 0), 2)
        # draw keypoints
        for name, v in self.kps.items():
            if v is not None:
                x, y, conf = v
                cv2.circle(vis, (int(x), int(y)), 5, (0, 255, 255), -1)
                cv2.putText(vis, name, (int(x) + 6, int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1)
        # convert BGR->RGB for PIL
        vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(vis_rgb)
        self.tkimg = ImageTk.PhotoImage(pil)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tkimg)

    def enable_add_mode(self):
        self.add_mode = True
        messagebox.showinfo("Add point", "Click on the canvas to add a point. Then choose which feature name to assign using the combobox, and click 'Calculate distance'.")

    def on_canvas_click(self, event):
        if not self.add_mode:
            return
        x, y = event.x, event.y
        # get currently selected name (we allow user to select either combobox to assign)
        # prefer the currently focused combobox; otherwise ask user (use A if ambiguous)
        # We'll open a small popup to let user choose which keypoint name to assign to this click
        assign = self._ask_assign_name()
        if assign is None:
            self.add_mode = False
            return
        self.kps[assign] = (float(x), float(y), 1.0)
        self.add_mode = False
        self._render_image()
        self.info_var.set(f"Added point: {assign} @ ({x},{y}). {self._scale_text()}")

    def _ask_assign_name(self):
        # small modal to pick which kp name to assign this click
        pick = tk.Toplevel(self.root)
        pick.title("Assign feature name")
        ttk.Label(pick, text="Assign clicked point to:").pack(padx=10, pady=6)
        var = tk.StringVar(value=KP_NAMES[0])
        cb = ttk.Combobox(pick, textvariable=var, values=KP_NAMES, state="readonly")
        cb.pack(padx=10, pady=6)
        result = {"name": None}

        def on_ok():
            result["name"] = var.get()
            pick.destroy()

        def on_cancel():
            pick.destroy()

        btns = ttk.Frame(pick)
        btns.pack(pady=6)
        ttk.Button(btns, text="OK", command=on_ok).pack(side="left", padx=6)
        ttk.Button(btns, text="Cancel", command=on_cancel).pack(side="left", padx=6)
        pick.grab_set()
        self.root.wait_window(pick)
        return result["name"]

    def auto_detect(self):
        if not self.model:
            messagebox.showwarning("No model", "Model not available.")
            return
        # run model (Ultralytics) on current image
        results = self.model.predict(self.orig, imgsz=640)
        r = results[0]
        kps = extract_kps_from_ultralytics_result(r)
        if kps is None:
            messagebox.showerror("Model", "Model did not return keypoints.")
            return
        # assign first N keypoints to KP_NAMES
        for i, name in enumerate(KP_NAMES):
            if i < kps.shape[0]:
                x, y, conf = float(kps[i, 0]), float(kps[i, 1]), float(kps[i, 2] if kps.shape[1] > 2 else 1.0)
                self.kps[name] = (x, y, conf)
        self._render_image()
        self.info_var.set("Auto-detection finished. " + self._scale_text())

    def calculate_distance(self):
        a = self.var_a.get()
        b = self.var_b.get()
        if a not in self.kps or b not in self.kps:
            messagebox.showerror("Error", "Unknown feature selected.")
            return
        pa = self.kps[a]
        pb = self.kps[b]
        if pa is None or pb is None:
            messagebox.showerror("Missing", "One or both features are not defined. Use Auto-detect or Add point.")
            return
        xa, ya = float(pa[0]), float(pa[1])
        xb, yb = float(pb[0]), float(pb[1])
        px = math.hypot(xa - xb, ya - yb)
        if self.scale:
            cm = px * self.scale
            inch = cm / 2.54
            text = f"{px:.1f} px | {cm:.2f} cm | {inch:.2f} in"
        else:
            cm = None
            inch = None
            text = f"{px:.1f} px (scale unknown)"
        # draw line and label on display image
        self.display = self.orig.copy()
        # redraw all kps as well
        for name, v in self.kps.items():
            if v is not None:
                x, y, conf = v
                cv2.circle(self.display, (int(x), int(y)), 5, (0, 255, 255), -1)
                cv2.putText(self.display, name, (int(x) + 6, int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1)
        pA = (int(xa), int(ya))
        pB = (int(xb), int(yb))
        cv2.line(self.display, pA, pB, (0, 200, 0), 2)
        mid = ((pA[0] + pB[0]) // 2, (pA[1] + pB[1]) // 2)
        cv2.putText(self.display, text, (mid[0] + 8, mid[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)

        # put sticker box if exists
        if getattr(self, "sticker_box", None) is not None:
            cv2.drawContours(self.display, [self.sticker_box], -1, (255, 0, 0), 2)

        # update canvas
        vis_rgb = cv2.cvtColor(self.display, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(vis_rgb)
        self.tkimg = ImageTk.PhotoImage(pil)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tkimg)

        self.info_var.set(f"Distance {a} ↔ {b}: {text}")

    def clear_points(self):
        self.kps = {name: None for name in KP_NAMES}
        self.display = self.orig.copy()
        self._render_image()
        self.info_var.set(self._scale_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", required=False)
    parser.add_argument("--sticker_cm", type=float, default=5.0)
    args = parser.parse_args()

    root = tk.Tk()
    app = Dashboard(root, args.image, sticker_cm=args.sticker_cm, model_path=args.model)
    root.mainloop()


if __name__ == "__main__":
    main()
