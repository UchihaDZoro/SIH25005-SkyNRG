#!/usr/bin/env python3
"""
dashboard_top.py

Top-view measurement dashboard (pixel units).

Usage:
    python dashboard_top.py --image top_view.jpg [--model top_view_model.pt]

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

# top-view keypoint names (indices 0..12)
KP_NAMES = [
    "wither",         # 0
    "shoulder_2",     # 1
    "spine",          # 2
    "shoulder_1",     # 3
    "hock_2",         # 4 (user: hock 2)
    "hock_1",         # 5 (user: hock 1)
    "tail_bone_2",    # 6 (user: tail bone 2)
    "tail_bone_1",    # 7 (user: tail bone 1)
    "pin_bone_2",     # 8 (user: pin bone 2)
    "pin_bone_1",     # 9 (user: pin bone 1)
    "hip_bone_2",     # 10 (user: hi
    "spine_bw_hips",  # 11 (kept from previous mapping; adjust if different)
    "hip_bone_1"      # 12 (user: hip bone 1)
]


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
    """Return kps array (N,3) from first detection if present."""
    if getattr(result, "keypoints", None) is None:
        return None
    # take first instance
    xy = result.keypoints.xy[0]    # (N,2)
    conf = result.keypoints.conf[0]  # (N,)
    kps = np.hstack([xy, conf.reshape(-1, 1)])
    return kps

class DashboardTop:
    def __init__(self, root, image_path, model_path=None):
        self.root = root
        self.root.title("Top-view Measurement (pixels)")

        # load image
        self.img_bgr = cv2.imread(image_path)
        if self.img_bgr is None:
            raise RuntimeError(f"Cannot read image {image_path}")
        self.orig = self.img_bgr.copy()
        self.display = self.orig.copy()

        # keypoint storage: name -> (x,y,conf) or None
        self.kps = {name: None for name in KP_NAMES}

        # model (optional)
        self.model = None
        if model_path:
            self.model = try_load_ultralytics(model_path)
            if self.model is None:
                messagebox.showwarning("Model", "Could not load ultralytics model; auto-detect disabled.")
                self.model = None

        # layout
        self.left_frame = ttk.Frame(root, padding=6)
        self.left_frame.grid(row=0, column=0, sticky="nsw")
        self.canvas_frame = ttk.Frame(root, padding=6)
        self.canvas_frame.grid(row=0, column=1)

        # Canvas for image
        h, w = self.orig.shape[:2]
        self.canvas = tk.Canvas(self.canvas_frame, width=w, height=h)
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

        self.info_var = tk.StringVar(value="Units: pixels (no reference sticker).")
        self.lbl_info = ttk.Label(self.left_frame, textvariable=self.info_var, wraplength=220)
        self.lbl_info.grid(row=9, column=0, pady=10, sticky="w")

        # state
        self.add_mode = False
        self.canvas.bind("<Button-1>", self.on_canvas_click)

    def _render_image(self):
        vis = self.display.copy()
        # draw keypoints
        for name, v in self.kps.items():
            if v is not None:
                x, y, conf = v
                cv2.circle(vis, (int(x), int(y)), 5, (0, 255, 255), -1)
                cv2.putText(vis, name, (int(x) + 6, int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1)
        vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(vis_rgb)
        self.tkimg = ImageTk.PhotoImage(pil)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tkimg)

    def enable_add_mode(self):
        self.add_mode = True
        messagebox.showinfo("Add point", "Click on the image to add a point, then assign it to a feature name.")

    def on_canvas_click(self, event):
        if not self.add_mode:
            return
        x, y = event.x, event.y
        assign = self._ask_assign_name()
        if assign is None:
            self.add_mode = False
            return
        self.kps[assign] = (float(x), float(y), 1.0)
        self.add_mode = False
        self.display = self.orig.copy()
        self._render_image()
        self.info_var.set(f"Added {assign} @ ({x},{y}). Units: pixels.")

    def _ask_assign_name(self):
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
        results = self.model.predict(self.orig, imgsz=640)
        r = results[0]
        kps = extract_kps_from_ultralytics_result(r)
        if kps is None:
            messagebox.showerror("Model", "Model did not return keypoints.")
            return
        for i, name in enumerate(KP_NAMES):
            if i < kps.shape[0]:
                x, y, conf = float(kps[i, 0]), float(kps[i, 1]), float(kps[i, 2] if kps.shape[1] > 2 else 1.0)
                self.kps[name] = (x, y, conf)
        self.display = self.orig.copy()
        self._render_image()
        self.info_var.set("Auto-detection finished. Units: pixels.")

    def calculate_distance(self):
        a = self.var_a.get()
        b = self.var_b.get()
        pa = self.kps.get(a)
        pb = self.kps.get(b)
        if pa is None or pb is None:
            messagebox.showerror("Missing", "One or both features are not defined. Use Auto-detect or Add point.")
            return
        xa, ya = float(pa[0]), float(pa[1])
        xb, yb = float(pb[0]), float(pb[1])
        px = math.hypot(xa - xb, ya - yb)
        text = f"{px:.1f} px"
        # draw line and label
        self.display = self.orig.copy()
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
        self.info_var.set("Cleared points. Units: pixels.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", required=False)
    args = parser.parse_args()

    root = tk.Tk()
    app = DashboardTop(root, args.image, model_path=args.model)
    root.mainloop()

if __name__ == "__main__":
    main()
