#!/usr/bin/env python3
"""
udder_view.py

Udder keypoint dashboard — improved auto-detection for overlapping teats.

Usage:
    python udder_view.py --image cow_udder.jpg --model udder_view_model.pt [--sticker_cm 21]

Notes:
- Model expected to produce 4 detections each with 2 keypoints (total 8 points).
- This version attempts to use detection class IDs if present, otherwise flattens
  and spatially orders points. If teats overlap, use "Auto-order points" or manual add.
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
    "pt_1", "pt_2",
    "pt_3", "pt_4",
    "pt_5", "pt_6",
    "pt_7", "pt_8"
]

DISPLAY_SIZE = 640
HSV_LOWER = np.array([25, 60, 60])
HSV_UPPER = np.array([90, 255, 255])

PALETTES = {
    "Teal":   {"line": (0,160,150), "text": (20,20,20)},
    "Blue":   {"line": (10,120,200), "text": (20,20,20)},
    "Orange": {"line": (230,120,20), "text": (20,20,20)},
    "Gray":   {"line": (80,80,80), "text": (10,10,10)},
}

# If your model uses class IDs to indicate which pair belongs to which anatomical position,
# configure this mapping: class_id -> index of KP_NAMES pair base (0 means pt_1/pt_2, 1 means pt_3/pt_4, ...)
# For example: {0:0, 1:1, 2:2, 3:3}
CLASS_TO_PAIR_INDEX = None  # set to None to use default sequential mapping


# ---------------- utilities ----------------

def letterbox(img, new=DISPLAY_SIZE, color=(114,114,114)):
    h, w = img.shape[:2]
    scale = new / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new, new, 3), color, dtype=np.uint8)
    top = (new - nh) // 2
    left = (new - nw) // 2
    canvas[top:top+nh, left:left+nw] = resized
    return canvas, scale, left, top


def detect_sticker_scale(img, sticker_cm=21):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None
    c = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    w, h = rect[1]
    px = max(w, h)
    if px < 1:
        return None, None
    box = cv2.boxPoints(rect).astype(int)
    cm_per_px = sticker_cm / px
    return cm_per_px, box


def px_dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def spatial_order_points(pts):
    """
    Deterministic spatial ordering for a list of (x,y) points.
    Strategy: sort by x ascending (left->right) then by y ascending (top->bottom).
    Returns list of points sorted.
    """
    return sorted(pts, key=lambda p: (p[0], p[1]))


# ---------------- dashboard ----------------

class UdderDashboard:
    def __init__(self, root, image_path, model_path=None, sticker_cm=None):
        self.root = root
        self.root.title("Udder Keypoint Dashboard (robust auto-detect)")

        img = cv2.imread(image_path)
        if img is None:
            raise RuntimeError("Cannot read image")

        # letterbox to DISPLAY_SIZE × DISPLAY_SIZE
        self.orig, _, _, _ = letterbox(img, DISPLAY_SIZE)

        # sticker scale
        self.scale_cm_per_px = None
        self.sticker_box = None
        if sticker_cm:
            scale, box = detect_sticker_scale(self.orig, sticker_cm)
            self.scale_cm_per_px = scale
            self.sticker_box = box

        # keypoints storage
        self.kp = {k: None for k in KP_NAMES}

        # try load ultralytics model
        self.model = None
        if model_path:
            try:
                from ultralytics import YOLO
                self.model = YOLO(model_path)
            except Exception as e:
                messagebox.showwarning("Model load", f"Could not load model: {e}")
                self.model = None

        # UI layout
        left = ttk.Frame(root, padding=8)
        left.grid(row=0, column=0, sticky="ns")
        right = ttk.Frame(root, padding=8)
        right.grid(row=0, column=1)

        self.canvas = tk.Canvas(right, width=DISPLAY_SIZE, height=DISPLAY_SIZE)
        self.canvas.pack()
        self.display = self.orig.copy()
        self._update_canvas()

        # controls
        ttk.Label(left, text="Feature A:").pack(anchor="w")
        self.var_a = tk.StringVar(value=KP_NAMES[0])
        ttk.Combobox(left, textvariable=self.var_a, values=KP_NAMES, state="readonly").pack(fill="x")

        ttk.Label(left, text="Feature B:").pack(anchor="w", pady=(6,0))
        self.var_b = tk.StringVar(value=KP_NAMES[1])
        ttk.Combobox(left, textvariable=self.var_b, values=KP_NAMES, state="readonly").pack(fill="x")

        ttk.Separator(left).pack(fill="x", pady=8)

        ttk.Button(left, text="Auto-detect keypoints", command=self.auto_detect).pack(fill="x", pady=3)
        ttk.Button(left, text="Auto-order points (spatial)", command=self.auto_order_points).pack(fill="x", pady=3)
        ttk.Button(left, text="Add point (click)", command=self.enable_add).pack(fill="x", pady=3)
        ttk.Button(left, text="Calculate distance", command=self.calculate).pack(fill="x", pady=3)
        ttk.Button(left, text="Clear", command=self.clear).pack(fill="x", pady=3)

        ttk.Separator(left).pack(fill="x", pady=8)
        ttk.Label(left, text="Color theme:").pack(anchor="w")
        self.color_var = tk.StringVar(value="Teal")
        for p in PALETTES:
            ttk.Radiobutton(left, text=p, variable=self.color_var, value=p).pack(anchor="w")

        self.info = tk.StringVar(value="Units: px" if self.scale_cm_per_px is None else f"{self.scale_cm_per_px:.4f} cm/px")
        ttk.Label(left, textvariable=self.info, wraplength=220).pack(anchor="w", pady=(8,0))

        ttk.Button(left, text="Quit", command=root.quit).pack(fill="x", pady=6)

        self.add_mode = False
        self.canvas.bind("<Button-1>", self._on_click)

    # ---------- canvas helper ----------
    def _update_canvas(self):
        rgb = cv2.cvtColor(self.display, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        self.tkimg = ImageTk.PhotoImage(pil)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tkimg)

    # ---------- manual add ----------
    def enable_add(self):
        self.add_mode = True
        messagebox.showinfo("Add", "Click on the image to add a point, then assign the point name.")

    def _on_click(self, event):
        if not self.add_mode:
            return
        x, y = event.x, event.y
        name = self._assign_dialog()
        if name:
            self.kp[name] = (float(x), float(y))
        self.add_mode = False
        self.refresh()

    def _assign_dialog(self):
        win = tk.Toplevel(self.root)
        win.title("Assign")
        ttk.Label(win, text="Assign to:").pack(padx=10, pady=6)
        var = tk.StringVar(value=KP_NAMES[0])
        cb = ttk.Combobox(win, textvariable=var, values=KP_NAMES, state="readonly")
        cb.pack(padx=10, pady=6)
        out = {"v": None}
        def ok(): out["v"] = var.get(); win.destroy()
        ttk.Button(win, text="OK", command=ok).pack(side="left", padx=10, pady=6)
        ttk.Button(win, text="Cancel", command=win.destroy).pack(side="left", padx=10, pady=6)
        win.grab_set()
        self.root.wait_window(win)
        return out["v"]

    # ---------- auto-detect (robust) ----------
    def auto_detect(self):
        if self.model is None:
            messagebox.showerror("Model", "No model loaded.")
            return

        results = self.model.predict(self.orig, imgsz=640)
        if len(results) == 0:
            messagebox.showerror("Model", "No detections found.")
            return

        r = results[0]
        det = getattr(r, "keypoints", None)
        if det is None or getattr(det, "xy", None) is None:
            messagebox.showerror("Model", "Model returned no keypoints.")
            return

        # Strategy A: try to use detection class ids if available and mapping provided
        used_strategy = None
        merged = [None] * len(KP_NAMES)

        # attempt to fetch class ids for each instance (r.boxes.cls)
        classes = None
        try:
            boxes = getattr(r, "boxes", None)
            if boxes is not None and getattr(boxes, "cls", None) is not None:
                classes = [int(x) for x in boxes.cls.cpu().numpy()] if hasattr(boxes.cls, "cpu") else [int(x) for x in boxes.cls]
        except Exception:
            classes = None

        if classes is not None and CLASS_TO_PAIR_INDEX is not None:
            # use the mapping provided by user to place keypoint pairs
            for i in range(len(det.xy)):
                cls_id = classes[i] if i < len(classes) else None
                kps_xy = det.xy[i]  # shape (2,2)
                base_idx = CLASS_TO_PAIR_INDEX.get(cls_id, None)
                if base_idx is not None:
                    # place two keypoints into merged at indices base_idx*2 and base_idx*2+1
                    for j, p in enumerate(kps_xy):
                        tgt_idx = base_idx*2 + j
                        if tgt_idx < len(merged):
                            merged[tgt_idx] = (float(p[0]), float(p[1]))
            used_strategy = "class_map"

        if not any(merged):
            # Strategy B: flatten all detections (in detection order)
            flat = []
            for i in range(len(det.xy)):
                for p in det.xy[i]:
                    flat.append((float(p[0]), float(p[1])))
            if len(flat) >= len(KP_NAMES):
                for i in range(len(KP_NAMES)):
                    merged[i] = flat[i]
                used_strategy = "flatten_first8"
            else:
                # Strategy C: flatten then spatial-order to fill missing
                # we will spatially sort points and assign in order
                if len(flat) > 0:
                    sorted_pts = spatial_order_points(flat)
                    for i in range(min(len(sorted_pts), len(KP_NAMES))):
                        merged[i] = sorted_pts[i]
                    used_strategy = "flatten_spatial"
                else:
                    used_strategy = "none"

        # finalize merged into kp map
        for idx, name in enumerate(KP_NAMES):
            self.kp[name] = merged[idx] if merged[idx] is not None else None

        # feedback for debugging
        print("[auto_detect] strategy:", used_strategy)
        total_pts = sum(1 for v in self.kp.values() if v is not None)
        print(f"[auto_detect] assigned {total_pts}/{len(KP_NAMES)} keypoints")

        self.refresh()
        self.info.set(f"Auto-detect ({used_strategy}), assigned {total_pts}/{len(KP_NAMES)}")

    # ---------- auto-order (spatial) ----------
    def auto_order_points(self):
        # collect existing points
        pts = [v for v in self.kp.values() if v is not None]
        if not pts:
            messagebox.showinfo("Auto-order", "No points to order.")
            return
        sorted_pts = spatial_order_points(pts)
        # map sorted into kp list
        for i, name in enumerate(KP_NAMES):
            if i < len(sorted_pts):
                self.kp[name] = sorted_pts[i]
            else:
                self.kp[name] = None
        self.refresh()
        self.info.set("Auto-ordered points spatially (left->right, top->bottom).")

    # ---------- calculate ----------
    def calculate(self):
        a = self.var_a.get()
        b = self.var_b.get()
        pa = self.kp.get(a)
        pb = self.kp.get(b)
        if pa is None or pb is None:
            messagebox.showerror("Missing", "Select two valid points.")
            return
        px = px_dist(pa, pb)
        if self.scale_cm_per_px:
            cm = px * self.scale_cm_per_px
            info = f"{px:.1f} px  |  {cm:.2f} cm"
        else:
            info = f"{px:.1f} px"
        # draw overlay
        pal = PALETTES[self.color_var.get()]
        vis = self.orig.copy()
        # draw all points
        for n, v in self.kp.items():
            if v:
                cv2.circle(vis, (int(v[0]), int(v[1])), 5, (0,255,255), -1)
                cv2.putText(vis, n, (int(v[0])+6, int(v[1])-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, pal["text"], 1)
        axy, bxy = (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1]))
        cv2.line(vis, axy, bxy, pal["line"], 2)
        mid = ((axy[0] + bxy[0])//2, (axy[1] + bxy[1])//2)
        cv2.putText(vis, info, (mid[0]+10, mid[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, pal["text"], 2)
        self.display = vis
        self._update_canvas()
        self.info.set("Distance: " + info)

    # ---------- clear / refresh ----------
    def clear(self):
        self.kp = {k: None for k in KP_NAMES}
        self.refresh()

    def refresh(self):
        pal = PALETTES[self.color_var.get()]
        vis = self.orig.copy()
        # draw sticker if present
        if self.sticker_box is not None:
            cv2.drawContours(vis, [self.sticker_box], -1, (0,0,255), 2)
            if self.scale_cm_per_px:
                cv2.putText(vis, f"scale: {self.scale_cm_per_px:.4f} cm/px", (8,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        # draw keypoints
        for n, v in self.kp.items():
            if v:
                cv2.circle(vis, (int(v[0]), int(v[1])), 5, (0,255,255), -1)
                cv2.putText(vis, n, (int(v[0])+6, int(v[1])-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, pal["text"], 1)
        self.display = vis
        self._update_canvas()
        if self.scale_cm_per_px:
            self.info.set(f"{self.scale_cm_per_px:.4f} cm/px")
        else:
            self.info.set("Units: px")


# ---------- main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", required=False)
    parser.add_argument("--sticker_cm", type=float, required=False)
    parser.add_argument("--class_map", required=False,
                        help="Optional mapping for class IDs to pair index, format 'cls0:0,cls1:1,cls2:2,cls3:3'")
    args = parser.parse_args()

    # parse optional class_map
    global CLASS_TO_PAIR_INDEX
    if args.class_map:
        m = {}
        for token in args.class_map.split(","):
            if ":" in token:
                k, v = token.split(":")
                try:
                    m[int(k)] = int(v)
                except:
                    pass
        if m:
            CLASS_TO_PAIR_INDEX = m
            print("[main] using CLASS_TO_PAIR_INDEX:", CLASS_TO_PAIR_INDEX)

    root = tk.Tk()
    UdderDashboard(root, args.image, args.model, args.sticker_cm)
    root.mainloop()


if __name__ == "__main__":
    main()
