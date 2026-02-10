#!/usr/bin/env python3
"""
cow_measure.py

- Loads 'cow.jpg' from the same directory.
- Detects a roughly square yellow/green sticker (assumed 3" x 3").
- Shows mask and detection windows.
- Click two points in the detection window to measure distance (pixel, cm, in).
- Press 'r' to re-run detection (useful after moving sliders / changing lighting externally).
- Press 'q' or ESC to quit and save 'measured_output.jpg'.
"""

import cv2
import numpy as np
import math
import sys

IMAGE_NAME = "cow.jpg"
OUTPUT_NAME = "measured_output.jpg"
KNOWN_STICKER_INCH = 7.62
KNOWN_STICKER_CM = KNOWN_STICKER_INCH * 2.54  # 7.62 cm

# HSV defaults for yellow / lime sticker - tune if needed
HSV_LOWER = np.array([25, 60, 60], dtype=np.uint8)
HSV_UPPER = np.array([80, 255, 255], dtype=np.uint8)


def get_sticker_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def find_best_sticker(mask, min_area=200):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # pick largest contour above area threshold
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)  # ((cx,cy),(w,h),angle)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    return rect, box


def pixel_distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


class MeasureApp:
    def __init__(self, img_bgr):
        self.orig = img_bgr.copy()
        self.display = img_bgr.copy()
        self.mask = get_sticker_mask(self.orig)
        found = find_best_sticker(self.mask)
        self.scale_cm_per_px = None
        self.sticker_rect = None
        self.sticker_box = None
        if found is not None:
            rect, box = found
            self.sticker_rect = rect
            self.sticker_box = box
            # use longer side of rect as sticker pixel size (approx)
            w, h = rect[1]
            sticker_px = max(w, h) if max(w, h) > 0 else 1.0
            self.scale_cm_per_px = KNOWN_STICKER_CM / sticker_px
        self.points = []
        self.last_image = self.display.copy()
        self.update_overlay()

    def update_overlay(self):
        disp = self.orig.copy()
        # draw sticker box if detected
        if self.sticker_box is not None:
            cv2.drawContours(disp, [self.sticker_box], -1, (0, 0, 255), 2)
            cx, cy = int(self.sticker_rect[0][0]), int(self.sticker_rect[0][1])
            w, h = self.sticker_rect[1]
            info = f"Sticker px (long side): {max(w,h):.1f}px"
            cv2.putText(disp, info, (max(10, cx+10), max(30, cy-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            if self.scale_cm_per_px is not None:
                cv2.putText(disp, f"Scale: {self.scale_cm_per_px:.4f} cm/px",
                            (max(10, cx+10), max(60, cy+10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        else:
            cv2.putText(disp, "Sticker not detected - distance will be pixels only",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # draw clicked points and measured line if 2 points exist
        for p in self.points:
            cv2.circle(disp, p, 6, (0, 255, 255), -1)
        if len(self.points) == 2:
            p1, p2 = self.points
            px = pixel_distance(p1, p2)
            if self.scale_cm_per_px is not None:
                real_cm = px * self.scale_cm_per_px
                real_in = real_cm / 2.54
                info = f"{px:.1f}px  |  {real_cm:.2f} cm  |  {real_in:.2f} in"
            else:
                info = f"{px:.1f}px  |  scale unknown"
            cv2.line(disp, p1, p2, (0, 255, 0), 2)
            mid = (int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2))
            cv2.putText(disp, info, (mid[0] + 10, mid[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # also show a small mask inset (resized) at top-left for reference
        try:
            mask_color = cv2.cvtColor(self.mask, cv2.COLOR_GRAY2BGR)
            h_m, w_m = mask_color.shape[:2]
            scale_factor = 0.25
            small = cv2.resize(mask_color, (int(w_m * scale_factor), int(h_m * scale_factor)))
            sh, sw = small.shape[:2]
            disp[10:10+sh, 10:10+sw] = cv2.addWeighted(disp[10:10+sh, 10:10+sw], 0.6, small, 0.4, 0)
        except Exception:
            pass

        self.last_image = disp

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) >= 2:
                self.points = []
            self.points.append((x, y))
            self.update_overlay()
        elif event == cv2.EVENT_RBUTTONDOWN:
            # right-click to clear
            self.points = []
            self.update_overlay()

    def re_detect(self):
        # re-compute mask and sticker (useful if you changed HSV externally)
        self.mask = get_sticker_mask(self.orig)
        found = find_best_sticker(self.mask)
        self.scale_cm_per_px = None
        self.sticker_rect = None
        self.sticker_box = None
        if found is not None:
            rect, box = found
            self.sticker_rect = rect
            self.sticker_box = box
            w, h = rect[1]
            sticker_px = max(w, h) if max(w, h) > 0 else 1.0
            self.scale_cm_per_px = KNOWN_STICKER_CM / sticker_px
        self.update_overlay()

    def run(self):
        WIN_DET = "Detected Sticker"
        WIN_MASK = "Sticker Mask"
        cv2.namedWindow(WIN_DET, cv2.WINDOW_NORMAL)
        cv2.namedWindow(WIN_MASK, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WIN_DET, self.on_mouse)

        print("Instructions:")
        print("- Left-click two points to measure distance.")
        print("- Right-click to clear points.")
        print("- Press 'r' to re-run sticker detection.")
        print("- Press 'q' or ESC to quit and save the result to", OUTPUT_NAME)

        while True:
            cv2.imshow(WIN_MASK, self.mask)
            cv2.imshow(WIN_DET, self.last_image)
            key = cv2.waitKey(20) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('r'):
                self.re_detect()

        # on exit save final annotated image
        cv2.imwrite(OUTPUT_NAME, self.last_image)
        print("Saved annotated image to", OUTPUT_NAME)
        cv2.destroyAllWindows()


def main():
    img = cv2.imread(IMAGE_NAME)
    if img is None:
        print(f"Error: cannot read '{IMAGE_NAME}'. Ensure image is in same directory and named exactly.")
        sys.exit(1)

    app = MeasureApp(img)
    app.run()


if __name__ == "__main__":
    main()
