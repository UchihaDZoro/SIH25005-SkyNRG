import cv2
import numpy as np
import math

# Your image
IMAGE = "poc2.jpg"
STICKER_CM = 5.3  # 5 × 5 cm

# HSV threshold for bright green sticker
LOWER = np.array([40, 70, 70])
UPPER = np.array([90, 255, 255])

# storage for clicked points
points = []
scale_cm_per_px = None

def detect_scale(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER, UPPER)

    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, mask, None

    # biggest green area = sticker
    c = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect).astype(int)
    w, h = rect[1]
    px_len = max(w, h)

    if px_len < 1:
        return None, mask, box

    scale = STICKER_CM / px_len
    return scale, mask, box


def on_click(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))

        # keep only 2 points
        if len(points) > 2:
            points = [(x, y)]


# ---------------------- MAIN LOOP ----------------------

img_orig = cv2.imread(IMAGE)
if img_orig is None:
    print("ERROR: image not found.")
    exit()

cv2.namedWindow("Measure")
cv2.setMouseCallback("Measure", on_click)

while True:
    img = img_orig.copy()

    # detect sticker every frame
    scale_cm_per_px, mask, box = detect_scale(img_orig)

    if box is not None:
        cv2.drawContours(img, [box], -1, (255, 0, 0), 2)
        if scale_cm_per_px:
            cv2.putText(img, f"Scale: {scale_cm_per_px:.4f} cm/px",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 0, 0), 2)
    else:
        cv2.putText(img, "Sticker NOT detected",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255), 2)

    # draw points + distance
    if len(points) >= 1:
        cv2.circle(img, points[0], 6, (0,255,255), -1)

    if len(points) == 2:
        p1, p2 = points
        cv2.circle(img, p1, 6, (0,255,255), -1)
        cv2.circle(img, p2, 6, (0,255,255), -1)
        cv2.line(img, p1, p2, (0,255,0), 2)

        px = math.dist(p1, p2)

        if scale_cm_per_px:
            cm = px * scale_cm_per_px
            text = f"{cm:.2f} cm"
        else:
            text = f"{px:.1f} px"

        mid = ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2)
        cv2.putText(img, text, (mid[0]+10, mid[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0), 2)

    cv2.imshow("Measure", img)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break
    if key == ord('c'):
        points = []  # clear clicks

cv2.destroyAllWindows()
