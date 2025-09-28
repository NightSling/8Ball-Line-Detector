import cv2
import numpy as np

# Globals
roi_bbox = None
click_stage = 0  # 0 = waiting for first click, 1 = waiting for second click
start_point = None
current_mouse = None
show_reflections = True  # Toggle with 'b'


def nothing(x):
    pass


def mouse_callback(event, x, y, flags, param):
    global click_stage, start_point, roi_bbox, current_mouse
    current_mouse = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        if click_stage == 0:
            start_point = (x, y)
            click_stage = 1
            roi_bbox = None
        elif click_stage == 1:
            x1, y1 = start_point
            x2, y2 = x, y
            roi_bbox = (min(x1, x2), min(y1, y2),
                        max(1, abs(x2 - x1)), max(1, abs(y2 - y1)))
            click_stage = 0


def draw_dotted_line(img, p1, p2, color=(0, 255, 0), thickness=2, gap=10):
    dist = int(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
    if dist == 0:
        return
    for i in range(0, dist, gap * 2):
        x_start = int(p1[0] + (p2[0] - p1[0]) * (i / dist))
        y_start = int(p1[1] + (p2[1] - p1[1]) * (i / dist))
        x_end = int(p1[0] + (p2[0] - p1[0]) * ((i + gap) / dist))
        y_end = int(p1[1] + (p2[1] - p1[1]) * ((i + gap) / dist))
        cv2.line(img, (x_start, y_start), (x_end, y_end), color, thickness)


def merge_and_select_top_lines(lines, max_lines=3):
    if lines is None:
        return []

    line_params = []
    for x1, y1, x2, y2 in lines[:, 0]:
        slope = np.inf if x2 - x1 == 0 else (y2 - y1) / (x2 - x1)
        intercept = x1 if slope == np.inf else y1 - slope * x1
        length = np.hypot(x2 - x1, y2 - y1)
        line_params.append((slope, intercept, length))

    merged = []
    used = [False] * len(line_params)

    for i, (slope_i, intercept_i, length_i) in enumerate(line_params):
        if used[i]:
            continue
        similar = [(slope_i, intercept_i, length_i)]
        used[i] = True
        for j, (slope_j, intercept_j, length_j) in enumerate(line_params):
            if used[j]:
                continue
            if slope_i == np.inf and slope_j == np.inf or \
               (slope_i != np.inf and slope_j != np.inf and abs(slope_i - slope_j) < 0.1):
                similar.append((slope_j, intercept_j, length_j))
                used[j] = True
        avg_slope = np.mean([l[0] for l in similar])
        avg_intercept = np.mean([l[1] for l in similar])
        total_length = sum([l[2] for l in similar])
        merged.append((avg_slope, avg_intercept, total_length))

    merged.sort(key=lambda x: x[2], reverse=True)

    selected = []
    for slope, intercept, _ in merged:
        if all(abs(slope - s[0]) > 0.1 for s in selected):
            selected.append((slope, intercept))
        if len(selected) >= max_lines:
            break
    return selected


def draw_reflections(img, p1, p2, bbox, color, count=3):
    x, y, w, h = bbox
    cx, cy = p1
    dx_orig, dy_orig = p2[0] - p1[0], p2[1] - p1[1]
    length = np.hypot(dx_orig, dy_orig)
    if length == 0:
        return
    for _ in range(count):
        dx, dy = dx_orig / length, dy_orig / length
        t_vals = []
        if dx > 0:
            t_vals.append((x + w - cx) / dx)
        if dx < 0:
            t_vals.append((x - cx) / dx)
        if dy > 0:
            t_vals.append((y + h - cy) / dy)
        if dy < 0:
            t_vals.append((y - cy) / dy)
        t = min([tv for tv in t_vals if tv > 0])
        nx, ny = int(cx + dx * t), int(cy + dy * t)
        draw_dotted_line(img, (cx, cy), (nx, ny), color)
        if nx <= x or nx >= x + w:
            dx_orig = -dx_orig
        if ny <= y or ny >= y + h:
            dy_orig = -dy_orig
        cx, cy = nx, ny


# --- Setup ---
cap = cv2.VideoCapture("/dev/video2", cv2.CAP_V4L2)
cv2.namedWindow("Hough Line Tuner", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Hough Line Tuner", mouse_callback)

# Trackbars
cv2.createTrackbar("Canny Min", "Hough Line Tuner", 120, 255, nothing)
cv2.createTrackbar("Canny Max", "Hough Line Tuner", 180, 255, nothing)
cv2.createTrackbar("Threshold", "Hough Line Tuner", 30, 200, nothing)
cv2.createTrackbar("MinLineLength", "Hough Line Tuner", 23, 500, nothing)
cv2.createTrackbar("MaxLineGap", "Hough Line Tuner", 10, 100, nothing)
cv2.createTrackbar("Gaussian Kernel", "Hough Line Tuner", 5, 31, nothing)
cv2.createTrackbar("Downscale (%)", "Hough Line Tuner", 100, 100, nothing)

print("Click once to start ROI, move mouse to see rectangle, click again to finalize.")
print("Press 'b' to toggle reflections, 'q' to quit.")

colors = [(0, 215, 255),  # gold
          (255, 0, 0),    # blue
          (0, 0, 255)]    # red

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    display_frame = frame.copy()

    # Draw temporary ROI
    if click_stage == 1 and start_point and current_mouse:
        cv2.rectangle(display_frame, start_point,
                      current_mouse, (0, 255, 0), 2)

    if roi_bbox:
        x, y, w, h = roi_bbox
        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Trackbar values
        cmin = cv2.getTrackbarPos("Canny Min", "Hough Line Tuner")
        cmax = cv2.getTrackbarPos("Canny Max", "Hough Line Tuner")
        thresh = cv2.getTrackbarPos("Threshold", "Hough Line Tuner")
        min_len = cv2.getTrackbarPos("MinLineLength", "Hough Line Tuner")
        max_gap = cv2.getTrackbarPos("MaxLineGap", "Hough Line Tuner")
        gk = cv2.getTrackbarPos("Gaussian Kernel", "Hough Line Tuner")
        downscale_percent = cv2.getTrackbarPos(
            "Downscale (%)", "Hough Line Tuner")

        gk = max(3, gk | 1)  # Ensure odd and >=3
        scale = max(0.01, downscale_percent / 100.0)

        roi = frame[y:y+h, x:x+w]
        if roi is None or roi.size == 0:
            continue
        roi_small = cv2.resize(roi, (0, 0), fx=scale, fy=scale)

        mask = cv2.inRange(roi_small, (240, 240, 240), (255, 255, 255))
        whitescaled = cv2.bitwise_and(roi_small, roi_small, mask=mask)
        gray = cv2.cvtColor(whitescaled, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (gk, gk), 0)
        edges = cv2.Canny(blurred, cmin, cmax, apertureSize=3)

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, thresh,
                                minLineLength=min_len, maxLineGap=max_gap)
        top_lines = merge_and_select_top_lines(lines)

        for idx, (slope, intercept) in enumerate(top_lines):
            color = colors[idx % len(colors)]
            if slope == np.inf:
                x1, y1 = int(intercept / scale) + x, y
                x2, y2 = int(intercept / scale) + x, y + h
            else:
                ex1, ey1 = 0, intercept
                ex2, ey2 = w / scale, slope * (w / scale) + intercept
                x1, y1 = int(ex1) + x, int(ey1) + y
                x2, y2 = int(ex2) + x, int(ey2) + y

            draw_dotted_line(display_frame, (x1, y1), (x2, y2), color)

            if show_reflections:
                draw_reflections(display_frame, (x1, y1),
                                 (x2, y2), roi_bbox, color)
                draw_reflections(display_frame, (x2, y2),
                                 (x1, y1), roi_bbox, color)

    cv2.imshow("Hough Line Tuner", display_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('b'):
        show_reflections = not show_reflections

cap.release()
cv2.destroyAllWindows()
