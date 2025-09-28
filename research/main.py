import cv2
import numpy as np

# Globals
points = []
setting_boundary = False
roi_bbox = None  # Bounding rectangle of ROI


def nothing(x):
    pass


def mouse_callback(event, x, y, flags, param):
    global points, setting_boundary, roi_bbox
    if setting_boundary and event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        if len(points) == 4:
            roi_bbox = cv2.boundingRect(np.array(points))  # x, y, w, h
            print("ROI set:", roi_bbox)
            setting_boundary = False


def extend_line(x1, y1, x2, y2, width):
    """Extend a line to the image edges along its slope."""
    if x2 - x1 == 0:
        return x1, 0, x2, width  # vertical line
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    new_x1 = 0
    new_y1 = int(intercept)
    new_x2 = width
    new_y2 = int(slope * width + intercept)
    return new_x1, new_y1, new_x2, new_y2


def merge_and_select_top_lines(lines, max_lines=3):
    """Merge lines with similar slopes and select top 3 most prominent."""
    if lines is None:
        return []

    line_params = []
    for x1, y1, x2, y2 in lines[:, 0]:
        if x2 - x1 == 0:
            slope = np.inf
            intercept = x1
        else:
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
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
            if slope_i == np.inf and slope_j == np.inf:
                similar.append((slope_j, intercept_j, length_j))
                used[j] = True
            elif slope_i != np.inf and slope_j != np.inf and abs(slope_i - slope_j) < 0.1:
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


def draw_dotted_line(img, p1, p2, color=(0, 255, 0), thickness=2, gap=10):
    """Draw a dotted line from p1 to p2."""
    dist = int(np.hypot(p2[0]-p1[0], p2[1]-p1[1]))
    if dist == 0:
        return
    for i in range(0, dist, gap*2):
        x_start = int(p1[0] + (p2[0]-p1[0])*(i/dist))
        y_start = int(p1[1] + (p2[1]-p1[1])*(i/dist))
        x_end = int(p1[0] + (p2[0]-p1[0])*((i+gap)/dist))
        y_end = int(p1[1] + (p2[1]-p1[1])*((i+gap)/dist))
        cv2.line(img, (x_start, y_start), (x_end, y_end), color, thickness)


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

print("Press 's' to start setting boundary")

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    display_frame = frame.copy()

    # Draw clicked points
    for p in points:
        cv2.circle(display_frame, p, 5, (0, 255, 0), -1)
    if len(points) == 4:
        cv2.polylines(display_frame, [np.array(points)], True, (0, 255, 0), 2)

    # --- Trackbar values ---
    cmin = cv2.getTrackbarPos("Canny Min", "Hough Line Tuner")
    cmax = cv2.getTrackbarPos("Canny Max", "Hough Line Tuner")
    thresh = cv2.getTrackbarPos("Threshold", "Hough Line Tuner")
    min_len = cv2.getTrackbarPos("MinLineLength", "Hough Line Tuner")
    max_gap = cv2.getTrackbarPos("MaxLineGap", "Hough Line Tuner")
    gk = cv2.getTrackbarPos("Gaussian Kernel", "Hough Line Tuner")
    downscale_percent = cv2.getTrackbarPos("Downscale (%)", "Hough Line Tuner")

    if gk % 2 == 0:
        gk += 1
    if gk < 3:
        gk = 3

    if roi_bbox is not None:
        x, y, w, h = roi_bbox
        roi = frame[y:y+h, x:x+w]

        # Downscale
        scale = downscale_percent / 100.0
        roi_small = cv2.resize(roi, (0, 0), fx=scale, fy=scale)

        # Whitescale filter
        mask = cv2.inRange(roi_small, (240, 240, 240), (255, 255, 255))
        whitescaled = cv2.bitwise_and(roi_small, roi_small, mask=mask)
        gray = cv2.cvtColor(whitescaled, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (gk, gk), 0)
        edges = cv2.Canny(blurred, cmin, cmax, apertureSize=3)

        # Merge and select top lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, thresh,
                                minLineLength=min_len,
                                maxLineGap=max_gap)
        top_lines = merge_and_select_top_lines(lines)

        # Draw main line and reflections
        for slope, intercept in top_lines:
            # Determine main line endpoints
            if slope == np.inf:
                x1, y1 = int(intercept / scale) + x, y
                x2, y2 = int(intercept / scale) + x, y+h
            else:
                ex1, ey1, ex2, ey2 = extend_line(
                    0, intercept, w/scale, slope*(w/scale)+intercept, int(w/scale))
                x1, y1 = int(ex1) + x, int(ey1) + y
                x2, y2 = int(ex2) + x, int(ey2) + y

            # Draw main line dotted red
            draw_dotted_line(display_frame, (x1, y1),
                             (x2, y2), color=(0, 0, 255))

            # Reflection lines (up to 3 bounces)
            cx, cy = x1, y1
            dx, dy = x2 - x1, y2 - y1
            length = (dx**2 + dy**2)**0.5
            if length == 0:
                continue
            dx /= length
            dy /= length

            for _ in range(3):
                # compute intersection distances with walls
                t_vals = []
                if dx > 0:
                    t_vals.append(((x + w) - cx) / dx)
                if dx < 0:
                    t_vals.append((x - cx) / dx)
                if dy > 0:
                    t_vals.append(((y + h) - cy) / dy)
                if dy < 0:
                    t_vals.append((y - cy) / dy)
                t = min([tv for tv in t_vals if tv > 0])
                nx, ny = int(cx + dx * t), int(cy + dy * t)

                # Draw reflection dotted green
                draw_dotted_line(display_frame, (cx, cy),
                                 (nx, ny), color=(0, 255, 0))

                # Reflect direction
                if nx <= x or nx >= x + w:
                    dx = -dx
                if ny <= y or ny >= y + h:
                    dy = -dy
                cx, cy = nx, ny

    cv2.imshow("Hough Line Tuner", display_frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('s'):
        setting_boundary = True
        points = []
        print("Click 4 points on the frame to set boundary")

cap.release()
cv2.destroyAllWindows()
