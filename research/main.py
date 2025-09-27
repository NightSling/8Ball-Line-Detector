import cv2
import numpy as np


def nothing(x):
    pass


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


def merge_and_select_top_lines(lines, img_width, max_lines=3):
    """Merge lines with similar slopes and select top 3 most prominent."""
    if lines is None:
        return []

    # Convert to slope/intercept
    line_params = []
    for x1, y1, x2, y2 in lines[:, 0]:
        if x2 - x1 == 0:  # vertical line
            slope = np.inf
            intercept = x1
        else:
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
        length = np.hypot(x2 - x1, y2 - y1)
        line_params.append((slope, intercept, length))

    # Merge lines with similar slopes
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
        # Average slope/intercept, sum lengths
        avg_slope = np.mean([l[0] for l in similar])
        avg_intercept = np.mean([l[1] for l in similar])
        total_length = sum([l[2] for l in similar])
        merged.append((avg_slope, avg_intercept, total_length))

    # Sort merged lines by total length descending
    merged.sort(key=lambda x: x[2], reverse=True)

    # Pick top lines with different slopes
    selected = []
    for slope, intercept, _ in merged:
        if all(abs(slope - s[0]) > 0.1 for s in selected):
            selected.append((slope, intercept))
        if len(selected) >= max_lines:
            break

    return selected


# --- Setup ---
cap = cv2.VideoCapture("/dev/video2", cv2.CAP_V4L2)
cv2.namedWindow("Hough Line Tuner", cv2.WINDOW_NORMAL)

# Trackbars
cv2.createTrackbar("Canny Min", "Hough Line Tuner", 120, 255, nothing)
cv2.createTrackbar("Canny Max", "Hough Line Tuner", 180, 255, nothing)
cv2.createTrackbar("Threshold", "Hough Line Tuner", 30, 200, nothing)
cv2.createTrackbar("MinLineLength", "Hough Line Tuner", 23, 500, nothing)
cv2.createTrackbar("MaxLineGap", "Hough Line Tuner", 10, 100, nothing)
cv2.createTrackbar("Gaussian Kernel", "Hough Line Tuner", 5, 31, nothing)

CROP_ROWS = 65

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # --- Crop frame vertically ---
    cropped_frame = frame[CROP_ROWS:frame.shape[0]-CROP_ROWS, :]

    # --- Whitescale filter ---
    mask = cv2.inRange(cropped_frame, (240, 240, 240), (255, 255, 255))
    whitescaled = cv2.bitwise_and(cropped_frame, cropped_frame, mask=mask)
    gray = cv2.cvtColor(whitescaled, cv2.COLOR_BGR2GRAY)

    # --- Trackbar values ---
    cmin = cv2.getTrackbarPos("Canny Min", "Hough Line Tuner")
    cmax = cv2.getTrackbarPos("Canny Max", "Hough Line Tuner")
    thresh = cv2.getTrackbarPos("Threshold", "Hough Line Tuner")
    min_len = cv2.getTrackbarPos("MinLineLength", "Hough Line Tuner")
    max_gap = cv2.getTrackbarPos("MaxLineGap", "Hough Line Tuner")
    gk = cv2.getTrackbarPos("Gaussian Kernel", "Hough Line Tuner")
    if gk % 2 == 0:
        gk += 1
    if gk < 3:
        gk = 3

    # --- Gaussian blur ---
    blurred = cv2.GaussianBlur(gray, (gk, gk), 0)

    # --- Edge detection ---
    edges = cv2.Canny(blurred, cmin, cmax, apertureSize=3)

    # --- Hough transform ---
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, thresh,
                            minLineLength=min_len,
                            maxLineGap=max_gap)

    # --- Merge lines and select top 3 ---
    top_lines = merge_and_select_top_lines(lines, frame.shape[1])

    debug = frame.copy()
    for slope, intercept in top_lines:
        if slope == np.inf:
            x1, y1 = int(intercept), CROP_ROWS
            x2, y2 = int(intercept), frame.shape[0] - CROP_ROWS
        else:
            x1, y1 = 0, int(intercept) + CROP_ROWS
            x2, y2 = frame.shape[1], int(
                slope * frame.shape[1] + intercept) + CROP_ROWS
        cv2.line(debug, (x1, y1), (x2, y2), (0, 0, 255), 3)

    # Stack views: Whitescale | Edges | Original+Lines (cropped)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    combined = np.hstack(
        (whitescaled, edges_bgr, debug[CROP_ROWS:frame.shape[0]-CROP_ROWS, :]))

    cv2.imshow("Hough Line Tuner", combined)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
