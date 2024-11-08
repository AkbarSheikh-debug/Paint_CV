import cv2
import numpy as np
import mediapipe as mp
from collections import deque

# Initialize different color queues
blue_points = [deque(maxlen=1024)]
green_points = [deque(maxlen=1024)]
red_points = [deque(maxlen=1024)]
yellow_points = [deque(maxlen=1024)]

# Index for each color queue to track points
index_blue = 0
index_green = 0
index_red = 0
index_yellow = 0

# Kernel for dilation
dilation_kernel = np.ones((5, 5), np.uint8)

# Color palette in BGR format
colors_palette = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
current_color_index = 0  # Initial color is blue

# Setup white canvas with color selector buttons
canvas = np.full((471, 636, 3), 255, dtype=np.uint8)
canvas = cv2.rectangle(canvas, (40, 1), (140, 65), (0, 0, 0), 2)
canvas = cv2.rectangle(canvas, (160, 1), (255, 65), (255, 0, 0), 2)
canvas = cv2.rectangle(canvas, (275, 1), (370, 65), (0, 255, 0), 2)
canvas = cv2.rectangle(canvas, (390, 1), (485, 65), (0, 0, 255), 2)
canvas = cv2.rectangle(canvas, (505, 1), (600, 65), (0, 255, 255), 2)

cv2.putText(canvas, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(canvas, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(canvas, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(canvas, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(canvas, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

# Initialize Mediapipe Hands and setup camera capture
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
drawing_utils = mp.solutions.drawing_utils

capture = cv2.VideoCapture(0)

while True:
    ret, frame = capture.read()
    if not ret:
        break

    # Prepare frame and canvas for processing
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Draw control panel on frame
    cv2.rectangle(frame, (40, 1), (140, 65), (0, 0, 0), 2)
    cv2.rectangle(frame, (160, 1), (255, 65), (255, 0, 0), 2)
    cv2.rectangle(frame, (275, 1), (370, 65), (0, 255, 0), 2)
    cv2.rectangle(frame, (390, 1), (485, 65), (0, 0, 255), 2)
    cv2.rectangle(frame, (505, 1), (600, 65), (0, 255, 255), 2)
    cv2.putText(frame, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

    # Detect hand landmarks
    hand_results = hands_detector.process(rgb_frame)

    if hand_results.multi_hand_landmarks:
        landmarks = []
        for hand_lms in hand_results.multi_hand_landmarks:
            for lm in hand_lms.landmark:
                lmx, lmy = int(lm.x * 640), int(lm.y * 480)
                landmarks.append([lmx, lmy])

            # Draw detected landmarks
            drawing_utils.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

        fingertip = (landmarks[8][0], landmarks[8][1])
        thumb_tip = (landmarks[4][0], landmarks[4][1])

        # Check finger distance from thumb for drawing
        if thumb_tip[1] - fingertip[1] < 30:
            blue_points.append(deque(maxlen=512))
            green_points.append(deque(maxlen=512))
            red_points.append(deque(maxlen=512))
            yellow_points.append(deque(maxlen=512))
            index_blue += 1
            index_green += 1
            index_red += 1
            index_yellow += 1
        elif fingertip[1] <= 65:
            if 40 <= fingertip[0] <= 140:
                blue_points = [deque(maxlen=512)]
                green_points = [deque(maxlen=512)]
                red_points = [deque(maxlen=512)]
                yellow_points = [deque(maxlen=512)]
                index_blue = index_green = index_red = index_yellow = 0
                canvas[67:, :, :] = 255
            elif 160 <= fingertip[0] <= 255:
                current_color_index = 0
            elif 275 <= fingertip[0] <= 370:
                current_color_index = 1
            elif 390 <= fingertip[0] <= 485:
                current_color_index = 2
            elif 505 <= fingertip[0] <= 600:
                current_color_index = 3
        else:
            color_queues = [blue_points, green_points, red_points, yellow_points]
            color_queues[current_color_index][
                index_blue if current_color_index == 0 else index_green if current_color_index == 1 else index_red if current_color_index == 2 else index_yellow].appendleft(
                fingertip)

    # Draw lines for each color
    points = [blue_points, green_points, red_points, yellow_points]
    for i, point_set in enumerate(points):
        for pts in point_set:
            for k in range(1, len(pts)):
                if pts[k - 1] is None or pts[k] is None:
                    continue
                cv2.line(frame, pts[k - 1], pts[k], colors_palette[i], 2)
                cv2.line(canvas, pts[k - 1], pts[k], colors_palette[i], 2)

    # Display the frame and canvas
    cv2.imshow("Frame", frame)
    cv2.imshow("Canvas", canvas)

    if cv2.waitKey(1) == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
