mport numpy as np
import cv2

# Initialize tracker
tracker = Sort(max_age=20)

def plot_boxes_cv2(img, det_result, detections, class_names=None):
    img = np.copy(img)
    colors = np.array([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=np.float32)

    # Define two lines
    line1 = [100, 200, 800, 200]  # Line format: [x1, y1, x2, y2]
    line2 = [100, 400, 800, 400]

    for det in det_result:
        c1 = (int(det[2][0]), int(det[2][1]))
        c2 = (int(det[2][2]), int(det[2][3]))
        cx = (c1[0] + c2[0]) // 2
        cy = (c1[1] + c2[1]) // 2

        currentArray = np.array([c1[0], c1[1], c2[0], c2[1], det[1]])
        detections = np.vstack((detections, currentArray))

    # Update tracker and plot
    tracked_objects = tracker.update(detections)
    crossed_lines = {}  # Stores the crossing sequence of each object

    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = obj
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # Check if the center point crosses line1 or line2
        if line1[1] - 10 < center_y < line1[1] + 10:
            if obj_id not in crossed_lines:
                crossed_lines[obj_id] = ['line1']
            elif 'line1' not in crossed_lines[obj_id]:
                crossed_lines[obj_id].append('line1')

        if line2[1] - 10 < center_y < line2[1] + 10:
            if obj_id not in crossed_lines:
                crossed_lines[obj_id] = ['line2']
            elif 'line2' not in crossed_lines[obj_id]:
                crossed_lines[obj_id].append('line2')

        # Draw circles and text to visualize crossing
        if obj_id in crossed_lines:
            if len(crossed_lines[obj_id]) == 1:
                text = f'{crossed_lines[obj_id][0]} first by {obj_id}'
                color = (0, 255, 0) if crossed_lines[obj_id][0] == 'line1' else (0, 0, 255)
            elif len(crossed_lines[obj_id]) == 2:
                text = f'{crossed_lines[obj_id][0]} then {crossed_lines[obj_id][1]} by {obj_id}'
                color = (255, 0, 0)
            cv2.putText(img, text, (center_x, center_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Draw lines
    cv2.line(img, (line1[0], line1[1]), (line1[2], line1[3]), (0, 255, 255), 2)
    cv2.line(img, (line2[0], line2[1]), (line2[2], line2[3]), (255, 0, 255), 2)

    return img

# Additional code setup, capture initialization, main loop, etc., would follow as previously defined

