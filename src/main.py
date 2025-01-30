import cv2
import numpy as np
from track import SORT, detect_moving_objects, count_cars_once
from utils import *

cap = cv2.VideoCapture(r'data\vecteezy_out-of-focus-street-traffic-high-angle-view.mov')
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
sort = SORT()
total_cars_counted = 0
counted_ids = set()
prev_centers_by_id = {}

if not cap.isOpened():
    raise cv2.error('Video could not be open.')

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        frame = cv2.resize(frame, (int(frame.shape[1] * 0.4), int(frame.shape[0] * 0.4)))

        line_position = int(frame.shape[0] * 0.3)

        contours, fg_mask = detect_moving_objects(frame, bg_subtractor)
        detections = [cv2.boundingRect(contour) for contour in contours if cv2.contourArea(contour) > 500]
        detections = np.array([np.concatenate((np.array([x, y]), np.array([w, h]))) for x, y, w, h in detections])

        merged_detections = merge_close_bounding_boxes(detections, distance_threshold=20)
        filtered_detections = filter_bounding_boxes(detections)
        filtered_detections = non_maximum_suppression(merged_detections, 0.2)  # More stringent overlap threshold

        tracked_objects = sort.update(filtered_detections)

        car_count = count_cars_once(sort, tracked_objects, line_position, counted_ids, prev_centers_by_id)
        total_cars_counted += car_count

        # Clear previous bounding boxes and draw only for active objects
        for tracker in sort.trackers:
            if tracker.lifespan < 2:
                x, y, w, h = tracker.bbox
                color = tracker.color
                cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
                cv2.putText(frame, f'ID: {tracker.object_id}', (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Draw line for object movement
                # for i in range(1, len(tracker.history)):
                #     cv2.line(frame, tracker.history[i - 1], tracker.history[i], color, 2)

        # Draw counting line
        cv2.line(frame, (0, line_position), (frame.shape[1], line_position), (0, 255, 255), 2)
        cv2.putText(frame, f'Cars Counted: {total_cars_counted}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow('Foreground Mask', fg_mask)
        cv2.imshow('Object Detection', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
