import cv2
import numpy as np
from KalmanFilter import KalmanFilter
import random
from scipy.optimize import linear_sum_assignment
from utils import calculate_center

class TrackedObject:
    def __init__(self, object_id, bbox):
        self.object_id = object_id
        self.kf = KalmanFilter()
        self.kf.kf.statePost[:2, 0] = bbox[:2]  # Set the initial state
        self.bbox = bbox
        self.color = self._generate_random_color()
        self.lifespan = 0
        self.history = []

    def predict(self):
        self.lifespan += 1
        return self.kf.predict()

    def update(self, bbox):
        self.kf.update(bbox[:2])
        self.bbox = bbox
        self.lifespan = 0
        self.history.append((bbox[0]+bbox[2] // 2, bbox[1]+bbox[3] // 2))

    def _generate_random_color(self):
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

class SORT:
    def __init__(self, max_age=1, iou_threshold=0.3):
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.next_id = 0
        self.removed_ids = set()

    def update(self, detections):
        # Predict
        for tracker in self.trackers:
            tracker.predict()

        matches_indices, unmatched_trackers, unmatched_detections = self.match_detections(detections)

        for tracker_idx, detection_idx in matches_indices:
            self.trackers[tracker_idx].update(detections[detection_idx])
            if self.trackers[tracker_idx].object_id in self.removed_ids:
                self.removed_ids.remove(self.trackers[tracker_idx].object_id)

        for detection_idx in unmatched_detections:
            new_id = self.get_next_id()
            self.trackers.append(TrackedObject(new_id, detections[detection_idx]))

        for tracker_idx in sorted(unmatched_trackers, reverse=True):
            self.removed_ids.add(self.trackers[tracker_idx].object_id)
            self.trackers.pop(tracker_idx)

        output = []
        for tracker in self.trackers:
            output.append(tracker.bbox)

        return output
    
    def get_next_id(self):
        while self.next_id in self.removed_ids:
            self.next_id += 1
        self.next_id += 1
        return self.next_id - 1
    
    def match_detections(self, detections):
        if len(self.trackers) == 0:
            return np.empty((0, 2), dtype=int), np.empty((0,), dtype=int), np.arange(len(detections))
        
        predicted_bboxes = np.array([tracker.bbox for tracker in self.trackers])
        iou_matrix = self.compute_iou_matrix(predicted_bboxes, detections)

        row_ind, col_ind = linear_sum_assignment(-iou_matrix)

        matched_indices = []
        unmatched_trackers = set(range(len(self.trackers)))
        unmatched_detections = set(range(len(detections)))

        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= self.iou_threshold:
                matched_indices.append((r, c))
                unmatched_trackers.remove(r)
                unmatched_detections.remove(c)

        return matched_indices, unmatched_trackers, unmatched_detections

    def compute_iou_matrix(self, boxes1, boxes2):
        iou_matrix = np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)
        for i, box1 in enumerate(boxes1):
            for j, box2 in enumerate(boxes2):
                iou_matrix[i, j] = self.iou(box1, box2)
        return iou_matrix

    def iou(self, box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area
    
def detect_moving_objects(frame, bg_subtractor):
    fg_mask = bg_subtractor.apply(frame)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel) 
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_DILATE, kernel)

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, fg_mask

def count_cars_once(sort, tracked_objects, line_position, counted_ids, prev_centers_by_id):
    count = 0
    for idx, obj in enumerate(tracked_objects):
        cx, cy = calculate_center(obj)
        tracker = sort.trackers[idx] if idx < len(sort.trackers) else None

        if tracker:
            tracker_id = tracker.object_id
            prev_center = prev_centers_by_id.get(tracker_id, (cx, cy))

            # Check for crossing the line
            if prev_center[1] < line_position <= cy or prev_center[1] > line_position >= cy:
                if tracker_id not in counted_ids:
                    counted_ids.add(tracker_id)
                    count += 1

            prev_centers_by_id[tracker_id] = (cx, cy)
    return count