import cv2
import numpy as np

def non_maximum_suppression(boxes, overlap_threshold):
    if len(boxes) == 0:
        return []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_threshold)[0])))

    return boxes[pick].astype("int")


def filter_bounding_boxes(detections, min_size=20, max_size=500):
    filtered_detections = []
    for x, y, w, h in detections:
        if min_size <= w <= max_size and min_size <= h <= max_size:
            filtered_detections.append([x, y, w, h])
    return np.array(filtered_detections)

def merge_close_bounding_boxes(boxes, distance_threshold):
    if len(boxes) == 0:
        return []

    boxes = boxes.tolist()
    merged_boxes = []
    while len(boxes) > 0:
        # Take the first box as the base
        base_box = boxes[0]
        boxes = boxes[1:]

        # Cluster boxes that are close to the base box
        close_boxes = [base_box]
        for box in boxes[:]:
            if np.linalg.norm(np.array([base_box[0], base_box[1]]) - np.array([box[0], box[1]])) < distance_threshold:
                close_boxes.append(box)
                boxes.remove(box)

        # Merge all close boxes into one bounding box
        x_coords = [box[0] for box in close_boxes]
        y_coords = [box[1] for box in close_boxes]
        widths = [box[0] + box[2] for box in close_boxes]
        heights = [box[1] + box[3] for box in close_boxes]

        x_min = min(x_coords)
        y_min = min(y_coords)
        x_max = max(widths)
        y_max = max(heights)

        merged_boxes.append([x_min, y_min, x_max - x_min, y_max - y_min])

    return np.array(merged_boxes)

def calculate_center(bbox):
    x, y, w, h = bbox
    center_x = x + w // 2
    center_y = y + h // 2
    return center_x, center_y