# Real-Time Vehicle Tracking and Counting System

A computer vision project for detecting, tracking, and counting vehicles in a video stream. It implements the SORT (Simple Online and Realtime Tracking) algorithm, which combines Kalman Filters for prediction and the Hungarian Algorithm for efficient data association.

Note on Accuracy: This project uses Background Subtraction for detection, a traditional method that is efficient but can be less robust to lighting changes and complex backgrounds. For higher accuracy, a deep learning-based detector (like YOLO) is recommended.

## Features

- Motion Detection: Uses Background Subtraction (MOG2) to identify moving objects.

- Object Tracking: Implements the core SORT algorithm for robust, real-time multi-object tracking.

- Persistence: Maintains object identities (object_id) across frames.

- Vehicle Counting: Counts vehicles as they cross a defined virtual line in the video.

- Bounding Box Management: Includes utilities for Non-Maximum Suppression (NMS) and merging close detections to reduce noise.

- Visualization: Displays tracked objects with unique colored bounding boxes and IDs, a counting line, and a live count.

## Data used in this project

The source code is available for learning and development purposes. The sample video data is owned by Vecteezy and is used under their Standard License. Please ensure you have the right to use any video you process with this code.

The video file can be downloaded from https://www.vecteezy.com/video/14055564-out-of-focus-street-traffic-high-angle-view.
