import cv2
import numpy as np

class KalmanFilter:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                            [0, 1, 0, 1],
                                            [0, 0, 1, 0],
                                            [0, 0, 0, 1]], dtype=np.float32)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                              [0, 1, 0, 0]], dtype=np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-3
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-2
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.kf.statePost = np.zeros((4, 1), dtype=np.float32)

    def predict(self):
        return self.kf.predict()

    def update(self, measurement):
        measurement = np.array([[measurement[0]], [measurement[1]]], dtype=np.float32)
        self.kf.correct(measurement)