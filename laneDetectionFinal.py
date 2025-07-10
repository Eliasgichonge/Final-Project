import cv2
import numpy as np
import threading
import time
import pygame
from collections import deque
from dataclasses import dataclass
from typing import Tuple, List, Optional

@dataclass
class KalmanState:
    x: np.ndarray
    P: np.ndarray
    F: np.ndarray
    Q: np.ndarray
    R: float
    H: np.ndarray

class LaneKalmanFilter:
    def __init__(self):
        self.left_filter = self._init_filter()
        self.right_filter = self._init_filter()
    
    def _init_filter(self) -> KalmanState:
        x = np.array([[0.0], [0.0]])
        P = np.eye(2) * 1000
        F = np.array([[1, 1], [0, 1]])
        Q = np.array([[0.1, 0], [0, 0.1]])
        R = 10.0
        H = np.array([[1, 0]])
        
        return KalmanState(x, P, F, Q, R, H)
    
    def predict_and_update(self, left_pos: Optional[float], right_pos: Optional[float]) -> Tuple[float, float]:
        left_smooth = self._filter_step(self.left_filter, left_pos)
        right_smooth = self._filter_step(self.right_filter, right_pos)
        return left_smooth, right_smooth
    
    def _filter_step(self, kf: KalmanState, measurement: Optional[float]) -> float:
        kf.x = kf.F @ kf.x
        kf.P = kf.F @ kf.P @ kf.F.T + kf.Q
        
        if measurement is not None:
            z = np.array([[measurement]])
            y = z - (kf.H @ kf.x)
            S = kf.H @ kf.P @ kf.H.T + kf.R
            K = kf.P @ kf.H.T @ np.linalg.inv(S)
            
            kf.x = kf.x + K @ y
            kf.P = (np.eye(2) - K @ kf.H) @ kf.P
        
        return float(kf.x[0, 0])

class AlarmSystem:
    def __init__(self):
        pygame.mixer.init()
        self.alarm_active = False
        self.alarm_thread = None
        
    def trigger_alarm(self):
        if not self.alarm_active:
            self.alarm_active = True
            self.alarm_thread = threading.Thread(target=self._play_alarm)
            self.alarm_thread.start()
    
    def stop_alarm(self):
        self.alarm_active = False
        if self.alarm_thread and self.alarm_thread.is_alive():
            self.alarm_thread.join()
    
    def _play_alarm(self):
        sample_rate = 22050
        duration = 0.5
        frequency = 800
        
        frames = int(duration * sample_rate)
        arr = np.sin(2 * np.pi * frequency * np.linspace(0, duration, frames))
        arr = (arr * 32767).astype(np.int16)
        arr = np.repeat(arr.reshape(frames, 1), 2, axis=1)
        
        while self.alarm_active:
            try:
                sound = pygame.sndarray.make_sound(arr)
                sound.play()
                time.sleep(0.6)
            except Exception as e:
                print(f"🚨 LANE DEPARTURE ALARM! 🚨 (Audio error: {e})")
                time.sleep(0.6)

class LaneDepartureDetector:
    def __init__(self, camera_ip: str, camera_port: int = 81):
        self.camera_url = f"http://{camera_ip}:{camera_port}/stream"
        self.kalman_filter = LaneKalmanFilter()
        self.alarm_system = AlarmSystem()
        
        self.roi_top = 0.4
        self.roi_bottom = 0.9
        self.lane_history = deque(maxlen=5)
        self.violation_count = 0
        self.violation_threshold = 3
        
        self.white_hsv_lower = np.array([0, 0, 180])
        self.white_hsv_upper = np.array([180, 40, 255])
        
        self.white_rgb_lower = np.array([180, 180, 180])
        self.white_rgb_upper = np.array([255, 255, 255])
        
        self.white_lab_lower = np.array([180, 0, 0])
        self.white_lab_upper = np.array([255, 140, 140])
        
    def preprocess_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        height, width = frame.shape[:2]
        roi = frame[int(height * self.roi_top):int(height * self.roi_bottom), :]
        
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        white_mask_hsv = cv2.inRange(hsv, self.white_hsv_lower, self.white_hsv_upper)
        
        white_mask_rgb = cv2.inRange(roi, self.white_rgb_lower, self.white_rgb_upper)
        
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        white_mask_lab = cv2.inRange(lab, self.white_lab_lower, self.white_lab_upper)
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, white_mask_gray = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        white_mask_adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                   cv2.THRESH_BINARY, 15, -5)
        
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        white_mask_combined = cv2.bitwise_or(white_mask_hsv, white_mask_rgb)
        white_mask_combined = cv2.bitwise_or(white_mask_combined, white_mask_lab)
        white_mask_combined = cv2.bitwise_or(white_mask_combined, white_mask_gray)
        white_mask_combined = cv2.bitwise_or(white_mask_combined, white_mask_adaptive)
        
        kernel_edge = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges_dilated = cv2.dilate(edges, kernel_edge, iterations=1)
        white_mask_combined = cv2.bitwise_or(white_mask_combined, edges_dilated)
        
        kernel_noise = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        white_mask_combined = cv2.morphologyEx(white_mask_combined, cv2.MORPH_OPEN, kernel_noise)
        
        kernel_fill = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        white_mask_combined = cv2.morphologyEx(white_mask_combined, cv2.MORPH_CLOSE, kernel_fill)
        
        kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
        white_mask_combined = cv2.morphologyEx(white_mask_combined, cv2.MORPH_CLOSE, kernel_vertical)
        
        white_mask_combined = cv2.GaussianBlur(white_mask_combined, (3, 3), 0)
        
        return roi, white_mask_combined
    
    def detect_lane_boundaries(self, mask: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
        height, width = mask.shape
        
        bottom_section = mask[int(height * 0.7):, :]
        histogram = np.sum(bottom_section, axis=0)
        
        kernel_size = max(3, width // 100)
        if kernel_size % 2 == 0:
            kernel_size += 1
        histogram_smooth = cv2.GaussianBlur(histogram.reshape(1, -1).astype(np.float32), 
                                           (kernel_size, 1), 0).flatten()
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_contours = []
        min_area = (height * width) * 0.001
        max_area = (height * width) * 0.3
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = h / max(w, 1)
                if aspect_ratio > 1.5:
                    valid_contours.append((contour, x + w//2))
        
        lines = cv2.HoughLinesP(mask, 1, np.pi/180, threshold=30, 
                               minLineLength=height//4, maxLineGap=height//8)
        
        line_x_positions = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                if abs(angle) > 60 or abs(angle - 90) < 30 or abs(angle + 90) < 30:
                    avg_x = (x1 + x2) / 2
                    line_x_positions.append(avg_x)
        
        all_candidates = []
        
        midpoint = width // 2
        left_histogram_peaks = self._find_peaks(histogram_smooth[:midpoint], min_height=height*2)
        right_histogram_peaks = [peak + midpoint for peak in 
                                self._find_peaks(histogram_smooth[midpoint:], min_height=height*2)]
        
        all_candidates.extend([(x, 'histogram') for x in left_histogram_peaks])
        all_candidates.extend([(x, 'histogram') for x in right_histogram_peaks])
        
        all_candidates.extend([(x, 'contour') for _, x in valid_contours])
        
        all_candidates.extend([(x, 'line') for x in line_x_positions])
        
        if not all_candidates:
            return None, None
        
        candidates_x = [x for x, _ in all_candidates]
        
        left_candidates = [x for x in candidates_x if x < midpoint]
        right_candidates = [x for x in candidates_x if x > midpoint]
        
        left_x = self._find_most_reliable_position(left_candidates) if left_candidates else None
        right_x = self._find_most_reliable_position(right_candidates) if right_candidates else None
        
        return left_x, right_x
    
    def _find_most_reliable_position(self, candidates: List[float]) -> float:
        if len(candidates) == 1:
            return candidates[0]
        
        candidates = sorted(candidates)
        
        if len(candidates) > 3:
            threshold = 50
            clusters = []
            
            for candidate in candidates:
                added_to_cluster = False
                for cluster in clusters:
                    if abs(candidate - np.mean(cluster)) < threshold:
                        cluster.append(candidate)
                        added_to_cluster = True
                        break
                
                if not added_to_cluster:
                    clusters.append([candidate])
            
            largest_cluster = max(clusters, key=len)
            return np.mean(largest_cluster)
        
        return np.median(candidates)
    
    def _find_peaks(self, histogram: np.ndarray, min_height: float = 50) -> List[int]:
        if len(histogram) == 0:
            return []
        
        peaks = []
        min_distance = len(histogram) // 20
        
        for i in range(min_distance, len(histogram) - min_distance):
            if (histogram[i] > min_height and
                histogram[i] > histogram[i-1] and 
                histogram[i] > histogram[i+1]):
                
                too_close = False
                for existing_peak in peaks:
                    if abs(i - existing_peak) < min_distance:
                        if histogram[i] > histogram[existing_peak]:
                            peaks.remove(existing_peak)
                        else:
                            too_close = True
                        break
                
                if not too_close:
                    peaks.append(i)
        
        peaks.sort(key=lambda x: histogram[x], reverse=True)
        return peaks[:2]
    
    def _sliding_window_search(self, mask: np.ndarray, base_x: int, is_left: bool) -> Optional[float]:
        height, width = mask.shape
        window_height = height // 10
        
        margin = 50
        min_pixels = 20
        
        current_x = base_x
        lane_pixels_x = []
        
        for window in range(10):
            win_y_low = height - (window + 1) * window_height
            win_y_high = height - window * window_height
            win_x_low = max(0, current_x - margin)
            win_x_high = min(width, current_x + margin)
            
            window_mask = mask[win_y_low:win_y_high, win_x_low:win_x_high]
            nonzero_pixels = np.nonzero(window_mask)
            
            if len(nonzero_pixels[1]) > min_pixels:
                mean_x = np.mean(nonzero_pixels[1]) + win_x_low
                lane_pixels_x.append(mean_x)
                current_x = int(mean_x)
        
        return np.mean(lane_pixels_x) if lane_pixels_x else None
    
    def check_lane_departure(self, left_x: Optional[float], right_x: Optional[float], 
                           frame_width: int) -> bool:
        if left_x is None or right_x is None:
            return False
        
        lane_center = (left_x + right_x) / 2
        lane_width = right_x - left_x
        frame_center = frame_width / 2
        
        tolerance = lane_width * 0.1
        departure = abs(frame_center - lane_center) > (lane_width / 2 - tolerance)
        
        if departure:
            self.violation_count += 1
        else:
            self.violation_count = max(0, self.violation_count - 1)
        
        return self.violation_count >= self.violation_threshold
    
    def draw_visualizations(self, frame: np.ndarray, roi: np.ndarray, 
                          left_x: Optional[float], right_x: Optional[float],
                          departed: bool) -> np.ndarray:
        height, width = frame.shape[:2]
        roi_height = roi.shape[0]
        roi_start = int(height * self.roi_top)
        
        overlay = frame.copy()
        
        if left_x is not None and right_x is not None:
            left_x_full = int(left_x)
            right_x_full = int(right_x)
            
            cv2.line(overlay, 
                    (left_x_full, roi_start), 
                    (left_x_full, roi_start + roi_height), 
                    (0, 255, 0), 8)
            cv2.line(overlay, 
                    (right_x_full, roi_start), 
                    (right_x_full, roi_start + roi_height), 
                    (0, 255, 0), 8)
            
            danger_alpha = 0.3
            
            left_danger = np.array([[0, roi_start], 
                                  [left_x_full, roi_start],
                                  [left_x_full, roi_start + roi_height],
                                  [0, roi_start + roi_height]], np.int32)
            cv2.fillPoly(overlay, [left_danger], (180, 105, 255))
            
            right_danger = np.array([[right_x_full, roi_start],
                                   [width, roi_start],
                                   [width, roi_start + roi_height],
                                   [right_x_full, roi_start + roi_height]], np.int32)
            cv2.fillPoly(overlay, [right_danger], (180, 105, 255))
            
            frame = cv2.addWeighted(frame, 1 - danger_alpha, overlay, danger_alpha, 0)
            
            lane_center = int((left_x_full + right_x_full) / 2)
            cv2.line(frame, 
                    (lane_center, roi_start), 
                    (lane_center, roi_start + roi_height), 
                    (255, 255, 0), 2)
        
        status_color = (0, 0, 255) if departed else (0, 255, 0)
        status_text = "LANE DEPARTURE!" if departed else "IN LANE"
        cv2.putText(frame, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        cv2.putText(frame, f"Violations: {self.violation_count}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def run(self):
        print(f"Connecting to ESP32 camera at {self.camera_url}")
        cap = cv2.VideoCapture(self.camera_url)
        
        if not cap.isOpened():
            print("Failed to connect to camera. Please check the IP address and connection.")
            return
        
        print("Lane Departure Detection System Active")
        print("Press 'q' to quit")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                roi, white_mask = self.preprocess_frame(frame)
                
                left_raw, right_raw = self.detect_lane_boundaries(white_mask)
                
                left_smooth, right_smooth = self.kalman_filter.predict_and_update(left_raw, right_raw)
                
                departed = self.check_lane_departure(left_smooth, right_smooth, frame.shape[1])
                
                if departed:
                    self.alarm_system.trigger_alarm()
                else:
                    self.alarm_system.stop_alarm()
                
                result_frame = self.draw_visualizations(frame, roi, left_smooth, right_smooth, departed)
                
                cv2.imshow('Lane Departure Detection', result_frame)
                cv2.imshow('White Detection Mask', white_mask)
                
                debug_frame = roi.copy()
                if left_smooth is not None:
                    cv2.line(debug_frame, (int(left_smooth), 0), (int(left_smooth), roi.shape[0]), (0, 255, 0), 3)
                if right_smooth is not None:
                    cv2.line(debug_frame, (int(right_smooth), 0), (int(right_smooth), roi.shape[0]), (0, 255, 0), 3)
                cv2.imshow('ROI with Detected Lines', debug_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.alarm_system.stop_alarm()
            cap.release()
            cv2.destroyAllWindows()

def main():
    CAMERA_IP = "192.168.192.102"
    CAMERA_PORT = 81
    
    detector = LaneDepartureDetector(CAMERA_IP, CAMERA_PORT)
    detector.run()

if __name__ == "__main__":
    main()