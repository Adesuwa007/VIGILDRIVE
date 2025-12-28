import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance
from collections import deque
import time
import torch
import torch.nn as nn
import pygame  # For beeping sound
from PIL import Image  # For Gradio (optional, for web frontend)

class DrowsinessDetector:
    def __init__(self):
        # MediaPipe Face Mesh initialization
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Eye landmarks indices for MediaPipe (468 landmarks)
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        
        # Mouth landmarks for yawn detection
        self.MOUTH = [61, 291, 0, 17, 269, 405]
        
        # Thresholds
        self.EAR_THRESH = 0.21
        self.MAR_THRESH = 0.6
        self.CONSEC_FRAMES = 20
        
        # Counters
        self.eye_closed_counter = 0
        self.yawn_counter = 0
        self.blink_counter = 0
        self.total_blinks = 0
        
        # Blink detection
        self.blink_detected = False
        self.previous_ear = 0.3
        
        # History for smoothing
        self.ear_history = deque(maxlen=5)
        self.mar_history = deque(maxlen=5)
        
        # Status tracking
        self.drowsy = False
        self.yawning = False
        self.previous_drowsy = False  # For sound trigger
        
        # Session stats
        self.session_start = time.time()
        self.drowsy_events = []
        
        # Predictive LSTM for fatigue forecasting
        self.ear_long_history = deque(maxlen=60)  # Longer history for prediction
        self.lstm_model = nn.LSTM(input_size=1, hidden_size=32, num_layers=1, batch_first=True)
        self.fc = nn.Linear(32, 1)  # Predict next EAR
        # Simple on-device training with dummy data (in production, fine-tune on user sessions)
        dummy_data = torch.randn(100, 60, 1)  # Simulate EAR sequences
        dummy_targets = torch.randn(100, 1)
        optimizer = torch.optim.Adam(list(self.lstm_model.parameters()) + list(self.fc.parameters()), lr=0.01)
        for _ in range(50):  # Quick pre-train
            out, _ = self.lstm_model(dummy_data)
            pred = self.fc(out[:, -1, :])
            loss = nn.MSELoss()(pred, dummy_targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Beeping sound setup (cross-platform via pygame)
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        duration = 0.5
        freq = 1000
        sample_rate = 22050
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        beep = np.sin(2 * np.pi * freq * t)
        audio = np.c_[beep * 0.5, beep * 0.5]  # Stereo, attenuated
        audio = (audio * 32767).astype(np.int16)
        self.beep_sound = pygame.mixer.Sound(array=audio)
    
    def eye_aspect_ratio(self, eye_landmarks):
        """Calculate Eye Aspect Ratio"""
        # Vertical distances
        A = distance.euclidean(eye_landmarks[1], eye_landmarks[5])
        B = distance.euclidean(eye_landmarks[2], eye_landmarks[4])
        # Horizontal distance
        C = distance.euclidean(eye_landmarks[0], eye_landmarks[3])
        
        ear = (A + B) / (2.0 * C)
        return ear
    
    def mouth_aspect_ratio(self, mouth_landmarks):
        """Calculate Mouth Aspect Ratio for yawn detection"""
        # Vertical distances
        A = distance.euclidean(mouth_landmarks[1], mouth_landmarks[5])
        B = distance.euclidean(mouth_landmarks[2], mouth_landmarks[4])
        # Horizontal distance
        C = distance.euclidean(mouth_landmarks[0], mouth_landmarks[3])
        
        mar = (A + B) / (2.0 * C)
        return mar
    
    def extract_eye_region(self, frame, eye_landmarks):
        """Extract eye region for CNN model input"""
        # Get bounding box for eye
        x_coords = [lm[0] for lm in eye_landmarks]
        y_coords = [lm[1] for lm in eye_landmarks]
        
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        
        # Add padding
        padding = 10
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(frame.shape[1], x_max + padding)
        y_max = min(frame.shape[0], y_max + padding)
        
        eye_crop = frame[y_min:y_max, x_min:x_max]
        
        # Resize for CNN input (e.g., 64x64)
        if eye_crop.size > 0:
            eye_crop = cv2.resize(eye_crop, (64, 64))
            return eye_crop, (x_min, y_min, x_max, y_max)
        return None, None
    
    def detect_blink(self, current_ear):
        """Detect blinks based on EAR changes"""
        blink = False
        
        # Blink: EAR drops below threshold then rises back
        if self.previous_ear > self.EAR_THRESH and current_ear < self.EAR_THRESH:
            self.blink_detected = True
        elif self.blink_detected and current_ear > self.EAR_THRESH:
            blink = True
            self.total_blinks += 1
            self.blink_detected = False
        
        self.previous_ear = current_ear
        return blink
    
    def process_frame(self, frame):
        """Main processing function for each frame"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        
        detection_data = {
            'ear': 0.0,
            'mar': 0.0,
            'drowsy': False,
            'yawning': False,
            'blink_detected': False,
            'total_blinks': self.total_blinks,
            'predicted_ear': 0.0,  # New for LSTM
            'left_eye_crop': None,
            'right_eye_crop': None,
            'face_detected': False
        }
        
        if not results.multi_face_landmarks:
            return frame, detection_data
        
        detection_data['face_detected'] = True
        face_landmarks = results.multi_face_landmarks[0]
        h, w = frame.shape[:2]
        
        # Extract landmark coordinates
        def get_landmarks(indices):
            return [(int(face_landmarks.landmark[idx].x * w),
                    int(face_landmarks.landmark[idx].y * h))
                    for idx in indices]
        
        left_eye = get_landmarks(self.LEFT_EYE)
        right_eye = get_landmarks(self.RIGHT_EYE)
        mouth = get_landmarks(self.MOUTH)
        
        # Calculate EAR
        left_ear = self.eye_aspect_ratio(left_eye)
        right_ear = self.eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0
        self.ear_history.append(ear)
        ear_smooth = np.mean(self.ear_history)
        
        # Calculate MAR
        mar = self.mouth_aspect_ratio(mouth)
        self.mar_history.append(mar)
        mar_smooth = np.mean(self.mar_history)
        
        detection_data['ear'] = ear_smooth
        detection_data['mar'] = mar_smooth
        
        # Predictive forecasting with LSTM
        self.ear_long_history.append(ear_smooth)
        if len(self.ear_long_history) == 60:
            seq = torch.tensor(list(self.ear_long_history), dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            with torch.no_grad():
                out, _ = self.lstm_model(seq)
                pred_ear = self.fc(out[:, -1, :]).item()
            detection_data['predicted_ear'] = pred_ear
        
        # Blink detection
        if self.detect_blink(ear_smooth):
            detection_data['blink_detected'] = True
        
        # Extract eye crops for CNN
        left_crop, left_bbox = self.extract_eye_region(frame, left_eye)
        right_crop, right_bbox = self.extract_eye_region(frame, right_eye)
        detection_data['left_eye_crop'] = left_crop
        detection_data['right_eye_crop'] = right_crop
        
        # Draw eye and mouth contours
        cv2.polylines(frame, [np.array(left_eye)], True, (0, 255, 0), 1)
        cv2.polylines(frame, [np.array(right_eye)], True, (0, 255, 0), 1)
        cv2.polylines(frame, [np.array(mouth)], True, (0, 255, 255), 1)
        
        # Drowsiness detection logic
        if ear_smooth < self.EAR_THRESH:
            self.eye_closed_counter += 1
            if self.eye_closed_counter >= self.CONSEC_FRAMES:
                self.drowsy = True
                detection_data['drowsy'] = True
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                
                # Log event and trigger beep (only on new detection)
                if not self.previous_drowsy:
                    self.beep_sound.play()
                if not self.drowsy_events or time.time() - self.drowsy_events[-1] > 5:
                    self.drowsy_events.append(time.time())
                self.previous_drowsy = True
        else:
            self.eye_closed_counter = 0
            self.drowsy = False
            self.previous_drowsy = False
        
        # Yawn detection
        if mar_smooth > self.MAR_THRESH:
            self.yawn_counter += 1
            if self.yawn_counter >= 10:
                self.yawning = True
                detection_data['yawning'] = True
                cv2.putText(frame, "YAWNING DETECTED!", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            self.yawn_counter = 0
            self.yawning = False
        
        # Predictive alert
        if 'predicted_ear' in detection_data and detection_data['predicted_ear'] < self.EAR_THRESH * 0.9:
            cv2.putText(frame, f"PREDICTED FATIGUE: {detection_data['predicted_ear']:.2f}", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Display metrics (modern styling: semi-transparent background for panel)
        panel_y = 30
        overlay = frame.copy()
        cv2.rectangle(overlay, (w - 200, 0), (w, 200), (0, 0, 0), -1)  # Black panel
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, f"EAR: {ear_smooth:.2f}", (w - 150, panel_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        panel_y += 30
        cv2.putText(frame, f"MAR: {mar_smooth:.2f}", (w - 150, panel_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        panel_y += 30
        cv2.putText(frame, f"Blinks: {self.total_blinks}", (w - 150, panel_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        panel_y += 30
        cv2.putText(frame, f"Closed: {self.eye_closed_counter}/{self.CONSEC_FRAMES}", 
                   (w - 150, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if 'predicted_ear' in detection_data:
            panel_y += 30
            cv2.putText(frame, f"Pred EAR: {detection_data['predicted_ear']:.2f}", (w - 150, panel_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        return frame, detection_data
    
    def get_session_stats(self):
        """Get statistics for the current session"""
        duration = time.time() - self.session_start
        return {
            'session_duration': duration,
            'total_blinks': self.total_blinks,
            'drowsy_events': len(self.drowsy_events),
            'blink_rate': self.total_blinks / (duration / 60) if duration > 0 else 0
        }

# Desktop OpenCV Demo (with modern panel styling and beep)
if __name__ == "__main__":
    detector = DrowsinessDetector()
    cap = cv2.VideoCapture(0)
    
    print("Starting drowsiness detection... Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame, detection_data = detector.process_frame(frame)
        
        cv2.imshow("Drowsiness Detection - Proactive Safety System", processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Print session stats
    stats = detector.get_session_stats()
    print("\n=== Session Statistics ===")
    print(f"Duration: {stats['session_duration']:.1f} seconds")
    print(f"Total Blinks: {stats['total_blinks']}")
    print(f"Drowsy Events: {stats['drowsy_events']}")
    print(f"Blink Rate: {stats['blink_rate']:.1f} blinks/min")
    
    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()

# Modern Web Frontend with Gradio (pip install gradio first)
# Run: python this_file.py --gradio (or separately)
import argparse
import gradio as gr

def run_gradio():
    detector = DrowsinessDetector()
    
    def process_image(img):
        if img is None:
            return None, 0.0, False, 0.0, False
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        processed_frame, data = detector.process_frame(frame)
        processed_pil = Image.fromarray(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
        return processed_pil, data['ear'], data['drowsy'], data.get('predicted_ear', 0.0), data['yawning']
    
    iface = gr.Interface(
        fn=process_image,
        inputs=gr.Image(source="webcam", type="pil", label="Live Webcam Feed"),
        outputs=[
            gr.Image(type="pil", label="Processed Video"),
            gr.Number(label="Current EAR"),
            gr.Checkbox(label="Drowsy?"),
            gr.Number(label="Predicted EAR"),
            gr.Checkbox(label="Yawning?")
        ],
        live=True,
        title="🚗 Proactive Drowsiness Detection System",
        description="Real-time monitoring with predictive AI. Beep alerts in desktop mode. Early warnings prevent fatigue!",
        theme=gr.themes.Soft()  # Modern theme
    )
    iface.launch(share=True)  # Share link for hackathon demos

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gradio", action="store_true", help="Run Gradio web demo")
    args = parser.parse_args()
    if args.gradio:
        run_gradio()