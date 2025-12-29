import os
IS_HF = os.environ.get("SPACE_ID") is not None
import cv2
import mediapipe as mp
import numpy as np
import time
from scipy.spatial import distance
from PIL import Image
import gradio as gr
import pygame

# LIGHT DETECTION
def is_dark(frame, threshold=60):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.mean(gray) < threshold

# GENTLE AUDIO ALERT
sound_enabled = False

if not IS_HF:
    try:
        import pygame
        pygame.mixer.init(frequency=22050, size=-16, channels=2)

        sr = 22050
        t = np.linspace(0, 0.5, int(sr * 0.5), False)
        tone = 0.15 * np.sin(2 * np.pi * 440 * t)
        tone = np.column_stack((tone, tone))
        alert_sound = pygame.sndarray.make_sound(
            (tone * 32767).astype(np.int16)
        )

        sound_enabled = True
    except Exception as e:
        print("Audio disabled:", e)

# NIGHT VISION
def night_vision(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    enhanced[:, :, 1] = np.clip(enhanced[:, :, 1] + 40, 0, 255)
    enhanced[:, :, 0] = enhanced[:, :, 0] * 0.6
    enhanced[:, :, 2] = enhanced[:, :, 2] * 0.6
    cv2.putText(enhanced, "NIGHT MODE", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    return enhanced


# DROWSINESS DETECTOR
class DrowsinessDetector:
    def __init__(self):
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        self.MOUTH = [61, 291, 0, 17, 269, 405]
        self.EAR_THRESH = 0.25
        self.CLOSED_FRAMES = 6
        self.eye_counter = 0
        self.prev_ear = 0.3
        self.blinks = 0
        self.last_alert = 0

    def ear(self, eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)

    def mar(self, mouth):
        A = distance.euclidean(mouth[1], mouth[5])
        B = distance.euclidean(mouth[2], mouth[4])
        C = distance.euclidean(mouth[0], mouth[3])
        return (A + B) / (2.0 * C)

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.mesh.process(rgb)
        ear_val = mar_val = 0.0
        drowsy = False

        if res.multi_face_landmarks:
            face = res.multi_face_landmarks[0]
            h, w = frame.shape[:2]

            def pts(ids):
                return [(int(face.landmark[i].x * w), int(face.landmark[i].y * h)) for i in ids]

            left = pts(self.LEFT_EYE)
            right = pts(self.RIGHT_EYE)
            mouth = pts(self.MOUTH)

            cv2.polylines(frame, [np.array(left)], True, (0, 255, 0), 1)
            cv2.polylines(frame, [np.array(right)], True, (0, 255, 0), 1)
            cv2.polylines(frame, [np.array(mouth)], True, (0, 255, 255), 1)

            ear_val = (self.ear(left) + self.ear(right)) / 2.0
            mar_val = self.mar(mouth)

            if self.prev_ear > self.EAR_THRESH and ear_val < self.EAR_THRESH:
                self.blinks += 1
            self.prev_ear = ear_val

            if ear_val < self.EAR_THRESH:
                self.eye_counter += 1
                if self.eye_counter >= self.CLOSED_FRAMES:
                    drowsy = True
                    if time.time() - self.last_alert > 3:
                        alert_sound.play()
                        self.last_alert = time.time()
                    cv2.putText(frame, "DROWSINESS ALERT!", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            else:
                self.eye_counter = 0

        # HUD on frame
        cv2.putText(frame, f"EAR: {ear_val:.3f}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"MAR: {mar_val:.3f}", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Blinks: {self.blinks}", (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return frame, ear_val, mar_val, self.blinks, drowsy

# STREAM 
def stream():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    detector = DrowsinessDetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Camera Error", (150, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            img = Image.fromarray(placeholder)
            yield img, init_ear, init_mar, init_blinks, init_alert
            time.sleep(1)
            continue

        small_frame = cv2.resize(frame, (320, 240))
        processed_small, ear, mar, blinks, drowsy = detector.process(small_frame)
        display = cv2.resize(processed_small, (640, 480))

        if is_dark(frame):
            display = night_vision(display)

        # Drowsy overlay
        if drowsy:
            overlay = display.copy()
            cv2.rectangle(overlay, (0, 0), (640, 100), (0, 0, 255), -1)
            cv2.putText(overlay, "DROWSY", (200, 70),
            cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 255, 255), 3)
            cv2.addWeighted(overlay, 0.5, display, 0.5, 0, display)

        img = Image.fromarray(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))

        ear_fill = int(ear * 360)
        ear_color = "#00ffff" if ear > 0.25 else "#ff00ff"
        mar_fill = int(mar * 360)

        ear_gauge = f"""
        <div class="gauge">
            <svg viewBox="0 0 240 240">
                <circle class="track" cx="120" cy="120" r="100"/>
                <circle class="fill" cx="120" cy="120" r="100" stroke="{ear_color}" stroke-dasharray="{ear_fill} 360"/>
            </svg>
            <div class="inner">EAR<br><span>{ear:.3f}</span></div>
        </div>
        """

        mar_gauge = ear_gauge.replace("EAR", "MAR").replace(str(ear_fill), str(mar_fill)).replace(ear_color, "#00ffff").replace(f"{ear:.3f}", f"{mar:.3f}")

        blinks_html = f"<div class='big-number'>{blinks}</div>"

        alert_html = "<div class='alert-banner safe'>SYSTEM ACTIVE • SAFE</div>" if not drowsy else "<div class='alert-banner danger pulse'>DROWSINESS DETECTED • ALERT</div>"

        yield img, ear_gauge, mar_gauge, blinks_html, alert_html
        time.sleep(0.25)

# Initials 
init_ear = "<div class='gauge'><div class='inner'>EAR<br><span>-.---</span></div></div>"
init_mar = init_ear.replace("EAR", "MAR")
init_blinks = "<div class='big-number'>0</div>"
init_alert = "<div class='alert-banner'>INITIALIZING...</div>"

# ENHANCED CSS WITH HOLOGRAPHIC SCAN LINES
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&family=Exo+2:wght@600&display=swap');

body { background: #000; margin: 0; }
.gradio-container { 
    position: relative; 
    background: radial-gradient(circle at center, #0f0033, #000000); 
    padding: 20px; 
    overflow: hidden;
}

/* Holographic Scan Lines Overlay */
.gradio-container::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    pointer-events: none;
    background: repeating-linear-gradient(
        0deg,
        rgba(0, 255, 255, 0.03),
        rgba(0, 255, 255, 0.03) 1px,
        transparent 1px,
        transparent 2px
    );
    z-index: 10;
}

.gradio-container::after {
    content: "";
    position: absolute;
    top: -100%;
    left: 0;
    width: 100%;
    height: 100px;
    background: linear-gradient(to bottom, transparent, rgba(0, 255, 255, 0.4));
    animation: scan 8s linear infinite;
    pointer-events: none;
    z-index: 11;
}

@keyframes scan {
    0% { top: -100%; }
    100% { top: 100%; }
}

h1 { font-family: 'Orbitron', sans-serif; color: #00ffff; font-size: 4.5em; text-shadow: 0 0 50px #00ffff; letter-spacing: 12px; }
.subtitle { color: #ffffff88; font-size: 1.3em; }

.gauge { position: relative; width: 220px; height: 220px; margin: 20px auto; }
.track { fill: none; stroke: #00ffff22; stroke-width: 20; }
.fill { fill: none; stroke-width: 20; stroke-linecap: round; transform: rotate(-90deg); transform-origin: center; transition: all 0.5s; }
.inner { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); text-align: center; color: #00ffff; font-family: 'Exo 2'; }
.inner span { font-size: 2em; display: block; margin-top: 10px; text-shadow: 0 0 20px; }

.big-number { font-size: 5em; color: #00ffff; text-align: center; text-shadow: 0 0 40px #00ffff; font-family: 'Orbitron'; }

.alert-banner { text-align: center; font-size: 2em; padding: 20px; margin: 30px 0; border-radius: 20px; font-family: 'Orbitron'; text-shadow: 0 0 30px; }
.alert-banner.safe { background: rgba(0, 255, 255, 0.2); color: #00ffff; border: 2px solid #00ffff; }
.alert-banner.danger { background: rgba(255, 0, 255, 0.3); color: #ff00ff; border: 3px solid #ff00ff; animation: pulse 1s infinite; }

@keyframes pulse { 0%,100% { box-shadow: 0 0 20px #ff00ff; } 50% { box-shadow: 0 0 80px #ff00ff; } }

.image-container { position: relative; border-radius: 30px; overflow: hidden; border: 4px solid #00ffff; box-shadow: 0 0 100px rgba(0, 255, 255, 0.6); }
.card { background: transparent; border: none; box-shadow: none; }
"""

with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("# VIGILDRIVE")
    gr.Markdown("<p class='subtitle'> Driver Vigilance System • Real-Time Monitoring • Instant Alerts</p>")

    with gr.Row():
        with gr.Column(scale=2):
            video_feed = gr.Image(show_label=False, height=600, container=True, elem_classes="image-container")

        with gr.Column(scale=1):
            with gr.Group(elem_classes="card"):
                ear_gauge = gr.HTML(init_ear)
                mar_gauge = gr.HTML(init_mar)
                gr.HTML("<div style='text-align:center; color:#00ffff; font-size:1.5em; margin:20px;'>BLINK COUNT</div>")
                blinks_html = gr.HTML(init_blinks)
                alert_html = gr.HTML(init_alert)

    demo.load(stream, outputs=[video_feed, ear_gauge, mar_gauge, blinks_html, alert_html])

demo.launch(share=True)