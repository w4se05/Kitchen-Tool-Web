"""Object detection demo with MobileNet SSD.
This model and code are based on
https://github.com/robmarkcole/object-detection-app
"""

import time
import logging
import queue
from pathlib import Path
from typing import List, NamedTuple

import aiortc
import av
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer, RTCConfiguration
from streamlit_webrtc import __version__ as st_webrtc_version

from utils.download import download_file

# ==========================================
# 1. GLOBAL CONFIG & CONSTANTS (Run once)
# ==========================================
# Cấu hình STUN server của Google (Miễn phí & Ổn định)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
        {"urls": ["stun:stun3.l.google.com:19302"]},
        {"urls": ["stun:stun4.l.google.com:19302"]},
    ]}
)

HERE = Path(__file__).parent
ROOT = HERE.parent

logger = logging.getLogger(__name__)

MODEL_URL = "https://github.com/robmarkcole/object-detection-app/raw/master/model/MobileNetSSD_deploy.caffemodel"
MODEL_LOCAL_PATH = ROOT / "./models/MobileNetSSD_deploy.caffemodel"
PROTOTXT_URL = "https://github.com/robmarkcole/object-detection-app/raw/master/model/MobileNetSSD_deploy.prototxt.txt"
PROTOTXT_LOCAL_PATH = ROOT / "./models/MobileNetSSD_deploy.prototxt.txt"

CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", 
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", 
    "sheep", "sofa", "train", "tvmonitor",
]

class Detection(NamedTuple):
    class_id: int
    label: str
    score: float
    box: np.ndarray
    note: str

@st.cache_resource
def generate_label_colors():
    return np.random.uniform(0, 255, size=(len(CLASSES), 3))

COLORS = generate_label_colors()

download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=23147564)
download_file(PROTOTXT_URL, PROTOTXT_LOCAL_PATH, expected_size=29353)

def get_model():
    return cv2.dnn.readNetFromCaffe(str(PROTOTXT_LOCAL_PATH), str(MODEL_LOCAL_PATH))

# Load model globally so it's fast
net = get_model()


# ==========================================
# 2. MAIN APP FUNCTION
# ==========================================
def app():
    # Load CSS File
    css_file_path = Path(__file__).parent / "static" / "home.css"
    with open(css_file_path, "r", encoding="utf-8") as css_file:
        st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)
    # -------------------------------------------

    st.title("📹 Object Detection Live Feed")
    st.caption("Real-time MobileNet SSD Object Detection")

    # --- A. LOGIC AUTOMATICALLY TURN ON CAM  ---
    should_auto_start = st.session_state.get("auto_start_trigger", False)
    
    if should_auto_start:
        desired_state = True
    else:
        desired_state = None
    # --------------------------------

    # --- B. CREATE LAYOUT ---
    col1, col2 = st.columns([0.65, 0.35])

    # --- C. CONTROLS AND CALLBACK (IMPORTANT) ---
    with col2:
        st.write("#### Controls")
        score_threshold = st.slider("Score threshold", 0.0, 1.0, 0.5, 0.05)
        
        # Initialize queue for this session
        result_queue: "queue.Queue[List[Detection]]" = queue.Queue()

    # ---------------------------------------------------------
    # DEFINE CALLBACK (Must be inside app() to see 'score_threshold')
    # ---------------------------------------------------------
    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")

        # Run inference
        blob = cv2.dnn.blobFromImage(
            image=cv2.resize(image, (300, 300)),
            scalefactor=0.007843,
            size=(300, 300),
            mean=(127.5, 127.5, 127.5),
        )
        net.setInput(blob)
        output = net.forward()

        h, w = image.shape[:2]

        # Convert output
        output = output.squeeze()
        output = output[output[:, 2] >= score_threshold]
        
        detections = [
            Detection(
                class_id=int(detection[1]),
                label=CLASSES[int(detection[1])],
                score=float(detection[2]),
                box=(detection[3:7] * np.array([w, h, w, h])),
                note=f"{CLASSES[int(detection[1])]}"
            )
            for detection in output
        ]

        # Draw boxes
        for detection in detections:
            caption = f"{detection.label}: {round(detection.score * 100, 2)}%"
            color = COLORS[detection.class_id]
            xmin, ymin, xmax, ymax = detection.box.astype("int")

            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(
                image, caption,
                (xmin, ymin - 15 if ymin - 15 > 15 else ymin + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
            )

        result_queue.put(detections)
        return av.VideoFrame.from_ndarray(image, format="bgr24")

    # --- D. DISPLAY CAMERA ---
    with col1:
        st.write("#### Live Camera")
        webrtc_ctx = webrtc_streamer(
            key = "object-detection",
            mode = WebRtcMode.SENDRECV,
            rtc_configuration = RTC_CONFIGURATION,
            video_frame_callback = video_frame_callback,
            media_stream_constraints = {
                "video": True, 
                "audio": False
            },
            async_processing = True,
            desired_playing_state = desired_state
        )

        if should_auto_start and webrtc_ctx.state.playing:
            st.session_state["auto_start_trigger"] = False

            #Rerun to update
            st.rerun()

    with col2:
        st.write("#### Results")
        if st.checkbox("Show detected labels", value=True):
            if webrtc_ctx.state.playing:
                labels_placeholder = st.empty()
                
                while True:
                    try:
                        result = result_queue.get(timeout=1.0)
                        while not result_queue.empty():
                            result = result_queue.get_nowait()
                    except queue.Empty:
                        result = None

                    if result is not None:
                        df = pd.DataFrame(result)
                        if not df.empty:
                            df['Action'] = df['label'].apply(
                                lambda x: f'<a href="./?nav=Wiki%20Search&tab={x}" target="_self" class="result-btn">Wiki ➜</a>'
                            )
                            df['Confidence'] = df['score'].apply(
                                lambda x: f'<span class="score-badge">{x*100:.1f}%</span>'
                            )
                            df['Object'] = df['label'].str.capitalize()

                            # Display result
                            display_df = df[["Object", "Confidence", "Action"]]

                            # Render HTML with class 'custom-table' for CSS
                            html_table = display_df.to_html(escape=False, index=False, classes="custom-table")
                            
                            # Border style
                            labels_placeholder.markdown(html_table, unsafe_allow_html=True)
                        else:
                            labels_placeholder.info("Waiting for object...")
                    
                    time.sleep(0.1)

    # Footer (Full Width)
    st.markdown("---")
    st.caption("Powered by 5 anh em siu nhan.")