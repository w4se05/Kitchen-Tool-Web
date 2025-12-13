"""Object detection demo with MobileNet SSD.
This model and code are based on
https://github.com/robmarkcole/object-detection-app
"""

import queue
from pathlib import Path
import time
from typing import List, NamedTuple
import av
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer, RTCConfiguration
from streamlit_webrtc import __version__ as st_webrtc_version
from ultralytics import YOLO
from utils.download import download_file
from urllib.parse import quote

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

MODEL_LOCAL_PATH = ROOT / "models" / "best.pt"
CLASSES = [
    "Spoon",
    "Fork",
    "Knife",
    "Tongs",
    "Bowl",
    "Plate",
    "Pot",
]
IDX_TO_NAME = {i: name for i, name in enumerate(CLASSES)}

class Detection(NamedTuple):
    class_id: int
    label: str
    score: float
    box: np.ndarray
    note: str

@st.cache_resource
def generate_label_colors() -> np.ndarray:
    # Deterministic color palette for consistent UX/screenshots
    rng = np.random.default_rng(42)
    return rng.uniform(0, 255, size=(len(CLASSES), 3))

COLORS = generate_label_colors()


@st.cache_resource
def load_model() -> YOLO:
    """Load the YOLO model once and cache as a resource."""
    if not MODEL_LOCAL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_LOCAL_PATH}. "
            "Please place your trained model at this path."
        )
    model = YOLO(MODEL_LOCAL_PATH)
    # Optional warm-up to stabilize first-frame latency
    try:
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        model.predict(source=dummy, conf=0.25, iou=0.5, verbose=False)
    except Exception:
        # If warm-up fails, continue; not critical
        pass
    return model

NET = load_model()
def home_app():
    # Load CSS once per session
    if "home_css_loaded" not in st.session_state:
        css_file_path = HERE / "static" / "home.css"
        try:
            with open(css_file_path, "r", encoding="utf-8") as css_file:
                st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)
        except FileNotFoundError:
            st.warning(f"CSS file not found at {css_file_path}. Continuing without custom styles.")
        st.session_state["home_css_loaded"] = True

    st.title("📹 Object Detection Live Feed")
    st.caption("Real-time YOLOv11 Kitchen Utils Detection!")

    # Controls and state
    should_auto_start = st.session_state.get("auto_start_trigger", False)
    desired_state = True if should_auto_start else None
    col1, col2 = st.columns([0.65, 0.35])

    # Shared queue for detection results (non-blocking updates)
    result_queue: "queue.Queue[List[Detection]]" = st.session_state.get("result_queue")
    if result_queue is None:
        result_queue = queue.Queue(maxsize=5)
        st.session_state["result_queue"] = result_queue

    with col2:
        st.write("#### Controls")
        score_threshold = st.slider("Score threshold", 0.0, 1.0, 0.5, 0.05)
        iou_threshold = st.slider("IoU threshold", 0.1, 0.9, 0.5, 0.05)
        max_det = st.number_input("Max detections per frame", min_value=1, max_value=300, value=100, step=1)
        target_resolution = st.selectbox(
            "Video resolution",
            options=["default", "640x480", "1280x720"],
            index=0,
            help="Lower resolutions reduce CPU usage."
        )
        media_constraints = {"video": True, "audio": False}
        if target_resolution != "default":
            w, h = map(int, target_resolution.split("x"))
            media_constraints["video"] = {"width": {"ideal": w}, "height": {"ideal": h}}

    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")  # (H, W, 3) BGR

        # Convert to RGB if required by model
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Run inference
        results = NET.predict(
            source=image_rgb,
            conf=score_threshold,
            iou=iou_threshold,
            max_det=int(max_det),
            verbose=False,
        )

        boxes = results[0].boxes
        output = boxes.cpu().numpy()  # has fields .xyxy, .conf, .cls

        detections: List[Detection] = []

        # Draw boxes
        num_det = len(output.cls) if hasattr(output, "cls") else 0
        for i in range(num_det):
            cls_id = int(output.cls[i])
            name = IDX_TO_NAME.get(cls_id, str(cls_id))
            score = float(output.conf[i])
            xmin, ymin, xmax, ymax = output.xyxy[i].astype("int")
            color = tuple(int(c) for c in COLORS[cls_id])

            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
            caption = f"{name}: {score * 100:.1f}%"
            cv2.putText(
                image,
                caption,
                (xmin, ymin - 15 if ymin - 15 > 15 else ymin + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

            detections.append(
                Detection(
                    class_id=cls_id,
                    label=name,
                    score=score,
                    box=np.array([xmin, ymin, xmax, ymax]),
                    note="",
                )
            )
        # Push detections once per frame, non-blocking
        if detections:
            try:
                # Drop stale data if queue is full
                if result_queue.full():
                    _ = result_queue.get_nowait()
                result_queue.put_nowait(detections)
            except queue.Full:
                pass

        return av.VideoFrame.from_ndarray(image, format="bgr24")

    # --- Live Camera ---
    with col1:
        st.write("#### Live Camera")
        webrtc_ctx = webrtc_streamer(
            key="object-detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_frame_callback=video_frame_callback,
            media_stream_constraints=media_constraints,
            async_processing=True,
            desired_playing_state=desired_state,
        )

        if should_auto_start and webrtc_ctx.state.playing:
            st.session_state["auto_start_trigger"] = False
            # Rerun to update UI state once
            st.rerun()

    with col2:
        st.write("#### Results")
        labels_placeholder = st.empty()

        # We only loop while the stream is playing. UI widgets won't be processed during this loop,
        # so avoid using a checkbox to control the loop itself.
        while webrtc_ctx.state.playing:
            # Pull the latest available detections without blocking too long
            try:
                result = result_queue.get(timeout=0.1)
                # Drain to keep only the most recent batch
                while not result_queue.empty():
                    result = result_queue.get_nowait()
            except queue.Empty:
                result = None

            if result:
                df = pd.DataFrame(result)
                if not df.empty:
                    df["Object"] = df["label"].str.capitalize()
                    df["Confidence"] = df["score"].apply(lambda x: f'<span class="score-badge">{x*100:.1f}%</span>')
                    df["Action"] = df["label"].apply(
                        lambda x: f'<a href="./?nav=Wiki%20Search&tab={quote(x)}" class="result-btn">Wiki ➜</a>'
                    )
                    display_df = df[["Object", "Confidence", "Action"]]
                    html_table = display_df.to_html(escape=False, index=False, classes="custom-table")
                    labels_placeholder.markdown(html_table, unsafe_allow_html=True)
                else:
                    labels_placeholder.info("Waiting for objects...")
            else:
                labels_placeholder.info("No detections yet. Move an object into the frame.")
            time.sleep(0.1)
        labels_placeholder.empty()

    # Footer (Full Width)
    st.markdown("---")
    st.caption("Powered by 5 anh em siu nhan.")