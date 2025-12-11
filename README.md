# Kitchen Tool Web: Real-Time Object Detection with Wiki Lookup

A real-time computer vision web app built with Streamlit that uses OpenCV’s DNN with MobileNet SSD to detect common kitchen tools from a webcam feed. Detected classes are presented alongside confidence scores, with quick links to a “Wiki” page for additional information.

## Key Features

- Real-time video processing via `streamlit-webrtc`
- Object detection using MobileNet SSD (Caffe)
- Adjustable confidence threshold
- Clickable labels linking to an in-app “Wiki” page for details
- Overlay bounding boxes, class labels, and confidence scores
- Streamlit-based UI for a simple, fast, and interactive experience

## Tech Stack

- Python 3.x
- [Streamlit](https://streamlit.io/)
- [OpenCV](https://opencv.org/) (DNN module)
- MobileNet SSD (pre-trained Caffe model)
- NumPy, Pandas

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/w4se05/Kitchen-Tool-Web.git
cd Kitchen-Tool-Web
```

### 2. Create and activate a virtual environment

Windows:

```bash
python -m venv venv
.\venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Optionally update Streamlit:

```bash
pip install --upgrade streamlit
```

### 4. Run the app

```bash
streamlit run Menu.py
```

## Project Structure

```text
├── .devcontainer/        # Dev container configuration (optional)
├── .dockerignore
├── .gitignore
├── .streamlit/
│   └── config.toml       # Streamlit UI/theme configuration
├── .vscode/              # Editor settings (optional)
├── Dockerfile            # Container build configuration
├── Menu.py               # Main Streamlit app entry point
├── models/               # MobileNet SSD Caffe model files
├── pages/                # Streamlit multipage files (About, Home, Wiki, etc.)
├── requirements.txt      # Python dependencies
├── runtime.txt           # Runtime pin (e.g., for certain platforms)
├── utils/                # Helper utilities
└── README.md             # Project documentation
```

Notes:

- The app relies on model files in `models/` (MobileNet SSD prototxt and caffemodel).
- Streamlit multipage support typically loads files in the `pages/` directory.

## Configuration

- Streamlit settings (theme, layout) can be adjusted in `.streamlit/config.toml`.
- Confidence thresholds and other parameters can be exposed via Streamlit widgets in `Menu.py`.
- If you deploy via Docker, see the `Dockerfile`.

## Usage Tips

- Ensure your webcam is accessible by your browser when prompted.
- Lowering the confidence threshold increases detections but may add false positives.
- Use the “Wiki” page link from the detection results to learn more about each object class.

## Docker (Optional)

Build and run locally:

```bash
docker build -t kitchen-tool-web .
docker run -p 8501:8501 kitchen-tool-web
```

Then open <http://localhost:8501> in your browser.

## License

This project is distributed under the MIT License.

## Acknowledgments

- Inspired by the Streamlit community discussion on `streamlit-webrtc`: [New component: streamlit-webrtc](https://discuss.streamlit.io/t/new-component-streamlit-webrtc-a-new-way-to-deal-with-real-time-media-streams/8669)
- MobileNet SSD model authors and OpenCV contributors
