# Multi-Object Detection & Persistent ID Tracking in Sports Footage

A computer vision pipeline for detecting and tracking all subjects in sports/event videos with **consistent unique IDs** across frames. Built with **YOLOv8** for detection and **ByteTrack** for multi-object tracking.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)

## Live Demo

🔗 **[Try the Live App](https://dharam9005-sports-tracker-ai.streamlit.app)**

## Features

- **Real-time Object Detection** using YOLOv8 (nano/small/medium models)
- **Persistent ID Tracking** via ByteTrack algorithm - IDs remain stable across frames
- **Trajectory Visualization** - movement paths drawn for each tracked subject
- **Movement Heatmap** generation showing activity zones
- **Detection Count Over Time** analytics
- **Web Interface** via Streamlit for easy interaction
- **CLI Pipeline** for batch processing
- Handles: occlusion, motion blur, camera motion, scale changes, similar-looking subjects

## Sample Results

| Detection & Tracking | Movement Heatmap |
|---|---|
| ![Tracking](output/screenshots/screenshot_1.png) | ![Heatmap](output/tracked_output_heatmap.png) |

## Source Video

**Video Used:** Public cricket/football footage from YouTube
**Link:** *(Include the public video URL you used here)*

> The video is publicly accessible and used solely for educational/assessment purposes.

## Installation

### Prerequisites
- Python 3.9+
- pip
- ffmpeg (optional, for video format conversion)

### Setup

```bash
# Clone the repository
git clone https://github.com/dharam9005/sports-tracker-ai.git
cd sports-tracker-ai

# Install dependencies
pip install -r requirements.txt

# YOLOv8 model will be auto-downloaded on first run
```

## Usage

### Option 1: Web App (Streamlit)

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser. You can:
- Upload a video file directly
- Paste a YouTube/public URL to download and process
- Adjust detection confidence and model size
- View annotated output, heatmaps, and statistics

### Option 2: Command Line

```bash
# Basic usage
python run_pipeline.py --input video.mp4 --output output/tracked.mp4

# With heatmap and screenshots
python run_pipeline.py --input video.mp4 --output output/tracked.mp4 --heatmap --screenshots

# Use a larger model for better accuracy
python run_pipeline.py --input video.mp4 --model yolov8m.pt --confidence 0.25
```

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--input`, `-i` | *(required)* | Path to input video |
| `--output`, `-o` | `output/tracked_output.mp4` | Output video path |
| `--model`, `-m` | `yolov8n.pt` | YOLOv8 model variant |
| `--confidence`, `-c` | `0.3` | Detection confidence threshold |
| `--iou` | `0.5` | IOU threshold for NMS |
| `--heatmap` | `False` | Generate movement heatmap |
| `--screenshots` | `False` | Save sample screenshots |

## Project Structure

```
sports-tracker-ai/
├── app.py                 # Streamlit web application
├── tracker.py             # Core detection + tracking pipeline
├── run_pipeline.py        # CLI entry point
├── video_downloader.py    # yt-dlp video download utility
├── requirements.txt       # Python dependencies
├── technical_report.md    # Detailed technical report
├── output/                # Generated outputs (gitignored)
│   ├── tracked_output.mp4
│   ├── screenshots/
│   └── tracked_output_heatmap.png
└── README.md
```

## Assumptions & Limitations

### Assumptions
- Input videos contain human subjects (players, athletes, participants)
- Videos are of reasonable quality (360p-1080p)
- Subjects are visible for enough frames for tracking initialization

### Limitations
- **ID Switches:** Prolonged full occlusion (>2 seconds) may cause ID reassignment
- **Crowded Scenes:** Very dense crowds (50+ overlapping people) can reduce tracking accuracy
- **Small Subjects:** Subjects occupying <20px in frame may be missed by the detector
- **Processing Speed:** ~15-30 FPS on GPU, ~2-5 FPS on CPU (YOLOv8n)
- **Person-only filtering:** Currently filters for person class; can be adjusted for other objects

## Dependencies

- `ultralytics` - YOLOv8 detection framework
- `opencv-python` - Video I/O and image processing
- `supervision` - Detection annotation and visualization
- `numpy` - Numerical operations
- `streamlit` - Web application framework
- `yt-dlp` - Video downloading from public sources
- `matplotlib` - Chart generation
- `scipy`, `lap` - ByteTrack optimization

## License

This project is for educational and assessment purposes only.
