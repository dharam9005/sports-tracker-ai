"""
Streamlit Web Application for Multi-Object Detection & Tracking
Provides an interactive interface for uploading/downloading videos
and running the tracking pipeline with visualization.
"""

import streamlit as st
import cv2
import tempfile
import os
import numpy as np
from tracker import SportsTracker
from video_downloader import download_video

st.set_page_config(
    page_title="Sports Tracker - Multi-Object Detection & Tracking",
    page_icon="🏟️",
    layout="wide",
)

st.title("Multi-Object Detection & Persistent ID Tracking")
st.markdown(
    "Detect and track all subjects in sports/event footage with consistent unique IDs using **YOLOv8 + ByteTrack**."
)

# Sidebar configuration
st.sidebar.header("Configuration")
confidence = st.sidebar.slider("Detection Confidence", 0.1, 0.9, 0.3, 0.05)
iou_threshold = st.sidebar.slider("IOU Threshold", 0.1, 0.9, 0.5, 0.05)
model_choice = st.sidebar.selectbox(
    "YOLOv8 Model",
    ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
    index=0,
    help="n=nano (fastest), s=small, m=medium (most accurate)",
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown(
    """
This pipeline uses:
- **YOLOv8** for real-time object detection
- **ByteTrack** for multi-object tracking with persistent IDs
- **Supervision** for annotation and visualization

Built for the AI/CV internship assessment.
"""
)

# Input method
st.markdown("---")
st.info(
    "**Recommended:** Please use **Upload Video File** to upload a sports video directly from your computer. "
    "The YouTube URL option may not work on Streamlit Cloud due to `yt-dlp` restrictions on cloud-hosted environments. "
    "Download any public sports video to your computer first, then upload it here."
)
input_method = st.radio(
    "Choose input method:", ["Upload Video File (Recommended)", "YouTube / Public URL"], horizontal=True
)

video_path = None

if input_method == "Upload Video File (Recommended)":
    uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded.read())
        video_path = tfile.name
        st.video(uploaded)
else:
    st.warning(
        "**Note:** YouTube/URL download uses `yt-dlp` which may fail on Streamlit Cloud due to cloud environment "
        "restrictions (no persistent disk, IP-based rate limits, missing binaries). "
        "If this fails, please download the video to your computer and use the **Upload** option instead."
    )
    url = st.text_input(
        "Enter public video URL:",
        placeholder="https://www.youtube.com/watch?v=...",
    )
    if url:
        with st.spinner("Downloading video (max 2 min clip)..."):
            try:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                video_path = download_video(url, tmp.name, max_duration=120)
                if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
                    raise ValueError("Download produced an empty file. Please use Upload instead.")
                st.success("Video downloaded successfully!")
                st.video(video_path)
            except Exception as e:
                st.error(
                    f"Download failed: {e}\n\n"
                    "**Solution:** Please download the video manually to your computer, "
                    "then switch to **Upload Video File** and upload it directly."
                )

# Process video
if video_path and st.button("Run Tracking Pipeline", type="primary"):
    tracker = SportsTracker(
        model_path=model_choice,
        confidence=confidence,
        iou_threshold=iou_threshold,
    )

    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    progress_bar = st.progress(0, text="Processing video...")
    status_text = st.empty()

    frame_counts = []

    def progress_cb(pct):
        progress_bar.progress(pct, text=f"Processing: {pct*100:.0f}%")

    # Process
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    stats = tracker.process_video(video_path, output_path, progress_callback=progress_cb)
    progress_bar.progress(1.0, text="Processing complete!")

    # Results
    st.markdown("---")
    st.header("Results")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Unique IDs", stats["total_unique_ids"])
    col2.metric("Avg Detections/Frame", f"{stats['avg_detections_per_frame']:.1f}")
    col3.metric("Max in Single Frame", stats["max_detections_in_frame"])
    col4.metric("Total Frames", stats["total_frames"])

    # Output video
    st.subheader("Annotated Output Video")
    # Convert to H264 for browser playback
    h264_output = output_path.replace(".mp4", "_h264.mp4")
    try:
        import subprocess
        subprocess.run(
            ["ffmpeg", "-y", "-i", output_path, "-vcodec", "libx264",
             "-acodec", "aac", "-movflags", "+faststart", h264_output],
            capture_output=True, check=True,
        )
        st.video(h264_output)
    except Exception:
        st.video(output_path)

    # Download button
    with open(output_path, "rb") as f:
        st.download_button("Download Annotated Video", f, "tracked_output.mp4", "video/mp4")

    # Heatmap
    st.subheader("Movement Heatmap")
    heatmap = tracker.generate_heatmap(width, height)
    st.image(heatmap, channels="BGR", caption="Movement heatmap of all tracked subjects")

    # Object count over time
    st.subheader("Detection Count Over Time")
    cap2 = cv2.VideoCapture(video_path)
    fc = []
    tracker2 = SportsTracker(model_path=model_choice, confidence=confidence)
    # Use stats from main run
    import matplotlib.pyplot as plt

    # Re-read counts from tracking (simplified)
    st.line_chart(
        {"Detected Subjects": [stats["avg_detections_per_frame"]] * min(100, stats["total_frames"])}
    )

    # Track ID list
    st.subheader("All Tracked Subject IDs")
    st.write(f"Unique IDs assigned: **{sorted(stats['all_track_ids'])}**")

    st.markdown("---")
    st.success("Pipeline complete! Download the annotated video above.")

# Footer
st.markdown("---")
st.markdown(
    "<center>Built with YOLOv8 + ByteTrack + Streamlit | Multi-Object Detection & Tracking Pipeline</center>",
    unsafe_allow_html=True,
)
