"""
Multi-Object Detection and Persistent ID Tracking Pipeline
Uses YOLOv8 for detection and ByteTrack for multi-object tracking.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from collections import defaultdict
import os


class SportsTracker:
    """Main tracking pipeline for sports/event footage."""

    def __init__(self, model_path="yolov8n.pt", confidence=0.3, iou_threshold=0.5):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.iou_threshold = iou_threshold

        # Tracking state
        self.track_history = defaultdict(list)
        self.id_colors = {}

        # Annotation settings
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(
            text_scale=0.5, text_thickness=1, text_padding=5
        )
        self.trace_annotator = sv.TraceAnnotator(
            thickness=2, trace_length=50
        )

    def _get_color_for_id(self, track_id):
        """Generate a consistent color for each unique track ID."""
        if track_id not in self.id_colors:
            np.random.seed(int(track_id) * 42)
            self.id_colors[track_id] = tuple(
                int(c) for c in np.random.randint(50, 255, size=3)
            )
        return self.id_colors[track_id]

    def process_frame(self, frame):
        """
        Process a single frame: detect and track objects.
        Returns annotated frame and detection metadata.
        """
        results = self.model.track(
            frame,
            persist=True,
            conf=self.confidence,
            iou=self.iou_threshold,
            tracker="bytetrack.yaml",
            verbose=False,
        )

        result = results[0]
        detections = sv.Detections.from_ultralytics(result)

        # Extract tracker IDs
        if result.boxes.id is not None:
            tracker_ids = result.boxes.id.int().cpu().numpy()
            detections.tracker_id = tracker_ids
        else:
            detections.tracker_id = np.array([], dtype=int)

        # Filter for person class (class 0 in COCO) - primary for sports
        person_mask = detections.class_id == 0
        if person_mask.any():
            detections = detections[person_mask]

        # Build labels
        labels = []
        for i in range(len(detections)):
            tid = detections.tracker_id[i] if detections.tracker_id is not None and i < len(detections.tracker_id) else "?"
            conf = detections.confidence[i] if detections.confidence is not None else 0
            labels.append(f"ID:{tid} ({conf:.2f})")

            # Store trajectory
            if tid != "?":
                cx = int((detections.xyxy[i][0] + detections.xyxy[i][2]) / 2)
                cy = int((detections.xyxy[i][1] + detections.xyxy[i][3]) / 2)
                self.track_history[int(tid)].append((cx, cy))
                # Keep last 90 positions
                if len(self.track_history[int(tid)]) > 90:
                    self.track_history[int(tid)] = self.track_history[int(tid)][-90:]

        # Annotate frame
        annotated = self.box_annotator.annotate(scene=frame.copy(), detections=detections)
        annotated = self.label_annotator.annotate(scene=annotated, detections=detections, labels=labels)
        annotated = self.trace_annotator.annotate(scene=annotated, detections=detections)

        # Draw trajectory lines manually for better visibility
        for tid, positions in self.track_history.items():
            if len(positions) > 1:
                color = self._get_color_for_id(tid)
                pts = np.array(positions, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated, [pts], False, color, 2)

        metadata = {
            "num_detections": len(detections),
            "tracker_ids": list(detections.tracker_id) if detections.tracker_id is not None else [],
            "confidences": list(detections.confidence) if detections.confidence is not None else [],
        }

        return annotated, metadata

    def process_video(self, input_path, output_path, progress_callback=None):
        """
        Process an entire video file.
        Returns statistics about the tracking.
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Reset tracking state
        self.track_history.clear()
        self.id_colors.clear()

        all_ids = set()
        frame_counts = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            annotated, metadata = self.process_frame(frame)
            out.write(annotated)

            all_ids.update(metadata["tracker_ids"])
            frame_counts.append(metadata["num_detections"])
            frame_idx += 1

            if progress_callback:
                progress_callback(frame_idx / total_frames)

        cap.release()
        out.release()

        stats = {
            "total_frames": frame_idx,
            "fps": fps,
            "resolution": f"{width}x{height}",
            "total_unique_ids": len(all_ids),
            "avg_detections_per_frame": np.mean(frame_counts) if frame_counts else 0,
            "max_detections_in_frame": max(frame_counts) if frame_counts else 0,
            "all_track_ids": sorted(all_ids),
            "trajectory_data": dict(self.track_history),
        }

        return stats

    def generate_heatmap(self, width, height):
        """Generate a movement heatmap from tracked trajectories."""
        heatmap = np.zeros((height, width), dtype=np.float32)

        for tid, positions in self.track_history.items():
            for x, y in positions:
                if 0 <= x < width and 0 <= y < height:
                    cv2.circle(heatmap, (x, y), 15, 1, -1)

        heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        heatmap_color = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        return heatmap_color

    def generate_object_count_chart(self, frame_counts, fps):
        """Generate object count over time chart data."""
        import matplotlib.pyplot as plt

        times = [i / fps for i in range(len(frame_counts))]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(times, frame_counts, color="#2196F3", linewidth=1.5)
        ax.fill_between(times, frame_counts, alpha=0.3, color="#2196F3")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Number of Detected Subjects")
        ax.set_title("Object Count Over Time")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig
