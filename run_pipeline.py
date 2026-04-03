"""
CLI script to run the tracking pipeline on a video file.
Usage: python run_pipeline.py --input video.mp4 --output output.mp4
"""

import argparse
import os
import cv2
import numpy as np
from tracker import SportsTracker


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Object Detection and Persistent ID Tracking"
    )
    parser.add_argument("--input", "-i", required=True, help="Path to input video file")
    parser.add_argument("--output", "-o", default="output/tracked_output.mp4", help="Path for output video")
    parser.add_argument("--model", "-m", default="yolov8n.pt", help="YOLOv8 model (yolov8n/s/m/l/x.pt)")
    parser.add_argument("--confidence", "-c", type=float, default=0.3, help="Detection confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="IOU threshold for NMS")
    parser.add_argument("--heatmap", action="store_true", help="Generate movement heatmap")
    parser.add_argument("--screenshots", action="store_true", help="Save sample screenshots")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input video not found: {args.input}")
        return

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else "output", exist_ok=True)

    print(f"Initializing tracker with model: {args.model}")
    tracker = SportsTracker(
        model_path=args.model,
        confidence=args.confidence,
        iou_threshold=args.iou,
    )

    def progress(pct):
        bar_len = 40
        filled = int(bar_len * pct)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"\rProcessing: [{bar}] {pct*100:.1f}%", end="", flush=True)

    print(f"Processing: {args.input}")
    stats = tracker.process_video(args.input, args.output, progress_callback=progress)
    print()

    # Print statistics
    print("\n" + "=" * 50)
    print("TRACKING RESULTS")
    print("=" * 50)
    print(f"Total Frames Processed : {stats['total_frames']}")
    print(f"Video FPS              : {stats['fps']}")
    print(f"Resolution             : {stats['resolution']}")
    print(f"Total Unique IDs       : {stats['total_unique_ids']}")
    print(f"Avg Detections/Frame   : {stats['avg_detections_per_frame']:.1f}")
    print(f"Max Detections (Frame) : {stats['max_detections_in_frame']}")
    print(f"All Track IDs          : {sorted(stats['all_track_ids'])}")
    print(f"Output saved to        : {args.output}")
    print("=" * 50)

    # Generate heatmap
    if args.heatmap:
        res = stats["resolution"].split("x")
        w, h = int(res[0]), int(res[1])
        heatmap = tracker.generate_heatmap(w, h)
        heatmap_path = args.output.replace(".mp4", "_heatmap.png")
        cv2.imwrite(heatmap_path, heatmap)
        print(f"Heatmap saved to: {heatmap_path}")

    # Save screenshots
    if args.screenshots:
        cap = cv2.VideoCapture(args.output)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        screenshot_dir = os.path.join(os.path.dirname(args.output) or "output", "screenshots")
        os.makedirs(screenshot_dir, exist_ok=True)

        for i, fnum in enumerate([0, total // 4, total // 2, 3 * total // 4, total - 1]):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fnum)
            ret, frame = cap.read()
            if ret:
                path = os.path.join(screenshot_dir, f"screenshot_{i+1}.png")
                cv2.imwrite(path, frame)
                print(f"Screenshot saved: {path}")
        cap.release()


if __name__ == "__main__":
    main()
