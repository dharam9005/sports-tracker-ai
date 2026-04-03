# Technical Report: Multi-Object Detection & Persistent ID Tracking

## 1. Model / Detector Used

**YOLOv8 (You Only Look Once v8)** by Ultralytics is used as the primary object detector.

- **Model variant:** YOLOv8n (nano) as default, with options for YOLOv8s (small) and YOLOv8m (medium)
- **Pre-trained on:** COCO dataset (80 classes, including "person" - class 0)
- **Input:** RGB video frames
- **Output:** Bounding boxes with class labels and confidence scores

**Why YOLOv8:**
- State-of-the-art real-time detection with excellent accuracy-speed tradeoff
- Native support for tracking integration via `model.track()` API
- Pre-trained COCO weights detect persons out-of-the-box without fine-tuning
- Active development and community support from Ultralytics

## 2. Tracking Algorithm Used

**ByteTrack** is used as the multi-object tracking (MOT) algorithm.

- Integrated via Ultralytics' built-in tracker (`bytetrack.yaml` config)
- Operates frame-by-frame using detection outputs from YOLOv8

**How ByteTrack works:**
1. **High-confidence detections** are matched to existing tracks using IoU (Intersection over Union) similarity
2. **Low-confidence detections** (that other trackers would discard) are matched to unmatched tracks in a second association step
3. **Kalman Filter** predicts the next position of each track for association
4. Tracks are initialized, maintained, and terminated based on matching success across frames

## 3. Why This Combination Was Selected

| Criteria | YOLOv8 + ByteTrack |
|---|---|
| **Speed** | YOLOv8n runs at 30+ FPS on GPU; ByteTrack adds minimal overhead |
| **Accuracy** | YOLOv8 achieves strong mAP on COCO; ByteTrack is SOTA on MOT benchmarks |
| **Integration** | Ultralytics natively integrates ByteTrack - single API call |
| **ID Stability** | ByteTrack's two-stage association recovers IDs even with low-confidence detections |
| **Robustness** | Handles occlusion, motion blur, and camera motion well |
| **Simplicity** | Clean, modular pipeline with minimal custom code |

Alternatives considered:
- **DeepSORT:** Uses appearance features (Re-ID) but is slower and can fail with similar-looking subjects (e.g., same jersey color)
- **BoT-SORT:** More complex, marginal improvement over ByteTrack for this use case
- **StrongSORT:** Heavier computational cost without significant benefit for sports footage

## 4. How ID Consistency Is Maintained

The pipeline maintains persistent IDs through several mechanisms:

1. **Kalman Filter Prediction:** Each track maintains a motion model that predicts where the subject will be in the next frame, enabling correct association even when detection temporarily fails.

2. **Two-Stage Association (ByteTrack):**
   - First pass: Match high-confidence detections to tracks using IoU
   - Second pass: Match remaining low-confidence detections to unmatched tracks
   - This recovers subjects that are partially occluded or blurred (low confidence)

3. **Track Lifecycle Management:**
   - New tracks are initialized only after consistent detection over multiple frames (prevents phantom IDs)
   - Lost tracks are kept alive for a configurable number of frames before termination (handles brief occlusion)
   - Re-identification through IoU matching when subjects reappear near their predicted position

4. **Trajectory History:** The pipeline maintains a position history buffer (last 90 frames) per ID, which visually confirms tracking consistency through drawn trajectories.

## 5. Challenges Faced

1. **Similar Appearance:** In team sports, players wearing identical jerseys are difficult to distinguish. ByteTrack relies on spatial proximity rather than appearance, which helps but can fail during close interactions.

2. **Camera Motion:** Rapid panning or zooming shifts all bounding boxes simultaneously, causing IoU-based matching to break. The Kalman filter partially compensates but cannot fully handle extreme camera motion.

3. **Dense Occlusion:** When multiple players cluster (e.g., during a tackle or scrum), detections merge or disappear, leading to ID reassignment after they separate.

4. **Scale Variation:** Subjects far from the camera are small and harder to detect reliably. Switching between wide and close-up shots causes detection inconsistency.

5. **Video Quality:** Compression artifacts, motion blur during fast actions, and low resolution reduce detection confidence.

## 6. Failure Cases Observed

- **Full Occlusion > 2 seconds:** When a player is completely hidden behind others for an extended period, the track is terminated and a new ID is assigned upon reappearance
- **Crossover Events:** Two players crossing paths at high speed may swap IDs if their predicted positions overlap
- **Entry/Exit Points:** Subjects entering from frame edges may briefly receive unstable IDs before the track stabilizes
- **Goalkeeper in Cricket:** Stationary subjects with minimal movement can occasionally lose tracking when briefly occluded

## 7. Possible Improvements

1. **Appearance-based Re-ID:** Add a lightweight appearance embedding (e.g., OSNet) for re-identification after long occlusion periods
2. **Camera Motion Compensation:** Apply homography estimation between frames to stabilize detections before tracking
3. **Sport-specific fine-tuning:** Fine-tune YOLOv8 on sport-specific datasets for improved detection of players in various poses
4. **Team Classification:** Use jersey color clustering (K-means on detected regions) to group players by team
5. **Bird's-eye View Projection:** Apply homography transformation to map positions onto a top-down field view
6. **Temporal Smoothing:** Apply smoothing to bounding boxes and trajectories to reduce jitter
7. **Ensemble Detection:** Combine multiple model outputs for more robust detection in challenging frames
8. **GPU Optimization:** Use TensorRT or ONNX Runtime for inference acceleration in production

## 8. Tools & Libraries

| Tool | Purpose |
|---|---|
| Ultralytics YOLOv8 | Object detection |
| ByteTrack | Multi-object tracking |
| OpenCV | Video I/O, image processing |
| Supervision | Annotation and visualization |
| Streamlit | Web application |
| NumPy | Numerical computation |
| Matplotlib | Chart generation |
| yt-dlp | Video downloading |
