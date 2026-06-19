# KasinoMonitoring

Realtime MVP for casino visitor monitoring from RTSP cameras or local video files.

Current stage:

- reads one RTSP stream, video file, or local camera index;
- detects people with Ultralytics YOLO;
- tracks people inside one camera with Ultralytics BoT-SORT or ByteTrack;
- detects faces with InsightFace `FaceAnalysis` when faces are visible;
- creates ArcFace embeddings for good face observations;
- can use TorchReID OSNet body embeddings when `torchreid` is installed;
- extracts stronger body/clothing visual descriptors from the person crop using HSV, LAB, spatial color bands, and gradient features;
- estimates crop quality and stores only useful visual samples;
- stores body proportions and movement history;
- keeps in-memory `Person ID` profiles;
- stores several samples per person and per modality;
- tries to recognize repeated appearances;
- draws `Person ID` labels over the live OpenCV window;
- can optionally save annotated video to disk.

## Run

Install dependencies with your Python environment manager, then run:

```bash
python main.py --source rtsp://user:password@host:554/stream --camera-id cam-001
```

Local video file:

```bash
python main.py --source /path/to/video.mp4 --camera-id file-001
```

Save annotated result to a file:

```bash
python main.py \
  --source video.mp4 \
  --camera-id test-video \
  --output result.mp4
```

Save annotated result without showing the realtime window:

```bash
python main.py \
  --source video.mp4 \
  --camera-id test-video \
  --output result.mp4 \
  --no-display
```

Local webcam:

```bash
python main.py --source 0 --camera-id webcam-001
```

Close the OpenCV window with `q` or `Esc`.

## Useful Options

```bash
python main.py \
  --source rtsp://user:password@host:554/stream \
  --camera-id entrance-01 \
  --yolo-model yolov8n.pt \
  --person-confidence 0.25 \
  --match-threshold 0.52 \
  --det-size 640x640 \
  --min-face-score 0.50 \
  --max-samples-per-person 20
```

## macOS Performance Presets

Stable ID preset with BoT-SORT:

```bash
python main.py \
  --source video.mp4 \
  --camera-id test-video \
  --output result.mp4 \
  --no-display \
  --debug-csv debug.csv \
  --tracker trackers/casino_botsort.yaml \
  --yolo-device mps \
  --yolo-imgsz 640 \
  --process-every-n-frames 1 \
  --face-every-n-frames 6 \
  --min-confirmed-hits 3 \
  --person-confidence 0.20
```

TorchReID / OSNet body ReID:

```bash
python main.py \
  --source video.mp4 \
  --camera-id test-video \
  --output result_osnet.mp4 \
  --no-display \
  --tracker trackers/casino_botsort.yaml \
  --body-reid-backend torchreid \
  --body-reid-model osnet_x0_25 \
  --body-reid-device auto \
  --yolo-device mps \
  --yolo-imgsz 640 \
  --process-every-n-frames 1
```

If TorchReID is not installed, use the default `auto` backend to fall back to handcrafted descriptors:

```bash
--body-reid-backend auto
```

Install TorchReID dependencies separately in your environment. The exact PyTorch install command depends on CPU/GPU/macOS setup; on macOS, start with the official PyTorch install for your machine, then install TorchReID.

Maximum quality preset:

```bash
uv run main.py \
  --source video.mp4 \
  --output result_max_quality.mp4 \
  --no-display \
  --debug-csv debug_max_quality.csv \
  --tracker trackers/casino_botsort.yaml \
  --yolo-model yolov8m-seg.pt \
  --yolo-device mps \
  --yolo-imgsz 1280 \
  --body-reid-backend torchreid \
  --body-reid-model osnet_x1_0 \
  --body-reid-device auto \
  --process-every-n-frames 1 \
  --face-every-n-frames 3 \
  --det-size 960x960 \
  --person-confidence 0.18 \
  --min-confirmed-hits 4 \
  --hold-missing-frames 8 \
  --hold-overlap-iou 0.15 \
  --hold-min-overlap 0.50 \
  --min-live-visual-quality 0.18 \
  --live-nms-min-overlap 0.60 \
  --new-track-match-threshold 0.84 \
  --match-threshold 0.64 \
  --max-samples-per-person 40
```

For this preset, use a segmentation model (`*-seg.pt`). When masks are available, body/clothing features and OSNet crops are extracted from the person mask instead of the full bbox, which improves quality during crossings and partial occlusions.

Fast Apple Silicon preset:

```bash
python main.py \
  --source video.mp4 \
  --camera-id test-video \
  --tracker trackers/casino_botsort.yaml \
  --yolo-device mps \
  --yolo-imgsz 416 \
  --process-every-n-frames 2 \
  --face-every-n-frames 6 \
  --person-confidence 0.20
```

Higher accuracy Apple Silicon preset:

```bash
python main.py \
  --source video.mp4 \
  --camera-id test-video \
  --yolo-device mps \
  --yolo-imgsz 640 \
  --process-every-n-frames 1 \
  --face-every-n-frames 3 \
  --person-confidence 0.20
```

CPU-only fallback:

```bash
python main.py \
  --source video.mp4 \
  --camera-id test-video \
  --yolo-device cpu \
  --yolo-imgsz 416 \
  --process-every-n-frames 3 \
  --face-every-n-frames 0
```

Performance knobs:

- `--yolo-device mps` uses Apple Silicon GPU acceleration for Ultralytics YOLO.
- `--yolo-imgsz 416` is much faster than `640`, but can miss small distant people.
- `--process-every-n-frames 2` runs YOLO/ReID every second frame and reuses the last labels between frames.
- `--process-every-n-frames 1` is recommended when `--tracker` is enabled and ID stability matters more than speed.
- `--tracker trackers/casino_botsort.yaml` is the default tuned tracker; `--tracker bytetrack.yaml` is a lighter alternative; `--tracker off` disables tracking.
- `--body-reid-backend torchreid` uses OSNet embeddings for body ReID.
- `--body-reid-model osnet_x0_25` is lighter; `osnet_x1_0` is heavier and usually more accurate.
- `--track-memory-seconds 30` keeps a local tracker ID mapped to the same internal `Person ID`.
- `--face-every-n-frames 6` runs InsightFace less often; use `0` to disable faces completely.
- `--max-people 8` limits work in crowded scenes to the top confidence detections.
- `--hold-missing-frames 6` keeps drawing the last known box briefly when detector/tracker misses a person for a few frames.
- `--hold-overlap-iou 0.25` removes a held box when it overlaps a current live detection, which reduces phantom boxes when people cross.
- `--hold-min-overlap 0.65` removes held boxes that are mostly covered by a live detection.
- `--min-live-visual-quality 0.22` removes weak live detections before ReID/drawing.
- `--live-nms-min-overlap 0.72` removes nested duplicate live boxes.
- `--min-confirmed-hits 3` keeps new IDs tentative until they are observed several times.
- `--debug-csv debug.csv` writes per-frame detection/filtering/matching diagnostics.

If boxes flicker because a person disappears for one or two frames, increase:

```bash
--hold-missing-frames 10
```

If stale boxes remain too long after a person exits, decrease:

```bash
--hold-missing-frames 2
```

If phantom boxes appear when people pass through each other, make held-box suppression stricter:

```bash
--hold-overlap-iou 0.15 --hold-min-overlap 0.50
```

If the phantom is a live detector/tracker box rather than a held box, make live filtering stricter:

```bash
--min-live-visual-quality 0.32 --live-nms-min-overlap 0.55
```

## Lifecycle And Debugging

New `Person ID`s start as `tentative` and are not drawn until they reach `--min-confirmed-hits`. This removes many short-lived false positives without forcing aggressive detector thresholds.

To draw tentative IDs for debugging:

```bash
--draw-tentative
```

To inspect why boxes are kept or dropped:

```bash
--debug-csv debug.csv
```

The CSV includes frame number, event type, drop reason, bbox, track ID, person ID, profile status, detection score, visual quality, and match scores.

## Tracker Tuning

The default tracker config is [trackers/casino_botsort.yaml](trackers/casino_botsort.yaml). It starts fewer weak tracks and keeps confirmed tracks alive longer through short occlusions.

Useful tracker fields:

- `new_track_thresh`: raise it to reduce phantom new tracks; lower it if real people are not starting tracks.
- `track_buffer`: raise it to keep tracks through longer occlusions; lower it if stale tracks linger.
- `match_thresh`: raise it to reduce ID switches; lower it if tracks break too often.
- `track_high_thresh`: raise it to use cleaner detections; lower it if people are missed.

## Reducing Wrong Recognition

The MVP is intentionally conservative with new tracker IDs. Once BoT-SORT keeps a track alive, that track keeps the same internal `Person ID`. When a tracker ID changes, the system only merges it into an existing `Person ID` if the evidence is strong.

Default conservative settings:

```bash
--new-track-match-threshold 0.72 \
--min-visual-sample-confidence 0.55
```

The label contains `vq=...`, the visual quality score used to decide whether a body/clothing sample is good enough for memory.

If different people are still merged into one ID, make new-track merging stricter:

```bash
--new-track-match-threshold 0.82 --match-threshold 0.62
```

If the same person often receives a new ID after the tracker loses them, loosen carefully:

```bash
--new-track-match-threshold 0.65
```

If low-quality detections pollute a person's memory, raise the sample quality gate:

```bash
--min-visual-sample-confidence 0.70
```

## Spatial ID Stability

Same-camera matching uses a spatial gate so an ID cannot jump to a similar-looking person on the other side of the frame. The gate compares the current detection center with the last known center for that `Person ID`, allowing more movement when more video time has passed.

There is also a short-term `position_score` based on bbox overlap and center distance. It keeps the same ID when a person stands in one place, turns around, and their clothing/body histogram changes.

Default behavior:

```bash
--spatial-gate \
--spatial-base-radius 80 \
--max-center-speed 900 \
--spatial-gate-seconds 4 \
--strong-face-threshold 0.62 \
--continuity-threshold 0.78
```

If IDs still jump across the frame, make the gate stricter:

```bash
--spatial-base-radius 50 --max-center-speed 600
```

If the same person gets a new ID after fast movement, camera shake, or low FPS processing, loosen it:

```bash
--spatial-base-radius 120 --max-center-speed 1300
```

If a standing person still gets a new ID after turning around, lower the continuity threshold:

```bash
--continuity-threshold 0.68
```

For videos with hard cuts between different camera views inside one file, disable the same-camera gate:

```bash
--no-spatial-gate
```

ONNX Runtime provider selection for InsightFace:

```bash
python main.py \
  --source 0 \
  --providers CUDAExecutionProvider CPUExecutionProvider
```

## Person Memory

Profiles are stored only in RAM for this MVP:

```python
{
    "person_id": "P-0001",
    "face_embeddings": [],
    "body_embeddings": [],
    "body_embedding_qualities": [],
    "clothing_features": [],
    "clothing_feature_qualities": [],
    "body_proportions": [],
    "movement_history": [],
    "tracks": [],
    "first_seen": timestamp,
    "last_seen": timestamp,
    "cameras": [],
}
```

The current matching logic is person-first and multi-modal. YOLO finds full people, then InsightFace adds a face embedding only when a usable face is visible. If face quality is low or no face is present, the face weight is redistributed to body, clothing, proportions, movement, and location-time features.

Default scoring:

```python
final_score = (
    0.45 * face_score
    + 0.25 * body_score
    + 0.15 * clothing_score
    + 0.05 * proportions_score
    + 0.05 * gait_score
    + 0.05 * location_time_score
)
```

Important limitation: when `--body-reid-backend handcrafted` is used, `body_embedding` and `clothing_features` are still handcrafted descriptors extracted locally from the person crop. Use `--body-reid-backend torchreid` for OSNet body embeddings.

For crowded scenes where similar uniforms or dark clothing cause accidental merges, increase:

```bash
--match-threshold 0.58
```

For sparse test videos where the same person is often treated as new, decrease:

```bash
--match-threshold 0.46
```

## Next Implementation Steps

1. Add model caching/config docs for OSNet weights in deployment.
2. Add CLIP embeddings for clothing and accessories.
3. Move vector memory to Qdrant and event/profile metadata to PostgreSQL.
4. Add multi-camera ingestion workers and a FastAPI/React dashboard.
