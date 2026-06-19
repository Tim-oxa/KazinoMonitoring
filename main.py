from __future__ import annotations

import argparse
import csv
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    import cv2
    import numpy as np
    from insightface.app import FaceAnalysis
    from ultralytics import YOLO


RFDETR_DETECTOR_CLASSES = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
}

RFDETR_SEGMENTATION_CLASSES = {
    "nano": "RFDETRSegNano",
    "small": "RFDETRSegSmall",
    "medium": "RFDETRSegMedium",
    "large": "RFDETRSegLarge",
    "xlarge": "RFDETRSegXLarge",
    "2xlarge": "RFDETRSeg2XLarge",
}


DEFAULT_MATCH_THRESHOLD = 0.52
DEFAULT_MIN_FACE_SCORE = 0.50
DEFAULT_MAX_SAMPLES_PER_PERSON = 20

BASE_WEIGHTS = {
    "face": 0.34,
    "body": 0.20,
    "clothing": 0.12,
    "proportions": 0.04,
    "position": 0.20,
    "gait": 0.05,
    "location_time": 0.05,
}


@dataclass(slots=True)
class PersonDetection:
    bbox: tuple[int, int, int, int]
    score: float
    track_id: str | None = None
    mask: "np.ndarray | None" = None


@dataclass(slots=True)
class FilteredDetection:
    detection: PersonDetection
    reason: str
    visual_quality: float
    area_ratio: float


@dataclass(slots=True)
class PersonObservation:
    camera_id: str
    bbox: tuple[int, int, int, int]
    frame_shape: tuple[int, int, int]
    body_embedding: "np.ndarray"
    clothing_features: "np.ndarray"
    body_proportions: "np.ndarray"
    center: tuple[float, float]
    timestamp: float
    detection_score: float
    visual_quality: float
    mask: "np.ndarray | None" = None
    face_embedding: "np.ndarray | None" = None
    face_bbox: tuple[int, int, int, int] | None = None
    face_quality: float = 0.0
    track_id: str | None = None


@dataclass(slots=True)
class DrawItem:
    observation: PersonObservation
    person_id: str
    breakdown: "MatchBreakdown"
    is_new: bool
    last_seen_frame: int


@dataclass(slots=True)
class MatchBreakdown:
    final_score: float
    track_score: float | None
    face_score: float | None
    body_score: float | None
    clothing_score: float | None
    proportions_score: float | None
    position_score: float | None
    gait_score: float | None
    location_time_score: float | None
    weights: dict[str, float]


@dataclass(slots=True)
class PersonProfile:
    person_id: str
    face_embeddings: list["np.ndarray"] = field(default_factory=list)
    body_embeddings: list["np.ndarray"] = field(default_factory=list)
    body_embedding_qualities: list[float] = field(default_factory=list)
    clothing_features: list["np.ndarray"] = field(default_factory=list)
    clothing_feature_qualities: list[float] = field(default_factory=list)
    body_proportions: list["np.ndarray"] = field(default_factory=list)
    movement_history: list[tuple[str, tuple[float, float], float]] = field(default_factory=list)
    bbox_history: list[tuple[str, tuple[int, int, int, int], float]] = field(default_factory=list)
    first_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    cameras: set[str] = field(default_factory=set)
    tracks: list[str] = field(default_factory=list)
    status: str = "tentative"
    observations: int = 0

    def update(self, observation: PersonObservation, max_samples: int, store_visual_sample: bool = True) -> None:
        self.last_seen = datetime.now(timezone.utc)
        self.cameras.add(observation.camera_id)
        self.observations += 1

        if observation.track_id is not None:
            self.tracks.append(observation.track_id)
            self.tracks = self.tracks[-max_samples:]

        if observation.face_embedding is not None and observation.face_quality >= 0.35:
            self.face_embeddings.append(observation.face_embedding)
            self.face_embeddings = self.face_embeddings[-max_samples:]

        if store_visual_sample:
            add_quality_sample(
                self.body_embeddings,
                self.body_embedding_qualities,
                observation.body_embedding,
                observation.visual_quality,
                max_samples,
            )
            add_quality_sample(
                self.clothing_features,
                self.clothing_feature_qualities,
                observation.clothing_features,
                observation.visual_quality,
                max_samples,
            )
            self.body_proportions.append(observation.body_proportions)

        self.movement_history.append((observation.camera_id, observation.center, observation.timestamp))
        self.bbox_history.append((observation.camera_id, observation.bbox, observation.timestamp))

        self.body_proportions = self.body_proportions[-max_samples:]
        self.movement_history = self.movement_history[-max_samples:]
        self.bbox_history = self.bbox_history[-max_samples:]


class PersonMemory:
    def __init__(
        self,
        match_threshold: float = DEFAULT_MATCH_THRESHOLD,
        max_samples_per_person: int = DEFAULT_MAX_SAMPLES_PER_PERSON,
        spatial_gate: bool = True,
        spatial_base_radius: float = 80.0,
        max_center_speed: float = 900.0,
        spatial_gate_seconds: float = 4.0,
        strong_face_threshold: float = 0.62,
        continuity_threshold: float = 0.78,
        track_memory_seconds: float = 30.0,
        new_track_match_threshold: float = 0.72,
        reentry_memory_seconds: float = 2.5,
        reentry_match_threshold: float = 0.58,
        reentry_position_threshold: float = 0.20,
        reentry_min_appearance_score: float = 0.55,
        long_reentry_memory_seconds: float = 600.0,
        long_reentry_match_threshold: float = 0.66,
        long_reentry_body_threshold: float = 0.62,
        long_reentry_clothing_threshold: float = 0.56,
        min_visual_sample_confidence: float = 0.55,
        min_confirmed_hits: int = 3,
    ) -> None:
        self.match_threshold = match_threshold
        self.max_samples_per_person = max_samples_per_person
        self.spatial_gate = spatial_gate
        self.spatial_base_radius = spatial_base_radius
        self.max_center_speed = max_center_speed
        self.spatial_gate_seconds = spatial_gate_seconds
        self.strong_face_threshold = strong_face_threshold
        self.continuity_threshold = continuity_threshold
        self.track_memory_seconds = track_memory_seconds
        self.new_track_match_threshold = new_track_match_threshold
        self.reentry_memory_seconds = reentry_memory_seconds
        self.reentry_match_threshold = reentry_match_threshold
        self.reentry_position_threshold = reentry_position_threshold
        self.reentry_min_appearance_score = reentry_min_appearance_score
        self.long_reentry_memory_seconds = long_reentry_memory_seconds
        self.long_reentry_match_threshold = long_reentry_match_threshold
        self.long_reentry_body_threshold = long_reentry_body_threshold
        self.long_reentry_clothing_threshold = long_reentry_clothing_threshold
        self.min_visual_sample_confidence = min_visual_sample_confidence
        self.min_confirmed_hits = min_confirmed_hits
        self._profiles: dict[str, PersonProfile] = {}
        self._track_to_person_id: dict[str, tuple[str, float]] = {}
        self._next_id = 1

    @property
    def profiles(self) -> Iterable[PersonProfile]:
        return self._profiles.values()

    def match_or_create(
        self,
        observation: PersonObservation,
        excluded_person_ids: set[str] | None = None,
    ) -> tuple[PersonProfile, MatchBreakdown, bool]:
        excluded_person_ids = excluded_person_ids or set()
        track_profile = self._profile_for_track(observation, excluded_person_ids)
        if track_profile is not None:
            breakdown = score_profile(observation, track_profile)
            if breakdown.track_score is None:
                breakdown.track_score = 1.0
            breakdown.final_score = max(breakdown.final_score, 1.0)
            track_profile.update(
                observation,
                self.max_samples_per_person,
                store_visual_sample=self._should_store_visual_sample(observation),
            )
            self._update_profile_status(track_profile)
            self._remember_track(observation, track_profile.person_id)
            return track_profile, breakdown, False

        profile, breakdown = self._best_match(observation, excluded_person_ids)

        if profile is not None and self._should_accept_match(observation, profile, breakdown):
            profile.update(
                observation,
                self.max_samples_per_person,
                store_visual_sample=self._should_store_visual_sample(observation),
            )
            self._update_profile_status(profile)
            self._remember_track(observation, profile.person_id)
            return profile, breakdown, False

        new_profile = self._create_profile()
        new_profile.update(observation, self.max_samples_per_person, store_visual_sample=True)
        self._update_profile_status(new_profile)
        self._remember_track(observation, new_profile.person_id)
        return new_profile, breakdown, True

    def _create_profile(self) -> PersonProfile:
        person_id = f"P-{self._next_id:04d}"
        self._next_id += 1
        profile = PersonProfile(person_id=person_id)
        self._profiles[person_id] = profile
        return profile

    def _best_match(
        self,
        observation: PersonObservation,
        excluded_person_ids: set[str],
    ) -> tuple[PersonProfile | None, MatchBreakdown]:
        best_profile: PersonProfile | None = None
        best_breakdown = empty_breakdown()

        for profile in self._profiles.values():
            if profile.person_id in excluded_person_ids:
                continue
            breakdown = score_profile(observation, profile)
            if not self._is_spatial_match_allowed(observation, profile, breakdown):
                continue
            if breakdown.final_score > best_breakdown.final_score:
                best_profile = profile
                best_breakdown = breakdown

        return best_profile, best_breakdown

    def _is_spatial_match_allowed(
        self,
        observation: PersonObservation,
        profile: PersonProfile,
        breakdown: MatchBreakdown,
    ) -> bool:
        if not self.spatial_gate:
            return True
        if breakdown.face_score is not None and breakdown.face_score >= self.strong_face_threshold:
            return True

        history = [
            (center, ts) for camera_id, center, ts in profile.movement_history if camera_id == observation.camera_id
        ]
        if not history:
            return True

        last_center, last_ts = history[-1]
        dt = max(0.0, observation.timestamp - last_ts)
        if dt > self.spatial_gate_seconds:
            return True

        allowed_radius = self._allowed_radius(observation, profile, dt)
        distance = float(np.linalg.norm(np.asarray(last_center) - np.asarray(observation.center)))
        return distance <= allowed_radius

    def _allowed_radius(self, observation: PersonObservation, profile: PersonProfile, dt: float) -> float:
        radius = self.spatial_base_radius + self.max_center_speed * dt
        last_bbox = last_camera_bbox(profile, observation.camera_id)
        if last_bbox is not None:
            radius += 0.35 * max(bbox_diagonal(last_bbox), bbox_diagonal(observation.bbox))
        return radius

    def _adaptive_threshold(self, observation: PersonObservation) -> float:
        threshold = self.match_threshold
        if observation.face_embedding is None:
            threshold -= 0.05
        if observation.detection_score < 0.55:
            threshold += 0.04
        return threshold

    def _should_accept_match(
        self,
        observation: PersonObservation,
        profile: PersonProfile,
        breakdown: MatchBreakdown,
    ) -> bool:
        if breakdown.face_score is not None and breakdown.face_score >= self.strong_face_threshold:
            return True
        if self._is_continuity_match(breakdown):
            return True
        if self._is_short_reentry_match(observation, profile, breakdown):
            return True
        if self._is_long_reentry_match(observation, profile, breakdown):
            return True
        if observation.track_id is not None:
            return breakdown.final_score >= self.new_track_match_threshold
        return breakdown.final_score >= self._adaptive_threshold(observation)

    def _should_store_visual_sample(self, observation: PersonObservation) -> bool:
        if observation.face_quality >= 0.55:
            return True
        return observation.visual_quality >= self.min_visual_sample_confidence

    def _update_profile_status(self, profile: PersonProfile) -> None:
        if profile.status == "tentative" and profile.observations >= self.min_confirmed_hits:
            profile.status = "confirmed"

    def _is_continuity_match(self, breakdown: MatchBreakdown) -> bool:
        return breakdown.position_score is not None and breakdown.position_score >= self.continuity_threshold

    def _is_short_reentry_match(
        self,
        observation: PersonObservation,
        profile: PersonProfile,
        breakdown: MatchBreakdown,
    ) -> bool:
        last_bbox_item = last_camera_bbox_item(profile, observation.camera_id)
        if last_bbox_item is None:
            return False

        last_bbox, last_ts = last_bbox_item
        dt = max(0.0, observation.timestamp - last_ts)
        if dt > self.reentry_memory_seconds:
            return False
        if breakdown.final_score < self.reentry_match_threshold:
            return False

        appearance_score = best_available_score(
            breakdown.face_score,
            breakdown.body_score,
            breakdown.clothing_score,
        )
        if appearance_score < self.reentry_min_appearance_score:
            return False

        position_ok = (
            breakdown.position_score is not None
            and breakdown.position_score >= self.reentry_position_threshold
        )
        edge_ok = (
            bbox_touches_frame_edge(last_bbox, observation.frame_shape)
            or bbox_touches_frame_edge(observation.bbox, observation.frame_shape)
        )
        return position_ok or edge_ok

    def _is_long_reentry_match(
        self,
        observation: PersonObservation,
        profile: PersonProfile,
        breakdown: MatchBreakdown,
    ) -> bool:
        if profile.status != "confirmed":
            return False
        if self.long_reentry_memory_seconds <= 0:
            return False

        last_seen_ts = last_camera_seen_timestamp(profile, observation.camera_id)
        if last_seen_ts is None:
            return False

        dt = max(0.0, observation.timestamp - last_seen_ts)
        if dt <= self.reentry_memory_seconds or dt > self.long_reentry_memory_seconds:
            return False
        if breakdown.final_score < self.long_reentry_match_threshold:
            return False
        if breakdown.face_score is not None and breakdown.face_score >= self.strong_face_threshold:
            return True
        return (
            breakdown.body_score is not None
            and breakdown.body_score >= self.long_reentry_body_threshold
            and breakdown.clothing_score is not None
            and breakdown.clothing_score >= self.long_reentry_clothing_threshold
        )

    def _profile_for_track(
        self,
        observation: PersonObservation,
        excluded_person_ids: set[str],
    ) -> PersonProfile | None:
        if observation.track_id is None:
            return None

        track_key = self._track_key(observation)
        match = self._track_to_person_id.get(track_key)
        if match is None:
            return None

        person_id, last_seen_ts = match
        if observation.timestamp - last_seen_ts > self.track_memory_seconds:
            self._track_to_person_id.pop(track_key, None)
            return None
        if person_id in excluded_person_ids:
            return None
        return self._profiles.get(person_id)

    def _remember_track(self, observation: PersonObservation, person_id: str) -> None:
        if observation.track_id is None:
            return
        self._track_to_person_id[self._track_key(observation)] = (person_id, observation.timestamp)

    def _track_key(self, observation: PersonObservation) -> str:
        return f"{observation.camera_id}:{observation.track_id}"


def load_runtime_dependencies() -> None:
    global FaceAnalysis, YOLO, cv2, np

    try:
        import cv2 as cv2_module
        import numpy as np_module
        from insightface.app import FaceAnalysis as FaceAnalysisClass
        from ultralytics import YOLO as YOLOClass
    except ModuleNotFoundError as exc:
        missing = exc.name or "runtime dependency"
        raise RuntimeError(
            f"Missing dependency '{missing}'. Install project dependencies before running video analysis."
        ) from exc

    cv2 = cv2_module
    np = np_module
    FaceAnalysis = FaceAnalysisClass
    YOLO = YOLOClass


def normalize_vector(vector: "np.ndarray") -> "np.ndarray":
    vector = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def top_k_cosine_score(query: "np.ndarray", candidates: list["np.ndarray"], k: int = 3) -> float | None:
    if not candidates:
        return None
    compatible_candidates = [candidate for candidate in candidates if candidate.shape == query.shape]
    if not compatible_candidates:
        return None
    matrix = np.vstack(compatible_candidates)
    scores = matrix @ query
    return float(np.mean(np.sort(scores)[-k:]))


def euclidean_similarity(query: "np.ndarray", candidates: list["np.ndarray"], scale: float) -> float | None:
    if not candidates:
        return None
    matrix = np.vstack(candidates)
    distances = np.linalg.norm(matrix - query, axis=1)
    return float(np.exp(-np.min(distances) / scale))


def add_quality_sample(
    samples: list["np.ndarray"],
    qualities: list[float],
    sample: "np.ndarray",
    quality: float,
    max_samples: int,
    near_duplicate_similarity: float = 0.985,
) -> None:
    if not samples:
        samples.append(sample)
        qualities.append(quality)
        return

    similarities = [float(existing @ sample) for existing in samples]
    nearest_index = int(np.argmax(similarities))
    if similarities[nearest_index] >= near_duplicate_similarity:
        if quality > qualities[nearest_index]:
            samples[nearest_index] = sample
            qualities[nearest_index] = quality
        return

    samples.append(sample)
    qualities.append(quality)
    if len(samples) > max_samples:
        remove_index = int(np.argmin(qualities))
        samples.pop(remove_index)
        qualities.pop(remove_index)


def empty_breakdown() -> MatchBreakdown:
    return MatchBreakdown(
        final_score=-1.0,
        track_score=None,
        face_score=None,
        body_score=None,
        clothing_score=None,
        proportions_score=None,
        position_score=None,
        gait_score=None,
        location_time_score=None,
        weights=BASE_WEIGHTS.copy(),
    )


def best_available_score(*scores: float | None) -> float:
    available_scores = [score for score in scores if score is not None]
    if not available_scores:
        return -1.0
    return max(available_scores)


def adaptive_weights(observation: PersonObservation) -> dict[str, float]:
    weights = BASE_WEIGHTS.copy()
    face_available = observation.face_embedding is not None

    if face_available and observation.face_quality >= 0.70:
        return weights

    if face_available and observation.face_quality >= 0.45:
        weights["face"] = 0.25
    else:
        weights["face"] = 0.0

    removed_face_weight = BASE_WEIGHTS["face"] - weights["face"]
    weights["body"] += removed_face_weight * 0.45
    weights["clothing"] += removed_face_weight * 0.30
    weights["position"] += removed_face_weight * 0.12
    weights["proportions"] += removed_face_weight * 0.08
    weights["gait"] += removed_face_weight * 0.05
    weights["location_time"] += removed_face_weight * 0.05
    return weights


def score_profile(observation: PersonObservation, profile: PersonProfile) -> MatchBreakdown:
    track_score = 1.0 if observation.track_id is not None and observation.track_id in profile.tracks else None
    face_score = None
    if observation.face_embedding is not None:
        face_score = top_k_cosine_score(observation.face_embedding, profile.face_embeddings)

    body_score = top_k_cosine_score(observation.body_embedding, profile.body_embeddings)
    clothing_score = top_k_cosine_score(observation.clothing_features, profile.clothing_features)
    proportions_score = euclidean_similarity(observation.body_proportions, profile.body_proportions, scale=0.30)
    position_score = position_continuity_similarity(observation, profile)
    gait_score = movement_similarity(observation, profile)
    location_time_score = location_time_similarity(observation, profile)

    weights = adaptive_weights(observation)
    score_map = {
        "face": face_score,
        "body": body_score,
        "clothing": clothing_score,
        "proportions": proportions_score,
        "position": position_score,
        "gait": gait_score,
        "location_time": location_time_score,
    }

    present_weight = sum(weights[name] for name, score in score_map.items() if score is not None)
    if present_weight == 0:
        final_score = -1.0
    else:
        final_score = sum(weights[name] * score for name, score in score_map.items() if score is not None)
        final_score /= present_weight

    return MatchBreakdown(
        final_score=float(final_score),
        track_score=track_score,
        face_score=face_score,
        body_score=body_score,
        clothing_score=clothing_score,
        proportions_score=proportions_score,
        position_score=position_score,
        gait_score=gait_score,
        location_time_score=location_time_score,
        weights=weights,
    )


def movement_similarity(observation: PersonObservation, profile: PersonProfile) -> float | None:
    same_camera_history = [
        (center, ts) for camera_id, center, ts in profile.movement_history if camera_id == observation.camera_id
    ]
    if len(same_camera_history) < 2:
        return None

    (prev_center, prev_ts), (last_center, last_ts) = same_camera_history[-2:]
    dt_profile = max(1e-3, last_ts - prev_ts)
    dt_current = max(1e-3, observation.timestamp - last_ts)
    profile_velocity = (
        (last_center[0] - prev_center[0]) / dt_profile,
        (last_center[1] - prev_center[1]) / dt_profile,
    )
    expected_center = (
        last_center[0] + profile_velocity[0] * dt_current,
        last_center[1] + profile_velocity[1] * dt_current,
    )
    distance = np.linalg.norm(np.asarray(expected_center) - np.asarray(observation.center))
    return float(np.exp(-distance / 180.0))


def position_continuity_similarity(observation: PersonObservation, profile: PersonProfile) -> float | None:
    last_bbox_item = last_camera_bbox_item(profile, observation.camera_id)
    if last_bbox_item is None:
        return None

    last_bbox, last_ts = last_bbox_item
    dt = max(0.0, observation.timestamp - last_ts)
    if dt > 3.0:
        return None

    last_center = bbox_center(last_bbox)
    distance = np.linalg.norm(np.asarray(last_center) - np.asarray(observation.center))
    scale = 35.0 + 420.0 * dt + 0.30 * max(bbox_diagonal(last_bbox), bbox_diagonal(observation.bbox))
    center_score = float(np.exp(-distance / max(1.0, scale)))
    overlap_score = bbox_iou(last_bbox, observation.bbox)
    time_score = float(np.exp(-dt / 2.0))
    return max(overlap_score, center_score) * time_score


def location_time_similarity(observation: PersonObservation, profile: PersonProfile) -> float | None:
    same_camera_history = [
        (center, ts) for camera_id, center, ts in profile.movement_history if camera_id == observation.camera_id
    ]
    if not same_camera_history:
        return None

    last_center, last_ts = same_camera_history[-1]
    dt = max(0.0, observation.timestamp - last_ts)
    distance = np.linalg.norm(np.asarray(last_center) - np.asarray(observation.center))
    max_reasonable_distance = 80.0 + 220.0 * min(dt, 2.0)
    return float(np.exp(-distance / max_reasonable_distance))


def parse_source(value: str) -> str | int:
    if value.isdigit():
        return int(value)
    return value


def parse_det_size(value: str) -> tuple[int, int]:
    parts = value.lower().replace("x", ",").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("det-size must look like 640x640")
    width, height = int(parts[0]), int(parts[1])
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("det-size values must be positive")
    return width, height


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return parsed


def build_face_app(det_size: tuple[int, int], providers: list[str]) -> "FaceAnalysis":
    app = FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(ctx_id=0, det_size=det_size)
    return app


def build_person_detector(model_name: str) -> "YOLO":
    return YOLO(model_name)


class RFDETRPersonDetector:
    def __init__(self, model_size: str, segmentation: bool) -> None:
        class_map = RFDETR_SEGMENTATION_CLASSES if segmentation else RFDETR_DETECTOR_CLASSES
        model_class_name = class_map.get(model_size)
        if model_class_name is None:
            available = ", ".join(sorted(class_map))
            raise RuntimeError(f"Unsupported RF-DETR model size '{model_size}'. Available sizes: {available}.")

        try:
            import rfdetr
            from rfdetr.assets.coco_classes import COCO_CLASSES
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "RF-DETR backend requires the 'rfdetr' package. Install it with: uv add rfdetr"
            ) from exc

        model_class = getattr(rfdetr, model_class_name, None)
        if model_class is None:
            raise RuntimeError(f"Installed rfdetr package does not expose {model_class_name}.")

        self.model = model_class()
        class_items = COCO_CLASSES.items() if hasattr(COCO_CLASSES, "items") else enumerate(COCO_CLASSES)
        self.person_class_ids = {
            int(class_id)
            for class_id, class_name in class_items
            if str(class_name).lower() == "person"
        }
        if not self.person_class_ids:
            self.person_class_ids = {0, 1}


class BodyReIDExtractor:
    def __init__(self, backend: str, model_name: str, device: str) -> None:
        self.backend = backend
        self.model_name = model_name
        self.device_name = device
        self._torch = None
        self._model = None
        self._device = None

        if backend in {"auto", "torchreid"}:
            try:
                self._init_torchreid()
                self.backend = "torchreid"
            except Exception as exc:
                if backend == "torchreid":
                    raise RuntimeError(
                        "TorchReID backend requested but could not be initialized. "
                        "Install torch and torchreid, or use --body-reid-backend handcrafted."
                    ) from exc
                print(f"Body ReID backend fallback: {exc}. Using handcrafted descriptors.")
                self.backend = "handcrafted"

    def extract(
        self,
        frame: "np.ndarray",
        bbox: tuple[int, int, int, int],
        mask: "np.ndarray | None" = None,
    ) -> "np.ndarray":
        if self.backend == "torchreid":
            return self._extract_torchreid(frame, bbox, mask)
        return extract_body_embedding(frame, bbox, mask)

    def _init_torchreid(self) -> None:
        import torch
        import torchreid

        if self.device_name == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(self.device_name)

        model = torchreid.models.build_model(
            name=self.model_name,
            num_classes=1000,
            pretrained=True,
        )
        model.eval()
        model.to(device)

        self._torch = torch
        self._device = device
        self._model = model

    def _extract_torchreid(
        self,
        frame: "np.ndarray",
        bbox: tuple[int, int, int, int],
        mask: "np.ndarray | None" = None,
    ) -> "np.ndarray":
        if self._torch is None or self._model is None or self._device is None:
            return extract_body_embedding(frame, bbox, mask)

        crop = crop_with_optional_mask(frame, bbox, mask)
        if crop.size == 0:
            return extract_body_embedding(frame, bbox, mask)

        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop = cv2.resize(crop, (128, 256), interpolation=cv2.INTER_AREA)
        tensor = self._torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0
        mean = self._torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = self._torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std
        tensor = tensor.unsqueeze(0).to(self._device)

        with self._torch.no_grad():
            embedding = self._model(tensor)

        embedding_np = embedding.detach().cpu().numpy().reshape(-1)
        return normalize_vector(embedding_np)


class DebugLogger:
    def __init__(self, path: str | None) -> None:
        self._file = None
        self._writer = None
        if path is not None:
            self._file = open(path, "w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(
                self._file,
                fieldnames=[
                    "frame",
                    "event",
                    "reason",
                    "person_id",
                    "person_status",
                    "track_id",
                    "bbox",
                    "det_score",
                    "visual_quality",
                    "area_ratio",
                    "match_score",
                    "face_score",
                    "body_score",
                    "clothing_score",
                    "position_score",
                    "is_new",
                    "drawn",
                ],
            )
            self._writer.writeheader()

    def log_detection_drop(self, frame_index: int, dropped: FilteredDetection) -> None:
        if self._writer is None:
            return
        self._writer.writerow(
            {
                "frame": frame_index,
                "event": "drop_detection",
                "reason": dropped.reason,
                "person_id": "",
                "person_status": "",
                "track_id": dropped.detection.track_id or "",
                "bbox": format_bbox(dropped.detection.bbox),
                "det_score": f"{dropped.detection.score:.4f}",
                "visual_quality": f"{dropped.visual_quality:.4f}",
                "area_ratio": f"{dropped.area_ratio:.6f}",
                "match_score": "",
                "face_score": "",
                "body_score": "",
                "clothing_score": "",
                "position_score": "",
                "is_new": "",
                "drawn": "",
            }
        )

    def log_match(
        self,
        frame_index: int,
        observation: PersonObservation,
        profile: PersonProfile,
        breakdown: MatchBreakdown,
        is_new: bool,
        drawn: bool,
    ) -> None:
        if self._writer is None:
            return
        self._writer.writerow(
            {
                "frame": frame_index,
                "event": "match",
                "reason": "matched_or_created",
                "person_id": profile.person_id,
                "person_status": profile.status,
                "track_id": observation.track_id or "",
                "bbox": format_bbox(observation.bbox),
                "det_score": f"{observation.detection_score:.4f}",
                "visual_quality": f"{observation.visual_quality:.4f}",
                "area_ratio": "",
                "match_score": f"{breakdown.final_score:.4f}",
                "face_score": format_optional_score(breakdown.face_score),
                "body_score": format_optional_score(breakdown.body_score),
                "clothing_score": format_optional_score(breakdown.clothing_score),
                "position_score": format_optional_score(breakdown.position_score),
                "is_new": int(is_new),
                "drawn": int(drawn),
            }
        )

    def close(self) -> None:
        if self._file is not None:
            self._file.close()


def format_bbox(bbox: tuple[int, int, int, int]) -> str:
    return ",".join(str(value) for value in bbox)


def format_optional_score(score: float | None) -> str:
    return "" if score is None else f"{score:.4f}"


def open_capture(source: str | int) -> "cv2.VideoCapture":
    capture = cv2.VideoCapture(source)
    if isinstance(source, str) and source.startswith("rtsp://"):
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def build_video_writer(
    output_path: str,
    capture: "cv2.VideoCapture",
    frame_shape: tuple[int, int, int],
    fallback_fps: float,
) -> "cv2.VideoWriter":
    frame_h, frame_w = frame_shape[:2]
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 240:
        fps = fallback_fps

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {output_path}")
    return writer


def clip_bbox(bbox: tuple[int, int, int, int], frame_shape: tuple[int, int, int]) -> tuple[int, int, int, int]:
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    return (
        int(np.clip(x1, 0, w - 1)),
        int(np.clip(y1, 0, h - 1)),
        int(np.clip(x2, 0, w - 1)),
        int(np.clip(y2, 0, h - 1)),
    )


def bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bbox_diagonal(bbox: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = bbox
    return float(np.hypot(max(0, x2 - x1), max(0, y2 - y1)))


def bbox_iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    first_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    second_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = first_area + second_area - intersection
    if union <= 0:
        return 0.0
    return float(intersection / union)


def bbox_min_overlap(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    first_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    second_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    min_area = min(first_area, second_area)
    if min_area <= 0:
        return 0.0
    return float(intersection / min_area)


def bbox_touches_frame_edge(
    bbox: tuple[int, int, int, int],
    frame_shape: tuple[int, int, int],
    margin_ratio: float = 0.06,
) -> bool:
    frame_h, frame_w = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    margin_x = frame_w * margin_ratio
    margin_y = frame_h * margin_ratio
    return x1 <= margin_x or y1 <= margin_y or x2 >= frame_w - margin_x or y2 >= frame_h - margin_y


def last_camera_bbox(profile: PersonProfile, camera_id: str) -> tuple[int, int, int, int] | None:
    for bbox_camera_id, bbox, _timestamp in reversed(profile.bbox_history):
        if bbox_camera_id == camera_id:
            return bbox
    return None


def last_camera_bbox_item(profile: PersonProfile, camera_id: str) -> tuple[tuple[int, int, int, int], float] | None:
    for bbox_camera_id, bbox, timestamp in reversed(profile.bbox_history):
        if bbox_camera_id == camera_id:
            return bbox, timestamp
    return None


def last_camera_seen_timestamp(profile: PersonProfile, camera_id: str) -> float | None:
    for history_camera_id, _center, timestamp in reversed(profile.movement_history):
        if history_camera_id == camera_id:
            return timestamp
    return None


def bbox_contains_point(bbox: tuple[int, int, int, int], point: tuple[float, float]) -> bool:
    x1, y1, x2, y2 = bbox
    x, y = point
    return x1 <= x <= x2 and y1 <= y <= y2


def estimate_face_quality(
    bbox: tuple[int, int, int, int],
    det_score: float,
    person_bbox: tuple[int, int, int, int],
) -> float:
    x1, y1, x2, y2 = bbox
    face_w = max(0, x2 - x1)
    face_h = max(0, y2 - y1)
    px1, py1, px2, py2 = person_bbox
    person_h = max(1, py2 - py1)
    min_side_score = min(1.0, min(face_w, face_h) / 80.0)
    relative_head_score = min(1.0, face_h / person_h * 4.0)
    quality = 0.55 * det_score + 0.30 * min_side_score + 0.15 * relative_head_score
    return float(np.clip(quality, 0.0, 1.0))


def extract_histogram(
    crop: "np.ndarray",
    bins: tuple[int, int, int],
    color_space: int | None = None,
    ranges: tuple[int, int, int, int, int, int] = (0, 180, 0, 256, 0, 256),
) -> "np.ndarray":
    if crop.size == 0:
        return np.zeros(bins[0] * bins[1] * bins[2], dtype=np.float32)
    if color_space is None:
        color_space = cv2.COLOR_BGR2HSV
    converted = cv2.cvtColor(crop, color_space)
    hist = cv2.calcHist([converted], [0, 1, 2], None, bins, list(ranges))
    return normalize_vector(hist.flatten())


def extract_gradient_features(crop: "np.ndarray", bins: int = 9) -> "np.ndarray":
    if crop.size == 0:
        return np.zeros(bins + 1, dtype=np.float32)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 128), interpolation=cv2.INTER_AREA)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude, angle = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=True)
    angle = np.mod(angle, 180.0)
    hist, _ = np.histogram(angle, bins=bins, range=(0.0, 180.0), weights=magnitude)
    edge_density = np.asarray([np.mean(magnitude > 24.0)], dtype=np.float32)
    return normalize_vector(np.concatenate([hist.astype(np.float32), edge_density]))


def extract_spatial_color_features(crop: "np.ndarray") -> "np.ndarray":
    if crop.size == 0:
        return np.zeros(192, dtype=np.float32)

    resized = cv2.resize(crop, (48, 96), interpolation=cv2.INTER_AREA)
    rows = np.array_split(resized, 3, axis=0)
    features = []
    for row in rows:
        hsv_hist = extract_histogram(row, bins=(8, 2, 2))
        lab_hist = extract_histogram(
            row,
            bins=(4, 4, 2),
            color_space=cv2.COLOR_BGR2LAB,
            ranges=(0, 256, 0, 256, 0, 256),
        )
        features.extend([hsv_hist * 0.65, lab_hist * 0.35])
    return normalize_vector(np.concatenate(features))


def crop_with_optional_mask(
    frame: "np.ndarray",
    bbox: tuple[int, int, int, int],
    mask: "np.ndarray | None" = None,
) -> "np.ndarray":
    x1, y1, x2, y2 = bbox
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0 or mask is None:
        return crop

    mask_crop = mask[y1:y2, x1:x2]
    if mask_crop.shape[:2] != crop.shape[:2]:
        mask_crop = cv2.resize(mask_crop.astype(np.uint8), (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask_crop = mask_crop.astype(bool)
    if not np.any(mask_crop):
        return crop

    masked_crop = np.zeros_like(crop)
    masked_crop[mask_crop] = crop[mask_crop]
    return masked_crop


def extract_body_embedding(
    frame: "np.ndarray",
    bbox: tuple[int, int, int, int],
    mask: "np.ndarray | None" = None,
) -> "np.ndarray":
    crop = crop_with_optional_mask(frame, bbox, mask)
    color = extract_spatial_color_features(crop)
    gradient = extract_gradient_features(crop)
    return normalize_vector(np.concatenate([color * 0.75, gradient * 0.25]))


def extract_clothing_features(
    frame: "np.ndarray",
    bbox: tuple[int, int, int, int],
    mask: "np.ndarray | None" = None,
) -> "np.ndarray":
    crop = crop_with_optional_mask(frame, bbox, mask)
    height = max(1, crop.shape[0])
    upper = crop[int(height * 0.18) : int(height * 0.55), :]
    lower = crop[int(height * 0.55) :, :]
    upper_hsv = extract_histogram(upper, bins=(16, 4, 4))
    lower_hsv = extract_histogram(lower, bins=(16, 4, 4))
    upper_lab = extract_histogram(
        upper,
        bins=(8, 4, 4),
        color_space=cv2.COLOR_BGR2LAB,
        ranges=(0, 256, 0, 256, 0, 256),
    )
    lower_lab = extract_histogram(
        lower,
        bins=(8, 4, 4),
        color_space=cv2.COLOR_BGR2LAB,
        ranges=(0, 256, 0, 256, 0, 256),
    )
    return normalize_vector(
        np.concatenate([
            upper_hsv * 0.42,
            lower_hsv * 0.28,
            upper_lab * 0.18,
            lower_lab * 0.12,
        ])
    )


def extract_body_proportions(
    bbox: tuple[int, int, int, int],
    frame_shape: tuple[int, int, int],
) -> "np.ndarray":
    x1, y1, x2, y2 = bbox
    frame_h, frame_w = frame_shape[:2]
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    center_x, center_y = bbox_center(bbox)
    area = width * height
    return np.asarray(
        [
            width / height,
            height / max(1, frame_h),
            width / max(1, frame_w),
            area / max(1, frame_w * frame_h),
            center_x / max(1, frame_w),
            center_y / max(1, frame_h),
        ],
        dtype=np.float32,
    )


def estimate_visual_quality(
    frame: "np.ndarray",
    bbox: tuple[int, int, int, int],
    detection_score: float,
    mask: "np.ndarray | None" = None,
) -> float:
    x1, y1, x2, y2 = bbox
    frame_h, frame_w = frame.shape[:2]
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    crop = crop_with_optional_mask(frame, bbox, mask)

    size_score = min(1.0, height / max(1, frame_h) * 3.2)
    area_score = min(1.0, (width * height) / max(1, frame_w * frame_h) * 18.0)
    if mask is not None and np.any(mask):
        mask_area_ratio = float(np.count_nonzero(mask) / max(1, frame_w * frame_h))
        area_score = max(area_score * 0.45, min(1.0, mask_area_ratio * 28.0))
    aspect_ratio = width / height
    aspect_score = float(np.exp(-abs(aspect_ratio - 0.42) / 0.55))

    if crop.size == 0:
        blur_score = 0.0
    else:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur_score = min(1.0, cv2.Laplacian(gray, cv2.CV_64F).var() / 180.0)

    quality = (
        0.35 * detection_score
        + 0.25 * size_score
        + 0.15 * area_score
        + 0.15 * blur_score
        + 0.10 * aspect_score
    )
    return float(np.clip(quality, 0.0, 1.0))


def detect_faces(
    app: "FaceAnalysis",
    frame: "np.ndarray",
    min_face_score: float,
) -> list[dict[str, object]]:
    detected_faces: list[dict[str, object]] = []
    for face in app.get(frame):
        det_score = float(getattr(face, "det_score", 0.0))
        if det_score < min_face_score:
            continue

        bbox = clip_bbox(tuple(np.asarray(face.bbox, dtype=np.int32).tolist()), frame.shape)
        raw_embedding = getattr(face, "normed_embedding", None)
        if raw_embedding is None:
            raw_embedding = getattr(face, "embedding", None)
        if raw_embedding is None:
            continue

        detected_faces.append(
            {
                "bbox": bbox,
                "center": bbox_center(bbox),
                "det_score": det_score,
                "embedding": normalize_vector(raw_embedding),
            }
        )
    return detected_faces


def detect_people(
    detector: "YOLO",
    frame: "np.ndarray",
    confidence: float,
    image_size: int,
    device: str | None,
    max_people: int,
) -> list[PersonDetection]:
    result = detector.predict(
        frame,
        classes=[0],
        conf=confidence,
        imgsz=image_size,
        device=device,
        verbose=False,
    )[0]
    people: list[PersonDetection] = []
    if result.boxes is None:
        return people

    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    masks = result_masks(result, frame.shape, len(boxes))
    for box, score, mask in sorted(zip(boxes, scores, masks, strict=False), key=lambda item: float(item[1]), reverse=True):
        bbox = clip_bbox(tuple(box.astype(np.int32).tolist()), frame.shape)
        x1, y1, x2, y2 = bbox
        if x2 - x1 < 20 or y2 - y1 < 40:
            continue
        people.append(PersonDetection(bbox=bbox, score=float(score), mask=mask))
        if max_people > 0 and len(people) >= max_people:
            break
    return people


def track_people(
    detector: "YOLO",
    frame: "np.ndarray",
    confidence: float,
    image_size: int,
    device: str | None,
    max_people: int,
    tracker_config: str,
) -> list[PersonDetection]:
    result = detector.track(
        frame,
        classes=[0],
        conf=confidence,
        imgsz=image_size,
        device=device,
        tracker=tracker_config,
        persist=True,
        verbose=False,
    )[0]
    people: list[PersonDetection] = []
    if result.boxes is None:
        return people

    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    track_ids = result.boxes.id
    ids = track_ids.cpu().numpy() if track_ids is not None else [None] * len(boxes)
    masks = result_masks(result, frame.shape, len(boxes))

    detections = sorted(
        zip(boxes, scores, ids, masks, strict=False),
        key=lambda item: float(item[1]),
        reverse=True,
    )
    for box, score, raw_track_id, mask in detections:
        bbox = clip_bbox(tuple(box.astype(np.int32).tolist()), frame.shape)
        x1, y1, x2, y2 = bbox
        if x2 - x1 < 20 or y2 - y1 < 40:
            continue
        track_id = None if raw_track_id is None else str(int(raw_track_id))
        people.append(PersonDetection(bbox=bbox, score=float(score), track_id=track_id, mask=mask))
        if max_people > 0 and len(people) >= max_people:
            break
    return people


def detect_people_rfdetr(
    detector: RFDETRPersonDetector,
    frame: "np.ndarray",
    confidence: float,
    max_people: int,
) -> list[PersonDetection]:
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detections = detector.model.predict(rgb_frame, threshold=confidence)
    xyxy = np.asarray(getattr(detections, "xyxy", []), dtype=np.float32)
    if xyxy.size == 0:
        return []

    scores = np.asarray(getattr(detections, "confidence", np.ones(len(xyxy))), dtype=np.float32)
    class_ids = getattr(detections, "class_id", None)
    if class_ids is None:
        class_ids = np.zeros(len(xyxy), dtype=np.int32)
    class_ids = np.asarray(class_ids, dtype=np.int32)

    masks = getattr(detections, "mask", None)
    if masks is not None:
        masks = np.asarray(masks)

    candidates: list[tuple["np.ndarray", float, "np.ndarray | None"]] = []
    for index, (box, score, class_id) in enumerate(zip(xyxy, scores, class_ids, strict=False)):
        if int(class_id) not in detector.person_class_ids:
            continue
        if float(score) < confidence:
            continue
        mask = None
        if masks is not None and index < len(masks):
            mask = normalize_detection_mask(masks[index], frame.shape)
        candidates.append((box, float(score), mask))

    candidates.sort(key=lambda item: item[1], reverse=True)
    people: list[PersonDetection] = []
    for box, score, mask in candidates:
        bbox = clip_bbox(tuple(box.astype(np.int32).tolist()), frame.shape)
        x1, y1, x2, y2 = bbox
        if x2 - x1 < 20 or y2 - y1 < 40:
            continue
        people.append(PersonDetection(bbox=bbox, score=score, mask=mask))
        if max_people > 0 and len(people) >= max_people:
            break
    return people


def normalize_detection_mask(mask: "np.ndarray", frame_shape: tuple[int, int, int]) -> "np.ndarray | None":
    frame_h, frame_w = frame_shape[:2]
    if mask.size == 0:
        return None
    normalized = mask > 0.5
    if normalized.shape != (frame_h, frame_w):
        normalized = cv2.resize(
            normalized.astype(np.uint8),
            (frame_w, frame_h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    return normalized


def result_masks(result: object, frame_shape: tuple[int, int, int], count: int) -> list["np.ndarray | None"]:
    masks_obj = getattr(result, "masks", None)
    if masks_obj is None or getattr(masks_obj, "data", None) is None:
        return [None] * count

    frame_h, frame_w = frame_shape[:2]
    raw_masks = masks_obj.data.cpu().numpy()
    masks: list["np.ndarray | None"] = []
    for raw_mask in raw_masks[:count]:
        mask = raw_mask > 0.5
        if mask.shape != (frame_h, frame_w):
            mask = cv2.resize(mask.astype(np.uint8), (frame_w, frame_h), interpolation=cv2.INTER_NEAREST).astype(bool)
        masks.append(mask)

    while len(masks) < count:
        masks.append(None)
    return masks


def filter_person_detections(
    frame: "np.ndarray",
    detections: list[PersonDetection],
    min_visual_quality: float,
    min_area_ratio: float,
    nms_iou: float,
    nms_min_overlap: float,
    max_people: int,
) -> tuple[list[PersonDetection], list[FilteredDetection]]:
    frame_h, frame_w = frame.shape[:2]
    filtered: list[tuple[PersonDetection, float, float]] = []
    dropped: list[FilteredDetection] = []

    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        area_ratio = max(0, x2 - x1) * max(0, y2 - y1) / max(1, frame_w * frame_h)
        quality = estimate_visual_quality(frame, detection.bbox, detection.score, detection.mask)
        if area_ratio < min_area_ratio:
            dropped.append(FilteredDetection(detection, "area_too_small", quality, area_ratio))
            continue
        if quality < min_visual_quality:
            dropped.append(FilteredDetection(detection, "visual_quality_too_low", quality, area_ratio))
            continue
        filtered.append((detection, quality, area_ratio))

    filtered.sort(key=lambda item: (item[0].score, item[1], item[2]), reverse=True)
    kept: list[PersonDetection] = []
    for detection, _quality, _area_ratio in filtered:
        if any(
            bbox_iou(detection.bbox, kept_detection.bbox) >= nms_iou
            or bbox_min_overlap(detection.bbox, kept_detection.bbox) >= nms_min_overlap
            for kept_detection in kept
        ):
            dropped.append(FilteredDetection(detection, "duplicate_overlap", _quality, _area_ratio))
            continue
        kept.append(detection)
        if max_people > 0 and len(kept) >= max_people:
            break
    return kept, dropped


def assign_face_to_person(
    person_bbox: tuple[int, int, int, int],
    faces: list[dict[str, object]],
) -> dict[str, object] | None:
    candidates = [face for face in faces if bbox_contains_point(person_bbox, face["center"])]
    if not candidates:
        return None

    def face_area(face: dict[str, object]) -> int:
        x1, y1, x2, y2 = face["bbox"]
        return max(0, x2 - x1) * max(0, y2 - y1)

    return max(candidates, key=face_area)


def build_observations(
    frame: "np.ndarray",
    camera_id: str,
    person_detections: list[PersonDetection],
    faces: list[dict[str, object]],
    timestamp: float,
    body_reid: BodyReIDExtractor,
) -> list[PersonObservation]:
    observations: list[PersonObservation] = []

    for detection in person_detections:
        person_bbox = detection.bbox
        assigned_face = assign_face_to_person(person_bbox, faces)
        face_embedding = None
        face_bbox = None
        face_quality = 0.0

        if assigned_face is not None:
            face_embedding = assigned_face["embedding"]
            face_bbox = assigned_face["bbox"]
            face_quality = estimate_face_quality(face_bbox, assigned_face["det_score"], person_bbox)

        observations.append(
            PersonObservation(
                camera_id=camera_id,
                bbox=person_bbox,
                frame_shape=frame.shape,
                body_embedding=body_reid.extract(frame, person_bbox, detection.mask),
                clothing_features=extract_clothing_features(frame, person_bbox, detection.mask),
                body_proportions=extract_body_proportions(person_bbox, frame.shape),
                center=bbox_center(person_bbox),
                timestamp=timestamp,
                detection_score=detection.score,
                visual_quality=estimate_visual_quality(frame, person_bbox, detection.score, detection.mask),
                mask=detection.mask,
                face_embedding=face_embedding,
                face_bbox=face_bbox,
                face_quality=face_quality,
                track_id=detection.track_id,
            )
        )

    return observations


def draw_person(
    frame: "np.ndarray",
    observation: PersonObservation,
    person_id: str,
    breakdown: MatchBreakdown,
    is_new: bool,
) -> None:
    x1, y1, x2, y2 = observation.bbox
    color = (65, 205, 245) if not is_new else (55, 175, 255)
    fill_alpha = 0.18 if not is_new else 0.22
    border_alpha = 0.72

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, fill_alpha, frame, 1.0 - fill_alpha, 0, frame)

    border_layer = frame.copy()
    cv2.rectangle(border_layer, (x1, y1), (x2, y2), color, 2)
    cv2.addWeighted(border_layer, border_alpha, frame, 1.0 - border_alpha, 0, frame)

    label = person_id.removeprefix("P-")
    label_scale = 0.56
    label_thickness = 2
    label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, label_scale, label_thickness)
    label_w, label_h = label_size
    pad_x = 8
    pad_y = 5
    label_x1 = x1
    label_y1 = max(0, y1 - label_h - baseline - pad_y * 2)
    label_x2 = min(frame.shape[1] - 1, label_x1 + label_w + pad_x * 2)
    label_y2 = min(frame.shape[0] - 1, label_y1 + label_h + baseline + pad_y * 2)

    label_layer = frame.copy()
    cv2.rectangle(label_layer, (label_x1, label_y1), (label_x2, label_y2), color, -1)
    cv2.addWeighted(label_layer, 0.72, frame, 0.28, 0, frame)
    cv2.putText(
        frame,
        label,
        (label_x1 + pad_x, label_y1 + pad_y + label_h),
        cv2.FONT_HERSHEY_SIMPLEX,
        label_scale,
        (255, 245, 210),
        label_thickness,
        cv2.LINE_AA,
    )


def should_drop_draw_item(
    item: DrawItem,
    current_frame: int,
    hold_missing_frames: int,
    hold_overlap_iou: float,
    hold_min_overlap: float,
    live_bboxes: list[tuple[int, int, int, int]],
    live_person_ids: set[str],
) -> bool:
    frames_missing = current_frame - item.last_seen_frame
    if frames_missing <= 0:
        return False
    if frames_missing > hold_missing_frames:
        return True
    if item.is_new:
        return True
    if item.person_id in live_person_ids:
        return False
    return any(
        bbox_iou(item.observation.bbox, live_bbox) >= hold_overlap_iou
        or bbox_min_overlap(item.observation.bbox, live_bbox) >= hold_min_overlap
        for live_bbox in live_bboxes
    )


def format_duration(seconds: float) -> str:
    if seconds < 0 or seconds == float("inf"):
        return "--:--"
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def report_progress(
    frame_index: int,
    total_frames: int,
    started_at: float,
    force_newline: bool = False,
) -> None:
    elapsed = max(time.monotonic() - started_at, 1e-6)
    processing_fps = frame_index / elapsed
    if total_frames > 0:
        percent = min(100.0, frame_index / total_frames * 100.0)
        remaining_frames = max(total_frames - frame_index, 0)
        eta = remaining_frames / processing_fps if processing_fps > 0 else float("inf")
        message = (
            f"\rProgress: {frame_index}/{total_frames} frames "
            f"({percent:5.1f}%) | {processing_fps:5.1f} fps | ETA {format_duration(eta)}"
        )
    else:
        message = f"\rProgress: {frame_index} frames | {processing_fps:5.1f} fps"

    sys.stderr.write(message)
    if force_newline:
        sys.stderr.write("\n")
    sys.stderr.flush()


def run_realtime(args: argparse.Namespace) -> int:
    load_runtime_dependencies()
    cv2.setUseOptimized(True)
    if args.opencv_threads > 0:
        cv2.setNumThreads(args.opencv_threads)
    if not args.display and not args.output:
        raise RuntimeError("--no-display requires --output so processed frames are not discarded.")

    stop = False

    def handle_signal(_signum: int, _frame: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    source = parse_source(args.source)
    face_app = build_face_app(args.det_size, args.providers)
    if args.detector_backend == "rfdetr":
        person_detector = RFDETRPersonDetector(args.rfdetr_model_size, args.rfdetr_segmentation)
    else:
        person_detector = build_person_detector(args.yolo_model)
    body_reid = BodyReIDExtractor(
        backend=args.body_reid_backend,
        model_name=args.body_reid_model,
        device=args.body_reid_device,
    )
    debug_logger = DebugLogger(args.debug_csv)
    memory = PersonMemory(
        match_threshold=args.match_threshold,
        max_samples_per_person=args.max_samples_per_person,
        spatial_gate=args.spatial_gate,
        spatial_base_radius=args.spatial_base_radius,
        max_center_speed=args.max_center_speed,
        spatial_gate_seconds=args.spatial_gate_seconds,
        strong_face_threshold=args.strong_face_threshold,
        continuity_threshold=args.continuity_threshold,
        track_memory_seconds=args.track_memory_seconds,
        new_track_match_threshold=args.new_track_match_threshold,
        reentry_memory_seconds=args.reentry_memory_seconds,
        reentry_match_threshold=args.reentry_match_threshold,
        reentry_position_threshold=args.reentry_position_threshold,
        reentry_min_appearance_score=args.reentry_min_appearance_score,
        long_reentry_memory_seconds=args.long_reentry_memory_seconds,
        long_reentry_match_threshold=args.long_reentry_match_threshold,
        long_reentry_body_threshold=args.long_reentry_body_threshold,
        long_reentry_clothing_threshold=args.long_reentry_clothing_threshold,
        min_visual_sample_confidence=args.min_visual_sample_confidence,
        min_confirmed_hits=args.min_confirmed_hits,
    )

    capture = open_capture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video source: {args.source}")

    window_name = f"KazinoMonitoring MVP - {args.camera_id}"
    if args.display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_count = 0
    source_frame_index = 0
    fps = 0.0
    fps_started_at = time.monotonic()
    draw_buffer: dict[str, DrawItem] = {}
    writer = None
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0 or source_fps > 240:
        source_fps = args.output_fps
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 0 or isinstance(source, str) and source.startswith("rtsp://"):
        total_frames = 0
    progress_started_at = time.monotonic()
    progress_last_reported_at = 0.0

    try:
        while not stop:
            ok, frame = capture.read()
            if not ok:
                if isinstance(source, str) and source.startswith("rtsp://"):
                    capture.release()
                    time.sleep(args.reconnect_delay)
                    capture = open_capture(source)
                    continue
                break

            if writer is None and args.output:
                writer = build_video_writer(args.output, capture, frame.shape, args.output_fps)

            should_process = source_frame_index % args.process_every_n_frames == 0
            live_bboxes: list[tuple[int, int, int, int]] = []
            live_person_ids: set[str] = set()
            if should_process:
                if args.detector_backend == "rfdetr":
                    people = detect_people_rfdetr(
                        person_detector,
                        frame,
                        args.person_confidence,
                        args.max_people,
                    )
                elif args.tracker == "off":
                    people = detect_people(
                        person_detector,
                        frame,
                        args.person_confidence,
                        args.yolo_imgsz,
                        args.yolo_device,
                        args.max_people,
                    )
                else:
                    people = track_people(
                        person_detector,
                        frame,
                        args.person_confidence,
                        args.yolo_imgsz,
                        args.yolo_device,
                        args.max_people,
                        args.tracker,
                    )
                people, dropped_detections = filter_person_detections(
                    frame,
                    people,
                    args.min_live_visual_quality,
                    args.min_live_area_ratio,
                    args.live_nms_iou,
                    args.live_nms_min_overlap,
                    args.max_people,
                )
                for dropped in dropped_detections:
                    debug_logger.log_detection_drop(source_frame_index, dropped)
                should_run_faces = args.face_every_n_frames > 0 and source_frame_index % args.face_every_n_frames == 0
                faces = detect_faces(face_app, frame, args.min_face_score) if should_run_faces else []
                frame_timestamp = source_frame_index / source_fps
                observations = build_observations(frame, args.camera_id, people, faces, frame_timestamp, body_reid)

                assigned_person_ids: set[str] = set()
                for observation in observations:
                    profile, breakdown, is_new = memory.match_or_create(observation, assigned_person_ids)
                    assigned_person_ids.add(profile.person_id)
                    live_person_ids.add(profile.person_id)
                    live_bboxes.append(observation.bbox)
                    drawn = args.draw_tentative or profile.status == "confirmed"
                    debug_logger.log_match(source_frame_index, observation, profile, breakdown, is_new, drawn)
                    if drawn:
                        draw_buffer[profile.person_id] = DrawItem(
                            observation=observation,
                            person_id=profile.person_id,
                            breakdown=breakdown,
                            is_new=is_new,
                            last_seen_frame=source_frame_index,
                        )

            expired_person_ids = [
                person_id
                for person_id, item in draw_buffer.items()
                if should_drop_draw_item(
                    item,
                    source_frame_index,
                    args.hold_missing_frames,
                    args.hold_overlap_iou,
                    args.hold_min_overlap,
                    live_bboxes,
                    live_person_ids,
                )
            ]
            for person_id in expired_person_ids:
                draw_buffer.pop(person_id, None)

            for item in draw_buffer.values():
                draw_person(frame, item.observation, item.person_id, item.breakdown, item.is_new)

            frame_count += 1
            source_frame_index += 1
            now = time.monotonic()
            elapsed = now - fps_started_at
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_started_at = now

            if writer is not None:
                writer.write(frame)
            if args.display:
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            if args.progress:
                progress_now = time.monotonic()
                is_final_frame = total_frames > 0 and source_frame_index >= total_frames
                if is_final_frame or progress_now - progress_last_reported_at >= args.progress_interval:
                    report_progress(source_frame_index, total_frames, progress_started_at, is_final_frame)
                    progress_last_reported_at = progress_now
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        debug_logger.close()
        if args.progress and (total_frames == 0 or source_frame_index < total_frames):
            report_progress(source_frame_index, total_frames, progress_started_at, force_newline=True)
        if args.display:
            cv2.destroyAllWindows()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Realtime casino visitor multi-modal Person ID MVP."
    )
    parser.add_argument(
        "--source",
        required=True,
        help="RTSP URL, video file path, or local camera index such as 0.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output video path for annotated frames, for example result.mp4.",
    )
    parser.add_argument(
        "--output-fps",
        type=float,
        default=25.0,
        help="Fallback output FPS when the source does not report a valid FPS.",
    )
    parser.add_argument(
        "--display",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show the realtime OpenCV window; use --no-display for file-only processing.",
    )
    parser.add_argument(
        "--debug-csv",
        default=None,
        help="Optional CSV path with per-frame detection, filtering, and matching diagnostics.",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print processing progress to stderr; use --no-progress to disable it.",
    )
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=1.0,
        help="Seconds between terminal progress updates.",
    )
    parser.add_argument("--camera-id", default="cam-001", help="Stable logical camera identifier.")
    parser.add_argument(
        "--detector-backend",
        choices=["yolo", "rfdetr"],
        default="yolo",
        help="Person detector backend. RF-DETR is experimental and currently runs without Ultralytics tracking IDs.",
    )
    parser.add_argument(
        "--yolo-model",
        default="yolov8n.pt",
        help="Ultralytics YOLO model for person detection.",
    )
    parser.add_argument(
        "--yolo-imgsz",
        type=positive_int,
        default=640,
        help="YOLO inference image size.",
    )
    parser.add_argument(
        "--yolo-device",
        default=None,
        help="Ultralytics device, for example mps on Apple Silicon, cpu, or cuda:0.",
    )
    parser.add_argument(
        "--rfdetr-model-size",
        choices=["nano", "small", "medium", "large", "xlarge", "2xlarge"],
        default="medium",
        help="RF-DETR model size. Detection supports nano/small/medium/large; segmentation also supports xlarge/2xlarge.",
    )
    parser.add_argument(
        "--rfdetr-segmentation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use RF-DETR segmentation models and pass masks into ReID/clothing extraction.",
    )
    parser.add_argument(
        "--person-confidence",
        type=float,
        default=0.25,
        help="Minimum detector confidence for person detections.",
    )
    parser.add_argument(
        "--tracker",
        default="trackers/casino_botsort.yaml",
        help="Ultralytics tracker config path, botsort.yaml, bytetrack.yaml, or off.",
    )
    parser.add_argument(
        "--body-reid-backend",
        choices=["auto", "handcrafted", "torchreid"],
        default="auto",
        help="Body ReID backend. auto tries TorchReID OSNet and falls back to handcrafted descriptors.",
    )
    parser.add_argument(
        "--body-reid-model",
        default="osnet_x0_25",
        help="TorchReID model name, for example osnet_x0_25 or osnet_x1_0.",
    )
    parser.add_argument(
        "--body-reid-device",
        default="auto",
        help="TorchReID device: auto, cpu, cuda, cuda:0, or mps.",
    )
    parser.add_argument(
        "--process-every-n-frames",
        type=positive_int,
        default=1,
        help="Run YOLO/ReID every N frames and reuse the last boxes between processed frames.",
    )
    parser.add_argument(
        "--face-every-n-frames",
        type=non_negative_int,
        default=3,
        help="Run InsightFace every N source frames; 0 disables face analysis.",
    )
    parser.add_argument(
        "--max-people",
        type=non_negative_int,
        default=0,
        help="Keep only the top N person detections per processed frame; 0 keeps all.",
    )
    parser.add_argument(
        "--hold-missing-frames",
        type=non_negative_int,
        default=6,
        help="Keep drawing the last known Person ID box for this many frames after a missed detection.",
    )
    parser.add_argument(
        "--hold-overlap-iou",
        type=float,
        default=0.25,
        help="Drop a held missing box when it overlaps a live detection by at least this IoU.",
    )
    parser.add_argument(
        "--hold-min-overlap",
        type=float,
        default=0.65,
        help="Drop a held missing box when this fraction of the smaller box overlaps a live detection.",
    )
    parser.add_argument(
        "--min-live-visual-quality",
        type=float,
        default=0.22,
        help="Drop live person detections below this visual quality before ReID and drawing.",
    )
    parser.add_argument(
        "--min-live-area-ratio",
        type=float,
        default=0.0008,
        help="Drop live person detections whose bbox area is below this fraction of the frame.",
    )
    parser.add_argument(
        "--live-nms-iou",
        type=float,
        default=0.55,
        help="Suppress duplicate live person boxes with IoU at or above this value.",
    )
    parser.add_argument(
        "--live-nms-min-overlap",
        type=float,
        default=0.72,
        help="Suppress duplicate live boxes when this fraction of the smaller box is overlapped.",
    )
    parser.add_argument(
        "--det-size",
        type=parse_det_size,
        default=(640, 640),
        help="InsightFace detector input size, for example 640x640.",
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        default=["CPUExecutionProvider"],
        help="ONNX Runtime providers, for example CUDAExecutionProvider CPUExecutionProvider.",
    )
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=DEFAULT_MATCH_THRESHOLD,
        help="Weighted multi-modal threshold for matching Person IDs.",
    )
    parser.add_argument(
        "--spatial-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reject same-camera matches that require an unrealistic jump across the frame.",
    )
    parser.add_argument(
        "--spatial-base-radius",
        type=float,
        default=80.0,
        help="Base allowed center movement in pixels before speed and bbox-size allowances.",
    )
    parser.add_argument(
        "--max-center-speed",
        type=float,
        default=900.0,
        help="Maximum plausible same-camera center speed in pixels per second.",
    )
    parser.add_argument(
        "--spatial-gate-seconds",
        type=float,
        default=4.0,
        help="Apply spatial gate only to profiles seen in the same camera within this many seconds.",
    )
    parser.add_argument(
        "--strong-face-threshold",
        type=float,
        default=0.62,
        help="Face cosine score that can override spatial gating.",
    )
    parser.add_argument(
        "--continuity-threshold",
        type=float,
        default=0.78,
        help="Recent same-position score that preserves an existing ID even when appearance changes.",
    )
    parser.add_argument(
        "--track-memory-seconds",
        type=float,
        default=30.0,
        help="Seconds to keep a camera-local tracker ID mapped to the same Person ID.",
    )
    parser.add_argument(
        "--new-track-match-threshold",
        type=float,
        default=0.72,
        help="Stricter score required before assigning a new tracker ID to an existing Person ID.",
    )
    parser.add_argument(
        "--reentry-memory-seconds",
        type=float,
        default=2.5,
        help="Seconds after a disappearance where a nearby/edge re-entry can reuse an existing Person ID.",
    )
    parser.add_argument(
        "--reentry-match-threshold",
        type=float,
        default=0.58,
        help="Lower match score accepted for short same-camera re-entry after a missed track.",
    )
    parser.add_argument(
        "--reentry-position-threshold",
        type=float,
        default=0.20,
        help="Minimum position continuity score for short re-entry matching.",
    )
    parser.add_argument(
        "--reentry-min-appearance-score",
        type=float,
        default=0.55,
        help="Minimum face/body/clothing score required for short re-entry matching.",
    )
    parser.add_argument(
        "--long-reentry-memory-seconds",
        type=float,
        default=600.0,
        help="Seconds to keep confirmed same-camera profiles eligible for later appearance-based re-entry.",
    )
    parser.add_argument(
        "--long-reentry-match-threshold",
        type=float,
        default=0.66,
        help="Minimum final score for matching a confirmed profile after a longer disappearance.",
    )
    parser.add_argument(
        "--long-reentry-body-threshold",
        type=float,
        default=0.62,
        help="Minimum body ReID score for matching a confirmed profile after a longer disappearance.",
    )
    parser.add_argument(
        "--long-reentry-clothing-threshold",
        type=float,
        default=0.56,
        help="Minimum clothing score for matching a confirmed profile after a longer disappearance.",
    )
    parser.add_argument(
        "--min-confirmed-hits",
        type=positive_int,
        default=3,
        help="Number of observations required before a new Person ID is drawn as confirmed.",
    )
    parser.add_argument(
        "--draw-tentative",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draw tentative Person IDs before they reach --min-confirmed-hits.",
    )
    parser.add_argument(
        "--min-visual-sample-confidence",
        type=float,
        default=0.55,
        help="Minimum visual crop quality for adding body/clothing samples to memory.",
    )
    parser.add_argument(
        "--min-face-score",
        type=float,
        default=DEFAULT_MIN_FACE_SCORE,
        help="Minimum InsightFace detection confidence.",
    )
    parser.add_argument(
        "--max-samples-per-person",
        type=positive_int,
        default=DEFAULT_MAX_SAMPLES_PER_PERSON,
        help="Number of recent samples kept per Person ID and modality.",
    )
    parser.add_argument(
        "--opencv-threads",
        type=non_negative_int,
        default=0,
        help="OpenCV thread count; 0 keeps OpenCV default.",
    )
    parser.add_argument(
        "--reconnect-delay",
        type=float,
        default=2.0,
        help="Seconds to wait before reconnecting an interrupted RTSP stream.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run_realtime(args)


if __name__ == "__main__":
    raise SystemExit(main())
