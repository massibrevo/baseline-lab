"""Version 1: pixels refer to the displayed, rotation-corrected image; times are seconds."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

Point = tuple[float, float]


class Calibration(BaseModel):
    # Clockwise singles outside painted corners: far-left, far-right, near-right, near-left.
    corners_px: list[Point] = Field(min_length=4, max_length=4)


class Submission(BaseModel):
    calibration: Calibration | None = None
    real_time_playback: bool = False
    preset: Literal["cpu-baseline"] = "cpu-baseline"
    idempotency_key: str = Field(min_length=1, max_length=80)


class Metadata(BaseModel):
    width: int
    height: int
    duration_s: float
    codec: str
    frames: int
    nominal_fps: float | None
    variable_frame_rate: bool
    rotation_deg: int
    timestamps_s: list[float]
    original_pts: list[int]
    time_base: str
    source_width: int | None = None
    source_height: int | None = None
    analysis_scale: float = 1.0
    warnings: list[str] = []


class Keypoint(BaseModel):
    name: str
    raw_xy_px: Point | None = None
    xy_px: Point | None = None
    state: Literal["detected", "missing"]
    visibility: float | None
    presence: float | None
    score: float | None = None
    score_semantics: str = "model visibility/presence scores; not calibrated probability"


class Observation(BaseModel):
    t_s: float
    frame: int
    state: Literal["detected", "interpolated", "missing", "rejected"]
    xy_px: Point | None = None
    raw_xy_px: Point | None = None
    apparent_diameter_px: float | None = None
    bbox_px: tuple[float, float, float, float] | None = None
    keypoints: list[Keypoint] = []
    pose_score: float | None = None
    pose_reason: str | None = "pose_not_run"
    footpoint_source: str = "bbox_bottom_approximate"
    court_m: Point | None = None
    score: float | None = None
    reason: str | None = None


class Track(BaseModel):
    id: str
    observations: list[Observation]
    coverage: float
    distance_m: float | None = None
    distance_reason: str | None = None


class ShotEvidence(BaseModel):
    speed_kmh: float | None = Field(default=None, ge=0, le=300, allow_inf_nan=False)
    spin_rpm: float | None = Field(default=None, ge=0, le=10000, allow_inf_nan=False)
    bounce_angle_deg: float | None = Field(default=None, ge=0, le=90, allow_inf_nan=False)
    source: str = Field(min_length=3, max_length=200)


class QualityFactor(BaseModel):
    value: float | None = None
    unit: str
    score: float | None = None
    weight: float
    source: str


class ShotQuality(BaseModel):
    score: float | None = None
    available_factors: int
    total_factors: int = 5
    evidence_coverage: float
    status: Literal["unavailable", "partial", "complete"]
    factors: dict[str, QualityFactor]
    method: str = (
        "groundstroke-pressure-v1; experimental weighted heuristic, not validated skill rating"
    )
    error_attribution: Literal["undetermined"] = "undetermined"


class Event(BaseModel):
    id: str
    t_s: float
    frame: int
    xy_px: Point
    evidence_score: float | None = None
    score_semantics: str = "heuristic temporal-evidence score; not probability"
    status: Literal["candidate", "confirmed", "rejected", "manual"] = "candidate"
    classification: Literal["estimated_in", "estimated_out", "uncertain"] = "uncertain"
    court_m: Point | None = None
    shot_depth_m: float | None = None
    shot_width_m: float | None = None
    shot_evidence: ShotEvidence | None = None
    shot_quality: ShotQuality | None = None
    depth_semantics: str = "signed distance inside receiving baseline; reviewed ground contact only"
    original_prediction: dict | None = None
    notes: str = ""


class Correction(BaseModel):
    status: Literal["confirmed", "rejected", "manual"]
    t_s: float = Field(ge=0)
    xy_px: Point
    notes: str = Field(default="", max_length=1000)
    shot_evidence: ShotEvidence | None = None


class LiveMetrics(BaseModel):
    """Measurements at one presentation timestamp; nulls are deliberate evidence gaps."""

    t_s: float
    frame: int
    ball_image_speed_px_s: float | None = None
    ball_image_speed_reason: str | None = None
    near_speed_m_s: float | None = None
    far_speed_m_s: float | None = None
    near_observed_distance_m: float | None = None
    far_observed_distance_m: float | None = None
    ball_observation_coverage_so_far: float
    contact_candidates_so_far: int
    near_visible_joints: int = 0
    far_visible_joints: int = 0
    near_left_knee_angle_deg_2d: float | None = None
    near_right_knee_angle_deg_2d: float | None = None
    far_left_knee_angle_deg_2d: float | None = None
    far_right_knee_angle_deg_2d: float | None = None


class ShotSignal(BaseModel):
    id: str
    t_s: float
    frame: int
    player: Literal["near", "far"]
    kind: Literal[
        "forehand", "backhand", "forehand_volley", "backhand_volley", "serve", "overhead", "unknown"
    ]
    status: Literal["estimated", "reviewed"] = "estimated"
    method: str = "ball/wrist proximity candidate; stroke type unclassified pending review"
    evidence_score: float
    score_semantics: str = "heuristic evidence, not probability"
    image_speed_px_s: float | None = None
    original_prediction: dict | None = None
    context_start_s: float | None = None
    context_end_s: float | None = None
    recognition_basis: str | None = None
    model_scores: dict[str, float] = {}
    technique: str | None = None


class ShotReview(BaseModel):
    kind: Literal[
        "forehand", "backhand", "forehand_volley", "backhand_volley", "serve", "overhead", "unknown"
    ]


class StrokeAnnotation(ShotReview):
    t_s: float = Field(ge=0)
    player: Literal["near", "far"]


class BallSpeedSample(BaseModel):
    t_s: float
    frame: int
    estimated_kmh: float | None = None
    low_kmh: float | None = None
    high_kmh: float | None = None
    status: Literal["experimental", "unavailable"] = "unavailable"
    reason: str | None = None
    support_start_s: float | None = None
    support_end_s: float | None = None
    reprojection_rms_px: float | None = None


class DisplayTrailPoint(BaseModel):
    t_s: float
    frame: int
    xy_px: Point | None
    state: Literal["observed", "interpolated", "inferred", "unavailable"]
    reason: str | None = None


class Result(BaseModel):
    schema_version: Literal["1.0", "1.1", "1.2", "1.3", "1.4"] = "1.4"
    metadata: Metadata
    tracks: list[Track]
    events: list[Event]
    calibration: dict | None
    frame_calibrations: list[dict | None] = []
    warnings: list[str]
    provenance: dict
    live_metrics: list[LiveMetrics] = []
    shots: list[ShotSignal] = []
    automatic_shots: list[ShotSignal] = []
    display_ball_trail: list[DisplayTrailPoint] = []
    stroke_analysis: dict = {}
    ball_speed_estimates: list[BallSpeedSample] = []
    speed_analysis: dict = {}
    ball_speed_m_s: float | None = None
    ball_speed_reason: str = "single_view_3d_speed_not_validated"
    elapsed_s: float

    @model_validator(mode="after")
    def no_unfounded_ball_speed(self) -> "Result":
        if self.ball_speed_m_s is not None:
            raise ValueError("Validated 3D speed is not implemented")
        return self


class LiveSessionInput(BaseModel):
    reference_jpeg_base64: str | None = Field(default=None, max_length=550000)
    width: int = Field(ge=320, le=1280)
    height: int = Field(ge=240, le=720)
    calibration: Calibration | None = None
    suitability_confirmed: Literal[True]


class CameraFrame(BaseModel):
    t_s: float = Field(ge=0, le=60)
    jpeg_base64: str = Field(min_length=20, max_length=550000)


class CameraBatch(BaseModel):
    sequence: int = Field(ge=0, le=1800)
    frames: list[CameraFrame] = Field(min_length=1, max_length=8)
