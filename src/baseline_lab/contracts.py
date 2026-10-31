"""Version 1: pixels refer to the displayed, rotation-corrected image; times are seconds."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

Point = tuple[float, float]


class Calibration(BaseModel):
    # Clockwise singles outside painted corners: far-left, far-right, near-right, near-left.
    corners_px: list[Point] = Field(min_length=4, max_length=4)


class Submission(BaseModel):
    calibration: Calibration | None = None
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
    warnings: list[str] = []


class Observation(BaseModel):
    t_s: float
    frame: int
    state: Literal["detected", "interpolated", "missing", "rejected"]
    xy_px: Point | None = None
    raw_xy_px: Point | None = None
    bbox_px: tuple[float, float, float, float] | None = None
    court_m: Point | None = None
    score: float | None = None
    reason: str | None = None


class Track(BaseModel):
    id: str
    observations: list[Observation]
    coverage: float
    distance_m: float | None = None
    distance_reason: str | None = None


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
    depth_semantics: str = "signed distance inside receiving baseline; reviewed ground contact only"
    original_prediction: dict | None = None
    notes: str = ""


class Correction(BaseModel):
    status: Literal["confirmed", "rejected", "manual"]
    t_s: float = Field(ge=0)
    xy_px: Point
    notes: str = Field(default="", max_length=1000)


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


class Result(BaseModel):
    schema_version: Literal["1.0", "1.1"] = "1.1"
    metadata: Metadata
    tracks: list[Track]
    events: list[Event]
    calibration: dict | None
    warnings: list[str]
    provenance: dict
    live_metrics: list[LiveMetrics] = []
    ball_speed_m_s: float | None = None
    ball_speed_reason: str = "single_view_3d_speed_not_validated"
    elapsed_s: float

    @model_validator(mode="after")
    def no_unfounded_ball_speed(self) -> "Result":
        if self.ball_speed_m_s is not None:
            raise ValueError("Validated 3D speed is not implemented")
        return self
