"""One configuration surface. Secrets and user artifacts never enter source control."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="BASELINE_", env_file=".env")
    data_dir: Path = ROOT / "data"
    max_upload_mb: int = 100
    min_duration_s: float = 5
    max_duration_s: float = 30
    max_pixels: int = 1920 * 1080
    max_source_pixels: int = 4096 * 2160
    max_frames: int = 1800
    job_timeout_s: int = 900
    player_model: str = "ssdlite"
    secure_cookies: bool = False
    allowed_origins: list[str] = ["http://127.0.0.1:8000", "http://localhost:8000"]

    def prepare(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)


def settings() -> Settings:
    return Settings()
