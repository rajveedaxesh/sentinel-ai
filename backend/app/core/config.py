from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    app_name: str = "SentinelAI"
    environment: str = "development"
    database_url: str = f"sqlite:///{(PROJECT_ROOT / 'data' / 'sentinelai.db').as_posix()}"
    data_path: Path = PROJECT_ROOT / "data" / "sample" / "metrics.csv"
    artifact_path: Path = PROJECT_ROOT / "models" / "isolation_forest.joblib"
    max_history_rows: int = 5000
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")


settings = Settings()