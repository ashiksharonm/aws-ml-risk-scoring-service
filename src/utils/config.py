from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, overridable through environment variables or .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    PROJECT_NAME: str = "Risk Scoring Service"
    VERSION: str = "2.0.0"
    LOG_LEVEL: str = "INFO"

    # Model artifacts (written by `make train`)
    MODEL_PATH: str = "models/model.pkl"
    CHALLENGER_PATH: str = "models/challenger.pkl"
    REFERENCE_PATH: str = "models/reference.json"
    MODEL_CARD_PATH: str = "models/model_card.json"
    PROCESSED_DATA_DIR: str = "data/processed"

    # Decisioning
    DECISION_THRESHOLD: float = 0.5
    REASON_CODE_COUNT: int = 4

    # Monitoring
    SHADOW_LOG_PATH: str = "logs/shadow_scores.jsonl"
    LIVE_BUFFER_SIZE: int = 1000
    MIN_DRIFT_ROWS: int = 100

    # AWS (only used by the optional Lambda/EC2 deploy path)
    AWS_REGION: str = "us-east-1"


settings = Settings()
