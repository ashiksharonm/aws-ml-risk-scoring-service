import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "AWS ML Risk Scoring Service"
    VERSION: str = "1.0.0"
    
    # Model Paths
    MODEL_PATH: str = "models/model.pkl"
    PREPROCESSOR_PATH: str = "models/preprocessor.pkl"
    
    # Data Paths
    # DATA_PATH DEPRECATED: We use ucimlrepo now. 
    # But we keep processed dir.
    PROCESSED_DATA_DIR: str = "data/processed"
    
    # AWS Config (loaded from env)
    AWS_REGION: str = "us-east-1"
    
    class Config:
        env_file = ".env"

settings = Settings()
