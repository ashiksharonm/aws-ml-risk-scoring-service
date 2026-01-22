import pandas as pd
from ucimlrepo import fetch_ucirepo
from src.utils.config import settings
from src.training.mock_data import generate_synthetic_data
import logging
import os

logger = logging.getLogger(__name__)

def load_data() -> pd.DataFrame:
    """
    Load the UCI Credit Default dataset.
    Uses 'USE_SYNTHETIC_DATA' env var to force synthetic data.
    Otherwise attempts to fetch from UCI ML Repo.
    """
    if os.getenv("USE_SYNTHETIC_DATA", "false").lower() == "true":
        logger.info("USE_SYNTHETIC_DATA is set. Using synthetic data.")
        return generate_synthetic_data()

    logger.info("Fetching dataset from UCI ML Repo...")
    try:
        # ID 350 is 'default of credit card clients'
        dataset = fetch_ucirepo(id=350)
        
        X = dataset.data.features
        y = dataset.data.targets
        
        df = pd.concat([X, y], axis=1)
        df.columns = [c.lower() for c in df.columns]
        
        if 'default payment next month' in df.columns:
            df = df.rename(columns={'default payment next month': 'target'})
        elif 'default_payment_next_month' in df.columns:
            df = df.rename(columns={'default_payment_next_month': 'target'})
            
        if 'pay_0' in df.columns:
            df = df.rename(columns={'pay_0': 'pay_1'})
        
        if 'id' in df.columns:
            df = df.drop(columns=['id'])
            
        logger.info(f"Dataset loaded from UCI. Shape: {df.shape}")
        return df
        
    except Exception as e:
        logger.warning(f"Failed to fetch data via ucimlrepo ({e}). Using SYNTHETIC data.")
        return generate_synthetic_data()
