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
        logger.warning("USE_SYNTHETIC_DATA is set! Using synthetic data (random noise).")
        return generate_synthetic_data()

    logger.info("Checking for local dataset...")
    local_path = "data/default_of_credit_card_clients.xls"
    if os.path.exists(local_path):
        try:
             # Header=1 because usually row 0 is just "X1, X2" and row 1 is actual names "ID, LIMIT_BAL"
            df = pd.read_excel(local_path, header=1)
            logger.info(f"Loaded local dataset from {local_path}")
            
            # Normalize columns
            df.columns = [c.lower() for c in df.columns]
            
            # Rename common targets
            if 'default payment next month' in df.columns:
                df = df.rename(columns={'default payment next month': 'target'})
            
            if 'pay_0' in df.columns:
                df = df.rename(columns={'pay_0': 'pay_1'})
                
            if 'id' in df.columns:
                df = df.drop(columns=['id'])
                
            return df
        except Exception as e:
            logger.error(f"Failed to read local excel file: {e}")

    logger.info("Fetching dataset from UCI ML Repo...")
    try:
        # ID 350 is 'default of credit card clients'
        dataset = fetch_ucirepo(id=350)
        
        X = dataset.data.features
        y = dataset.data.targets
        
        X = dataset.data.features
        y = dataset.data.targets
        
        df = pd.concat([X, y], axis=1)
        # Verify valid columns
        
        if 'Y' in df.columns:
            df = df.rename(columns={'Y': 'target'})
            
        # Mapping for raw UCI columns (X1...X23)
        column_mapping = {
            'X1': 'limit_bal',
            'X2': 'sex',
            'X3': 'education',
            'X4': 'marriage',
            'X5': 'age',
            'X6': 'pay_1',
            'X7': 'pay_2',
            'X8': 'pay_3',
            'X9': 'pay_4',
            'X10': 'pay_5',
            'X11': 'pay_6',
            'X12': 'bill_amt1',
            'X13': 'bill_amt2',
            'X14': 'bill_amt3',
            'X15': 'bill_amt4',
            'X16': 'bill_amt5',
            'X17': 'bill_amt6',
            'X18': 'pay_amt1',
            'X19': 'pay_amt2',
            'X20': 'pay_amt3',
            'X21': 'pay_amt4',
            'X22': 'pay_amt5',
            'X23': 'pay_amt6'
        }
        df = df.rename(columns=column_mapping)
        
        # Lowercase any remaining
        df.columns = [c.lower() for c in df.columns]

        # Handle legacy names just in case
        if 'default payment next month' in df.columns:
            df = df.rename(columns={'default payment next month': 'target'})
            
        logger.info(f"Dataset loaded from UCI. Shape: {df.shape}")
        return df
        
    except Exception as e:
        logger.warning(f"Failed to fetch data via ucimlrepo ({e}). Using SYNTHETIC data.")
        return generate_synthetic_data()
