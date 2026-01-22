import pandas as pd
import numpy as np
from src.utils.config import settings
import logging

logger = logging.getLogger(__name__)

def generate_synthetic_data(n_rows=1000):
    """
    Generates synthetic data matching the UCI Credit Default schema.
    Used when external download fails.
    """
    logger.info(f"Generating {n_rows} rows of synthetic data...")
    np.random.seed(42)
    
    data = {
        'limit_bal': np.random.randint(10000, 1000000, n_rows),
        'sex': np.random.choice([1, 2], n_rows),
        'education': np.random.choice([1, 2, 3, 4], n_rows),
        'marriage': np.random.choice([1, 2, 3], n_rows),
        'age': np.random.randint(21, 79, n_rows),
    }
    
    for i in range(1, 7):
        data[f'pay_{i}'] = np.random.randint(-2, 9, n_rows)
        data[f'bill_amt{i}'] = np.random.randn(n_rows) * 50000 + 50000
        data[f'pay_amt{i}'] = np.random.randn(n_rows) * 5000 + 2000
        
    data['target'] = np.random.randint(0, 2, n_rows)
    
    df = pd.DataFrame(data)
    
    # Ensure correct types for pay columns (int)
    # The generated are int/float mixed potentially, verify
    return df
