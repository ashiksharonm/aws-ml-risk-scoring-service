from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np

def get_preprocessor(categorical_features: list[str], numerical_features: list[str]) -> ColumnTransformer:
    """
    Creates a scikit-learn preprocessing pipeline.
    """
    
    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ])
    
    return preprocessor

def split_features_target(df: pd.DataFrame, target_col: str = 'target'):
    """
    Separates features and target.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y
