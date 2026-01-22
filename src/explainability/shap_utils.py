import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.utils.config import settings

def get_explainer(model, X_background):
    """
    Returns a SHAP TreeExplainer for the XGBoost model.
    model: The trained XGBoost model (underlying booster).
    X_background: Background dataset for SHAP (optional for TreeExplainer but good practice).
    """
    return shap.TreeExplainer(model)

def generate_shap_plots(pipeline, X_sample):
    """
    Generates summary plots for features.
    """
    model = pipeline.named_steps['classifier']
    preprocessor = pipeline.named_steps['preprocessor']
    
    # Transform data to match model input
    X_transformed = preprocessor.transform(X_sample)
    feature_names = get_feature_names(preprocessor)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed)
    
    plt.figure()
    shap.summary_plot(shap_values, X_transformed, feature_names=feature_names, show=False)
    plt.savefig('reports/shap_summary.png', bbox_inches='tight')
    plt.close()

def get_feature_names(preprocessor):
    """
    Extracts feature names from the sklearn pipeline preprocessor.
    """
    output_features = []
    
    # Check transformers
    for name, pipe, features in preprocessor.transformers_:
        if name == 'remainder':
            continue
        if hasattr(pipe, 'get_feature_names_out'):
            # Sklearn 1.0+
            output_features.extend(pipe.get_feature_names_out(features))
        elif hasattr(pipe, 'named_steps'):
            # If it's a pipeline inside column transformer
             # Check the last step of the sub-pipeline
            last_step = pipe.steps[-1][1]
            if hasattr(last_step, 'get_feature_names_out'):
                 output_features.extend(last_step.get_feature_names_out(features))
            else:
                 output_features.extend(features)
        else:
            output_features.extend(features)
            
    return output_features
