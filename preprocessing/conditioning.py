import pandas as pd
import numpy as np

def compute_physiological_features(df):
    """
    Computes derived physiological features aligned with the user dataset.
    Handles WBC, NEUTp, and LYMp columns for medical logic.
    """
    conditioned_df = df.copy()

    # 1. Oxygen-Carrying Capacity (HGB x RBC)
    conditioned_df["oxygen_capacity"] = conditioned_df["HGB"] * conditioned_df["RBC"]

    # 2. Hemoglobin Efficiency (HGB / RBC)
    conditioned_df["hb_efficiency"] = conditioned_df["HGB"] / (conditioned_df["RBC"] + 1e-5)

    # 3. Cell Volume-Hemoglobin Mismatch (Cell Size Disparity)
    # Ensure MCH exists or calculate it: (HGB/RBC)*10
    if "MCH" not in conditioned_df.columns or conditioned_df["MCH"].isnull().any():
        conditioned_df["MCH"] = (conditioned_df["HGB"] / (conditioned_df["RBC"] + 1e-5)) * 10
        
    conditioned_df["cell_size_gap"] = abs(conditioned_df["MCV"] - (conditioned_df["MCH"] * 10))

    # 4. Inflammation Intensity (Using WBC, NEUTp, LYMp)
    # Formula: (WBC * Neutrophil%) / (Lymphocyte% + 1)
    # Aligned to your dataset headers
    wbc_val = conditioned_df.get("WBC", 0)
    neut_p = conditioned_df.get("NEUTp", 0)
    lym_p = conditioned_df.get("LYMp", 0)
    
    conditioned_df["inflammation_index"] = (wbc_val * neut_p) / (lym_p + 1)

    # 5. Platelet Stress Score (PDW x PLT)
    conditioned_df["platelet_stress"] = conditioned_df.get("PDW", 0) * conditioned_df.get("PLT", 0)

    return conditioned_df

def normalize_conditioned_features(df, features_to_scale):
    for col in features_to_scale:
        if col in df.columns:
            min_val = df[col].min()
            max_val = df[col].max()
            df[col] = (df[col] - min_val) / (max_val - min_val + 1e-9)
    return df