import pandas as pd
import numpy as np

def calculate_reliability_score(row):
    """
    Assigns a weight between 0.0 and 1.0 based on physiological consistency.
    """
    score = 1.0
    
    # Rule 1: Extreme Hemoglobin (HGB) Check
    if row["HGB"] < 3 or row["HGB"] > 20:
        score *= 0.5
        
    # Rule 2: Erythrocyte Indices Coherence (HCT ≈ RBC × MCV)
    hct_estimate = (row["RBC"] * row["MCV"]) / 10
    if abs(row["HCT"] - hct_estimate) > 15:
        score *= 0.6

    # Rule 3: MCV-MCH Compatibility
    if row["MCV"] < 60 or row["MCV"] > 120:
        score *= 0.7
        
    # Rule 4: White Blood Cell (WBC) Balance
    if "NEUTp" in row and "LYMp" in row:
        if (row["NEUTp"] + row["LYMp"]) > 105: 
            score *= 0.8

    # Rule 5: Zero/Negative values (Strictly impossible)
    essential_cols = ["HGB", "RBC", "MCV", "WBC"]
    for col in essential_cols:
        if row[col] <= 0:
            score = 0.1 
            break

    return round(score, 2)

def apply_reliability_to_dataset(df):
    df["reliability"] = df.apply(calculate_reliability_score, axis=1)
    return df

def get_consistency_flags(df):
    """
    Generates binary indicators for rule violations.
    Used as input for Path B (Physiological Rule Violation Learner).
    """
    flags = pd.DataFrame(index=df.index)
    flags["err_hgb_range"] = ((df["HGB"] < 3) | (df["HGB"] > 20)).astype(int)
    flags["err_rbc_mcv_hct"] = (abs(df["HCT"] - (df["RBC"] * df["MCV"] / 10)) > 15).astype(int)
    flags["err_mcv_range"] = ((df["MCV"] < 60) | (df["MCV"] > 120)).astype(int)
    
    return flags