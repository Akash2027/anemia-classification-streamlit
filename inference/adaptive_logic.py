import pandas as pd
import numpy as np
import joblib
import json
import os
from preprocessing.conditioning import compute_physiological_features

def perform_adaptive_inference(user_input_dict):
    """
    Combines Path A and Path B outputs with shifts mapped to specific labels.
    Includes robustness fixes for MCHC and feature selection.
    """
    # 1. Load models and thresholds
    model_path = 'models/statistical_model.pkl'
    rule_path = 'models/rule_model.pkl'
    threshold_path = 'models/thresholds.json'

    if not all(os.path.exists(p) for p in [model_path, rule_path, threshold_path]):
        raise FileNotFoundError("Required model artifacts missing. Please run training pipeline.")

    stat_model = joblib.load(model_path)
    rule_model = joblib.load(rule_path)
    with open(threshold_path, 'r') as f:
        thresholds = json.load(f)

    # 2. Convert input to DataFrame and Calculate Indices
    raw_df = pd.DataFrame([user_input_dict])
    
    # --- ISSUE FIX: HCT Normalization for MCHC ---
    # Detect if HCT is fractional (e.g., 0.4) vs percentage (e.g., 40.0)
    hct_val = raw_df["HCT"].iloc[0]
    if hct_val > 0 and hct_val < 1:
        hct_norm = hct_val * 100
    else:
        hct_norm = hct_val

    # Explicitly calculate MCH and MCHC if not provided
    if "MCH" not in raw_df.columns:
        raw_df["MCH"] = (raw_df["HGB"] / (raw_df["RBC"] + 1e-5)) * 10
    if "MCHC" not in raw_df.columns:
        raw_df["MCHC"] = (raw_df["HGB"] / (hct_norm + 1e-5)) * 100

    # Apply Physiological Conditioning
    conditioned_df = compute_physiological_features(raw_df)
    
    # --- ISSUE FIX: Feature Robustness ---
    required_features = thresholds["feature_names"]
    missing_features = [f for f in required_features if f not in conditioned_df.columns]
    if missing_features:
        raise ValueError(f"Inference failed: Conditioned data missing features: {missing_features}")
    
    # 3. Path A: Get Statistical Probabilities
    X = conditioned_df[required_features]
    stat_probs = stat_model.predict_proba(X)[0]
    classes = stat_model.classes_
    prob_dict = dict(zip(classes, stat_probs))

    # 4. Path B: Detect Medical Contradictions
    # Predicts the probability that this feature set represents a physiological rule violation
    violation_detected = rule_model.predict(X)[0]

    # 5. Adaptive Probability Shifting
    inf_idx = conditioned_df["inflammation_index"].iloc[0]
    
    if inf_idx > thresholds.get("inflammation_high", 500):
        if "Iron deficiency anemia" in prob_dict:
            prob_dict["Iron deficiency anemia"] *= 0.8
        if "Leukemia" in prob_dict:
            prob_dict["Leukemia"] *= 1.4
        if "Normocytic hypochromic anemia" in prob_dict:
            prob_dict["Normocytic hypochromic anemia"] *= 1.2

    # If Path B detected a rule violation, flatten the distribution (increase uncertainty)
    if violation_detected:
        for key in prob_dict:
            prob_dict[key] = (prob_dict[key] + 0.1) / (1.0 + (0.1 * len(prob_dict)))

    # 6. Re-normalize Probabilities to sum to 1.0
    total_prob = sum(prob_dict.values())
    final_probs = {k: v / total_prob for k, v in prob_dict.items()}

    return final_probs, bool(violation_detected)