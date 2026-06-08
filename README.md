# 🩸 Adaptive Physiology-Aware Anemia Classification System

A biology-constrained, dual-path diagnostic decision support system that integrates statistical machine learning with symbolic medical logic to classify anemia types from Complete Blood Count (CBC) parameters.

This system resolves the "black-box" limitations of standard classifiers by enforcing clinical rules, assessing data reliability, and dynamically adapting diagnostic outputs based on secondary physiological stress markers.

---


## 🌐 Live Demo

**Streamlit Application:**
https://anemia-classification-system.streamlit.app/

## 📂 Project Architecture

```text
├── data/
│   └── diagnosed_cbc_data_v4.csv      # Training dataset containing diagnosed CBC records
├── preprocessing/
│   ├── conditioning.py                # Computes derived mathematical/biological indices
│   └── reliability.py                 # Scores data consistency & identifies flag violations
├── models/
│   ├── training_pipeline.py           # Coordinates dual-path training & saves artifacts
│   ├── statistical_model.pkl          # Path A: Random Forest Classifier
│   ├── rule_model.pkl                 # Path B: Gradient Boosting Classifier
│   └── thresholds.json                # Saved feature names, rules, and dynamic thresholds
├── inference/
│   ├── plausibility.py                # Pre-inference hard biological validation gate
│   ├── adaptive_logic.py              # Probability shifting & multi-path model fusion
│   └── confidence.py                  # Evaluates final response boundaries and tiers
└── app_streamlit.py                   # Main interactive clinician UI and PDF generator

```

---

## ⚙️ Core Technical Workflow

```text
[User Input Module] ──► [Plausibility Gate] ──► [Physiological Conditioning] 
                                                         │
                                        ┌────────────────┴────────────────┐
                                        ▼                                 ▼
                                [Path A: Statistical]             [Path B: Symbolic]
                                (Random Forest Probabilities)     (Gradient Boosting Audit)
                                        │                                 │
                                        └────────────────┬────────────────┘
                                                         ▼
                                        [Adaptive Probability Shifting] 
                                                         │
                                                         ▼
                                            [Confidence Evaluation] 
                                                         │
                                                         ▼
                                             [Result Reporting Module]

```

1. **Plausibility Gate Module:** Performs immediate hard boundary validation on incoming raw metrics to intercept data corruption, impossible biological ranges, or lab transposition errors.
2. **Physiological Feature Conditioning Module:** Synthesizes raw inputs into multidimensional biological markers (Oxygen Capacity, Inflammation Index, Platelet Stress, and Hemoglobin Efficiency).
3. **Dual-Path Inference Engine:**
* **Path A (Statistical):** Maps statistical patterns across all features using a reliability-weighted model.
* **Path B (Rule Violation Detector):** Audits the input feature architecture to compute the mathematical probability of an active physiological contradiction.


4. **Adaptive Probability Adjustment Module:** Shifts output probabilities based on structural context—dampening iron deficiency and boosting inflammatory pathways during high-inflammation states, while flattening distributions when contradictions emerge.
5. **Confidence Tiering Module:** Fuses outputs into distinct clarity bands: **High, Moderate, Low/Suspicious, or Ambiguous**.
6. **Result Reporting Module:** Renders complete breakdowns on the UI and handles professional PDF generations.

---

## 🛠️ Installation & Setup

### 1. Environment Setup

Clone the repository and ensure you have Python installed alongside the required package stack:

```bash
# Clone the repository
git clone https://github.com/Akash2027/Anemia-Classification-System.git
cd Anemia-Classification-System

# Install required dependencies
pip install pandas numpy scikit-learn joblib streamlit fpdf

```

### 2. Execute Training Pipeline

To condition the source dataset, map reliability weights, and train both Path A and Path B models, execute the training file:

```bash
python -m models.training_pipeline

```

*This produces `statistical_model.pkl`, `rule_model.pkl`, and `thresholds.json` inside the `models/` directory.*

### 3. Launch UI Application

Run the interface locally using Streamlit:

```bash
streamlit run app_streamlit.py

```

---

## 🧪 System Validation Scenarios

Use the following reference metrics to validate the multi-layer logic checks within your localized environment:

| Test Case | Target Scenario | HGB | RBC | MCV | HCT | WBC | PLT | NEUTp | LYMp | System Expected Behavior |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Case 1** | Healthy Control | 14.5 | 4.8 | 88.0 | 43.0 | 7.0 | 250 | 60% | 30% | **Healthy**: Classified with High Confidence. |
| **Case 2** | Iron Deficiency | 9.0 | 3.8 | 68.0 | 28.0 | 6.5 | 300 | 55% | 35% | **Microcytic Hypochromic Anemia** pattern matches. |
| **Case 4** | Extreme Low Crash | **0.5** | 4.0 | 85.0 | 40.0 | 6.0 | 250 | 60% | 30% | **REJECTED**: Intercepted at Plausibility Gate ($HGB < 2.0$). |
| **Case 5** | Erroneous MCHC | 14.0 | 5.0 | 85.0 | **15.0** | 7.0 | 250 | 60% | 30% | **REJECTED**: Intercepted due to non-viable MCHC saturation ($>50$). |
| **Case 6** | Rule Contradiction | 13.0 | **2.0** | **70.0** | 40.0 | 7.0 | 250 | 60% | 30% | **Low/Suspicious Tier**: Path B forces probability flattening. |
| **Case 7** | Inflammation Shift | 10.5 | 4.1 | 84.0 | 32.0 | **22.0** | 350 | **92%** | 5% | **Adaptive Boost**: Weight shifted to Normocytic/Infectious markers. |

---

## 📑 Core Implementation Assets

### Preprocessing Module: `preprocessing/reliability.py`

```python
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

```

### Inference Module: `inference/adaptive_logic.py`

```python
import pandas as pd
import numpy as np
import joblib
import json
import os
from preprocessing.conditioning import compute_physiological_features

def perform_adaptive_inference(user_input_dict):
    """
    Combines Path A and Path B outputs with adaptive structural adjustments.
    Includes explicit error shielding and HCT fractional normalization checks.
    """
    model_path = 'models/statistical_model.pkl'
    rule_path = 'models/rule_model.pkl'
    threshold_path = 'models/thresholds.json'

    if not all(os.path.exists(p) for p in [model_path, rule_path, threshold_path]):
        raise FileNotFoundError("Required model artifacts missing. Please run the training pipeline.")

    stat_model = joblib.load(model_path)
    rule_model = joblib.load(rule_path)
    with open(threshold_path, 'r') as f:
        thresholds = json.load(f)

    raw_df = pd.DataFrame([user_input_dict])
    
    # Normalization boundary check for fractional HCT inputs
    hct_val = raw_df["HCT"].iloc[0]
    if 0 < hct_val < 1:
        hct_norm = hct_val * 100
    else:
        hct_norm = hct_val

    # Derive dependent indices if absent from root payload
    if "MCH" not in raw_df.columns:
        raw_df["MCH"] = (raw_df["HGB"] / (raw_df["RBC"] + 1e-5)) * 10
    if "MCHC" not in raw_df.columns:
        raw_df["MCHC"] = (raw_df["HGB"] / (hct_norm + 1e-5)) * 100

    conditioned_df = compute_physiological_features(raw_df)
    
    # Verify input feature vector integrity against schema
    required_features = thresholds["feature_names"]
    missing_features = [f for f in required_features if f not in conditioned_df.columns]
    if missing_features:
        raise ValueError(f"Inference blocked: Schema validation missing: {missing_features}")
    
    # Path A Evaluation
    X = conditioned_df[required_features]
    stat_probs = stat_model.predict_proba(X)[0]
    classes = stat_model.classes_
    prob_dict = dict(zip(classes, stat_probs))

    # Path B Evaluation
    violation_detected = rule_model.predict(X)[0]

    # Adaptive Structural Shifting Engine
    inf_idx = conditioned_df["inflammation_index"].iloc[0]
    if inf_idx > thresholds.get("inflammation_high", 500):
        if "Iron deficiency anemia" in prob_dict:
            prob_dict["Iron deficiency anemia"] *= 0.8
        if "Leukemia" in prob_dict:
            prob_dict["Leukemia"] *= 1.4
        if "Normocytic hypochromic anemia" in prob_dict:
            prob_dict["Normocytic hypochromic anemia"] *= 1.2

    # Uncertainty flattening layer via entropy injection during structural conflicts
    if violation_detected:
        for key in prob_dict:
            prob_dict[key] = (prob_dict[key] + 0.1) / (1.0 + (0.1 * len(prob_dict)))

    # Re-normalize vector probability density
    total_prob = sum(prob_dict.values())
    final_probs = {k: v / total_prob for k, v in prob_dict.items()}

    return final_probs, bool(violation_detected)

```

## ⚖️ Intellectual Property Protection Profile

This implementation maps directly to the formal claims defined in the published patent application:
**"A Physiology-Aware Dual-Path System and Method for Automated Anemia Classification Using Hematological Parameters."**

### 🔒 Protected Core Innovations:
1. **Parallel Contradiction Mapping (Dual-Path Engine):** The unique framework running structural machine learning models (Path A) alongside a parallel symbolic rule-learner (Path B) strictly to predict the activation probability of clinical contradictions.
2. **Dynamic Context Distribution Shifts:** The proprietary method of utilizing an unlinked systemic stress score (Inflammation Index) to act as a dynamic bias modulator over an algorithmic probability array.
3. **Reliability-Weighted Training Logic:** The automated validation layer that weights clinical records during training based on their adherence to physiological constraints.

*Unauthorized reproduction, reverse engineering, or commercial exploitation of this architecture or its specific logical transitions is strictly prohibited under applicable patent laws.*

### Non-Obvious Novelty Core:

1. **Parallel Contradiction Mapping:** Running structural machine learning models strictly to predict the activation probability of a clinical contradiction based on feature inputs rather than target classification labels.
2. **Dynamic Context Distribution Shifts:** Explicitly using an unlinked operational system stress score (Inflammation Index) to act as a dynamic bias modulator over an algorithmic probability array.

---

## ⚠️ Medical Intelligence Disclaimer

This tool is configured strictly to serve as a **Clinical Decision Support System (CDSS)**. It evaluates mathematical and biological relationships derived from input telemetry and should never displace definitive independent clinical evaluations by certified practitioners or laboratory medical officers.
