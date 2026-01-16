import json

def check_input_plausibility(user_input):
    """
    Hard biological validation gate.
    Returns: (is_plausible, error_message)
    """
    
    # 1. Physical Constraints (Non-negativity)
    for param, value in user_input.items():
        if value < 0:
            return False, f"Invalid input: {param} cannot be negative."

    # 2. Hard Biological Ranges (Living Human Limits)
    ranges = {
        "HGB": (2.0, 25.0),    
        "RBC": (0.5, 10.0),    
        "MCV": (40.0, 150.0),  
        "PLT": (5.0, 2000.0),  
        "WBC": (0.1, 300.0)    
    }

    for param, (low, high) in ranges.items():
        if param in user_input:
            val = user_input[param]
            if val < low or val > high:
                return False, f"Physiologically impossible value for {param}: {val}"

    # 3. Ratio-Based Hard Checks (MCHC Check)
    # Physically cannot exceed ~45-50 g/dL (Hb saturation point)
    if user_input.get("HGB") and user_input.get("HCT"):
        # Prevent division by zero
        hct_val = user_input["HCT"] if user_input["HCT"] > 0 else 1e-5
        mchc_val = (user_input["HGB"] / hct_val) * 100
        if mchc_val > 50: 
            return False, f"MCHC ({round(mchc_val, 1)}) exceeds biological saturation limit."

    return True, "Success"

def soft_consistency_check(user_input, thresholds_path='models/thresholds.json'):
    """
    Checks for suspicious but physically possible combinations (e.g., Rule of Three).
    """
    try:
        with open(thresholds_path, 'r') as f:
            thresholds = json.load(f)
    except FileNotFoundError:
        return ["Model metadata not found. Please run training."]

    warnings = []
    
    # Check "Rule of Three" deviation: HCT should be ~ RBC * MCV / 10
    hct_calc = (user_input["RBC"] * user_input["MCV"]) / 10
    if abs(user_input["HCT"] - hct_calc) > thresholds.get("cell_gap_limit", 15):
        warnings.append("High discrepancy between HCT and RBC/MCV indices.")
        
    return warnings