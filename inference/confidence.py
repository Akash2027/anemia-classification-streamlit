def determine_confidence_tier(final_probs, violation_detected, plausibility_warnings):
    """
    Evaluates the 'trustworthiness' of the final prediction.
    """
    max_prob = max(final_probs.values())
    
    # Logic for Confidence Tiers
    if max_prob > 0.85 and not violation_detected and len(plausibility_warnings) == 0:
        tier = "High"
        message = "Consistent physiological patterns detected."
    elif max_prob > 0.60 and not violation_detected:
        tier = "Moderate"
        message = "Probable diagnosis, but some overlapping indices observed."
    elif violation_detected or len(plausibility_warnings) > 0:
        tier = "Low / Suspicious"
        message = "Physiological contradictions detected. Clinical correlation required."
    else:
        tier = "Ambiguous"
        message = "Data does not align clearly with a specific anemia profile."

    return tier, message

def generate_final_output(final_probs, violation_detected, plausibility_warnings):
    """
    Constructs the final JSON object for the user/API.
    """
    # Get the diagnosis with the highest adjusted probability
    predicted_diagnosis = max(final_probs, key=final_probs.get)
    
    # Determine the confidence tier based on inference flags
    confidence_tier, reason = determine_confidence_tier(
        final_probs, violation_detected, plausibility_warnings
    )

    # Build the structured output
    output = {
        "diagnosis": predicted_diagnosis,
        "confidence": {
            "tier": confidence_tier,
            "score": round(max(final_probs.values()), 4),
            "logic_flags": reason
        },
        "probabilities": {k: round(v, 4) for k, v in final_probs.items()},
        "warnings": plausibility_warnings,
        "disclaimer": "This is a decision support tool, not a clinical diagnosis. Consult a hematologist."
    }

    return output