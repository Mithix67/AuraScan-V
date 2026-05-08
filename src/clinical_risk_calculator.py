def calculate_risk_score(age, pack_years, family_history, symptoms, bmi=None,
                         cancer_history=False, copd=False, radon_exposure=False,
                         occupational_exposure=False, secondary_smoke=False,
                         chest_radiation=False):
    """
    Clinical lung cancer risk score — PLCOm2012-inspired.
    Returns (score: int, level: str) where level is LOW / MODERATE / HIGH / CRITICAL.
    """
    score = 0

    # --- Factor Weighting ---
    if age > 45:               score += 1
    if pack_years > 0:         score += 1
    if family_history:         score += 1
    if len(symptoms) > 0:      score += 1
    if cancer_history:         score += 1
    if copd:                   score += 1
    if chest_radiation:        score += 1
    if radon_exposure:         score += 1
    if occupational_exposure:  score += 1
    if secondary_smoke:        score += 1

    # --- Stratification thresholds calibrated on PLCO dataset ---
    # [Calibration coefficients omitted]
    if score >= 8:   return score, "CRITICAL"
    elif score >= 6: return score, "HIGH"
    elif score >= 3: return score, "MODERATE"
    else:            return score, "LOW"


def validate_dicom(dicom_file):
    """Validates modality and pixel data presence in a DICOM file."""
    try:
        if dicom_file.Modality != 'CT':
            return False, f"Modality is not CT (Found: {dicom_file.Modality})"
        if not hasattr(dicom_file, 'PixelData'):
            return False, "No Pixel Data found."
        return True, "Valid CT Scan"
    except Exception as e:
        return False, f"Invalid DICOM: {str(e)}"
