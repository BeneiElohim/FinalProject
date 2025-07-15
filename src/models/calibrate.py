from sklearn.calibration import CalibratedClassifierCV

def calibrate_prefit(model, X_val, y_val, method: str = "isotonic"):
    """
    Wrap an already-fitted classifier with a probability-calibration layer.
    Returns a *new fitted model* (CalibratedClassifierCV).
    """
    cal = CalibratedClassifierCV(
        base_estimator=model,
        method=method,
        cv="prefit"
    )
    return cal.fit(X_val, y_val)
