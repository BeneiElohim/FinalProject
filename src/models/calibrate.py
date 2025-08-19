from sklearn.calibration import CalibratedClassifierCV

def calibrate_prefit(model, X_val, y_val, method: str = "isotonic"):
    """
    Wrap an already-fitted model with a probability-calibration layer.
    Tries both old (base_estimator) and new (estimator) signatures.
    If calibration fails (e.g. model isn't fully classifier-like), returns the original model.
    """
    try:
        calibrator = CalibratedClassifierCV(estimator=model, method=method, cv="prefit")
    except TypeError:
        calibrator = CalibratedClassifierCV(base_estimator=model, method=method, cv="prefit")
    
    try:
        # fit on your validation set
        return calibrator.fit(X_val, y_val)
    except Exception:
        # fallback: calibration didn’t work, so just return the uncalibrated model
        return model
