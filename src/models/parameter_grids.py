from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from imodels import RuleFitClassifier

# =============================================================================
# 1. Strategic Parameters
# =============================================================================
STRATEGIC_PARAMS = [
    {"hold_days": 5, "vol_mult": 0.75},
    {"hold_days": 5, "vol_mult": 1.25},
    {"hold_days": 10, "vol_mult": 1.0},
    {"hold_days": 10, "vol_mult": 1.5},
]

# =============================================================================
# 2. Feature Generation Parameters
# =============================================================================
FEATURE_PARAMS = {
    # Existing params
    "ret_windows": [1, 3, 5, 10, 21],
    "sma_windows": [10, 20, 50],
    "ema_windows": [10, 20, 50],
    "roc_windows": [10, 21],
    "rsi_windows": [7, 14, 21],
    "atr_windows": [7, 14, 21],
    "vol_windows": [20, 60],
    "bb_windows": [20, 40],
    "adx_windows": [7, 14, 21],
    
    # New params for the new features
    "regime_windows": [100, 200],
    "stoch_windows": [14, 28],
    "cmf_windows": [20, 40],
    "vortex_windows": [14, 28],
}

# =============================================================================
# 3. Model Hyperparameter Grids
# =============================================================================
MODEL_PARAMS = {
    "dt": {
        "estimator": DecisionTreeClassifier(random_state=42),
        "grid": {
            "max_depth": [3, 4, 5, 6, 7, 8],
            "min_samples_leaf": [20, 35, 50, 100],
            "criterion": ["gini", "entropy"],
        }
    },
    "xgb": {
        "estimator": XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42
        ),
        "grid": {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 4, 5],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.8, 0.9, 1.0],
            "colsample_bytree": [0.8, 0.9, 1.0],
        }
    },
    "rulefit": {
        "estimator": RuleFitClassifier(include_linear=False, random_state=42),
        "grid": {
            "n_estimators": [50, 100, 150],
            "tree_size": [4, 6, 8],
            "max_rules": [100, 200],
        }
    }
}
