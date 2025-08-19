import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.models.engines.decision_tree import DecisionTreeModel
from src.models.engines.rulefit import RulefitModel
from src.models.engines.xgb_surrogate import XgbSurrogateModel
from src.models.engines.gp import GpModel
MODEL_MAP = {
    "dt": DecisionTreeModel,
    "rulefit": RulefitModel,
    "xgb": XgbSurrogateModel,
    "gp": GpModel,
}

def main(args=None):
    """
    This main function is legacy and should not be run directly.
    The primary entry point for the application is `signal_engine.cli`.
    """
    print("[WARN] This `train.py` script is a legacy entry point.")
    print("[WARN] Please use `python -m signal_engine.cli flow` instead.")
    
    if args is None:
        parser = argparse.ArgumentParser(description="Legacy model training pipeline.")
        parser.add_argument("--syms", nargs="+", required=True)
        parser.add_argument("--models", nargs="+", required=True)
        args = parser.parse_args()
    
    print("\nRun complete.")

if __name__ == "__main__":
    main()
