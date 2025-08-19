from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
import json

from src.utils.windows import build_walkforward_windows
from src.models.calibrate import calibrate_prefit
from src.models.db import Strategy, Backtest, generate_strategy_id
from src.models.backtest_utils import run_backtest
from src.models.config import est_cost
from src.models.parameter_grids import MODEL_PARAMS

MIN_TRADES_FOR_STRATEGY = 5

class BaseModel(ABC):
    @property
    @abstractmethod
    def key(self) -> str:
        pass

    def __init__(self, symbol: str, features: pd.DataFrame, price: pd.Series, target: pd.Series):
        self.symbol = symbol
        self.model_type = self.key
        self.X = features.loc[target.index]
        self.price = price.reindex(self.X.index)
        self.y = target

    @abstractmethod
    def _get_estimator_and_grid(self) -> Tuple[Any, Dict[str, Any]]:
        pass

    @abstractmethod
    def _extract_rules(self, model: Any, feature_names: list) -> Dict[str, Any]:
        pass

    def run_pipeline(self, db: Session):
        print(f"--- Running pipeline for {self.model_type.upper()} on {self.symbol} ---")
        if len(self.X) < 500 or self.y.nunique() < 2:
            print(f"  [SKIP] Insufficient data for {self.symbol}.")
            return

        windows = build_walkforward_windows(self.X.index)
        if not windows:
            print(f"  [SKIP] No valid walk-forward windows for {self.symbol}.")
            return

        all_oos_signals = pd.DataFrame(index=self.X.index)
        all_oos_signals['entries'] = False
        all_oos_signals['exits'] = False
        latest_window_artifacts = {}

        for t0, t1, v0, v1, test_start, test_end in windows:
            tr_mask = (self.X.index >= t0) & (self.X.index <= v1)
            va_mask = (self.X.index >= v0) & (self.X.index <= v1)
            test_mask = (self.X.index >= test_start) & (self.X.index <= test_end)

            if va_mask.sum() < 60 or test_mask.sum() < 1: continue

            X_tr, y_tr = self.X.loc[tr_mask], self.y.loc[tr_mask]
            X_va, y_va = self.X.loc[va_mask], self.y.loc[va_mask]
            X_test = self.X.loc[test_mask]

            print(f"  Training window: {t0.date()} to {v1.date()}, Testing: {test_start.date()} to {test_end.date()}...")
            
            model, artifacts = self.train_window(X_tr, y_tr)
            
            if model is None:
                print("    Skipping window due to training failure.")
                continue

            calibrated_model = calibrate_prefit(model, X_va, y_va)
            best_long_thresh, best_flat_thresh = self._find_best_thresholds(calibrated_model, X_va, y_va)
            artifacts['prob_thresholds'] = (best_long_thresh, best_flat_thresh)
            print(f"    Found best thresholds: Long >= {best_long_thresh:.2f}, Flat <= {best_flat_thresh:.2f}")

            entries, exits = self.predict_signals(calibrated_model, X_test, artifacts)
            all_oos_signals.loc[X_test.index, 'entries'] = entries
            all_oos_signals.loc[X_test.index, 'exits'] = exits
            
            latest_window_artifacts = artifacts
            latest_window_artifacts['model_object'] = calibrated_model

        if not latest_window_artifacts:
            print(f"  [FAIL] No windows were successfully trained for {self.symbol}.")
            return

        final_entries = all_oos_signals['entries']
        final_exits = all_oos_signals['exits']
        atr_col = next((col for col in self.X.columns if 'atr' in col), None)
        atr14 = self.X[atr_col] if atr_col else pd.Series(0.01, index=self.price.index)
        fees = est_cost(atr14, self.price)
        result = run_backtest(self.symbol, self.price, final_entries, final_exits, fees=fees, benchmark=self.price)
        metrics = result.get("metrics", {})
        num_trades = metrics.get('num_trades', 0)
        print(f"  Backtest complete. Sharpe: {metrics.get('sharpe_ratio', 0):.2f}, Trades: {num_trades}")
        if num_trades < MIN_TRADES_FOR_STRATEGY:
            print(f"  [DISCARD] Strategy generated only {num_trades} trades (min is {MIN_TRADES_FOR_STRATEGY}). Not saving.")
            return
        strategy_def = {
            "rules": latest_window_artifacts.get('rules', {}),
            "hyperparameters": latest_window_artifacts.get('hyperparameters', {})
        }
        if 'hyperparameters' in strategy_def:
            strategy_def['hyperparameters'] = {k: str(v) for k, v in strategy_def['hyperparameters'].items()}
        strategy_id = generate_strategy_id(self.symbol, self.model_type, **strategy_def)
        existing_strat = db.query(Strategy).filter(Strategy.id == strategy_id).first()
        if existing_strat:
            db.delete(existing_strat)
            db.commit()
        strategy = Strategy(id=strategy_id, symbol=self.symbol, model_type=self.model_type, **strategy_def)
        db.add(strategy)
        db.commit()
        backtest = Backtest(
            strategy_id=strategy.id, sharpe_ratio=metrics.get('sharpe_ratio'), max_drawdown=metrics.get('max_drawdown'),
            annual_return=metrics.get('annual_return'), win_rate=metrics.get('win_rate'),
            num_trades=metrics.get('num_trades'), metrics=metrics,
            equity_curve=result.get('equity_curve'), trade_log=result.get('trade_log'),
            test_start=self.price.index.min(), test_end=self.price.index.max(),
        )
        db.add(backtest)
        db.commit()
        print(f"  [OK] Saved strategy {strategy_id} to database.")

    def train_window(self, X_tr: pd.DataFrame, y_tr: pd.Series) -> Tuple[Any, Dict[str, Any]]:
        estimator, grid = self._get_estimator_and_grid()
        
        scoring_metric = "accuracy" if self.model_type == "rulefit" else "roc_auc"
        print(f"    Tuning {self.model_type.upper()} with RandomizedSearchCV (scoring: {scoring_metric})...")
        try:
            search = RandomizedSearchCV(
                estimator, param_distributions=grid, n_iter=10, cv=3,
                scoring=scoring_metric, n_jobs=-1, random_state=42, verbose=0
            )
            search.fit(X_tr, y_tr)
            best_model = search.best_estimator_
            print(f"    Best params: {search.best_params_}")
            artifacts = {
                "hyperparameters": search.best_params_,
                "rules": self._extract_rules(best_model, list(X_tr.columns))
            }
            return best_model, artifacts
        except Exception as e:
            print(f"    [FAIL] Tuning failed for {self.model_type.upper()}: {e}")
            return None, {}

    def _find_best_thresholds(self, model: Any, X_va: pd.DataFrame, y_va: pd.Series) -> Tuple[float, float]:
        if not hasattr(model, "predict_proba"): return 0.5, 0.5
        proba = model.predict_proba(X_va)[:, 1]
        best_acc = -1
        best_thresh = (0.6, 0.4)
        for hi in np.arange(0.55, 0.8, 0.05):
            for lo in np.arange(0.2, 0.45, 0.05):
                if lo >= hi: continue
                preds = np.full(len(y_va), -1)
                preds[proba >= hi] = 1
                preds[proba <= lo] = 0
                mask = preds != -1
                if not np.any(mask): continue
                acc = accuracy_score(y_va[mask], preds[mask])
                if acc > best_acc:
                    best_acc = acc
                    best_thresh = (hi, lo)
        return best_thresh

    def predict_signals(self, model: Any, X_test: pd.DataFrame, artifacts: dict) -> Tuple[pd.Series, pd.Series]:
        entries = pd.Series(False, index=X_test.index)
        exits = pd.Series(False, index=X_test.index)
        
        if self.model_type == 'gp':
            feature_args = {f.replace('-', '_').replace(':', '_'): X_test[f].values for f in X_test.columns}
            # The compiled GP function can return floats, ints, or booleans. We must be explicit.
            raw_entries = model(**feature_args)
            entries = np.asarray(raw_entries, dtype=bool)
            exits = ~entries
            return pd.Series(entries, index=X_test.index), pd.Series(exits, index=X_test.index)
        
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)[:, 1]
            long_thresh, flat_thresh = artifacts.get("prob_thresholds", (0.6, 0.4))
            entries = proba >= long_thresh
            exits = proba <= flat_thresh
        return pd.Series(entries, index=X_test.index), pd.Series(exits, index=X_test.index)