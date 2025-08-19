from typing import Any, Dict, Tuple
import random
import pandas as pd
import numpy as np
from deap import base, creator, gp, tools, algorithms
from sklearn.metrics import accuracy_score

from src.models.base_model import BaseModel

# --- GP Primitives (Helper functions) ---
def _safe_div(a, b): return np.divide(a, b + 1e-9)

class GpModel(BaseModel):
    @property
    def key(self) -> str:
        return "gp"

    def __init__(self, symbol: str, features: pd.DataFrame, price: pd.Series, target: pd.Series):
        super().__init__(symbol, features, price, target)
        self.pset = self._setup_pset()
        self._setup_deap_types()
        self.toolbox = self._setup_toolbox()

    def _get_estimator_and_grid(self) -> Tuple[Any, Dict[str, Any]]:
        return None, {}

    def _extract_rules(self, model: Any, feature_names: list) -> Dict[str, Any]:
        return {"text": "See strategy definition for the evolved tree structure."}

    def train_window(self, X_tr: pd.DataFrame, y_tr: pd.Series) -> Tuple[Any, Dict[str, Any]]:
        # DEAP cannot handle certain characters in feature names. Sanitize them.
        self.feature_names = [f.replace('-', '_').replace(':', '_') for f in X_tr.columns]
        
        # Re-register the evaluation function with the correct (sanitized) feature names for this window
        self.toolbox.register("evaluate", self._evaluate, X_tr=X_tr, y_tr=y_tr)
        
        pop = self.toolbox.population(n=100)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("max", np.max)
        
        print(f"    Evolving GP population for {self.symbol}...")
        algorithms.eaSimple(pop, self.toolbox, 0.7, 0.2, 20, stats=stats, halloffame=hof, verbose=False)
                
        best_individual = hof[0]
        model_object = self.toolbox.compile(expr=best_individual)
        
        artifacts = {
            "hyperparameters": {"population": 100, "generations": 20, "cxpb": 0.7, "mutpb": 0.2},
            "rules": {"text": str(best_individual)}
        }
        return model_object, artifacts

    def _evaluate(self, individual, X_tr: pd.DataFrame, y_tr: pd.Series):
        try:
            func = self.toolbox.compile(expr=individual)
            feature_args = {name: X_tr.iloc[:, i].values for i, name in enumerate(self.feature_names)}
            pred = func(**feature_args)
            return (accuracy_score(y_tr, pred > 0),)
        except Exception:
            return (-1.0,)

    def _setup_pset(self) -> gp.PrimitiveSet:
        feature_names = [f.replace('-', '_').replace(':', '_') for f in self.X.columns]
        pset = gp.PrimitiveSet("MAIN", len(feature_names))
        pset.renameArguments(**{f"ARG{i}": name for i, name in enumerate(feature_names)})

        pset.addPrimitive(np.add, 2)
        pset.addPrimitive(np.subtract, 2)
        pset.addPrimitive(np.multiply, 2)
        pset.addPrimitive(_safe_div, 2)
        pset.addPrimitive(np.logical_and, 2)
        pset.addPrimitive(np.logical_or, 2)
        pset.addPrimitive(np.logical_not, 1)
        pset.addPrimitive(np.greater, 2)
        pset.addPrimitive(np.less, 2)
        
        pset.addEphemeralConstant("rand", lambda: random.uniform(-1, 1))

        return pset

    def _setup_deap_types(self):
        if not hasattr(creator, "FitnessGP"):
            creator.create("FitnessGP", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessGP)

    def _setup_toolbox(self) -> base.Toolbox:
        toolbox = base.Toolbox()
        toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=3)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=self.pset)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=self.pset)
        toolbox.decorate("mate", gp.staticLimit(key=len, max_value=90))
        toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=90))
        return toolbox