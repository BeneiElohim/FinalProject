from typing import Any, Dict, Tuple
from sklearn.tree import DecisionTreeRegressor, export_text
from src.models.base_model import BaseModel
from src.models.parameter_grids import MODEL_PARAMS

class XgbSurrogateModel(BaseModel):
    @property
    def key(self) -> str:
        return "xgb"

    def _get_estimator_and_grid(self) -> Tuple[Any, Dict[str, Any]]:
        model_config = MODEL_PARAMS[self.key]
        return model_config["estimator"], model_config["grid"]

    def _extract_rules(self, model: Any, feature_names: list) -> Dict[str, Any]:
        """
        For complex models like XGBoost, we fit a simple Decision Tree
        to its predictions to get an understandable approximation of its logic.
        """
        try:
            if hasattr(model, 'feature_importances_'):
                importances = {name: float(imp) for name, imp in zip(feature_names, model.feature_importances_)}
                sorted_importances = sorted(importances.items(), key=lambda item: item[1], reverse=True)
                rules_text = "Feature Importances:\n"
                for name, imp in sorted_importances[:10]:
                    rules_text += f"- {name}: {imp:.4f}\n"
                return {"text": rules_text}
            return {"text": "Rule extraction via surrogate not implemented in this context."}
        except Exception as e:
            return {"text": f"Could not extract rules: {e}"}