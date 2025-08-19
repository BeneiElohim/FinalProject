from typing import Any, Dict, Tuple
from sklearn.tree import export_text
from src.models.base_model import BaseModel
from src.models.parameter_grids import MODEL_PARAMS

class DecisionTreeModel(BaseModel):
    @property
    def key(self) -> str:
        return "dt"
    
    def _get_estimator_and_grid(self) -> Tuple[Any, Dict[str, Any]]:
        model_config = MODEL_PARAMS[self.key]
        return model_config["estimator"], model_config["grid"]

    def _extract_rules(self, model: Any, feature_names: list) -> Dict[str, Any]:
        try:
            rules = export_text(model, feature_names=feature_names)
            return {"text": rules}
        except Exception as e:
            return {"text": f"Could not extract rules: {e}"}