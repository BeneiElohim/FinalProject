from typing import Any, Dict, Tuple
import pandas as pd
from src.models.base_model import BaseModel
from src.models.parameter_grids import MODEL_PARAMS

class RulefitModel(BaseModel):
    @property
    def key(self) -> str:
        return "rulefit"

    def _get_estimator_and_grid(self) -> Tuple[Any, Dict[str, Any]]:
        model_config = MODEL_PARAMS[self.key]
        return model_config["estimator"], model_config["grid"]

    def _extract_rules(self, model: Any, feature_names: list) -> Dict[str, Any]:
        try:
            rules_df = pd.DataFrame(model.rules_)
            if rules_df.empty:
                return {"text": "RuleFit model found no significant rules."}
            if 'importance' in rules_df.columns:
                sort_by_col = 'importance'
                rules_df = rules_df.sort_values(by=sort_by_col, ascending=False)
            elif 'coef' in rules_df.columns:
                sort_by_col = 'coef'
                rules_df['sort_key'] = rules_df[sort_by_col].abs()
                rules_df = rules_df.sort_values(by='sort_key', ascending=False)
            else:
                return {"text": f"No 'importance' or 'coef' column found in rules.\n{rules_df.to_string()}"}

            top_rules = rules_df[rules_df['type'] == 'rule'].head(15)
            
            rules_text_list = [f"Top 15 Rules by {sort_by_col.capitalize()}:"]
            for _, row in top_rules.iterrows():
                rule = row.get('rule')
                value = row.get(sort_by_col)
                rules_text_list.append(f"- IF {rule} (Value: {value:.4f})")
            
            return {"text": "\n".join(rules_text_list)}
        except Exception as e:
            return {"text": f"Could not extract rules: {e}"}