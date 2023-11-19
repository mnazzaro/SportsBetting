from typing import Dict, Any

import pandas as pd
import xgboost as xgb

from .base_model import BaseModel, Metric

class XGBoostModel (BaseModel):
    def __init__ (self, **params):
        super().__init__(xgb.XGBClassifier, 'XGBoost', **params)

    def find_optimal_hyperparameters(self) -> Dict[str, Any]:
        """ Unimplemented """

    @Metric('most important feature')
    def _most_important_feature (self, X_test: pd.DataFrame, y_test: pd.DataFrame):
        feature_importance = self.model.get_booster().get_score(importance_type='weight')
        keys = list(feature_importance.keys())
        values = list(feature_importance.values())
        data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by="score", ascending=False)
        return data.iloc[0].index.name
