import numpy as np
import xgboost as xgb
import lightgbm as lgb

from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, clone


class AveragingModels(BaseEstimator, ClassifierMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        for model in self.models_:
            model.fit(X, y)

        return self
    
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        prediction = np.array(predictions)
        result = [int(np.ceil(np.median(item))) for item in prediction]
        return result
    
    
def get_avgmodel(x, y, seed=10):
    avg_model = AveragingModels(models = (
        RandomForestClassifier(random_state=seed),        
        lgb.LGBMClassifier(random_state=seed),
        xgb.XGBClassifier(random_state=seed),
        AdaBoostClassifier(random_state=seed),
        GradientBoostingClassifier(random_state=seed), 
    ))
    
    avg_model.fit(x, y)
    
    return avg_model