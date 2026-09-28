"""Classical estimator adapters with task-aware prediction and persistence."""
from abc import ABC, abstractmethod
from pathlib import Path
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

class BaseModel(ABC):
    def __init__(self, task="classification"): self.task=task
    @abstractmethod
    def fit(self,X,y): ...
    @abstractmethod
    def predict(self,X): ...
    def predict_proba(self,X):
        if not hasattr(self.estimator,"predict_proba"): raise NotImplementedError("probability output unavailable")
        return self.estimator.predict_proba(X)[:,1]
    def save(self,path,metadata=None):
        Path(path).parent.mkdir(parents=True,exist_ok=True); joblib.dump({"model":self,"metadata":metadata or {}},path)
    @staticmethod
    def load(path): return joblib.load(path)["model"]

class LogisticRegressionModel(BaseModel):
    def __init__(self,**params):
        super().__init__("classification"); defaults={"max_iter":1000,"class_weight":"balanced"}; defaults.update(params); self.estimator=LogisticRegression(**defaults)
    def fit(self,X,y): self.estimator.fit(X,y); return self
    def predict(self,X): return self.estimator.predict(X)

class RandomForestModel(BaseModel):
    def __init__(self,task="classification",**params):
        super().__init__(task); defaults=dict(n_estimators=250,max_depth=None,min_samples_leaf=2,n_jobs=-1,random_state=42)
        defaults.update(params); self.estimator=(RandomForestClassifier(class_weight="balanced_subsample",**defaults) if task=="classification" else RandomForestRegressor(**defaults))
    def fit(self,X,y): self.estimator.fit(X,y); return self
    def predict(self,X): return self.estimator.predict(X)
    @property
    def feature_importances_(self): return self.estimator.feature_importances_

class XGBoostModel(BaseModel):
    def __init__(self,task="classification",**params):
        super().__init__(task)
        try: from xgboost import XGBClassifier,XGBRegressor
        except ImportError as e: raise ImportError("Install the optional XGBoost extra: pip install -e '.[xgboost]'") from e
        defaults=dict(n_estimators=250,max_depth=5,learning_rate=.05,subsample=.85,colsample_bytree=.85,n_jobs=1,random_state=42)
        defaults.update(params); defaults.setdefault("eval_metric","logloss" if task=="classification" else "rmse")
        self.estimator=(XGBClassifier(**defaults) if task=="classification" else XGBRegressor(**defaults))
    def fit(self,X,y): self.estimator.fit(X,y); return self
    def predict(self,X): return self.estimator.predict(X)
    @property
    def feature_importances_(self): return self.estimator.feature_importances_
