from typing import Dict, Any, Callable
import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, 
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    matthews_corrcoef,
    precision_recall_curve,
    cohen_kappa_score
)

from .exceptions import UnfittedModelError

class MetricReportType (type):
    def __init__ (cls, name, bases, attrs):
        cls._metrics = {}
        for name, method in attrs.items():
            if hasattr (method, 'metric_name'):
                cls._metrics[method.metric_name] = method

class Metric:
    def __init__(self, metric_name: str, is_figure=False, func: Callable = None):
        self.metric_name = metric_name
        self.is_figure = is_figure
        self.func = func

    def __call__(self, func):
        if self.func is None:
            self.func = func
        return self

    def __get__ (self, instance, owner=None):
        def wrapper (*args, **kwargs):
            if not instance.fitted:
                raise UnfittedModelError(f"{instance.name} model has not been fit")
            return self.func(instance, *args, **kwargs)
        return wrapper

class BaseModel (metaclass=MetricReportType):
    def __init__ (self, model_class: type, name: str, **supplied_params):
        self.model_class = model_class
        self.name = name
        self.model = None
        self.fitted = False
        if supplied_params:
            self.set_model(**supplied_params)
    
    @property
    def metrics (self) -> Dict[str, Callable]:
        metrics = {}
        for t in type(self).__mro__:
            if hasattr(t, '_metrics'):
                for name, metric in t._metrics.items():
                    metrics[name] = metric
        return metrics

    @abstractmethod
    def find_optimal_hyperparameters (self) -> Dict[str, Any]:
        """ e.g. Optuna tuning for xgb. Not applicable to all models """
        ...

    def set_model (self, **params):
        self.model = self.model_class(**params)

    def fit (self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        if not self.model:
            raise UnfittedModelError (f"{self.name} model has not been set yet")
        if self.fitted:
            logging.warn(f"{self.name} model has already been fit")
        self.model.fit(X_train, y_train)
        self.fitted = True

    def predict (self, X_test: pd.DataFrame) -> np.ndarray:
        if not self.model:
            raise UnfittedModelError (f"{self.name} model has not been fit")
        return self.model.predict(X_test)

    def report (self, X_test: pd.DataFrame, y_test: pd.DataFrame):
        for metric_name, metric in self.metrics.items():
            print (f'{self.name} {metric_name}: {metric.__get__(self)(X_test, y_test)}')

    @Metric('Accuracy')
    def _get_accuracy (self, X_test: pd.DataFrame, y_test: pd.DataFrame) -> float: 
        return accuracy_score(y_test, self.predict(X_test))
    
    @Metric('Precision')
    def _get_precision (self, X_test: pd.DataFrame, y_test: pd.DataFrame) -> float:
        return precision_score(y_test, self.predict(X_test))
    
    @Metric('Recall')
    def _get_recall (self, X_test: pd.DataFrame, y_test: pd.DataFrame) -> float:
        return recall_score(y_test, self.predict(X_test))
    
    @Metric('F1 Score')
    def _get_f1_score (self, X_test: pd.DataFrame, y_test: pd.DataFrame) -> float:
        return f1_score(y_test, self.predict(X_test))
    
    @Metric('ROC AUC')
    def _get_auc (self, X_test: pd.DataFrame, y_test: pd.DataFrame) -> float:
        return roc_auc_score(y_test, self.predict(X_test))
    
    @Metric('MCC')
    def _get_mcc (self, X_test: pd.DataFrame, y_test: pd.DataFrame) -> float:
        return matthews_corrcoef(y_test, self.predict(X_test))
    
    @Metric('Cohen\'s Kappa')
    def _get_cohen_kappa (self, X_test: pd.DataFrame, y_test: pd.DataFrame) -> float:
        return cohen_kappa_score(y_test, self.predict(X_test))
    
    # @Metric('Confusion Matrix', is_figure=True)
    # def _get_confusion_matrix (self, X_test: pd.DataFrame, y_test: pd.DataFrame) -> plt.Axes:
        
