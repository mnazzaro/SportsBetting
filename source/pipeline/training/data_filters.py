import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def remove_wmma (df: pd.DataFrame) -> pd.DataFrame:
    return df[~df.filter(like='women').any(axis=1)]

def get_most_predictive_features (train: pd.DataFrame, outcome: pd.DataFrame) -> pd.DataFrame:
    model = 
    feature_importance = {}
    for feature in X_train.columns:
        # Fit the model using only this feature
        model.fit(X_train[[feature]], y_train)
        
        # Make predictions on the test set
        predictions = model.predict_proba(X_test[[feature]])[:, 1]
        
        # Compute the ROC AUC score and store it
        score = roc_auc_score(y_test, predictions)
        feature_importance[feature] = score
