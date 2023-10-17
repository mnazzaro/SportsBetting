import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score

def train_xgb (train: pd.DataFrame, test: pd.DataFrame):
    train_stat_cols = list(set(train.columns) - set(['event', 'bout', 'outcome', 'fighter_red', 'fighter_blue', 'url_red', 'url_blue']))
    model = xgb.XGBClassifier(verbosity=0,
                                reg_lambda=0.023385762997113632,
                                reg_alpha=0.003694895205081855,
                                tree_method="hist",
                                objective="binary:logistic",
                                n_jobs=-1,
                                learning_rate=0.0059107879099318415,
                                min_child_weight=15,
                                max_depth=7,
                                max_delta_step=10,
                                subsample=0.5370056644955932,
                                colsample_bytree=0.5742787613391558,
                                gamma=0.09815563994539223,
                                n_estimators=143,
                                eta=0.1134711359195081,
                                seed=1)
    model.fit(train[train_stat_cols], train['outcome'])
    y_pred = model.predict(test[train_stat_cols])
    accuracy = accuracy_score(test['outcome'], y_pred)
    print(f'Model Accuracy: {accuracy}')

    print (len(train.index))
    print(len(test.index))
    