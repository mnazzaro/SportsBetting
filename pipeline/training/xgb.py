import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from .get_prediction_data import make_matchup
from .test_against_odds import compare_predictions_to_odds, clean_odds_data

def train_xgb (train: pd.DataFrame, test: pd.DataFrame, matchup: pd.DataFrame):

    # train_stat_cols = list(set(train.columns) - set(['event', 'bout', 'outcome', 'fighter_red', 'fighter_blue', 'url_red', 'url_blue']))
    # correlation_matrix = train[train_stat_cols].corr().abs()
    # upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    # to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)] 
    # train = train.drop(train[to_drop], axis=1)
    # test = test.drop(test[to_drop], axis=1)
    # print (to_drop)
    decisions = set(['ko_tko', 'unanimous_decision', 'split_decision', 'submission', 'dr_stoppage', 'other'])
    train_stat_cols = list(set(train.columns) - set(['event', 'date', 'bout', 'outcome', 'fighter_red', 'fighter_blue', 'url_red', 'url_blue']) -
                           decisions - set(map(lambda x: x + '_direct_difference', decisions)))
                        # - set(to_drop))
    # list(filter(lambda x: x.endswith('_direct_difference'), 
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

    platt_calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
    platt_calibrated_model.fit(train[train_stat_cols], train['outcome'])

    isotonic_calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
    isotonic_calibrated_model.fit(train[train_stat_cols], train['outcome'])

    y_pred = model.predict(test[train_stat_cols])
    print (y_pred)
    print(len(train['outcome']))
    print(train['outcome'].sum())
    print (y_pred.sum())
    print (len(y_pred))
    accuracy = accuracy_score(test['outcome'], y_pred)
    print(f'Model Accuracy: {accuracy}')
    

    # feature_important = model.get_booster().get_score(importance_type='weight')
    # keys = list(feature_important.keys())
    # values = list(feature_important.values())

    # data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)
    # data.nlargest(50, columns="score").plot(kind='barh', figsize = (20,10)) ## plot top 40 features
    # plt.show()

    # print (len(train.index))
    # print(len(test.index))

    # while True:
    #     bout = input("Enter a fight: ")
    #     try:
    #         prediction = model.predict(test[test['bout'] == bout][train_stat_cols])[0]
    #         prob = model.predict_proba(test[test['bout'] == bout][train_stat_cols])
    #         platt_prob = platt_calibrated_model.predict_proba(test[test['bout'] == bout][train_stat_cols])
    #         isotonic_prob = isotonic_calibrated_model.predict_proba(test[test['bout'] == bout][train_stat_cols])
    #         fighter_red = test[test['bout'] == bout]['fighter_red'].iloc[0]
    #         fighter_blue = test[test['bout'] == bout]['fighter_blue'].iloc[0]
    #         if prediction == 1:
    #             answer = fighter_red
    #         else:
    #             answer = fighter_blue
    #         if test[test['bout'] == bout]['outcome'].iloc[0]:
    #             correct = fighter_red
    #         else:
    #             correct = fighter_blue
    #         print (f"The model predicted: {answer} to win with probabilities {prob} (Uncalibrated), {platt_prob} (Platt), {isotonic_prob} (Isotonic Regression). The true winner was: {correct}.\n")
    #     except Exception as e:
    #         print (e)
    #         print (f"Could not find {bout} in test data\n")
    
    # matchup = make_matchup('Jiri Prochazka', 'Alex Pereira', datetime(2023, 11, 11))
    # matchup.to_csv('matchup.csv')
    # matchup[train_stat_cols] = matchup[train_stat_cols].apply(pd.to_numeric)
    prediction = model.predict(matchup[train_stat_cols])[0]
    prob = model.predict_proba(matchup[train_stat_cols])
    platt_prob = platt_calibrated_model.predict_proba(matchup[train_stat_cols])
    isotonic_prob = isotonic_calibrated_model.predict_proba(matchup[train_stat_cols])
    fighter_red = matchup['fighter_red'].iloc[0]
    fighter_blue = matchup['fighter_blue'].iloc[0]

    if prediction == 1:
        answer = fighter_red
    else:
        answer = fighter_blue
    print (f"FIGHTER_RED: {fighter_red}\nFIGHTER_BLUE: {fighter_blue}\nThe model predicted: {answer} to win with probabilities {prob} (Uncalibrated), {platt_prob} (Platt), {isotonic_prob} (Isotonic Regression)")

    print (test.head())
    print ('date' in test.columns)
    test['prediction'] = (platt_calibrated_model.predict_proba(test[train_stat_cols])[:, 1] + \
          isotonic_calibrated_model.predict_proba(test[train_stat_cols])[:, 1]) / 2
    test['winner'] = model.predict(test[train_stat_cols])
    compare_predictions_to_odds(test, clean_odds_data('moneyline_data_at_close.csv'), 
                                2000, 0.1, 0.25)
