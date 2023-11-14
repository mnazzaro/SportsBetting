import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import clone
import xgboost as xgb
from xgboost import plot_tree
from sklearn.metrics import accuracy_score
from sklearn.calibration import CalibratedClassifierCV, calibration_curve, cross_val_predict, CalibrationDisplay
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import numpy as np
import shap
from datetime import datetime

from .get_prediction_data import make_matchup
from .test_against_odds import compare_predictions_to_odds, clean_odds_data, compare_predictions_to_odds_groupby_date

def plot_calibration_curve(model, X, y, name):
    """
    Plots a calibration curve for the given model and dataset.
    
    :param model: A trained model (like XGBoost) or a model wrapped in CalibratedClassifierCV
    :param X: Feature set
    :param y: Target variable
    """
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # model.fit(X_train, y_train)
    # model = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
    # model.fit(X_train, y_train)
    # Predict probabilities on the test set
    prob_pos = model.predict_proba(X_test)[:, 1]

    cross_val_predict(estimator=model, X=X, y=y, method='predict_proba', cv=5)

    # Calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(y_test, prob_pos, n_bins=10)

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns
    ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label=name)
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    ax1.set_ylabel('Fraction of positives')
    ax1.set_xlabel('Mean predicted probability')
    ax1.set_title('Calibration Plot (Reliability Curve)')

    display = CalibrationDisplay.from_estimator(
        model,
        X,
        y,
        n_bins=10,
        name=name,
        ax=ax2,
        color='red',
    )


    ax2.hist(
        display.y_prob,
        range=(0, 1),
        bins=10,
        label=name,
        color='red',
    )
    plt.legend()
    plt.show()

# Example usage:
# plot_calibration_curve(your_model, X_data, y_data)

def train_xgb (train: pd.DataFrame, test: pd.DataFrame):

    # train_stat_cols = list(set(train.columns) - set(['event', 'bout', 'outcome', 'fighter_red', 'fighter_blue', 'url_red', 'url_blue']))
    # correlation_matrix = train[train_stat_cols].corr().abs()
    # upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    # to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)] 
    # train = train.drop(train[to_drop], axis=1)
    # test = test.drop(test[to_drop], axis=1)
    # print (to_drop)
    # train = train.drop(columns=['wl_percentage_direct_difference'])
    # test = test.drop(columns=['wl_percentage_direct_difference'])
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
    
    temp = train.sample(frac=1).reset_index(drop=True)
    # temp['is_woman'] = temp.filter(like='women').any(axis=1).astype(bool)
    # test = test.filter(like='women').any(axis=1).astype(bool)
    # model.fit(temp[train_stat_cols], temp['outcome'])

    # plot_tree(model)
    # plt.show()

    # explainer = shap.Explainer(model)
    # shap_values = explainer(test[train_stat_cols])

    # shap.summary_plot(shap_values, test[train_stat_cols], max_display=50)


    platt_calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv=5)
    # platt_calibrated_model.fit(train[train_stat_cols], train['outcome'])

    isotonic_calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=5)
    # isotonic_calibrated_model.fit(train[train_stat_cols], train['outcome'])

    y_pred = model.predict(test[train_stat_cols])

    ### CALIBRATION TIME

    plot_calibration_curve(model, train[train_stat_cols], train['outcome'], 'Platt Calibrated')
    # plot_calibration_curve(isotonic_calibrated_model, train[train_stat_cols], train['outcome'])
    # plot_calibration_curve(platt_calibrated_model, train[train_stat_cols], train['outcome'])



    accuracy = accuracy_score(test['outcome'], y_pred)
    print(f'Model Accuracy: {accuracy}')
    

    feature_important = model.get_booster().get_score(importance_type='weight')
    keys = list(feature_important.keys())
    values = list(feature_important.values())

    data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)
    data.nlargest(50, columns="score").plot(kind='barh', figsize = (20,10)) ## plot top 40 features
    plt.show()

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
    

    # test['prediction'] = (platt_calibrated_model.predict_proba(test[train_stat_cols])[:, 1] + \
    #       isotonic_calibrated_model.predict_proba(test[train_stat_cols])[:, 1]) / 2

    def make_mean_prediction (group):
        try:
            pred1 = group.iloc[0]['pre_prediction']
            pred2 = group.iloc[1]['pre_prediction']
            group['prediction'] = [np.mean([pred1, 1 - pred2]), np.mean([1 - pred1, pred2])]
            return group
        except Exception as e:
            group['prediction'] = [group.iloc[0]['pre_prediction']]
            return group
    

    test['pre_prediction'] = list(platt_calibrated_model.predict_proba(test[train_stat_cols])[:, 1])
    # test['pre_prediction_iso'] = list(isotonic_calibrated_model.predict_proba(test[train_stat_cols])[:, 1])
    # test['pre_prediction'] = (test['pre_prediction_platt'] + test['pre_prediction_iso']) / 2
    test = test.groupby(['event', 'bout'], group_keys=False).apply(make_mean_prediction)

    test['winner'] = test['prediction'].map(lambda x: int(round(x)))
    # test['winner'] = model.predict(test[train_stat_cols])
    compare_predictions_to_odds_groupby_date(test, clean_odds_data('moneyline_data_at_close.csv'), 
                                2000, 0.03, 0.3)

    return train_stat_cols, platt_calibrated_model

def train_xgb_all (train: pd.DataFrame):

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
    
    temp = train.sample(frac=1).reset_index(drop=True)
    model.fit(temp[train_stat_cols], temp['outcome'])

    # plot_tree(model)
    # plt.show()

    # platt_calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv=5)
    # platt_calibrated_model.fit(temp[train_stat_cols], temp['outcome'])

    isotonic_calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=5)
    isotonic_calibrated_model.fit(temp[train_stat_cols], temp['outcome'])

    # plot_calibration_curve(model, train[train_stat_cols], train['outcome'], 'Platt Calibrated')
    # plot_calibration_curve(isotonic_calibrated_model, train[train_stat_cols], train['outcome'])
    plot_calibration_curve(isotonic_calibrated_model, temp[train_stat_cols], temp['outcome'], 'Isotonic Calibrated Model')
    

    # feature_important = model.get_booster().get_score(importance_type='weight')
    # keys = list(feature_important.keys())
    # values = list(feature_important.values())

    # data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)
    # data.nlargest(50, columns="score").plot(kind='barh', figsize = (20,10)) ## plot top 40 features
    # plt.show()

    print (len(train.index))

    return train_stat_cols, isotonic_calibrated_model