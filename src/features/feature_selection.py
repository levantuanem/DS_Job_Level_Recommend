import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif

def select_features(X_train, y_train, X_test, k=1000,):
    k = min(k, X_train.shape[1])
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    return ( X_train_selected, X_test_selected, selector)

def get_selected_feature_names(selector, feature_names):
    feature_names = pd.Series(feature_names)
    mask = selector.get_support()
    return feature_names[mask].tolist()