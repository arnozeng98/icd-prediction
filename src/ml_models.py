from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV

def train_svm(X_train_fused, y_train_main):
    svm = SVC(decision_function_shape='ovr')
    param_distributions = {
        'C': [0.01, 0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
    random_search = RandomizedSearchCV(svm, param_distributions, n_iter=9, cv=3, n_jobs=-1, random_state=42)  # 将 n_iter 设置为参数空间的大小
    random_search.fit(X_train_fused, y_train_main)
    best_svm = random_search.best_estimator_
    return best_svm

def train_rf(X_train_fused, y_train_main):
    rf = RandomForestClassifier()
    param_distributions = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30]
    }
    random_search = RandomizedSearchCV(rf, param_distributions, n_iter=9, cv=3, n_jobs=-1, random_state=42)  # 将 n_iter 设置为参数空间的大小
    random_search.fit(X_train_fused, y_train_main)
    best_rf = random_search.best_estimator_
    return best_rf

def train_xgb(X_train_fused, y_train_main):
    xgb = XGBClassifier()
    param_distributions = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2]
    }
    random_search = RandomizedSearchCV(xgb, param_distributions, n_iter=9, cv=3, n_jobs=-1, random_state=42)  # 将 n_iter 设置为参数空间的大小
    random_search.fit(X_train_fused, y_train_main)
    best_xgb = random_search.best_estimator_
    return best_xgb

def train_stacking_model(X_train_fused, y_train_main):
    # Define base learners
    estimators = [
        ('svm', train_svm(X_train_fused, y_train_main)),
        ('rf', train_rf(X_train_fused, y_train_main)),
        ('xgb', train_xgb(X_train_fused, y_train_main))
    ]

    # Define Stacking model
    stacking_clf = StackingClassifier(estimators=estimators, final_estimator=SVC())

    # Train Stacking model
    stacking_clf.fit(X_train_fused, y_train_main)
    return stacking_clf
