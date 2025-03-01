from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

def train_svm(X_train_fused, y_train_main):
    svm = SVC(decision_function_shape='ovr')
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto'],
        'coef0': [0.0, 0.1, 1.0]
    }
    grid_search = GridSearchCV(svm, param_grid, cv=3, n_jobs=-1, scoring='accuracy')
    grid_search.fit(X_train_fused, y_train_main)
    best_svm = grid_search.best_estimator_
    return best_svm
