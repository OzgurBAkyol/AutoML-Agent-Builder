# optimizer.py

import optuna
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np

def optimize_model(model, X, y, task_type):
    def objective(trial):
        # Parametre aralıkları
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        }

        if task_type == "classification":
            clf = RandomForestClassifier(**params)
            score = cross_val_score(clf, X, y, cv=3, scoring="accuracy").mean()
        else:
            clf = RandomForestRegressor(**params)
            score = cross_val_score(clf, X, y, cv=3, scoring="neg_root_mean_squared_error").mean()

        return score

    print("⚙️ Optuna ile en iyi parametreler aranıyor...")
    study = optuna.create_study(direction="maximize" if task_type == "classification" else "minimize")
    study.optimize(objective, n_trials=20, show_progress_bar=True)

    print(f"🏁 En iyi skor: {study.best_value}")
    print(f"🔧 En iyi parametreler: {study.best_params}")

    best_params = study.best_params
    if task_type == "classification":
        best_model = RandomForestClassifier(**best_params)
    else:
        best_model = RandomForestRegressor(**best_params)

    best_model.fit(X, y)
    return best_model
