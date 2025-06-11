# optimizer.py

import optuna
from sklearn.model_selection import cross_val_score
import numpy as np

def optimize_model(models: list, X, y, task_type: str, metric: str):
    best_score = None
    best_model = None
    best_model_class = None

    for model_class in models:
        def objective(trial):
            # 👇 Model'e özel parametre alanı
            if "RandomForest" in model_class.__name__:
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                    "max_depth": trial.suggest_int("max_depth", 3, 20),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                }
            elif "GradientBoosting" in model_class.__name__:
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                    "max_depth": trial.suggest_int("max_depth", 3, 20),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                }
            else:
                params = {}  # fallback/default (ileride uyarı bastırabiliriz)

            model = model_class(**params)
            scores = cross_val_score(model, X, y, cv=3, scoring=metric)
            return scores.mean()

        direction = "maximize" if not metric.startswith("neg_") else "minimize"
        print(f"\n⚙️ {model_class.__name__} modeli optimize ediliyor... [{metric}]")
        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=15, show_progress_bar=False)

        print(f"📊 {model_class.__name__} → En iyi skor: {study.best_value:.4f}")

        if best_score is None or (
            direction == "maximize" and study.best_value > best_score
        ) or (
            direction == "minimize" and study.best_value < best_score
        ):
            best_score = study.best_value
            best_model_class = model_class
            best_model = model_class(**study.best_params)

    print(f"\n🏆 En iyi model: {best_model_class.__name__} → Skor: {best_score:.4f}")
    best_model.fit(X, y)
    return best_model
