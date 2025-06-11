from profiler import generate_profiling_report
from preprocess import preprocess_data
from model_selector import detect_task_type
from optimizer import optimize_model
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor


from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import joblib

class AutoMLAgent:
    def __init__(self, dataframe, metric="accuracy"):
        self.df = dataframe
        self.metric = metric
        self.task_type = None
        self.target_column = None
        self.best_model = None

    def run(self):
        print("\n🚀 [1/5] Profiling başlatılıyor...")
        generate_profiling_report(self.df)

        print("\n📌 Hedef değişkeni (target) olarak kullanılacak kolonu belirtin:")
        self.target_column = input("→ Target column ismi: ")

        if self.target_column not in self.df.columns:
            print("❌ Hedef kolon bulunamadı. İşlem iptal.")
            return

        print("\n🧼 [2/5] Preprocessing uygulanıyor...")
        X, y = preprocess_data(self.df, self.target_column)

        print("\n🔍 [3/5] Görev türü belirleniyor...")
        self.task_type = detect_task_type(y)
        print(f"✅ Belirlenen görev: {self.task_type.upper()}")

        print("\n🤖 [4/5] En uygun model aranıyor (Optuna + Cross-Validation)...")

        # Çoklu model listesi
        if self.task_type == "classification":
            models = [
                RandomForestClassifier,
                GradientBoostingClassifier,
                XGBClassifier,
                LGBMClassifier,
                CatBoostClassifier
            ]
        else:
            models = [
                RandomForestRegressor,
                GradientBoostingRegressor,
                XGBRegressor,
                LGBMRegressor,
                CatBoostRegressor
            ]

        self.best_model = optimize_model(models, X, y, self.task_type, self.metric)

        print("\n💾 [5/5] En iyi model 'outputs/best_model.pkl' olarak kaydediliyor...")
        joblib.dump(self.best_model, "outputs/best_model.pkl")

        print("\n✅ AutoML işlemi başarıyla tamamlandı!")
