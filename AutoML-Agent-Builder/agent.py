# agent.py

from profiler import generate_profiling_report
from preprocess import preprocess_data
from model_selector import detect_task_type, select_model
from optimizer import optimize_model

import joblib

class AutoMLAgent:
    def __init__(self, dataframe):
        self.df = dataframe
        self.task_type = None
        self.target_column = None
        self.model = None
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

        print("\n🤖 [4/5] Model seçiliyor ve optimize ediliyor...")
        self.model = select_model(self.task_type)
        self.best_model = optimize_model(self.model, X, y, self.task_type)

        print("\n💾 [5/5] En iyi model 'outputs/best_model.pkl' olarak kaydediliyor...")
        joblib.dump(self.best_model, "outputs/best_model.pkl")

        print("\n✅ AutoML işlemi başarıyla tamamlandı!")

