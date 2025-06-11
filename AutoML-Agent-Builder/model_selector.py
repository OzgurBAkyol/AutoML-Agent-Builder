# model_selector.py

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def detect_task_type(y):
    """
    Hedef değişkenin yapısına göre classification mı regression mı olduğunu otomatik belirler.
    """
    if y.dtype == "object" or len(set(y)) < 30:
        return "classification"
    else:
        return "regression"

def select_model(task_type):
    """
    Görev türüne uygun sklearn modeli döner.
    """
    if task_type == "classification":
        print("🎯 RandomForestClassifier seçildi.")
        return RandomForestClassifier()
    elif task_type == "regression":
        print("🎯 RandomForestRegressor seçildi.")
        return RandomForestRegressor()
    else:
        raise ValueError("❌ Geçersiz görev türü! 'classification' veya 'regression' bekleniyor.")
