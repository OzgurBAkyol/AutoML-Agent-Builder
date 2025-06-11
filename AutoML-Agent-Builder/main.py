# main.py

import pandas as pd
from agent import AutoMLAgent

def main():
    print("🔍 AutoML-Agent-Builder başlatılıyor...\n")

    # 1. Veri dosyasını al
    file_path = input("Lütfen veri dosyasının yolunu girin (örn: data/sample.csv): ")

    try:
        df = pd.read_csv(file_path)
        print(f"✅ Veri başarıyla yüklendi. Şekil: {df.shape}")
    except Exception as e:
        print(f"❌ Veri okunamadı: {e}")
        return

    # 2. Kullanıcıdan metrik girilmesini iste
    print("\n📏 Kullanmak istediğiniz değerlendirme metriğini girin:")
    print("📌 Örnekler: accuracy, f1, roc_auc, r2, neg_root_mean_squared_error")
    metric = input("→ Skor metriği: ").strip()

    # 3. Agent'i başlat ve çalıştır
    agent = AutoMLAgent(df, metric=metric)
    agent.run()

if __name__ == "__main__":
    main()
