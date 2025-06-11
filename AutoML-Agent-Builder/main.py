# main.py

import pandas as pd
from agent import AutoMLAgent

def main():
    print("🔍 AutoML-Agent-Builder başlatılıyor...\n")

    # Kullanıcıdan dosya yolu al
    file_path = input("Lütfen veri dosyasının yolunu girin (örn: data/sample.csv): ")

    try:
        df = pd.read_csv(file_path)
        print(f"✅ Veri başarıyla yüklendi. Şekil: {df.shape}")
    except Exception as e:
        print(f"❌ Veri okunamadı: {e}")
        return

    # Agent'i başlat
    agent = AutoMLAgent(df)
    agent.run()

if __name__ == "__main__":
    main()
