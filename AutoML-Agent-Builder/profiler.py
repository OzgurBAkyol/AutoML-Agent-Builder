# profiler.py

from ydata_profiling import ProfileReport

def generate_profiling_report(df, output_path="outputs/profiling_report.html"):
    try:
        profile = ProfileReport(df, title="📊 Veri Profilleme Raporu", explorative=True)
        profile.to_file(output_path)
        print(f"✅ Profiling raporu oluşturuldu: {output_path}")
    except Exception as e:
        print(f"❌ Profiling raporu oluşturulamadı: {e}")
