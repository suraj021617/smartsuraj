"""
⚡ CACHE WARMER - Pre-build OCR cache on startup
Runs in background to ensure instant predictions
"""
import pandas as pd
import os
import sys
from datetime import datetime

def warm_ocr_cache():
    """Pre-build OCR cache for all providers"""
    try:
        # Import after path setup
        from auto_ocr_predictor import learn_ocr_patterns
        
        # Load data
        csv_paths = ['4d_results_history.csv', '../4d_results_history.csv']
        df = None
        
        for csv_path in csv_paths:
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path, on_bad_lines='skip', encoding='utf-8', 
                               encoding_errors='ignore', dtype=str, keep_default_na=False)
                if not df.empty:
                    break
        
        if df is None or df.empty:
            print("❌ No data found for cache warming")
            return
        
        # Parse dates
        df['date_parsed'] = pd.to_datetime(df['date'], errors='coerce', infer_datetime_format=True)
        df = df[df['date_parsed'].notna()].copy()
        
        if df.empty:
            print("❌ No valid dates in data")
            return
        
        print(f"⏳ Warming cache for {len(df)} rows...")
        start = datetime.now()
        
        # Warm cache for 'all' provider
        patterns = learn_ocr_patterns(df, 'all')
        print(f"✅ Cached 'all' provider: {len(patterns)} patterns")
        
        # Warm cache for top providers
        providers = df['provider'].value_counts().head(5).index.tolist()
        for provider in providers:
            if provider and str(provider).strip():
                provider_df = df[df['provider'] == provider]
                patterns = learn_ocr_patterns(provider_df, provider)
                print(f"✅ Cached '{provider}': {len(patterns)} patterns")
        
        elapsed = (datetime.now() - start).total_seconds()
        print(f"🎉 Cache warming complete in {elapsed:.2f}s")
        print(f"💾 Next load will be INSTANT!")
        
    except Exception as e:
        print(f"❌ Cache warming failed: {e}")

if __name__ == "__main__":
    warm_ocr_cache()
