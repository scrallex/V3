import xgboost as xgb
import os
import glob
import json

model_dir = "/sep/models"
models = glob.glob(os.path.join(model_dir, "model_*.json"))

print(f"Found {len(models)} models.")

for m_path in models:
    if "feature_builder" in m_path: continue
    if "model_high" in m_path or "model_low" in m_path: continue
    if "gold_standard" in m_path: continue
    
    try:
        clf = xgb.XGBClassifier()
        clf.load_model(m_path)
        
        # Check features
        n_features = clf.get_booster().num_features()
        print(f"{os.path.basename(m_path)}: Expects {n_features} features.")
        
        # Try to inspect feature names if saved
        try:
            fnames = clf.get_booster().feature_names
            if fnames:
                print(f"  Features: {fnames}")
        except:
            pass
            
    except Exception as e:
        print(f"Error loading {m_path}: {e}")
