import xgboost as xgb


model = xgb.XGBClassifier()
model.load_model("/sep/models/model_EUR_USD.json")

# Dump to text
model.get_booster().dump_model("/sep/model_dump.txt")
print("Dumped.")
