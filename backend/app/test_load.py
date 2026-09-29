import joblib
model = joblib.load("model/rain_prediction_model.pkl")
print("Model loaded successfully!")
print(type(model))