import os
import pickle

print("Current directory:", os.getcwd())

try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    print("✅ Model loaded successfully!")
except FileNotFoundError:
    print("❌ model.pkl not found. Check the path.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
