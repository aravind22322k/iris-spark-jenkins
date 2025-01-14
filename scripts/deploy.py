from fastapi import FastAPI
import numpy as np
from xgboost import XGBClassifier
import json

app = FastAPI()

# Load the trained model
model = XGBClassifier()
model.load_model("models/xgboost_model.json")

@app.post("/predict/")
async def predict(features: list):
    try:
        prediction = model.predict(np.array(features).reshape(1, -1))
        return {"prediction": int(prediction[0])}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    print("Fast api is running")
