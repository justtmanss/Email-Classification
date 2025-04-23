# app.py
import uvicorn
import os
from api import create_app

# Path to the trained model
MODEL_PATH = os.environ.get("MODEL_PATH", "trained_model.pkl")

# Create FastAPI application
app = create_app(MODEL_PATH)

if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "app:app", 
        host="0.0.0.0", 
        port=int(os.environ.get("PORT", 8000)),
        reload=os.environ.get("DEBUG", "False").lower() == "true"
    )



