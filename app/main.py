# app/main.py

from fastapi import FastAPI
from app.routes.environment import router as environment_router
from app.routes.predict import router as predict_router
from app.routes.multiple_predict import router as multiple_predict_router

app = FastAPI(title="Sound Recognition API")

app.include_router(environment_router, prefix="/environment")
app.include_router(predict_router, prefix="/predict")
app.include_router(multiple_predict_router, prefix="/multiple_predict")


@app.get("/")
def root():
    return {"message": "FastAPI is working 🚀"}

