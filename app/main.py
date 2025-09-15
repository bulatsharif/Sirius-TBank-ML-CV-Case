from fastapi import FastAPI
from app.api.routers import predict

app = FastAPI(title="Sirius x T-Bank Computer Vision Case")

app.include_router(predict.router, prefix="/api/v1", tags=["detection"])

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"status": "ok"}