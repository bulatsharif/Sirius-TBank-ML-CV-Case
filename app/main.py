from fastapi import FastAPI

app = FastAPI(title="Sirius x T-Bank Computer Vision Case")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"status": "ok"}