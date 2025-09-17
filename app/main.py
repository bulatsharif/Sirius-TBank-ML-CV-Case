from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY
import logging
from app.api.routers import predict
from app.schemas.predict import ErrorResponse
from contextlib import asynccontextmanager
from app.models.detect_YOLO import _get_model

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _ = _get_model()
    except Exception:
        # Модель может не загрузиться;
        pass
    yield

app = FastAPI(title="Sirius x T-Bank Computer Vision Case", lifespan=lifespan)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app.include_router(predict.router, prefix="/api/v1", tags=["detection"])

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(error="HTTPException", detail=str(exc.detail)).model_dump(),
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning("Validation error: %s", exc)
    return JSONResponse(
        status_code=HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(error="ValidationError", detail=str(exc)).model_dump(),
    )

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(error="InternalServerError", detail="Unexpected error").model_dump(),
    )

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"status": "ok"}