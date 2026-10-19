from contextlib import asynccontextmanager
import json
import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from app.api.routes import router
from app.core.config import settings
from app.db import models
from app.db.session import Base, SessionLocal, engine
from app.services.monitoring import MonitoringService


class StructuredFormatter(logging.Formatter):
    def format(self, record):
        payload = {"level": record.levelname.lower(), "logger": record.name, "message": record.getMessage()}
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=True)


logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
for handler in logging.getLogger().handlers:
    handler.setFormatter(StructuredFormatter())
logger = logging.getLogger("sentinelai")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        settings.data_path.parent.mkdir(parents=True, exist_ok=True)
        settings.artifact_path.parent.mkdir(parents=True, exist_ok=True)
        Base.metadata.create_all(bind=engine)
        service = MonitoringService.build()
        with SessionLocal() as session:
            service.seed_database(session)
        app.state.monitoring = service
        app.state.startup_error = None
        logger.info("SentinelAI initialized with %s synthetic records", len(service.data))
    except Exception as exc:
        app.state.monitoring = None
        app.state.startup_error = str(exc)
        logger.exception("SentinelAI initialization failed")
    yield


app = FastAPI(
    title="SentinelAI",
    description="Predictive operations and anomaly intelligence using clearly labeled synthetic demo data.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "PATCH"],
    allow_headers=["Content-Type"],
)
app.include_router(router)


@app.exception_handler(SQLAlchemyError)
async def database_error_handler(_request: Request, _exc: SQLAlchemyError):
    logger.exception("Database operation failed")
    return JSONResponse(status_code=503, content={"detail": "Database is temporarily unavailable"})