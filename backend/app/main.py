import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from app.routes.chat import router as chat_router


CORS_ORIGINS_ENV = "UNIGUIDE_CORS_ORIGINS"


def get_cors_origins() -> list[str]:
    """Return explicitly allowed browser origins for optional direct API access."""
    configured = os.getenv(CORS_ORIGINS_ENV, "http://localhost:3000")
    return [
        origin.strip().rstrip("/")
        for origin in configured.split(",")
        if origin.strip()
    ]


app = FastAPI(
    title="UniGuide AI API",
    description="Backend API for the UniGuide AI student assistant.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

app.include_router(
    chat_router,
    prefix="/api/v1/chat",
    tags=["Chat"],
)


@app.get("/health", tags=["Health"])
def health_check() -> dict[str, str]:
    return {"status": "ok"}
