import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from app.routes.chat import router as chat_router


CORS_ORIGINS_ENV = "UNIGUIDE_CORS_ORIGINS"


def get_cors_origins() -> list[str]:
    """Return explicitly allowed browser origins for optional direct API access."""
    configured = os.getenv(CORS_ORIGINS_ENV, "http://localhost:3000")
    if configured.strip() == "*":
        return ["*"]
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


@app.middleware("http")
async def normalize_vercel_paths(request: Request, call_next):
    """Restore original path if Vercel serverless proxy mangles it."""
    matched_path = request.headers.get("x-matched-path")
    if matched_path and matched_path != request.scope.get("path"):
        request.scope["path"] = matched_path

    path = request.scope.get("path", "")
    for prefix in ("/api/index.py", "/api/index", "/index.py"):
        if path == prefix:
            request.scope["path"] = "/"
            break
        elif path.startswith(prefix + "/"):
            request.scope["path"] = path[len(prefix):]
            break

    response = await call_next(request)
    return response


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


@app.get("/", tags=["Health"])
@app.get("/health", tags=["Health"])
@app.get("/api/health", tags=["Health"])
@app.get("/api/index.py", tags=["Health"])
@app.get("/api/index.py/health", tags=["Health"])
def health_check() -> dict[str, str]:
    return {"status": "ok", "service": "UniGuide AI API"}
