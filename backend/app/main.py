from fastapi import FastAPI

from app.routes.chat import router as chat_router


app = FastAPI(
    title="UniGuide AI API",
    description="Backend API for the UniGuide AI student assistant.",
    version="0.1.0",
)

app.include_router(
    chat_router,
    prefix="/api/v1/chat",
    tags=["Chat"],
)


@app.get("/health", tags=["Health"])
def health_check() -> dict[str, str]:
    return {"status": "ok"}
