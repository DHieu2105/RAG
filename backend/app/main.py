from fastapi import FastAPI


app = FastAPI(
    title="UniGuide AI API",
    description="Backend API for the UniGuide AI student assistant.",
    version="0.1.0",
)


@app.get("/health", tags=["Health"])
def health_check() -> dict[str, str]:
    return {"status": "ok"}
