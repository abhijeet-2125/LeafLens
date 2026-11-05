from fastapi import FastAPI

app = FastAPI(
    title="LeafLens API",
    description="AI-powered plant leaf disease detection and monitoring system",
    version="0.1.0",
)


@app.get("/", tags=["General"])
def root() -> dict[str, str]:
    """Return basic application information."""
    return {
        "application": "LeafLens",
        "version": "0.1.0",
        "message": "Welcome to LeafLens API",
    }


@app.get("/health", tags=["Operations"])
def health_check() -> dict[str, str]:
    """Check whether the API application is responding."""
    return {
        "status": "healthy",
        "service": "leaflens-api",
    }
