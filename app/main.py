from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.routes import router as api_router
from app.model import load_model

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model()
    print("Model loaded successfully.")
    yield
    print("Shutting down application.")

app = FastAPI(
    title="CAPTCHA Recognition API",
    description="API for recognizing text from CAPTCHA images",
    lifespan=lifespan
)

app.include_router(api_router)
