from fastapi import FastAPI
from app.core.database import engine, Base
from app.api import auth

app = FastAPI()

Base.metadata.create_all(bind=engine)

app.include_router(auth.router)