from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import or_
from app.core.security import (
    create_access_token, 
    get_password_hash
)
from app.core.security import create_access_token, verify_password
from app.core.database import get_db
from app.models.auth import User
from app.schemas.auth import UserCreate, UserResponse, UserLogin, Token

router = APIRouter()

@router.post("/signup", response_model=UserResponse, tags=["Auth"])
def signup(user: UserCreate, db: Session = Depends(get_db)):
    existing_user = db.query(User).filter(or_(User.username == user.username, User.email == user.email)).first()

    if existing_user:
        if existing_user.username == user.username:
            raise HTTPException(status_code=400, detail="이미 등록된 닉네임 입니다.")
        else:
            raise HTTPException(status_code=400, detail="이미 등록된 이메일 입니다.")
    
    hashed_password = get_password_hash(user.password)
    new_user = User(email=user.email, hashed_password=hashed_password, username=user.username)
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return new_user