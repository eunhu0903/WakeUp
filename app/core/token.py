from fastapi import HTTPException, status, Header
from datetime import datetime
from sqlalchemy.orm import Session
from models.auth import User
from .security import decode_access_token

def get_token_from_header(authorization: str = Header(None)) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="인증 토큰이 없거나 잘못되었습니다.")
    
    token = authorization[7:]
    return token

def verify_token(token: str, db: Session) -> str:
    try:
        payload = decode_access_token(token)
        user_id: str = payload.get("sub")

        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="토큰에 user_id가 없습니다.",
            )

        user = db.query(User).filter(User.id == int(user_id)).first()
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="잘못된 사용자 ID 또는 토큰입니다.",
            )

        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="정지된 계정입니다.",
            )

        expiration: int = payload.get("exp")
        if expiration and expiration < int(datetime.utcnow().timestamp()):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="토큰이 만료되었습니다.",
            )

        return user_id

    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="자격 증명을 확인할 수 없습니다.",
        )