from datetime import datetime, timedelta, timezone

import jwt

from rag_permission.models import User


def create_access_token(
    user_id: str,
    groups: tuple[str, ...],
    secret: str,
    ttl_minutes: int = 60,
) -> str:
    if not secret:
        raise ValueError("auth_secret must be configured")
    if len(secret.encode("utf-8")) < 32:
        raise ValueError("auth_secret must contain at least 32 bytes")
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "groups": list(groups),
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=ttl_minutes)).timestamp()),
    }
    return jwt.encode(payload, secret, algorithm="HS256")


def decode_access_token(token: str, secret: str) -> User:
    if not secret:
        raise jwt.InvalidTokenError("auth_secret is not configured")
    if len(secret.encode("utf-8")) < 32:
        raise jwt.InvalidTokenError("auth_secret is too short")
    payload = jwt.decode(token, secret, algorithms=["HS256"])
    user_id = payload.get("sub")
    if not isinstance(user_id, str) or not user_id:
        raise jwt.InvalidTokenError("token subject is missing")
    raw_groups = payload.get("groups", [])
    if not isinstance(raw_groups, list) or not all(
        isinstance(group, str) for group in raw_groups
    ):
        raise jwt.InvalidTokenError("token groups are invalid")
    return User(id=user_id, groups=frozenset(raw_groups))
