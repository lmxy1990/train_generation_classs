import os
from fastapi import HTTPException
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from starlette.requests import Request

# 创建API密钥验证器
VALID_TOKENS = os.getenv("VALID_TOKENS", "123456").split(",")  # 固定令牌


class TokenAuth(HTTPBearer):
    def __init__(self, auto_error: bool = True):
        super().__init__(auto_error=auto_error)

    async def __call__(self, request: Request):
        print("正在验证令牌...")
        credentials: HTTPAuthorizationCredentials = await super().__call__(request)
        print("已获取认证令牌")
        if not credentials:
            print("未提供认证令牌")
            raise HTTPException(status_code=401, detail="未提供认证令牌")

        token = credentials.credentials
        if token not in VALID_TOKENS:
            print("无效的认证令牌")
            raise HTTPException(status_code=403, detail="无效的认证令牌")
        return token
