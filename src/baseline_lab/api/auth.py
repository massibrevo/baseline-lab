"""Browser capability identity: local deployment only, signed HTTP-only cookie."""

import hashlib
import hmac
import secrets
from pathlib import Path

from fastapi import HTTPException, Request


def secret(directory: Path) -> bytes:
    path = directory / ".cookie-key"
    if not path.exists():
        try:
            with path.open("xb") as f:
                f.write(secrets.token_bytes(32))
            path.chmod(0o600)
        except FileExistsError:
            pass
    return path.read_bytes()


def mint(key: bytes) -> str:
    identity = secrets.token_hex(24)
    return identity + "." + hmac.new(key, identity.encode(), hashlib.sha256).hexdigest()


def owner(request: Request) -> str:
    token = request.cookies.get("baseline_owner", "")
    try:
        identity, signature = token.split(".")
        expected = hmac.new(request.app.state.key, identity.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected):
            raise ValueError("invalid")
    except ValueError as exc:
        raise HTTPException(401, "browser_identity_required") from exc
    return hashlib.sha256(identity.encode()).hexdigest()
