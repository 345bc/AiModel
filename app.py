"""AI model manager: python app.py."""

import base64
import json
import os
import sqlite3
import tempfile
import secrets
import hashlib
import hmac
import time
from datetime import datetime, timezone
from contextlib import contextmanager
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import uvicorn
from fastapi import FastAPI, HTTPException, Request as WebRequest
from fastapi.responses import FileResponse, RedirectResponse, JSONResponse
from pydantic import BaseModel, Field

import server
from config import ROOT, DATA_DIR, MAX_MODEL_MB
from starlette.concurrency import run_in_threadpool

CATALOG = {
    "yolo": ("YOLO", "Nhận diện quần áo", ".pt"),
    "resnet": ("ResNet18", "Phân loại màu quần áo", ".pt"),
    "rasa": ("Rasa", "Chatbot thời trang", ".tar.gz"),
}


@contextmanager
def database():
    connection = sqlite3.connect(DATA_DIR / "manager.sqlite3")
    try:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS models (id TEXT PRIMARY KEY, enabled INTEGER NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS versions (id TEXT PRIMARY KEY, model_id TEXT NOT NULL, created_at TEXT NOT NULL, size_bytes INTEGER NOT NULL, sha256 TEXT NOT NULL, local_file TEXT NOT NULL, r2_key TEXT)"
        )
        for model_id in CATALOG:
            connection.execute(
                "INSERT OR IGNORE INTO models VALUES (?, 1)", (model_id,)
            )
        connection.commit()
        yield connection
        connection.commit()
    finally:
        connection.close()


def catalog():
    with database() as connection:
        enabled = dict(connection.execute("SELECT id, enabled FROM models"))
    result = []
    from storage import active_model
    for model_id, (name, task, suffix) in CATALOG.items():
        try:
            metadata = active_model(model_id)
        except Exception as exc:
            raise HTTPException(503, "Không đọc được model từ R2. Kiểm tra cấu hình và kết nối.") from exc
        result.append(dict(id=model_id, name=name, task=task,
                           file=metadata["r2_key"] if metadata else None,
                           enabled=bool(enabled[model_id]), available=bool(metadata),
                           size_bytes=metadata["size_bytes"] if metadata else 0,
                           modified_at=metadata["created_at"] if metadata else None,
                           format=suffix, storage="Cloudflare R2",
                           active_version=metadata["id"] if metadata else None))
    return result


def require_models(*ids):
    models = {model["id"]: model for model in catalog()}
    for model_id in ids:
        if not models[model_id]["enabled"] or not models[model_id]["available"]:
            raise HTTPException(503, f"Model {model_id} đang tắt hoặc thiếu file.")


app = FastAPI(
    title="Fashion AI API",
    version="1.0.0",
    description="Quản lý model và sử dụng pipeline YOLO + ResNet, chatbot Rasa.",
)

COOKIE_NAME = "fashion_session"
SESSION_SECONDS = 8 * 60 * 60


def session_user(request):
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        return None
    with database() as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS sessions (token_hash TEXT PRIMARY KEY, username TEXT NOT NULL, expires_at REAL NOT NULL)")
        connection.execute("DELETE FROM sessions WHERE expires_at <= ?", (time.time(),))
        row = connection.execute("SELECT username FROM sessions WHERE token_hash = ?", (hashlib.sha256(token.encode()).hexdigest(),)).fetchone()
    return row[0] if row else None


@app.middleware("http")
async def require_login(request: WebRequest, call_next):
    # Java can call image inference directly without a Python admin session.
    if request.method == "POST" and request.url.path.rstrip("/") == "/api/predict":
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response
    if request.url.path in ("/login", "/api/auth/login", "/health"):
        return await call_next(request)
    username = await run_in_threadpool(session_user, request)
    if not username:
        if request.url.path.startswith("/api/"):
            return JSONResponse({"detail": "Vui lòng đăng nhập."}, status_code=401)
        return RedirectResponse("/login", status_code=303)
    request.state.username = username
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        origin = request.headers.get("origin")
        if origin and origin != f"{request.url.scheme}://{request.headers.get('host')}":
            return JSONResponse({"detail": "Nguồn yêu cầu không hợp lệ."}, status_code=403)
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    return response


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=1, max_length=200)


@app.get("/login", include_in_schema=False)
def login_page():
    return FileResponse(ROOT / "login.html", headers={"Cache-Control": "no-store"})


@app.get("/health", include_in_schema=False)
def health():
    return {"status": "ok"}


@app.post("/api/auth/login")
def login(body: LoginRequest, request: WebRequest):
    username = os.getenv("ADMIN_USERNAME", "admin")
    password = os.getenv("ADMIN_PASSWORD", "Admin@123")
    valid_user = hmac.compare_digest(body.username.encode(), username.encode())
    valid_password = hmac.compare_digest(body.password.encode(), password.encode())
    if not (valid_user and valid_password):
        raise HTTPException(401, "Tài khoản hoặc mật khẩu không đúng.")
    token = secrets.token_urlsafe(32)
    with database() as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS sessions (token_hash TEXT PRIMARY KEY, username TEXT NOT NULL, expires_at REAL NOT NULL)")
        old_token = request.cookies.get(COOKIE_NAME)
        if old_token:
            connection.execute("DELETE FROM sessions WHERE token_hash=?", (hashlib.sha256(old_token.encode()).hexdigest(),))
        connection.execute("INSERT INTO sessions VALUES (?, ?, ?)",
                           (hashlib.sha256(token.encode()).hexdigest(), username, time.time() + SESSION_SECONDS))
    response = JSONResponse({"username": username})
    response.set_cookie(COOKIE_NAME, token, max_age=SESSION_SECONDS, httponly=True,
                        secure=os.getenv("COOKIE_SECURE", "false").lower() == "true", samesite="strict")
    response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/api/auth/me")
def current_user(request: WebRequest):
    return {"username": request.state.username}


@app.post("/api/auth/logout")
def logout(request: WebRequest):
    with database() as connection:
        token = request.cookies.get(COOKIE_NAME, "")
        connection.execute("DELETE FROM sessions WHERE token_hash=?", (hashlib.sha256(token.encode()).hexdigest(),))
    response = JSONResponse({"message": "Đã đăng xuất."})
    response.delete_cookie(COOKIE_NAME)
    return response


class ModelUpdate(BaseModel):
    enabled: bool


class ImageRequest(BaseModel):
    image: str = Field(
        min_length=1,
        max_length=20 * 1024 * 1024,
        description="Ảnh base64 hoặc data URL",
    )
    confidence: float = Field(default=0.25, ge=0.01, le=1)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    sender: str = Field(default="web-user", min_length=1, max_length=100)


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(ROOT / "manager.html")


@app.get("/demo", include_in_schema=False)
def demo():
    return FileResponse(ROOT / "index.html")


@app.get("/api/models")
def list_models():
    return {"models": catalog()}


@app.get("/api/models/{model_id}")
def model_detail(model_id: str):
    if model_id not in CATALOG:
        raise HTTPException(404, "Không tìm thấy model.")
    model = next(item for item in catalog() if item["id"] == model_id)
    with database() as connection:
        rows = connection.execute(
            "SELECT id, created_at, size_bytes, sha256, r2_key FROM versions WHERE model_id = ? ORDER BY created_at DESC",
            (model_id,),
        ).fetchall()
    model["versions"] = [
        dict(zip(("id", "created_at", "size_bytes", "sha256", "r2_key"), row))
        for row in rows
    ]
    return model


@app.get("/api/storage")
def storage_status():
    from storage import r2_configured, r2_enabled

    return {
        "provider": "Cloudflare R2",
        "configured": r2_configured(),
        "enabled": r2_enabled(),
        "max_model_mb": MAX_MODEL_MB,
    }


@app.get("/api/models/{model_id}/download")
def download_model(model_id: str):
    if model_id not in CATALOG:
        raise HTTPException(404, "Không tìm thấy model.")
    from storage import active_model, r2_client
    try:
        metadata = active_model(model_id)
        if not metadata:
            raise HTTPException(404, "Model chưa được upload lên R2.")
        url = r2_client().generate_presigned_url("get_object",
            Params={"Bucket": os.environ["R2_BUCKET"], "Key": metadata["r2_key"]}, ExpiresIn=300)
        return RedirectResponse(url, status_code=307)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(503, "Không lấy được model từ R2.") from exc


@app.post("/api/storage/check")
def check_storage():
    from storage import r2_client

    try:
        r2_client().head_bucket(Bucket=os.getenv("R2_BUCKET"))
        return {"message": "Kết nối Cloudflare R2 thành công."}
    except Exception as exc:
        raise HTTPException(
            503,
            "Không kết nối được R2. Kiểm tra cấu hình, bucket và quyền truy cập trong .env.",
        ) from exc


@app.put(
    "/api/models/{model_id}/file",
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/octet-stream": {
                    "schema": {"type": "string", "format": "binary"}
                }
            },
        },
        "parameters": [
            {
                "name": "X-Filename",
                "in": "header",
                "required": True,
                "schema": {"type": "string"},
                "description": "Tên file .pt hoặc .tar.gz",
            }
        ],
    },
)
async def replace_model(model_id: str, request: WebRequest):
    if model_id not in CATALOG:
        raise HTTPException(404, "Không tìm thấy model.")
    filename = request.headers.get("X-Filename", "").lower()
    suffix = ".tar.gz" if model_id == "rasa" else ".pt"
    if not filename.endswith(suffix):
        raise HTTPException(422, f"Model cần file {suffix}.")
    from storage import install_model

    upload_dir = Path(tempfile.gettempdir()) / "fashion-ai-r2-uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=upload_dir, suffix=suffix)
    stage = Path(name)
    try:
        size = 0
        with os.fdopen(descriptor, "wb") as stream:
            async for chunk in request.stream():
                size += len(chunk)
                if size > MAX_MODEL_MB * 1024 * 1024:
                    raise HTTPException(413, f"File vượt quá {MAX_MODEL_MB} MB.")
                stream.write(chunk)
        if not size:
            raise HTTPException(422, "File không được rỗng.")
        try:
            return await run_in_threadpool(install_model, model_id, stage)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                503,
                "Thay thế model thất bại. Kiểm tra cấu hình R2, quyền ghi và log server.",
            ) from exc
    finally:
        stage.unlink(missing_ok=True)


@app.post("/api/models/{model_id}/versions/{version_id}/restore")
def restore_model(model_id: str, version_id: str):
    if model_id not in CATALOG:
        raise HTTPException(404, "Không tìm thấy model.")
    with database() as connection:
        row = connection.execute(
            "SELECT local_file, r2_key, sha256 FROM versions WHERE id = ? AND model_id = ?",
            (version_id, model_id),
        ).fetchone()
    if row is None:
        raise HTTPException(404, "Không tìm thấy phiên bản.")
    from storage import restore_version

    try:
        return restore_version(model_id, row)
    except Exception as exc:
        raise HTTPException(
            503, "Không khôi phục được phiên bản. Kiểm tra file và kết nối R2."
        ) from exc


@app.patch("/api/models/{model_id}")
def update_model(model_id: str, body: ModelUpdate):
    if model_id not in CATALOG:
        raise HTTPException(404, "Không tìm thấy model.")
    with database() as connection:
        connection.execute(
            "UPDATE models SET enabled = ? WHERE id = ?", (body.enabled, model_id)
        )
    return next(model for model in catalog() if model["id"] == model_id)


@app.post("/api/models/reload")
def reload_models():
    from storage import active_model, model_path, validate_model
    downloaded = []
    missing = []
    try:
        for model_id in CATALOG:
            metadata = active_model(model_id)
            if not metadata:
                missing.append(model_id)
                continue
            path = model_path(model_id, metadata)
            validate_model(model_id, path)
            with database() as connection:
                connection.execute("INSERT OR IGNORE INTO versions VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (metadata["id"], model_id, metadata["created_at"], metadata["size_bytes"],
                     metadata["sha256"], "", metadata["r2_key"]))
            downloaded.append(model_id)
        if not downloaded:
            raise HTTPException(404, "Không tìm thấy model trên R2. Kiểm tra R2_BUCKET và R2_PREFIX trong .env.")
        with server.LOCK:
            server.MODELS.clear()
        # Warm the vision pipeline so the next real request does not need model loading.
        vision_ready = all(model_id in downloaded for model_id in ("yolo", "resnet"))
        if vision_ready:
            import io
            from PIL import Image
            image = io.BytesIO()
            Image.new("RGB", (32, 32), "white").save(image, format="PNG")
            server.predict({"image": base64.b64encode(image.getvalue()).decode()})
        message = f"Đã lấy {len(downloaded)} model từ R2."
        if vision_ready:
            message += " YOLO và ResNet đã được nạp, sẵn sàng nhận diện."
        if "rasa" in downloaded:
            message += " File Rasa đã tải; chatbot cần dịch vụ Rasa đang chạy."
        if missing:
            message += " Chưa có trên R2: " + ", ".join(missing) + "."
        return {"message": message, "downloaded": downloaded, "missing": missing}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(503, "Không tải hoặc nạp được model R2. Kiểm tra cấu hình, kết nối và checkpoint.") from exc


@app.get("/api/status")
def status():
    models = {m["id"]: m for m in catalog()}
    try:
        with urlopen(server.RASA_URL + "/status", timeout=2) as response:
            rasa = response.status == 200
    except (URLError, TimeoutError, OSError):
        rasa = False
    return {
        "yolo": models["yolo"]["enabled"] and models["yolo"]["available"],
        "resnet": models["resnet"]["enabled"] and models["resnet"]["available"],
        "rasa": models["rasa"]["enabled"] and rasa,
    }


@app.post("/api/predict")
def predict(body: ImageRequest):
    require_models("yolo", "resnet")
    try:
        base64.b64decode(body.image.split(",", 1)[-1], validate=True)
    except ValueError:
        raise HTTPException(422, "Ảnh base64 không hợp lệ.")
    try:
        return server.predict(body.model_dump())
    except (ValueError, OSError) as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            503,
            "Không chạy được model. Kiểm tra dependencies và checkpoint trên server.",
        ) from exc


@app.post("/api/chat")
def chat(body: ChatRequest):
    require_models("rasa")
    if not body.message.strip():
        raise HTTPException(422, "Tin nhắn không được rỗng.")
    request = Request(
        server.RASA_URL + "/webhooks/rest/webhook",
        data=json.dumps(body.model_dump()).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=60) as response:
            return {"messages": json.load(response)}
    except (URLError, TimeoutError, OSError) as exc:
        raise HTTPException(
            503, "Rasa chưa kết nối. Hãy chạy Rasa trên cổng 5005."
        ) from exc


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8080")),
    )
