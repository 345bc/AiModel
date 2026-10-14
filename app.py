"""AI model manager: python app.py."""

import base64
import json
import os
import sqlite3
import tempfile
from datetime import datetime, timezone
from contextlib import contextmanager
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import uvicorn
from fastapi import FastAPI, HTTPException, Request as WebRequest
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import server
from config import ROOT, DATA_DIR, MAX_MODEL_MB
from starlette.concurrency import run_in_threadpool

CATALOG = {
    "yolo": ("YOLO", "Nhận diện quần áo", "yolo/best.pt"),
    "resnet": ("ResNet18", "Phân loại màu quần áo", "resnet/best.pt"),
    "rasa": ("Rasa", "Chatbot thời trang", "rasa/fashion_chatbot_final_v2.tar.gz"),
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
    for model_id, (name, task, filename) in CATALOG.items():
        path = ROOT / filename
        exists = path.is_file()
        result.append(
            dict(
                id=model_id,
                name=name,
                task=task,
                file=filename,
                enabled=bool(enabled[model_id]),
                available=exists,
                size_bytes=path.stat().st_size if exists else 0,
                modified_at=(
                    datetime.fromtimestamp(
                        path.stat().st_mtime, timezone.utc
                    ).isoformat()
                    if exists
                    else None
                ),
                format=".tar.gz" if model_id == "rasa" else ".pt",
            )
        )
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
        "provider": "Cloudflare R2" if r2_enabled() else "Local",
        "configured": r2_configured(),
        "enabled": r2_enabled(),
        "max_model_mb": MAX_MODEL_MB,
    }


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

    upload_dir = DATA_DIR / "uploads"
    upload_dir.mkdir(exist_ok=True)
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
    with server.LOCK:
        server.MODELS.clear()
    return {
        "message": "Đã xóa cache. Model sẽ được nạp lại khi nhận yêu cầu nhận diện tiếp theo."
    }


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
