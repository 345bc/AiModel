"""Versioned local model storage with optional private Cloudflare R2 backup."""
import hashlib
import os
import shutil
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from config import DATA_DIR, ROOT


def r2_enabled():
    return os.getenv("R2_ENABLED", "false").lower() == "true"


def r2_configured():
    return all(os.getenv(key) for key in ("R2_ENDPOINT_URL", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET"))


def r2_client():
    if not r2_enabled() or not r2_configured():
        raise ValueError("Cloudflare R2 chưa được cấu hình đầy đủ.")
    import boto3
    from botocore.config import Config
    return boto3.client("s3", endpoint_url=os.environ["R2_ENDPOINT_URL"],
                        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
                        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
                        region_name="auto", config=Config(connect_timeout=10, read_timeout=60,
                        retries={"max_attempts": 2}))


def digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_model(model_id, path):
    # Files must be trusted training artifacts: YOLO checkpoints contain Python objects.
    if model_id == "rasa":
        try:
            with tarfile.open(path, "r:gz") as archive:
                if not any(Path(member.name).name == "metadata.json" for member in archive):
                    raise ValueError("Archive Rasa thiếu metadata.json.")
        except tarfile.TarError as exc:
            raise ValueError("File Rasa không phải archive tar.gz hợp lệ.") from exc
        return
    try:
        if model_id == "yolo":
            from ultralytics import YOLO
            model = YOLO(str(path), task="detect")
            if model.task != "detect":
                raise ValueError("YOLO phải là model detection.")
        else:
            import torch
            from torchvision import models
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            names = checkpoint.get("class_names", checkpoint.get("colors"))
            if checkpoint.get("architecture") != "resnet18" or not isinstance(names, (list, tuple)) or len(names) < 3:
                raise ValueError("Cần checkpoint ResNet18 với ít nhất 3 nhãn màu.")
            net = models.resnet18(weights=None)
            net.fc = torch.nn.Linear(net.fc.in_features, len(names))
            net.load_state_dict(checkpoint.get("state_dict", checkpoint.get("model_state_dict")))
    except Exception as exc:
        raise ValueError("Checkpoint không tương thích với model được chọn.") from exc


def save_version(model_id, source, connection):
    version_id = uuid4().hex
    suffix = ".tar.gz" if model_id == "rasa" else ".pt"
    folder = DATA_DIR / "versions" / model_id
    folder.mkdir(parents=True, exist_ok=True)
    saved = folder / (version_id + suffix)
    shutil.copy2(source, saved)
    key = None
    try:
        if r2_enabled():
            key = os.getenv("R2_PREFIX", "models").strip("/") + f"/{model_id}/{version_id}{suffix}"
            r2_client().upload_file(str(saved), os.environ["R2_BUCKET"], key)
        connection.execute("INSERT INTO versions VALUES (?, ?, ?, ?, ?, ?, ?)",
                           (version_id, model_id, datetime.now(timezone.utc).isoformat(),
                            saved.stat().st_size, digest(saved), str(saved.relative_to(DATA_DIR)), key))
        return version_id
    except Exception:
        saved.unlink(missing_ok=True)
        raise


def install_model(model_id, stage):
    from app import CATALOG, database
    import server
    validate_model(model_id, stage)
    target = ROOT / CATALOG[model_id][2]
    target.parent.mkdir(parents=True, exist_ok=True)
    with server.LOCK:
        with database() as connection:
            if target.exists():
                current_hash = digest(target)
                if not connection.execute("SELECT 1 FROM versions WHERE model_id = ? AND sha256 = ?", (model_id, current_hash)).fetchone():
                    save_version(model_id, target, connection)
            version_id = save_version(model_id, stage, connection)
            # Commit backup records before activation so they remain available on failure.
            connection.commit()
        descriptor, replacement_name = tempfile.mkstemp(dir=target.parent, suffix=target.suffix)
        os.close(descriptor)
        replacement = Path(replacement_name)
        try:
            shutil.copy2(stage, replacement)
            os.replace(replacement, target)
        finally:
            replacement.unlink(missing_ok=True)
        server.MODELS.clear()
    message = "Đã thay thế model và lưu phiên bản dự phòng."
    if model_id == "rasa":
        message += " Khởi động lại dịch vụ Rasa với file mới để áp dụng."
    return {"message": message, "version_id": version_id}


def restore_version(model_id, row):
    relative, key, checksum = row
    source = DATA_DIR / relative
    if source.exists():
        if digest(source) != checksum:
            raise ValueError("Checksum không khớp.")
        return install_model(model_id, source)
    if not key:
        raise ValueError("Không có bản lưu R2.")
    suffix = ".tar.gz" if model_id == "rasa" else ".pt"
    descriptor, name = tempfile.mkstemp(dir=DATA_DIR, suffix=suffix)
    os.close(descriptor)
    stage = Path(name)
    try:
        r2_client().download_file(os.environ["R2_BUCKET"], key, str(stage))
        if digest(stage) != checksum:
            raise ValueError("Checksum không khớp.")
        return install_model(model_id, stage)
    finally:
        stage.unlink(missing_ok=True)
