"""R2 model versions and activation, with temporary runtime downloads."""
import hashlib
import os
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4



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


def manifest_key(model_id):
    return os.getenv("R2_PREFIX", "models").strip("/") + f"/{model_id}/active.json"


def active_model(model_id):
    import json
    client = r2_client()
    try:
        response = client.get_object(Bucket=os.environ["R2_BUCKET"], Key=manifest_key(model_id))
    except Exception as exc:
        code = getattr(exc, "response", {}).get("Error", {}).get("Code")
        if code in ("NoSuchKey", "404", "NotFound"):
            return None
        raise
    body = response["Body"]
    try:
        return json.loads(body.read())
    finally:
        body.close()


def activate(model_id, metadata):
    import json
    r2_client().put_object(Bucket=os.environ["R2_BUCKET"], Key=manifest_key(model_id),
                           Body=json.dumps(metadata).encode(), ContentType="application/json")


def model_path(model_id, metadata=None):
    metadata = metadata or active_model(model_id)
    if not metadata:
        raise ValueError(f"Model {model_id} chưa có trên R2.")
    folder = Path(tempfile.gettempdir()) / "fashion-ai-r2-cache" / model_id
    folder.mkdir(parents=True, exist_ok=True)
    suffix = ".tar.gz" if model_id == "rasa" else ".pt"
    target = folder / (metadata["sha256"] + suffix)
    if target.is_file() and digest(target) == metadata["sha256"]:
        return target
    descriptor, name = tempfile.mkstemp(dir=folder, suffix=suffix)
    os.close(descriptor)
    stage = Path(name)
    try:
        r2_client().download_file(os.environ["R2_BUCKET"], metadata["r2_key"], str(stage))
        if digest(stage) != metadata["sha256"]:
            raise ValueError("Checksum model R2 không khớp.")
        os.replace(stage, target)
        return target
    finally:
        stage.unlink(missing_ok=True)


def install_model(model_id, stage):
    from app import database
    import server
    validate_model(model_id, stage)
    version_id = uuid4().hex
    suffix = ".tar.gz" if model_id == "rasa" else ".pt"
    key = os.getenv("R2_PREFIX", "models").strip("/") + f"/{model_id}/{version_id}{suffix}"
    metadata = dict(id=version_id, model_id=model_id,
                    created_at=datetime.now(timezone.utc).isoformat(),
                    size_bytes=stage.stat().st_size, sha256=digest(stage), r2_key=key)
    with server.LOCK:
        client = r2_client()
        client.upload_file(str(stage), os.environ["R2_BUCKET"], key)
        # R2 manifest is the source of truth; failed upload never changes active model.
        with database() as connection:
            connection.execute("INSERT INTO versions VALUES (?, ?, ?, ?, ?, ?, ?)",
                               (version_id, model_id, metadata["created_at"], metadata["size_bytes"],
                                metadata["sha256"], "", key))
        activate(model_id, metadata)
        server.MODELS.clear()
    message = "Đã upload và kích hoạt model trên R2."
    if model_id == "rasa":
        message += " Dịch vụ Rasa cần nạp lại model qua URL tải mới."
    return {"message": message, "version_id": version_id}


def restore_version(model_id, row):
    from app import database
    import server
    _, key, checksum = row
    if not key:
        raise ValueError("Phiên bản này chưa được lưu trên R2.")
    with database() as connection:
        version = connection.execute("SELECT id, created_at, size_bytes FROM versions WHERE model_id=? AND r2_key=?", (model_id, key)).fetchone()
    metadata = dict(id=version[0], model_id=model_id, created_at=version[1],
                    size_bytes=version[2], sha256=checksum, r2_key=key)
    # Check object is still present before switching the manifest.
    r2_client().head_object(Bucket=os.environ["R2_BUCKET"], Key=key)
    with server.LOCK:
        activate(model_id, metadata)
        server.MODELS.clear()
    return {"message": "Đã kích hoạt lại phiên bản trên R2.", "version_id": version[0]}
