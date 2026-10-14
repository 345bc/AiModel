"""Local test UI: python server.py, then http://127.0.0.1:8080."""

import base64
import io
import json
import os
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock
from urllib.request import Request, urlopen
from urllib.error import URLError
from config import ROOT

RASA_URL = os.environ.get("RASA_URL", "http://127.0.0.1:5005").rstrip("/")
LOCK = Lock()
MODELS = {}


def predict(data):
    import torch
    from PIL import Image, ImageOps
    from torchvision import models, transforms
    from ultralytics import YOLO

    raw = base64.b64decode(data["image"].split(",", 1)[-1], validate=True)
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    with LOCK:
        model_paths = [ROOT / "resnet/best.pt", ROOT / "yolo/best.pt"]
        signature = tuple((p.stat().st_mtime_ns, p.stat().st_size) for p in model_paths)
        if not MODELS or MODELS.get("signature") != signature:
            checkpoint = torch.load(
                ROOT / "resnet/best.pt", map_location="cpu", weights_only=True
            )
            if checkpoint["architecture"] != "resnet18":
                raise ValueError("Kiến trúc ResNet chưa được hỗ trợ.")
            names = checkpoint.get("class_names", checkpoint.get("colors"))
            if not isinstance(names, (list, tuple)) or not names:
                raise ValueError("Checkpoint ResNet cần danh sách nhãn trong 'class_names' hoặc 'colors'.")
            state = checkpoint.get("state_dict", checkpoint.get("model_state_dict"))
            if state is None:
                raise ValueError("Checkpoint ResNet thiếu 'state_dict' hoặc 'model_state_dict'.")
            if state["fc.weight"].shape[0] != len(names):
                raise ValueError("Số nhãn màu không khớp số đầu ra của ResNet.")
            net = models.resnet18(weights=None)
            net.fc = torch.nn.Linear(net.fc.in_features, len(names))
            net.load_state_dict(state)
            net.eval()
            detector = YOLO(str(ROOT / "yolo/best.pt"))
            MODELS.update(net=net, names=names, yolo=detector, signature=signature,
                          image_size=int(checkpoint.get("image_size", 224)),
                          mean=checkpoint.get("mean", [0.485, 0.456, 0.406]),
                          std=checkpoint.get("std", [0.229, 0.224, 0.225]))
        transform = transforms.Compose(
            [
                transforms.Resize((MODELS["image_size"], MODELS["image_size"])),
                transforms.ToTensor(),
                transforms.Normalize(MODELS["mean"], MODELS["std"]),
            ]
        )

        def color(crop):
            with torch.inference_mode():
                probs = MODELS["net"](transform(crop).unsqueeze(0)).softmax(1)[0]
            values, indices = probs.topk(3)
            return [
                {"label": MODELS["names"][i], "confidence": float(v)}
                for v, i in zip(values, indices)
            ]

        threshold = float(data.get("confidence", 0.25))
        if not 0.01 <= threshold <= 1:
            raise ValueError("Ngưỡng confidence phải từ 0.01 đến 1.")
        result = MODELS["yolo"].predict(
            img, conf=threshold, device="cpu", verbose=False
        )[0]
        detections = []
        for box in result.boxes:
            coords = box.xyxy[0].tolist()
            crop = img.crop(tuple(coords))
            detections.append(
                {
                    "label": result.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "box": coords,
                    "colors": color(crop),
                }
            )
        return {
            "width": img.width,
            "height": img.height,
            "detections": detections,
        }


class Handler(BaseHTTPRequestHandler):
    def handle(self):
        try:
            super().handle()
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            # The client disconnected; no response can be sent on this socket.
            self.close_connection = True

    def send_json(self, status, data):
        payload = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            payload = (ROOT / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        elif self.path == "/api/status":
            try:
                with urlopen(RASA_URL + "/status", timeout=2) as response:
                    rasa = response.status == 200
            except (URLError, TimeoutError):
                rasa = False
            self.send_json(
                200,
                {
                    "yolo": (ROOT / "yolo/best.pt").exists(),
                    "resnet": (ROOT / "resnet/best.pt").exists(),
                    "rasa": rasa,
                },
            )
        else:
            self.send_json(404, {"error": "Không tìm thấy."})

    def do_POST(self):
        try:
            size = int(self.headers.get("Content-Length", 0))
            if size <= 0 or size > 20 * 1024 * 1024:
                return self.send_json(
                    413, {"error": "Dữ liệu quá lớn hoặc rỗng (tối đa 20 MB)."}
                )
            data = json.loads(self.rfile.read(size))
            if self.path == "/api/predict":
                self.send_json(200, predict(data))
            elif self.path == "/api/chat":
                message = str(data.get("message", "")).strip()
                if not message:
                    raise ValueError("Nhập tin nhắn trước khi gửi.")
                req = Request(
                    RASA_URL + "/webhooks/rest/webhook",
                    data=json.dumps(
                        {"sender": data.get("sender", "html-test"), "message": message}
                    ).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urlopen(req, timeout=60) as response:
                    self.send_json(200, {"messages": json.load(response)})
            else:
                self.send_json(404, {"error": "Không tìm thấy API."})
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            self.close_connection = True
        except URLError:
            self.send_json(
                503,
                {
                    "error": "Không kết nối được Rasa. Hãy chạy Rasa với REST channel trên cổng 5005."
                },
            )
        except Exception as exc:
            self.send_json(400, {"error": str(exc)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    httpd = None
    for candidate in dict.fromkeys([port, 8088, 8765, 18080, 0]):
        try:
            httpd = ThreadingHTTPServer(("127.0.0.1", candidate), Handler)
            break
        except OSError as exc:
            if exc.errno not in (13, 98, 10013, 10048) and getattr(
                exc, "winerror", None
            ) not in (10013, 10048):
                raise
            print(f"Không mở được cổng {candidate}; đang thử cổng khác.", flush=True)
    if httpd is None:
        raise SystemExit(
            "Windows không cho mở cổng localhost. Kiểm tra quyền truy cập mạng của Python trong firewall/phần mềm bảo mật."
        )
    port = httpd.server_address[1]
    print(f"Mở http://127.0.0.1:{port} để test 3 model.", flush=True)
    httpd.serve_forever()
