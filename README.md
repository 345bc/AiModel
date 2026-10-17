# Test 3 model

## Ứng dụng quản lý model và API

Chạy từ thư mục `model_final`:

```powershell
python -m pip install -r requirements.txt
python app.py
```

Mở http://127.0.0.1:8080 để quản lý YOLO, ResNet và Rasa, bật/tắt model,
xem chi tiết, thay thế checkpoint và khôi phục phiên bản. Frontend và backend chạy chung một server.
Tài liệu API tương tác: http://127.0.0.1:8080/docs.
Giao diện nhận diện cũ: http://127.0.0.1:8080/demo.

| API                                                   | Chức năng                                                      |
| ----------------------------------------------------- | -------------------------------------------------------------- |
| `GET /api/models`                                     | Danh sách, kích thước file, trạng thái bật/tắt                 |
| `GET /api/models/{id}`                                | Thông tin model và lịch sử phiên bản                           |
| `PUT /api/models/{id}/file`                           | Upload file nhị phân, header `X-Filename`, thay thế checkpoint |
| `POST /api/models/{id}/versions/{version_id}/restore` | Kích hoạt lại phiên bản trên R2                       |
| `GET /api/storage`                                    | Trạng thái cấu hình lưu trữ, không trả khóa bí mật             |
| `POST /api/storage/check`                             | Kiểm tra kết nối bucket R2                                     |
| `PATCH /api/models/{id}`                              | Bật/tắt với JSON `{"enabled": true}`                           |
| `POST /api/models/reload`                             | Tải model R2, kiểm tra và nạp ngay YOLO/ResNet       |
| `GET /api/status`                                     | Kiểm tra file model và kết nối Rasa                            |
| `POST /api/predict`                                   | JSON `{"image": "<base64 hoặc data URL>", "confidence": 0.25}` |
| `POST /api/chat`                                      | JSON `{"message": "Xin chào", "sender": "user-1"}`             |

Ví dụ gọi nhận diện từ Python:

```python
import base64
import json
from pathlib import Path
from urllib.request import Request, urlopen

payload = {
    "image": base64.b64encode(Path("../samples/sample_aothun.jpg").read_bytes()).decode(),
    "confidence": 0.25,
}
request = Request("http://127.0.0.1:8080/api/predict",
                  data=json.dumps(payload).encode(),
                  headers={"Content-Type": "application/json"})
with urlopen(request, timeout=120) as response:
    print(json.load(response))
```

Trạng thái bật/tắt được lưu trong `manager.sqlite3`. Nhận diện cần cả YOLO và
ResNet được bật. Model được nạp khi nhận yêu cầu nhận diện đầu tiên.
Model được upload, lưu phiên bản và kích hoạt trên Cloudflare R2. Không cần file model trong project.
Chỉ dùng checkpoint do bạn huấn luyện hoặc từ nguồn tin cậy vì YOLO dùng định dạng
PyTorch chứa đối tượng Python. ResNet cần kiến trúc ResNet18 và ít nhất 3 nhãn.
File upload được lưu tạm ngoài project để kiểm tra, sau đó upload R2 và xóa bản tạm.
Model chạy inference được tải từ R2 về cache trong thư mục tạm của hệ điều hành;
cache có thể xóa và sẽ tự tải lại. R2 là nguồn lưu chính, database chỉ lưu metadata.
Việc kiểm tra checkpoint không bảo đảm độ chính xác của model trên dữ liệu thực tế.
Chat cần chạy dịch vụ Rasa riêng như hướng dẫn bên dưới. Thay file Rasa cần khởi
động lại dịch vụ Rasa; thao tác nạp lại trong UI chỉ xóa cache YOLO/ResNet.
Ứng dụng dành cho localhost, chưa có đăng nhập để triển khai công khai.
Đổi cổng bằng biến môi trường `PORT` nếu cổng 8080 đang được sử dụng.

### Cấu hình .env và Cloudflare R2

Ứng dụng tự đọc `.env` trong `model_final`; `.env.example` là mẫu không chứa bí mật.
Biến môi trường hệ thống ưu tiên hơn giá trị trong file.

```dotenv
HOST=127.0.0.1
PORT=8080
RASA_URL=http://127.0.0.1:5005
MAX_MODEL_MB=256
R2_ENABLED=true
R2_ENDPOINT_URL=https://YOUR_ACCOUNT_ID.r2.cloudflarestorage.com
R2_ACCESS_KEY_ID=your-access-key
R2_SECRET_ACCESS_KEY=your-secret-key
R2_BUCKET=your-private-bucket
R2_PREFIX=fashion-ai/models
```

Tạo bucket R2 riêng và API key có quyền đọc/ghi object cho bucket đó, điền vào `.env`
rồi khởi động lại ứng dụng. Dùng nút **Kiểm tra kết nối** trong trang quản lý.
R2 bắt buộc được bật. Mỗi lần upload tạo một object phiên bản riêng và cập nhật
`{R2_PREFIX}/{model_id}/active.json` để chỉ định phiên bản đang dùng.
Nếu lưu R2 thất bại, file đang hoạt động được giữ nguyên. File upload đi qua backend,
không đưa khóa R2 ra frontend và không cần mở public bucket.
Khôi phục chuyển manifest sang phiên bản R2 cũ, không upload lại hay lưu bản model local.
Model tải để chạy được kiểm tra SHA-256. API `GET /api/models/{id}/download`
chuyển hướng đến URL R2 ký sẵn có hiệu lực 5 phút, dùng cho tải file hoặc dịch vụ Rasa.
Các model hiện có không tự đồng bộ lên R2 chỉ vì bật cấu hình.
Tích hợp theo [Cloudflare R2 boto3](https://developers.cloudflare.com/r2/examples/aws/boto3/).

### Build và chạy Docker

Docker Desktop cần chạy Linux containers. Từ `model_final`:

```powershell
docker compose up --build -d
docker compose logs -f ai-manager
```

Mở http://127.0.0.1:8080. Compose đọc `.env`, lưu database và lịch sử trong volume
`manager-data`. Không mount thư mục model; model được đọc từ R2 và cache tạm khi chạy. `.env` không được đóng gói vào image. Docker dùng CPU và Python 3.11.
Rasa trên Windows host được gọi qua `host.docker.internal:5005`; tùy chỉnh bằng
`DOCKER_RASA_URL` trong `.env` nếu chạy Rasa ở địa chỉ khác.

Build image riêng:

```powershell
docker build -t fashion-ai-manager .
```

## Server ảnh và giao diện

```powershell
python -m pip install -r requirements.txt
python server.py
```

Mở http://127.0.0.1:8080. Không mở trực tiếp index.html vì trang cần API Python.

Nếu cổng 8080 bị chiếm hoặc Windows chặn, chọn cổng khác:

```powershell
$env:PORT = '8088'
python server.py
```

Sau đó mở http://127.0.0.1:8088.

YOLO nhận diện AoThun, AoSoMi, QuanDai, VayNu. ResNet18 phân loại màu theo danh sách nhãn và cấu hình tiền xử lý trong checkpoint. Chỉ các vùng quần áo YOLO phát hiện được cắt để phân loại màu. Nếu không phát hiện quần áo, server không chạy phân loại màu.

## Rasa

Trong môi trường Python tương thích với phiên bản Rasa đã dùng để huấn luyện, chạy từ thư mục dự án:

```powershell
rasa run --enable-api --credentials credentials.yml --model http://127.0.0.1:8080/api/models/rasa/download --port 5005
```

Rasa cần môi trường riêng nếu phiên bản Python hiện tại không tương thích. Python 3.14 trên máy hiện tại chưa chạy được CLI Rasa. Nếu bot có custom actions, cần mã nguồn actions và action server của dự án huấn luyện; file model không chứa mã nguồn đó.

Mặc định server gọi http://127.0.0.1:5005. Đổi địa chỉ nếu cần:

```powershell
$env:RASA_URL = 'http://127.0.0.1:5005'
python server.py
```

Server chỉ lắng nghe localhost. Ảnh được xử lý trong bộ nhớ, không lưu lên đĩa.

Nhấn **Lấy model từ R2** để tải các phiên bản active về cache tạm và nạp YOLO/ResNet.
Nếu trang báo chưa upload dù bucket có model, kiểm tra `R2_PREFIX` khớp đường dẫn
manifest `fashion-ai/models/{id}/active.json` và khởi động lại để nhận `.env` mới.

### Đăng nhập

Tài khoản mẫu: `admin` / `Admin@123`. Backend so sánh trực tiếp với
`ADMIN_USERNAME` và `ADMIN_PASSWORD` trong `.env`; frontend không giữ mật khẩu.
Sau khi đăng nhập, cookie HttpOnly lưu phiên 8 giờ; đăng xuất hủy phiên trên server.
Trang quản lý, demo, docs và các API quản lý model đều yêu cầu phiên đăng nhập.
`POST /api/predict` được gọi trực tiếp, không cần đăng nhập hoặc cookie Python.
Service Java gửi JSON ảnh base64 và confidence thẳng tới API này.
`POST /api/chat` vẫn yêu cầu đăng nhập; yêu cầu đăng nhập của chat trên store giữ riêng.
`GET /health` dành cho Docker không cần đăng nhập. Đặt `COOKIE_SECURE=true` khi dùng HTTPS.
Đổi mật khẩu mẫu trước khi deploy. Client gọi API quản trị cần đăng nhập qua
`POST /api/auth/login` và gửi lại cookie; API nhận diện không cần bước này.
API tải Rasa cũng cần cookie; tải archive bằng phiên đăng nhập rồi đưa vào dịch vụ Rasa.
