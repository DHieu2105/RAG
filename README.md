# UniGuide AI

UniGuide AI là trợ lý hỏi đáp dành cho sinh viên, sử dụng kiến trúc Retrieval-Augmented Generation (RAG) để tìm kiếm thông tin trong sổ tay sinh viên và cung cấp câu trả lời có ngữ cảnh.

## Tech Stack

- **Frontend:** Next.js 16, React 19, TypeScript, Tailwind CSS 4
- **Backend:** Python 3.12, FastAPI, Uvicorn
- **Document processing:** PyMuPDF
- **Data pipeline:** PDF loading, text preprocessing và chunking cho RAG

## Cấu Trúc Dự Án

```text
RAG-simple/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   └── rag/
│   └── requirements.txt
├── data/
│   ├── raw/
│   └── processed/
└── frontend/
```

Các file trong `data/raw` và `data/processed` không được đưa lên Git. Đặt tài liệu nguồn tại `data/raw/handbook_k67.pdf` trước khi chạy pipeline xử lý dữ liệu.

## Yêu Cầu

- Python 3.12+
- Node.js 20+
- npm

## Chạy Backend

```bash
cd backend
python -m venv .venv
```

Kích hoạt virtual environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS/Linux
source .venv/bin/activate
```

Cài dependencies và chạy FastAPI:

```bash
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

Backend chạy tại `http://localhost:8000`:

- Health check: `GET http://localhost:8000/health`
- Swagger UI: `http://localhost:8000/docs`

## Chạy Frontend

Mở terminal khác:

```bash
cd frontend
npm install
npm run dev
```

Frontend chạy tại `http://localhost:3000`.

## Chuẩn bị deploy Vercel + Cloudflare Tunnel

Luồng request sau khi deploy:

```text
Trình duyệt -> Vercel /api/chat -> Cloudflare Tunnel -> FastAPI -> RAG/Ollama
```

Trình duyệt không gọi trực tiếp FastAPI và không nhận được token backend. Route
`frontend/src/app/api/chat/route.ts` chạy trên Vercel, tự gắn Bearer token rồi
chuyển tiếp request tới máy Windows.

### 1. Tạo token dùng chung

Tạo một token ngẫu nhiên dài:

```powershell
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Không commit token này lên GitHub và không đặt tên biến có tiền tố
`NEXT_PUBLIC_`.

### 2. Chạy local

Tạo `backend/.env` từ `backend/.env.example`, sau đó đặt token vừa tạo:

```dotenv
UNIGUIDE_API_TOKEN=<token-cua-ban>
UNIGUIDE_CORS_ORIGINS=http://localhost:3000
UNIGUIDE_RETRIEVAL_MODE=hybrid
```

Tạo `frontend/.env.local` từ `frontend/.env.example`:

```dotenv
UNIGUIDE_BACKEND_URL=http://localhost:8000
UNIGUIDE_API_TOKEN=<cung-token-o-tren>
```

Sau đó chạy backend và frontend như bình thường. Frontend luôn gọi `/api/chat`;
proxy của Next.js sẽ gọi `http://localhost:8000/api/v1/chat` khi chạy local.

### 3. Cấu hình khi deploy

Trên máy Windows:

- Chạy FastAPI tại `http://localhost:8000`.
- Đặt `UNIGUIDE_API_TOKEN` trong `backend/.env`.
- Cho Cloudflare Tunnel chuyển tiếp hostname HTTPS tới
  `http://localhost:8000`.

Trên Vercel, cấu hình hai Environment Variables cho project frontend:

```dotenv
UNIGUIDE_BACKEND_URL=https://api.ten-mien-cua-ban.example
UNIGUIDE_API_TOKEN=<cung-token-o-backend>
```

Redeploy frontend sau khi thêm hoặc đổi biến môi trường. Không cần thêm domain
Vercel vào CORS khi dùng proxy cùng origin này; `UNIGUIDE_CORS_ORIGINS` chỉ dùng
cho trường hợp một trình duyệt được phép gọi thẳng FastAPI.

### 4. Kiểm tra trước khi công khai

```powershell
Test-Path .\frontend\src\app\api\chat\route.ts
```

Kết quả phải là `True`. Kiểm tra thêm:

- `GET https://<hostname-tunnel>/health` trả `{"status":"ok"}`.
- Gọi `/api/v1/chat` không có Bearer token phải trả HTTP `401`.
- Gửi câu hỏi từ website Vercel phải đi qua `/api/chat` và nhận câu trả lời.

Token chỉ ngăn truy cập trực tiếp trái phép vào FastAPI. Vì `/api/chat` là API
công khai cho người dùng website, nên trước khi chia sẻ rộng cần cấu hình thêm
rate limiting/WAF trên Vercel hoặc Cloudflare để tránh lạm dụng tài nguyên máy.

## Chạy Preprocessing

Sau khi đặt PDF vào `data/raw/handbook_k67.pdf`, chạy:

```bash
cd backend
python -m app.rag.preprocessing
```

Các chunks sau xử lý được ghi vào `data/processed/handbook_k67.jsonl`.
