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

## Chạy Preprocessing

Sau khi đặt PDF vào `data/raw/handbook_k67.pdf`, chạy:

```bash
cd backend
python -m app.rag.preprocessing
```

Các chunks sau xử lý được ghi vào `data/processed/handbook_k67.jsonl`.
