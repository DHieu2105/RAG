# UniGuide AI — Trợ Lý Tra Cứu Sổ Tay Sinh Viên (RAG)

UniGuide AI là trợ lý ảo hỏi đáp thông minh dành cho sinh viên, sử dụng kiến trúc **RAG (Retrieval-Augmented Generation)** kết hợp tìm kiếm đa tầng (**TF-IDF + BM25 + Vector Embedding**) và LLM local qua **Ollama**.

Hệ thống hỗ trợ tra cứu chính xác quy chế đào tạo tín chỉ, học bổng, điểm rèn luyện, chuẩn đầu ra... có trích dẫn rõ ràng (Điều, Khoản, số trang). Dự án được thiết kế linh hoạt, dễ dàng áp dụng cho sổ tay của bất kỳ trường nào.

---

## 📋 Yêu Cầu Cần Chuẩn Bị

Cài đặt sẵn trên máy tính:

1. **Python 3.12+**
2. **Node.js 20+** & **npm**
3. **[Ollama](https://ollama.com/)** (để chạy AI trên máy của bạn)

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy Local Từ A Đến Z

### Bước 1: Clone repo về máy

```bash
git clone <URL_REPO>
cd RAG-simple
```

---

### Bước 2: Bật mô hình AI với Ollama

1. Tải và cài đặt Ollama từ [ollama.com](https://ollama.com).
2. Mở terminal bất kỳ và tải model về máy:
   ```bash
   ollama pull qwen3:4b
   ```

   *(Nếu máy cấu hình vừa phải, bạn có thể dùng `ollama pull qwen2.5:3b`)*.

---

### Bước 3: Đặt tài liệu Sổ tay & Logo trường

1. Đặt file PDF sổ tay sinh viên vào thư mục `data/raw/` (hệ thống sẽ tự động quét và nhận file `.pdf` bất kỳ).
2. *(Tùy chọn)* Đặt file logo trường vào `frontend/public/logo.png`.

---

### Bước 4: Thiết lập Backend

1. Mở terminal tại thư mục `backend`:

   ```bash
   cd backend
   ```
2. Tạo và kích hoạt môi trường ảo:

   - **Windows (PowerShell):**
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   - **macOS / Linux:**
     ```bash
     python -m venv .venv
     source .venv/bin/activate
     ```
3. Cài đặt các thư viện cần thiết:

   ```bash
   pip install -r requirements.txt
   ```
4. Tạo file cấu hình `backend/.env`:

   - Copy từ file `.env.example`:
     - Windows: `copy .env.example .env`
     - macOS/Linux: `cp .env.example .env`
   - File `backend/.env` chạy local cơ bản:
     ```dotenv
     UNIGUIDE_API_TOKEN=uniguide-secret-local-token
     UNIGUIDE_CORS_ORIGINS=http://localhost:3000
     UNIGUIDE_RETRIEVAL_MODE=hybrid
     OLLAMA_URL=http://127.0.0.1:11434/api/chat
     OLLAMA_MODEL=qwen3:4b
     ```

---

### Bước 5: Xử lý dữ liệu Sổ tay (Chạy 1 lần đầu)

Đảm bảo terminal đang ở thư mục `backend` và đã kích hoạt môi trường ảo (`.venv`):

1. **Cắt đoạn văn bản (Preprocessing):**

   ```bash
   python -m app.rag.preprocessing
   ```
2. **Tạo Vector Embeddings (mô hình BAAI/bge-m3):**

   ```bash
   python -m app.rag.embeddings
   ```

> 💡 *Chỉ mục TF-IDF & BM25 sẽ được hệ thống tự động sinh khi ứng dụng khởi chạy.*

---

### Bước 6: Thiết lập Frontend

1. Mở một cửa sổ terminal mới và vào thư mục `frontend`:

   ```bash
   cd frontend
   npm install
   ```
2. Tạo file cấu hình `frontend/.env.local`:

   - Tạo file `frontend/.env.local` với nội dung:
     ```dotenv
     UNIGUIDE_BACKEND_URL=http://localhost:8000
     UNIGUIDE_API_TOKEN=uniguide-secret-local-token
     ```

   *(Lưu ý: `UNIGUIDE_API_TOKEN` phải giống nhau ở cả `backend/.env` và `frontend/.env.local`)*.

---

## 🏃 Chạy Ứng Dụng

### Cách 1: Chạy giao diện Web (Fullstack)

Mở **2 terminal** song song:

- **Terminal 1 — Backend (FastAPI):**

  ```bash
  cd backend
  # Đảm bảo đã activate .venv
  uvicorn app.main:app --reload
  ```

  Backend chạy tại: `http://localhost:8000` (Tài liệu API: `http://localhost:8000/docs`).
- **Terminal 2 — Frontend (Next.js):**

  ```bash
  cd frontend
  npm run dev
  ```

  Mở trình duyệt truy cập: **`http://localhost:3000`**.

---

### Cách 2: Chạy trực tiếp trên Terminal (CLI Test)

Nếu chỉ muốn test nhanh câu trả lời của AI và thuật toán tìm kiếm RAG mà không cần bật web:

```bash
cd backend
python run_rag_local.py
```

Nhập câu hỏi trực tiếp trên terminal (gõ `exit` để thoát).

---

## 🔄 Cách Đổi Sổ Tay Của Trường Khác

Giao diện đã được thiết kế sẵn dạng tổng quát. Khi muốn đổi sổ tay trường khác:

1. Đặt file PDF sổ tay mới vào `data/raw/` (xóa file cũ).
2. *(Tùy chọn)* Thay logo mới tại `frontend/public/logo.png`.
3. Xóa các file trong `data/processed/` (nếu có).
4. Chạy lại 2 lệnh xử lý:
   ```bash
   cd backend
   python -m app.rag.preprocessing
   python -m app.rag.embeddings
   ```
5. Khởi động lại backend là xong!

---

## 🧪 Kiểm Thử (Testing)

Chạy bộ unit test tự động để kiểm tra toàn bộ pipeline:

```bash
cd backend
pytest
```

*(138/138 tests sẽ tự động chạy và xác minh hệ thống hoạt động chính xác)*.
