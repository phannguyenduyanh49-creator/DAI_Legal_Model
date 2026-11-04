# DAI Legal Model

**Trợ lý AI hỗ trợ tra cứu pháp luật và thủ tục hành chính Việt Nam**  
Academic research prototype | React + FastAPI + RAG + Qwen2.5-7B + MCP

DAI Legal Model là dự án nghiên cứu xây dựng trợ lý AI giúp người dùng đặt câu hỏi về pháp luật và thủ tục hành chính bằng tiếng Việt. Hệ thống kết hợp mô hình ngôn ngữ lớn, truy xuất tài liệu nội bộ (RAG), xây dựng bằng chứng (Evidence Builder), kiểm tra trích dẫn (Verifier) và MCP để truy vấn một số nguồn dữ liệu bên ngoài khi được người dùng cho phép.

> **Lưu ý:** Đây là sản phẩm thử nghiệm phục vụ học tập và nghiên cứu, không phải dịch vụ tư vấn pháp lý chuyên nghiệp. Câu trả lời của AI không thay thế văn bản pháp luật hoặc ý kiến của luật sư/cơ quan có thẩm quyền. Với thủ tục, thời hạn, lệ phí và quy định quan trọng, hãy kiểm tra lại nguồn chính thức và hiệu lực văn bản.

## Mục lục

1. [Giới thiệu dự án](#1-giới-thiệu-dự-án)
2. [Chức năng chính](#2-chức-năng-chính)
3. [Kiến trúc hệ thống](#3-kiến-trúc-hệ-thống)
4. [Công nghệ sử dụng](#4-công-nghệ-sử-dụng)
5. [Cấu trúc thư mục](#5-cấu-trúc-thư-mục)
6. [Yêu cầu môi trường](#6-yêu-cầu-môi-trường)
7. [Cài đặt](#7-cài-đặt)
8. [Cấu hình trước khi chạy](#8-cấu-hình-trước-khi-chạy)
9. [Khởi chạy hệ thống](#9-khởi-chạy-hệ-thống)
10. [Hướng dẫn sử dụng](#10-hướng-dẫn-sử-dụng)
11. [API cơ bản](#11-api-cơ-bản)
12. [Chạy kiểm thử](#12-chạy-kiểm-thử)
13. [Xử lý lỗi thường gặp](#13-xử-lý-lỗi-thường-gặp)
14. [Giới hạn và an toàn](#14-giới-hạn-và-an-toàn)

---

## 1. Giới thiệu dự án

DAI Legal Model hướng đến việc hỗ trợ người dùng tra cứu thông tin pháp luật và thủ tục hành chính Việt Nam thông qua giao diện hội thoại.

Thay vì chỉ yêu cầu mô hình trả lời dựa trên kiến thức đã học, hệ thống sử dụng RAG để tìm kiếm tài liệu liên quan, đóng gói nội dung tìm được thành Evidence và kiểm tra mối liên hệ giữa câu trả lời với các nguồn được truy xuất.

Mục tiêu chính:

- Hỗ trợ hỏi đáp pháp luật bằng tiếng Việt.
- Tìm kiếm thông tin trong kho dữ liệu nội bộ theo từng lĩnh vực.
- Hiển thị nguồn và căn cứ được sử dụng trong câu trả lời.
- Hạn chế câu trả lời không có bằng chứng hoặc trích dẫn không hợp lệ.
- Cho phép truy vấn nguồn bên ngoài thông qua MCP theo chính sách và sự đồng ý của người dùng.

## 2. Chức năng chính

| Chức năng | Mô tả |
|---|---|
| Hỏi đáp pháp luật | Tiếp nhận câu hỏi tự nhiên bằng tiếng Việt. |
| Phân loại lĩnh vực | Định tuyến câu hỏi tới domain phù hợp. |
| Local RAG | Tìm kiếm tài liệu trong kho dữ liệu nội bộ trước. |
| Embedding và reranking | Biểu diễn truy vấn, tìm ứng viên và xếp hạng kết quả theo cấu hình. |
| Evidence Builder | Chuẩn hóa và đóng gói tài liệu truy xuất thành bằng chứng. |
| Citation và Verifier | Kiểm tra ID bằng chứng, trích dẫn và trạng thái xác minh. |
| MCP | Truy vấn nguồn bên ngoài được cấu hình khi chính sách cho phép và người dùng đồng ý. |
| Quản lý hội thoại | Lưu phiên và lịch sử hội thoại cục bộ theo cấu hình backend. |
| Domain documents | Hỗ trợ tải lên tài liệu .txt, .md, .xlsx theo domain. |

**Lưu ý:** HTTP 200 chỉ cho biết Gateway đã xử lý request ở tầng HTTP. Hãy đọc trạng thái trong response body và thông tin verifier/citation; không được mặc định mọi câu trả lời đều đã được xác minh.

## 3. Kiến trúc hệ thống

Luồng tổng quan:

```text
User
 |
 v
React / Vite Frontend (:5173)
 |
 v
Core Gateway (:8002)
 |
 v
Core API (:8000)
 |             |                  |
 |             |                  +--> Domain Router / Query Analysis
 |             |
 |             +--> Local Retrieval
 |                    |
 |                    +--> Embedding (BGE-M3)
 |                    +--> Qdrant
 |                    +--> Reranker (when configured)
 |                    |
 |                    v
 |                Evidence Builder
 |                    |
 |                    v
 |                Prompt Assembly
 |                    |
 |                    v
 |                Model API (:8001)
 |                    |
 |                    v
 |             Qwen2.5-7B-Instruct + LoRA V4
 |                    |
 |                    v
 |             Citation / Evidence Verifier
 |
 +--> MCP (permission-controlled external retrieval)
          |
          v
     Configured external source
```

### Luồng xử lý câu hỏi

1. Người dùng chọn lĩnh vực và gửi câu hỏi từ Frontend.
2. Gateway tiếp nhận request và chuyển tới Core API.
3. Query Analysis / Domain Router phân tích ý định và lĩnh vực.
4. Hệ thống ưu tiên truy xuất dữ liệu nội bộ (Local RAG).
5. Kết quả được lọc, xếp hạng và chuyển tới Evidence Builder.
6. Prompt Assembly tạo ngữ cảnh có bằng chứng cho Model API.
7. Qwen2.5-7B-Instruct kết hợp LoRA V4 sinh câu trả lời.
8. Citation và Verifier kiểm tra các thông tin liên quan đến Evidence.
9. Gateway trả kết quả cùng trạng thái xử lý cho Frontend.

### Nguyên tắc MCP

MCP là lớp truy xuất nguồn bên ngoài, không thay thế Qdrant và Local RAG. Khi dữ liệu nội bộ chưa đủ, hệ thống áp dụng luồng xin phép người dùng trước khi gọi nguồn bên ngoài theo policy. MCP hiện được mô tả trong dự án với tích hợp Cổng Dịch vụ công Quốc gia; không nên hiểu đây là công cụ tìm kiếm Internet không giới hạn.

## 4. Công nghệ sử dụng

| Thành phần | Công nghệ |
|---|---|
| Frontend | React 19, Vite, Tailwind CSS, Lucide React |
| Backend | Python, FastAPI |
| LLM | Qwen2.5-7B-Instruct |
| Fine-tuning adapter | LoRA V4 |
| Quantization | 4-bit NF4, compute dtype BF16 (theo cấu hình Model API) |
| Embedding | BGE-M3 cho pipeline Local RAG được mô tả |
| Vector database | Qdrant |
| Reranking | BGE Reranker (khi được cấu hình) |
| External tools | MCP |
| Database / search bổ sung | PostgreSQL, Elasticsearch (nhánh triển khai tùy chọn) |
| Testing | Pytest, Node.js test runner |

## 5. Cấu trúc thư mục

Cấu trúc rút gọn:

```text
DAI-Legal-Model/
├── FE/                         # React/Vite frontend
├── backend/
│   ├── answer/                 # Generation, citation, diagnostics
│   ├── catalog/                # Catalog lookup và database
│   ├── evidence/               # Evidence Builder, normalization
│   ├── mcp/                    # MCP adapter và tích hợp nguồn ngoài
│   ├── rag/                    # Retrieval pipeline và policy
│   ├── search/                 # PostgreSQL / Elasticsearch (tùy chọn)
│   ├── tests/                  # Backend tests
│   ├── verifier/               # Evidence / answer verification
│   ├── core_gateway.py         # Gateway API :8002
│   ├── main.py                 # Core API :8000
│   └── model_api.py            # Model API :8001
├── configs/                    # Cấu hình mẫu
├── data/                       # Dataset và dữ liệu cục bộ
├── evals/                      # Công cụ đánh giá
├── mcp_server.py               # MCP server entry point
├── run_gateway_stack.py        # Khởi chạy 3 Python services
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

Một số file dữ liệu, model weights, LoRA adapter hoặc index có thể không được lưu trong Git vì dung lượng lớn hoặc điều kiện phân phối.

## 6. Yêu cầu môi trường

- Windows 10/11 hoặc môi trường tương thích.
- Python phiên bản tương thích với các package trong requirements.txt.
- Node.js và npm.
- Git.
- Docker Desktop / Docker Engine để chạy Qdrant và PostgreSQL bằng Docker Compose.
- GPU NVIDIA/CUDA được khuyến nghị để chạy model 7B với cấu hình 4-bit. Khả năng chạy thực tế phụ thuộc VRAM, RAM, CUDA, PyTorch và phiên bản bitsandbytes.
- Có sẵn model Qwen2.5-7B-Instruct, LoRA V4 adapter và dữ liệu/index cần thiết.

Kiểm tra công cụ:

```powershell
python --version
node --version
npm --version
docker --version
git --version
```

## 7. Cài đặt

### Bước 1. Clone repository

```powershell
git clone https://github.com/phannguyenduyanh49-creator/DAI_Legal_Model.git
cd DAI_Legal_Model
```

### Bước 2. Tạo môi trường Python

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
```

Nếu PowerShell chặn việc activate môi trường, có thể gọi trực tiếp Python trong môi trường:

```powershell
.\\.venv\\Scripts\\python.exe --version
```

### Bước 3. Cài dependencies backend

```powershell
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

Nếu cài PyTorch/CUDA gặp lỗi, cần đối chiếu phiên bản CUDA, driver NVIDIA và hướng dẫn cài PyTorch tương ứng với môi trường máy. Không tự ý thay đổi phiên bản package đã pin nếu chưa kiểm tra tương thích.

### Bước 4. Cài dependencies frontend

Mở terminal tại thư mục repository:

```powershell
cd FE
npm install
```

Sau khi cài xong, quay về thư mục gốc nếu cần chạy backend:

```powershell
cd ..
```

## 8. Cấu hình trước khi chạy

### 8.1. Kiểm tra đường dẫn model và project

Một số đường dẫn trong backend hiện đang được khai báo trực tiếp theo môi trường phát triển Windows. Trước khi chạy trên máy khác, hãy kiểm tra và sửa cho khớp với vị trí repository và model adapter:

- backend/main.py: biến BASE_DIR và đường dẫn tới mcp_server.py.
- backend/model_api.py: biến LORA_MODEL trỏ tới thư mục adapter LoRA V4.

Ví dụ thư mục adapter:

```text
outputs/qwen-legal-lora-v4/
```

Đảm bảo thư mục adapter có các file cần thiết (ví dụ adapter_config.json, adapter_model.safetensors). Không commit model weights hoặc adapter nếu chưa xác nhận quyền phân phối.

### 8.2. Frontend Gateway URL

Có thể tạo file FE/.env:

```env
VITE_GATEWAY_API_URL=http://127.0.0.1:8002
```

Đây là địa chỉ Gateway local mặc định. Không đưa API key, mật khẩu hoặc thông tin nhạy cảm vào file được commit.

### 8.3. Khởi động Qdrant và PostgreSQL

Từ thư mục gốc repository:

```powershell
docker compose -f backend/docker-compose.yml up -d qdrant postgres
```

Kiểm tra container:

```powershell
docker compose -f backend/docker-compose.yml ps
```

Qdrant mặc định sử dụng:

- REST API: http://127.0.0.1:6333
- Dashboard: http://127.0.0.1:6333/dashboard

PostgreSQL được khai báo trong Compose cho các thành phần catalog/search tương ứng. Qdrant vẫn là Local RAG backend mặc định theo cấu hình hiện tại.

### 8.4. Dữ liệu và vector collection

Hệ thống cần có dữ liệu pháp lý đã được nạp vào kho tương ứng trước khi có thể trả lời dựa trên tài liệu nội bộ.

Repository có script import Dịch vụ công:

```powershell
.\\.venv\\Scripts\\python.exe backend/import_dvc_to_qdrant.py
```

Chỉ chạy import sau khi đã đọc script, xác nhận nguồn dữ liệu, collection đích và hiểu rõ tác động lên dữ liệu đang có. Một script import khác trong repository có thể tạo lại collection; không chạy trên môi trường có dữ liệu quan trọng nếu chưa backup và xác nhận.

## 9. Khởi chạy hệ thống

Cần giữ các terminal đang chạy dịch vụ mở trong suốt quá trình sử dụng.

### Terminal 1 – Khởi chạy backend stack

Tại thư mục gốc repository:

```powershell
.\\.venv\\Scripts\\python.exe run_gateway_stack.py
```

Script lần lượt khởi chạy:

| Service | URL | Vai trò |
|---|---|---|
| Core API | http://127.0.0.1:8000 | Retrieval và xử lý câu hỏi |
| Model API | http://127.0.0.1:8001 | Qwen + LoRA inference |
| Core Gateway | http://127.0.0.1:8002 | API cho Frontend |

Model API cần tải model và adapter khi khởi động. Lần chạy đầu có thể mất thời gian và cần đủ bộ nhớ. Launcher giám sát tiến trình con nhưng không thay thế việc kiểm tra health/readiness của từng service.

Dừng stack bằng Ctrl+C.

### Terminal 2 – Khởi chạy Frontend

```powershell
cd FE
npm run dev
```

Mở URL Vite hiển thị trong terminal, mặc định thường là:

```text
http://localhost:5173
```

### Kiểm tra Gateway

```powershell
Invoke-RestMethod http://127.0.0.1:8002/health
```

Nếu frontend không kết nối được backend, kiểm tra VITE_GATEWAY_API_URL, CORS và port 8002.

## 10. Hướng dẫn sử dụng

### 10.1. Đặt câu hỏi pháp lý

1. Mở giao diện tại http://localhost:5173.
2. Chọn lĩnh vực/domain phù hợp nếu giao diện có hiển thị lựa chọn.
3. Nhập câu hỏi pháp lý cụ thể vào khung chat.
4. Gửi câu hỏi và chờ hệ thống truy xuất tài liệu, xử lý bằng chứng và tạo câu trả lời.
5. Đọc nội dung trả lời cùng phần nguồn/citation và trạng thái xác minh.

Ví dụ câu hỏi:

- “Hồ sơ đăng ký thành lập doanh nghiệp gồm những giấy tờ gì?”
- “Người lao động được nghỉ phép năm tối thiểu bao nhiêu ngày?”
- “Thủ tục đăng ký tạm trú được thực hiện như thế nào?”
- “Điều kiện để kinh doanh dịch vụ lưu trú du lịch là gì?”

Nên nêu rõ bối cảnh, thời điểm và địa phương nếu các yếu tố đó ảnh hưởng đến câu trả lời.

### 10.2. Sử dụng MCP khi Local RAG chưa đủ dữ liệu

1. Đặt câu hỏi như bình thường.
2. Hệ thống ưu tiên tìm kiếm trong dữ liệu nội bộ.
3. Nếu cần nguồn ngoài, giao diện có thể hiển thị yêu cầu cho phép truy vấn.
4. Đọc nội dung yêu cầu và chỉ chọn đồng ý nếu muốn hệ thống gọi nguồn ngoài.
5. Sau khi MCP trả dữ liệu, kiểm tra nguồn, nội dung và thời điểm của tài liệu được sử dụng.

Nếu không đồng ý, hệ thống không được tự ý truy cập nguồn ngoài. MCP chỉ truy vấn các nguồn và công cụ đã cấu hình, không mặc định là tìm kiếm web tổng quát.

### 10.3. Tải tài liệu vào domain

Gateway cung cấp endpoint:

- GET /api/v2/domains/
- POST /api/v2/domains/{domain_id}/documents

Các định dạng được mô tả là hỗ trợ: .txt, .md, .xlsx; giới hạn upload hiện tại là 2 MiB. Tài liệu được gắn với domain được chọn để tránh trộn dữ liệu giữa các lĩnh vực.

Với Qdrant backend mặc định, collection upload có dạng domain_documents_{domain_id}. Tài liệu upload không tự động có URL công khai; hệ thống nhận diện theo filename, document ID và chunk location.

### 10.4. Đọc trạng thái câu trả lời

Không nên chỉ nhìn nội dung câu trả lời. Hãy kiểm tra:

- Evidence có được truy xuất hay không.
- Citation có tham chiếu tới Evidence ID hợp lệ không.
- Verifier có thực sự chạy và đạt trạng thái passed không.
- Câu trả lời là verified, unverified, partial hay chưa đủ dữ liệu theo response backend.

Nếu hệ thống báo chưa đủ bằng chứng hoặc chưa xác minh, không sử dụng câu trả lời đó như kết luận pháp lý đã được kiểm chứng.

## 11. API cơ bản

Gateway mặc định chạy tại http://127.0.0.1:8002.

| Method | Endpoint | Mục đích |
|---|---|---|
| GET | /health | Kiểm tra Gateway |
| POST | /api/v2/legal-query | Gửi câu hỏi pháp lý |
| GET | /api/v2/domains/ | Lấy danh sách domain |
| POST | /api/v2/domains/{domain_id}/documents | Upload tài liệu theo domain |

Ví dụ request:

```json
{
  "query": "Thủ tục đăng ký thành lập doanh nghiệp gồm những gì?",
  "domain_hint": "enterprise_law",
  "allow_external": false
}
```

Đây là ví dụ minh họa các trường request cơ bản; schema thực tế có thể có thêm trường tùy phiên bản. Xem gateway_openapi.json hoặc OpenAPI của service đang chạy để biết schema đầy đủ.

- Gateway OpenAPI: http://127.0.0.1:8002/docs
- Core API OpenAPI: http://127.0.0.1:8000/docs
- Model API OpenAPI: http://127.0.0.1:8001/docs

## 12. Chạy kiểm thử

### Backend

Từ thư mục gốc:

```powershell
.\\.venv\\Scripts\\python.exe -m pytest backend/tests -q
```

### Frontend

Từ thư mục FE:

```powershell
npm test
npm run build
```

Nếu Windows không thể xóa thư mục dist do file lock, dừng dev server rồi thử build sang thư mục khác:

```powershell
npm run build -- --outDir build-check
```

## 13. Xử lý lỗi thường gặp

| Hiện tượng | Nguyên nhân cần kiểm tra |
|---|---|
| Frontend không kết nối Gateway | Gateway chưa chạy, sai VITE_GATEWAY_API_URL, sai port hoặc CORS. |
| Port 8000/8001/8002 đã được sử dụng | Một tiến trình cũ đang chạy; kiểm tra tiến trình trước khi dừng. |
| Qdrant connection error | Container chưa chạy, sai URL hoặc port 6333 chưa mở. |
| Không tìm thấy Evidence | Collection chưa có dữ liệu, sai domain/collection hoặc retrieval gặp lỗi. |
| MCP không được gọi | Local evidence có thể đã đủ, người dùng chưa cho phép, policy chặn hoặc MCP upstream không khả dụng. |
| Model API không khởi động | Sai đường dẫn LoRA, thiếu model, thiếu VRAM/RAM hoặc lỗi tương thích CUDA/bitsandbytes. |
| Citation/verifier không đạt | Evidence ID thiếu, citation không khớp, grounding chưa đạt hoặc verifier chưa hoàn tất. |
| PostgreSQL connection error | Container/database chưa sẵn sàng hoặc cấu hình kết nối không đúng. |

HTTP 200 không đồng nghĩa với truy xuất thành công hoặc câu trả lời đã được xác minh. Hãy xem log theo request/trace ID và các trường trạng thái trong response.

## 14. Giới hạn và an toàn

- Dự án là prototype học thuật, chưa phải hệ thống tư vấn pháp lý thương mại.
- Chất lượng phụ thuộc dữ liệu, độ bao phủ retrieval, embedding, reranker, Evidence Builder và model.
- Citation hợp lệ không tự động chứng minh nội dung pháp lý đúng.
- Văn bản pháp luật có thể được sửa đổi, thay thế hoặc hết hiệu lực; cần xác minh hiệu lực tại thời điểm sử dụng.
- MCP chỉ được sử dụng theo nguồn, công cụ và policy đã cấu hình.
- Không đưa mật khẩu, API key, dữ liệu cá nhân hoặc hồ sơ nhạy cảm vào repository.
- Không commit .venv/, node_modules/, file .env, chat history, model checkpoints, adapter weights hoặc build output.
- Trước khi chia sẻ dataset hoặc model artifacts, cần kiểm tra quyền sử dụng và điều kiện phân phối.
- Các service mặc định bind vào loopback trong môi trường local. Không expose ra Internet nếu chưa bổ sung xác thực, phân quyền, TLS và các biện pháp bảo mật phù hợp.

---

**DAI Legal Model — Academic Research Prototype**  
Developed for learning and research on Vietnamese legal question answering, Retrieval-Augmented Generation (RAG), evidence grounding, citation verification and MCP-based external retrieval.
