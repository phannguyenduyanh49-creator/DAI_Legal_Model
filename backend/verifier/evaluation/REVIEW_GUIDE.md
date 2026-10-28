# Hướng dẫn gán nhãn ứng viên claim–Evidence

## Quy trình

1. Dùng `extract_catalog_candidates.py` để xuất tối đa 50–100 cặp từ Catalog.
   Script chỉ đọc bảng `procedures`; không sửa dữ liệu DB.
2. Người review mở nguồn theo `source_record_id`/`source_url`, kiểm tra phiên
   bản và thẩm quyền của nguồn, rồi chỉnh claim thành một nhận định có thể đánh
   giá độc lập. Giữ nguyên Evidence được trích và ghi rõ nếu thiếu ngữ cảnh.
3. Để trống `gold_label` cho tới khi được đánh giá. Gán
   `review_status=in_review` khi đang làm; chỉ dùng `expert_confirmed` sau khi
   chuyên gia xác nhận và bất đồng đã được xử lý.
4. Ưu tiên ít nhất hai người có chuyên môn pháp lý; ghi quy tắc adjudication
   trong manifest riêng, không đưa thông tin cá nhân vào JSONL dùng chung.
5. Chia train/calibration/test theo `source_record_id`, văn bản pháp luật và
   thời gian. Không để cùng thủ tục hoặc điều khoản gần trùng xuất hiện ở nhiều
   split.

## Review bằng Excel

Tạo workbook review riêng (không ghi đè JSONL nguồn):

```powershell
.\.venv\Scripts\python.exe -m backend.verifier.evaluation.review_workbook export
```

Mặc định workbook được tạo tại
`backend/verifier/evaluation/review/catalog_candidates_review.xlsx`. Mỗi dòng
hiển thị claim, toàn văn Evidence, nguồn, dự đoán NLI và xác suất ba nhãn.
`gold_label`, `reviewer_notes` và `adjudication_reason` để trống; trạng thái
ban đầu được giữ từ dataset (`unreviewed`). Dự đoán chỉ là thông tin tham khảo.

31 mẫu `input_too_long` được tô vàng và giữ nguyên toàn bộ Evidence. Không sửa,
cắt hoặc bỏ mẫu đó trong workbook. Reviewer cần kiểm tra nguồn/ngữ cảnh và ghi
quyết định xử lý riêng; nếu cần tạo một biến thể có phạm vi Evidence hẹp hơn,
hãy tạo sample mới có provenance và review độc lập, đồng thời giữ mẫu gốc.

## Bất đồng và adjudication

- Hai reviewer đánh giá độc lập, không dùng dự đoán model để làm nhãn chuẩn.
- Nếu nhãn khác nhau hoặc Evidence thiếu ngữ cảnh, để nhãn chuẩn trống, đặt
  `review_status=needs_adjudication` và ghi vấn đề vào `reviewer_notes`.
- Adjudicator có chuyên môn kiểm tra nguồn gốc/phiên bản và lý do của từng bên,
  rồi ghi nhãn cuối vào `gold_label`, đặt `review_status=expert_confirmed` và
  ghi căn cứ vào `adjudication_reason`. Lưu danh tính reviewer trong manifest
  riêng có kiểm soát truy cập.
- Không thể adjudicate thì giữ nhãn trống và trạng thái cần review.

Sau khi review, tạo JSONL đã áp dụng vào một file mới:

```powershell
.\.venv\Scripts\python.exe -m backend.verifier.evaluation.review_workbook apply `
  --workbook backend/verifier/evaluation/review/catalog_candidates_review.xlsx `
  --output backend/verifier/evaluation/review/catalog_candidates_reviewed.jsonl
```

Lệnh kiểm tra sample IDs và nhãn/trạng thái hợp lệ; file gốc không bị sửa.
Runner benchmark có thể dùng JSONL mới. Chỉ các mẫu có nhãn hợp lệ và trạng
thái `expert_confirmed` mới được tính metric. Lưu dataset adjudicated thành
phiên bản bất biến trước khi đánh giá, ghi rõ version, reviewer policy và cách
chia tập; không dùng cùng nguồn/thủ tục ở cả calibration và test.

## Nhãn

- `supported`: Evidence hỗ trợ toàn bộ nội dung trọng yếu của claim.
- `unsupported`: Evidence không đủ căn cứ xác nhận claim nhưng cũng không phủ
  định claim.
- `contradicted`: Evidence trực tiếp mâu thuẫn với claim.
- `partially_supported`: Evidence hỗ trợ một phần claim ghép nhưng bỏ ngỏ hoặc
  không hỗ trợ phần trọng yếu còn lại.
- `insufficient_evidence`: Đoạn Evidence được trích thiếu phạm vi hoặc ngữ cảnh
  liên quan nên chưa thể xác định claim được hỗ trợ hay bị mâu thuẫn. Dùng nhãn
  này khi không thể đánh giá; dùng `unsupported` khi Evidence đủ để đánh giá
  nhưng không xác lập claim.

Nếu xác minh được đoạn trích thiếu phạm vi/ngữ cảnh cần thiết để trả lời claim,
dùng nhãn `insufficient_evidence`. Nếu không thể xác minh nguồn/ngữ cảnh liên
quan hoặc chưa thể phân xử nhãn, để `gold_label` trống và đặt
`review_status=needs_adjudication`. Không dùng dự đoán NLI để quyết định nhãn chuẩn.

## Giới hạn

Các cặp trích xuất tự động chỉ là ứng viên chưa gán nhãn. Chúng không chứng minh
Catalog hoặc nội dung là luật hiện hành. Bản trích xuất mặc định lấy một đoạn
nguyên văn từ một trường của mỗi bản ghi; reviewer cần xác thực nguồn, ngữ cảnh,
ngày hiệu lực, thẩm quyền và chỉnh claim để tập đánh giá không chỉ đo khả năng
nhận diện trùng văn bản.
