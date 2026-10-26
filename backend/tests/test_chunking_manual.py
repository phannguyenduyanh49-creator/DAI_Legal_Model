
import ast
from pathlib import Path

source_path = Path(__file__).resolve().parents[1] / "domain_upload.py"
source = source_path.read_text(encoding="utf-8")
tree = ast.parse(source)

function_node = next(
    node for node in tree.body
    if isinstance(node, ast.FunctionDef)
    and node.name == "_chunk_text"
)

isolated_module = ast.Module(
    body=[function_node],
    type_ignores=[],
)
namespace = {}
exec(compile(isolated_module, str(source_path), "exec"), namespace)

chunk_text = namespace["_chunk_text"]

# Test 1: Văn bản ngắn
short_text = "Điểm đến: Đà Lạt\nMô tả: Thành phố ngàn hoa."
result = chunk_text(short_text, "recursive", 1024, 128)
assert result == [short_text], f"Test 1 FAILED: {result}"
print("Test 1 PASS: Văn bản ngắn")

# Test 2: Văn bản dài
long_text = "A" * 2500
result = chunk_text(long_text, "recursive", 1024, 128)
assert len(result) > 1, "Test 2 FAILED: Chưa chia thành nhiều chunk"
assert all(len(chunk) <= 1024 for chunk in result), "Test 2 FAILED: Chunk vượt giới hạn"
print(f"Test 2 PASS: {len(result)} chunks, max length={max(map(len, result))}")

# Test 3: Overlap
overlap_text = "A" * 1024 + "B" * 1024 + "C" * 300
result = chunk_text(overlap_text, "recursive", 1024, 128)
has_overlap = any(
    result[i - 1][-128:] in result[i]
    for i in range(1, len(result))
)
print("Test 3:", "PASS" if has_overlap else "FAIL", "- Kiểm tra overlap")
