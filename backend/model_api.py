from fastapi import FastAPI, Header, HTTPException
import math
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch
import uvicorn
import time
import hashlib
import sys
import os
import atexit
from backend.model_runtime import (
    pytorch_cuda_memory_snapshot,
    pytorch_cuda_peak_snapshot,
)
from backend.model_api_runtime import (
    InferenceAdmission,
    ModelInstanceLock,
    with_inference_admission,
)


def configure_utf8_stdio() -> None:
    """Keep Windows console encoding from preventing API startup on Vietnamese logs."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError):
            pass


configure_utf8_stdio()

# ============================================================
# CONFIG
# ============================================================

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LORA_MODEL = r"D:\Prj\DAI-Legal-Model\outputs\qwen-legal-lora-v4"

HOST = os.getenv("MODEL_API_HOST", "127.0.0.1")
PORT = int(os.getenv("MODEL_API_PORT", "8001"))


def _configured_generation_limit() -> int:
    try:
        value = int(os.getenv("MODEL_MAX_NEW_TOKENS", "256"))
    except (TypeError, ValueError):
        value = 256
    return max(1, min(value, 512))


MODEL_MAX_NEW_TOKENS = _configured_generation_limit()


def _bounded_int_setting(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(value, maximum))


def _positive_float_setting(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return value if math.isfinite(value) and value > 0 else default


MODEL_MAX_INPUT_TOKENS = _bounded_int_setting(
    "MODEL_MAX_INPUT_TOKENS", 8192, 256, 32768
)
MODEL_INFERENCE_CONCURRENCY = _bounded_int_setting(
    "MODEL_INFERENCE_CONCURRENCY", 1, 1, 4
)
MODEL_INFERENCE_QUEUE_CAPACITY = _bounded_int_setting(
    "MODEL_INFERENCE_QUEUE_CAPACITY", 2, 0, 32
)
MODEL_INFERENCE_QUEUE_TIMEOUT_SECONDS = _positive_float_setting(
    "MODEL_INFERENCE_QUEUE_TIMEOUT_SECONDS", 120.0
)
inference_admission = InferenceAdmission(
    concurrency=MODEL_INFERENCE_CONCURRENCY,
    queue_capacity=MODEL_INFERENCE_QUEUE_CAPACITY,
    queue_timeout_seconds=MODEL_INFERENCE_QUEUE_TIMEOUT_SECONDS,
)


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="DAI Legal Model API",
    description="Qwen 2.5 7B + LoRA V4",
    version="1.0"
)


# ============================================================
# REQUEST
# ============================================================

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = Field(default=MODEL_MAX_NEW_TOKENS, ge=1, le=MODEL_MAX_NEW_TOKENS)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("LOAD QWEN 2.5 7B + LORA V4")
print("=" * 60)
_model_instance_lock = ModelInstanceLock().acquire()
atexit.register(_model_instance_lock.release)
print("[MODEL INSTANCE] single-process model lock acquired", flush=True)
_model_init_started = time.perf_counter()

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

print("Đang load tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True
)

print("Đang load base model...")

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

print("Đang load LoRA V4...")

model = PeftModel.from_pretrained(
    model,
    LORA_MODEL
)

model.eval()

_device_map = getattr(model, "hf_device_map", {}) or {}
_placement_counts = {}
for _device in _device_map.values():
    _key = str(_device)
    _placement_counts[_key] = _placement_counts.get(_key, 0) + 1
_qwen_memory = pytorch_cuda_memory_snapshot(torch)
print(
    "[MODEL INIT]",
    f"model_name={BASE_MODEL}",
    f"adapter={LORA_MODEL}",
    "quantization=nf4_4bit",
    "compute_dtype=bfloat16",
    "device_map=auto",
    f"layer_placement_counts={_placement_counts}",
    "status=loaded",
    f"duration_ms={round((time.perf_counter() - _model_init_started) * 1000, 2)}",
    f"cuda_allocated_bytes={_qwen_memory['allocated_bytes']}",
    f"cuda_reserved_bytes={_qwen_memory['reserved_bytes']}",
    "cuda_memory_scope=pytorch_process_allocator",
    flush=True,
)

print("=" * 60)
print("MODEL V4 LOADED SUCCESSFULLY")
print("=" * 60)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def root():
    return {
        "success": True,
        "message": "DAI Legal Model API đang chạy",
        "model": "Qwen2.5-7B-Instruct + LoRA V4"
    }


# ============================================================
# GENERATE
# ============================================================

@app.post("/generate")
@with_inference_admission(
    inference_admission,
    concurrency=MODEL_INFERENCE_CONCURRENCY,
    queue_capacity=MODEL_INFERENCE_QUEUE_CAPACITY,
)
def generate(
    request: GenerateRequest,
    request_id: str | None = Header(default=None, alias="X-Request-ID"),
    citation_attempt: int | None = Header(
        default=None, alias="X-Citation-Attempt"
    ),
):
    started_at = time.perf_counter()
    prompt = request.prompt

    if not prompt.strip():
        from fastapi import HTTPException
        raise HTTPException(
            status_code=400,
            detail="prompt không được để trống"
        )

    # Chỉ log metadata, không in toàn bộ nội dung pháp lý của người dùng.
    has_evidence_section = "=== EVIDENCE ===" in prompt
    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    print(
        "[MODEL API] request received:",
        f"request_id={request_id or 'not_provided'}",
        f"citation_attempt={citation_attempt or 'not_provided'}",
        f"prompt_chars={len(prompt)}",
        f"prompt_sha256={prompt_sha256}",
        f"has_evidence_section={has_evidence_section}",
        flush=True,
    )

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=False,
    )

    input_token_count = int(inputs["input_ids"].shape[1])
    if input_token_count > MODEL_MAX_INPUT_TOKENS:
        raise HTTPException(
            status_code=413,
            detail={
                "code": "MODEL_PROMPT_TOO_LONG",
                "input_tokens": input_token_count,
                "max_input_tokens": MODEL_MAX_INPUT_TOKENS,
                "message": "Prompt exceeds the configured context budget; no evidence was truncated.",
            },
        )
    inputs = inputs.to(model.device)
    formatted_prompt_sha256 = hashlib.sha256(
        formatted_prompt.encode("utf-8")
    ).hexdigest()
    print(
        "[MODEL API] chat_template applied:",
        f"formatted_prompt_chars={len(formatted_prompt)}",
        f"formatted_prompt_sha256={formatted_prompt_sha256}",
        f"input_tokens={input_token_count}",
        flush=True,
    )

    _device_map = getattr(model, "hf_device_map", {}) or {}
    _qwen_uses_cuda = (
        any(
            "cuda" in str(device).lower()
            or isinstance(device, int)
            for device in _device_map.values()
        )
        if _device_map
        else str(getattr(model, "device", "")).lower().startswith("cuda")
    )
    if _qwen_uses_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
        _cuda_before = pytorch_cuda_memory_snapshot(torch)
        torch.cuda.reset_peak_memory_stats()
    else:
        _cuda_before = {"allocated_bytes": None, "reserved_bytes": None}

    generation_started_at = time.perf_counter()
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id
            )
        if _qwen_uses_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        generation_duration_ms = round(
            (time.perf_counter() - generation_started_at) * 1000, 2
        )
    except Exception as exc:
        if _qwen_uses_cuda and torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
        _cuda_peak_failure = (
            pytorch_cuda_peak_snapshot(torch)
            if _qwen_uses_cuda
            else {"peak_allocated_bytes": None, "peak_reserved_bytes": None}
        )
        print(
            "[MODEL API] generation failed:",
            f"request_id={request_id or 'not_provided'}",
            f"citation_attempt={citation_attempt or 'not_provided'}",
            f"exception_type={type(exc).__name__}",
            f"elapsed_ms={round((time.perf_counter() - generation_started_at) * 1000, 2)}",
            f"cuda_peak_allocated_bytes={_cuda_peak_failure['peak_allocated_bytes']}",
            f"cuda_peak_reserved_bytes={_cuda_peak_failure['peak_reserved_bytes']}",
            "cuda_memory_scope=pytorch_process_allocator",
            flush=True,
        )
        raise

    generated_tokens = outputs[0][input_token_count:]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    processing_time_ms = round(
        (time.perf_counter() - started_at) * 1000,
        2
    )

    output_token_count = int(generated_tokens.shape[0])
    tokens_per_second = (
        round(output_token_count / (generation_duration_ms / 1000), 3)
        if generation_duration_ms > 0
        else None
    )
    _cuda_after = (
        pytorch_cuda_memory_snapshot(torch)
        if _qwen_uses_cuda
        else {"allocated_bytes": None, "reserved_bytes": None}
    )
    _cuda_peak = (
        pytorch_cuda_peak_snapshot(torch)
        if _qwen_uses_cuda
        else {"peak_allocated_bytes": None, "peak_reserved_bytes": None}
    )

    print(
        "[MODEL API] generation complete:",
        f"request_id={request_id or 'not_provided'}",
        f"citation_attempt={citation_attempt or 'not_provided'}",
        f"input_tokens={input_token_count}",
        f"output_tokens={output_token_count}",
        f"generation_duration_ms={generation_duration_ms}",
        f"tokens_per_second={tokens_per_second}",
        "ttft_ms=not_measured",
        f"cuda_allocated_before_bytes={_cuda_before['allocated_bytes']}",
        f"cuda_reserved_before_bytes={_cuda_before['reserved_bytes']}",
        f"cuda_allocated_after_bytes={_cuda_after['allocated_bytes']}",
        f"cuda_reserved_after_bytes={_cuda_after['reserved_bytes']}",
        f"cuda_peak_allocated_bytes={_cuda_peak['peak_allocated_bytes']}",
        f"cuda_peak_reserved_bytes={_cuda_peak['peak_reserved_bytes']}",
        "cuda_memory_scope=pytorch_process_allocator",
        f"processing_time_ms={processing_time_ms}",
        flush=True,
    )

    return {
        "success": True,
        "answer": answer,
        "processing_time_ms": processing_time_ms,
        "input_tokens": input_token_count,
        "output_tokens": output_token_count,
        "prompt_chars": len(prompt),
        "evidence_section_detected": has_evidence_section,
    }



# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    uvicorn.run(
        app,
        host=HOST,
        port=PORT
    )
