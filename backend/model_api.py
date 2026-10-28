from fastapi import FastAPI, Header
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch
import uvicorn
import time
import hashlib
import sys
import os


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
    ).to(model.device)

    input_token_count = int(inputs["input_ids"].shape[1])
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
    except Exception as exc:
        print(
            "[MODEL API] generation failed:",
            f"request_id={request_id or 'not_provided'}",
            f"citation_attempt={citation_attempt or 'not_provided'}",
            f"exception_type={type(exc).__name__}",
            f"elapsed_ms={round((time.perf_counter() - generation_started_at) * 1000, 2)}",
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

    print(
        "[MODEL API] generation complete:",
        f"request_id={request_id or 'not_provided'}",
        f"citation_attempt={citation_attempt or 'not_provided'}",
        f"input_tokens={input_token_count}",
        f"output_tokens={output_token_count}",
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
