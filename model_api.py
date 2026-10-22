import os
import time
import hashlib
import logging
from pathlib import Path
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel

# ============================================================
# CONFIG
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")

# Override with $env:LORA_MODEL="D:\\path\\to\\adapter" if needed.
# The default matches the project folder layout on this machine.
DEFAULT_LORA = PROJECT_DIR / "outputs" / "qwen-legal-lora-v4"
LORA_MODEL = Path(os.getenv("LORA_MODEL", str(DEFAULT_LORA))).expanduser()

HOST = os.getenv("MODEL_API_HOST", "0.0.0.0")
PORT = int(os.getenv("MODEL_API_PORT", "8001"))
LOAD_IN_4BIT = os.getenv("LOAD_IN_4BIT", "true").strip().lower() in {
    "1", "true", "yes", "on"
}
MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", "8192"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("dai-model-api")

# ============================================================
# APP / REQUEST
# ============================================================

app = FastAPI(
    title="DAI Legal Model API",
    description="Qwen2.5-7B-Instruct with optional LoRA adapter",
    version="1.1",
)


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    max_new_tokens: int = Field(default=512, ge=1, le=2048)


# ============================================================
# MODEL STATE
# ============================================================

tokenizer = None
model = None
active_adapter: Optional[str] = None
model_loaded = False


def load_model() -> None:
    """Load base Qwen and load LoRA only when a valid local adapter config exists."""
    global tokenizer, model, active_adapter, model_loaded

    logger.info("=" * 60)
    logger.info("DAI MODEL API - MODEL LOADING")
    logger.info("Project: %s", PROJECT_DIR)
    logger.info("Base model: %s", BASE_MODEL)
    logger.info("Configured LoRA path: %s", LORA_MODEL)
    logger.info("CUDA available: %s", torch.cuda.is_available())

    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            logger.info(
                "GPU %s: %s (%.2f GB VRAM)",
                idx,
                torch.cuda.get_device_name(idx),
                props.total_memory / (1024 ** 3),
            )

    adapter_config = LORA_MODEL / "adapter_config.json"
    adapter_weights = LORA_MODEL / "adapter_model.safetensors"
    adapter_weights_bin = LORA_MODEL / "adapter_model.bin"
    adapter_exists = (
        adapter_config.is_file()
        and (adapter_weights.is_file() or adapter_weights_bin.is_file())
    )

    if adapter_exists:
        logger.info("Valid LoRA adapter found: %s", LORA_MODEL)
    else:
        logger.warning(
            "No complete local LoRA adapter found at %s. "
            "Will load Qwen base model ONLY for testing.",
            LORA_MODEL,
        )
        logger.warning(
            "This is not the fine-tuned legal model. To use LoRA, set LORA_MODEL "
            "to a directory containing adapter_config.json and adapter weights."
        )

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    model_kwargs = {"device_map": "auto", "trust_remote_code": True}

    if LOAD_IN_4BIT:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "4-bit loading is enabled but CUDA is unavailable. "
                "Use a CUDA-enabled Python/PyTorch environment."
            )
        compute_dtype = (
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
        )
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = quantization_config
    else:
        model_kwargs["torch_dtype"] = (
            torch.float16 if torch.cuda.is_available() else torch.float32
        )

    logger.info("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **model_kwargs)
    logger.info("Base model loaded.")

    if adapter_exists:
        # If an adapter is present but invalid/incompatible, fail loudly rather
        # than silently returning an un-fine-tuned model.
        logger.info("Loading LoRA adapter...")
        model = PeftModel.from_pretrained(
            model,
            str(LORA_MODEL.resolve()),
            is_trainable=False,
        )
        active_adapter = str(LORA_MODEL.resolve())
        logger.info("LoRA adapter loaded successfully.")
    else:
        active_adapter = None

    model.eval()
    model_loaded = True
    logger.info("Model loaded. Active adapter: %s", active_adapter or "NONE (BASE ONLY)")
    logger.info("=" * 60)


@app.on_event("startup")
def startup_event() -> None:
    load_model()


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def root():
    return {
        "success": True,
        "message": "DAI Legal Model API is running",
        "base_model": BASE_MODEL,
        "model_loaded": model_loaded,
        "adapter_loaded": active_adapter is not None,
        "adapter_path": active_adapter,
        "warning": None if active_adapter else "Base model only; LoRA is not loaded.",
    }


@app.get("/health")
def health():
    return {
        "success": model_loaded,
        "model_loaded": model_loaded,
        "base_model": BASE_MODEL,
        "adapter_loaded": active_adapter is not None,
    }


# ============================================================
# GENERATE
# ============================================================

@app.post("/generate")
def generate(
    request: GenerateRequest,
    request_id: Optional[str] = Header(default=None, alias="X-Request-ID"),
    citation_attempt: Optional[int] = Header(default=None, alias="X-Citation-Attempt"),
):
    if not model_loaded or model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    started_at = time.perf_counter()
    prompt = request.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt không được để trống")

    has_evidence_section = "=== EVIDENCE ===" in prompt
    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    logger.info(
        "[MODEL API] request received: request_id=%s citation_attempt=%s "
        "prompt_chars=%s prompt_sha256=%s has_evidence_section=%s",
        request_id or "not_provided",
        citation_attempt if citation_attempt is not None else "not_provided",
        len(prompt),
        prompt_sha256,
        has_evidence_section,
    )

    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    )

    # Use the input-embedding device; this is safer than assuming model.device
    # when device_map="auto" has dispatched the model across devices.
    input_device = model.get_input_embeddings().weight.device
    inputs = {key: value.to(input_device) for key, value in inputs.items()}
    input_token_count = int(inputs["input_ids"].shape[1])

    generation_started_at = time.perf_counter()
    try:
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    except Exception as exc:
        logger.exception(
            "[MODEL API] generation failed: request_id=%s citation_attempt=%s "
            "exception_type=%s elapsed_ms=%.2f",
            request_id or "not_provided",
            citation_attempt if citation_attempt is not None else "not_provided",
            type(exc).__name__,
            (time.perf_counter() - generation_started_at) * 1000,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Model generation failed: {type(exc).__name__}",
        ) from exc

    generated_tokens = outputs[0][input_token_count:]
    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    processing_time_ms = round((time.perf_counter() - started_at) * 1000, 2)
    output_token_count = int(generated_tokens.shape[0])

    logger.info(
        "[MODEL API] generation complete: request_id=%s input_tokens=%s "
        "output_tokens=%s processing_time_ms=%s",
        request_id or "not_provided",
        input_token_count,
        output_token_count,
        processing_time_ms,
    )
    return {
        "success": True,
        "answer": answer,
        "processing_time_ms": processing_time_ms,
        "input_tokens": input_token_count,
        "output_tokens": output_token_count,
        "prompt_chars": len(prompt),
        "evidence_section_detected": has_evidence_section,
        "adapter_loaded": active_adapter is not None,
    }


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
