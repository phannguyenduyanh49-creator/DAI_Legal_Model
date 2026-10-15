from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch
import uvicorn


# ============================================================
# CONFIG
# ============================================================

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LORA_MODEL = r"D:\Prj\DAI-Legal-Model\outputs\qwen-legal-lora-v4"

HOST = "0.0.0.0"
PORT = 8001


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
    max_new_tokens: int = 512


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
def generate(request: GenerateRequest):

    prompt = request.prompt

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_new_tokens,
            do_sample=False,
            temperature=0.1,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    return {
        "success": True,
        "answer": answer
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