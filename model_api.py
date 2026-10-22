import os
import json
import torch

from fastapi import FastAPI
from pydantic import BaseModel

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from peft import PeftModel


# =========================================================
# 1. CẤU HÌNH
# =========================================================

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


# =========================================================
# 2. TỰ NHẬN DIỆN MÔI TRƯỜNG
# =========================================================

if os.path.exists("/app"):
    # Docker / Linux
    PROJECT_DIR = "/app"

    HF_HOME = "/root/.cache/huggingface"

    HF_CACHE = "/root/.cache/huggingface/hub"

else:
    # Windows
    PROJECT_DIR = os.path.dirname(
        os.path.abspath(__file__)
    )

    HF_HOME = r"D:\AI\HuggingFace"

    HF_CACHE = r"D:\AI\HuggingFace\hub"


os.environ["HF_HOME"] = HF_HOME
os.environ["HF_HUB_CACHE"] = HF_CACHE


# =========================================================
# 3. ĐƯỜNG DẪN LORA
# =========================================================

FALLBACK_LORA = os.path.join(
    PROJECT_DIR,
    "outputs",
    "qwen-legal-lora-v3"
)

REGISTRY_FILE = os.path.join(
    PROJECT_DIR,
    "outputs",
    "active",
    "registry.json"
)


# =========================================================
# 4. ACTIVE ADAPTER
# =========================================================

def get_active_adapter():

    print("\n" + "=" * 60)
    print("AUTO MODEL REGISTRY")
    print("=" * 60)

    print(
        "Project:",
        PROJECT_DIR
    )

    print(
        "Registry:",
        REGISTRY_FILE
    )

    print(
        "Fallback:",
        FALLBACK_LORA
    )

    # -----------------------------------------------------
    # Registry không tồn tại
    # -----------------------------------------------------

    if not os.path.exists(REGISTRY_FILE):

        print(
            "Registry không tồn tại."
        )

        print(
            "Fallback → V3"
        )

        return FALLBACK_LORA


    # -----------------------------------------------------
    # Đọc registry
    # -----------------------------------------------------

    try:

        with open(
            REGISTRY_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            registry = json.load(f)

    except Exception as e:

        print(
            "Không đọc được registry:",
            e
        )

        print(
            "Fallback → V3"
        )

        return FALLBACK_LORA


    status = registry.get(
        "status",
        "unknown"
    )

    active_adapter = registry.get(
        "active_adapter"
    )


    print(
        "Registry status:",
        status
    )

    print(
        "Registry adapter:",
        active_adapter
    )


    # -----------------------------------------------------
    # Không có adapter
    # -----------------------------------------------------

    if not active_adapter:

        print(
            "Registry không có active_adapter."
        )

        print(
            "Fallback → V3"
        )

        return FALLBACK_LORA


    # -----------------------------------------------------
    # QUAN TRỌNG:
    # Registry được tạo trên Windows có thể dùng "\"
    #
    # Docker Linux cần "/"
    # -----------------------------------------------------

    active_adapter = active_adapter.replace(
        "\\",
        "/"
    )


    # -----------------------------------------------------
    # Relative path → absolute path
    # -----------------------------------------------------

    if not os.path.isabs(
        active_adapter
    ):

        active_adapter = os.path.join(
            PROJECT_DIR,
            active_adapter
        )


    active_adapter = os.path.normpath(
        active_adapter
    )


    # -----------------------------------------------------
    # Kiểm tra adapter
    # -----------------------------------------------------

    adapter_file = os.path.join(
        active_adapter,
        "adapter_model.safetensors"
    )

    config_file = os.path.join(
        active_adapter,
        "adapter_config.json"
    )


    if not os.path.exists(
        adapter_file
    ):

        print(
            "Active adapter không tồn tại:"
        )

        print(
            active_adapter
        )

        print(
            "Fallback → V3"
        )

        return FALLBACK_LORA


    if not os.path.exists(
        config_file
    ):

        print(
            "adapter_config.json không tồn tại:"
        )

        print(
            active_adapter
        )

        print(
            "Fallback → V3"
        )

        return FALLBACK_LORA


    # -----------------------------------------------------
    # ACTIVE
    # -----------------------------------------------------

    print(
        "ACTIVE ADAPTER:"
    )

    print(
        active_adapter
    )

    return active_adapter


# =========================================================
# 5. ACTIVE LORA
# =========================================================

LORA_MODEL = get_active_adapter()


# =========================================================
# 6. FASTAPI
# =========================================================

app = FastAPI(
    title="DAI Legal Model API"
)


# =========================================================
# 7. REQUEST
# =========================================================

class ChatRequest(BaseModel):

    prompt: str


# =========================================================
# 8. GPU CHECK
# =========================================================

print("\n" + "=" * 60)
print("GPU CHECK")
print("=" * 60)

print(
    "CUDA:",
    torch.cuda.is_available()
)

if not torch.cuda.is_available():

    raise RuntimeError(
        "Không tìm thấy CUDA."
    )


print(
    "GPU:",
    torch.cuda.get_device_name(0)
)


vram = (
    torch.cuda.get_device_properties(0).total_memory
    / 1024**3
)


print(
    f"VRAM: {vram:.1f} GB"
)


# =========================================================
# 9. 4-BIT QUANTIZATION
# =========================================================

bnb_config = BitsAndBytesConfig(

    load_in_4bit=True,

    bnb_4bit_quant_type="nf4",

    bnb_4bit_compute_dtype=torch.bfloat16,

    bnb_4bit_use_double_quant=True,
)


# =========================================================
# 10. LOAD TOKENIZER
# =========================================================

print("\n" + "=" * 60)
print("LOADING TOKENIZER")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(

    BASE_MODEL,

    cache_dir=HF_CACHE,
)


if tokenizer.pad_token is None:

    tokenizer.pad_token = tokenizer.eos_token


# =========================================================
# 11. LOAD BASE MODEL
# =========================================================

print("\n" + "=" * 60)
print("LOADING QWEN2.5-7B")
print("=" * 60)

model = AutoModelForCausalLM.from_pretrained(

    BASE_MODEL,

    quantization_config=bnb_config,

    device_map="auto",

    cache_dir=HF_CACHE,
)


print(
    "Base model loaded."
)


# =========================================================
# 12. LOAD ACTIVE LORA
# =========================================================

print("\n" + "=" * 60)
print("LOADING ACTIVE LORA")
print("=" * 60)

print(
    "Adapter:",
    LORA_MODEL
)


model = PeftModel.from_pretrained(

    model,

    LORA_MODEL,
)


model.eval()


print(
    "Active LoRA loaded successfully."
)


# =========================================================
# 13. MODEL INFO
# =========================================================

print("\n" + "=" * 60)
print("MODEL READY")
print("=" * 60)

print(
    "Base:",
    BASE_MODEL
)

print(
    "LoRA:",
    LORA_MODEL
)

print(
    "Registry:",
    REGISTRY_FILE
)


# =========================================================
# 14. HÀM INFERENCE
# =========================================================

def ask_model(
    question: str
):

    messages = [

        {
            "role": "system",

            "content": (
                "Bạn là trợ lý hỗ trợ "
                "tra cứu thủ tục hành chính. "
                "Trả lời ngắn gọn và rõ ràng. "
                "Chỉ cung cấp thông tin "
                "liên quan đến câu hỏi."
            ),
        },

        {
            "role": "user",

            "content": question,
        },

    ]


    prompt = tokenizer.apply_chat_template(

        messages,

        tokenize=False,

        add_generation_prompt=True,
    )


    inputs = tokenizer(

        prompt,

        return_tensors="pt",
    )


    inputs = {

        key: value.to(model.device)

        for key, value in inputs.items()

    }


    with torch.no_grad():

        outputs = model.generate(

            **inputs,

            max_new_tokens=384,

            do_sample=False,

            repetition_penalty=1.05,

            pad_token_id=tokenizer.eos_token_id,
        )


    input_length = (
        inputs["input_ids"].shape[1]
    )


    generated_tokens = outputs[

        0

    ][

        input_length:

    ]


    answer = tokenizer.decode(

        generated_tokens,

        skip_special_tokens=True,
    )


    return answer.strip()


# =========================================================
# 15. ROOT
# =========================================================

@app.get("/")
def root():

    return {

        "status": "running",

        "base_model": BASE_MODEL,

        "active_adapter": LORA_MODEL,

        "registry": REGISTRY_FILE,

    }


# =========================================================
# 16. GENERATE
# =========================================================

def ask_model(prompt_text: str):
    messages = [
        {
            "role": "user",
            "content": prompt_text,
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=384,
            do_sample=False,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
        )

    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return answer.strip()



# =========================================================
# 17. RUN SERVER
# =========================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(

        app,

        host="0.0.0.0",

        port=8001

    )