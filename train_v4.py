import os
import pandas as pd
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig


# =========================================================
# 1. CONFIG
# =========================================================

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

DATA_PATH = "data/vietnamese_legal_instruct_cleaned.xlsx"

OUTPUT_DIR = "outputs/qwen-legal-lora-v4"

HF_CACHE = r"D:\AI\HuggingFace\hub"

os.environ["HF_HOME"] = r"D:\AI\HuggingFace"
os.environ["HF_HUB_CACHE"] = HF_CACHE


# =========================================================
# 2. CHECK GPU
# =========================================================

print("=" * 60)
print("CHECK GPU")
print("=" * 60)

print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("Không tìm thấy GPU CUDA.")

print("GPU:", torch.cuda.get_device_name(0))

gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"VRAM: {gpu_memory:.2f} GB")


# =========================================================
# 3. LOAD EXCEL
# =========================================================

print("\n" + "=" * 60)
print("LOAD DATASET")
print("=" * 60)

df = pd.read_excel(DATA_PATH)

print("Tổng số mẫu:", len(df))
print("Columns:", df.columns.tolist())


required_columns = [
    "system_prompt",
    "user_prompt",
    "assistant_response",
]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Thiếu column: {column}")


# Xóa dòng thiếu dữ liệu quan trọng
df = df.dropna(
    subset=required_columns
).reset_index(drop=True)

print("Số mẫu sau khi loại dòng thiếu:", len(df))


# Lấy 2.000 mẫu để train V4
df = df.sample(n=2000, random_state=42).reset_index(drop=True)

print("Số mẫu sử dụng cho V4:", len(df))


# =========================================================
# 4. TRAIN / VALIDATION / TEST
# =========================================================

print("\n" + "=" * 60)
print("SPLIT DATASET")
print("=" * 60)


df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 90% train
# 5% validation
# 5% test
n = len(df)

train_end = int(n * 0.90)
val_end = int(n * 0.95)

train_df = df.iloc[:train_end].copy()
val_df = df.iloc[train_end:val_end].copy()
test_df = df.iloc[val_end:].copy()

print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))


# =========================================================
# 5. CONVERT TO CHAT FORMAT
# =========================================================

def convert_to_conversations(row):

    return {
        "conversations": [
            {
                "role": "system",
                "content": str(row["system_prompt"]),
            },
            {
                "role": "user",
                "content": str(row["user_prompt"]),
            },
            {
                "role": "assistant",
                "content": str(row["assistant_response"]),
            },
        ]
    }


train_data = Dataset.from_pandas(
    train_df,
    preserve_index=False,
).map(
    convert_to_conversations
)

val_data = Dataset.from_pandas(
    val_df,
    preserve_index=False,
).map(
    convert_to_conversations
)

test_data = Dataset.from_pandas(
    test_df,
    preserve_index=False,
).map(
    convert_to_conversations
)

print("\nDataset đã chuyển sang conversations.")


# =========================================================
# 6. TOKENIZER
# =========================================================

print("\n" + "=" * 60)
print("LOAD TOKENIZER")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    cache_dir=HF_CACHE,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# =========================================================
# 7. QLoRA CONFIG
# =========================================================

print("\n" + "=" * 60)
print("LOAD QLORA MODEL")
print("=" * 60)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)


model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.bfloat16,
    cache_dir=HF_CACHE,
)

model.config.use_cache = False


# =========================================================
# 8. LORA
# =========================================================

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",

    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)


# =========================================================
# 9. TRAINING CONFIG
# =========================================================

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,

    # Batch
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,

    gradient_accumulation_steps=8,

    # Training
    num_train_epochs=1,
    learning_rate=2e-5,

    # Sequence
    max_length=2048,

    # Precision
    bf16=True,

    # Memory
    gradient_checkpointing=True,

    # Optimizer
    optim="paged_adamw_8bit",

    # Evaluation
    eval_strategy="steps",
    eval_steps=100,

    # Save
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,

    logging_steps=10,

    report_to="none",

    seed=42,
)


# =========================================================
# 10. FORMATTING
# =========================================================

def formatting_func(example):

    return tokenizer.apply_chat_template(
        example["conversations"],
        tokenize=False,
        add_generation_prompt=False,
    )


# =========================================================
# 11. TRAINER
# =========================================================

print("\n" + "=" * 60)
print("CREATE TRAINER")
print("=" * 60)

trainer = SFTTrainer(
    model=model,
    args=training_args,

    train_dataset=train_data,
    eval_dataset=val_data,

    peft_config=peft_config,

    formatting_func=formatting_func,

    processing_class=tokenizer,
)


# =========================================================
# 12. TRAIN
# =========================================================

print("\n" + "=" * 60)
print("START TRAINING V4")
print("=" * 60)

trainer.train()


# =========================================================
# 13. SAVE
# =========================================================

print("\n" + "=" * 60)
print("SAVE MODEL")
print("=" * 60)

trainer.save_model(OUTPUT_DIR)

tokenizer.save_pretrained(OUTPUT_DIR)

print("\nLoRA V4 saved at:")
print(OUTPUT_DIR)

print("\nTRAINING COMPLETED.")