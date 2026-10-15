import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LORA_MODEL = r"outputs/qwen-legal-lora-v4"

print("=" * 60)
print("LOAD QWEN 2.5 7B + LORA V4")
print("=" * 60)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.bfloat16,
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(
    model,
    LORA_MODEL,
)

model.eval()

print("\nModel V4 loaded successfully!\n")

while True:
    question = input("\nNhập câu hỏi pháp luật (gõ 'exit' để thoát): ")

    if question.lower() == "exit":
        break

    messages = [
        {
            "role": "system",
            "content": "Bạn là AI hỗ trợ tra cứu thông tin pháp luật Việt Nam. Hãy trả lời rõ ràng, chính xác và dựa trên thông tin được cung cấp."
        },
        {
            "role": "user",
            "content": question
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.2,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
        )

    answer = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    print("\n" + "=" * 60)
    print("TRẢ LỜI CỦA MODEL V4")
    print("=" * 60)
    print(answer)