import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_ID = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

def main():
    print("Loading tokenizer from cache...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, local_files_only=True
    )

    print("Loading model from cache...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, local_files_only=True
    )
    model.eval()

    print("Device: CPU")
    print("Labels:", model.config.id2label)

    examples = [
        {
            "claim": "Người nộp hồ sơ phải cung cấp giấy tờ theo quy định.",
            "evidence": "Hồ sơ gồm đơn đề nghị và bản sao giấy tờ tùy thân.",
        },
        {
            "claim": "Người dân không cần nộp bất kỳ giấy tờ nào.",
            "evidence": "Hồ sơ gồm đơn đề nghị và bản sao giấy tờ tùy thân.",
        },
        {
            "claim": "Thủ tục này được thực hiện hoàn toàn miễn phí.",
            "evidence": "Hồ sơ gồm đơn đề nghị và bản sao giấy tờ tùy thân.",
        },
    ]

    for i, item in enumerate(examples, 1):
        inputs = tokenizer(
            item["evidence"],
            item["claim"],
            return_tensors="pt",
            truncation=False,
        )

        token_count = inputs["input_ids"].shape[1]
        if token_count > 512:
            print(f"Example {i}: {token_count} tokens; skipped.")
            continue

        start = time.perf_counter()
        with torch.inference_mode():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]
        elapsed = time.perf_counter() - start

        labels = model.config.id2label
        pred = int(probs.argmax())

        print(f"\n--- Example {i} ---")
        print("Claim:", item["claim"])
        print("Evidence:", item["evidence"])
        print("Predicted:", labels[pred])
        print("Probabilities:", {
            labels[j]: round(float(probs[j]), 4)
            for j in range(len(probs))
        })
        print(f"Inference time: {elapsed:.3f}s")

    print("\nSmoke test completed.")

if __name__ == "__main__":
    main()
