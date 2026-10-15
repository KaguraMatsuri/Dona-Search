#!/usr/bin/env python3
"""Train query processor via SFT"""
import json, torch
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

DATA = Path("data/query_train.jsonl")
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT = Path("checkpoints/query-processor")

def format_example(ex):
    return f"""Extract search keywords from the query. Output JSON only.

Query: {ex['query']}

Output format: {{"keywords": [...], "filters": {{...}}}}

JSON: {json.dumps({"keywords": ex["keywords"], "filters": ex["filters"]}, ensure_ascii=False)}"""

def main():
    # Load data
    with open(DATA) as f:
        data = [json.loads(line) for line in f]

    dataset = Dataset.from_list(data).train_test_split(test_size=0.1)

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # LoRA config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    # Tokenize
    def tokenize(examples):
        texts = [format_example(ex) for ex in examples]
        return tokenizer(texts, truncation=True, max_length=512)

    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset["train"].column_names)

    # Train
    args = TrainingArguments(
        output_dir=str(OUTPUT),
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=50,
        save_steps=500,
        eval_steps=500,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
    )

    trainer.train()
    model.save_pretrained(OUTPUT)
    tokenizer.save_pretrained(OUTPUT)
    print(f"✓ Saved to {OUTPUT}")

if __name__ == "__main__":
    main()
