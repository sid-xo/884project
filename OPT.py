#!/usr/bin/env python
# intent_counterspeech.py

import os
import torch
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftConfig,
    PeftModel
)
import evaluate
from sklearn.metrics import accuracy_score

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROMPT = "Intent: {intent}\nHate: {hate}\nResponse: "

def train():
    # 1. Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b", use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b")

    # 2. Load and encode dataset
    dataset = load_dataset("csv", data_files={
        "train": "train.csv",
        "validation": "validation.csv"
    })

    def encode(row):
        intent_map = {
            "informative": "informative",
            "denouncing": "denouncing",
            "questioning": "question",
            "positive": "positive"
        }
        intent = intent_map[row["csType"].strip().lower()]
        prompt = PROMPT.format(intent=intent.capitalize(), hate=row["hatespeech"].strip())
        full = prompt + row["counterspeech"].strip()
        enc = tokenizer(full, truncation=True, padding="max_length", max_length=256)
        enc["labels"] = enc["input_ids"].copy()
        return enc

    dataset = dataset.map(encode, remove_columns=dataset["train"].column_names)

    # 3. Apply LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=("q_proj", "v_proj"),
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(base_model, lora_config).to(DEVICE)

    # 4. Training arguments
    training_args = TrainingArguments(
        output_dir="outputs/opt1.3b_lora_final",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        fp16=True,
        save_steps=400,
        logging_dir="outputs/opt1.3b_lora_final/logs",
    )

    # 5. Trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    trainer.train()
    trainer.save_model("outputs/opt1.3b_lora_final")
    tokenizer.save_pretrained("outputs/opt1.3b_lora_final")

def evaluate_model():
    # 1. Load model and tokenizer
    tok = AutoTokenizer.from_pretrained("outputs/opt1.3b_lora_final")
    peft_cfg = PeftConfig.from_pretrained("outputs/opt1.3b_lora_final")
    base = AutoModelForCausalLM.from_pretrained(peft_cfg.base_model_name_or_path)
    model = PeftModel.from_pretrained(base, "outputs/opt1.3b_lora_final").to(DEVICE)

    # 2. Load test set
    ds = load_dataset("csv", data_files={"test": "test.csv"})["test"]
    ds = ds.select(range(300))

    preds, refs, intents_true, intents_pred = [], [], [], []
    intent_map = {
        "informative": "informative",
        "denouncing": "denouncing",
        "questioning": "question",
        "positive": "positive"
    }

    # 3. Generate predictions
    for row in tqdm(ds):
        intent = intent_map[row["csType"].strip().lower()]
        prompt = PROMPT.format(intent=intent.capitalize(), hate=row["hatespeech"].strip())
        input_ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)

        with torch.no_grad():
            out = model.generate(input_ids, max_new_tokens=64)
        gen = tok.decode(out[0], skip_special_tokens=True)[len(prompt):].strip()

        preds.append(gen)
        refs.append(row["counterspeech"].strip())
        intents_true.append(intent)
        intents_pred.append(
            "question" if gen.endswith("?") else
            "positive" if any(w in gen for w in ["love", "respect", "support"]) else
            "denouncing" if any(w in gen for w in ["wrong", "shame", "hate"]) else
            "informative"
        )

    # 4. Compute metrics
    rouge = evaluate.load("rouge")
    bleu  = evaluate.load("bleu")
    bert  = evaluate.load("bertscore")

    results = {
        "rougeL": rouge.compute(predictions=preds, references=refs)["rougeL"],
        "bleu": bleu.compute(predictions=preds, references=refs)["bleu"],
        "bertscore_f1": sum(bert.compute(predictions=preds, references=refs, lang="en")["f1"]) / len(preds),
        "intent_accuracy": accuracy_score(intents_true, intents_pred)
    }

    print(" Evaluation Complete")
    print(results)

    # 5. Save predictions
    df_out = pd.DataFrame({
        "Hate": [r["hatespeech"] for r in ds],
        "Intent": intents_true,
        "PredIntent": intents_pred,
        "Reference": refs,
        "Generated": preds
    })
    df_out.to_csv("final_outputs.csv", index=False)
    print(" Saved outputs to final_outputs.csv")

if __name__ == "__main__":
    train()
    evaluate_model()
