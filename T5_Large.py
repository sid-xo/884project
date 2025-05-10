import os
import torch
import pandas as pd
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments, Seq2SeqTrainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import evaluate

# File paths — change as per your HPC structure
train_path      = "train.csv"
validation_path = "validation.csv"
test_path       = "test.csv"

# Intent tokens
intent_tokens = {
    "Informative": "<intent_informative>",
    "Questioning": "<intent_question>",
    "Denouncing":  "<intent_denouncing>",
    "Positive":    "<intent_positive>",
}

# Load dataset
print("📁 Loading dataset...")
data_files = {
    "train": train_path,
    "validation": validation_path,
    "test": test_path
}
raw_datasets = load_dataset("csv", data_files=data_files)

# Preprocessing
def preprocess_fn(ex):
    tok = intent_tokens[ex["csType"]]
    ex["input_text"]  = f"{tok} {ex['hatespeech']}"
    ex["target_text"] = ex["counterspeech"]
    return ex

processed = raw_datasets.map(preprocess_fn, remove_columns=raw_datasets["train"].column_names)

# Load model and tokenizer
model_name = "t5-large"
print(f"🚀 Loading model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Tokenization
max_input_len  = 256
max_target_len = 128

def tokenize_fn(ex):
    inputs = tokenizer(ex["input_text"], truncation=True, padding="max_length", max_length=max_input_len)
    targets = tokenizer(ex["target_text"], truncation=True, padding="max_length", max_length=max_target_len)
    inputs["labels"] = targets["input_ids"]
    return inputs

tokenized = processed.map(tokenize_fn, batched=True, remove_columns=["input_text", "target_text"])

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, label_pad_token_id=tokenizer.pad_token_id)

# Evaluation metrics
bleu      = evaluate.load("bleu")
rouge     = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = [[(l if l != -100 else tokenizer.pad_token_id) for l in label] for label in labels]
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = {}
    result.update(bleu.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels]))
    result.update(rouge.compute(predictions=decoded_preds, references=decoded_labels))
    bs = bertscore.compute(predictions=decoded_preds, references=decoded_labels, lang="en")
    result["bertscore_f1"] = sum(bs["f1"]) / len(bs["f1"])
    return {
        "bleu":       result["bleu"],
        "rouge1":     result["rouge1"],
        "rouge2":     result["rouge2"],
        "rougeL":     result["rougeL"],
        "bertscore":  result["bertscore_f1"]
    }

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="intentconan2-t5large-lora",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    predict_with_generate=True,
    logging_dir="./logs",
    logging_steps=100,
    num_train_epochs=3,
    warmup_steps=500,
    learning_rate=5e-5,
    weight_decay=0.01,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    report_to=[],
)

# LoRA configuration
print("🔧 Applying LoRA...")
model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_2_SEQ_LM",
)
model = get_peft_model(model, lora_config)

# Trainer setup
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Training
print(" Training started...")
trainer.train()

# Evaluation
print(" Evaluating...")
metrics = trainer.evaluate(eval_dataset=tokenized["test"])
print(" Test Evaluation:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

# Save model
save_dir = "intentconan2-t5large-lora-final"
trainer.save_model(save_dir)
tokenizer.save_pretrained(save_dir)
print(f" Model and tokenizer saved to {save_dir}/")
