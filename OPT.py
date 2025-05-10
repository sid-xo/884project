from __future__ import annotations
import json, logging, torch
from typing import Dict
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, PeftModel, PeftConfig
from sklearn.metrics import accuracy_score
import evaluate

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("IntentCS")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🔧 Running on", DEVICE)

PROMPT = "Intent: {intent}\nHate: {hate}\nResponse: "

# --- Tokenizer Wrapper ---
class IntentTokenizer:
    def __init__(self, base_model: str):
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def save(self, path: str) -> None:
        self.tokenizer.save_pretrained(path)

# --- LoRA Model Wrapper ---
class LoRAOPT:
    def __init__(self, base_model="facebook/opt-350m", lora_r=8, lora_alpha=16, lora_dropout=0.05, quant4bit=False):
        self.base_model_name = base_model
        self.model = self._load(lora_r, lora_alpha, lora_dropout, quant4bit)

    def _load(self, r, alpha, dropout, quant4bit):
        base = AutoModelForCausalLM.from_pretrained(self.base_model_name)
        base.gradient_checkpointing_enable()
        base = prepare_model_for_kbit_training(base)

        cfg = LoraConfig(
            r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none",
            target_modules=("q_proj", "v_proj"), task_type="CAUSAL_LM"
        )
        return get_peft_model(base, cfg)

# --- Trainer Wrapper ---
class CSTrainer:
    def __init__(self, model: LoRAOPT, tokenizer, datasets, out_dir="outputs/opt_cs_lora",
                 epochs=1, bs=2, grad_acc=4, lr=2e-4):
        self.model = model.model
        self.tokenizer = tokenizer
        self.args = TrainingArguments(
            output_dir=out_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=bs,
            gradient_accumulation_steps=grad_acc,
            learning_rate=lr,
            fp16=True,
            save_steps=400,
            logging_dir=f"{out_dir}/logs"
        )
        self.trainer = Trainer(
            model=self.model,
            args=self.args,
            train_dataset=datasets["train"],
            eval_dataset=datasets["validation"],
            tokenizer=tokenizer,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
        )

    def fit(self):
        self.trainer.train()
        self.trainer.save_model(self.args.output_dir)
        self.tokenizer.save_pretrained(self.args.output_dir)

# --- Evaluator ---
class CSEvaluator:
    def __init__(self, model_dir: str, dataset_cache: str, n=300):
        from transformers import pipeline
        import os

        self.tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        peft_cfg = PeftConfig.from_pretrained(model_dir)
        base = AutoModelForCausalLM.from_pretrained(peft_cfg.base_model_name_or_path)
        self.model = PeftModel.from_pretrained(base, model_dir).to(DEVICE)

        self.gen = pipeline(
            "text-generation", model=self.model, tokenizer=self.tok,
            max_new_tokens=64, do_sample=False,
            device=0 if torch.cuda.is_available() else -1
        )

        #  Only select up to available size
        full_test_ds = load_from_disk(dataset_cache)["test"]
        self.ds = load_dataset("csv", data_files={"test": "test.csv"})["test"]
        self.ds = self.ds.select(range(min(n, len(self.ds))))  # limit to first n

    def _pred_intent_heuristic(self, text: str) -> str:
        text = text.lower()
        if text.endswith("?"): return "question"
        if any(w in text for w in ["love", "respect", "support"]): return "positive"
        if any(w in text for w in ["wrong", "shame", "hate"]): return "denouncing"
        return "informative"

    def evaluate(self):
        preds, refs, intents_true, intents_pred = [], [], [], []
        for row in self.ds:
            intent_raw = row["csType"].strip().lower()
            intent_map = {
                "informative": "informative",
                "denouncing": "denouncing",
                "questioning": "question",
                "positive": "positive"
            }
            intent = intent_map.get(intent_raw)
            if intent is None:
                continue
            prompt = PROMPT.format(intent=intent.capitalize(), hate=row["hatespeech"].strip())
            out = self.gen(prompt)[0]["generated_text"][len(prompt):].strip()
            preds.append(out)
            refs.append(row["counterspeech"].strip())
            intents_true.append(intent)
            intents_pred.append(self._pred_intent_heuristic(out))

        rouge = evaluate.load("rouge")
        bleu = evaluate.load("bleu")
        bertscore = evaluate.load("bertscore")
        rouge_result = rouge.compute(predictions=preds, references=refs, rouge_types=["rougeL"])
        bleu_result = bleu.compute(predictions=preds, references=refs)
        bert_result = bertscore.compute(predictions=preds, references=refs, lang="en")
        intent_acc = accuracy_score(intents_true, intents_pred)

        log.info({
            "rougeL": rouge_result["rougeL"],
            "bleu": bleu_result["bleu"],
            "bertscore_f1": sum(bert_result["f1"]) / len(bert_result["f1"]),
            "intent_accuracy": intent_acc
        })

        print("\n Sample Predictions:")
        for i in range(min(5, len(preds))):
            print(f"\n--- Example {i+1} ---")
            print(f"Hate Speech        : {self.ds[i]['hatespeech'].strip()}")
            print(f"Intended Intent    : {intents_true[i]}")
            print(f"Predicted Intent   : {intents_pred[i]}")
            print(f"Reference Response : {refs[i]}")
            print(f"Generated Response : {preds[i]}")

# --- Main Orchestrator ---
def main(cfg):
    if isinstance(cfg, str):
        cfg = json.load(open(cfg))

    tokenizer_obj = IntentTokenizer(cfg["model_name"])
    tokenizer_obj.save(cfg["tokenizer_dir"])

    dataset = load_dataset("csv", data_files={
        "train": "train.csv",
        "validation": "validation.csv",
        "test": "test.csv"
    })

    if cfg.get("debug_mode", False):
        dataset["train"] = dataset["train"].select(range(20))
        dataset["validation"] = dataset["validation"].select(range(10))
        dataset["test"] = dataset["test"].select(range(10))
        print("🔎 DEBUG MODE: Using small dataset subset.")

    class PatchedData:
        def __init__(self, dataset, tokenizer, max_len):
            self.tokenizer = tokenizer
            self.ds = dataset
            self.max_len = max_len
            self.enc_ds = self._tokenize()

        def _tokenize(self):
            def encode(row):
                intent_raw = row["csType"].strip().lower()
                intent_map = {
                    "informative": "informative",
                    "denouncing": "denouncing",
                    "questioning": "question",
                    "positive": "positive"
                }
                intent = intent_map.get(intent_raw)
                prompt = PROMPT.format(intent=intent.capitalize(), hate=row["hatespeech"].strip())
                full = prompt + row["counterspeech"].strip()
                encoded = self.tokenizer(full, truncation=True, padding="max_length", max_length=self.max_len)
                encoded["labels"] = encoded["input_ids"].copy()
                return encoded

            return self.ds.map(encode, remove_columns=self.ds["train"].column_names)

        def cache(self, path="cached_dataset"):
            self.enc_ds.save_to_disk(path)

    data_obj = PatchedData(dataset, tokenizer_obj.tokenizer, cfg["max_len"])
    data_obj.cache(cfg["dataset_cache"])

    model = LoRAOPT(cfg["model_name"], **cfg["lora"])
    trainer = CSTrainer(model, tokenizer_obj.tokenizer, data_obj.enc_ds,
                        out_dir=cfg["output_dir"], epochs=cfg["epochs"],
                        bs=cfg["batch_size"], grad_acc=cfg["grad_acc"], lr=cfg["lr"])
    trainer.fit()

    evaluator = CSEvaluator(cfg["output_dir"], cfg["dataset_cache"])
    evaluator.evaluate()

# --- Config ---
inline_config = {
    "model_name": "facebook/opt-350m",
    "tokenizer_dir": "tokenizer",
    "dataset_cache": "cached_dataset",
    "output_dir": "outputs/opt350m_lora_final",
    "max_len": 256,
    "epochs": 1,
    "batch_size": 2,
    "grad_acc": 2,
    "lr": 2e-4,
    "lora": {
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "quant4bit": False
    },
    "debug_mode": False
}

# Run the pipeline
main(inline_config)
