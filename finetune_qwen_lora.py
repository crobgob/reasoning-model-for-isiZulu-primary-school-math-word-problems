import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    default_data_collator,
)
from peft import LoraConfig, get_peft_model
import os

print("🔍 Starting script...")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
print(f"GPU Name: {torch.cuda.get_device_name(0)}")
print(f"Allocated GPU memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
print(f"Reserved GPU memory: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
print(f"Environment PATH: {os.environ.get('PATH')}")
print(f"Conda env: {os.environ.get('CONDA_DEFAULT_ENV')}")

# Load and randomise dataset from Parquet
df = pd.read_parquet("QZuluAEnglish.parquet").sample(frac=1.0, random_state=42).reset_index(drop=True)

# Format each row as a string: "Q: question\nA: chain-of-thought"
formatted_texts = [f"Q: {q}\nA: {cot}" for q, cot in zip(df["question"], df["CoT"])]

# Load tokenizer and model
model_name = "Qwen/Qwen2.5-Math-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
except Exception as e:
    print(f"Model loading failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

model.gradient_checkpointing_enable()
model.config.use_cache = False

# Configure LoRA
peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, peft_config)

# Define a PyTorch-compatible dataset
class QADataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer(
            text, 
            truncation=True, 
            max_length=self.max_length, 
            padding= "max_length",
            return_tensors="pt"
        )
        input_ids = tokens["input_ids"].squeeze()
        attention_mask = tokens["attention_mask"].squeeze()
        
        # Create labels (same as input_ids for causal LM)
        labels = input_ids.clone()
        
        # Mask the prompt part
        prompt_part = text.split("A:")[0] + "A: "
        prompt_tokens = self.tokenizer(prompt_part, add_special_tokens=False)["input_ids"]
        prompt_len = min(len(prompt_tokens), len(labels))
        labels[:prompt_len] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

# Create dataset and split into 80% train, 20% validation
full_dataset = QADataset(formatted_texts, tokenizer)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, eval_dataset = random_split(full_dataset, [train_size, val_size])

# Simple data collator that doesn't do additional processing
from transformers import default_data_collator
data_collator = default_data_collator

# Training configuration
training_args = TrainingArguments(
    output_dir="output",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=16,  # maybe increase
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",
    gradient_checkpointing=True,
)

# Train using Hugging Face Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

trainer.train()

# Save final LoRA adapter weights
trainer.model.save_pretrained("output/QZAE")