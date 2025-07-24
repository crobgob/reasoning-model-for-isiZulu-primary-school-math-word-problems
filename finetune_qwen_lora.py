from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

# Load the GSM8K Zulu dataset from the Parquet file
dataset = load_dataset("parquet", data_files={"train": "GSM8K_Zulu.parquet"})["train"]

# Format each example as a prompt (question) + response (chain-of-thought with answer)
def format_qa(example):
    # Assuming the Parquet columns: question, chain_of_thought, answer
    q = example["question"] if "question" in example else example[list(example.keys())[0]]
    cot = example["CoT"] if "CoT" in example else example[list(example.keys())[1]]
    return f"Q: {q}\nA: {cot}"

dataset = dataset.map(lambda ex: {"text": format_qa(ex)})  # add a combined text field

# Initialize tokenizer and model
model_name = "Qwen/Qwen2.5-Math-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # use EOS token for padding if no pad token

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
model.gradient_checkpointing_enable()         # enable grad checkpointing for lower memory
model.config.use_cache = False               # disable cache to reduce memory during training

# Set up LoRA (Low-Rank Adaptation) configuration
peft_config = LoraConfig(
    r=8,                # rank of LoRA updates (low-rank adaptation)
    lora_alpha=16,      # LoRA scaling factor
    lora_dropout=0.05,  # dropout on LoRA weights
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)
model = get_peft_model(model, peft_config)    # wrap the model with LoRA adapters

# Tokenize the dataset for causal LM, masking out the prompt tokens in labels
def tokenize_example(example):
    text = example["text"]
    # Tokenize the full prompt+response text
    tokens = tokenizer(text, truncation=True, max_length=512, padding=False)
    input_ids = tokens["input_ids"]
    # Determine prompt length in tokens (so we can mask prompt in labels)
    prompt_part = text.split("A:")[0] + "A: "   # include the "A: " separator in prompt
    prompt_tokens = tokenizer(prompt_part, add_special_tokens=False)["input_ids"]
    prompt_len = len(prompt_tokens)
    # Create labels: use -100 for prompt tokens, and actual IDs for the response part
    labels = [-100] * prompt_len + input_ids[prompt_len:]
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}

tokenized_ds = dataset.map(tokenize_example, remove_columns=dataset.column_names)

# Data collator for padding sequences to the same length in a batch
data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8, return_tensors="pt")

# Training configuration
training_args = TrainingArguments(
    output_dir="output",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,    # accumulate to simulate a larger batch (adjust as needed)
    learning_rate=2e-4,
    fp16=True,                       # use FP16 for memory efficiency
    logging_steps=50,
    save_strategy="epoch",
    save_total_limit=1,
    report_to="none",
    gradient_checkpointing=True      # also enable grad checkpointing in Trainer
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds,
    data_collator=data_collator
)
trainer.train()

# Save the LoRA adapter weights (so you can reload or merge with base model later)
trainer.model.save_pretrained("output/final_lora")
