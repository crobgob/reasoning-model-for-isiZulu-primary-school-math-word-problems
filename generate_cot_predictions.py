#Python script to generate Chain of Thought (CoT) for a set of questions using a pre-trained model.

import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
import torch
import re

# Load test dataset
df = pd.read_csv("hf://datasets/masakhane/afrimgsm/data/zul/test.tsv", sep='\t')
questions = df["question"].tolist()

# Load base model and tokenizer
base_model = "Qwen/Qwen2.5-Math-7B"
tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)

# Load base + LoRA adapter
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    device_map="auto"
)
model = PeftModel.from_pretrained(model, "output/final_lora")
model.eval()

# Text generation pipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    torch_dtype=torch.float16,
)

# Inference config
generation_args = {
    "max_new_tokens": 512,
    "do_sample": False,
    "temperature": 0.7,
    "pad_token_id": tokenizer.eos_token_id,
}

# Generate CoT for each question
generated_cots = []
for i, q in enumerate(questions):
    prompt = f"Give the correct answer to this math problem. Show working out concisely and step-by-step. Q: {q}\nA:"
    output = generator(prompt, **generation_args)[0]["generated_text"]
    
    # Only keep the CoT part (after "A:")
    match = re.search(r"Q:.*?\nA:(.*)", output, re.DOTALL)
    cot = match.group(1).strip() if match else ""
    
    print(f"\n📍 Q{i+1}: {q}\n🧠 Generated CoT: {cot}")
    generated_cots.append(cot)

# Save results
df["generated_cot"] = generated_cots
df.to_csv("generated_predictions.csv", index=False)
