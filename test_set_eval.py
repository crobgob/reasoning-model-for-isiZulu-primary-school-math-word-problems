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
model = PeftModel.from_pretrained(model, "RLmodel/best_model")
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
    prompt = f"Q: {q}\nA:"
    output = generator(prompt, **generation_args)[0]["generated_text"]
    
    # Only keep the CoT part (after "A:")
    match = re.search(r"Q:.*?\nA:(.*)", output, re.DOTALL)
    cot = match.group(1).strip() if match else ""
    
    print(f"\n📍 Q{i+1}: {q}\n🧠 Generated CoT: {cot}")
    generated_cots.append(cot)

# Save results
df["generated_cot"] = generated_cots
df.to_csv("generated_predictions.csv", index=False)

# Load generated predictions
df = pd.read_csv("generated_predictions.csv")

correct = 0
total = 0
bad_format = 0
results = []

for i, row in df.iterrows():
    gold_answer = str(row['answer_number']).strip()
    cot = str(row['generated_cot'])

    # Extract the final number after the last ####
    match = re.search(r"####\s*(-?\d+)\s*$", cot)

    if match:
        pred_answer = match.group(1).strip()
    else:
        pred_answer = None
        bad_format += 1

    is_correct = pred_answer == gold_answer
    if is_correct:
        correct += 1
    total += 1

    results.append({
        "question": row['question'],
        "gold_answer": gold_answer,
        "pred_answer": pred_answer,
        "correct": is_correct
    })

# Summary
accuracy = correct / total * 100
print(f"\nTotal: {total}")
print(f"Correct: {correct}")
print(f"Incorrect: {total - correct}")
print(f"CoT with no answer found (no ####): {bad_format}")
print(f"\nAccuracy: {accuracy:.2f}%")

# Optional: save per-question evaluation
eval_df = pd.DataFrame(results)
eval_df.to_csv("predictions/evaluation_results.csv", index=False)
print("\n📁 Saved detailed results to predictions/evaluation_results.csv")
