import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
import torch
import re
from collections import Counter

# Load test dataset
df = pd.read_csv("hf://datasets/masakhane/afrimgsm/data/zul/test.tsv", sep="\t")
questions = df["question"].tolist()
answers = df["answer_number"].astype(str).tolist()

# Load model and tokenizer
base_model = "Qwen/Qwen2.5-Math-7B"
tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model, torch_dtype=torch.float16, device_map="auto"
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

# Inference config (non-zero temperature for sampling)
generation_args = {
    "max_new_tokens": 512,
    "do_sample": True,
    "temperature": 0.7,
    "pad_token_id": tokenizer.eos_token_id,
    "num_return_sequences": 8,
}

# Extract final answer from CoT (last number in the response)
def extract_final_answer(text):
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    return numbers[-1] if numbers else ""

# Generate and evaluate
pass_at_8 = 0
maj_at_8 = 0
all_preds = []

for i, (q, gt) in enumerate(zip(questions, answers)):
    prompt = f"\nQ: {q}\nA:"
    outputs = generator(prompt, **generation_args)

    cots = []
    preds = []
    for out in outputs:
        full = out["generated_text"]
        match = re.search(r"Q:.*?\nA:(.*)", full, re.DOTALL)
        cot = match.group(1).strip() if match else ""
        pred = extract_final_answer(cot)
        cots.append(cot)
        preds.append(pred)

    correct_count = preds.count(gt)
    if correct_count >= 1:
        pass_at_8 += 1
    if correct_count >= 5:
        maj_at_8 += 1

    print(f"\n Q{i+1}: {q}\n Ground Truth: {gt}\n Predictions: {preds}")
    all_preds.append({
        "question": q,
        "ground_truth": gt,
        "predictions": preds,
        "generated_cots": cots,
        "pass@8": int(correct_count >= 1),
        "maj@8": int(correct_count >= 5),
    })

# Final metrics
total = len(questions)
print(f"\n pass@8: {pass_at_8}/{total} = {pass_at_8/total:.3f}")
print(f"maj@8: {maj_at_8}/{total} = {maj_at_8/total:.3f}")

# Save results
pd.DataFrame(all_preds).to_json("cot_predictions_pass8_maj8.json", orient="records", indent=2)
