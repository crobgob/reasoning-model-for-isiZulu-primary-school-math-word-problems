# Python script to generate Chain of Thought (CoT) for a set of questions using a pre-trained model,
# then evaluate exact-match accuracy. Extended to also parse answers from
# <think>...</think><answer>...</answer> format (ignoring the think part).

import os
import re
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

# ----------------------------
# Config
# ----------------------------
TEST_PATH = "hf://datasets/masakhane/afrimgsm/data/zul/test.tsv"
BASE_MODEL = "/scratch/wbbcor001/Qwen_QZAE_Finetune"
ADAPTER_PATH = "output/QZAE_GRPO/checkpoint-12500"
MAX_NEW_TOKENS = 512
OUT_CSV = "generated_predictions_grpo.csv"
EVAL_PATH = "predictions"

# ----------------------------
# Helpers
# ----------------------------
def extract_pred_answer(cot: str):
    """
    Extract the predicted numeric answer from the model's CoT/output.
    Priority:
      1) '#### {answer}' anywhere in text (modified format).
    Returns:
      - pred_answer (str | None): stringified number if found, else None
      - from_answer_tags (bool): True if extracted via <answer> tags, False if via ####
    """
    # 1) Try #### {answer} format first
    # Look for #### followed by any content in curly braces
    pattern = r"####\s*(-?\d+)\s*$"
    match = re.search(pattern, cot, re.DOTALL | re.IGNORECASE)
    
    if match:
        answer_content = match.group(1).strip()
        # Extract number from the content, handling negatives, decimals, and commas
        # Remove commas first, then look for number pattern
        cleaned = answer_content.replace(',', '')
        # Pattern for negative numbers with optional decimal points
        num_match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
        if num_match:
            number = num_match.group(0)
            # Convert to integer if it's a whole number, otherwise keep as string
            if '.' in number and number.endswith('.0'):
                return str(int(float(number))), False
            elif '.' not in number:
                return number, False
            else:
                # Return as string to preserve decimal
                return number, False
    # Nothing found
    return None, False


# ----------------------------
# Load test dataset
# ----------------------------
df = pd.read_csv(TEST_PATH, sep="\t")
questions = df["question"].tolist()

# ----------------------------
# Load model & tokenizer
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
)
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()

# ----------------------------
# Generation pipeline
# ----------------------------
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    torch_dtype=torch.float16,
)

generation_args = {
    "max_new_tokens": MAX_NEW_TOKENS,
    "do_sample": False,
    "pad_token_id": tokenizer.eos_token_id,
}

# ----------------------------
# Generate CoT for each question
# ----------------------------
generated_cots = []
for i, q in enumerate(questions):
    prompt = f"Q: {q}\nA: "
    output = generator(prompt, **generation_args)[0]["generated_text"]
    
    # Keep only the text after "A:"
    match = re.search(r"A:\s*(.*)", output, re.DOTALL | re.IGNORECASE)
    cot = match.group(1).strip() if match else ""
    
    print(f"\nQ{i+1}: {q}\nGenerated CoT: {cot}")
    generated_cots.append(cot)

# ----------------------------
# Save generations
# ----------------------------
df["generated_cot"] = generated_cots
df.to_csv(OUT_CSV, index=False)

# ----------------------------
# Evaluate predictions
# ----------------------------
df = pd.read_csv(OUT_CSV)

correct = 0
total = 0
bad_format = 0
results = []

for i, row in enumerate(df.itertuples(index=False)):
    gold_answer = str(getattr(row, "answer_number")).strip()
    cot = str(getattr(row, "generated_cot"))

    pred_answer, used_alt = extract_pred_answer(cot)

    if pred_answer is None:
        bad_format += 1

    is_correct = (pred_answer == gold_answer)
    if is_correct:
        correct += 1
    total += 1

    results.append({
        "question": getattr(row, "question"),
        "gold_answer": gold_answer,
        "pred_answer": pred_answer,
        "correct": is_correct,
        "extracted_from_tags": used_alt,
    })

accuracy = (correct / total * 100.0) if total else 0.0

print(f"\nTotal: {total}")
print(f"Correct: {correct}")
print(f"Incorrect: {total - correct}")
print(f"CoT with no answer found (no ####): {bad_format}")
print(f"\nAccuracy: {accuracy:.2f}%")

# ----------------------------
# Persist
