# Python script to generate Chain of Thought (CoT) for a set of questions using a pre-trained model,
# then evaluate exact-match accuracy.
# \boxed{} format

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
BASE_MODEL = "Qwen/Qwen2.5-Math-7B"
ADAPTER_PATH = "output/QZAE_GRPO_BASE/checkpoint-2000" #TODO add correct checkpoint
MAX_NEW_TOKENS = 512
OUT_CSV = "generated_predictions_grpo_base.csv"
EVAL_DIR = "predictions"
EVAL_PATH = os.path.join(EVAL_DIR, "evaluation_QZAE_GRPO_BASE.csv")


def extract_pred_answer(cot: str):
    """
    Extract the predicted numeric answer from the model's CoT/output.
    Only extracts answers in \boxed{} format.
    """
    # Look for \boxed{answer} pattern
    m = re.search(r"\\boxed\{([^}]*)\}", cot)
    if m:
        answer_content = m.group(1).strip()
        
        # Try to extract a numeric value from the boxed content
        # Handle negative numbers and decimals
        num_match = re.search(r"-?\d+(?:\.\d+)?", answer_content)
        if num_match:
            return num_match.group(0), True
        
        # If no numeric value found, return the raw content
        return answer_content, True
    
    # No \boxed{} format found
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
    prompt = f"Question: {q}\nPlease reason step by step, and put your final answer within \\boxed{{}}"
    output = generator(prompt, **generation_args)[0]["generated_text"]

    # Keep only the text after "A:"
    match = re.search(r"Question:.*?\nPlease reason step by step, and put your final answer within(.*)", output, re.DOTALL)
    cot = match.group(1).strip() if match else ""

    print(f"\n Q{i+1}: {q}\n Generated CoT: {cot}")
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
print(f"CoT with no boxed: {bad_format}")
print(f"\nAccuracy: {accuracy:.2f}%")

# ----------------------------
# Persist
