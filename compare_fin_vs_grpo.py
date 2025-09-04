# Python script to compare BASE vs BASE+PEFT on AfriMGSM Zulu (exact-match with #### <number>)

import os
import re
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ----------------------------
# Config
# ----------------------------
TEST_PATH = "hf://datasets/masakhane/afrimgsm/data/zul/test.tsv"
BASE_MODEL = "/scratch/wbbcor001/Qwen_QZAE_Finetune"
ADAPTER_PATH = "output/QZAE_GRPO/final_model"

MAX_NEW_TOKENS = 512
OUT_DIR = "predictions"
os.makedirs(OUT_DIR, exist_ok=True)

CSV_BASE   = os.path.join(OUT_DIR, "generated_predictions_BASE.csv")
CSV_PEFT   = os.path.join(OUT_DIR, "generated_predictions_PEFT.csv")
CSV_COMBO  = os.path.join(OUT_DIR, "generated_predictions_COMBINED.csv")
EVAL_BASE  = os.path.join(OUT_DIR, "evaluation_BASE.csv")
EVAL_PEFT  = os.path.join(OUT_DIR, "evaluation_PEFT.csv")

# ----------------------------
# Helpers
# ----------------------------
def extract_pred_answer(cot: str):
    """
    Extract the predicted numeric answer from the model's CoT/output.

    Priority:
      1) '#### <number>'
    """
    # Handle standard and early-occurring #### markers
    m = re.search(r"####\s*(-?\d+)\s*$", cot)
    if not m:
        m = re.search(r"####\s*(-?\d+)", cot)
    if m:
        return m.group(1).strip()
    return None

@torch.inference_mode()
def generate_cot(model, tokenizer, question: str, device: str):
    prompt = f"Q: {question}\nA: "
    toks = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **toks,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    # Decode only new tokens beyond the prompt length
    gen_text = tokenizer.decode(out[0][toks['input_ids'].shape[1]:], skip_special_tokens=True)
    return gen_text

def evaluate(df_with_cot: pd.DataFrame, label_col: str, cot_col: str):
    correct = 0
    total = 0
    bad_format = 0
    rows = []
    for row in df_with_cot.itertuples(index=False):
        gold = str(getattr(row, label_col)).strip()
        cot  = str(getattr(row, cot_col))
        pred = extract_pred_answer(cot)
        if pred is None:
            bad_format += 1
        is_correct = (pred == gold)
        if is_correct:
            correct += 1
        total += 1
        rows.append({
            "question": getattr(row, "question"),
            "gold_answer": gold,
            "pred_answer": pred,
            "correct": is_correct,
            "cot": cot,
        })
    acc = (correct / total * 100.0) if total else 0.0
    return acc, bad_format, total, pd.DataFrame(rows)

# ----------------------------
# Load test dataset
# ----------------------------
df_test = pd.read_csv(TEST_PATH, sep="\t")
questions = df_test["question"].tolist()

# ----------------------------
# Load tokenizer & two separate base instances
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token_id

# A) Pristine base for baseline
base_A = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map=None,
    low_cpu_mem_usage=False,
).to(device).eval()

# B) Fresh base that we'll wrap with PEFT
base_B = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map=None,
    low_cpu_mem_usage=False,
).to(device).eval()

peft_B = PeftModel.from_pretrained(base_B, ADAPTER_PATH, is_trainable=False).to(device).eval()

# ----------------------------
# Generate CoTs for both models
# ----------------------------
print("Generating CoTs for BASE and PEFT...")
cots_base = []
cots_peft = []
for i, q in enumerate(questions, 1):
    cot_base = generate_cot(base_A, tokenizer, q, device)
    cot_peft = generate_cot(peft_B, tokenizer, q, device)
    cots_base.append(cot_base)
    cots_peft.append(cot_peft)
    if i % 25 == 0 or i == len(questions):
        print(f"  done {i}/{len(questions)}")

# Save generations
df_out_base = df_test.copy()
df_out_base["generated_cot"] = cots_base
df_out_base.to_csv(CSV_BASE, index=False)

df_out_peft = df_test.copy()
df_out_peft["generated_cot"] = cots_peft
df_out_peft.to_csv(CSV_PEFT, index=False)

# ----------------------------
# Evaluate both and print summary
# ----------------------------
acc_base, bad_base, total, eval_base_df = evaluate(df_out_base, "answer_number", "generated_cot")
acc_peft, bad_peft, _,     eval_peft_df = evaluate(df_out_peft, "answer_number", "generated_cot")

eval_base_df.to_csv(EVAL_BASE, index=False)
eval_peft_df.to_csv(EVAL_PEFT, index=False)

# Combined comparison
combo = pd.DataFrame({
    "question": df_test["question"],
    "gold_answer": df_test["answer_number"].astype(str),
    "cot_base": cots_base,
    "cot_peft": cots_peft,
})
combo["pred_base"] = combo["cot_base"].apply(extract_pred_answer)
combo["pred_peft"] = combo["cot_peft"].apply(extract_pred_answer)
combo["diff_pred"] = combo["pred_base"] != combo["pred_peft"]
combo.to_csv(CSV_COMBO, index=False)

diff_rate = 100.0 * combo["diff_pred"].mean()

print("\n================ SUMMARY ================")
print(f"Total examples:                {total}")
print(f"BASE  accuracy (EM):           {acc_base:.2f}%   | bad-format: {bad_base}")
print(f"PEFT  accuracy (EM):           {acc_peft:.2f}%   | bad-format: {bad_peft}")
print(f"Predictions different (BASE vs PEFT): {diff_rate:.2f}%")
print("Outputs:")
print(f"  - BASE generations: {CSV_BASE}")
print(f"  - PEFT generations: {CSV_PEFT}")
print(f"  - Combined compare: {CSV_COMBO}")
print(f"  - BASE eval rows:   {EVAL_BASE}")
print(f"  - PEFT eval rows:   {EVAL_PEFT}")
