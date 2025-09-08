# Python script to generate Chain of Thought (CoT) for a set of questions using a pre-trained model.

import os
import re
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

# ----------------------------
# Config
# ----------------------------
NAME = "QZAZ"
TEST_PATH = "hf://datasets/masakhane/afrimgsm/data/zul/test.tsv"
BASE_MODEL = "/scratch/wbbcor001/Qwen_QZAZ_Finetune"
ADAPTER_PATH = "output/QZAZ_GRPO/final_model"
MAX_NEW_TOKENS = 512
OUT_CSV = f"generated_predictions_{NAME}.csv"
EVAL_DIR = "predictions"
EVAL_PATH = os.path.join(EVAL_DIR, f"evaluation_{NAME}.csv")


#Answer extraction - looks for #### <number> pattern
def extract_pred_answer(cot: str):
    # 1) Try #### pattern
    m = re.search(r"####\s*(-?\d+)\s*$", cot)
    if not m:
        m = re.search(r"####\s*(-?\d+)", cot)
    if m:
        return m.group(1).strip(), False
    return None, False # No answer found


#Test dataset
df = pd.read_csv(TEST_PATH, sep="\t")
questions = df["question"].tolist()

#Loading model + adapter path and tokeniser
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
base_model_clean = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
)

# Load another instance of the base and apply the adapter to that instance
base_model_for_adapter = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
)

# Apply the saved adapter to the separate base instance. The resulting object is the
# model we will use for generation/evaluation.
model = PeftModel.from_pretrained(base_model_for_adapter, ADAPTER_PATH).eval()

# Generation pipeline - is this necessary?
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


# Prompt for CoT on the 250 test questions
generated_cots = []
for i, q in enumerate(questions):
    prompt = f"Q: {q}\n A: "
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the new tokens
    generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    print(f"\nQ{i+1}: {q}\n Generated CoT: {generated_text}")
    generated_cots.append(generated_text)


# ----------------------------
# Save generations
# ----------------------------
df["generated_cot"] = generated_cots
df.to_csv(OUT_CSV, index=False)


# Evaluate
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
