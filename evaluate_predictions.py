import pandas as pd
import re

# Load generated predictions
df = pd.read_csv("QZAE/2 epoch train/generated_predictions.csv")

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
