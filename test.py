# first_questions_with_cot_testset.py
import re
from pathlib import Path

import pandas as pd

try:
    from datasets import load_dataset
except ImportError as e:
    raise SystemExit(
        "The 'datasets' library is required for GSM8K. Install with:\n"
        "  pip install datasets\n"
        "Then re-run this script."
    ) from e


def extract_cot_from_gsm8k_answer(answer_text: str) -> str:
    """
    Extract CoT trace from GSM8K 'answer' field.
    Strips off the final '#### <final answer>'.
    """
    if answer_text is None:
        return ""
    parts = answer_text.split("####", 1)
    cot = parts[0].rstrip()
    return cot


def get_first_gsm8k_test():
    """
    Load GSM8K (main) from HF datasets and return
    first test sample (question, CoT).
    """
    ds = load_dataset("openai/gsm8k", "main")
    first = ds["test"][0]   # explicitly use TEST split
    question = first.get("question", "").strip()
    answer_full = first.get("answer", "")
    cot = extract_cot_from_gsm8k_answer(answer_full)
    return question, cot


def get_first_from_parquet(parquet_path: str, q_col="question", cot_col="CoT"):
    """
    Load first row from a parquet dataset.
    """
    p = Path(parquet_path)
    if not p.exists():
        raise FileNotFoundError(f"Parquet not found: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    if df.empty:
        raise ValueError(f"No rows in {parquet_path}")

    # handle case-insensitive column names
    cols = {c.lower(): c for c in df.columns}
    q_key = cols.get(q_col.lower())
    cot_key = cols.get(cot_col.lower())
    if q_key is None or cot_key is None:
        raise KeyError(f"Columns '{q_col}' or '{cot_col}' not found in {parquet_path}. Found: {list(df.columns)}")

    question = str(df.iloc[0][q_key]).strip()
    cot = str(df.iloc[0][cot_key]).strip()
    return question, cot


def pretty_print(title: str, question: str, cot: str):
    print("=" * 80)
    print(title)
    print("-" * 80)
    print("Question:\n", question, "\n", sep="")
    print("CoT trace:\n", cot, "\n", sep="")
    print("=" * 80, "\n")


if __name__ == "__main__":
    # GSM8K test split
    gsm_q, gsm_cot = get_first_gsm8k_test()
    pretty_print("GSM8K (main, TEST split) — First Question + CoT", gsm_q, gsm_cot)

    # Zulu GSM8K
    zulu_q, zulu_cot = get_first_from_parquet("GSM8K_Zulu.parquet")
    pretty_print("GSM8K_Zulu.parquet — First Question + CoT", zulu_q, zulu_cot)

    # QZuluAEnglish
    qzae_q, qzae_cot = get_first_from_parquet("QZuluAEnglish.parquet")
    pretty_print("QZuluAEnglish.parquet — First Question + CoT", qzae_q, qzae_cot)
