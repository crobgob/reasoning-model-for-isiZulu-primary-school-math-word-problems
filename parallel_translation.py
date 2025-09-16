# Parallelised version of TranslateGSM8K.py
# Note line 61 I have reduced dataset size to 10 for safety.
# Program that creates a parquet dataframe for GSM8K translated to Zulu. We will have three columns: question, CoT and answer.
# Corey Webb, BSc Hons 2025, UCT
# 10/07/2025

import pandas as pd
from google.cloud import translate
from google.oauth2 import service_account
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Config
PROJECT_ID = "boreal-rain-465311-e2"
CREDENTIALS_PATH = "D:/HonoursProject/keys/boreal-rain-465311-e2-65805b7c44c1.json"
PARQUET_OUTPUT = "GSM8K_Zulu.parquet"
NUM_THREADS = 20

# Translation setup (reuse client globally)
credentials = service_account.Credentials.from_service_account_file(CREDENTIALS_PATH)
client = translate.TranslationServiceClient(credentials=credentials)
location = "global"
parent = f"projects/{PROJECT_ID}/locations/{location}"

def translate_to_zul(text):
    try:
        response = client.translate_text(
            request={
                "parent": parent,
                "contents": [text],
                "mime_type": "text/plain",
                "source_language_code": "en-US",
                "target_language_code": "zu",
            }
        )
        return response.translations[0].translated_text
    except Exception as e:
        print(f"[ERROR] Translation failed: {e}")
        time.sleep(2)
        return translate_to_zul(text)  # Retry once recursively

def extract_ans(cot_answer):
    split = cot_answer.split("####")
    if len(split) == 2:
        return int(split[1].replace(",", "").strip())
    else:
        raise ValueError(f"Expected one answer, found {len(split) - 1}")

def process_row(index, question, cot):
    print(f"Processing row {index}")
    try:
        q_trans = translate_to_zul(question)
        cot_trans = translate_to_zul(cot)
        ans = extract_ans(cot)
        return {"index": index, "question": q_trans, "CoT": cot_trans, "answer": ans}
    except Exception as e:
        print(f"[ERROR] Row {index} failed: {e}")
        return None

def main():
    df = pd.read_parquet("hf://datasets/openai/gsm8k/main").head(10) #For safety
    rows = [(i, row['question'], row['answer']) for i, row in df.iterrows()]

    results = []
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = [executor.submit(process_row, idx, q, a) for idx, q, a in rows]
        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

    # Sort by original index to maintain order
    results_sorted = sorted(results, key=lambda x: x["index"])
    zulu_df = pd.DataFrame(results_sorted).drop(columns=["index"])

    zulu_df.to_parquet(PARQUET_OUTPUT, index=False)
    print(f"[DONE] Saved {len(zulu_df)} rows to {PARQUET_OUTPUT}")

if __name__ == "__main__":
    main()
