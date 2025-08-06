import pandas as pd

# Load Zulu dataset
df_zulu = pd.read_parquet('GSM8K_Zulu.parquet')

# Load GSM8K English CoT answers
df_gsm8k_main = pd.read_parquet("hf://datasets/openai/gsm8k/main")

# Replace Zulu CoT with English CoT
df_zulu["CoT"] = df_gsm8k_main["answer"].values

# Save result
df_zulu.to_parquet("QZuluAEnglish.parquet")

print("QZuluAEnglish.parquet saved successfully.")
