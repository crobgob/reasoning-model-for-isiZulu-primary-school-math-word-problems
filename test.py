import pandas as pd

df = pd.read_parquet("GSM8K_Zulu.parquet")
gsm8k = pd.read_parquet("hf://datasets/openai/gsm8k/main")

print(len(gsm8k))
print(len(df))


print(gsm8k.iloc[0:10]['question'])

print(df.iloc[0:10]['question'])

