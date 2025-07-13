import pandas as pd

gsm8k = pd.read_parquet("hf://datasets/openai/gsm8k/main").head(10) 
df = pd.read_parquet("GSM8K_Zulu.parquet")
print(gsm8k.iloc[0]['answer'])
print(df.iloc[0]['CoT'])

