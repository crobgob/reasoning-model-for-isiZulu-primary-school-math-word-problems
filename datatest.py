import pandas as pd

df = pd.read_parquet("hf://datasets/openai/gsm8k/main")

first_question = df.iloc[0]['question']
first_answer = df.iloc[0]['answer']
# print(first_question)
print(first_answer)

