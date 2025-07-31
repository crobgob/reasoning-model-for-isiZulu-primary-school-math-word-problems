import pandas as pd


# parquet_file = 'GSM8K_Zulu.parquet'
# df = pd.read_parquet(parquet_file)



# Zulu gsm8k sample
# print(df.head())

# #GSM8K
# gsm8k = pd.read_parquet("hf://datasets/openai/gsm8k/main")
# first_question = gsm8k.iloc[0]['question']
# first_answer = gsm8k.iloc[0]['answer']
# print(first_question)
# print(first_answer)

# #Zulu gsm8k parts
# print('#################\n\n\n')
# print(f'Question:\n{df.iloc[0]['question']}\n')
# print(f'Chain-of-thought:\n{df.iloc[0]['CoT']}\n')
# print(f'Answer:{df.iloc[0]['answer']}\n')

csv_file = pd.read_csv('predictions/predictions_from_raw_model/generated_predictions_base_model.csv')

for i, row in csv_file.iterrows():
    print(f"QUESTION {i}:")
    print(csv_file.iloc[i]['question'])
    print("ANSWER:")
    print(csv_file.iloc[i]['answer_number'])
    print("GENERATED COT:")
    print(csv_file.iloc[i]['generated_cot'])


