import pandas as pd
# parquet_file = 'GSM8K_Zulu.parquet'
# df = pd.read_parquet(parquet_file)

# print(df.iloc[1]['CoT'])

# # Display the first few rows
# print(df.head())

csv_file = pd.read_csv('predictions/generated_predictions.csv')

for i, row in csv_file.iterrows():
    print(f"QUESTION {i}:")
    print(csv_file.iloc[i]['question'])
    print("ANSWER:")
    print(csv_file.iloc[i]['answer_number'])
    print("GENERATED COT:")
    print(csv_file.iloc[i]['generated_cot'])
