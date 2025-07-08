import pandas as pd

# Reads in test set
df = pd.read_csv("hf://datasets/masakhane/afrimgsm/data/eng/test.tsv", sep='\t')
q_num = 1
first_question = df.iloc[q_num]['question']
first_answer = df.iloc[q_num]['answer_number']
print(len(df))
print(first_question)
print("################################")
print(first_answer)
