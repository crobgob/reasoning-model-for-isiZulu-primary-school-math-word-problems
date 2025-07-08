import pandas as pd

df = pd.read_parquet("hf://datasets/openai/gsm8k/main")

first_question = df.iloc[0]['question']
first_answer = df.iloc[0]['answer']
# print(first_question)
print(first_answer)

from transformers import AutoProcessor, SeamlessM4Tv2Model


processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")


prompt = """Please translate the following text to Zulu. Preserve any math symbols and formatting like "<< >>" exactly. 
Here is an example:

ENGLISH:
Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.
She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market.
#### 18

ZULU:
UJanet uthengisa 16 - 3 - 4 = <<16-3-4=9>>9 amaqanda edada ngosuku.
Wenza u-9 * 2 = $<<9*2=18>>18 nsuku zonke emakethe yomlimi.
#### 18

Now translate this:
"""
combined_input = prompt + first_answer
text_inputs = processor(text = combined_input, src_lang="eng", return_tensors="pt")
output_tokens = model.generate(**text_inputs, tgt_lang="zul", generate_speech=False)
translated_text_from_text = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
print(translated_text_from_text)