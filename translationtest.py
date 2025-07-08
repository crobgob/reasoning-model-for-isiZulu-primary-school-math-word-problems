from transformers import AutoProcessor, SeamlessM4Tv2Model


processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")

prompt = "Please make sure to preserve all numbers and math notation. "
text = "Janet sells 16 - 3 - 4 = 9 duck eggs a day. She makes 9 * 2 = $18 every day at the farmer’s market."

# Combine prompt and text
combined_input = prompt + text

text_inputs = processor(
    text=combined_input,
    src_lang="eng",
    return_tensors="pt"
)

output_tokens = model.generate(**text_inputs, tgt_lang="zul", generate_speech=False)
translated_text = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
print(translated_text)
