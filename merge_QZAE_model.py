
from peft import PeftModel
from transformers import AutoModelForCausalLM


base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Math-7B")
model = PeftModel.from_pretrained(base_model, "output/QZAE")

merged_model = model.merge_and_unload()
merged_model.save_pretrained("./output/Qwen_QZAE_Finetune/")