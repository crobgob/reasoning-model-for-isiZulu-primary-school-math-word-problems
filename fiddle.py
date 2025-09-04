
from peft import PeftModel
from transformers import AutoModelForCausalLM


base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Math-7B")
model = PeftModel.from_pretrained(base_model, "/home/wbbcor001/finetuning-qwen/output/QZAE_GRPO_BASE/checkpoint-13188")

model.save_pretrained("output/QZAE_GRPO_BASE/pretrained")