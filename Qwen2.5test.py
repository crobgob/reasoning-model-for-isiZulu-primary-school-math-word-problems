# Load model directly
#TODO: Add TODO comments to the code
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Math-7B")

