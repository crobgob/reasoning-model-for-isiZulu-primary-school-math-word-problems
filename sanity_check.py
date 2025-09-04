import torch
from safetensors import safe_open

# Load your trained adapter weights
with safe_open("output/QZAE_GRPO/checkpoint-13188/adapter_model.safetensors", framework="pt") as f:
    trained_weights = {k: f.get_tensor(k) for k in f.keys()}

# Create a fresh adapter with identical config for comparison
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained("/scratch/wbbcor001/Qwen_QZAE_Finetune", torch_dtype=torch.float16)
peft_config = LoraConfig(
    task_type="CAUSAL_LM", r=16, lora_alpha=32, lora_dropout=0.1,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], bias="none",
)
fresh_model = get_peft_model(base_model, peft_config)

print("=== WEIGHT COMPARISON ===")
for name, param in fresh_model.named_parameters():
    if 'lora_' in name:
        fresh_weight = param.data
        # Find corresponding trained weight
        for trained_name, trained_weight in trained_weights.items():
            if trained_name in name or name.replace('base_model.model.', '') == trained_name:
                diff = torch.abs(trained_weight - fresh_weight).mean().item()
                print(f"{name}: mean_diff = {diff:.8f}")
                if diff < 1e-6:
                    print(f"  ⚠️  IDENTICAL to fresh weights!")
                break