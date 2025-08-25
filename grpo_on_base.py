#grpo for eliciting reasoning in Zulu math problems
# This script uses the GRPOTrainer from the trl library to train a model on Zulu math problems based on the GSM8K dataset.
#Corey Webb, UCT, BSc Hons 2025
import re
import pandas as pd
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig
from transformers import AutoModelForCausalLM
from peft import PeftModel
import wandb

#wandb logging
wandb.init(project="grpo-base-AfriMGSM", name="qwen-grpo-base")


# 4. Define reward function
def format_reward_func(completions, ground_truth, **kwargs):
    """Reward function:
    - 1 if format is correct AND answer matches ground_truth
    - -0.5 if format is correct only (contains \\boxed{})
    - -1 otherwise
    """
    rewards = []

    # Handle different completion formats
    completion_contents = []
    for completion in completions:
        if isinstance(completion, list) and len(completion) > 0:
            completion_contents.append(completion[0]["content"])
        elif isinstance(completion, str):
            completion_contents.append(completion)
        else:
            completion_contents.append("")

    num_generations = len(completion_contents) // len(ground_truth)  # num generations doesn't match ground truth so we need to tile
    ground_truth_tiled = [gt for gt in ground_truth for _ in range(num_generations)]

    # Look for \\boxed{} pattern
    for content, gt in zip(completion_contents, ground_truth_tiled):
        # Look for \\boxed{answer} pattern (LaTeX format)
        boxed_match = re.search(r"\\boxed\{([^}]+)\}", content)
        
        if boxed_match:
            # Extract answer from boxed format
            answer = boxed_match.group(1).strip()
            # Handle numeric answers - extract just the number if it's numeric
            numeric_match = re.search(r'-?\d+(?:\.\d+)?', answer)
            if numeric_match:
                answer = numeric_match.group(0)
            
            if answer == gt.strip():
                rewards.append(1)  # Correct format + correct answer
            else:
                rewards.append(-0.5)  # Correct format, wrong answer
        else:
            rewards.append(-1)  # Wrong format (no \\boxed{} found)
    
    return rewards

# 1. Load the Zulu GSM8K dataset
parquet_file = 'QZuluAEnglish.parquet'
df = pd.read_parquet(parquet_file)

# 2. Create prompts for each question
prompts = []
for question in df['question']:
    prompt = f"Question: {question}\n Please reason step by step, and put your final answer within \\boxed{{}}" # Prompy according to the Qwen2.5B Math paper - figure 5
    prompts.append(prompt)

# 3. Create dataset with prompts and ground truth
ground_truth = df['answer'].astype(str).tolist()
dataset = Dataset.from_dict({
    "prompt": prompts,
    "ground_truth": ground_truth
})


# 5. Configure LoRA (Parameter-efficient training)
peft_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=16,                    # Rank (lower = less memory)
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    bias="none",
)

# 6. Configure training settings
training_args = GRPOConfig(
    output_dir="output/QZAE_GRPO_BASE",
    per_device_train_batch_size=2,        # Small batch size due to memory constraints
    gradient_accumulation_steps=4,
    bf16=True,                 # Saves VRAM
    max_completion_length=512,         
    max_prompt_length=512,  
    num_generations=8,                    # Number of generations per prompt
    remove_unused_columns=False,          # Keep ground_truth column
    logging_steps=10,
    save_strategy="steps",
    save_steps=500,
    report_to="wandb",
    save_total_limit=3,  # Keep only the last 3 checkpoints
)


# Load base model and tokenizer
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Math-7B")

# 7. Create trainer
trainer = GRPOTrainer(
    model=model,      #Qwen/Qwen2-0.5B-Instruct; "Qwen/Qwen2.5-Math-7B"
    peft_config=peft_config,              # Add LoRA configuration
    reward_funcs=[format_reward_func],    # Fixed: use list and correct function name
    args=training_args,
    train_dataset=dataset,
)

# 8. Start training
print(f"Starting training with {len(dataset)} examples...")
trainer.train(resume_from_checkpoint=False)
print("Training completed!")