#python script to add a GRPO layer on top of an already fine-tuned LLM.
#This script uses the GRPOTrainer from the trl library to train a model using a dataset of Zulu math problems.
#Corey Webb, UCT, BSc Hons 2025
import re
import pandas as pd
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
# Will need to log into your wandb account first
# import wandb 


#####################
# Config
#####################
#Make sure to check all of these
dataset = "GSM8K_Zulu.parquet"
job_name = "QZAZ_GRPO"
base_path = "/scratch/wbbcor001/Qwen_QZAZ_Finetune" # Path to the fine-tuned model


#wandb logging
# wandb.init(project=f"{job_name}", name=f"{job_name}")


# 4. Define reward function
def format_reward_func(completions, ground_truth, **kwargs):
        """Reward function:
        - 1 if format is correct AND answer matches ground_truth
        - -0.5 if format is correct only
        - -1 otherwise
        """
        rewards = []
    
        completion_contents = []
        for completion in completions:
            if isinstance(completion, list) and len(completion) > 0:
                completion_contents.append(completion[0]["content"])
            elif isinstance(completion, str):
                completion_contents.append(completion)
            else:
                completion_contents.append("")
    
        num_generations = len(completion_contents) // len(ground_truth) # num generations doesn't match ground truth (We need 8 ground truth because we sample 8 times for each question) so we need to tile
        ground_truth_tiled = [gt for gt in ground_truth for _ in range(num_generations)]

        for content, gt in zip(completion_contents, ground_truth_tiled):
            match = re.search(r"####\s*(-?\d+)\s*$", content)
            if not match:
                match = re.search(r"####\s*(-?\d+)", content)
            
            if match:
                answer = match.group(1).strip()
                if answer == gt.strip():
                    rewards.append(1)  # Correct format + correct answer
                else:
                    rewards.append(-0.5)  # Correct format, wrong answer
            else:
                rewards.append(-1)      # Wrong format
        return rewards

parquet_file = dataset
df = pd.read_parquet(parquet_file)
# Prompt format
prompts = []
for question in df['question']:
    prompt = f"Q: {question}\nA: "
    prompts.append(prompt)

# Create dataset with prompts and ground truth
ground_truth = df['answer'].astype(str).tolist()
dataset = Dataset.from_dict({
    "prompt": prompts,
    "ground_truth": ground_truth                        
})


#  LoRA (Parameter-efficient training)
lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=16,                    # 16 retains almost all performance of full fine-tuning
    lora_alpha=32, #this value is usually 2x the r value
    lora_dropout=0.1, #Seemed to work well
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], #Not sure which target modules are best for Qwen2.5
    bias="none",
)

#  training settings
training_args = GRPOConfig(
    output_dir=f"output/{job_name}",
    per_device_train_batch_size=2,        # Small batch size due to memory constraints
    gradient_accumulation_steps=4,
    bf16=True,                 # Saves VRAM
    max_completion_length=512,         #Generated output usually completed before 512 tokens
    max_prompt_length=512,  
    num_generations=8,  # Number of generations per prompt
    remove_unused_columns=False,      # Keep ground_truth column
    logging_steps=10,
    save_strategy="steps",
    save_steps=500,
    report_to="wandb", # Look at comments at top of script - will need to log into wandb and initialise
    save_total_limit=3,  # Keep only the last 3 checkpoints bc limited disk space
)


model = AutoModelForCausalLM.from_pretrained(base_path) #Load the fine-tuned base model
model = get_peft_model(model, lora_config) #Add LoRA layers
print(model.print_trainable_parameters()) # For debug - making sure that params are trainable

# GRPO trainer from trl library
trainer = GRPOTrainer(
    model=model,  
    reward_funcs=[format_reward_func], 
    args=training_args,
    train_dataset=dataset,
)

# training
print(f"Starting training with {len(dataset)} examples...") # For debug 
try:
    trainer.train(resume_from_checkpoint=True)
except:
    trainer.train(resume_from_checkpoint=False)
trainer.model.save_pretrained(f"output/{job_name}/final_model")
print("Training completed!")