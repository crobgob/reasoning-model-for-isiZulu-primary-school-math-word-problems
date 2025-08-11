# train_grpo.py
from datasets import load_dataset
import re
from trl import GRPOConfig, GRPOTrainer
import pandas as pd

# Load the Zulu GSM8K dataset
parquet_file = 'GSM8K_Zulu.parquet'
df = pd.read_parquet(parquet_file)

# Extract ground truth answers as strings
ground_truth = df['answer'].astype(str).tolist()

def format_reward_func(completions, ground_truth, **kwargs):
    """Reward function:
    - 2 if format is correct AND answer matches ground_truth
    - 1 if format is correct only
    - 0 otherwise
    """
    rewards = []
    pattern = r"^<cabanga>.*?</cabanga><phendula>(.*?)</phendula>$"
    
    completion_contents = [completion[0]["content"] for completion in completions]
    
    for content, gt in zip(completion_contents, ground_truth):
        match = re.match(pattern, content, re.DOTALL)
        if match:
            answer = match.group(1).strip()
            if answer == gt.strip():
                rewards.append(2.0)
            else:
                rewards.append(1.0)
        else:
            rewards.append(0.0)

    return rewards


def main():
    training_args = GRPOConfig(
        output_dir="Qwen2-0.5B-GRPO",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4, 
        bf16=True
    )


    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs=reward_len,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()

if __name__ == "__main__":
    main()

